// Gemini Replicas-style GDR + D2H microbenchmark.
//
// Each process binds to one GPU, connects to one peer rank, then repeatedly:
//   1. pre-posts RDMA receives into a CPU buffer,
//   2. sends a GPU buffer to the peer using RDMA SEND in chunk/batch form,
//   3. mirrors completed send batches from GPU to CPU with cudaMemcpyAsync.
//
// This isolates the Gemini save hot path: GDR read from GPU plus local D2H
// mirror overlap, without involving model/training/checkpoint serialization.

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <ifaddrs.h>
#include <infiniband/verbs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kIbPort = 1;
constexpr int kGidIndex = 1;
constexpr int kMaxWr = 256;
constexpr int kMaxSge = 1;

std::string errno_message(const std::string& what) {
    return what + ": errno=" + std::to_string(errno) + " (" + std::strerror(errno) + ")";
}

double now_sec() {
    static auto t0 = std::chrono::steady_clock::now();
    auto t1 = std::chrono::steady_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1e6;
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

std::string get_interface_ipv4(const std::string& iface_name) {
    ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) != 0) {
        throw std::runtime_error(errno_message("getifaddrs failed"));
    }

    std::string ip;
    for (ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || iface_name != ifa->ifa_name ||
            ifa->ifa_addr->sa_family != AF_INET) {
            continue;
        }

        char buf[INET_ADDRSTRLEN] = {};
        auto* addr = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
        if (inet_ntop(AF_INET, &addr->sin_addr, buf, sizeof(buf))) {
            ip = buf;
            break;
        }
    }

    freeifaddrs(ifaddr);
    if (ip.empty()) {
        throw std::runtime_error("No IPv4 address found for interface " + iface_name);
    }
    return ip;
}

ibv_device* find_rdma_device_by_ip(
    const std::string& ip,
    ibv_device** dev_list,
    int num_devices,
    int gid_index) {
    in_addr target{};
    if (inet_pton(AF_INET, ip.c_str(), &target) != 1) {
        throw std::runtime_error("Invalid IPv4 address for RDMA device lookup: " + ip);
    }

    for (int i = 0; i < num_devices; ++i) {
        ibv_context* ctx = ibv_open_device(dev_list[i]);
        if (!ctx) continue;

        ibv_gid gid{};
        bool matched = false;
        if (ibv_query_gid(ctx, kIbPort, gid_index, &gid) == 0) {
            matched = std::memcmp(&gid.raw[12], &target.s_addr, 4) == 0;
        }
        ibv_close_device(ctx);

        if (matched) {
            return dev_list[i];
        }
    }

    return nullptr;
}

size_t parse_size_arg(std::string value) {
    size_t mul = 1;
    if (!value.empty()) {
        char suffix = value.back();
        if (suffix == 'K' || suffix == 'k') {
            mul = 1024ULL;
            value.pop_back();
        } else if (suffix == 'M' || suffix == 'm') {
            mul = 1024ULL * 1024ULL;
            value.pop_back();
        } else if (suffix == 'G' || suffix == 'g') {
            mul = 1024ULL * 1024ULL * 1024ULL;
            value.pop_back();
        }
    }
    return static_cast<size_t>(std::stoull(value)) * mul;
}

struct QPInfo {
    uint32_t qp_num;
    uint16_t lid;
    uint8_t gid[16];
} __attribute__((packed));

struct Config {
    int rank = 0;
    int world_size = 16;
    int local_rank = 0;
    int cuda_device = 0;
    int node_rank = 0;
    int peer_rank = -1;
    int base_port = 36000;
    int gid_index = kGidIndex;
    std::string master_addr = "127.0.0.1";
    size_t bytes = 512ULL * 1024ULL * 1024ULL;
    int iters = 10;
    int warmup = 2;
    size_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
    int batch_wr = 4;
    bool verify = false;
    bool debug = false;
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return std::string(argv[++i]);
        };
        if (arg == "--rank") cfg.rank = std::stoi(need_value("--rank"));
        else if (arg == "--world-size") cfg.world_size = std::stoi(need_value("--world-size"));
        else if (arg == "--local-rank") cfg.local_rank = std::stoi(need_value("--local-rank"));
        else if (arg == "--cuda-device") cfg.cuda_device = std::stoi(need_value("--cuda-device"));
        else if (arg == "--node-rank") cfg.node_rank = std::stoi(need_value("--node-rank"));
        else if (arg == "--peer-rank") cfg.peer_rank = std::stoi(need_value("--peer-rank"));
        else if (arg == "--master-addr") cfg.master_addr = need_value("--master-addr");
        else if (arg == "--base-port") cfg.base_port = std::stoi(need_value("--base-port"));
        else if (arg == "--gid-index") cfg.gid_index = std::stoi(need_value("--gid-index"));
        else if (arg == "--bytes") cfg.bytes = parse_size_arg(need_value("--bytes"));
        else if (arg == "--size-mb") cfg.bytes = static_cast<size_t>(std::stoull(need_value("--size-mb"))) * 1024ULL * 1024ULL;
        else if (arg == "--iters") cfg.iters = std::stoi(need_value("--iters"));
        else if (arg == "--warmup") cfg.warmup = std::stoi(need_value("--warmup"));
        else if (arg == "--chunk-mb") cfg.chunk_bytes = static_cast<size_t>(std::stoull(need_value("--chunk-mb"))) * 1024ULL * 1024ULL;
        else if (arg == "--batch-wr") cfg.batch_wr = std::stoi(need_value("--batch-wr"));
        else if (arg == "--verify") cfg.verify = true;
        else if (arg == "--debug") cfg.debug = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: gemini_gdr_sendrecv_bench --rank R --world-size W --local-rank L\n"
                << "       --node-rank N --master-addr IP [options]\n"
                << "Options:\n"
                << "  --cuda-device N   CUDA device ordinal after CUDA_VISIBLE_DEVICES (default: 0)\n"
                << "  --peer-rank R     Override peer rank (default: rank +/- world_size/2)\n"
                << "  --base-port P     Base TCP port (default: 36000)\n"
                << "  --gid-index N     GID index for RoCE global routing (default: 1)\n"
                << "  --bytes N|512M    Transfer bytes per iteration\n"
                << "  --size-mb MB      Transfer size in MiB\n"
                << "  --iters N         Timed iterations (default: 10)\n"
                << "  --warmup N        Warmup iterations (default: 2)\n"
                << "  --chunk-mb MB     RDMA chunk size (default: 64)\n"
                << "  --batch-wr N      Chunks per signaled batch (default: 4)\n"
                << "  --verify          Sample-check received data\n"
                << "  --debug           Print mirror task details\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (cfg.world_size <= 1 || cfg.world_size % 2 != 0) {
        throw std::runtime_error("--world-size must be an even value >= 2");
    }
    if (cfg.peer_rank < 0) {
        int half = cfg.world_size / 2;
        cfg.peer_rank = (cfg.rank < half) ? cfg.rank + half : cfg.rank - half;
    }
    if (cfg.cuda_device < 0) {
        throw std::runtime_error("--cuda-device must be >= 0");
    }
    if (cfg.batch_wr <= 0 || cfg.batch_wr > kMaxWr) {
        throw std::runtime_error("--batch-wr must be in [1, 256]");
    }
    if (cfg.gid_index < 0) {
        throw std::runtime_error("--gid-index must be >= 0");
    }
    if (cfg.chunk_bytes == 0 || cfg.bytes == 0) {
        throw std::runtime_error("Transfer size and chunk size must be non-zero");
    }
    return cfg;
}

ibv_device* select_rdma_device(
    const Config& cfg,
    ibv_device** dev_list,
    int num_devices,
    std::string* selected_iface,
    std::string* selected_ip) {
    std::string env_name = "GEMINI_REPLICAS_LOCAL_RANK_NIC_" + std::to_string(cfg.local_rank);
    const char* iface = std::getenv(env_name.c_str());
    if (!iface || std::strlen(iface) == 0) {
        iface = std::getenv("GEMINI_REPLICAS_INTERFACE");
    }
    if (!iface || std::strlen(iface) == 0) {
        iface = std::getenv("NETIFACES_INTERFACE");
    }

    if (!iface || std::strlen(iface) == 0) {
        *selected_iface = "";
        *selected_ip = "";
        return dev_list[0];
    }

    for (int i = 0; i < num_devices; ++i) {
        if (std::strcmp(ibv_get_device_name(dev_list[i]), iface) == 0) {
            *selected_iface = iface;
            *selected_ip = "";
            return dev_list[i];
        }
    }

    *selected_iface = iface;
    *selected_ip = get_interface_ipv4(*selected_iface);
    ibv_device* matched = find_rdma_device_by_ip(*selected_ip, dev_list, num_devices, cfg.gid_index);
    if (!matched) {
        std::cerr << "WARNING: IP " << *selected_ip << " from interface " << *selected_iface
                  << " not found in RDMA GID table at gid_index=" << cfg.gid_index
                  << ", falling back to first device " << ibv_get_device_name(dev_list[0])
                  << std::endl;
        return dev_list[0];
    }
    return matched;
}

int tcp_listen(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket() failed");
    int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        throw std::runtime_error("bind() failed on port " + std::to_string(port));
    }
    if (listen(fd, 16) < 0) throw std::runtime_error("listen() failed");
    return fd;
}

int tcp_accept(int listen_fd) {
    sockaddr_in peer{};
    socklen_t len = sizeof(peer);
    int fd = accept(listen_fd, reinterpret_cast<sockaddr*>(&peer), &len);
    if (fd < 0) throw std::runtime_error("accept() failed");
    return fd;
}

int tcp_connect_retry(const std::string& ip, int port) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error("socket() failed");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) {
            close(fd);
            throw std::runtime_error("inet_pton failed for " + ip);
        }
        if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            return fd;
        }
        close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("connect failed to " + ip + ":" + std::to_string(port));
}

void tcp_barrier(int fd, bool lower_rank) {
    uint8_t byte = 7;
    if (lower_rank) {
        if (send(fd, &byte, 1, 0) != 1) throw std::runtime_error("barrier send failed");
        if (recv(fd, &byte, 1, MSG_WAITALL) != 1) throw std::runtime_error("barrier recv failed");
    } else {
        if (recv(fd, &byte, 1, MSG_WAITALL) != 1) throw std::runtime_error("barrier recv failed");
        if (send(fd, &byte, 1, 0) != 1) throw std::runtime_error("barrier send failed");
    }
}

struct MirrorTask {
    size_t offset;
    size_t bytes;
};

class MirrorWorker {
public:
    MirrorWorker(uint8_t* cpu_base, const uint8_t* gpu_base, bool debug)
        : cpu_base_(cpu_base), gpu_base_(gpu_base), debug_(debug) {
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
        worker_ = std::thread(&MirrorWorker::run, this);
    }

    ~MirrorWorker() {
        if (!joined_) {
            finish();
        }
    }

    void push(size_t offset, size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push({offset, bytes});
            tasks_++;
            bytes_ += bytes;
        }
        cv_.notify_one();
    }

    double finish() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();
        double t0 = now_sec();
        if (worker_.joinable()) worker_.join();
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        check_cuda(cudaStreamDestroy(stream_), "cudaStreamDestroy");
        joined_ = true;
        double elapsed = now_sec() - t0;
        if (debug_) {
            std::cout << "mirror_tasks=" << tasks_
                      << " mirror_bytes=" << bytes_
                      << " mirror_finish_s=" << elapsed << std::endl;
        }
        return elapsed;
    }

    size_t tasks() const { return tasks_; }

private:
    void run() {
        while (true) {
            MirrorTask task{};
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) {
                    if (stop_) return;
                    continue;
                }
                task = queue_.front();
                queue_.pop();
            }
            check_cuda(
                cudaMemcpyAsync(
                    cpu_base_ + task.offset,
                    gpu_base_ + task.offset,
                    task.bytes,
                    cudaMemcpyDeviceToHost,
                    stream_),
                "cudaMemcpyAsync D2H");
        }
    }

    uint8_t* cpu_base_;
    const uint8_t* gpu_base_;
    bool debug_;
    cudaStream_t stream_{};
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<MirrorTask> queue_;
    bool stop_{false};
    bool joined_{false};
    size_t tasks_{0};
    size_t bytes_{0};
};

class Lane {
public:
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* send_cq = nullptr;
    ibv_cq* recv_cq = nullptr;
    ibv_qp* qp = nullptr;
    int tcp_fd = -1;

    ~Lane() {
        if (qp) ibv_destroy_qp(qp);
        if (recv_cq) ibv_destroy_cq(recv_cq);
        if (send_cq) ibv_destroy_cq(send_cq);
        if (tcp_fd >= 0) close(tcp_fd);
    }

    void connect_qp(int fd, int gid_index) {
        tcp_fd = fd;
        send_cq = ibv_create_cq(ctx, kMaxWr, nullptr, nullptr, 0);
        recv_cq = ibv_create_cq(ctx, kMaxWr, nullptr, nullptr, 0);
        if (!send_cq || !recv_cq) throw std::runtime_error("ibv_create_cq failed");

        ibv_qp_init_attr qp_attr{};
        qp_attr.send_cq = send_cq;
        qp_attr.recv_cq = recv_cq;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr = kMaxWr;
        qp_attr.cap.max_recv_wr = kMaxWr;
        qp_attr.cap.max_send_sge = kMaxSge;
        qp_attr.cap.max_recv_sge = kMaxSge;
        qp = ibv_create_qp(pd, &qp_attr);
        if (!qp) throw std::runtime_error("ibv_create_qp failed");

        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = kIbPort;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        if (ibv_modify_qp(qp, &attr, flags)) throw std::runtime_error(errno_message("modify QP INIT failed"));

        ibv_port_attr port_attr{};
        if (ibv_query_port(ctx, kIbPort, &port_attr)) throw std::runtime_error("ibv_query_port failed");

        QPInfo local{};
        local.qp_num = qp->qp_num;
        local.lid = port_attr.lid;
        ibv_gid gid{};
        if (ibv_query_gid(ctx, kIbPort, gid_index, &gid) == 0) {
            std::memcpy(local.gid, &gid, sizeof(local.gid));
        } else {
            throw std::runtime_error(errno_message("ibv_query_gid failed"));
        }
        QPInfo remote{};
        if (send(tcp_fd, &local, sizeof(local), 0) != sizeof(local)) {
            throw std::runtime_error("send QP info failed");
        }
        if (recv(tcp_fd, &remote, sizeof(remote), MSG_WAITALL) != sizeof(remote)) {
            throw std::runtime_error("recv QP info failed");
        }

        std::memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = port_attr.active_mtu;
        attr.dest_qp_num = remote.qp_num;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = (remote.lid == 0) ? 1 : 0;
        attr.ah_attr.dlid = remote.lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = kIbPort;
        if (remote.lid == 0) {
            attr.ah_attr.grh.dgid.global.subnet_prefix =
                reinterpret_cast<uint64_t*>(remote.gid)[0];
            attr.ah_attr.grh.dgid.global.interface_id =
                reinterpret_cast<uint64_t*>(remote.gid)[1];
            attr.ah_attr.grh.sgid_index = gid_index;
            attr.ah_attr.grh.hop_limit = 255;
        }
        flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
        if (ibv_modify_qp(qp, &attr, flags)) throw std::runtime_error(errno_message("modify QP RTR failed"));

        std::memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
        if (ibv_modify_qp(qp, &attr, flags)) throw std::runtime_error(errno_message("modify QP RTS failed"));
    }

    void poll_cq(ibv_cq* cq, int count) {
        int done = 0;
        while (done < count) {
            ibv_wc wc{};
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) throw std::runtime_error("ibv_poll_cq failed");
            if (n == 0) continue;
            if (wc.status != IBV_WC_SUCCESS) {
                throw std::runtime_error("WC status " + std::to_string(wc.status));
            }
            done++;
        }
    }

    void post_recvs(uint8_t* buffer, size_t total_size, ibv_mr* mr, size_t chunk_bytes, int batch_wr) {
        size_t chunk_count = (total_size + chunk_bytes - 1) / chunk_bytes;
        for (size_t batch_start = 0; batch_start < chunk_count; batch_start += batch_wr) {
            size_t batch_end = std::min(chunk_count, batch_start + static_cast<size_t>(batch_wr));
            std::vector<ibv_sge> sges(batch_end - batch_start);
            std::vector<ibv_recv_wr> wrs(batch_end - batch_start);
            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t local = i - batch_start;
                size_t offset = i * chunk_bytes;
                size_t bytes = std::min(chunk_bytes, total_size - offset);
                sges[local].addr = reinterpret_cast<uint64_t>(buffer + offset);
                sges[local].length = static_cast<uint32_t>(bytes);
                sges[local].lkey = mr->lkey;
                wrs[local] = {};
                wrs[local].sg_list = &sges[local];
                wrs[local].num_sge = 1;
                wrs[local].next = (local + 1 < batch_end - batch_start) ? &wrs[local + 1] : nullptr;
            }
            ibv_recv_wr* bad = nullptr;
            if (ibv_post_recv(qp, &wrs[0], &bad)) throw std::runtime_error("ibv_post_recv failed");
        }
    }

    void send_gpu_batches(
        const uint8_t* gpu_buffer,
        size_t total_size,
        ibv_mr* mr,
        size_t chunk_bytes,
        int batch_wr,
        MirrorWorker* mirror) {
        size_t chunk_count = (total_size + chunk_bytes - 1) / chunk_bytes;
        for (size_t batch_start = 0; batch_start < chunk_count; batch_start += batch_wr) {
            size_t batch_end = std::min(chunk_count, batch_start + static_cast<size_t>(batch_wr));
            std::vector<ibv_sge> sges(batch_end - batch_start);
            std::vector<ibv_send_wr> wrs(batch_end - batch_start);
            size_t batch_offset = batch_start * chunk_bytes;
            size_t batch_bytes = std::min(total_size - batch_offset, (batch_end - batch_start) * chunk_bytes);
            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t local = i - batch_start;
                size_t offset = i * chunk_bytes;
                size_t bytes = std::min(chunk_bytes, total_size - offset);
                sges[local].addr = reinterpret_cast<uint64_t>(gpu_buffer + offset);
                sges[local].length = static_cast<uint32_t>(bytes);
                sges[local].lkey = mr->lkey;
                wrs[local] = {};
                wrs[local].sg_list = &sges[local];
                wrs[local].num_sge = 1;
                wrs[local].opcode = IBV_WR_SEND;
                wrs[local].send_flags = 0;
                wrs[local].next = (local + 1 < batch_end - batch_start) ? &wrs[local + 1] : nullptr;
            }
            wrs.back().send_flags = IBV_SEND_SIGNALED;
            ibv_send_wr* bad = nullptr;
            if (ibv_post_send(qp, &wrs[0], &bad)) throw std::runtime_error("ibv_post_send failed");
            poll_cq(send_cq, 1);
            if (mirror) mirror->push(batch_offset, batch_bytes);
        }
    }
};

bool verify_recv(const uint8_t* recv_buf, size_t bytes, int peer_rank) {
    if (bytes == 0) return true;
    std::vector<size_t> positions = {0, bytes / 2, bytes - 1};
    for (size_t pos : positions) {
        uint8_t expected = static_cast<uint8_t>((peer_rank + pos) & 0xFF);
        if (recv_buf[pos] != expected) {
            std::cerr << "verify failed at offset " << pos
                      << ": got=" << static_cast<int>(recv_buf[pos])
                      << " expected=" << static_cast<int>(expected) << std::endl;
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        bool lower_rank = cfg.rank < cfg.peer_rank;
        int port = cfg.base_port + std::min(cfg.rank, cfg.peer_rank);

        check_cuda(cudaSetDevice(cfg.cuda_device), "cudaSetDevice");

        int tcp_fd = -1;
        if (lower_rank) {
            int listen_fd = tcp_listen(port);
            tcp_fd = tcp_accept(listen_fd);
            close(listen_fd);
        } else {
            tcp_fd = tcp_connect_retry(cfg.master_addr, port);
        }

        int num_devices = 0;
        ibv_device** dev_list = ibv_get_device_list(&num_devices);
        if (!dev_list || num_devices == 0) throw std::runtime_error("No IB devices found");
        std::string selected_iface;
        std::string selected_ip;
        ibv_device* dev = select_rdma_device(cfg, dev_list, num_devices, &selected_iface, &selected_ip);
        std::string rdma_device_name = ibv_get_device_name(dev);
        std::cout << "RDMA_BIND,"
                  << "rank=" << cfg.rank
                  << ",local_rank=" << cfg.local_rank
                  << ",nic=" << (selected_iface.empty() ? "default" : selected_iface)
                  << ",nic_ip=" << (selected_ip.empty() ? "default" : selected_ip)
                  << ",rdma_device=" << rdma_device_name
                  << ",gid_index=" << cfg.gid_index
                  << std::endl;
        ibv_context* ctx = ibv_open_device(dev);
        if (!ctx) throw std::runtime_error("ibv_open_device failed for " + rdma_device_name);
        ibv_pd* pd = ibv_alloc_pd(ctx);
        if (!pd) throw std::runtime_error("ibv_alloc_pd failed");

        Lane lane;
        lane.ctx = ctx;
        lane.pd = pd;
        lane.connect_qp(tcp_fd, cfg.gid_index);

        uint8_t* gpu_send = nullptr;
        check_cuda(cudaMalloc(&gpu_send, cfg.bytes), "cudaMalloc gpu_send");
        check_cuda(cudaMemset(gpu_send, cfg.rank & 0xFF, cfg.bytes), "cudaMemset gpu_send");

        uint8_t* host_pattern = nullptr;
        check_cuda(cudaMallocHost(&host_pattern, cfg.bytes), "cudaMallocHost pattern");
        for (size_t i = 0; i < cfg.bytes; ++i) host_pattern[i] = static_cast<uint8_t>((cfg.rank + i) & 0xFF);
        check_cuda(cudaMemcpy(gpu_send, host_pattern, cfg.bytes, cudaMemcpyHostToDevice), "cudaMemcpy pattern H2D");

        uint8_t* cpu_recv = nullptr;
        uint8_t* cpu_mirror = nullptr;
        check_cuda(cudaMallocHost(&cpu_recv, cfg.bytes), "cudaMallocHost cpu_recv");
        check_cuda(cudaMallocHost(&cpu_mirror, cfg.bytes), "cudaMallocHost cpu_mirror");
        std::memset(cpu_recv, 0, cfg.bytes);
        std::memset(cpu_mirror, 0, cfg.bytes);

        ibv_mr* send_mr = ibv_reg_mr(
            pd, gpu_send, cfg.bytes,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!send_mr) throw std::runtime_error("GDR ibv_reg_mr failed for GPU buffer");
        ibv_mr* recv_mr = ibv_reg_mr(
            pd, cpu_recv, cfg.bytes,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!recv_mr) throw std::runtime_error("ibv_reg_mr failed for CPU recv buffer");

        tcp_barrier(tcp_fd, lower_rank);

        int total_iters = cfg.warmup + cfg.iters;
        std::vector<double> sendrecv_ms;
        std::vector<double> mirror_wait_ms;
        std::vector<double> total_ms;
        sendrecv_ms.reserve(cfg.iters);
        mirror_wait_ms.reserve(cfg.iters);
        total_ms.reserve(cfg.iters);
        size_t last_mirror_tasks = 0;

        for (int iter = 0; iter < total_iters; ++iter) {
            bool timed = iter >= cfg.warmup;
            lane.post_recvs(cpu_recv, cfg.bytes, recv_mr, cfg.chunk_bytes, cfg.batch_wr);
            tcp_barrier(tcp_fd, lower_rank);

            MirrorWorker mirror(cpu_mirror, gpu_send, cfg.debug);
            double t0 = now_sec();
            std::atomic<bool> recv_done{false};
            std::exception_ptr recv_error = nullptr;

            std::thread recv_thread([&] {
                try {
                    size_t chunk_count = (cfg.bytes + cfg.chunk_bytes - 1) / cfg.chunk_bytes;
                    lane.poll_cq(lane.recv_cq, static_cast<int>(chunk_count));
                    recv_done = true;
                } catch (...) {
                    recv_error = std::current_exception();
                }
            });

            lane.send_gpu_batches(gpu_send, cfg.bytes, send_mr, cfg.chunk_bytes, cfg.batch_wr, &mirror);
            if (recv_thread.joinable()) recv_thread.join();
            if (recv_error) std::rethrow_exception(recv_error);

            double t_sendrecv = now_sec();
            double mirror_wait = mirror.finish();
            double t_done = now_sec();
            last_mirror_tasks = mirror.tasks();

            tcp_barrier(tcp_fd, lower_rank);
            if (timed) {
                double iter_sendrecv_ms = (t_sendrecv - t0) * 1000.0;
                double iter_mirror_wait_ms = mirror_wait * 1000.0;
                double iter_total_ms = (t_done - t0) * 1000.0;
                double iter_gbps = (static_cast<double>(cfg.bytes) * 2.0 * 8.0) /
                                   (iter_total_ms / 1000.0) / 1e9;
                int timed_iter = iter - cfg.warmup;
                sendrecv_ms.push_back(iter_sendrecv_ms);
                mirror_wait_ms.push_back(iter_mirror_wait_ms);
                total_ms.push_back(iter_total_ms);

                std::cout << std::fixed << std::setprecision(3)
                          << "ITER_RESULT,"
                          << "rank=" << cfg.rank
                          << ",node_rank=" << cfg.node_rank
                          << ",local_rank=" << cfg.local_rank
                          << ",cuda_device=" << cfg.cuda_device
                          << ",peer_rank=" << cfg.peer_rank
                          << ",gid_index=" << cfg.gid_index
                          << ",nic=" << (selected_iface.empty() ? "default" : selected_iface)
                          << ",nic_ip=" << (selected_ip.empty() ? "default" : selected_ip)
                          << ",rdma_device=" << rdma_device_name
                          << ",iter=" << iter
                          << ",timed_iter=" << timed_iter
                          << ",size_mb=" << (cfg.bytes / 1024.0 / 1024.0)
                          << ",chunk_mb=" << (cfg.chunk_bytes / 1024.0 / 1024.0)
                          << ",batch_wr=" << cfg.batch_wr
                          << ",sendrecv_ms=" << iter_sendrecv_ms
                          << ",mirror_wait_ms=" << iter_mirror_wait_ms
                          << ",total_ms=" << iter_total_ms
                          << ",gbps=" << iter_gbps
                          << ",mirror_tasks=" << last_mirror_tasks
                          << std::endl;
            }
        }

        if (cfg.verify && !verify_recv(cpu_recv, cfg.bytes, cfg.peer_rank)) {
            throw std::runtime_error("verification failed");
        }

        auto avg = [](const std::vector<double>& values) {
            return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
        };
        double avg_sendrecv = avg(sendrecv_ms);
        double avg_mirror_wait = avg(mirror_wait_ms);
        double avg_total = avg(total_ms);
        double gbps = (static_cast<double>(cfg.bytes) * 2.0 * 8.0) / (avg_total / 1000.0) / 1e9;

        std::cout << std::fixed << std::setprecision(3)
                  << "RESULT,"
                  << "rank=" << cfg.rank
                  << ",node_rank=" << cfg.node_rank
                  << ",local_rank=" << cfg.local_rank
                  << ",cuda_device=" << cfg.cuda_device
                  << ",peer_rank=" << cfg.peer_rank
                  << ",gid_index=" << cfg.gid_index
                  << ",nic=" << (selected_iface.empty() ? "default" : selected_iface)
                  << ",nic_ip=" << (selected_ip.empty() ? "default" : selected_ip)
                  << ",rdma_device=" << rdma_device_name
                  << ",size_mb=" << (cfg.bytes / 1024.0 / 1024.0)
                  << ",iters=" << cfg.iters
                  << ",chunk_mb=" << (cfg.chunk_bytes / 1024.0 / 1024.0)
                  << ",batch_wr=" << cfg.batch_wr
                  << ",sendrecv_ms=" << avg_sendrecv
                  << ",mirror_wait_ms=" << avg_mirror_wait
                  << ",total_ms=" << avg_total
                  << ",gbps=" << gbps
                  << ",mirror_tasks=" << last_mirror_tasks
                  << std::endl;

        ibv_dereg_mr(send_mr);
        ibv_dereg_mr(recv_mr);
        cudaFreeHost(cpu_mirror);
        cudaFreeHost(cpu_recv);
        cudaFreeHost(host_pattern);
        cudaFree(gpu_send);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
