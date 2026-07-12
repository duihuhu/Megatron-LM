// Pure GDR RDMA WRITE microbenchmark.
//
// Each process binds to one GPU and one RDMA device, connects to one peer rank,
// then repeatedly writes a local GPU buffer into the peer's GPU buffer. There is
// no CPU receive buffer and no GPU-to-host mirror path, so the measured data path
// is GPU memory -> RNIC -> peer RNIC -> peer GPU memory.

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <ifaddrs.h>
#include <infiniband/verbs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
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
               std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) /
           1e6;
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
        if (!ifa->ifa_addr || iface_name != ifa->ifa_name || ifa->ifa_addr->sa_family != AF_INET) {
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
    const std::string& ip, ibv_device** dev_list, int num_devices, int gid_index) {
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

        if (matched) return dev_list[i];
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

struct MRInfo {
    uint64_t addr;
    uint32_t rkey;
} __attribute__((packed));

struct Config {
    int rank = 0;
    int world_size = 16;
    int local_rank = 0;
    int cuda_device = 0;
    int node_rank = 0;
    int peer_rank = -1;
    int base_port = 39000;
    int gid_index = kGidIndex;
    std::string master_addr = "127.0.0.1";
    size_t bytes = 1024ULL * 1024ULL * 1024ULL;
    int iters = 10;
    int warmup = 4;
    size_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
    int batch_wr = 4;
    int send_window = 8;
    bool verify = false;
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
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
        else if (arg == "--send-window") cfg.send_window = std::stoi(need_value("--send-window"));
        else if (arg == "--verify") cfg.verify = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: gemini_gdr_pure_write_bench --rank R --world-size W --local-rank L\n"
                << "       --node-rank N --master-addr IP [options]\n"
                << "Options:\n"
                << "  --cuda-device N   CUDA device ordinal after CUDA_VISIBLE_DEVICES (default: 0)\n"
                << "  --peer-rank R     Override peer rank (default: rank +/- world_size/2)\n"
                << "  --base-port P     Base TCP port (default: 39000)\n"
                << "  --gid-index N     GID index for RoCE global routing (default: 1)\n"
                << "  --bytes N|1G      Transfer bytes per iteration\n"
                << "  --size-mb MB      Transfer size in MiB\n"
                << "  --iters N         Timed iterations (default: 10)\n"
                << "  --warmup N        Warmup iterations (default: 4)\n"
                << "  --chunk-mb MB     RDMA WRITE chunk size (default: 64)\n"
                << "  --batch-wr N      Chunks per signaled batch (default: 4)\n"
                << "  --send-window N   Outstanding signaled batches (default: 8)\n"
                << "  --verify          Sample-check remote writes after the run\n";
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
    if (cfg.batch_wr <= 0 || cfg.batch_wr > kMaxWr) {
        throw std::runtime_error("--batch-wr must be in [1, 256]");
    }
    if (cfg.send_window <= 0 || cfg.send_window * cfg.batch_wr > kMaxWr) {
        throw std::runtime_error("--send-window must be > 0 and send_window * batch_wr <= 256");
    }
    if (cfg.chunk_bytes == 0 || cfg.bytes == 0) {
        throw std::runtime_error("Transfer size and chunk size must be non-zero");
    }
    return cfg;
}

ibv_device* select_rdma_device(
    const Config& cfg, ibv_device** dev_list, int num_devices, std::string* selected_iface,
    std::string* selected_ip) {
    std::string env_name = "GEMINI_REPLICAS_LOCAL_RANK_NIC_" + std::to_string(cfg.local_rank);
    const char* iface = std::getenv(env_name.c_str());
    if (!iface || std::strlen(iface) == 0) iface = std::getenv("GEMINI_REPLICAS_INTERFACE");
    if (!iface || std::strlen(iface) == 0) iface = std::getenv("NETIFACES_INTERFACE");

    if (!iface || std::strlen(iface) == 0) {
        *selected_iface = "";
        *selected_ip = "";
        return dev_list[0];
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
        if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) return fd;
        close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("connect failed to " + ip + ":" + std::to_string(port));
}

void send_all(int fd, const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    while (bytes > 0) {
        ssize_t n = send(fd, p, bytes, 0);
        if (n <= 0) throw std::runtime_error("tcp send failed");
        p += n;
        bytes -= static_cast<size_t>(n);
    }
}

void recv_all(int fd, void* data, size_t bytes) {
    auto* p = static_cast<uint8_t*>(data);
    while (bytes > 0) {
        ssize_t n = recv(fd, p, bytes, MSG_WAITALL);
        if (n <= 0) throw std::runtime_error("tcp recv failed");
        p += n;
        bytes -= static_cast<size_t>(n);
    }
}

template <typename T>
void tcp_exchange(int fd, bool lower_rank, const T& local, T* remote) {
    if (lower_rank) {
        send_all(fd, &local, sizeof(local));
        recv_all(fd, remote, sizeof(*remote));
    } else {
        recv_all(fd, remote, sizeof(*remote));
        send_all(fd, &local, sizeof(local));
    }
}

void tcp_barrier(int fd, bool lower_rank) {
    uint8_t byte = 7;
    tcp_exchange(fd, lower_rank, byte, &byte);
}

class Lane {
public:
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* send_cq = nullptr;
    ibv_qp* qp = nullptr;
    int tcp_fd = -1;

    ~Lane() {
        if (qp) ibv_destroy_qp(qp);
        if (send_cq) ibv_destroy_cq(send_cq);
        if (tcp_fd >= 0) close(tcp_fd);
    }

    void connect_qp(int fd, int gid_index, bool lower_rank) {
        tcp_fd = fd;
        send_cq = ibv_create_cq(ctx, kMaxWr, nullptr, nullptr, 0);
        if (!send_cq) throw std::runtime_error("ibv_create_cq failed");

        ibv_qp_init_attr qp_attr{};
        qp_attr.send_cq = send_cq;
        qp_attr.recv_cq = send_cq;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr = kMaxWr;
        qp_attr.cap.max_recv_wr = 1;
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
        if (ibv_query_gid(ctx, kIbPort, gid_index, &gid)) {
            throw std::runtime_error(errno_message("ibv_query_gid failed"));
        }
        std::memcpy(local.gid, &gid, sizeof(local.gid));

        QPInfo remote{};
        tcp_exchange(tcp_fd, lower_rank, local, &remote);

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
            attr.ah_attr.grh.dgid.global.subnet_prefix = reinterpret_cast<uint64_t*>(remote.gid)[0];
            attr.ah_attr.grh.dgid.global.interface_id = reinterpret_cast<uint64_t*>(remote.gid)[1];
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

    uint64_t poll_one_cq() {
        while (true) {
            ibv_wc wc{};
            int n = ibv_poll_cq(send_cq, 1, &wc);
            if (n < 0) throw std::runtime_error("ibv_poll_cq failed");
            if (n == 0) continue;
            if (wc.status != IBV_WC_SUCCESS) {
                throw std::runtime_error("WC status " + std::to_string(wc.status));
            }
            return wc.wr_id;
        }
    }

    void write_gpu_batches(
        const uint8_t* local_gpu, const MRInfo& remote_mr, size_t total_size, ibv_mr* local_mr,
        size_t chunk_bytes, int batch_wr, int send_window) {
        size_t chunk_count = (total_size + chunk_bytes - 1) / chunk_bytes;
        size_t batch_count = (chunk_count + static_cast<size_t>(batch_wr) - 1) / static_cast<size_t>(batch_wr);
        size_t next_batch = 0;
        size_t completed = 0;
        size_t outstanding = 0;

        auto post_batch = [&](size_t batch_idx) {
            size_t batch_start = batch_idx * static_cast<size_t>(batch_wr);
            size_t batch_end = std::min(chunk_count, batch_start + static_cast<size_t>(batch_wr));
            std::vector<ibv_sge> sges(batch_end - batch_start);
            std::vector<ibv_send_wr> wrs(batch_end - batch_start);

            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t local = i - batch_start;
                size_t offset = i * chunk_bytes;
                size_t bytes = std::min(chunk_bytes, total_size - offset);
                sges[local].addr = reinterpret_cast<uint64_t>(local_gpu + offset);
                sges[local].length = static_cast<uint32_t>(bytes);
                sges[local].lkey = local_mr->lkey;

                wrs[local] = {};
                wrs[local].sg_list = &sges[local];
                wrs[local].num_sge = 1;
                wrs[local].opcode = IBV_WR_RDMA_WRITE;
                wrs[local].wr.rdma.remote_addr = remote_mr.addr + offset;
                wrs[local].wr.rdma.rkey = remote_mr.rkey;
                wrs[local].send_flags = 0;
                wrs[local].next = (local + 1 < batch_end - batch_start) ? &wrs[local + 1] : nullptr;
            }
            wrs.back().send_flags = IBV_SEND_SIGNALED;
            wrs.back().wr_id = static_cast<uint64_t>(batch_idx);
            ibv_send_wr* bad = nullptr;
            if (ibv_post_send(qp, &wrs[0], &bad)) throw std::runtime_error("ibv_post_send failed");
        };

        while (completed < batch_count) {
            while (next_batch < batch_count && outstanding < static_cast<size_t>(send_window)) {
                post_batch(next_batch++);
                outstanding++;
            }
            uint64_t wr_id = poll_one_cq();
            if (wr_id >= batch_count) {
                throw std::runtime_error("Invalid write completion wr_id " + std::to_string(wr_id));
            }
            outstanding--;
            completed++;
        }
    }
};

bool verify_pattern(const uint8_t* buffer, size_t bytes, int peer_rank) {
    if (bytes == 0) return true;
    std::vector<size_t> positions = {0, bytes / 2, bytes - 1};
    for (size_t pos : positions) {
        uint8_t expected = static_cast<uint8_t>((peer_rank + pos) & 0xFF);
        if (buffer[pos] != expected) {
            std::cerr << "verify failed at offset " << pos << ": got="
                      << static_cast<int>(buffer[pos]) << " expected=" << static_cast<int>(expected)
                      << std::endl;
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
        std::cout << "RDMA_BIND,rank=" << cfg.rank << ",local_rank=" << cfg.local_rank
                  << ",nic=" << (selected_iface.empty() ? "default" : selected_iface)
                  << ",nic_ip=" << (selected_ip.empty() ? "default" : selected_ip)
                  << ",rdma_device=" << rdma_device_name << ",gid_index=" << cfg.gid_index
                  << std::endl;

        ibv_context* ctx = ibv_open_device(dev);
        if (!ctx) throw std::runtime_error("ibv_open_device failed for " + rdma_device_name);
        ibv_pd* pd = ibv_alloc_pd(ctx);
        if (!pd) throw std::runtime_error("ibv_alloc_pd failed");

        Lane lane;
        lane.ctx = ctx;
        lane.pd = pd;
        lane.connect_qp(tcp_fd, cfg.gid_index, lower_rank);

        uint8_t* gpu_send = nullptr;
        uint8_t* gpu_recv = nullptr;
        check_cuda(cudaMalloc(&gpu_send, cfg.bytes), "cudaMalloc gpu_send");
        check_cuda(cudaMalloc(&gpu_recv, cfg.bytes), "cudaMalloc gpu_recv");

        uint8_t* host_pattern = nullptr;
        check_cuda(cudaMallocHost(&host_pattern, cfg.bytes), "cudaMallocHost pattern");
        for (size_t i = 0; i < cfg.bytes; ++i) host_pattern[i] = static_cast<uint8_t>((cfg.rank + i) & 0xFF);
        check_cuda(cudaMemcpy(gpu_send, host_pattern, cfg.bytes, cudaMemcpyHostToDevice), "cudaMemcpy pattern H2D");
        check_cuda(cudaMemset(gpu_recv, 0, cfg.bytes), "cudaMemset gpu_recv");

        ibv_mr* send_mr = ibv_reg_mr(pd, gpu_send, cfg.bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!send_mr) throw std::runtime_error("GDR ibv_reg_mr failed for GPU send buffer");
        ibv_mr* recv_mr = ibv_reg_mr(
            pd, gpu_recv, cfg.bytes,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!recv_mr) throw std::runtime_error("GDR ibv_reg_mr failed for GPU recv buffer");

        MRInfo local_mr{reinterpret_cast<uint64_t>(gpu_recv), recv_mr->rkey};
        MRInfo remote_mr{};
        tcp_exchange(tcp_fd, lower_rank, local_mr, &remote_mr);
        tcp_barrier(tcp_fd, lower_rank);

        int total_iters = cfg.warmup + cfg.iters;
        std::vector<double> total_ms;
        total_ms.reserve(cfg.iters);

        for (int iter = 0; iter < total_iters; ++iter) {
            bool timed = iter >= cfg.warmup;
            tcp_barrier(tcp_fd, lower_rank);
            double t0 = now_sec();
            lane.write_gpu_batches(
                gpu_send, remote_mr, cfg.bytes, send_mr, cfg.chunk_bytes, cfg.batch_wr,
                cfg.send_window);
            double t1 = now_sec();
            tcp_barrier(tcp_fd, lower_rank);

            if (timed) {
                double iter_total_ms = (t1 - t0) * 1000.0;
                double iter_gbps = (static_cast<double>(cfg.bytes) * 8.0) /
                                   (iter_total_ms / 1000.0) / 1e9;
                int timed_iter = iter - cfg.warmup;
                total_ms.push_back(iter_total_ms);
                std::cout << std::fixed << std::setprecision(3)
                          << "ITER_RESULT,rank=" << cfg.rank
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
                          << ",send_window=" << cfg.send_window
                          << ",total_ms=" << iter_total_ms
                          << ",gbps=" << iter_gbps
                          << std::endl;
            }
        }

        if (cfg.verify) {
            uint8_t* host_recv = nullptr;
            check_cuda(cudaMallocHost(&host_recv, cfg.bytes), "cudaMallocHost verify");
            check_cuda(cudaMemcpy(host_recv, gpu_recv, cfg.bytes, cudaMemcpyDeviceToHost), "cudaMemcpy verify D2H");
            bool ok = verify_pattern(host_recv, cfg.bytes, cfg.peer_rank);
            cudaFreeHost(host_recv);
            if (!ok) throw std::runtime_error("verification failed");
        }

        double avg_total = std::accumulate(total_ms.begin(), total_ms.end(), 0.0) /
                           static_cast<double>(total_ms.size());
        double gbps = (static_cast<double>(cfg.bytes) * 8.0) / (avg_total / 1000.0) / 1e9;
        std::cout << std::fixed << std::setprecision(3)
                  << "RESULT,rank=" << cfg.rank
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
                  << ",send_window=" << cfg.send_window
                  << ",total_ms=" << avg_total
                  << ",gbps=" << gbps
                  << std::endl;

        ibv_dereg_mr(send_mr);
        ibv_dereg_mr(recv_mr);
        cudaFreeHost(host_pattern);
        cudaFree(gpu_recv);
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
