// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// Two-rank RDMA microbenchmark aligned with Megatron EC/Gemini checkpoint path:
//   resolve_ip (Python) -> find_rdma_device_by_ip -> GID index 1 -> QP exchange over TCP
//   -> chunked IBV_WR_SEND / IBV_WR_RECV (64 MiB chunks, same as ecnaive_native).

#include <arpa/inet.h>
#include <cerrno>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <infiniband/verbs.h>

#include "rdma_device_utils.h"

namespace {

constexpr int kGidIndex = 1;
constexpr size_t kChunkSize = 64ULL * 1024 * 1024;
constexpr int kMaxWr = 64;

struct RdmaConnInfo {
    uint32_t qp_num;
    uint16_t lid;
    uint8_t gid[16];
} __attribute__((packed));

struct BenchConfig {
    std::string bind_ip;
    std::string peer_ip;
    int rank = 0;
    int peer_rank = 1;
    uint16_t port = 19987;
    size_t message_size = 64ULL * 1024 * 1024;
    int warmup_iters = 5;
    int bench_iters = 50;
    bool phase_send = true;
    bool phase_recv = true;
    bool phase_duplex = true;
};

void die(const std::string& msg) {
    std::cerr << "[rdma_ec_bind_bench] ERROR: " << msg << std::endl;
    std::exit(1);
}

BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                die(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--bind-ip") {
            cfg.bind_ip = need("--bind-ip");
        } else if (arg == "--peer-ip") {
            cfg.peer_ip = need("--peer-ip");
        } else if (arg == "--rank") {
            cfg.rank = std::stoi(need("--rank"));
        } else if (arg == "--peer-rank") {
            cfg.peer_rank = std::stoi(need("--peer-rank"));
        } else if (arg == "--port") {
            cfg.port = static_cast<uint16_t>(std::stoul(need("--port")));
        } else if (arg == "--size-mb") {
            cfg.message_size = static_cast<size_t>(std::stoull(need("--size-mb"))) * 1024 * 1024;
        } else if (arg == "--warmup") {
            cfg.warmup_iters = std::stoi(need("--warmup"));
        } else if (arg == "--iterations") {
            cfg.bench_iters = std::stoi(need("--iterations"));
        } else if (arg == "--skip-send") {
            cfg.phase_send = false;
        } else if (arg == "--skip-recv") {
            cfg.phase_recv = false;
        } else if (arg == "--skip-duplex") {
            cfg.phase_duplex = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: rdma_ec_bind_bench --bind-ip IP --peer-ip IP --rank R --peer-rank P\n"
                << "       [--port PORT] [--size-mb N] [--warmup N] [--iterations N]\n"
                << "       [--skip-send] [--skip-recv] [--skip-duplex]\n"
                << "Phases (default all): send-only, recv-only, concurrent send+recv\n";
            std::exit(0);
        } else {
            die("Unknown argument: " + arg);
        }
    }
    if (cfg.bind_ip.empty() || cfg.peer_ip.empty()) {
        die("--bind-ip and --peer-ip are required");
    }
    return cfg;
}

int tcp_listen(const std::string& bind_ip, uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        die("socket() failed");
    }
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, bind_ip.c_str(), &addr.sin_addr) != 1) {
        die("invalid bind-ip: " + bind_ip);
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        die("bind() failed on " + bind_ip + ":" + std::to_string(port));
    }
    if (listen(fd, 1) < 0) {
        die("listen() failed");
    }
    return fd;
}

int tcp_connect_with_retry(const std::string& peer_ip, uint16_t port, int timeout_ms = 30000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    int attempt = 0;
    while (true) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            die("socket() failed");
        }
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (inet_pton(AF_INET, peer_ip.c_str(), &addr.sin_addr) != 1) {
            close(fd);
            die("invalid peer-ip: " + peer_ip);
        }
        if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            if (attempt > 0) {
                std::cout << "[bench] connect ok to " << peer_ip << ":" << port
                          << " after " << attempt << " retries" << std::endl;
            }
            return fd;
        }
        const int err = errno;
        close(fd);
        if (std::chrono::steady_clock::now() >= deadline) {
            die("connect() failed to " + peer_ip + ":" + std::to_string(port) + " after " +
                std::to_string(attempt + 1) + " attempts: " + std::strerror(err));
        }
        if (err != ECONNREFUSED && err != ETIMEDOUT && err != EINTR) {
            die("connect() failed to " + peer_ip + ":" + std::to_string(port) + ": " +
                std::strerror(err));
        }
        ++attempt;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

int tcp_accept(int listen_fd) {
    sockaddr_in peer{};
    socklen_t len = sizeof(peer);
    int fd = accept(listen_fd, reinterpret_cast<sockaddr*>(&peer), &len);
    if (fd < 0) {
        die("accept() failed");
    }
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &peer.sin_addr, ip, sizeof(ip));
    std::cout << "[bench] TCP accepted from " << ip << std::endl;
    return fd;
}

void send_all(int fd, const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, 0);
        if (n <= 0) {
            die("send() failed");
        }
        sent += static_cast<size_t>(n);
    }
}

void recv_all(int fd, void* buf, size_t len) {
    char* p = static_cast<char*>(buf);
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, p + got, len - got, MSG_WAITALL);
        if (n <= 0) {
            die("recv() failed");
        }
        got += static_cast<size_t>(n);
    }
}

RdmaConnInfo get_local_conn_info(ibv_context* ctx, ibv_qp* qp) {
    RdmaConnInfo info{};
    info.qp_num = qp->qp_num;
    ibv_port_attr port_attr{};
    if (ibv_query_port(ctx, 1, &port_attr) == 0) {
        info.lid = port_attr.lid;
    }
    ibv_gid gid{};
    if (ibv_query_gid(ctx, 1, kGidIndex, &gid) == 0) {
        std::memcpy(info.gid, &gid, 16);
    }
    return info;
}

void connect_qp(ibv_context* ctx, ibv_qp* qp, const RdmaConnInfo& remote) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        die("QP INIT failed");
    }

    ibv_port_attr port_attr{};
    if (ibv_query_port(ctx, 1, &port_attr) != 0) {
        die("ibv_query_port failed");
    }
    const bool use_gid = (remote.lid == 0);

    attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = port_attr.active_mtu;
    attr.dest_qp_num = remote.qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = use_gid ? 1 : 0;
    attr.ah_attr.dlid = remote.lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
    if (use_gid) {
        std::memcpy(&attr.ah_attr.grh.dgid, remote.gid, 16);
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.sgid_index = kGidIndex;
        attr.ah_attr.grh.hop_limit = 255;
        attr.ah_attr.grh.traffic_class = 0;
    }
    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        die("QP RTR failed");
    }

    attr = {};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
        die("QP RTS failed");
    }
}

void exchange_conn_info(int sock, ibv_context* ctx, ibv_qp* qp, RdmaConnInfo& remote_out) {
    RdmaConnInfo local = get_local_conn_info(ctx, qp);
    send_all(sock, &local, sizeof(local));
    recv_all(sock, &remote_out, sizeof(remote_out));
    connect_qp(ctx, qp, remote_out);
    std::cout << "[bench] QP connected local=" << local.qp_num << " remote=" << remote_out.qp_num
              << " use_gid=" << (remote_out.lid == 0 ? "yes" : "no") << std::endl;
}

std::mutex g_send_mutex;
std::mutex g_recv_mutex;

void poll_cq(ibv_cq* cq, int expected, const char* where, int rank, int timeout_sec = 120) {
    int done = 0;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (done < expected) {
        if (std::chrono::steady_clock::now() >= deadline) {
            die(std::string("poll_cq timeout rank=") + std::to_string(rank) + " at " + where +
                " (" + std::to_string(done) + "/" + std::to_string(expected) + " completions)");
        }
        ibv_wc wc{};
        const int ret = ibv_poll_cq(cq, 1, &wc);
        if (ret < 0) {
            die("ibv_poll_cq failed");
        }
        if (ret == 0) {
            continue;
        }
        if (wc.status != IBV_WC_SUCCESS) {
            die(std::string("WC error at ") + where + ": " + ibv_wc_status_str(wc.status));
        }
        ++done;
    }
}

void post_send_chunked(ibv_qp* qp, ibv_cq* send_cq, const uint8_t* data, size_t total, ibv_mr* mr,
                       int rank, const char* where) {
    size_t offset = 0;
    size_t remaining = total;
    while (remaining > 0) {
        const size_t chunk = std::min(remaining, kChunkSize);
        ibv_sge sge{};
        sge.addr = reinterpret_cast<uint64_t>(data + offset);
        sge.length = static_cast<uint32_t>(chunk);
        sge.lkey = mr->lkey;

        ibv_send_wr wr{};
        wr.wr_id = 1;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;

        ibv_send_wr* bad = nullptr;
        if (ibv_post_send(qp, &wr, &bad)) {
            die("ibv_post_send failed");
        }
        poll_cq(send_cq, 1, where, rank);
        offset += chunk;
        remaining -= chunk;
    }
}

void post_recv_chunked(ibv_qp* qp, ibv_cq* recv_cq, uint8_t* data, size_t total, ibv_mr* mr,
                       int rank, const char* where) {
    size_t offset = 0;
    size_t remaining = total;
    while (remaining > 0) {
        const size_t chunk = std::min(remaining, kChunkSize);
        ibv_sge sge{};
        sge.addr = reinterpret_cast<uint64_t>(data + offset);
        sge.length = static_cast<uint32_t>(chunk);
        sge.lkey = mr->lkey;

        ibv_recv_wr wr{};
        wr.wr_id = 1;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        ibv_recv_wr* bad = nullptr;
        if (ibv_post_recv(qp, &wr, &bad)) {
            die("ibv_post_recv failed");
        }
        poll_cq(recv_cq, 1, where, rank);
        offset += chunk;
        remaining -= chunk;
    }
}

// Post all recv WRs for the message, but do NOT poll CQ here.
// The caller must poll for exactly the returned number of completions.
int post_recv_chunked_only(ibv_qp* qp, uint8_t* data, size_t total, ibv_mr* mr) {
    size_t offset = 0;
    size_t remaining = total;
    int posted = 0;
    while (remaining > 0) {
        const size_t chunk = std::min(remaining, kChunkSize);
        ibv_sge sge{};
        sge.addr = reinterpret_cast<uint64_t>(data + offset);
        sge.length = static_cast<uint32_t>(chunk);
        sge.lkey = mr->lkey;

        ibv_recv_wr wr{};
        wr.wr_id = 1;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        ibv_recv_wr* bad = nullptr;
        if (ibv_post_recv(qp, &wr, &bad)) {
            die("ibv_post_recv failed");
        }
        offset += chunk;
        remaining -= chunk;
        ++posted;
    }
    return posted;
}

// Same control + RDMA order as ecnaive_native RdmaConnectionChannel.
void send_message(int sock, ibv_qp* qp, ibv_cq* send_cq, const uint8_t* buf, size_t size,
                  ibv_mr* mr, int rank, int iter, const char* phase) {
    uint64_t sz_net = htobe64(size);
    send_all(sock, &sz_net, sizeof(sz_net));
    uint8_t ack = 0;
    recv_all(sock, &ack, sizeof(ack));
    std::lock_guard<std::mutex> lock(g_send_mutex);
    const std::string where = std::string(phase) + " send iter=" + std::to_string(iter);
    post_send_chunked(qp, send_cq, buf, size, mr, rank, where.c_str());
}

void recv_message(int sock, ibv_qp* qp, ibv_cq* recv_cq, uint8_t* buf, size_t capacity,
                  ibv_mr* mr, size_t& out_size, int rank, int iter, const char* phase) {
    uint64_t sz_net = 0;
    recv_all(sock, &sz_net, sizeof(sz_net));
    out_size = static_cast<size_t>(be64toh(sz_net));
    if (out_size > capacity) {
        die("peer message exceeds local buffer");
    }
    // More robust ordering than the legacy path:
    // Post ALL recv WRs first, then ACK, so sender won't post_send before we are ready.
    std::lock_guard<std::mutex> lock(g_recv_mutex);
    const std::string where = std::string(phase) + " recv iter=" + std::to_string(iter);
    const int expected = post_recv_chunked_only(qp, buf, out_size, mr);
    uint8_t ack = 1;
    send_all(sock, &ack, sizeof(ack));
    poll_cq(recv_cq, expected, where.c_str(), rank);
}

struct RdmaCtx {
    ibv_context* context = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* send_cq = nullptr;
    ibv_cq* recv_cq = nullptr;
    ibv_qp* qp = nullptr;
    ibv_mr* mr = nullptr;
    std::vector<uint8_t> buffer;
    // Separate buffer for duplex recv thread to avoid racing on the same MR region.
    std::vector<uint8_t> duplex_recv_buffer;
    ibv_mr* duplex_recv_mr = nullptr;

    ~RdmaCtx() {
        if (duplex_recv_mr) {
            ibv_dereg_mr(duplex_recv_mr);
        }
        if (mr) {
            ibv_dereg_mr(mr);
        }
        if (qp) {
            ibv_destroy_qp(qp);
        }
        if (send_cq) {
            ibv_destroy_cq(send_cq);
        }
        if (recv_cq) {
            ibv_destroy_cq(recv_cq);
        }
        if (pd) {
            ibv_dealloc_pd(pd);
        }
        if (context) {
            ibv_close_device(context);
        }
    }
};

RdmaCtx init_rdma(const std::string& bind_ip, size_t buffer_size) {
    if (ibv_fork_init() != 0) {
        std::cerr << "[bench] WARNING: ibv_fork_init() failed\n";
    }

    int num_devices = 0;
    ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        die("no RDMA devices found");
    }

    ibv_device* dev = find_rdma_device_by_ip(bind_ip, dev_list, num_devices);
    RdmaCtx ctx;
    ctx.context = ibv_open_device(dev);
    ibv_free_device_list(dev_list);
    if (!ctx.context) {
        die("ibv_open_device failed");
    }

    ctx.pd = ibv_alloc_pd(ctx.context);
    if (!ctx.pd) {
        die("ibv_alloc_pd failed");
    }
    ctx.send_cq = ibv_create_cq(ctx.context, kMaxWr, nullptr, nullptr, 0);
    ctx.recv_cq = ibv_create_cq(ctx.context, kMaxWr, nullptr, nullptr, 0);
    if (!ctx.send_cq || !ctx.recv_cq) {
        die("ibv_create_cq failed");
    }

    ibv_qp_init_attr qpa{};
    qpa.send_cq = ctx.send_cq;
    qpa.recv_cq = ctx.recv_cq;
    qpa.qp_type = IBV_QPT_RC;
    qpa.cap.max_send_wr = kMaxWr;
    qpa.cap.max_recv_wr = kMaxWr;
    qpa.cap.max_send_sge = 1;
    qpa.cap.max_recv_sge = 1;
    ctx.qp = ibv_create_qp(ctx.pd, &qpa);
    if (!ctx.qp) {
        die("ibv_create_qp failed");
    }

    ctx.buffer.resize(buffer_size);
    std::memset(ctx.buffer.data(), 0xAB, buffer_size);
    ctx.mr = ibv_reg_mr(ctx.pd, ctx.buffer.data(), buffer_size,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!ctx.mr) {
        die("ibv_reg_mr failed");
    }
    ctx.duplex_recv_buffer.resize(buffer_size);
    ctx.duplex_recv_mr = ibv_reg_mr(
        ctx.pd, ctx.duplex_recv_buffer.data(), buffer_size,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!ctx.duplex_recv_mr) {
        die("ibv_reg_mr (duplex recv) failed");
    }
    return ctx;
}

void sync_barrier(int sock, bool send_first) {
    char b = 1;
    if (send_first) {
        send_all(sock, &b, 1);
        recv_all(sock, &b, 1);
    } else {
        recv_all(sock, &b, 1);
        send_all(sock, &b, 1);
    }
}

double gib_per_sec(size_t bytes, int iters, double seconds) {
    if (seconds <= 0) {
        return 0.0;
    }
    const double gib = (static_cast<double>(bytes) * static_cast<double>(iters)) /
                       (1024.0 * 1024.0 * 1024.0);
    return gib / seconds;
}

// Direction higher_rank -> lower_rank: higher uses sock_out, lower uses sock_in.
void run_send_only_phase(const BenchConfig& cfg, int sock_out, int sock_in, RdmaCtx& rdma,
                         int iters, const char* label) {
    const int sender_rank = (cfg.rank > cfg.peer_rank) ? cfg.rank : cfg.peer_rank;
    const bool is_sender = (cfg.rank == sender_rank);
    const int sock = is_sender ? sock_out : sock_in;

    std::cout << "[bench] rank " << cfg.rank << " " << label << " phase=send-only starting ("
              << iters << " iters, role=" << (is_sender ? "sender" : "receiver") << ")"
              << std::endl;
    std::cout.flush();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (is_sender) {
            send_message(sock, rdma.qp, rdma.send_cq, rdma.buffer.data(), cfg.message_size,
                         rdma.mr, cfg.rank, i, label);
        } else {
            size_t got = 0;
            recv_message(sock, rdma.qp, rdma.recv_cq, rdma.buffer.data(), rdma.buffer.size(),
                         rdma.mr, got, cfg.rank, i, label);
            (void)got;
        }
        if (i == 0 || (i + 1) % 5 == 0) {
            std::cout << "[bench] rank " << cfg.rank << " " << label << " progress " << (i + 1)
                      << "/" << iters << std::endl;
            std::cout.flush();
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    const double sec =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

    if (is_sender) {
        const double gbps = gib_per_sec(cfg.message_size, iters, sec);
        std::cout << "[bench] rank " << cfg.rank << " " << label
                  << " phase=send (peer_rank->rank, outbound) " << gbps << " GiB/s" << std::endl;
    }
}

void run_recv_only_phase(const BenchConfig& cfg, int sock_out, int sock_in, RdmaCtx& rdma,
                         int iters, const char* label) {
    const int sender_rank = (cfg.rank < cfg.peer_rank) ? cfg.rank : cfg.peer_rank;
    const bool is_receiver = (cfg.rank != sender_rank);
    const int sock = is_receiver ? sock_in : sock_out;

    std::cout << "[bench] rank " << cfg.rank << " " << label << " phase=recv-only starting ("
              << iters << " iters, role=" << (is_receiver ? "receiver" : "sender") << ")"
              << std::endl;
    std::cout.flush();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (is_receiver) {
            size_t got = 0;
            recv_message(sock, rdma.qp, rdma.recv_cq, rdma.buffer.data(), rdma.buffer.size(),
                         rdma.mr, got, cfg.rank, i, label);
            (void)got;
        } else {
            send_message(sock, rdma.qp, rdma.send_cq, rdma.buffer.data(), cfg.message_size,
                         rdma.mr, cfg.rank, i, label);
        }
        if (i == 0 || (i + 1) % 5 == 0) {
            std::cout << "[bench] rank " << cfg.rank << " " << label << " progress " << (i + 1)
                      << "/" << iters << std::endl;
            std::cout.flush();
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    const double sec =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

    if (is_receiver) {
        const double gbps = gib_per_sec(cfg.message_size, iters, sec);
        std::cout << "[bench] rank " << cfg.rank << " " << label
                  << " phase=recv (peer_rank->rank, inbound) " << gbps << " GiB/s" << std::endl;
    }
}

struct DuplexStats {
    double send_gib_per_s = 0.0;
    double recv_gib_per_s = 0.0;
};

DuplexStats run_duplex_phase(const BenchConfig& cfg, int sock_out, int sock_in, int sock_sync, bool send_first,
                             RdmaCtx& rdma, int iters, const char* label) {
    std::atomic<bool> go{false};
    DuplexStats stats;

    std::thread send_thread([&]() {
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            send_message(sock_out, rdma.qp, rdma.send_cq, rdma.buffer.data(), cfg.message_size,
                         rdma.mr, cfg.rank, i, label);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double sec =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        stats.send_gib_per_s = gib_per_sec(cfg.message_size, iters, sec);
    });

    std::thread recv_thread([&]() {
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            size_t got = 0;
            recv_message(sock_in, rdma.qp, rdma.recv_cq, rdma.duplex_recv_buffer.data(),
                         rdma.duplex_recv_buffer.size(), rdma.duplex_recv_mr, got, cfg.rank, i,
                         label);
            (void)got;
        }
        auto t1 = std::chrono::steady_clock::now();
        const double sec =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        stats.recv_gib_per_s = gib_per_sec(cfg.message_size, iters, sec);
    });

    // Duplex phase uses the same TCP sync socket; follow deterministic ordering.
    sync_barrier(sock_sync, send_first);
    go.store(true, std::memory_order_release);
    send_thread.join();
    recv_thread.join();

    std::cout << "[bench] rank " << cfg.rank << " " << label
              << " phase=duplex (concurrent) send=" << stats.send_gib_per_s
              << " GiB/s recv=" << stats.recv_gib_per_s << " GiB/s" << std::endl;
    return stats;
}

}  // namespace

int main(int argc, char** argv) {
    const BenchConfig cfg = parse_args(argc, argv);

    std::cout << "[bench] rank=" << cfg.rank << " bind_ip=" << cfg.bind_ip
              << " peer_ip=" << cfg.peer_ip << " port=" << cfg.port
              << " size=" << (cfg.message_size / (1024 * 1024)) << " MiB"
              << " warmup=" << cfg.warmup_iters << " iters=" << cfg.bench_iters << std::endl;

    RdmaCtx rdma = init_rdma(cfg.bind_ip, cfg.message_size);

    // Two TCP control channels (opposite directions) for full-duplex phase.
    //   sock_a @ port P:   peer_rank -> rank  (rank0 listen, rank1 connect)
    //   sock_b @ port P+1: rank -> peer_rank (rank1 listen, rank0 connect)
    const uint16_t port_a = cfg.port;
    const uint16_t port_b = static_cast<uint16_t>(cfg.port + 1);
    const bool is_low_rank = (cfg.rank < cfg.peer_rank);

    int listen_a = -1;
    int listen_b = -1;
    int sock_a = -1;
    int sock_b = -1;

    if (is_low_rank) {
        listen_a = tcp_listen(cfg.bind_ip, port_a);
        std::cout << "[bench] listen (peer->me) " << cfg.bind_ip << ":" << port_a << std::endl;
        sock_a = tcp_accept(listen_a);
        std::cout << "[bench] connect (me->peer) " << cfg.peer_ip << ":" << port_b << std::endl;
        sock_b = tcp_connect_with_retry(cfg.peer_ip, port_b);
    } else {
        std::cout << "[bench] connect (me->peer) " << cfg.peer_ip << ":" << port_a << std::endl;
        sock_a = tcp_connect_with_retry(cfg.peer_ip, port_a);
        listen_b = tcp_listen(cfg.bind_ip, port_b);
        std::cout << "[bench] listen (peer->me) " << cfg.bind_ip << ":" << port_b << std::endl;
        sock_b = tcp_accept(listen_b);
    }

    // sock_a carries peer_rank->rank; sock_b carries rank->peer_rank.
    int sock_in = -1;
    int sock_out = -1;
    if (is_low_rank) {
        sock_in = sock_a;
        sock_out = sock_b;
    } else {
        sock_in = sock_b;
        sock_out = sock_a;
    }

    RdmaConnInfo remote{};
    exchange_conn_info(sock_a, rdma.context, rdma.qp, remote);

    // Barriers use sock_a. Use deterministic send/recv order by rank to avoid
    // any possible TCP control-message race in single-node 2-rank simulation.
    const bool send_first = is_low_rank;
    if (cfg.rank == 0) {
        std::cout << "[bench] entering sync_barrier #0 (send_first=" << (send_first ? "yes" : "no") << ")"
                  << std::endl;
    }
    sync_barrier(sock_a, send_first);
    if (cfg.rank == 0) {
        std::cout << "[bench] leaving sync_barrier #0" << std::endl;
    }

    if (cfg.bind_ip == cfg.peer_ip) {
        std::cout << "[bench] WARNING rank=" << cfg.rank
                  << ": bind_ip == peer_ip (" << cfg.bind_ip
                  << "). Single-node RoCE often needs different NIC IPs per local_rank "
                  << "(ECNAIVE_LOCAL_RANK_NIC_0/1 on eth0/eth1) or use two physical nodes."
                  << std::endl;
    }

    if (cfg.warmup_iters > 0) {
        std::cout << "[bench] rank " << cfg.rank << " starting warmup" << std::endl;
        std::cout.flush();
        run_send_only_phase(cfg, sock_out, sock_in, rdma, cfg.warmup_iters, "warmup");
        sync_barrier(sock_a, send_first);
    }

    if (cfg.phase_send) {
        run_send_only_phase(cfg, sock_out, sock_in, rdma, cfg.bench_iters, "measure");
        sync_barrier(sock_a, send_first);
    }

    if (cfg.phase_recv) {
        run_recv_only_phase(cfg, sock_out, sock_in, rdma, cfg.bench_iters, "measure");
        sync_barrier(sock_a, send_first);
    }

    if (cfg.phase_duplex) {
        run_duplex_phase(cfg, sock_out, sock_in, sock_a, send_first, rdma, cfg.bench_iters, "measure");
        sync_barrier(sock_a, send_first);
    }

    if (listen_a >= 0) {
        close(listen_a);
    }
    if (listen_b >= 0) {
        close(listen_b);
    }
    close(sock_a);
    close(sock_b);
    return 0;
}
