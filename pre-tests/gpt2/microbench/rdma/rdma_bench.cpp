// RDMA microbenchmark: measure per-QP throughput vs QP count.
// Mirrors Concord's RDMA usage: same libibverbs API, SEND/RECV,
// dedicated CQ per QP, chunked transfers, TCP handshake per transfer.
//
// Build:  g++ -std=c++17 -O2 -o rdma_bench rdma_bench.cpp -libverbs -lpthread
// Run:    see run_rdma_bench.sh

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---- config ----
static constexpr size_t kRdmaChunk = 64ULL * 1024 * 1024;  // 64 MB per RDMA op (matches Concord)

// ---- utilities ----
static double now_sec() {
    static auto t0 = std::chrono::steady_clock::now();
    auto t1 = std::chrono::steady_clock::now();
    return (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
}

struct QPInfo {
    uint32_t qp_num;
    uint16_t lid;
    uint8_t  gid[16];
} __attribute__((packed));

// ---- A single QP with dedicated CQs (matches ConcordRdmaChannel) ----
struct Lane {
    ibv_context* ctx;
    ibv_pd*     pd;
    ibv_cq*     send_cq;
    ibv_cq*     recv_cq;
    ibv_qp*     qp;
    int         tcp_fd = -1;

    ~Lane() {
        if (qp)      ibv_destroy_qp(qp);
        if (recv_cq) ibv_destroy_cq(recv_cq);
        if (send_cq) ibv_destroy_cq(send_cq);
        if (tcp_fd >= 0) close(tcp_fd);
    }

    // Exchange QP info over TCP
    void connect_qp(int fd, ibv_mr* mr, int peer_rank) {
        tcp_fd = fd;

        send_cq = ibv_create_cq(ctx, 256, nullptr, nullptr, 0);
        recv_cq = ibv_create_cq(ctx, 256, nullptr, nullptr, 0);
        if (!send_cq || !recv_cq) throw std::runtime_error("create CQ failed");

        ibv_qp_init_attr qp_attr = {};
        qp_attr.send_cq = send_cq;
        qp_attr.recv_cq = recv_cq;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr  = 256;
        qp_attr.cap.max_recv_wr  = 256;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp = ibv_create_qp(pd, &qp_attr);
        if (!qp) throw std::runtime_error("create QP failed");

        // INIT
        {
            ibv_qp_attr attr = {};
            attr.qp_state = IBV_QPS_INIT;
            attr.pkey_index = 0;
            attr.port_num = 1;
            attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
            int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
            if (ibv_modify_qp(qp, &attr, flags))
                throw std::runtime_error("INIT failed");
        }

        // Exchange local info
        QPInfo local = {};
        local.qp_num = qp->qp_num;
        ibv_port_attr pa;
        if (ibv_query_port(ctx, 1, &pa) == 0) local.lid = pa.lid;
        {
            ibv_gid gid;
            if (ibv_query_gid(ctx, 1, 1, &gid) == 0)
                memcpy(local.gid, &gid, sizeof(local.gid));
        }

        QPInfo remote;
        if (send(tcp_fd, &local, sizeof(local), 0) != sizeof(local))
            throw std::runtime_error("send QP info failed");
        if (recv(tcp_fd, &remote, sizeof(remote), MSG_WAITALL) != sizeof(remote))
            throw std::runtime_error("recv QP info failed");

        // RTR
        {
            ibv_qp_attr attr = {};
            attr.qp_state = IBV_QPS_RTR;
            attr.path_mtu = pa.active_mtu;
            attr.dest_qp_num = remote.qp_num;
            attr.rq_psn = 0;
            attr.max_dest_rd_atomic = 1;
            attr.min_rnr_timer = 12;
            attr.ah_attr.is_global = (remote.lid == 0) ? 1 : 0;
            attr.ah_attr.dlid = remote.lid;
            attr.ah_attr.sl = 0;
            attr.ah_attr.src_path_bits = 0;
            attr.ah_attr.port_num = 1;
            if (remote.lid == 0) {
                attr.ah_attr.grh.dgid.global.subnet_prefix = ((uint64_t*)remote.gid)[0];
                attr.ah_attr.grh.dgid.global.interface_id   = ((uint64_t*)remote.gid)[1];
                attr.ah_attr.grh.sgid_index = 0;
                attr.ah_attr.grh.hop_limit = 255;
            }
            int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
            if (ibv_modify_qp(qp, &attr, flags))
                throw std::runtime_error("RTR failed");
        }

        // RTS
        {
            ibv_qp_attr attr = {};
            attr.qp_state = IBV_QPS_RTS;
            attr.timeout = 14;
            attr.retry_cnt = 7;
            attr.rnr_retry = 7;
            attr.sq_psn = 0;
            attr.max_rd_atomic = 1;
            int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
            if (ibv_modify_qp(qp, &attr, flags))
                throw std::runtime_error("RTS failed");
        }
    }

    void poll_cq(ibv_cq* cq, int count) {
        int done = 0;
        while (done < count) {
            ibv_wc wc;
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) throw std::runtime_error("poll CQ error");
            if (n == 0) continue;
            if (wc.status != IBV_WC_SUCCESS)
                throw std::runtime_error("WC status " + std::to_string(wc.status));
            done++;
        }
    }

    // RDMA SEND (matches Concord: chunked, poll after each post)
    void send_data(const uint8_t* data, size_t size, ibv_mr* mr) {
        size_t remaining = size, offset = 0;
        while (remaining > 0) {
            size_t nchunks = (std::min(remaining, kRdmaChunk) + kRdmaChunk - 1) / kRdmaChunk;
            std::vector<ibv_sge> sge(nchunks);
            std::vector<ibv_send_wr> wr(nchunks);
            for (size_t i = 0; i < nchunks; ++i) {
                size_t cur = std::min(kRdmaChunk, remaining);
                sge[i].addr   = (uint64_t)(data + offset);
                sge[i].length = (uint32_t)cur;
                sge[i].lkey   = mr->lkey;
                memset(&wr[i], 0, sizeof(wr[i]));
                wr[i].sg_list   = &sge[i];
                wr[i].num_sge   = 1;
                wr[i].opcode    = IBV_WR_SEND;
                wr[i].send_flags = (i + 1 == nchunks) ? IBV_SEND_SIGNALED : 0;
                wr[i].next = (i + 1 < nchunks) ? &wr[i + 1] : nullptr;
                offset    += cur;
                remaining -= cur;
            }
            ibv_send_wr* bad = nullptr;
            if (ibv_post_send(qp, &wr[0], &bad))
                throw std::runtime_error("post_send failed");
            poll_cq(send_cq, (int)nchunks);
        }
    }

    // RDMA RECV (matches Concord)
    void recv_data(uint8_t* buf, size_t size, ibv_mr* mr, bool sync = true) {
        // Pre-post RECV
        ibv_sge sge;
        sge.addr   = (uint64_t)buf;
        sge.length = (uint32_t)size;
        sge.lkey   = mr->lkey;
        ibv_recv_wr wr, *bad;
        memset(&wr, 0, sizeof(wr));
        wr.sg_list = &sge;
        wr.num_sge = 1;
        if (ibv_post_recv(qp, &wr, &bad))
            throw std::runtime_error("post_recv failed");
        if (sync)
            poll_cq(recv_cq, 1);
    }
};

// ---- TCP accept/connect helpers ----
static int tcp_listen(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket() failed");
    int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0)
        throw std::runtime_error("bind() failed");
    if (listen(fd, 16) < 0)
        throw std::runtime_error("listen() failed");
    return fd;
}

static int tcp_accept(int listen_fd) {
    sockaddr_in peer;
    socklen_t len = sizeof(peer);
    int fd = accept(listen_fd, (sockaddr*)&peer, &len);
    if (fd < 0) throw std::runtime_error("accept() failed");
    return fd;
}

static int tcp_connect(const char* ip, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket() failed");
    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    inet_pton(AF_INET, ip, &addr.sin_addr);
    if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0)
        throw std::runtime_error("connect() failed: " + std::string(ip) + ":" + std::to_string(port));
    return fd;
}

// ---- benchmark ----
struct Config {
    int    n_qps       = 1;
    size_t block_size  = 64 * 1024 * 1024;  // bytes per transfer
    int    n_iters     = 10;
    int    base_port   = 12000;
    std::string server_ip = "127.0.0.1";
    bool   is_server   = true;
    bool   warmup      = true;
};

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n_qps" && i + 1 < argc)       cfg.n_qps      = std::stoi(argv[++i]);
        else if (arg == "--block_size" && i + 1 < argc) {
            std::string v = argv[++i];
            size_t mul = 1;
            if (v.back() == 'M' || v.back() == 'm') { mul = 1024*1024; v.pop_back(); }
            else if (v.back() == 'K' || v.back() == 'k') { mul = 1024; v.pop_back(); }
            else if (v.back() == 'G' || v.back() == 'g') { mul = 1024*1024*1024; v.pop_back(); }
            cfg.block_size = (size_t)std::stoul(v) * mul;
        }
        else if (arg == "--n_iters" && i + 1 < argc) cfg.n_iters    = std::stoi(argv[++i]);
        else if (arg == "--port" && i + 1 < argc)    cfg.base_port  = std::stoi(argv[++i]);
        else if (arg == "--ip" && i + 1 < argc)      cfg.server_ip  = argv[++i];
        else if (arg == "--server")                   cfg.is_server  = true;
        else if (arg == "--client")                   cfg.is_server  = false;
        else if (arg == "--no_warmup")                cfg.warmup     = false;
    }
    return cfg;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    // Open device
    int num_devices = 0;
    ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        std::cerr << "No IB devices found" << std::endl;
        return 1;
    }
    ibv_device* dev = dev_list[0];
    ibv_context* ctx = ibv_open_device(dev);
    if (!ctx) { std::cerr << "ibv_open_device failed" << std::endl; return 1; }
    ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd)  { std::cerr << "ibv_alloc_pd failed" << std::endl; return 1; }

    std::cout << "Device: " << ibv_get_device_name(dev) << std::endl;
    std::cout << "Config: n_qps=" << cfg.n_qps
              << " block_size=" << (cfg.block_size / (1024*1024)) << "MB"
              << " n_iters=" << cfg.n_iters
              << " role=" << (cfg.is_server ? "server" : "client")
              << std::endl;

    // Allocate buffers
    size_t total_buf = (size_t)cfg.n_qps * cfg.block_size * 2;
    std::vector<uint8_t> send_buf(total_buf, 0);
    std::vector<uint8_t> recv_buf(total_buf, 0);
    // Fill send buffer with pattern
    for (size_t i = 0; i < total_buf; ++i) send_buf[i] = (uint8_t)(i & 0xFF);

    ibv_mr* send_mr = ibv_reg_mr(pd, send_buf.data(), total_buf,
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    ibv_mr* recv_mr = ibv_reg_mr(pd, recv_buf.data(), total_buf,
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!send_mr || !recv_mr) {
        std::cerr << "ibv_reg_mr failed" << std::endl;
        return 1;
    }

    // ---- Setup TCP connections + QPs ----
    std::vector<Lane> lanes(cfg.n_qps);
    std::vector<int>  conn_fds(cfg.n_qps);

    if (cfg.is_server) {
        int listen_fd = tcp_listen(cfg.base_port);
        std::cout << "Server listening on port " << cfg.base_port << std::endl;
        for (int i = 0; i < cfg.n_qps; ++i) {
            conn_fds[i] = tcp_accept(listen_fd);
        }
        close(listen_fd);

        // Create QPs and exchange info
        for (int i = 0; i < cfg.n_qps; ++i) {
            lanes[i].ctx = ctx;
            lanes[i].pd  = pd;
            lanes[i].connect_qp(conn_fds[i], send_mr, 0);
        }
    } else {
        // Connect to server: one TCP connection per QP
        for (int i = 0; i < cfg.n_qps; ++i) {
            conn_fds[i] = tcp_connect(cfg.server_ip.c_str(), cfg.base_port);
        }

        for (int i = 0; i < cfg.n_qps; ++i) {
            lanes[i].ctx = ctx;
            lanes[i].pd  = pd;
            lanes[i].connect_qp(conn_fds[i], send_mr, 1);
        }
    }

    // Barrier: send/recv a byte on first connection
    {
        uint8_t b = 0;
        if (cfg.is_server) {
            recv(conn_fds[0], &b, 1, MSG_WAITALL);
            send(conn_fds[0], &b, 1, 0);
        } else {
            send(conn_fds[0], &b, 1, 0);
            recv(conn_fds[0], &b, 1, MSG_WAITALL);
        }
    }
    if (!cfg.is_server)
        std::cout << "All " << cfg.n_qps << " QPs connected. Starting benchmark..." << std::endl;

    // ---- Warmup ----
    if (cfg.warmup) {
        size_t warm_sz = cfg.block_size;
        for (int q = 0; q < cfg.n_qps; ++q) {
            uint8_t* s = send_buf.data() + (size_t)q * cfg.block_size * 2;
            uint8_t* r = recv_buf.data() + (size_t)q * cfg.block_size * 2;
            if (cfg.is_server) {
                lanes[q].recv_data(r, warm_sz, recv_mr);
                lanes[q].send_data(s, warm_sz, send_mr);
            } else {
                lanes[q].send_data(s, warm_sz, send_mr);
                lanes[q].recv_data(r, warm_sz, recv_mr);
            }
        }
        std::cout << "Warmup done." << std::endl;
    }

    // ---- Benchmark ----
    // Each iteration: server sends to client, client receives, then client sends back.
    // We time the send+recv on the initiator side.
    struct IterResult {
        double total_s;
        double bw_gbps;   // effective bandwidth = (n_qps * block_size * 2) / total_s
    };
    std::vector<IterResult> results;
    results.reserve(cfg.n_iters);

    for (int iter = 0; iter < cfg.n_iters; ++iter) {
        double t0 = now_sec();

        // Launch all QP transfers in parallel using threads
        std::vector<std::thread> threads;
        threads.reserve(cfg.n_qps);
        std::atomic<int> errors{0};

        for (int q = 0; q < cfg.n_qps; ++q) {
            uint8_t* s = send_buf.data() + (size_t)q * cfg.block_size * 2;
            uint8_t* r = recv_buf.data() + (size_t)q * cfg.block_size * 2;
            threads.emplace_back([&, q, s, r]() {
                try {
                    if (cfg.is_server) {
                        lanes[q].recv_data(r, cfg.block_size, recv_mr);
                        lanes[q].send_data(s, cfg.block_size, send_mr);
                    } else {
                        lanes[q].send_data(s, cfg.block_size, send_mr);
                        lanes[q].recv_data(r, cfg.block_size, recv_mr);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "QP " << q << " error: " << e.what() << std::endl;
                    errors++;
                }
            });
        }
        for (auto& t : threads) t.join();

        if (errors > 0) {
            std::cerr << "Iter " << iter << ": " << errors << " errors, aborting" << std::endl;
            break;
        }

        double t1 = now_sec();
        double elapsed = t1 - t0;
        double total_bytes = (double)cfg.n_qps * (double)cfg.block_size * 2.0; // send + recv
        double bw = total_bytes * 8.0 / elapsed / 1e9; // Gbps

        results.push_back({elapsed, bw});
    }

    // ---- Report ----
    if (results.empty()) {
        std::cerr << "No results collected." << std::endl;
        return 1;
    }

    std::cout << "\n========== Results ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "Iter " << i << ": " << results[i].total_s << "s  "
                  << results[i].bw_gbps << " Gbps" << std::endl;
    }

    // Stats
    double sum = 0, min_bw = 1e9, max_bw = 0;
    for (auto& r : results) {
        sum += r.bw_gbps;
        if (r.bw_gbps < min_bw) min_bw = r.bw_gbps;
        if (r.bw_gbps > max_bw) max_bw = r.bw_gbps;
    }
    double avg = sum / (double)results.size();

    // Per-QP effective bandwidth
    double per_qp_bw = (double)cfg.block_size * 2.0 * 8.0 / (avg / 1e9) / (double)cfg.n_qps;
    // Actually compute per-QP from average iteration time
    double avg_time = sum / (double)results.size();
    double per_qp_gbps = (double)cfg.block_size * 2.0 * 8.0 / (avg_time * (double)cfg.n_qps / (double)cfg.n_qps) / 1e9;
    // Simpler: per-QP gbps = total_bw / n_qps
    double per_qp_gbps_simple = avg / (double)cfg.n_qps;

    std::cout << "\n----- Summary -----" << std::endl;
    std::cout << "n_qps      : " << cfg.n_qps << std::endl;
    std::cout << "block_size : " << (cfg.block_size / (1024*1024)) << " MB" << std::endl;
    std::cout << "n_iters    : " << results.size() << std::endl;
    std::cout << "avg time   : " << (sum / (double)results.size()) << " s" << std::endl;
    std::cout << "avg BW     : " << avg << " Gbps  (aggregate)" << std::endl;
    std::cout << "min BW     : " << min_bw << " Gbps" << std::endl;
    std::cout << "max BW     : " << max_bw << " Gbps" << std::endl;
    std::cout << "per-QP BW  : " << (avg / (double)cfg.n_qps) << " Gbps" << std::endl;
    std::cout << "per-QP MB/s: " << (avg / (double)cfg.n_qps * 1e9 / 8.0 / (1024*1024)) << " MB/s" << std::endl;

    // Cleanup
    for (auto& l : lanes) {
        if (l.qp)      { ibv_destroy_qp(l.qp); l.qp = nullptr; }
        if (l.recv_cq) { ibv_destroy_cq(l.recv_cq); l.recv_cq = nullptr; }
        if (l.send_cq) { ibv_destroy_cq(l.send_cq); l.send_cq = nullptr; }
    }
    for (int fd : conn_fds) close(fd);

    ibv_dereg_mr(send_mr);
    ibv_dereg_mr(recv_mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dev_list);

    return 0;
}
