#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <iostream>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

// RDMA headers
#include <infiniband/verbs.h>

#include "rdma_device_utils.h"

// ISA-L erasure coding
#include <isa-l/erasure_code.h>

// RS encode thread pool
static constexpr int kRsPoolSize = 16;
static constexpr const char* kRsCpuListEnv = "FRCHECK_RS_CPU_LIST";

struct RsEncodeJob {
    int len = 0;
    int k = 0;
    int m = 0;
    unsigned char* g_tbls = nullptr;
    unsigned char** data_ptrs = nullptr;
    unsigned char** parity_ptrs = nullptr;
};

struct RsPoolWorkerCtx {
    class FRCheckNative* self = nullptr;
    int wid = 0;
};

namespace py = pybind11;

static inline uint64_t frcheck_now_us() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// RDMA structures
// ---------------------------------------------------------------------------
struct RdmaConnInfo {
    uint32_t qp_num;
    uint16_t lid;
    uint8_t gid[16];
} __attribute__((packed));

struct RdmaBuffer {
    ibv_mr* mr;
    uintptr_t addr;
    size_t size;
};

// ---------------------------------------------------------------------------
// RdmaConnectionChannel — wraps a TCP socket for QP exchange + RDMA send/recv
// ---------------------------------------------------------------------------
static constexpr size_t FRCHECK_RDMA_CHUNK = 64ULL * 1024 * 1024;     // 64 MB per RDMA op
static constexpr int    FRCHECK_MAX_WR = 64;

struct ConnectHello {
    int32_t rank;
    int32_t lane_id;
};

class FRCheckRdmaChannel {
public:
    // ECLATIN/ECNaive-aligned: each channel owns dedicated send/recv CQs.
    FRCheckRdmaChannel(ibv_context* ctx, ibv_pd* pd,
                       int tcp_sock, int peer_rank,
                       std::map<uintptr_t, RdmaBuffer>* bufs,
                       std::mutex* buf_mtx)
        : ctx_(ctx), pd_(pd),
          send_cq_(nullptr), recv_cq_(nullptr),
          tcp_sock_(tcp_sock), peer_rank_(peer_rank),
          bufs_(bufs), buf_mtx_(buf_mtx),
          qp_(nullptr),
          connected_(false)
    {
        send_cq_ = ibv_create_cq(ctx_, 256, nullptr, nullptr, 0);
        recv_cq_ = ibv_create_cq(ctx_, 256, nullptr, nullptr, 0);
        if (!send_cq_ || !recv_cq_) {
            if (send_cq_) { ibv_destroy_cq(send_cq_); send_cq_ = nullptr; }
            if (recv_cq_) { ibv_destroy_cq(recv_cq_); recv_cq_ = nullptr; }
            throw std::runtime_error("FRCheck RDMA: failed to create per-channel CQs");
        }

        ibv_qp_init_attr attr{};
        attr.send_cq = send_cq_;
        attr.recv_cq = recv_cq_;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = FRCHECK_MAX_WR;
        attr.cap.max_recv_wr = FRCHECK_MAX_WR;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        qp_ = ibv_create_qp(pd_, &attr);
        if (!qp_)
            throw std::runtime_error("FRCheck RDMA: failed to create QP");
    }

    ~FRCheckRdmaChannel() {
        if (qp_) ibv_destroy_qp(qp_);
        if (recv_cq_) { ibv_destroy_cq(recv_cq_); recv_cq_ = nullptr; }
        if (send_cq_) { ibv_destroy_cq(send_cq_); send_cq_ = nullptr; }
        if (tcp_sock_ >= 0) close(tcp_sock_);
    }

    int peer_rank() const { return peer_rank_; }
    bool is_connected() const { return connected_; }

    // Break blocking send/recv during shutdown (ECLATIN-style conn cleanup).
    void abort_connection() {
        connected_ = false;
        if (tcp_sock_ >= 0) {
            shutdown(tcp_sock_, SHUT_RDWR);
        }
    }

    ibv_mr* find_mr(uintptr_t addr, size_t size) {
        std::lock_guard<std::mutex> lock(*buf_mtx_);
        for (auto& [reg_addr, buf] : *bufs_) {
            if (addr >= reg_addr && (addr + size) <= (reg_addr + buf.size))
                return buf.mr;
        }
        return nullptr;
    }

    // Exchange QP info with peer over TCP then connect QP
    void exchange_and_connect() {
        RdmaConnInfo local = get_local_info();
        RdmaConnInfo remote;
        memset(&remote, 0, sizeof(remote));

        // Both sides send then recv (symmetric, safe since TCP is full-duplex)
        ssize_t ns = send(tcp_sock_, &local, sizeof(local), 0);
        if (ns != (ssize_t)sizeof(local))
            throw std::runtime_error("FRCheck RDMA: failed to send conn info");
        ssize_t nr = recv(tcp_sock_, &remote, sizeof(remote), MSG_WAITALL);
        if (nr != (ssize_t)sizeof(remote))
            throw std::runtime_error("FRCheck RDMA: failed to recv conn info");

        connect_qp(remote);
    }

    // RDMA SEND data to peer (blocking)
    void send_data(const uint8_t* data, size_t size,
                   const std::function<void()>& wait_cb = nullptr,
                   const std::function<void()>& done_cb = nullptr) {
        std::lock_guard<std::mutex> lock(send_mtx_);
        if (!connected_)
            throw std::runtime_error("FRCheck RDMA: channel not connected");

        if (wait_cb) wait_cb();
        // Send size over TCP
        uint64_t net_sz = htobe64(size);
        if (send(tcp_sock_, &net_sz, sizeof(net_sz), 0) != sizeof(net_sz))
            throw std::runtime_error("FRCheck RDMA: failed to send size");
        uint8_t ack;
        if (recv(tcp_sock_, &ack, 1, MSG_WAITALL) != 1)
            throw std::runtime_error("FRCheck RDMA: failed to recv ack");
        if (done_cb) done_cb();

        ibv_mr* mr = find_mr((uintptr_t)data, size);
        if (!mr) {
            throw std::runtime_error(
                "FRCheck RDMA: unregistered send buffer (addr=0x" +
                std::to_string((uintptr_t)data) + " size=" + std::to_string(size) + ")");
        }
        send_chunked(data, size, mr, wait_cb, done_cb);
    }

    // RDMA RECEIVE data from peer (blocking)
    size_t recv_data(uint8_t* buf, size_t buf_size,
                     const std::function<void()>& wait_cb = nullptr,
                     const std::function<void()>& done_cb = nullptr) {
        std::lock_guard<std::mutex> lock(recv_mtx_);
        if (!connected_)
            throw std::runtime_error("FRCheck RDMA: channel not connected");

        if (wait_cb) wait_cb();
        uint64_t net_sz;
        if (recv(tcp_sock_, &net_sz, sizeof(net_sz), MSG_WAITALL) != sizeof(net_sz))
            throw std::runtime_error("FRCheck RDMA: failed to recv size");
        size_t size = be64toh(net_sz);
        if (size > buf_size)
            throw std::runtime_error("FRCheck RDMA: recv size " + std::to_string(size) +
                " exceeds buffer " + std::to_string(buf_size) +
                " (delta=" + std::to_string(size - buf_size) + ")");

        uint8_t ack = 1;
        if (send(tcp_sock_, &ack, 1, 0) != 1)
            throw std::runtime_error("FRCheck RDMA: failed to send ack");
        if (done_cb) done_cb();

        ibv_mr* mr = find_mr((uintptr_t)buf, size);
        if (!mr) {
            throw std::runtime_error(
                "FRCheck RDMA: unregistered recv buffer (addr=0x" +
                std::to_string((uintptr_t)buf) + " size=" + std::to_string(size) + ")");
        }
        recv_chunked(buf, size, mr, wait_cb, done_cb);
        return size;
    }

private:
    RdmaConnInfo get_local_info() {
        RdmaConnInfo info;
        memset(&info, 0, sizeof(info));
        info.qp_num = qp_->qp_num;
        ibv_port_attr pa;
        if (ibv_query_port(ctx_, 1, &pa) == 0)
            info.lid = pa.lid;
        ibv_gid gid;
        if (ibv_query_gid(ctx_, 1, 1, &gid) == 0)
            memcpy(info.gid, &gid, 16);
        return info;
    }

    void connect_qp(const RdmaConnInfo& remote) {
        // INIT
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = 1;
        attr.pkey_index = 0;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(qp_, &attr,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
            throw std::runtime_error("FRCheck RDMA: INIT failed");

        ibv_port_attr pa;
        if (ibv_query_port(ctx_, 1, &pa) != 0)
            throw std::runtime_error("FRCheck RDMA: query port failed");
        ibv_mtu mtu = pa.active_mtu;
        bool use_gid = (remote.lid == 0);

        // RTR
        attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = mtu;
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
            memcpy(&attr.ah_attr.grh.dgid, remote.gid, 16);
            attr.ah_attr.grh.flow_label = 0;
            attr.ah_attr.grh.sgid_index = 1;
            attr.ah_attr.grh.hop_limit = 255;
            attr.ah_attr.grh.traffic_class = 0;
        }
        if (ibv_modify_qp(qp_, &attr,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
            throw std::runtime_error("FRCheck RDMA: RTR failed");

        // RTS
        attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        if (ibv_modify_qp(qp_, &attr,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
            throw std::runtime_error("FRCheck RDMA: RTS failed");
        connected_ = true;
    }

    void send_chunked(const uint8_t* data, size_t total, ibv_mr* mr,
                      const std::function<void()>& wait_cb = nullptr,
                      const std::function<void()>& done_cb = nullptr) {
        _send_chunked(data, total, mr, wait_cb, done_cb);
    }

    void recv_chunked(uint8_t* buf, size_t total, ibv_mr* mr,
                      const std::function<void()>& wait_cb = nullptr,
                      const std::function<void()>& done_cb = nullptr) {
        _recv_chunked(buf, total, mr, wait_cb, done_cb);
    }

    void _send_chunked(const uint8_t* data, size_t total, ibv_mr* mr,
                       const std::function<void()>& wait_cb,
                       const std::function<void()>& done_cb) {
        size_t remaining = total, offset = 0;
        while (remaining > 0) {
            if (wait_cb) wait_cb();
            size_t nchunks = (std::min(remaining, FRCHECK_RDMA_CHUNK) + FRCHECK_RDMA_CHUNK - 1) / FRCHECK_RDMA_CHUNK;
            std::vector<ibv_sge> sge(nchunks);
            std::vector<ibv_send_wr> wr(nchunks);
            for (size_t i = 0; i < nchunks; ++i) {
                size_t cur = std::min(FRCHECK_RDMA_CHUNK, remaining);
                sge[i].addr = (uint64_t)(data + offset);
                sge[i].length = (uint32_t)cur;
                sge[i].lkey = mr->lkey;
                memset(&wr[i], 0, sizeof(wr[i]));
                wr[i].sg_list = &sge[i];
                wr[i].num_sge = 1;
                wr[i].opcode = IBV_WR_SEND;
                wr[i].send_flags = (i + 1 == nchunks) ? IBV_SEND_SIGNALED : 0;
                wr[i].next = (i + 1 < nchunks) ? &wr[i + 1] : nullptr;
                offset += cur;
                remaining -= cur;
            }
            ibv_send_wr* bad = nullptr;
            if (ibv_post_send(qp_, &wr[0], &bad))
                throw std::runtime_error("FRCheck RDMA: post_send failed");
            poll_cq(send_cq_, (int)nchunks);
            if (done_cb) done_cb();
        }
    }

    void _recv_chunked(uint8_t* buf, size_t total, ibv_mr* mr,
                       const std::function<void()>& wait_cb,
                       const std::function<void()>& done_cb) {
        size_t remaining = total, offset = 0;
        while (remaining > 0) {
            if (wait_cb) wait_cb();
            size_t nchunks = (std::min(remaining, FRCHECK_RDMA_CHUNK) + FRCHECK_RDMA_CHUNK - 1) / FRCHECK_RDMA_CHUNK;
            std::vector<ibv_sge> sge(nchunks);
            std::vector<ibv_recv_wr> wr(nchunks);
            for (size_t i = 0; i < nchunks; ++i) {
                size_t cur = std::min(FRCHECK_RDMA_CHUNK, remaining);
                sge[i].addr = (uint64_t)(buf + offset);
                sge[i].length = (uint32_t)cur;
                sge[i].lkey = mr->lkey;
                memset(&wr[i], 0, sizeof(wr[i]));
                wr[i].sg_list = &sge[i];
                wr[i].num_sge = 1;
                wr[i].next = (i + 1 < nchunks) ? &wr[i + 1] : nullptr;
                offset += cur;
                remaining -= cur;
            }
            ibv_recv_wr* bad = nullptr;
            if (ibv_post_recv(qp_, &wr[0], &bad))
                throw std::runtime_error("FRCheck RDMA: post_recv failed");
            poll_cq(recv_cq_, (int)nchunks);
            if (done_cb) done_cb();
        }
    }

    // ECLATIN-style: poll one completion at a time on this channel's private CQ.
    void poll_cq(ibv_cq* cq, int count) {
        int done = 0;
        while (done < count) {
            ibv_wc wc;
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) throw std::runtime_error("FRCheck RDMA: poll CQ error");
            if (n > 0) {
                if (wc.status != IBV_WC_SUCCESS)
                    throw std::runtime_error(
                        "FRCheck RDMA: CQ error status=" + std::to_string(wc.status));
                ++done;
            }
        }
    }

    ibv_context* ctx_;
    ibv_pd* pd_;
    ibv_cq* send_cq_;
    ibv_cq* recv_cq_;
    int tcp_sock_;
    int peer_rank_;
    std::map<uintptr_t, RdmaBuffer>* bufs_;
    std::mutex* buf_mtx_;
    ibv_qp* qp_;
    bool connected_;
    std::mutex send_mtx_;
    std::mutex recv_mtx_;
};

// ---------------------------------------------------------------------------
// StripePlan — pre-compiled role assignment for a POA row
// ---------------------------------------------------------------------------
enum class StripeRole { SOURCE, ENCODER, PARITY_TARGET };

struct StripePlan {
    int stripe_id;
    std::vector<int> row;               // POA row (node IDs 1..n)
    StripeRole role;                    // My role in this stripe
    int encoder_node_id;                // node ID of encoder (row[n-2])
    int parity_target_node_id;          // node ID of parity target (row[n-1])
    std::vector<int> source_node_ids;   // node IDs of sources (row[0..n-3])
};

struct RecoveryFailedTarget {
    int failed_node = 0;
    int failed_pos = 0;
    int original_role = 0;
};

struct RecoveryStripePlan {
    int stripe_id = 0;
    bool dual_failure = false;
    int failed_node = 0;
    int failed_pos = 0;
    int decoder_node = 0;
    int decoder_pos = 0;
    std::vector<int> helper_nodes;
    std::vector<int> helper_positions;
    std::vector<int> survivor_positions;
    std::vector<int> failed_nodes;
    std::vector<RecoveryFailedTarget> failed_targets;
    int original_role = 0;
};

// ---------------------------------------------------------------------------
// POA file helpers
// ---------------------------------------------------------------------------
namespace {

void trim_inplace(std::string& s) {
    while (!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
}

bool is_comment_or_empty(const std::string& line) {
    std::string t = line;
    trim_inplace(t);
    return t.empty() || t[0] == '#';
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// FRCheckNative — main class
// ---------------------------------------------------------------------------
class FRCheckNative {
public:
    FRCheckNative(const std::string& poa_path)
        : poa_path_(poa_path), stopped_(false),
          rdma_ctx_(nullptr), rdma_pd_(nullptr),
          group_size_(0), rank_in_group_(-1),
          acceptor_fd_(-1), acceptor_thread_stop_(false)
    {
        std::ifstream in(poa_path);
        if (in.good()) {
            load_poa_(in);
        } else {
            // File not found — auto-generate POA from n encoded in filename
            int n = parse_n_from_path_(poa_path);
            generate_poa_(n);
        }
        init_encode_tables_();
    }

    FRCheckNative(int n)
        : poa_path_("(generated)"), stopped_(false),
          rdma_ctx_(nullptr), rdma_pd_(nullptr),
          group_size_(0), rank_in_group_(-1),
          acceptor_fd_(-1), acceptor_thread_stop_(false)
    {
        generate_poa_(n);
        init_encode_tables_();
    }

    ~FRCheckNative() {
        // Lightweight destructor: free POA/ISA-L tables only.
        // Full RDMA/worker teardown is explicit via stop() (ECLATIN-style).
        if (a_mat_) { free(a_mat_); a_mat_ = nullptr; }
        if (g_tbls_) { free(g_tbls_); g_tbls_ = nullptr; }
        if (decode_tbls_) { free(decode_tbls_); decode_tbls_ = nullptr; }
    }

    // ---- POA query (existing) ----
    int n() const { return n_; }
    int num_stripes() const { return (int)table_.size(); }
    int entry(int r, int c) const { return table_.at(r).at(c); }
    std::vector<int> row(int r) const { return table_.at(r); }
    std::string path() const { return poa_path_; }
    void abort_all_channels_() {
        for (auto& peer_row : channels_) {
            for (auto* ch : peer_row) {
                if (ch) ch->abort_connection();
            }
        }
    }
    void stop_recovery_runtime() {
        recovery_workers_stop_ = true;
        helper_cv_.notify_all();
        decoder_cv_.notify_all();
        failed_cv_.notify_all();
        recovery_wait_cv_.notify_all();
        recovery_workers_join_();
    }

    void cleanup_recovery_runtime() {
        stop_recovery_runtime();
        cleanup_rdma_();
    }

    void stop() {
        if (stopped_.exchange(true)) return;

        stop_recovery_runtime();
        mirror_cv_.notify_all();

        abort_all_channels_();
        shutdown_stripe_workers();
        mirror_worker_shutdown();
        rs_pool_shutdown();
        cleanup_rdma_();
    }
    bool is_stopped() const { return stopped_; }

    // ---- New: RDMA init ----
    void init_rdma(int group_size, int rank_in_group,
                   uint16_t base_port,
                   const std::string& my_ip,
                   const std::vector<std::string>& peer_ips,
                   bool use_rdma)
    {
        if (!use_rdma) {
            throw std::runtime_error("FRCheck: RDMA is required (no ASIO fallback)");
        }
        if (group_size != n_) {
            throw std::runtime_error("FRCheck: group_size must equal POA n");
        }
        group_size_ = group_size;
        rank_in_group_ = rank_in_group;
        my_ip_ = my_ip;
        stopped_ = false;

        // Init RDMA device
        init_ibv_();

        // Build full-mesh TCP topology:
        // Each node: acceptor on base_port + rank_in_group
        // For j < rank: connect to j; for j > rank: accept from j
        uint16_t listen_port = base_port + (uint16_t)rank_in_group;
        if (debug_)
            std::cout << "[FRCheck RDMA] rank_in_group=" << rank_in_group
                      << " listening on port " << listen_port << std::endl;

        acceptor_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (acceptor_fd_ < 0)
            throw std::runtime_error("FRCheck RDMA: failed to create acceptor socket");
        int reuse = 1;
        setsockopt(acceptor_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        struct sockaddr_in listen_addr;
        memset(&listen_addr, 0, sizeof(listen_addr));
        listen_addr.sin_family = AF_INET;
        listen_addr.sin_port = htons(listen_port);
        if (inet_pton(AF_INET, my_ip.c_str(), &listen_addr.sin_addr) <= 0)
            throw std::runtime_error("FRCheck RDMA: invalid listen IP: " + my_ip);
        if (bind(acceptor_fd_, (struct sockaddr*)&listen_addr, sizeof(listen_addr)) < 0)
            throw std::runtime_error("FRCheck RDMA: bind failed on port " + std::to_string(listen_port));
        if (listen(acceptor_fd_, std::max(group_size * 16, 128)) < 0)
            throw std::runtime_error("FRCheck RDMA: listen failed");

        num_lanes_ = (int)table_.size();
        if (num_lanes_ <= 0)
            throw std::runtime_error("FRCheck RDMA: POA table has no stripes");
        if (debug_)
            std::cout << "[FRCheck RDMA] rank=" << rank_in_group
                      << " num_lanes=" << num_lanes_
                      << " (connections per peer=" << num_lanes_ << ")" << std::endl;

        // Launch accept thread
        acceptor_thread_stop_ = false;
        accept_thread_ = std::thread([this]() { accept_loop_(); });

        // channels_[peer_rg][lane_id] = channel (lane_id == stripe_id)
        channels_.assign(group_size, std::vector<FRCheckRdmaChannel*>(num_lanes_, nullptr));

        auto connect_outbound = [&](int peer, int lane) {
            uint16_t peer_port = base_port + (uint16_t)peer;
            std::string peer_ip = (peer < (int)peer_ips.size()) ? peer_ips[peer] : my_ip;
            int sock = create_tcp_connect(peer_ip, peer_port);
            ConnectHello hello{rank_in_group_, lane};
            if (send(sock, &hello, sizeof(hello), 0) != (ssize_t)sizeof(hello)) {
                close(sock);
                throw std::runtime_error(
                    "FRCheck RDMA: failed to send hello to peer=" + std::to_string(peer) +
                    " lane=" + std::to_string(lane));
            }
            auto ch = std::make_unique<FRCheckRdmaChannel>(
                rdma_ctx_, rdma_pd_,
                sock, peer, &registered_bufs_, &buf_mtx_);
            ch->exchange_and_connect();
            channels_[peer][lane] = ch.release();
            channel_owners_.push_back(
                std::unique_ptr<FRCheckRdmaChannel>(channels_[peer][lane]));
        };

        // Connect to lower ranks: one TCP+QP per (peer, stripe lane)
        for (int peer = 0; peer < rank_in_group; ++peer) {
            for (int lane = 0; lane < num_lanes_; ++lane) {
                connect_outbound(peer, lane);
            }
        }

        // Wait for higher ranks to connect (one connection per lane)
        for (int peer = rank_in_group + 1; peer < group_size; ++peer) {
            for (int lane = 0; lane < num_lanes_; ++lane) {
                int sock;
                {
                    std::unique_lock<std::mutex> lk(accept_mtx_);
                    accept_cv_.wait(lk, [this, peer, lane]() {
                        return (accepted_queue_.count(peer) &&
                                accepted_queue_[peer].count(lane)) ||
                               acceptor_thread_stop_.load();
                    });
                    if (acceptor_thread_stop_)
                        throw std::runtime_error("FRCheck: acceptor thread stopped prematurely");
                    sock = accepted_queue_[peer][lane];
                    accepted_queue_[peer].erase(lane);
                    if (accepted_queue_[peer].empty())
                        accepted_queue_.erase(peer);
                }
                auto ch = std::make_unique<FRCheckRdmaChannel>(
                    rdma_ctx_, rdma_pd_,
                    sock, peer, &registered_bufs_, &buf_mtx_);
                ch->exchange_and_connect();
                channels_[peer][lane] = ch.release();
                channel_owners_.push_back(
                    std::unique_ptr<FRCheckRdmaChannel>(channels_[peer][lane]));
            }
        }

        n_connected_ = group_size_;
        if (debug_)
            std::cout << "[FRCheck RDMA] rank=" << rank_in_group
                      << " all " << group_size_ << " peers x " << num_lanes_
                  << " lanes connected (per-channel CQ pairs="
                  << (group_size_ - 1) * num_lanes_ << ")" << std::endl;

        // Init RS encode thread pool
        rs_pool_init();

        // Pre-compile stripe plans
        compile_stripe_plans_();

        // Start per-stripe workers (needs stripe_plans_ populated)
        mirror_worker_init();
        init_stripe_workers();
    }

    int group_size() const { return group_size_; }
    int rank_in_group() const { return rank_in_group_; }

    // ---- Buffer registration ----
    void register_buffer(uintptr_t addr, size_t size) {
        if (!rdma_pd_) return;
        ibv_mr* mr = ibv_reg_mr(rdma_pd_, (void*)addr, size,
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                 IBV_ACCESS_REMOTE_READ);
        if (!mr) {
            if (require_registered_mr_) {
                throw std::runtime_error(
                    "FRCheck RDMA: ibv_reg_mr failed for save pipeline (addr=0x" +
                    std::to_string(addr) + " size=" + std::to_string(size) +
                    ", GPU pointers need nvidia-peermem for GDR)");
            }
            std::cerr << "[FRCheck RDMA] WARNING: ibv_reg_mr failed for addr=0x"
                      << std::hex << addr << " size=" << std::dec << size
                      << " (GPU pointers need nvidia-peermem for GDR)" << std::endl;
            return;
        }
        std::lock_guard<std::mutex> lk(buf_mtx_);
        registered_bufs_[addr] = {mr, addr, size};
    }

    void unregister_buffer(uintptr_t addr) {
        std::lock_guard<std::mutex> lk(buf_mtx_);
        auto it = registered_bufs_.find(addr);
        if (it != registered_bufs_.end()) {
            if (it->second.mr) ibv_dereg_mr(it->second.mr);
            registered_bufs_.erase(it);
        }
    }

    // ---- GDR capability detection ----
    static bool gdr_available() {
        // Check for nvidia-peermem kernel module — enables GPU memory as ibv MR
        std::ifstream mod("/proc/modules");
        if (!mod) return false;
        std::string line;
        while (std::getline(mod, line)) {
            if (line.rfind("nvidia_peermem", 0) == 0) return true;
            if (line.rfind("nvidia-peermem", 0) == 0) return true;
        }
        return false;
    }

    // ---- Stripe decode (hardware recovery) ----
    void submit_stripe_decode(
        int k,
        const std::vector<int>& survivor_positions,
        int lost_position,
        const std::vector<uintptr_t>& survivor_addrs,
        uintptr_t recovered_addr,
        size_t block_size)
    {
        if (stopped_) return;
        int surviving_count = k;
        if ((int)survivor_positions.size() != surviving_count ||
            (int)survivor_addrs.size() != surviving_count) {
            std::cerr << "FRCheck decode: survivor count mismatch" << std::endl;
            return;
        }

        std::lock_guard<std::mutex> lk(decode_mtx_);

        // Build decode tables for this stripe
        init_decode_tables_(k, survivor_positions, lost_position);
        if (!decode_tbls_) {
            std::cerr << "FRCheck decode: failed to init decode tables" << std::endl;
            return;
        }

        // Run parallel decode via RS pool (reuses encode pool)
        if (rs_pool_inited_.load(std::memory_order_acquire)) {
            RsEncodeJob job;
            job.len = (int)block_size;
            job.k = surviving_count;
            job.m = 1;  // recover 1 block
            job.g_tbls = decode_tbls_;
            std::vector<unsigned char*> data_ptrs(surviving_count);
            for (int i = 0; i < surviving_count; ++i)
                data_ptrs[i] = (unsigned char*)survivor_addrs[i];
            job.data_ptrs = data_ptrs.data();
            unsigned char* out[1] = { (unsigned char*)recovered_addr };
            job.parity_ptrs = out;
            rs_pool_run_parallel_encode(job);
        } else {
            // Fallback: single-threaded decode
            std::vector<unsigned char*> data_ptrs(surviving_count);
            for (int i = 0; i < surviving_count; ++i)
                data_ptrs[i] = (unsigned char*)survivor_addrs[i];
            unsigned char* out[1] = { (unsigned char*)recovered_addr };
            ec_encode_data((int)block_size, surviving_count, 1,
                          decode_tbls_, data_ptrs.data(), out);
        }
    }

    // ---- Point-to-point RDMA (for recovery) ----
    void send_to_peer(int peer_rig, int stripe_id, uintptr_t addr, size_t size) {
        FRCheckRdmaChannel* ch = get_channel_(peer_rig, stripe_id);
        if (!ch) {
            throw std::runtime_error(
                "FRCheck send_to_peer: no channel to rig " + std::to_string(peer_rig) +
                " lane " + std::to_string(stripe_id));
        }
        ch->send_data((const uint8_t*)addr, size);
    }

    void recv_from_peer(int peer_rig, int stripe_id, uintptr_t addr, size_t size) {
        FRCheckRdmaChannel* ch = get_channel_(peer_rig, stripe_id);
        if (!ch) {
            throw std::runtime_error(
                "FRCheck recv_from_peer: no channel from rig " + std::to_string(peer_rig) +
                " lane " + std::to_string(stripe_id));
        }
        ch->recv_data((uint8_t*)addr, size);
    }

    void set_require_registered_mr(bool require) { require_registered_mr_ = require; }
    void set_debug(bool d) { debug_ = d; }

    // ---- Hardware recovery batch pipeline ----
    void init_recovery_plans(const std::vector<int>& failed_nodes_1based) {
        if (failed_nodes_1based.empty())
            throw std::runtime_error("FRCheck: init_recovery_plans requires failed nodes");
        if (failed_nodes_1based.size() > 2)
            throw std::runtime_error("FRCheck: at most 2 failed nodes per group");
        if (stripe_plans_.empty())
            throw std::runtime_error("FRCheck: compile_plans must run before init_recovery_plans");

        recovery_plans_.clear();
        recovery_dual_failure_ = (failed_nodes_1based.size() == 2);

        if (recovery_dual_failure_) {
            compile_recovery_plans_dual_(failed_nodes_1based);
        } else {
            compile_recovery_plans_single_(failed_nodes_1based[0]);
        }
    }

    void reset_recovery_batch() {
        if (recovery_batch_active_)
            throw std::runtime_error("FRCheck: recovery batch already active");
        if (encoding_batch_active_)
            throw std::runtime_error("FRCheck: encode batch active, cannot start recovery batch");

        ensure_recovery_workers_();
        reset_recovery_completion_();
        recovery_batch_active_ = true;
    }

    void submit_recovery_stripe(
        int stripe_id,
        size_t block_size,
        uintptr_t helper_block_addr,
        uintptr_t decoder_self_block_addr,
        const std::vector<uintptr_t>& decoder_helper_recv_addrs,
        const std::vector<uintptr_t>& decoder_recovered_addrs,
        uintptr_t failed_recv_buf_addr,
        uintptr_t failed_layer_buf_addr,
        size_t failed_layer_offset,
        size_t failed_ncopy,
        bool store_to_layer_buf)
    {
        if (stopped_) return;
        if (!recovery_batch_active_)
            throw std::runtime_error("FRCheck: submit_recovery_stripe without reset_recovery_batch");
        if (stripe_id < 0 || stripe_id >= (int)recovery_plans_.size())
            throw std::runtime_error("FRCheck: invalid recovery stripe_id");

        const RecoveryStripePlan& plan = recovery_plans_[stripe_id];
        int my_node = rank_in_group_ + 1;

        auto is_helper = [&]() {
            return std::find(plan.helper_nodes.begin(), plan.helper_nodes.end(), my_node)
                   != plan.helper_nodes.end();
        };

        if (is_helper() && helper_block_addr != 0) {
            RecoveryHelperTask task;
            task.stripe_id = stripe_id;
            task.block_size = block_size;
            task.helper_block = helper_block_addr;
            task.decoder_rig = plan.decoder_node - 1;
            pending_recovery_chunks_.fetch_add(1, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lk(helper_mtx_);
                helper_q_.push(std::move(task));
                wait_helper_ = true;
                helper_completed_ = false;
            }
            helper_cv_.notify_all();
        }

        if (my_node == plan.decoder_node && decoder_self_block_addr != 0) {
            RecoveryDecoderTask task;
            task.stripe_id = stripe_id;
            task.block_size = block_size;
            task.self_block = decoder_self_block_addr;
            task.helper_recv_bufs = decoder_helper_recv_addrs;
            for (int hn : plan.helper_nodes)
                task.helper_rigs.push_back(hn - 1);
            task.recovered_bufs = decoder_recovered_addrs;
            task.dual_failure = plan.dual_failure;
            task.survivor_positions = plan.survivor_positions;
            if (!plan.dual_failure) {
                task.failed_pos = plan.failed_pos;
                task.failed_rigs.push_back(plan.failed_node - 1);
            } else {
                for (const auto& ft : plan.failed_targets) {
                    task.failed_positions.push_back(ft.failed_pos);
                    task.failed_rigs.push_back(ft.failed_node - 1);
                }
            }
            pending_recovery_chunks_.fetch_add(1, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lk(decoder_mtx_);
                decoder_q_.push(std::move(task));
                wait_decoder_ = true;
                decoder_completed_ = false;
            }
            decoder_cv_.notify_all();
        }

        if (!plan.dual_failure) {
            if (my_node == plan.failed_node && failed_recv_buf_addr != 0) {
                RecoveryFailedTask task;
                task.stripe_id = stripe_id;
                task.block_size = block_size;
                task.recv_buf = failed_recv_buf_addr;
                task.decoder_rig = plan.decoder_node - 1;
                task.layer_buf = failed_layer_buf_addr;
                task.layer_offset = failed_layer_offset;
                task.ncopy = failed_ncopy;
                task.store_to_layer = store_to_layer_buf;
                pending_recovery_chunks_.fetch_add(1, std::memory_order_acq_rel);
                {
                    std::lock_guard<std::mutex> lk(failed_mtx_);
                    failed_q_.push(std::move(task));
                    wait_failed_ = true;
                    failed_completed_ = false;
                }
                failed_cv_.notify_all();
            }
        } else {
            for (const auto& ft : plan.failed_targets) {
                if (my_node != ft.failed_node || failed_recv_buf_addr == 0)
                    continue;
                RecoveryFailedTask task;
                task.stripe_id = stripe_id;
                task.block_size = block_size;
                task.recv_buf = failed_recv_buf_addr;
                task.decoder_rig = plan.decoder_node - 1;
                task.layer_buf = failed_layer_buf_addr;
                task.layer_offset = failed_layer_offset;
                task.ncopy = failed_ncopy;
                task.store_to_layer = store_to_layer_buf;
                pending_recovery_chunks_.fetch_add(1, std::memory_order_acq_rel);
                {
                    std::lock_guard<std::mutex> lk(failed_mtx_);
                    failed_q_.push(std::move(task));
                    wait_failed_ = true;
                    failed_completed_ = false;
                }
                failed_cv_.notify_all();
                break;
            }
        }
    }

    void submit_recovery_sentinel() {
        RecoveryHelperTask h_sentinel{};
        RecoveryDecoderTask d_sentinel{};
        RecoveryFailedTask f_sentinel{};
        if (wait_helper_) {
            std::lock_guard<std::mutex> lk(helper_mtx_);
            for (size_t i = 0; i < helper_threads_.size(); ++i)
                helper_q_.push(h_sentinel);
        }
        if (wait_decoder_) {
            std::lock_guard<std::mutex> lk(decoder_mtx_);
            for (size_t i = 0; i < decoder_threads_.size(); ++i)
                decoder_q_.push(d_sentinel);
        }
        if (wait_failed_) {
            std::lock_guard<std::mutex> lk(failed_mtx_);
            for (size_t i = 0; i < failed_threads_.size(); ++i)
                failed_q_.push(f_sentinel);
        }
        helper_cv_.notify_all();
        decoder_cv_.notify_all();
        failed_cv_.notify_all();
        maybe_complete_recovery_batch_();
    }

    void wait_recovery_batch() {
        std::unique_lock<std::mutex> lk(recovery_wait_mtx_);
        recovery_wait_cv_.wait(lk, [this] {
            return stopped_.load() || recovery_batch_completed_;
        });
        recovery_batch_active_ = false;
        lk.unlock();
        print_recovery_batch_profile_();
    }

    // ---- StripePlan queries for Python ----
    int get_role_for_stripe(int stripe_id) const {
        if (stripe_id < 0 || stripe_id >= (int)stripe_plans_.size())
            throw std::runtime_error("FRCheck: invalid stripe_id");
        return (int)stripe_plans_[stripe_id].role;
    }

    std::vector<int> get_source_node_ids(int stripe_id) const {
        if (stripe_id < 0 || stripe_id >= (int)stripe_plans_.size())
            throw std::runtime_error("FRCheck: invalid stripe_id");
        return stripe_plans_[stripe_id].source_node_ids;
    }

    int get_encoder_node_id(int stripe_id) const {
        if (stripe_id < 0 || stripe_id >= (int)stripe_plans_.size())
            throw std::runtime_error("FRCheck: invalid stripe_id");
        return stripe_plans_[stripe_id].encoder_node_id;
    }

    int get_parity_target_node_id(int stripe_id) const {
        if (stripe_id < 0 || stripe_id >= (int)stripe_plans_.size())
            throw std::runtime_error("FRCheck: invalid stripe_id");
        return stripe_plans_[stripe_id].parity_target_node_id;
    }

private:
    // ---- POA loading ----
    void load_poa_(std::istream& in) {
        table_.clear();
        n_ = -1;
        std::string line;
        while (std::getline(in, line)) {
            if (is_comment_or_empty(line)) continue;
            std::istringstream iss(line);
            std::vector<int> rv;
            int v;
            while (iss >> v) rv.push_back(v);
            if (rv.empty()) continue;
            if (n_ < 0) {
                n_ = (int)rv.size();
                if (n_ < 3)
                    throw std::runtime_error("FRCheck: POA must have >=3 columns");
            } else if ((int)rv.size() != n_) {
                throw std::runtime_error("FRCheck: inconsistent row width");
            }
            for (int x : rv)
                if (x < 1 || x > n_)
                    throw std::runtime_error("FRCheck: entry out of range [1,n]");
            table_.push_back(std::move(rv));
        }
        if (n_ < 0 || table_.empty())
            throw std::runtime_error("FRCheck: empty POA file");
        int expected = n_ * (n_ - 1);
        if ((int)table_.size() != expected)
            throw std::runtime_error("FRCheck: expected " + std::to_string(expected) +
                                     " rows, got " + std::to_string(table_.size()));
    }

    // ---- POA auto-generation ----
    int parse_n_from_path_(const std::string& path) {
        // Parse n from common filename patterns: poa_n{}.txt, n{}.poa, frcheck_poa_n{}.txt
        std::string fname = path;
        auto slash = fname.rfind('/');
        if (slash != std::string::npos) fname = fname.substr(slash + 1);
        // Try "n{N}" pattern
        std::string marker = "n";
        auto pos = fname.find(marker);
        if (pos == std::string::npos) marker = "N", pos = fname.find(marker);
        if (pos != std::string::npos) {
            pos += marker.size();
            int v = 0;
            while (pos < fname.size() && std::isdigit(fname[pos])) {
                v = v * 10 + (fname[pos] - '0');
                ++pos;
            }
            if (v >= 3) return v;
        }
        throw std::runtime_error("FRCheck: cannot determine n from filename: " + path +
                                 " (use path like poa_n4.txt or construct with int n)");
    }

    static bool is_prime_(int n) {
        if (n < 2) return false;
        for (int i = 2; i * i <= n; ++i)
            if (n % i == 0) return false;
        return true;
    }

    static bool is_power_of_two_(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    // GF(2^k) multiplication via exponent/log tables with primitive polynomial
    static int gf2k_mul_(int a, int b, int n, const int exp[], const int log[]) {
        if (a == 0 || b == 0) return 0;
        int sum = log[a] + log[b];
        int mod = n - 1;  // 2^k - 1
        if (sum >= mod) sum -= mod;
        return exp[sum];
    }

    // Build exponent/log tables for GF(2^k) with primitive poly
    // polys[k] = primitive polynomial for GF(2^k), k=2..8
    static void gf2k_build_tables_(int n, int exp[], int log[], int poly) {
        int val = 1;
        for (int i = 0; i < n - 1; ++i) {
            exp[i] = val;
            log[val] = i;
            val <<= 1;
            if (val >= n) val ^= poly;
        }
        exp[n - 1] = 1;  // α^{n-1} = 1
        log[1] = 0;
        log[0] = -1;  // sentinel
    }

    // Primitive polynomials for GF(2^k), k=2..8
    static int gf2k_poly_(int n) {
        switch (n) {
            case 4:   return 0b111;     // x^2 + x + 1
            case 8:   return 0b1011;    // x^3 + x + 1
            case 16:  return 0b10011;   // x^4 + x + 1
            case 32:  return 0b100101;  // x^5 + x^2 + 1
            case 64:  return 0b1000011; // x^6 + x + 1
            case 128: return 0b10001001; // x^7 + x^3 + 1
            case 256: return 0b100011101; // x^8 + x^4 + x^3 + x^2 + 1
            default:  return 0;
        }
    }

    void generate_poa_(int n) {
        if (n < 3) throw std::runtime_error("FRCheck: n must be >= 3");
        table_.clear();
        n_ = n;
        poa_path_ = "(generated n=" + std::to_string(n) + ")";

        if (is_prime_(n)) {
            // For prime n: use modular arithmetic
            // row(r, c) = [(r+0*c)%n, (r+1*c)%n, (r+2*c)%n, ..., (r+(n-1)*c)%n]
            // c ranges 1..n-1 (nonzero), r ranges 0..n-1
            for (int c = 1; c < n; ++c) {
                for (int r = 0; r < n; ++r) {
                    std::vector<int> row(n);
                    for (int k = 0; k < n; ++k) {
                        row[k] = ((r + k * c) % n) + 1;  // 1-based
                    }
                    table_.push_back(std::move(row));
                }
            }
        } else if (is_power_of_two_(n)) {
            // For n = 2^k: use GF(2^k) arithmetic with (r ⊕ α^i ⊗ c) + 1
            int poly = gf2k_poly_(n);
            if (poly == 0)
                throw std::runtime_error("FRCheck: no primitive polynomial for GF(" +
                                         std::to_string(n) + ")");
            // Build exp/log tables
            int exp[256], log[256];
            gf2k_build_tables_(n, exp, log, poly);
            log[0] = -1;

            // Precompute multipliers: α^0, α^1, ..., α^{n-2}
            // α^i = exp[i] (since exp[i] = α^i)
            // For n = 2^k, α^{n-1} = 1
            // We need multipliers for columns 1..n-1
            std::vector<int> mult(n - 1);
            for (int i = 0; i < n - 1; ++i) {
                mult[i] = exp[i];  // α^i
            }

            // c ranges over non-zero GF(2^k) elements, r over all elements
            for (int c = 1; c < n; ++c) {
                for (int r = 0; r < n; ++r) {
                    std::vector<int> row(n);
                    row[0] = r + 1;  // column 0: just r
                    for (int i = 1; i < n; ++i) {
                        int product = gf2k_mul_(mult[i - 1], c, n, exp, log);
                        row[i] = (r ^ product) + 1;  // GF addition = XOR, then 1-based
                    }
                    table_.push_back(std::move(row));
                }
            }
        } else {
            throw std::runtime_error(
                "FRCheck: cannot auto-generate POA for n=" + std::to_string(n) +
                ". n must be prime or a power of 2. Provide a POA file instead.");
        }

        // Validate
        int expected = n_ * (n_ - 1);
        if ((int)table_.size() != expected)
            throw std::runtime_error("FRCheck generate: expected " +
                                     std::to_string(expected) + " rows, got " +
                                     std::to_string(table_.size()));
    }

    // ---- ISA-L RS encode table init ----
    void init_encode_tables_() {
        int k = n_ - 2;       // data blocks per stripe
        int rows = 2;         // parity blocks
        int m = k + rows;     // total blocks per stripe (= n)
        // RS matrix: m × k (matches ecnaive); top k rows are identity, bottom rows parity
        a_mat_ = (unsigned char*)malloc((size_t)k * (size_t)m);
        if (!a_mat_) throw std::runtime_error("FRCheck: failed to alloc RS matrix");
        gf_gen_rs_matrix(a_mat_, m, k);

        // Encoding tables: 32 * k * rows (matches ecnaive)
        size_t gtbls_size = 32 * (size_t)k * (size_t)rows;
        void* tmp = nullptr;
        if (posix_memalign(&tmp, 32, gtbls_size) != 0) tmp = nullptr;
        if (tmp == nullptr) tmp = malloc(gtbls_size);
        g_tbls_ = (unsigned char*)tmp;
        if (!g_tbls_) {
            free(a_mat_); a_mat_ = nullptr;
            throw std::runtime_error("FRCheck: failed to alloc encode tables");
        }
        // ec_init_tables expects k×rows parity coeffs only; skip identity rows 0..k-1
        ec_init_tables(k, rows, a_mat_ + k * k, g_tbls_);
    }

    // ---- RS decode table init (hardware recovery) ----
    // Builds decode tables for recovering one lost block from k=n-2 surviving blocks.
    // survivor_positions: k positions in [0, n-1] of the surviving blocks
    //   positions 0..k-1 are data blocks (identity); positions k, k+1 are parity
    // lost_position: position in [0, n-1] of the failed rank's block
    void init_decode_tables_(int k,
                             const std::vector<int>& survivor_positions,
                             int lost_position) {
        int m_parity = 2;
        int full_rows = k + m_parity;  // = n

        // Free old decode tables
        if (decode_tbls_) { free(decode_tbls_); decode_tbls_ = nullptr; }

        // Step 1: Build full (k+2) x k Vandermonde encoding matrix
        std::vector<unsigned char> encode_mat((size_t)k * (size_t)full_rows);
        gf_gen_rs_matrix(encode_mat.data(), full_rows, k);

        // Step 2: Build k x k survivor matrix A
        // For each surviving position:
        //   - pos < k: identity row (1 at column pos)
        //   - pos >= k: Vandermonde row from encode_mat
        std::vector<unsigned char> A((size_t)k * (size_t)k, 0);
        int a_row = 0;
        for (int pos : survivor_positions) {
            if (pos < 0 || pos >= full_rows) continue;
            if (pos < k) {
                A[a_row * k + pos] = 1;
            } else {
                // Parity row from Vandermonde
                for (int col = 0; col < k; ++col)
                    A[a_row * k + col] = encode_mat[pos * k + col];
            }
            ++a_row;
        }

        // Step 3: Invert A in GF(2^8)
        std::vector<unsigned char> inv_workspace((size_t)k * 2 * k);
        std::vector<unsigned char> A_inv((size_t)k * k);
        for (int i = 0; i < k * k; ++i) inv_workspace[i] = A[i];
        int ret = gf_invert_matrix(inv_workspace.data(), A_inv.data(), k);
        if (ret != 0) {
            std::cerr << "FRCheck: gf_invert_matrix failed (singular), ret=" << ret << std::endl;
            return;
        }

        // Step 4: Extract decode coefficients from A_inv
        // A_inv is k x k, mapping survivor outputs → original data blocks
        // Row j of A_inv recovers data block j from the k survivor inputs
        std::vector<unsigned char> decode_mat((size_t)k);  // 1 row, k cols
        if (lost_position < k) {
            // Lost is a data block: use row lost_position of A_inv directly
            for (int s = 0; s < k; ++s)
                decode_mat[s] = A_inv[lost_position * k + s];
        } else {
            // Lost is a parity block at position P (k or k+1)
            // Parity P = sum_{j=0}^{k-1} encode_mat[P*k + j] * data_j
            // data_j = sum_{s=0}^{k-1} A_inv[j*k + s] * survivor_s
            // => Parity P = sum_{s} (sum_{j} encode_mat[P*k + j] * A_inv[j*k + s]) * survivor_s
            for (int s = 0; s < k; ++s) {
                unsigned char coeff = 0;
                for (int j = 0; j < k; ++j)
                    coeff ^= gf_mul(encode_mat[lost_position * k + j],
                                    A_inv[j * k + s]);
                decode_mat[s] = coeff;
            }
        }

        // Step 5: Generate decode tables (1 output row, k input columns)
        size_t tbl_size = 32 * (size_t)k * 1;
        void* tmp = nullptr;
        if (posix_memalign(&tmp, 32, tbl_size) != 0) tmp = nullptr;
        if (tmp == nullptr) tmp = malloc(tbl_size);
        decode_tbls_ = (unsigned char*)tmp;
        if (!decode_tbls_) return;
        ec_init_tables(k, 1, decode_mat.data(), decode_tbls_);
    }

    // ---- RS encode thread pool (matches ecnaive xor_pool) ----
    static std::array<int, kRsPoolSize> rs_parse_cpus_or_default() {
        std::array<int, kRsPoolSize> cpus{};
        const char* env = std::getenv(kRsCpuListEnv);
        if (!env || !*env) {
            for (int i = 0; i < kRsPoolSize; ++i) cpus[i] = i;
            return cpus;
        }
        std::vector<int> parsed;
        const char* p = env;
        while (*p) {
            while (*p && (std::isspace((unsigned char)*p) || *p == ',')) ++p;
            if (!*p) break;
            char* end = nullptr;
            long v = std::strtol(p, &end, 10);
            if (end == p || v < 0 || v > 65535)
                throw std::runtime_error(std::string(kRsCpuListEnv) + ": invalid CPU id");
            parsed.push_back((int)v);
            p = end;
        }
        if (parsed.size() != (size_t)kRsPoolSize)
            throw std::runtime_error(std::string(kRsCpuListEnv) +
                                     " must have exactly " + std::to_string(kRsPoolSize) + " CPU ids");
        for (int i = 0; i < kRsPoolSize; ++i) cpus[i] = parsed[i];
        return cpus;
    }

    void rs_pool_init() {
        if (rs_pool_inited_.load(std::memory_order_acquire)) return;
        rs_pool_cpus_ = rs_parse_cpus_or_default();
        rs_pool_stop_.store(false, std::memory_order_release);
        rs_pool_epoch_.store(0, std::memory_order_release);
        rs_pool_remaining_.store(0, std::memory_order_release);
        for (auto& e : rs_pool_last_epoch_) e = 0;
        for (int i = 0; i < kRsPoolSize; ++i) {
            rs_pool_ctx_[i].self = this;
            rs_pool_ctx_[i].wid = i;
            int rc = pthread_create(&rs_pool_threads_[i], nullptr,
                                    &FRCheckNative::rs_pool_entry, &rs_pool_ctx_[i]);
            if (rc != 0) {
                rs_pool_stop_.store(true, std::memory_order_release);
                rs_pool_worker_cv_.notify_all();
                for (int j = 0; j < i; ++j) pthread_join(rs_pool_threads_[j], nullptr);
                throw std::runtime_error("FRCheck: pthread_create for RS pool failed: " +
                                         std::string(std::strerror(rc)));
            }
        }
        rs_pool_inited_.store(true, std::memory_order_release);
        if (debug_)
            std::cout << "FRCheck: RS encode pool (" << kRsPoolSize << " workers) initialized" << std::endl;
    }

    void rs_pool_shutdown() {
        if (!rs_pool_inited_.load(std::memory_order_acquire)) return;
        rs_pool_stop_.store(true, std::memory_order_release);
        rs_pool_worker_cv_.notify_all();
        rs_pool_coordinator_cv_.notify_all();
        for (int i = 0; i < kRsPoolSize; ++i) pthread_join(rs_pool_threads_[i], nullptr);
        rs_pool_inited_.store(false, std::memory_order_release);
        if (debug_)
            std::cout << "FRCheck: RS encode pool shut down" << std::endl;
    }

    static void* rs_pool_entry(void* arg) {
        auto* ctx = (RsPoolWorkerCtx*)arg;
        ctx->self->rs_pool_worker(ctx->wid);
        return nullptr;
    }

    void rs_pool_worker(int wid) {
        int cpu = rs_pool_cpus_[wid];
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (cpu >= 0 && (unsigned)cpu < CPU_SETSIZE) {
            CPU_SET((unsigned)cpu, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        }
        while (true) {
            std::unique_lock<std::mutex> lk(rs_pool_mutex_);
            rs_pool_worker_cv_.wait(lk, [&] {
                return rs_pool_stop_.load(std::memory_order_acquire) ||
                       (rs_pool_last_epoch_[wid] < rs_pool_epoch_.load(std::memory_order_acquire));
            });
            if (rs_pool_stop_.load(std::memory_order_acquire)) break;
            uint64_t e = rs_pool_epoch_.load(std::memory_order_acquire);
            RsEncodeJob job = rs_pool_shared_job_;
            lk.unlock();

            rs_pool_execute_slice(job, wid);

            {
                std::lock_guard<std::mutex> guard(rs_pool_mutex_);
                rs_pool_last_epoch_[wid] = e;
            }
            int left = rs_pool_remaining_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            if (left == 0) rs_pool_coordinator_cv_.notify_one();
        }
    }

    void rs_pool_execute_slice(const RsEncodeJob& job, int wid) {
        int total = job.len;
        int base = total / kRsPoolSize;
        int rem = total % kRsPoolSize;
        int off, len;
        if (wid < kRsPoolSize - 1) {
            off = wid * base;
            len = base;
        } else {
            off = (kRsPoolSize - 1) * base;
            len = base + rem;
        }
        if (len <= 0) return;

        // Build per-worker source and parity pointers offset by 'off'
        std::vector<unsigned char*> src((size_t)job.k);
        for (int i = 0; i < job.k; ++i)
            src[i] = job.data_ptrs[i] + off;
        unsigned char* dest[2] = { job.parity_ptrs[0] + off, job.parity_ptrs[1] + off };

        ec_encode_data(len, job.k, job.m, job.g_tbls, src.data(), dest);
    }

    void rs_pool_run_parallel_encode(const RsEncodeJob& job) {
        {
            std::lock_guard<std::mutex> publish(rs_pool_mutex_);
            if (stopped_) return;
            rs_pool_shared_job_ = job;
            rs_pool_epoch_.fetch_add(1, std::memory_order_acq_rel);
            rs_pool_remaining_.store(kRsPoolSize, std::memory_order_release);
        }
        rs_pool_worker_cv_.notify_all();
        std::unique_lock<std::mutex> lk(rs_pool_mutex_);
        rs_pool_coordinator_cv_.wait(lk, [&] {
            return rs_pool_remaining_.load(std::memory_order_acquire) == 0 ||
                   stopped_.load() || rs_pool_stop_.load(std::memory_order_acquire);
        });
    }

    // ---- EC-aligned role encoding workers (1 thread per role) ----
    struct StripeChunkTask {
        int stripe_id = 0;
        StripeRole role = StripeRole::SOURCE;
        size_t block_size = 0;
        uintptr_t source_data = 0;
        uintptr_t source_mirror = 0;
        uintptr_t recv_buf_addr = 0;
        uintptr_t p1_addr = 0;
        uintptr_t p2_out_addr = 0;
        uintptr_t p2_in_addr = 0;
        int n_src = 0;
        std::vector<int> src_peer_rigs;
        std::vector<uint8_t> source_active_mask;  // 1=active (has data), 0=skip (zero-fill)
        int enc_peer_rig = -1;
        int par_peer_rig = -1;
    };

    struct MirrorTask {
        uintptr_t gpu_addr = 0;
        uintptr_t cpu_addr = 0;
        size_t block_size = 0;
    };

    std::thread mirror_thread_;
    std::mutex mirror_mtx_;
    std::condition_variable mirror_cv_;
    std::queue<MirrorTask> mirror_q_;
    std::atomic<bool> mirror_stop_{false};
    std::atomic<bool> mirror_idle_{true};
    cudaStream_t d2h_stream_ = nullptr;

    // ---- Role-based queues: n workers each, encoder split into RECV→encode+SEND ----
    struct StripeInfo {
        std::vector<int> src_peer_rigs;
        int enc_peer_rig = -1;
        int par_peer_rig = -1;
    };
    std::vector<StripeInfo> stripe_info_;

    struct SourceTask   { int sid; uintptr_t data, mirror; size_t bs; };
    struct EncRecvTask  { int sid; uintptr_t recv, p1, p2; size_t bs; std::vector<uint8_t> mask; };
    struct EncSendTask  { int sid; uintptr_t recv, p1, p2; size_t bs; int n_src; std::vector<int> src_peer_rigs; int par_peer_rig; bool parity_send_only = false; uint64_t async_order = 0; };
    struct ParityTask   { int sid; uintptr_t parity_in; size_t bs; bool async_p2_only = false; uint64_t async_order = 0; };

    std::queue<SourceTask>   source_q_;   std::mutex source_mtx_;   std::condition_variable source_cv_;
    std::queue<EncRecvTask>  enc_recv_q_; std::mutex enc_recv_mtx_; std::condition_variable enc_recv_cv_;
    std::queue<EncSendTask>  enc_send_q_; std::mutex enc_send_mtx_; std::condition_variable enc_send_cv_;
    std::queue<ParityTask>   parity_q_;   std::mutex parity_mtx_;   std::condition_variable parity_cv_;

    std::vector<std::thread> source_workers_, enc_recv_workers_, enc_send_workers_, parity_workers_;
    std::atomic<bool> all_stop_{false};
    std::atomic<int> task_total_{0}, task_done_{0};
    std::atomic<int> task_encode_total_{0}, task_encode_done_{0};
    std::atomic<int> task_async_total_{0}, task_async_done_{0};
    std::mutex encode_done_mtx_;
    std::condition_variable encode_done_cv_;
    std::atomic<int> async_p2p_pause_count_{0};
    std::mutex async_parity_send_order_mtx_;
    std::mutex async_p2_order_mtx_;
    std::condition_variable async_p2_order_cv_;
    std::vector<uint64_t> async_p2_send_enq_order_;
    std::vector<uint64_t> async_p2_send_run_order_;
    std::vector<uint64_t> async_p2_recv_enq_order_;
    std::vector<uint64_t> async_p2_recv_run_order_;
    std::mutex layer_done_mtx_;
    std::condition_variable layer_done_cv_;
    std::mutex async_done_mtx_;
    std::condition_variable async_done_cv_;
    std::mutex encoder_encode_mtx_;
    std::atomic<int> async_p2p_inflight_{0};
    std::mutex async_p2p_inflight_mtx_;
    std::condition_variable async_p2p_inflight_cv_;
    bool debug_ = false;

    void check_done_() {
        if (task_done_.fetch_add(1, std::memory_order_acq_rel) + 1 == task_total_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lk(layer_done_mtx_);
            layer_done_cv_.notify_all();
        }
    }

    void check_encode_done_() {
        if (task_encode_done_.fetch_add(1, std::memory_order_acq_rel) + 1
            == task_encode_total_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lk(encode_done_mtx_);
            encode_done_cv_.notify_all();
        }
    }

    void check_async_done_() {
        int done = task_async_done_.fetch_add(1, std::memory_order_acq_rel) + 1;
        int total = task_async_total_.load(std::memory_order_acquire);
        if (done >= total) {
            std::lock_guard<std::mutex> lk(async_done_mtx_);
            async_done_cv_.notify_all();
        }
    }

    void _wait_if_paused() {
        while (async_p2p_pause_count_.load(std::memory_order_acquire) > 0) {
            if (debug_) {
                static int log_counter = 0;
                if (++log_counter % 1000 == 1)
                    std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                              << " async P2 paused (refcount="
                              << async_p2p_pause_count_.load() << ")" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    void _async_rdma_begin() {
        while (true) {
            async_p2p_inflight_.fetch_add(1, std::memory_order_acq_rel);
            if (async_p2p_pause_count_.load(std::memory_order_acquire) == 0) {
                return;
            }
            _async_rdma_end();
            _wait_if_paused();
        }
    }

    void _async_rdma_end() {
        int prev = async_p2p_inflight_.fetch_sub(1, std::memory_order_acq_rel);
        if (prev <= 1) {
            std::lock_guard<std::mutex> lk(async_p2p_inflight_mtx_);
            async_p2p_inflight_cv_.notify_all();
        }
    }

    void _wait_async_rdma_idle() {
        if (async_p2p_inflight_.load(std::memory_order_acquire) == 0) return;
        std::unique_lock<std::mutex> lk(async_p2p_inflight_mtx_);
        async_p2p_inflight_cv_.wait(lk, [&] {
            return async_p2p_inflight_.load(std::memory_order_acquire) == 0;
        });
    }

    void wait_async_p2_send_turn_(int sid, uint64_t order) {
        std::unique_lock<std::mutex> lk(async_p2_order_mtx_);
        while (!all_stop_.load(std::memory_order_acquire) &&
               sid >= 0 && sid < (int)async_p2_send_run_order_.size() &&
               async_p2_send_run_order_[(size_t)sid] != order) {
            if (debug_) {
                std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                          << " async_p2_send wait_turn sid=" << sid
                          << " order=" << order
                          << " expect=" << async_p2_send_run_order_[(size_t)sid]
                          << std::endl;
            }
            async_p2_order_cv_.wait_for(lk, std::chrono::seconds(1));
        }
        if (debug_) {
            uint64_t expect = (sid >= 0 && sid < (int)async_p2_send_run_order_.size())
                                ? async_p2_send_run_order_[(size_t)sid] : 0;
            std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                      << " async_p2_send turn_ready sid=" << sid
                      << " order=" << order
                      << " expect=" << expect << std::endl;
        }
    }

    void advance_async_p2_send_turn_(int sid) {
        {
            std::lock_guard<std::mutex> lk(async_p2_order_mtx_);
            if (sid >= 0 && sid < (int)async_p2_send_run_order_.size())
                ++async_p2_send_run_order_[(size_t)sid];
        }
        async_p2_order_cv_.notify_all();
    }

    void wait_async_p2_recv_turn_(int sid, uint64_t order) {
        std::unique_lock<std::mutex> lk(async_p2_order_mtx_);
        while (!all_stop_.load(std::memory_order_acquire) &&
               sid >= 0 && sid < (int)async_p2_recv_run_order_.size() &&
               async_p2_recv_run_order_[(size_t)sid] != order) {
            if (debug_) {
                std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                          << " async_p2_recv wait_turn sid=" << sid
                          << " order=" << order
                          << " expect=" << async_p2_recv_run_order_[(size_t)sid]
                          << std::endl;
            }
            async_p2_order_cv_.wait_for(lk, std::chrono::seconds(1));
        }
        if (debug_) {
            uint64_t expect = (sid >= 0 && sid < (int)async_p2_recv_run_order_.size())
                                ? async_p2_recv_run_order_[(size_t)sid] : 0;
            std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                      << " async_p2_recv turn_ready sid=" << sid
                      << " order=" << order
                      << " expect=" << expect << std::endl;
        }
    }

    void advance_async_p2_recv_turn_(int sid) {
        {
            std::lock_guard<std::mutex> lk(async_p2_order_mtx_);
            if (sid >= 0 && sid < (int)async_p2_recv_run_order_.size())
                ++async_p2_recv_run_order_[(size_t)sid];
        }
        async_p2_order_cv_.notify_all();
    }

    void source_worker_() {
        while (!all_stop_) {
            SourceTask t;
            { std::unique_lock<std::mutex> lk(source_mtx_); source_cv_.wait(lk, [&]{ return all_stop_ || !source_q_.empty(); });
              if (all_stop_ && source_q_.empty()) break; t = source_q_.front(); source_q_.pop(); }
            auto& si = stripe_info_[(size_t)t.sid];
            auto* ch = get_channel_(si.enc_peer_rig, t.sid);
            if (!ch) { std::cerr << "FRCheck source " << t.sid << ": no channel\n"; continue; }
            ch->send_data((const uint8_t*)t.data, t.bs);
            if (t.mirror) push_mirror_task_(t.data, t.mirror, t.bs);
            check_encode_done_();
            check_done_();
        }
    }

    void enc_recv_worker_() {
        while (!all_stop_) {
            EncRecvTask t;
            { std::unique_lock<std::mutex> lk(enc_recv_mtx_); enc_recv_cv_.wait(lk, [&]{ return all_stop_ || !enc_recv_q_.empty(); });
              if (all_stop_ && enc_recv_q_.empty()) break; t = std::move(enc_recv_q_.front()); enc_recv_q_.pop(); }
            auto& si = stripe_info_[(size_t)t.sid];
            int n_src = (int)si.src_peer_rigs.size();
            std::vector<std::thread> recv_threads;
            for (int i = 0; i < n_src; ++i) {
                if (!t.mask.empty() && i < (int)t.mask.size() && t.mask[(size_t)i] == 0) continue;
                recv_threads.emplace_back([&, i]() {
                    if (si.src_peer_rigs[(size_t)i] == rank_in_group_) return;
                    uintptr_t dst = t.recv + (uintptr_t)i * t.bs;
                    auto* ch = get_channel_(si.src_peer_rigs[(size_t)i], t.sid);
                    if (ch) ch->recv_data((uint8_t*)dst, t.bs);
                });
            }
            for (auto& th : recv_threads) th.join();
            // Push to enc_send_q_
            EncSendTask es{t.sid, t.recv, t.p1, t.p2, t.bs, n_src, si.src_peer_rigs, si.par_peer_rig};
            { std::lock_guard<std::mutex> lk(enc_send_mtx_); enc_send_q_.push(std::move(es)); }
            enc_send_cv_.notify_one();
        }
    }

    void enc_send_worker_() {
        while (!all_stop_) {
            EncSendTask t;
            {
                std::unique_lock<std::mutex> lk(enc_send_mtx_);
                enc_send_cv_.wait(lk, [&]{ return all_stop_ || !enc_send_q_.empty(); });
                if (all_stop_ && enc_send_q_.empty()) break;
                t = std::move(enc_send_q_.front()); enc_send_q_.pop();
            }
            if (t.parity_send_only) {
                wait_async_p2_send_turn_(t.sid, t.async_order);
                _wait_if_paused();
                auto wait_cb = [this]() { this->_async_rdma_begin(); };
                auto done_cb = [this]() { this->_async_rdma_end(); };
                {
                    auto* ch = get_channel_(t.par_peer_rig, t.sid);
                    if (debug_) {
                        std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                                  << " async_p2_send begin sid=" << t.sid
                                  << " order=" << t.async_order
                                  << " peer=" << t.par_peer_rig
                                  << " bytes=" << t.bs
                                  << " done=" << task_async_done_.load(std::memory_order_acquire)
                                  << "/" << task_async_total_.load(std::memory_order_acquire)
                                  << " ch=" << (ch ? 1 : 0) << std::endl;
                    }
                    if (ch) ch->send_data((const uint8_t*)t.p1, t.bs, wait_cb, done_cb);
                }
                advance_async_p2_send_turn_(t.sid);
                check_async_done_();
                if (debug_) {
                    std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                              << " async_p2_send done sid=" << t.sid
                              << " order=" << t.async_order
                              << " peer=" << t.par_peer_rig
                              << " bytes=" << t.bs
                              << " done=" << task_async_done_.load(std::memory_order_acquire)
                              << "/" << task_async_total_.load(std::memory_order_acquire)
                              << std::endl;
                }
            } else {
                // ---- sync phase: RS encode ----
                int n_src = t.n_src;
                std::vector<unsigned char*> data_ptrs((size_t)n_src);
                for (int i = 0; i < n_src; ++i)
                    data_ptrs[(size_t)i] = (unsigned char*)(t.recv + (uintptr_t)i * t.bs);
                unsigned char* parity_ptrs[2] = { (unsigned char*)t.p1, (unsigned char*)t.p2 };
                RsEncodeJob rs{(int)t.bs, n_src, 2, g_tbls_, data_ptrs.data(), parity_ptrs};
                { std::lock_guard<std::mutex> lk(encoder_encode_mtx_); rs_pool_run_parallel_encode(rs); }
                check_encode_done_();
                check_done_();
            }
        }
    }

    void parity_worker_() {
        while (!all_stop_) {
            ParityTask t;
            { std::unique_lock<std::mutex> lk(parity_mtx_); parity_cv_.wait(lk, [&]{ return all_stop_ || !parity_q_.empty(); });
              if (all_stop_ && parity_q_.empty()) break; t = parity_q_.front(); parity_q_.pop(); }
            auto& si = stripe_info_[(size_t)t.sid];
            auto* ch = get_channel_(si.enc_peer_rig, t.sid);
            if (t.async_p2_only) wait_async_p2_recv_turn_(t.sid, t.async_order);
            if (debug_ && t.async_p2_only) {
                std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                          << " async_p2_recv begin sid=" << t.sid
                          << " order=" << t.async_order
                          << " peer=" << si.enc_peer_rig
                          << " bytes=" << t.bs
                          << " done=" << task_async_done_.load(std::memory_order_acquire)
                          << "/" << task_async_total_.load(std::memory_order_acquire)
                          << " ch=" << (ch ? 1 : 0) << std::endl;
            }
            // Do not pause receive-side posts: a sender that already entered RDMA
            // needs the matching recv to make progress and drain before PP starts.
            if (ch) ch->recv_data((uint8_t*)t.parity_in, t.bs);
            if (t.async_p2_only) advance_async_p2_recv_turn_(t.sid);
            check_async_done_();
            if (debug_ && t.async_p2_only) {
                std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                          << " async_p2_recv done sid=" << t.sid
                          << " order=" << t.async_order
                          << " peer=" << si.enc_peer_rig
                          << " bytes=" << t.bs
                          << " done=" << task_async_done_.load(std::memory_order_acquire)
                          << "/" << task_async_total_.load(std::memory_order_acquire)
                          << std::endl;
            }
            check_done_();
        }
    }

    void push_mirror_task_(uintptr_t gpu_addr, uintptr_t cpu_addr, size_t block_size) {
        MirrorTask mt{gpu_addr, cpu_addr, block_size};
        {
            std::lock_guard<std::mutex> lk(mirror_mtx_);
            mirror_q_.push(mt);
            mirror_idle_ = false;
        }
        mirror_cv_.notify_one();
    }

    void mirror_worker_() {
        while (!mirror_stop_.load()) {
            MirrorTask task;
            {
                std::unique_lock<std::mutex> lk(mirror_mtx_);
                mirror_cv_.wait(lk, [this] {
                    return mirror_stop_.load() || !mirror_q_.empty();
                });
                if (mirror_stop_.load() && mirror_q_.empty()) break;
                task = mirror_q_.front();
                mirror_q_.pop();
                mirror_idle_ = false;
            }
            cudaError_t err = cudaMemcpyAsync(
                reinterpret_cast<void*>(task.cpu_addr),
                reinterpret_cast<const void*>(task.gpu_addr),
                task.block_size,
                cudaMemcpyDeviceToHost,
                d2h_stream_);
            if (err != cudaSuccess) {
                std::cerr << "FRCheck mirror_worker: async D2H failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
            {
                std::lock_guard<std::mutex> lk(mirror_mtx_);
                if (mirror_q_.empty()) mirror_idle_ = true;
            }
            mirror_cv_.notify_all();
        }
    }

    void mirror_worker_init() {
        mirror_stop_ = false;
        mirror_idle_ = true;
        if (d2h_stream_ == nullptr) {
            cudaError_t err = cudaStreamCreate(&d2h_stream_);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("FRCheck: cudaStreamCreate(d2h_stream) failed: ") +
                    cudaGetErrorString(err));
            }
        }
        mirror_thread_ = std::thread([this]() { mirror_worker_(); });
    }

    void mirror_worker_shutdown() {
        mirror_stop_ = true;
        mirror_cv_.notify_all();
        if (mirror_thread_.joinable()) mirror_thread_.join();
        if (d2h_stream_ != nullptr) {
            cudaStreamSynchronize(d2h_stream_);
            cudaStreamDestroy(d2h_stream_);
            d2h_stream_ = nullptr;
        }
        {
            std::lock_guard<std::mutex> lk(mirror_mtx_);
            while (!mirror_q_.empty()) mirror_q_.pop();
            mirror_idle_ = true;
        }
    }

public:
    void init_stripe_workers() {
        int ns = (int)stripe_plans_.size();
        stripe_info_.resize((size_t)ns);
        async_p2_send_enq_order_.assign((size_t)ns, 0);
        async_p2_send_run_order_.assign((size_t)ns, 0);
        async_p2_recv_enq_order_.assign((size_t)ns, 0);
        async_p2_recv_run_order_.assign((size_t)ns, 0);
        for (int i = 0; i < ns; ++i) {
            auto& si = stripe_info_[(size_t)i];
            si.src_peer_rigs.resize(stripe_plans_[i].source_node_ids.size());
            for (size_t j = 0; j < stripe_plans_[i].source_node_ids.size(); ++j)
                si.src_peer_rigs[j] = stripe_plans_[i].source_node_ids[j] - 1;
            si.enc_peer_rig = stripe_plans_[i].encoder_node_id - 1;
            si.par_peer_rig = stripe_plans_[i].parity_target_node_id - 1;
        }
        all_stop_.store(false, std::memory_order_release);
        int nw = n_;
        for (int w = 0; w < nw; ++w) {
            source_workers_.emplace_back(&FRCheckNative::source_worker_, this);
            enc_recv_workers_.emplace_back(&FRCheckNative::enc_recv_worker_, this);
            enc_send_workers_.emplace_back(&FRCheckNative::enc_send_worker_, this);
            parity_workers_.emplace_back(&FRCheckNative::parity_worker_, this);
        }
        // Reuse enc_send workers for deferred P2 sends (n parallel workers,
        // same as sync path).  P2 send tasks are pushed to enc_send_q_ so
        // the existing n workers pick them up in FIFO order.
        if (debug_)
            std::cout << "FRCheck: " << nw << " workers/role for " << ns << " stripes" << std::endl;
    }

    void shutdown_stripe_workers() {
        all_stop_.store(true, std::memory_order_release);
        source_cv_.notify_all(); enc_recv_cv_.notify_all();
        enc_send_cv_.notify_all(); parity_cv_.notify_all();
        async_p2_order_cv_.notify_all();
        for (auto* v : {&source_workers_, &enc_recv_workers_, &enc_send_workers_, &parity_workers_})
            for (auto& t : *v) if (t.joinable()) t.join();
        source_workers_.clear(); enc_recv_workers_.clear();
        enc_send_workers_.clear(); parity_workers_.clear();
    }

    void submit_source(int sid, uintptr_t data, uintptr_t mirror, size_t bs) {
        task_total_.fetch_add(1, std::memory_order_acq_rel);
        task_encode_total_.fetch_add(1, std::memory_order_acq_rel);
        { std::lock_guard<std::mutex> lk(source_mtx_); source_q_.push({sid, data, mirror, bs}); }
        source_cv_.notify_one();
    }

    void submit_enc_recv(int sid, uintptr_t recv, uintptr_t p1, uintptr_t p2, size_t bs,
                         const std::vector<uint8_t>& mask) {
        task_total_.fetch_add(1, std::memory_order_acq_rel);
        task_encode_total_.fetch_add(1, std::memory_order_acq_rel);
        { std::lock_guard<std::mutex> lk(enc_recv_mtx_); enc_recv_q_.push({sid, recv, p1, p2, bs, mask}); }
        enc_recv_cv_.notify_one();
    }

    void submit_parity(int sid, uintptr_t parity_in, size_t bs) {
        task_total_.fetch_add(1, std::memory_order_acq_rel);
        task_async_total_.fetch_add(1, std::memory_order_acq_rel);
        { std::lock_guard<std::mutex> lk(parity_mtx_); parity_q_.push({sid, parity_in, bs}); }
        parity_cv_.notify_one();
    }

    void submit_p2_send(int sid, uintptr_t p2_addr, size_t bs) {
        submit_p1_send(sid, p2_addr, bs);
    }

    void submit_p1_send(int sid, uintptr_t p1_addr, size_t bs) {
        auto& si = stripe_info_[(size_t)sid];
        EncSendTask t;
        t.sid = sid;
        t.p1 = p1_addr;
        t.bs = bs;
        t.par_peer_rig = si.par_peer_rig;
        t.parity_send_only = true;
        task_async_total_.fetch_add(1, std::memory_order_acq_rel);
        { std::lock_guard<std::mutex> lk(enc_send_mtx_); enc_send_q_.push(std::move(t)); }
        enc_send_cv_.notify_one();
    }

    void submit_async_p2_batch(
        const std::vector<std::tuple<int, uintptr_t, size_t>>& parity_tasks,
        const std::vector<std::tuple<int, uintptr_t, size_t>>& p2_send_tasks) {
        int total = static_cast<int>(parity_tasks.size() + p2_send_tasks.size());
        if (total == 0) return;
        task_total_.fetch_add(static_cast<int>(parity_tasks.size()), std::memory_order_acq_rel);
        task_async_total_.fetch_add(total, std::memory_order_acq_rel);
        if (!parity_tasks.empty()) {
            std::lock_guard<std::mutex> lk(parity_mtx_);
            for (const auto& task : parity_tasks) {
                int sid;
                uintptr_t parity_in;
                size_t bs;
                std::tie(sid, parity_in, bs) = task;
                parity_q_.push({sid, parity_in, bs});
            }
        }
        if (!p2_send_tasks.empty()) {
            std::lock_guard<std::mutex> lk(enc_send_mtx_);
            for (const auto& task : p2_send_tasks) {
                int sid;
                uintptr_t p2_addr;
                size_t bs;
                std::tie(sid, p2_addr, bs) = task;
                auto& si = stripe_info_[(size_t)sid];
                EncSendTask t;
                t.sid = sid;
                t.p1 = p2_addr;
                t.bs = bs;
                t.par_peer_rig = si.par_peer_rig;
                t.parity_send_only = true;
                enc_send_q_.push(std::move(t));
            }
        }
        if (!parity_tasks.empty()) parity_cv_.notify_all();
        if (!p2_send_tasks.empty()) enc_send_cv_.notify_all();
    }

    std::vector<int> submit_async_p2_layer(
        const std::vector<uintptr_t>& p2_addrs,
        size_t bs) {
        int parity_count = 0;
        int send_count = 0;
        int ns = std::min((int)p2_addrs.size(), (int)stripe_info_.size());
        for (int sid = 0; sid < ns; ++sid) {
            if (p2_addrs[(size_t)sid] == 0) continue;
            int role = get_role_for_stripe(sid);
            if (role == (int)StripeRole::PARITY_TARGET) {
                ++parity_count;
            } else if (role == (int)StripeRole::ENCODER) {
                ++send_count;
            }
        }
        int total = parity_count + send_count;
        if (total == 0) return {0, 0};

        task_total_.fetch_add(parity_count, std::memory_order_acq_rel);
        task_async_total_.fetch_add(total, std::memory_order_acq_rel);

        std::vector<ParityTask> parity_to_push;
        if (parity_count > 0) {
            parity_to_push.reserve((size_t)parity_count);
            for (int sid = 0; sid < ns; ++sid) {
                uintptr_t addr = p2_addrs[(size_t)sid];
                if (addr == 0) continue;
                if (get_role_for_stripe(sid) == (int)StripeRole::PARITY_TARGET) {
                    ParityTask t;
                    t.sid = sid;
                    t.parity_in = addr;
                    t.bs = bs;
                    t.async_p2_only = true;
                    {
                        std::lock_guard<std::mutex> order_lk(async_p2_order_mtx_);
                        if (sid >= 0 && sid < (int)async_p2_recv_enq_order_.size())
                            t.async_order = async_p2_recv_enq_order_[(size_t)sid]++;
                    }
                    parity_to_push.push_back(std::move(t));
                }
            }
            std::lock_guard<std::mutex> lk(parity_mtx_);
            for (auto& t : parity_to_push) parity_q_.push(std::move(t));
        }

        std::vector<EncSendTask> send_to_push;
        if (send_count > 0) {
            send_to_push.reserve((size_t)send_count);
            for (int sid = 0; sid < ns; ++sid) {
                uintptr_t addr = p2_addrs[(size_t)sid];
                if (addr == 0) continue;
                if (get_role_for_stripe(sid) == (int)StripeRole::ENCODER) {
                    auto& si = stripe_info_[(size_t)sid];
                    EncSendTask t;
                    t.sid = sid;
                    t.p1 = addr;
                    t.bs = bs;
                    t.par_peer_rig = si.par_peer_rig;
                    t.parity_send_only = true;
                    {
                        std::lock_guard<std::mutex> order_lk(async_p2_order_mtx_);
                        if (sid >= 0 && sid < (int)async_p2_send_enq_order_.size())
                            t.async_order = async_p2_send_enq_order_[(size_t)sid]++;
                    }
                    send_to_push.push_back(std::move(t));
                }
            }
            std::lock_guard<std::mutex> lk(enc_send_mtx_);
            for (auto& t : send_to_push) enc_send_q_.push(std::move(t));
        }

        if (parity_count > 0) parity_cv_.notify_all();
        if (send_count > 0) enc_send_cv_.notify_all();
        return {parity_count, send_count};
    }

    void reset_async_parity() {
        task_async_done_.store(0, std::memory_order_release);
        task_async_total_.store(0, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(parity_mtx_);
            while (!parity_q_.empty()) parity_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lk(enc_send_mtx_);
            std::queue<EncSendTask> keep;
            while (!enc_send_q_.empty()) {
                EncSendTask t = std::move(enc_send_q_.front());
                enc_send_q_.pop();
                if (!t.parity_send_only) keep.push(std::move(t));
            }
            enc_send_q_.swap(keep);
        }
        {
            std::lock_guard<std::mutex> lk(async_p2_order_mtx_);
            std::fill(async_p2_send_enq_order_.begin(), async_p2_send_enq_order_.end(), 0);
            std::fill(async_p2_send_run_order_.begin(), async_p2_send_run_order_.end(), 0);
            std::fill(async_p2_recv_enq_order_.begin(), async_p2_recv_enq_order_.end(), 0);
            std::fill(async_p2_recv_run_order_.begin(), async_p2_recv_run_order_.end(), 0);
        }
        async_p2_order_cv_.notify_all();
    }

    void reset_layer() {
        task_done_.store(0, std::memory_order_release);
        task_total_.store(0, std::memory_order_release);
        task_encode_done_.store(0, std::memory_order_release);
        task_encode_total_.store(0, std::memory_order_release);
        task_async_done_.store(0, std::memory_order_release);
        task_async_total_.store(0, std::memory_order_release);
        { std::lock_guard<std::mutex> lk(source_mtx_);   while (!source_q_.empty()) source_q_.pop(); }
        { std::lock_guard<std::mutex> lk(enc_recv_mtx_); while (!enc_recv_q_.empty()) enc_recv_q_.pop(); }
        { std::lock_guard<std::mutex> lk(enc_send_mtx_); while (!enc_send_q_.empty()) enc_send_q_.pop(); }
        { std::lock_guard<std::mutex> lk(parity_mtx_);   while (!parity_q_.empty()) parity_q_.pop(); }
    }

    void wait_layer() {
        int total = task_total_.load(std::memory_order_acquire);
        if (total == 0) return;
        std::unique_lock<std::mutex> lk(layer_done_mtx_);
        layer_done_cv_.wait_for(lk, std::chrono::seconds(5), [&]{ return task_done_.load() >= total; });
        if (task_done_.load() < total)
            std::cerr << "[FRCHECK-DIAG] rank " << rank_in_group_ << " stuck: done="
                      << task_done_.load() << "/" << total << std::endl;
        layer_done_cv_.wait(lk, [&]{ return task_done_.load() >= total; });
    }

    void wait_encode_only() {
        int total = task_encode_total_.load(std::memory_order_acquire);
        if (total == 0) return;
        if (debug_)
            std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                      << " wait_encode_only begin (total=" << total
                      << " done=" << task_encode_done_.load() << ")" << std::endl;
        std::unique_lock<std::mutex> lk(encode_done_mtx_);
        encode_done_cv_.wait(lk, [&]{
            return task_encode_done_.load(std::memory_order_acquire) >= total;
        });
        if (debug_)
            std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                      << " wait_encode_only done" << std::endl;
    }

    void wait_parity_flush() {
        int total = task_async_total_.load(std::memory_order_acquire);
        if (total > 0 && debug_)
            std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                      << " wait_parity_flush begin (total=" << total
                      << " done=" << task_async_done_.load() << ")"
                      << std::endl;
        if (total == 0) return;
        std::unique_lock<std::mutex> lk(async_done_mtx_);
        async_done_cv_.wait(lk, [&]{
            return task_async_done_.load(std::memory_order_acquire) >=
                   task_async_total_.load(std::memory_order_acquire);
        });
        if (total > 0 && debug_)
            std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                      << " wait_parity_flush done" << std::endl;
    }

    void inc_pause_async_p2p() {
        // Block new async P2 RDMA before PP communication starts, then wait for
        // already-started async RDMA chunks to drain.
        int prev = async_p2p_pause_count_.fetch_add(1, std::memory_order_acq_rel);
        _wait_async_rdma_idle();
        if (debug_)
            std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                      << " inc_pause_async_p2p prev=" << prev
                      << " -> async RDMA idle" << std::endl;
    }

    void dec_pause_async_p2p() {
        int prev = async_p2p_pause_count_.fetch_sub(1, std::memory_order_acq_rel);
        if (prev <= 1) {
            async_p2p_pause_count_.store(0, std::memory_order_release);
        }
        if (debug_)
            std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                      << " dec_pause_async_p2p prev=" << prev << std::endl;
    }

    void wait_mirror_completion() {
        std::unique_lock<std::mutex> lk(mirror_mtx_);
        mirror_cv_.wait(lk, [this] {
            return (mirror_idle_.load() && mirror_q_.empty()) ||
                   stopped_.load() || mirror_stop_.load();
        });
        if (d2h_stream_ != nullptr) {
            cudaError_t err = cudaStreamSynchronize(d2h_stream_);
            if (err != cudaSuccess) {
                std::cerr << "FRCheck wait_mirror: d2h_stream sync failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    // ---- Recovery worker tasks ----
    struct RecoveryHelperTask {
        int stripe_id = 0;
        size_t block_size = 0;
        uintptr_t helper_block = 0;
        int decoder_rig = -1;
    };

    struct RecoveryDecoderTask {
        int stripe_id = 0;
        size_t block_size = 0;
        uintptr_t self_block = 0;
        std::vector<uintptr_t> helper_recv_bufs;
        std::vector<int> helper_rigs;
        std::vector<uintptr_t> recovered_bufs;
        bool dual_failure = false;
        std::vector<int> survivor_positions;
        int failed_pos = 0;
        std::vector<int> failed_positions;
        std::vector<int> failed_rigs;
    };

    struct RecoveryFailedTask {
        int stripe_id = 0;
        size_t block_size = 0;
        uintptr_t recv_buf = 0;
        int decoder_rig = -1;
        uintptr_t layer_buf = 0;
        size_t layer_offset = 0;
        size_t ncopy = 0;
        bool store_to_layer = false;
    };

    static bool is_recovery_helper_sentinel_(const RecoveryHelperTask& t) {
        return t.block_size == 0 && t.helper_block == 0;
    }
    static bool is_recovery_decoder_sentinel_(const RecoveryDecoderTask& t) {
        return t.block_size == 0 && t.self_block == 0;
    }
    static bool is_recovery_failed_sentinel_(const RecoveryFailedTask& t) {
        return t.block_size == 0 && t.recv_buf == 0;
    }

    static StripeRole role_for_node_in_row_(const std::vector<int>& row, int node_id, int n) {
        auto it = std::find(row.begin(), row.end(), node_id);
        if (it == row.end())
            throw std::runtime_error("FRCheck recovery: node not in POA row");
        int pos = (int)std::distance(row.begin(), it);
        if (pos < n - 2) return StripeRole::SOURCE;
        if (pos == n - 2) return StripeRole::ENCODER;
        return StripeRole::PARITY_TARGET;
    }

    void compile_recovery_plans_single_(int failed_rank_node) {
        recovery_plans_.resize(stripe_plans_.size());
        for (const auto& sp : stripe_plans_) {
            const auto& row = sp.row;
            int sid = sp.stripe_id;
            auto it = std::find(row.begin(), row.end(), failed_rank_node);
            if (it == row.end()) continue;

            int failed_pos = (int)std::distance(row.begin(), it);
            int decoder_pos = (failed_pos + 1) % n_;
            std::vector<int> helper_positions;
            for (int i = 0; i < n_ - 3; ++i)
                helper_positions.push_back((failed_pos + 2 + i) % n_);

            RecoveryStripePlan plan;
            plan.stripe_id = sid;
            plan.dual_failure = false;
            plan.failed_node = failed_rank_node;
            plan.failed_pos = failed_pos;
            plan.decoder_node = row[decoder_pos];
            plan.decoder_pos = decoder_pos;
            for (int p : helper_positions) {
                plan.helper_positions.push_back(p);
                plan.helper_nodes.push_back(row[p]);
            }
            plan.survivor_positions = {decoder_pos};
            plan.survivor_positions.insert(
                plan.survivor_positions.end(),
                helper_positions.begin(), helper_positions.end());
            plan.original_role = (int)sp.role;
            recovery_plans_[sid] = std::move(plan);
        }
    }

    void compile_recovery_plans_dual_(const std::vector<int>& failed_nodes) {
        std::set<int> failed_set(failed_nodes.begin(), failed_nodes.end());
        recovery_plans_.resize(stripe_plans_.size());
        for (const auto& sp : stripe_plans_) {
            const auto& row = sp.row;
            int sid = sp.stripe_id;

            RecoveryStripePlan plan;
            plan.stripe_id = sid;
            plan.dual_failure = true;
            plan.failed_nodes = failed_nodes;

            for (int fn : failed_nodes) {
                auto it = std::find(row.begin(), row.end(), fn);
                if (it == row.end()) {
                    throw std::runtime_error(
                        "FRCheck dual recovery: failed node " + std::to_string(fn) +
                        " missing in stripe " + std::to_string(sid));
                }
                int fp = (int)std::distance(row.begin(), it);
                RecoveryFailedTarget ft;
                ft.failed_node = fn;
                ft.failed_pos = fp;
                ft.original_role = (int)role_for_node_in_row_(row, fn, n_);
                plan.failed_targets.push_back(ft);
            }

            for (int i = 0; i < n_; ++i) {
                if (failed_set.count(row[i]) == 0)
                    plan.survivor_positions.push_back(i);
            }
            if ((int)plan.survivor_positions.size() != n_ - 2) {
                throw std::runtime_error(
                    "FRCheck dual recovery: stripe " + std::to_string(sid) +
                    " survivor count mismatch");
            }
            int decoder_pos = plan.survivor_positions[0];
            plan.decoder_pos = decoder_pos;
            plan.decoder_node = row[decoder_pos];
            for (size_t i = 1; i < plan.survivor_positions.size(); ++i) {
                int p = plan.survivor_positions[i];
                plan.helper_positions.push_back(p);
                plan.helper_nodes.push_back(row[p]);
            }
            plan.original_role = (int)sp.role;
            recovery_plans_[sid] = std::move(plan);
        }
    }

    void execute_recovery_helper_(const RecoveryHelperTask& task) {
        uint64_t t0 = frcheck_now_us();
        send_to_peer(task.decoder_rig, task.stripe_id,
                     task.helper_block, task.block_size);
        recovery_helper_send_us_.fetch_add(frcheck_now_us() - t0, std::memory_order_relaxed);
        recovery_helper_tasks_.fetch_add(1, std::memory_order_relaxed);
    }

    void execute_recovery_decoder_(const RecoveryDecoderTask& task) {
        uint64_t t_recv = frcheck_now_us();
        std::vector<std::thread> recv_threads;
        std::vector<std::exception_ptr> recv_errors(task.helper_rigs.size());

        for (size_t hi = 0; hi < task.helper_rigs.size(); ++hi) {
            recv_errors[hi] = nullptr;
            recv_threads.emplace_back([&, hi]() {
                try {
                    if (hi >= task.helper_recv_bufs.size())
                        throw std::runtime_error("FRCheck recovery: missing helper recv buf");
                    recv_from_peer(
                        task.helper_rigs[hi], task.stripe_id,
                        task.helper_recv_bufs[hi], task.block_size);
                } catch (...) {
                    recv_errors[hi] = std::current_exception();
                }
            });
        }
        for (auto& t : recv_threads) t.join();
        for (size_t hi = 0; hi < recv_errors.size(); ++hi) {
            if (recv_errors[hi]) std::rethrow_exception(recv_errors[hi]);
        }
        recovery_decoder_recv_us_.fetch_add(frcheck_now_us() - t_recv, std::memory_order_relaxed);

        int k = n_ - 2;
        std::vector<uintptr_t> survivor_addrs;
        survivor_addrs.push_back(task.self_block);
        for (uintptr_t addr : task.helper_recv_bufs)
            survivor_addrs.push_back(addr);

        if (!task.dual_failure) {
            if (task.recovered_bufs.empty())
                throw std::runtime_error("FRCheck recovery: missing recovered buffer");
            uint64_t t_decode = frcheck_now_us();
            submit_stripe_decode(
                k, task.survivor_positions, task.failed_pos,
                survivor_addrs, task.recovered_bufs[0], task.block_size);
            recovery_decoder_decode_us_.fetch_add(frcheck_now_us() - t_decode, std::memory_order_relaxed);
            if (task.failed_rigs.empty())
                throw std::runtime_error("FRCheck recovery: missing failed rig");
            uint64_t t_send = frcheck_now_us();
            send_to_peer(task.failed_rigs[0], task.stripe_id,
                         task.recovered_bufs[0], task.block_size);
            recovery_decoder_send_us_.fetch_add(frcheck_now_us() - t_send, std::memory_order_relaxed);
        } else {
            if (task.recovered_bufs.size() < task.failed_positions.size())
                throw std::runtime_error("FRCheck recovery: insufficient recovered buffers");
            std::vector<std::thread> send_threads;
            for (size_t slot = 0; slot < task.failed_positions.size(); ++slot) {
                uint64_t t_decode = frcheck_now_us();
                submit_stripe_decode(
                    k, task.survivor_positions, task.failed_positions[slot],
                    survivor_addrs, task.recovered_bufs[slot], task.block_size);
                recovery_decoder_decode_us_.fetch_add(frcheck_now_us() - t_decode, std::memory_order_relaxed);
                int frig = task.failed_rigs[slot];
                uintptr_t rec = task.recovered_bufs[slot];
                send_threads.emplace_back([this, frig, rec, sid = task.stripe_id, bs = task.block_size]() {
                    uint64_t t_send = frcheck_now_us();
                    send_to_peer(frig, sid, rec, bs);
                    recovery_decoder_send_us_.fetch_add(frcheck_now_us() - t_send, std::memory_order_relaxed);
                });
            }
            for (auto& t : send_threads) t.join();
        }
        recovery_decoder_tasks_.fetch_add(1, std::memory_order_relaxed);
    }

    void execute_recovery_failed_(const RecoveryFailedTask& task) {
        uint64_t t_recv = frcheck_now_us();
        recv_from_peer(task.decoder_rig, task.stripe_id,
                       task.recv_buf, task.block_size);
        recovery_failed_recv_us_.fetch_add(frcheck_now_us() - t_recv, std::memory_order_relaxed);
        if (task.store_to_layer && task.layer_buf != 0 && task.ncopy > 0) {
            uint64_t t_copy = frcheck_now_us();
            std::memcpy(
                reinterpret_cast<void*>(task.layer_buf + task.layer_offset),
                reinterpret_cast<const void*>(task.recv_buf),
                task.ncopy);
            recovery_failed_copy_us_.fetch_add(frcheck_now_us() - t_copy, std::memory_order_relaxed);
        }
        recovery_failed_tasks_.fetch_add(1, std::memory_order_relaxed);
    }

    void helper_worker_loop_() {
        while (!recovery_workers_stop_) {
            RecoveryHelperTask task;
            {
                std::unique_lock<std::mutex> lk(helper_mtx_);
                helper_cv_.wait(lk, [this] {
                    return recovery_workers_stop_ || !helper_q_.empty();
                });
                if (recovery_workers_stop_) break;
                task = helper_q_.front();
                helper_q_.pop();
            }
            if (is_recovery_helper_sentinel_(task)) {
                helper_sentinels_received_.fetch_add(1, std::memory_order_acq_rel);
                maybe_complete_helper_after_sentinel_();
                continue;
            }
            helper_active_.fetch_add(1, std::memory_order_acq_rel);
            try {
                execute_recovery_helper_(task);
            } catch (const std::exception& e) {
                helper_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                std::cerr << "FRCheck helper_worker: " << e.what() << std::endl;
                return;
            } catch (...) {
                helper_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                throw;
            }
            helper_active_.fetch_sub(1, std::memory_order_acq_rel);
            pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
            maybe_complete_helper_after_sentinel_();
        }
    }

    void decoder_worker_loop_() {
        while (!recovery_workers_stop_) {
            RecoveryDecoderTask task;
            {
                std::unique_lock<std::mutex> lk(decoder_mtx_);
                decoder_cv_.wait(lk, [this] {
                    return recovery_workers_stop_ || !decoder_q_.empty();
                });
                if (recovery_workers_stop_) break;
                task = decoder_q_.front();
                decoder_q_.pop();
            }
            if (is_recovery_decoder_sentinel_(task)) {
                decoder_sentinels_received_.fetch_add(1, std::memory_order_acq_rel);
                maybe_complete_decoder_after_sentinel_();
                continue;
            }
            decoder_active_.fetch_add(1, std::memory_order_acq_rel);
            try {
                execute_recovery_decoder_(task);
            } catch (const std::exception& e) {
                decoder_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                std::cerr << "FRCheck decoder_worker: " << e.what() << std::endl;
                return;
            } catch (...) {
                decoder_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                throw;
            }
            decoder_active_.fetch_sub(1, std::memory_order_acq_rel);
            pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
            maybe_complete_decoder_after_sentinel_();
        }
    }

    void failed_worker_loop_() {
        while (!recovery_workers_stop_) {
            RecoveryFailedTask task;
            {
                std::unique_lock<std::mutex> lk(failed_mtx_);
                failed_cv_.wait(lk, [this] {
                    return recovery_workers_stop_ || !failed_q_.empty();
                });
                if (recovery_workers_stop_) break;
                task = failed_q_.front();
                failed_q_.pop();
            }
            if (is_recovery_failed_sentinel_(task)) {
                failed_sentinels_received_.fetch_add(1, std::memory_order_acq_rel);
                maybe_complete_failed_after_sentinel_();
                continue;
            }
            failed_active_.fetch_add(1, std::memory_order_acq_rel);
            try {
                execute_recovery_failed_(task);
            } catch (const std::exception& e) {
                failed_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                std::cerr << "FRCheck failed_worker: " << e.what() << std::endl;
                return;
            } catch (...) {
                failed_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                throw;
            }
            failed_active_.fetch_sub(1, std::memory_order_acq_rel);
            pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
            maybe_complete_failed_after_sentinel_();
        }
    }

    void ensure_recovery_workers_() {
        if (recovery_workers_inited_) return;
        recovery_workers_stop_ = false;
        int nw = std::max(1, n_);
        helper_threads_.reserve((size_t)nw);
        decoder_threads_.reserve((size_t)nw);
        failed_threads_.reserve((size_t)nw);
        for (int w = 0; w < nw; ++w) {
            helper_threads_.emplace_back(&FRCheckNative::helper_worker_loop_, this);
            decoder_threads_.emplace_back(&FRCheckNative::decoder_worker_loop_, this);
            failed_threads_.emplace_back(&FRCheckNative::failed_worker_loop_, this);
        }
        recovery_workers_inited_ = true;
        std::cout << "FRCheck: recovery workers started (helper/decoder/failed x"
                  << nw << ")" << std::endl;
    }

    void recovery_workers_join_() {
        for (auto& t : helper_threads_) if (t.joinable()) t.join();
        for (auto& t : decoder_threads_) if (t.joinable()) t.join();
        for (auto& t : failed_threads_) if (t.joinable()) t.join();
        helper_threads_.clear();
        decoder_threads_.clear();
        failed_threads_.clear();
        recovery_workers_inited_ = false;
    }

    void notify_recovery_wait_() { recovery_wait_cv_.notify_all(); }

    void maybe_complete_recovery_batch_() {
        if (pending_recovery_chunks_.load(std::memory_order_acquire) != 0) return;
        if (helper_active_.load(std::memory_order_acquire) != 0) return;
        if (decoder_active_.load(std::memory_order_acquire) != 0) return;
        if (failed_active_.load(std::memory_order_acquire) != 0) return;
        if (wait_helper_ && !helper_completed_) return;
        if (wait_decoder_ && !decoder_completed_) return;
        if (wait_failed_ && !failed_completed_) return;
        std::lock_guard<std::mutex> lk(recovery_wait_mtx_);
        if (recovery_batch_completed_) return;
        recovery_batch_completed_ = true;
        notify_recovery_wait_();
    }

    void maybe_complete_helper_after_sentinel_() {
        if (helper_sentinels_received_.load(std::memory_order_acquire)
            < (int)helper_threads_.size()) return;
        if (helper_active_.load(std::memory_order_acquire) != 0) return;
        std::lock_guard<std::mutex> lk(helper_mtx_);
        if (helper_q_.empty()) {
            helper_completed_ = true;
            maybe_complete_recovery_batch_();
        }
    }

    void maybe_complete_decoder_after_sentinel_() {
        if (decoder_sentinels_received_.load(std::memory_order_acquire)
            < (int)decoder_threads_.size()) return;
        if (decoder_active_.load(std::memory_order_acquire) != 0) return;
        std::lock_guard<std::mutex> lk(decoder_mtx_);
        if (decoder_q_.empty()) {
            decoder_completed_ = true;
            maybe_complete_recovery_batch_();
        }
    }

    void maybe_complete_failed_after_sentinel_() {
        if (failed_sentinels_received_.load(std::memory_order_acquire)
            < (int)failed_threads_.size()) return;
        if (failed_active_.load(std::memory_order_acquire) != 0) return;
        std::lock_guard<std::mutex> lk(failed_mtx_);
        if (failed_q_.empty()) {
            failed_completed_ = true;
            maybe_complete_recovery_batch_();
        }
    }

    void reset_recovery_batch_profile_() {
        recovery_helper_send_us_.store(0, std::memory_order_relaxed);
        recovery_decoder_recv_us_.store(0, std::memory_order_relaxed);
        recovery_decoder_decode_us_.store(0, std::memory_order_relaxed);
        recovery_decoder_send_us_.store(0, std::memory_order_relaxed);
        recovery_failed_recv_us_.store(0, std::memory_order_relaxed);
        recovery_failed_copy_us_.store(0, std::memory_order_relaxed);
        recovery_helper_tasks_.store(0, std::memory_order_relaxed);
        recovery_decoder_tasks_.store(0, std::memory_order_relaxed);
        recovery_failed_tasks_.store(0, std::memory_order_relaxed);
    }

    void print_recovery_batch_profile_() {
        auto us_to_s = [](uint64_t us) { return (double)us / 1000000.0; };
        const int helper_tasks = recovery_helper_tasks_.load(std::memory_order_relaxed);
        const int decoder_tasks = recovery_decoder_tasks_.load(std::memory_order_relaxed);
        const int failed_tasks = recovery_failed_tasks_.load(std::memory_order_relaxed);
        if (helper_tasks == 0 && decoder_tasks == 0 && failed_tasks == 0) return;
        std::cout << "FRCheck native profile: rank_in_group=" << rank_in_group_
                  << " helper_tasks=" << helper_tasks
                  << " decoder_tasks=" << decoder_tasks
                  << " failed_tasks=" << failed_tasks
                  << " helper_send_s=" << us_to_s(recovery_helper_send_us_.load(std::memory_order_relaxed))
                  << " decoder_recv_s=" << us_to_s(recovery_decoder_recv_us_.load(std::memory_order_relaxed))
                  << " decoder_decode_s=" << us_to_s(recovery_decoder_decode_us_.load(std::memory_order_relaxed))
                  << " decoder_send_s=" << us_to_s(recovery_decoder_send_us_.load(std::memory_order_relaxed))
                  << " failed_recv_s=" << us_to_s(recovery_failed_recv_us_.load(std::memory_order_relaxed))
                  << " failed_copy_s=" << us_to_s(recovery_failed_copy_us_.load(std::memory_order_relaxed))
                  << std::endl;
    }

    void reset_recovery_completion_() {
        recovery_batch_completed_ = false;
        pending_recovery_chunks_.store(0, std::memory_order_release);
        helper_active_.store(0, std::memory_order_release);
        decoder_active_.store(0, std::memory_order_release);
        failed_active_.store(0, std::memory_order_release);
        wait_helper_ = false;
        wait_decoder_ = false;
        wait_failed_ = false;
        helper_completed_ = false;
        decoder_completed_ = false;
        failed_completed_ = false;
        helper_sentinels_received_.store(0, std::memory_order_release);
        decoder_sentinels_received_.store(0, std::memory_order_release);
        failed_sentinels_received_.store(0, std::memory_order_release);
        reset_recovery_batch_profile_();
        {
            std::lock_guard<std::mutex> lk(helper_mtx_);
            while (!helper_q_.empty()) helper_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lk(decoder_mtx_);
            while (!decoder_q_.empty()) decoder_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lk(failed_mtx_);
            while (!failed_q_.empty()) failed_q_.pop();
        }
    }

    // ---- RDMA device init ----
    void init_ibv_() {
        if (ibv_fork_init() != 0)
            std::cerr << "[FRCheck RDMA] WARNING: ibv_fork_init() failed" << std::endl;

        int ndev;
        ibv_device** devs = ibv_get_device_list(&ndev);
        if (!devs || ndev == 0)
            throw std::runtime_error("FRCheck RDMA: no IB devices found");
        rdma_ctx_ = ibv_open_device(find_rdma_device_by_ip(my_ip_, devs, ndev));
        ibv_free_device_list(devs);
        if (!rdma_ctx_)
            throw std::runtime_error("FRCheck RDMA: failed to open device");

        rdma_pd_ = ibv_alloc_pd(rdma_ctx_);
        if (!rdma_pd_)
            throw std::runtime_error("FRCheck RDMA: failed to alloc PD");
    }

    void cleanup_rdma_() {
        if (rdma_cleaned_up_.exchange(true)) return;

        if (!stopped_.load()) {
            shutdown_stripe_workers();
        }
        channel_owners_.clear();
        channels_.clear();

        acceptor_thread_stop_ = true;
        if (acceptor_fd_ >= 0) {
            shutdown(acceptor_fd_, SHUT_RDWR);
            close(acceptor_fd_);
            acceptor_fd_ = -1;
        }
        accept_cv_.notify_all();
        if (accept_thread_.joinable())
            accept_thread_.join();

        for (auto& kv : registered_bufs_) {
            if (kv.second.mr) ibv_dereg_mr(kv.second.mr);
        }
        registered_bufs_.clear();

        if (rdma_pd_) { ibv_dealloc_pd(rdma_pd_); rdma_pd_ = nullptr; }
        if (rdma_ctx_) { ibv_close_device(rdma_ctx_); rdma_ctx_ = nullptr; }
        n_connected_ = 0;
    }

    // ---- TCP helpers ----
    int create_tcp_connect(const std::string& ip, uint16_t port) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error("FRCheck: socket() failed");
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
            close(fd);
            throw std::runtime_error("FRCheck: invalid IP: " + ip);
        }

        // Retry with backoff — needed for single-machine multi-node sim
        // where the remote acceptor may not have completed bind+listen yet.
        int attempt = 0;
        const int max_attempts = 20;
        const int base_delay_us = 5000;  // 5 ms
        while (true) {
            if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0)
                return fd;
            if (++attempt >= max_attempts) {
                int err = errno;
                close(fd);
                throw std::runtime_error("FRCheck: connect to " + ip + ":" +
                                         std::to_string(port) + " failed after " +
                                         std::to_string(max_attempts) + " attempts: " +
                                         std::strerror(err));
            }
            usleep(base_delay_us * attempt);  // linear backoff: 5, 10, 15, ... ms
        }
    }

    void accept_loop_() {
        while (!acceptor_thread_stop_) {
            try {
                if (acceptor_fd_ < 0) break;
                struct pollfd pfd;
                pfd.fd = acceptor_fd_;
                pfd.events = POLLIN;
                int pr = poll(&pfd, 1, 100);
                if (pr < 0) {
                    if (errno == EINTR) continue;
                    break;
                }
                if (pr == 0) continue;

                struct sockaddr_in peer_addr;
                socklen_t addrlen = sizeof(peer_addr);
                int fd = accept(acceptor_fd_,
                                (struct sockaddr*)&peer_addr, &addrlen);
                if (fd < 0) {
                    if (errno == EINTR) continue;
                    if (errno == EINVAL || errno == EBADF) break; // closed
                    continue;
                }
                // Receive peer rank + lane (stripe_id)
                ConnectHello hello{};
                ssize_t nr = recv(fd, &hello, sizeof(hello), MSG_WAITALL);
                if (nr != (ssize_t)sizeof(hello)) {
                    std::cerr << "[FRCheck RDMA] accept: failed to recv connect hello" << std::endl;
                    close(fd);
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lk(accept_mtx_);
                    accepted_queue_[hello.rank][hello.lane_id] = fd;
                }
                accept_cv_.notify_all();
            } catch (const std::exception& e) {
                if (!acceptor_thread_stop_)
                    std::cerr << "[FRCheck RDMA] accept error: " << e.what() << std::endl;
                break;
            }
        }
    }

    // ---- Channel access ----
    FRCheckRdmaChannel* get_channel_(int peer_rg, int stripe_id) {
        if (peer_rg < 0 || peer_rg >= group_size_ || peer_rg == rank_in_group_)
            return nullptr;
        if (stripe_id < 0 || stripe_id >= num_lanes_)
            return nullptr;
        return channels_[peer_rg][stripe_id];
    }

public:
    // ---- StripePlan compilation ----
    void compile_plans(int rank_in_group) {
        rank_in_group_ = rank_in_group;
        compile_stripe_plans_();
    }

private:
    void compile_stripe_plans_() {
        stripe_plans_.clear();
        for (int sid = 0; sid < (int)table_.size(); ++sid) {
            const auto& poa_row = table_[sid];
            StripePlan plan;
            plan.stripe_id = sid;
            plan.row = poa_row;
            plan.encoder_node_id = poa_row[n_ - 2];
            plan.parity_target_node_id = poa_row[n_ - 1];
            plan.source_node_ids.assign(poa_row.begin(), poa_row.begin() + (n_ - 2));

            int my_node = rank_in_group_ + 1; // rank_in_group is 0-based, POA IDs are 1-based
            if (my_node == plan.encoder_node_id) {
                plan.role = StripeRole::ENCODER;
            } else if (my_node == plan.parity_target_node_id) {
                plan.role = StripeRole::PARITY_TARGET;
            } else {
                // Check if I'm a source
                bool is_src = false;
                for (int src : plan.source_node_ids) {
                    if (src == my_node) { is_src = true; break; }
                }
                if (is_src) plan.role = StripeRole::SOURCE;
                else throw std::runtime_error(
                    "FRCheck: node " + std::to_string(my_node) +
                    " not found in stripe " + std::to_string(sid));
            }
            stripe_plans_.push_back(std::move(plan));
        }
    }

    // ---- data members ----
    // POA
    std::string poa_path_;
    int n_ = -1;
    std::vector<std::vector<int>> table_;
    unsigned char* a_mat_ = nullptr;  // ISA-L RS generator matrix (rows_ × k_)
    unsigned char* g_tbls_ = nullptr; // ISA-L RS encode tables (32*k_*rows_)
    unsigned char* decode_tbls_ = nullptr; // ISA-L RS decode tables (for recovery)
    std::mutex decode_mtx_;
    std::atomic<bool> stopped_;
    std::atomic<bool> rdma_cleaned_up_{false};

    // RDMA
    ibv_context* rdma_ctx_;
    ibv_pd* rdma_pd_;

    // Topology
    int group_size_ = 0;
    int rank_in_group_ = -1;
    int n_connected_ = 0;
    std::string my_ip_;

    // Channels: index by peer rank_in_group and stripe lane (lane_id == stripe_id)
    int num_lanes_ = 0;
    std::vector<std::vector<FRCheckRdmaChannel*>> channels_;
    std::vector<std::unique_ptr<FRCheckRdmaChannel>> channel_owners_;
    bool require_registered_mr_ = true;

    // Registered buffers (shared across channels)
    std::map<uintptr_t, RdmaBuffer> registered_bufs_;
    std::mutex buf_mtx_;

    // TCP acceptor (raw socket)
    int acceptor_fd_ = -1;
    std::thread accept_thread_;
    std::atomic<bool> acceptor_thread_stop_{false};
    std::map<int, std::map<int, int>> accepted_queue_; // peer_rank -> lane_id -> fd
    std::mutex accept_mtx_;
    std::condition_variable accept_cv_;

    // Stripe plans (pre-compiled)
    std::vector<StripePlan> stripe_plans_;

    // Hardware recovery
    std::vector<RecoveryStripePlan> recovery_plans_;
    bool recovery_dual_failure_ = false;
    bool recovery_batch_active_ = false;
    bool encoding_batch_active_ = false;
    bool recovery_workers_inited_ = false;
    std::atomic<bool> recovery_workers_stop_{false};
    std::queue<RecoveryHelperTask> helper_q_;
    std::mutex helper_mtx_;
    std::condition_variable helper_cv_;
    std::atomic<int> helper_sentinels_received_{0};
    bool helper_completed_ = false;
    bool wait_helper_ = false;
    std::queue<RecoveryDecoderTask> decoder_q_;
    std::mutex decoder_mtx_;
    std::condition_variable decoder_cv_;
    std::atomic<int> decoder_sentinels_received_{0};
    bool decoder_completed_ = false;
    bool wait_decoder_ = false;
    std::queue<RecoveryFailedTask> failed_q_;
    std::mutex failed_mtx_;
    std::condition_variable failed_cv_;
    std::atomic<int> failed_sentinels_received_{0};
    bool failed_completed_ = false;
    bool wait_failed_ = false;
    std::vector<std::thread> helper_threads_;
    std::vector<std::thread> decoder_threads_;
    std::vector<std::thread> failed_threads_;
    std::atomic<int> pending_recovery_chunks_{0};
    std::atomic<int> helper_active_{0};
    std::atomic<int> decoder_active_{0};
    std::atomic<int> failed_active_{0};
    bool recovery_batch_completed_ = false;
    std::mutex recovery_wait_mtx_;
    std::condition_variable recovery_wait_cv_;
    std::atomic<uint64_t> recovery_helper_send_us_{0};
    std::atomic<uint64_t> recovery_decoder_recv_us_{0};
    std::atomic<uint64_t> recovery_decoder_decode_us_{0};
    std::atomic<uint64_t> recovery_decoder_send_us_{0};
    std::atomic<uint64_t> recovery_failed_recv_us_{0};
    std::atomic<uint64_t> recovery_failed_copy_us_{0};
    std::atomic<int> recovery_helper_tasks_{0};
    std::atomic<int> recovery_decoder_tasks_{0};
    std::atomic<int> recovery_failed_tasks_{0};

    // ---- RS encode thread pool (matches ecnaive xor_pool pattern) ----
    static constexpr int kRsPoolWorkers = 16;
    std::array<pthread_t, kRsPoolWorkers> rs_pool_threads_{};
    std::array<RsPoolWorkerCtx, kRsPoolWorkers> rs_pool_ctx_{};
    std::array<int, kRsPoolWorkers> rs_pool_cpus_{};
    std::atomic<bool> rs_pool_inited_{false};
    std::atomic<bool> rs_pool_stop_{false};
    std::mutex rs_pool_mutex_;
    std::condition_variable rs_pool_worker_cv_;
    std::condition_variable rs_pool_coordinator_cv_;
    std::atomic<uint64_t> rs_pool_epoch_{0};
    std::array<uint64_t, kRsPoolWorkers> rs_pool_last_epoch_{};
    std::atomic<int> rs_pool_remaining_{0};
    RsEncodeJob rs_pool_shared_job_{};
};

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(frcheck_native, m) {
    py::class_<FRCheckNative>(m, "FRCheckNative")
        .def(py::init<const std::string&>(), py::arg("poa_file_path"))
        .def(py::init<int>(), py::arg("n"))

        // POA queries
        .def("n", &FRCheckNative::n)
        .def("num_stripes", &FRCheckNative::num_stripes)
        .def("entry", &FRCheckNative::entry, py::arg("row"), py::arg("col"))
        .def("row", &FRCheckNative::row, py::arg("row"))
        .def("path", &FRCheckNative::path)
        .def("stop", &FRCheckNative::stop)
        .def("is_stopped", &FRCheckNative::is_stopped)

        // Stripe plan compilation (standalone, no RDMA needed)
        .def("compile_plans", &FRCheckNative::compile_plans,
             py::arg("rank_in_group"))

        // RDMA init
        .def("init_rdma", &FRCheckNative::init_rdma,
             py::arg("group_size"),
             py::arg("rank_in_group"),
             py::arg("base_port"),
             py::arg("my_ip"),
             py::arg("peer_ips"),
             py::arg("use_rdma") = true)

        .def("group_size", &FRCheckNative::group_size)
        .def("rank_in_group", &FRCheckNative::rank_in_group)

        // Buffer registration
        .def("register_buffer", &FRCheckNative::register_buffer,
             py::arg("addr"), py::arg("size"))
        .def("unregister_buffer", &FRCheckNative::unregister_buffer,
             py::arg("addr"))

        // GDR capability
        .def_static("gdr_available", &FRCheckNative::gdr_available)

        // Stripe decode (hardware recovery)
        .def("submit_stripe_decode", &FRCheckNative::submit_stripe_decode,
             py::arg("k"),
             py::arg("survivor_positions"),
             py::arg("lost_position"),
             py::arg("survivor_addrs"),
             py::arg("recovered_addr"),
             py::arg("block_size"))

        // Point-to-point RDMA send/recv (for recovery, GIL released for threading)
        .def("send_to_peer", &FRCheckNative::send_to_peer,
             py::arg("peer_rig"), py::arg("stripe_id"),
             py::arg("addr"), py::arg("size"),
             py::call_guard<py::gil_scoped_release>())
        .def("recv_from_peer", &FRCheckNative::recv_from_peer,
             py::arg("peer_rig"), py::arg("stripe_id"),
             py::arg("addr"), py::arg("size"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_require_registered_mr", &FRCheckNative::set_require_registered_mr,
             py::arg("require"))
        .def("set_debug", &FRCheckNative::set_debug,
             py::arg("debug"))

        // Hardware recovery batch pipeline
        .def("init_recovery_plans", &FRCheckNative::init_recovery_plans,
             py::arg("failed_nodes_1based"))
        .def("reset_recovery_batch", &FRCheckNative::reset_recovery_batch)
        .def("submit_recovery_stripe", &FRCheckNative::submit_recovery_stripe,
             py::arg("stripe_id"),
             py::arg("block_size"),
             py::arg("helper_block_addr"),
             py::arg("decoder_self_block_addr"),
             py::arg("decoder_helper_recv_addrs"),
             py::arg("decoder_recovered_addrs"),
             py::arg("failed_recv_buf_addr"),
             py::arg("failed_layer_buf_addr"),
             py::arg("failed_layer_offset"),
             py::arg("failed_ncopy"),
             py::arg("store_to_layer_buf"))
        .def("submit_recovery_sentinel", &FRCheckNative::submit_recovery_sentinel)
        .def("wait_recovery_batch", &FRCheckNative::wait_recovery_batch,
             py::call_guard<py::gil_scoped_release>())
        .def("stop_recovery_runtime", &FRCheckNative::stop_recovery_runtime,
             py::call_guard<py::gil_scoped_release>())
        .def("cleanup_recovery_runtime", &FRCheckNative::cleanup_recovery_runtime,
             py::call_guard<py::gil_scoped_release>())

        // StripePlan queries
        .def("get_role_for_stripe", &FRCheckNative::get_role_for_stripe,
             py::arg("stripe_id"))
        .def("get_source_node_ids", &FRCheckNative::get_source_node_ids,
             py::arg("stripe_id"))
        .def("get_encoder_node_id", &FRCheckNative::get_encoder_node_id,
             py::arg("stripe_id"))
        .def("get_parity_target_node_id", &FRCheckNative::get_parity_target_node_id,
             py::arg("stripe_id"))

        // Per-stripe worker encoding API
        .def("reset_layer", &FRCheckNative::reset_layer)
        .def("submit_source", &FRCheckNative::submit_source,
             py::arg("stripe_id"), py::arg("data_addr"), py::arg("mirror_addr"), py::arg("block_size"))
        .def("submit_enc_recv", &FRCheckNative::submit_enc_recv,
             py::arg("stripe_id"), py::arg("recv_addr"), py::arg("p1_addr"),
             py::arg("p2_addr"), py::arg("block_size"),
             py::arg("source_active_mask") = std::vector<uint8_t>())
        .def("submit_parity", &FRCheckNative::submit_parity,
             py::arg("stripe_id"), py::arg("parity_in_addr"), py::arg("block_size"))
        .def("submit_p1_send", &FRCheckNative::submit_p1_send,
             py::arg("stripe_id"), py::arg("p1_addr"), py::arg("block_size"))
        .def("submit_p2_send", &FRCheckNative::submit_p2_send,
             py::arg("stripe_id"), py::arg("p2_addr"), py::arg("block_size"))
        .def("submit_async_p2_batch", &FRCheckNative::submit_async_p2_batch,
             py::arg("parity_tasks"), py::arg("p2_send_tasks"))
        .def("submit_async_p2_layer", &FRCheckNative::submit_async_p2_layer,
             py::arg("p2_addrs"), py::arg("block_size"))
        .def("reset_async_parity", &FRCheckNative::reset_async_parity)
        .def("wait_layer", &FRCheckNative::wait_layer,
             py::call_guard<py::gil_scoped_release>())
        .def("wait_encode_only", &FRCheckNative::wait_encode_only,
             py::call_guard<py::gil_scoped_release>())
        .def("wait_parity_flush", &FRCheckNative::wait_parity_flush,
             py::call_guard<py::gil_scoped_release>())
        .def("inc_pause_async_p2p", &FRCheckNative::inc_pause_async_p2p)
        .def("dec_pause_async_p2p", &FRCheckNative::dec_pause_async_p2p)
        .def("wait_mirror_completion", &FRCheckNative::wait_mirror_completion);
}
