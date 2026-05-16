#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <iostream>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// RDMA headers
#include <infiniband/verbs.h>

#include "rdma_device_utils.h"

// ISA-L erasure coding
#include <isa-l/erasure_code.h>

// Per-stripe file header (matches legacy_io_utils binary format)
struct StripeFileHeader {
    char magic[4];        // "FRBK"
    uint32_t stripe_id;
    uint32_t role;        // 0=SOURCE, 1=ENCODER, 2=PARITY_TARGET
    uint64_t data_size;   // actual payload bytes following header
    uint64_t block_size;  // nominal block size
};

static void write_stripe_file(const std::string& layer_dir, int stripe_id, int role,
                               int rank, const uint8_t* data, uint64_t data_size,
                               uint64_t block_size, const char* suffix = "") {
    std::string dir = layer_dir + "/stripe_" + std::to_string(stripe_id);
    mkdir(dir.c_str(), 0755);
    std::string path = dir + "/frcheck_shard_rank" + std::to_string(rank) + suffix + ".pt";
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        std::cerr << "[FRCheck] WARNING: failed to open " << path << std::endl;
        return;
    }
    StripeFileHeader hdr{};
    memcpy(hdr.magic, "FRBK", 4);
    hdr.stripe_id = (uint32_t)stripe_id;
    hdr.role = (uint32_t)role;
    hdr.data_size = data_size;
    hdr.block_size = block_size;
    fwrite(&hdr, sizeof(hdr), 1, fp);
    if (data_size > 0 && data)
        fwrite(data, 1, (size_t)data_size, fp);
    fclose(fp);
}

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
static constexpr size_t FRCHECK_TEMP_BUF_SIZE = 128ULL * 1024 * 1024; // 128 MB
static constexpr size_t FRCHECK_RDMA_CHUNK = 64ULL * 1024 * 1024;     // 64 MB per RDMA op
static constexpr int    FRCHECK_MAX_WR = 64;

class FRCheckRdmaChannel {
public:
    FRCheckRdmaChannel(ibv_context* ctx, ibv_pd* pd,
                       ibv_cq* send_cq, ibv_cq* recv_cq,
                       int tcp_sock, int peer_rank,
                       std::map<uintptr_t, RdmaBuffer>* bufs,
                       std::mutex* buf_mtx)
        : ctx_(ctx), pd_(pd),
          send_cq_(send_cq), recv_cq_(recv_cq),
          tcp_sock_(tcp_sock), peer_rank_(peer_rank),
          bufs_(bufs), buf_mtx_(buf_mtx),
          qp_(nullptr), temp_send_mr_(nullptr), temp_recv_mr_(nullptr),
          connected_(false)
    {
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

        temp_send_.resize(FRCHECK_TEMP_BUF_SIZE);
        temp_recv_.resize(FRCHECK_TEMP_BUF_SIZE);
        temp_send_mr_ = ibv_reg_mr(pd_, temp_send_.data(), FRCHECK_TEMP_BUF_SIZE,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        temp_recv_mr_ = ibv_reg_mr(pd_, temp_recv_.data(), FRCHECK_TEMP_BUF_SIZE,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!temp_send_mr_ || !temp_recv_mr_)
            throw std::runtime_error("FRCheck RDMA: failed to register temp MRs");
    }

    ~FRCheckRdmaChannel() {
        if (temp_recv_mr_) ibv_dereg_mr(temp_recv_mr_);
        if (temp_send_mr_) ibv_dereg_mr(temp_send_mr_);
        if (qp_) ibv_destroy_qp(qp_);
        if (tcp_sock_ >= 0) close(tcp_sock_);
    }

    int peer_rank() const { return peer_rank_; }
    bool is_connected() const { return connected_; }

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
    void send_data(const uint8_t* data, size_t size) {
        std::lock_guard<std::mutex> lock(send_mtx_);
        if (!connected_)
            throw std::runtime_error("FRCheck RDMA: channel not connected");

        // Send size over TCP
        uint64_t net_sz = htobe64(size);
        if (send(tcp_sock_, &net_sz, sizeof(net_sz), 0) != sizeof(net_sz))
            throw std::runtime_error("FRCheck RDMA: failed to send size");
        uint8_t ack;
        if (recv(tcp_sock_, &ack, 1, MSG_WAITALL) != 1)
            throw std::runtime_error("FRCheck RDMA: failed to recv ack");

        ibv_mr* mr = find_mr((uintptr_t)data, size);
        if (!mr) {
            if (size > FRCHECK_TEMP_BUF_SIZE)
                throw std::runtime_error("FRCheck RDMA: data exceeds temp buffer");
            memcpy(temp_send_.data(), data, size);
            mr = temp_send_mr_;
            data = temp_send_.data();
        }
        send_chunked(data, size, mr);
    }

    // RDMA RECEIVE data from peer (blocking)
    size_t recv_data(uint8_t* buf, size_t buf_size) {
        std::lock_guard<std::mutex> lock(recv_mtx_);
        if (!connected_)
            throw std::runtime_error("FRCheck RDMA: channel not connected");

        uint64_t net_sz;
        if (recv(tcp_sock_, &net_sz, sizeof(net_sz), MSG_WAITALL) != sizeof(net_sz))
            throw std::runtime_error("FRCheck RDMA: failed to recv size");
        size_t size = be64toh(net_sz);
        if (size > buf_size)
            throw std::runtime_error("FRCheck RDMA: recv size exceeds buffer");

        uint8_t ack = 1;
        if (send(tcp_sock_, &ack, 1, 0) != 1)
            throw std::runtime_error("FRCheck RDMA: failed to send ack");

        ibv_mr* mr = find_mr((uintptr_t)buf, size);
        uint8_t* target = buf;
        bool use_temp = false;
        if (!mr) {
            if (size > FRCHECK_TEMP_BUF_SIZE)
                throw std::runtime_error("FRCheck RDMA: recv exceeds temp buffer");
            mr = temp_recv_mr_;
            target = temp_recv_.data();
            use_temp = true;
        }
        recv_chunked(target, size, mr);
        if (use_temp) memcpy(buf, temp_recv_.data(), size);
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

    void send_chunked(const uint8_t* data, size_t total, ibv_mr* mr) {
        _send_chunked(data, total, mr, 0, true);
    }

    void recv_chunked(uint8_t* buf, size_t total, ibv_mr* mr) {
        _recv_chunked(buf, total, mr, 0, true);
    }

public:
    // Async post (no TCP handshake, no poll). Only last WR signaled → 1 CQ completion.
    void post_send(const uint8_t* data, size_t total, ibv_mr* mr, uint64_t wr_id) {
        _send_chunked(data, total, mr, wr_id, false);
    }

    void post_recv(uint8_t* buf, size_t total, ibv_mr* mr, uint64_t wr_id) {
        _recv_chunked(buf, total, mr, wr_id, false);
    }

    // Poll a single completion, return its wr_id.
    uint64_t poll_one_send_cq() { return _poll_one(send_cq_); }
    uint64_t poll_one_recv_cq() { return _poll_one(recv_cq_); }

    ibv_cq* get_send_cq() const { return send_cq_; }
    ibv_cq* get_recv_cq() const { return recv_cq_; }

    void _send_chunked(const uint8_t* data, size_t total, ibv_mr* mr, uint64_t wr_id, bool sync) {
        size_t remaining = total, offset = 0;
        while (remaining > 0) {
            size_t nchunks = (std::min(remaining, FRCHECK_RDMA_CHUNK) + FRCHECK_RDMA_CHUNK - 1) / FRCHECK_RDMA_CHUNK;
            std::vector<ibv_sge> sge(nchunks);
            std::vector<ibv_send_wr> wr(nchunks);
            for (size_t i = 0; i < nchunks; ++i) {
                size_t cur = std::min(FRCHECK_RDMA_CHUNK, remaining);
                sge[i].addr = (uint64_t)(data + offset);
                sge[i].length = (uint32_t)cur;
                sge[i].lkey = mr->lkey;
                memset(&wr[i], 0, sizeof(wr[i]));
                wr[i].wr_id = wr_id;
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
            if (sync) poll_cq(send_cq_, (int)nchunks);
        }
    }

    void _recv_chunked(uint8_t* buf, size_t total, ibv_mr* mr, uint64_t wr_id, bool sync) {
        size_t remaining = total, offset = 0;
        while (remaining > 0) {
            size_t nchunks = (std::min(remaining, FRCHECK_RDMA_CHUNK) + FRCHECK_RDMA_CHUNK - 1) / FRCHECK_RDMA_CHUNK;
            std::vector<ibv_sge> sge(nchunks);
            std::vector<ibv_recv_wr> wr(nchunks);
            for (size_t i = 0; i < nchunks; ++i) {
                size_t cur = std::min(FRCHECK_RDMA_CHUNK, remaining);
                sge[i].addr = (uint64_t)(buf + offset);
                sge[i].length = (uint32_t)cur;
                sge[i].lkey = mr->lkey;
                memset(&wr[i], 0, sizeof(wr[i]));
                wr[i].wr_id = wr_id;
                wr[i].sg_list = &sge[i];
                wr[i].num_sge = 1;
                wr[i].next = (i + 1 < nchunks) ? &wr[i + 1] : nullptr;
                offset += cur;
                remaining -= cur;
            }
            ibv_recv_wr* bad = nullptr;
            if (ibv_post_recv(qp_, &wr[0], &bad))
                throw std::runtime_error("FRCheck RDMA: post_recv failed");
            if (sync) poll_cq(recv_cq_, (int)nchunks);
        }
    }

    void poll_cq(ibv_cq* cq, int count) {
        ibv_wc wc[FRCHECK_MAX_WR];
        int done = 0;
        while (done < count) {
            int n = ibv_poll_cq(cq, std::min(count - done, FRCHECK_MAX_WR), wc);
            if (n < 0) throw std::runtime_error("FRCheck RDMA: poll CQ error");
            for (int i = 0; i < n; ++i) {
                if (wc[i].status != IBV_WC_SUCCESS)
                    throw std::runtime_error("FRCheck RDMA: CQ error status=" + std::to_string(wc[i].status));
                ++done;
            }
        }
    }

    uint64_t _poll_one(ibv_cq* cq) {
        ibv_wc wc;
        while (true) {
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) throw std::runtime_error("FRCheck RDMA: poll CQ error");
            if (n == 0) { usleep(1); continue; }
            if (wc.status != IBV_WC_SUCCESS)
                throw std::runtime_error("FRCheck RDMA: CQ error status=" + std::to_string(wc.status));
            return wc.wr_id;
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
    std::vector<uint8_t> temp_send_;
    std::vector<uint8_t> temp_recv_;
    ibv_mr* temp_send_mr_;
    ibv_mr* temp_recv_mr_;
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
        stop();
        cleanup_rdma_();
        if (a_mat_) { free(a_mat_); a_mat_ = nullptr; }
        if (g_tbls_) { free(g_tbls_); g_tbls_ = nullptr; }
    }

    // ---- POA query (existing) ----
    int n() const { return n_; }
    int num_stripes() const { return (int)table_.size(); }
    int entry(int r, int c) const { return table_.at(r).at(c); }
    std::vector<int> row(int r) const { return table_.at(r); }
    std::string path() const { return poa_path_; }
    void stop() {
        if (stopped_.exchange(true)) return; // already stopped
        // Notify RS pool threads (matches ecnaive pattern)
        rs_pool_worker_cv_.notify_all();
        rs_pool_coordinator_cv_.notify_one();
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
        if (listen(acceptor_fd_, group_size) < 0)
            throw std::runtime_error("FRCheck RDMA: listen failed");

        // Launch accept thread
        acceptor_thread_stop_ = false;
        accept_thread_ = std::thread([this]() { accept_loop_(); });

        // For i = 0..group_size-1, i != rank_in_group
        // If i < rank_in_group: connect to i's acceptor
        // (if i > rank_in_group, i will connect to us, handled by accept_loop_)

        // Collection for channels: channels_[peer_rg] = channel
        // First reserve space
        channels_.assign(group_size, nullptr);

        // Connect to lower ranks
        for (int peer = 0; peer < rank_in_group; ++peer) {
            uint16_t peer_port = base_port + (uint16_t)peer;
            std::string peer_ip = (peer < (int)peer_ips.size()) ? peer_ips[peer] : my_ip;
            std::cout << "[FRCheck RDMA] rank=" << rank_in_group
                      << " connecting to peer=" << peer
                      << " at " << peer_ip << ":" << peer_port << std::endl;

            int sock = create_tcp_connect(peer_ip, peer_port);
            std::cout << "[FRCheck RDMA] rank=" << rank_in_group
                      << " connected to peer=" << peer << " fd=" << sock << std::endl;

            // Send my rank to the acceptor before QP exchange
            int my_rank = rank_in_group_;
            if (send(sock, &my_rank, sizeof(my_rank), 0) != (ssize_t)sizeof(my_rank))
                throw std::runtime_error("FRCheck RDMA: failed to send rank to peer=" + std::to_string(peer));

            auto ch = std::make_unique<FRCheckRdmaChannel>(
                rdma_ctx_, rdma_pd_, rdma_send_cq_, rdma_recv_cq_,
                sock, peer, &registered_bufs_, &buf_mtx_);
            ch->exchange_and_connect();
            std::cout << "[FRCheck RDMA] rank=" << rank_in_group
                      << " RDMA connected to peer=" << peer << std::endl;
            channels_[peer] = ch.release(); // owned, will be freed in cleanup
            channel_owners_.push_back(std::unique_ptr<FRCheckRdmaChannel>(channels_[peer]));
        }

        // Wait for higher ranks to connect to us
        // The accept_loop_ collects accepted connections into accepted_queue_
        for (int peer = rank_in_group + 1; peer < group_size; ++peer) {
            std::cout << "[FRCheck RDMA] rank=" << rank_in_group
                      << " waiting for accept from peer=" << peer << std::endl;
            int sock;
            {
                std::unique_lock<std::mutex> lk(accept_mtx_);
                accept_cv_.wait(lk, [this, peer]() {
                    return accepted_queue_.count(peer) > 0 || acceptor_thread_stop_;
                });
                if (acceptor_thread_stop_)
                    throw std::runtime_error("FRCheck: acceptor thread stopped prematurely");
                sock = accepted_queue_[peer];
                accepted_queue_.erase(peer);
            }
            std::cout << "[FRCheck RDMA] rank=" << rank_in_group
                      << " accepted from peer=" << peer << " fd=" << sock << std::endl;

            auto ch = std::make_unique<FRCheckRdmaChannel>(
                rdma_ctx_, rdma_pd_, rdma_send_cq_, rdma_recv_cq_,
                sock, peer, &registered_bufs_, &buf_mtx_);
            ch->exchange_and_connect();
            std::cout << "[FRCheck RDMA] rank=" << rank_in_group
                      << " RDMA connected to peer=" << peer << std::endl;
            channels_[peer] = ch.release();
            channel_owners_.push_back(std::unique_ptr<FRCheckRdmaChannel>(channels_[peer]));
        }

        n_connected_ = group_size_;
        std::cout << "[FRCheck RDMA] rank=" << rank_in_group
                  << " all " << group_size_ << " nodes connected" << std::endl;

        // Init RS encode thread pool
        rs_pool_init();

        // Pre-compile stripe plans
        compile_stripe_plans_();
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
            // ibv_reg_mr can fail for GPU pointers without nvidia-peermem.
            // Don't throw — omit from registered bufs so send/recv will
            // transparently fall back to temp-buffer memcpy.
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

    // ---- Stripe encode ----
    // Called by ALL ranks in the group for the same stripe.
    // Each rank passes its own data buffer and parity output buffer.
    // Internally determines role from pre-compiled StripePlan and acts accordingly.
    void submit_stripe_encode(
        int stripe_id,
        uintptr_t my_data_addr,    // This rank's data (source) or zeros (if IDLE/ENCODER with own data)
        size_t block_size,         // Size per data block
        uintptr_t recv_buf_addr,   // Buffer to receive incoming data (encoder/parity_target)
        size_t recv_buf_size,      // Size of recv buffer
        uintptr_t parity1_out_addr, // Parity1 output (encoder only, kept locally)
        uintptr_t parity2_out_addr, // Parity2 output (encoder only, sent to target)
        uintptr_t parity2_in_addr   // Parity2 receive (parity_target only)
    ) {
        if (stopped_) return;
        if (stripe_id < 0 || stripe_id >= (int)stripe_plans_.size())
            throw std::runtime_error("FRCheck: invalid stripe_id");

        const auto& plan = stripe_plans_[stripe_id];

        switch (plan.role) {
        case StripeRole::SOURCE: {
            // Send my data to encoder
            FRCheckRdmaChannel* ch = get_channel_(plan.encoder_node_id - 1);
            if (!ch) throw std::runtime_error("FRCheck: no channel to encoder");
            ch->send_data((const uint8_t*)my_data_addr, block_size);
            break;
        }
        case StripeRole::ENCODER: {
            // Receive data from all sources
            size_t n_src = plan.source_node_ids.size();
            // Each source places its data sequentially into recv_buf
            for (size_t i = 0; i < n_src; ++i) {
                int src_node = plan.source_node_ids[i] - 1; // 0-based
                FRCheckRdmaChannel* ch = get_channel_(src_node);
                if (!ch) throw std::runtime_error("FRCheck: no channel from source");
                uintptr_t this_recv = recv_buf_addr + i * block_size;
                ch->recv_data((uint8_t*)this_recv, recv_buf_size);
            }

            // RS encode via ISA-L (16-worker pool matching ecnaive xor_pool)
            {
                int k = (int)n_src;
                int m = 2;
                std::vector<unsigned char*> data_ptrs(k);
                for (int i = 0; i < k; ++i)
                    data_ptrs[i] = (unsigned char*)recv_buf_addr + i * block_size;
                unsigned char* parity_ptrs[2] = {
                    (unsigned char*)parity1_out_addr,
                    (unsigned char*)parity2_out_addr,
                };

                if (rs_pool_inited_.load(std::memory_order_acquire)) {
                    RsEncodeJob job;
                    job.len = (int)block_size;
                    job.k = k;
                    job.m = m;
                    job.g_tbls = g_tbls_;
                    job.data_ptrs = data_ptrs.data();
                    job.parity_ptrs = parity_ptrs;
                    rs_pool_run_parallel_encode(job);
                } else {
                    ec_encode_data((int)block_size, k, m,
                                   g_tbls_, data_ptrs.data(), parity_ptrs);
                }
            }

            // Send parity2 to target
            FRCheckRdmaChannel* ch = get_channel_(plan.parity_target_node_id - 1);
            if (!ch) throw std::runtime_error("FRCheck: no channel to parity target");
            ch->send_data((const uint8_t*)parity2_out_addr, block_size);
            break;
        }
        case StripeRole::PARITY_TARGET: {
            // Receive parity2 from encoder
            FRCheckRdmaChannel* ch = get_channel_(plan.encoder_node_id - 1);
            if (!ch) throw std::runtime_error("FRCheck: no channel from encoder");
            ch->recv_data((uint8_t*)parity2_in_addr, block_size);
            break;
        }
        }
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
        // RS matrix: k × m (matches ecnaive)
        a_mat_ = (unsigned char*)malloc((size_t)k * (size_t)m);
        if (!a_mat_) throw std::runtime_error("FRCheck: failed to alloc RS matrix");
        gf_gen_rs_matrix(a_mat_, k, rows);

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
        ec_init_tables(k, rows, a_mat_, g_tbls_);
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
        std::cout << "FRCheck: RS encode pool (" << kRsPoolSize << " workers) initialized" << std::endl;
    }

    void rs_pool_shutdown() {
        if (!rs_pool_inited_.load(std::memory_order_acquire)) return;
        rs_pool_stop_.store(true, std::memory_order_release);
        rs_pool_worker_cv_.notify_all();
        rs_pool_coordinator_cv_.notify_one(); // unblock any stuck rs_pool_run_parallel_encode
        for (int i = 0; i < kRsPoolSize; ++i) pthread_join(rs_pool_threads_[i], nullptr);
        rs_pool_inited_.store(false, std::memory_order_release);
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

    // ---- Stripe async pipeline ----
    struct StripeAsyncState {
        int stripe_id;
        int role;
        int slot;
        int ops_done = 0;
        bool all_done = false;
        // ENCODER:
        int n_src = 0;
        int recvs_done = 0;
        bool encode_queued = false;
        bool encode_done = false;
    };

    static uint64_t wr_id_encode(int stripe_id, int sub) {
        return ((uint64_t)(unsigned)stripe_id << 16) | (uint64_t)(sub & 0xFFFF);
    }
    static int wr_id_stripe(uint64_t id) { return (int)(id >> 16); }
    static int wr_id_sub(uint64_t id) { return (int)(id & 0xFFFF); }

    std::vector<StripeAsyncState> async_stripes_;
    std::thread async_poller_;
    std::atomic<int> async_done_count_{0};
    int async_total_stripes_ = 0;
    bool async_active_ = false;
    std::vector<uintptr_t> async_data_sizes_;

public:
    void submit_stripes_post_recvs(
        int num_stripes,
        const std::vector<int>& roles,
        size_t block_size,
        const std::vector<uintptr_t>& recv_bufs,
        const std::vector<uintptr_t>& parity2_addrs)
    {
        if (async_active_) throw std::runtime_error("FRCheck: async pipeline already active");

        async_stripes_.clear();
        async_stripes_.reserve(num_stripes);
        for (int sid = 0; sid < num_stripes; ++sid) {
            StripeAsyncState st{};
            st.stripe_id = sid;
            st.role = roles[sid];
            st.slot = sid;
            st.ops_done = 0;
            st.all_done = false;
            st.recvs_done = 0;
            st.encode_queued = false;
            st.encode_done = false;
            st.n_src = (st.role == 1) ? (int)stripe_plans_[sid].source_node_ids.size() : 0;
            async_stripes_.push_back(std::move(st));
        }
        async_total_stripes_ = num_stripes;
        async_done_count_ = 0;
        async_active_ = true;

        // Post all recv WRs (before barrier, before any sends)
        for (int sid = 0; sid < num_stripes; ++sid) {
            const auto& plan = stripe_plans_[sid];
            auto& st = async_stripes_[sid];
            if (st.role == (int)StripeRole::ENCODER) {
                for (int i = 0; i < st.n_src; ++i) {
                    int peer = plan.source_node_ids[i] - 1;
                    FRCheckRdmaChannel* ch = get_channel_(peer);
                    uint8_t* dst = (uint8_t*)(recv_bufs[sid] + i * block_size);
                    ch->post_recv(dst, block_size, get_recv_mr(recv_bufs[sid], block_size),
                                  wr_id_encode(sid, i));
                }
            } else if (st.role == (int)StripeRole::PARITY_TARGET) {
                int peer = plan.encoder_node_id - 1;
                FRCheckRdmaChannel* ch = get_channel_(peer);
                ch->post_recv((uint8_t*)parity2_addrs[sid], block_size,
                              get_recv_mr(parity2_addrs[sid], block_size),
                              wr_id_encode(sid, 0));
            }
        }
    }

    void submit_stripes_post_sends(
        const std::vector<uintptr_t>& data_addrs,
        const std::vector<uintptr_t>& actual_sizes,
        size_t block_size,
        const std::vector<uintptr_t>& recv_bufs,
        const std::vector<uintptr_t>& parity1_addrs,
        const std::vector<uintptr_t>& parity2_addrs,
        uintptr_t g_tbls,
        const std::string& output_dir,
        const std::string& layer_name,
        int rank)
    {
        if (!async_active_) throw std::runtime_error("FRCheck: async pipeline not active");
        unsigned char* tbls = reinterpret_cast<unsigned char*>(g_tbls);

        // Store actual sizes for later file write
        async_data_sizes_ = actual_sizes;

        // Post all send WRs (after barrier, recvs already posted on all ranks)
        for (int sid = 0; sid < async_total_stripes_; ++sid) {
            const auto& plan = stripe_plans_[sid];
            auto& st = async_stripes_[sid];
            if (st.role == (int)StripeRole::SOURCE) {
                int peer = plan.encoder_node_id - 1;
                FRCheckRdmaChannel* ch = get_channel_(peer);
                const uint8_t* data = (const uint8_t*)data_addrs[sid];
                ibv_mr* mr = ch->find_mr((uintptr_t)data, block_size);
                if (!mr) {
                    ch->send_data(data, block_size); // fallback sync
                } else {
                    ch->post_send(data, block_size, mr, wr_id_encode(sid, 0));
                }
            }
        }

        // Start poller thread
        std::string layer_dir = output_dir + "/" + layer_name;
        async_poller_ = std::thread([this,
            rbufs=recv_bufs, p1a=parity1_addrs, p2a=parity2_addrs,
            daddrs=data_addrs, block_size, tbls, layer_dir, rank]() {
            async_poller_loop_(async_total_stripes_, tbls, rbufs, p1a, p2a, daddrs,
                               block_size, layer_dir, rank);
        });
    }

    void wait_stripes_async() {
        if (async_poller_.joinable()) async_poller_.join();
        async_active_ = false;
    }

    uintptr_t _get_g_tbls_ptr() const { return (uintptr_t)g_tbls_; }

    void async_poller_loop_(
        int num_stripes, unsigned char* g_tbls,
        std::vector<uintptr_t> recv_bufs,
        std::vector<uintptr_t> parity1_addrs,
        std::vector<uintptr_t> parity2_addrs,
        std::vector<uintptr_t> data_addrs,
        size_t block_size, const std::string& layer_dir, int rank)
    {
        ibv_wc wc_s, wc_r;
        while (async_done_count_ < num_stripes && !stopped_) {
            int ns = ibv_poll_cq(rdma_send_cq_, 1, &wc_s);
            int nr = ibv_poll_cq(rdma_recv_cq_, 1, &wc_r);
            if (ns == 0 && nr == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                continue;
            }
            if (ns < 0 || nr < 0) break;

            if (ns > 0) {
                int sid = wr_id_stripe(wc_s.wr_id);
                auto& st = async_stripes_[sid];
                st.ops_done++;
                check_stripe_done_(sid, parity1_addrs, parity2_addrs, data_addrs, block_size, layer_dir, rank);
            }
            if (nr > 0) {
                int sid = wr_id_stripe(wc_r.wr_id);
                auto& st = async_stripes_[sid];
                if (st.role == (int)StripeRole::ENCODER) {
                    st.recvs_done++;
                    if (st.recvs_done == st.n_src && !st.encode_queued) {
                        st.encode_queued = true;
                        int sidx = st.stripe_id;
                        RsEncodeJob job;
                        job.len = (int)block_size;
                        job.k = st.n_src;
                        job.m = 2;
                        job.g_tbls = g_tbls;
                        std::vector<unsigned char*> dp(job.k);
                        for (int i = 0; i < job.k; ++i)
                            dp[i] = (unsigned char*)(recv_bufs[sidx] + i * block_size);
                        job.data_ptrs = dp.data();
                        unsigned char* pptr[2] = {
                            (unsigned char*)parity1_addrs[sidx],
                            (unsigned char*)parity2_addrs[sidx] };
                        job.parity_ptrs = pptr;
                        rs_pool_run_parallel_encode(job);
                        st.encode_done = true;

                        const auto& plan = stripe_plans_[sidx];
                        int peer = plan.parity_target_node_id - 1;
                        FRCheckRdmaChannel* ch = get_channel_(peer);
                        ibv_mr* mr = ch->find_mr(parity2_addrs[sidx], block_size);
                        ch->post_send((uint8_t*)parity2_addrs[sidx], block_size, mr,
                                      wr_id_encode(sidx, st.n_src));
                    }
                }
                st.ops_done++;
                check_stripe_done_(sid, parity1_addrs, parity2_addrs, data_addrs, block_size, layer_dir, rank);
            }
        }
    }

    void check_stripe_done_(int sid,
                            const std::vector<uintptr_t>& parity1_addrs,
                            const std::vector<uintptr_t>& parity2_addrs,
                            const std::vector<uintptr_t>& data_addrs,
                            size_t blk, const std::string& layer_dir, int rank) {
        auto& st = async_stripes_[sid];
        if (st.all_done) return;
        int need = 1;
        if (st.role == (int)StripeRole::ENCODER)
            need = st.n_src + 1;
        if (st.ops_done >= need && (!(st.role == (int)StripeRole::ENCODER) || st.encode_done)) {
            if (st.role == (int)StripeRole::SOURCE) {
                // SOURCE write handled in Python (GPU buffers not fwrite-safe)
            } else if (st.role == (int)StripeRole::ENCODER) {
                write_stripe_file(layer_dir, sid, 1, rank,
                    (const uint8_t*)parity1_addrs[sid], blk, blk, "_p1");
                write_stripe_file(layer_dir, sid, 2, rank,
                    (const uint8_t*)parity2_addrs[sid], blk, blk, "_p2");
            } else if (st.role == (int)StripeRole::PARITY_TARGET) {
                write_stripe_file(layer_dir, sid, st.role, rank,
                    (const uint8_t*)parity2_addrs[sid], blk, blk);
            }
            st.all_done = true;
            async_done_count_++;
        }
    }

    ibv_mr* get_recv_mr(uintptr_t addr, size_t size) {
        for (auto* ch : channels_) {
            if (ch) {
                ibv_mr* mr = ch->find_mr(addr, size);
                if (mr) return mr;
            }
        }
        throw std::runtime_error("FRCheck async: recv buffer not registered");
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

        rdma_send_cq_ = ibv_create_cq(rdma_ctx_, FRCHECK_MAX_WR * 2, nullptr, nullptr, 0);
        rdma_recv_cq_ = ibv_create_cq(rdma_ctx_, FRCHECK_MAX_WR * 2, nullptr, nullptr, 0);
        if (!rdma_send_cq_ || !rdma_recv_cq_)
            throw std::runtime_error("FRCheck RDMA: failed to create CQs");
    }

    void cleanup_rdma_() {
        rs_pool_shutdown();
        channel_owners_.clear();
        channels_.clear();

        if (acceptor_thread_stop_.load()) {
            // already stopped
        }
        acceptor_thread_stop_ = true;
        if (acceptor_fd_ >= 0) {
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

        if (rdma_recv_cq_) { ibv_destroy_cq(rdma_recv_cq_); rdma_recv_cq_ = nullptr; }
        if (rdma_send_cq_) { ibv_destroy_cq(rdma_send_cq_); rdma_send_cq_ = nullptr; }
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
                struct sockaddr_in peer_addr;
                socklen_t addrlen = sizeof(peer_addr);
                int fd = accept(acceptor_fd_,
                                (struct sockaddr*)&peer_addr, &addrlen);
                if (fd < 0) {
                    if (errno == EINVAL || errno == EBADF) break; // closed
                    continue;
                }
                // Receive peer rank identifier
                int peer_rank = -1;
                ssize_t nr = recv(fd, &peer_rank, sizeof(peer_rank), MSG_WAITALL);
                if (nr != sizeof(peer_rank)) {
                    std::cerr << "[FRCheck RDMA] accept: failed to recv peer rank" << std::endl;
                    close(fd);
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lk(accept_mtx_);
                    accepted_queue_[peer_rank] = fd;
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
    FRCheckRdmaChannel* get_channel_(int peer_rg) {
        if (peer_rg < 0 || peer_rg >= group_size_ || peer_rg == rank_in_group_)
            return nullptr;
        return channels_[peer_rg];
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
    std::atomic<bool> stopped_;

    // RDMA
    ibv_context* rdma_ctx_;
    ibv_pd* rdma_pd_;
    ibv_cq* rdma_send_cq_ = nullptr;
    ibv_cq* rdma_recv_cq_ = nullptr;

    // Topology
    int group_size_ = 0;
    int rank_in_group_ = -1;
    int n_connected_ = 0;
    std::string my_ip_;

    // Channels: index by peer rank_in_group (0..group_size-1), null for self
    std::vector<FRCheckRdmaChannel*> channels_;
    std::vector<std::unique_ptr<FRCheckRdmaChannel>> channel_owners_;

    // Registered buffers (shared across channels)
    std::map<uintptr_t, RdmaBuffer> registered_bufs_;
    std::mutex buf_mtx_;

    // TCP acceptor (raw socket)
    int acceptor_fd_ = -1;
    std::thread accept_thread_;
    std::atomic<bool> acceptor_thread_stop_{false};
    std::map<int, int> accepted_queue_; // peer_rank -> fd
    std::mutex accept_mtx_;
    std::condition_variable accept_cv_;

    // Stripe plans (pre-compiled)
    std::vector<StripePlan> stripe_plans_;

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

        // Stripe encode
        .def("submit_stripe_encode", &FRCheckNative::submit_stripe_encode,
             py::arg("stripe_id"),
             py::arg("my_data_addr"),
             py::arg("block_size"),
             py::arg("recv_buf_addr"),
             py::arg("recv_buf_size"),
             py::arg("parity1_out_addr"),
             py::arg("parity2_out_addr"),
             py::arg("parity2_in_addr"))

        // StripePlan queries
        .def("get_role_for_stripe", &FRCheckNative::get_role_for_stripe,
             py::arg("stripe_id"))
        .def("get_source_node_ids", &FRCheckNative::get_source_node_ids,
             py::arg("stripe_id"))
        .def("get_encoder_node_id", &FRCheckNative::get_encoder_node_id,
             py::arg("stripe_id"))
        .def("get_parity_target_node_id", &FRCheckNative::get_parity_target_node_id,
             py::arg("stripe_id"))

        // Async stripe pipeline (three-phase: recvs → barrier → sends+poller → wait)
        .def("submit_stripes_post_recvs", &FRCheckNative::submit_stripes_post_recvs,
             py::arg("num_stripes"), py::arg("roles"),
             py::arg("block_size"), py::arg("recv_bufs"), py::arg("parity2_addrs"))
        .def("submit_stripes_post_sends", &FRCheckNative::submit_stripes_post_sends,
             py::arg("data_addrs"), py::arg("actual_sizes"), py::arg("block_size"),
             py::arg("recv_bufs"), py::arg("parity1_addrs"), py::arg("parity2_addrs"),
             py::arg("g_tbls"),
             py::arg("output_dir"), py::arg("layer_name"), py::arg("rank"))
        .def("wait_stripes_async", &FRCheckNative::wait_stripes_async)
        .def("_get_g_tbls_ptr", &FRCheckNative::_get_g_tbls_ptr);
}
