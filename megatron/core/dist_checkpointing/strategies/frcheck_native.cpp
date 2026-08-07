#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
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
#include <limits>
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
static constexpr int kRsMaxDataBlocks = 254;
static constexpr int kRsMaxParityBlocks = 2;
static constexpr size_t kRsInitialPlanSpans = 4096;
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

struct RsPoolSpan {
    uint32_t job_index = 0;
    int offset = 0;
    int length = 0;
};

namespace py = pybind11;

static inline uint64_t frcheck_now_us() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Each Megatron rank owns one GPU; LOCAL_RANK selects the device in-process.
static int resolve_cuda_device() {
    if (const char* local_rank = std::getenv("LOCAL_RANK")) {
        return std::max(0, std::atoi(local_rank));
    }
    return 0;
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
static constexpr size_t FRCHECK_DEFAULT_RDMA_CHUNK = 64ULL * 1024 * 1024;

static size_t frcheck_rdma_chunk_size() {
    static const size_t chunk = []() -> size_t {
        const char* env = std::getenv("FRCHECK_RDMA_CHUNK_MB");
        if (!env || !*env) return FRCHECK_DEFAULT_RDMA_CHUNK;
        char* end = nullptr;
        errno = 0;
        unsigned long long value = std::strtoull(env, &end, 10);
        if (errno != 0 || end == env || *end != '\0' || value == 0 || value > 4095)
            throw std::runtime_error("FRCheck RDMA: invalid FRCHECK_RDMA_CHUNK_MB");
        return static_cast<size_t>(value) * 1024ULL * 1024ULL;
    }();
    return chunk;
}
static constexpr int    FRCHECK_MAX_WR = 64;
static constexpr int    FRCHECK_DEFAULT_MAX_SEND_SGE = 16;

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
          max_send_sge_(1), max_recv_sge_(1), segment_sge_limit_(1),
          segment_chunk_size_(frcheck_rdma_chunk_size()), connected_(false)
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
        int requested_max_send_sge = FRCHECK_DEFAULT_MAX_SEND_SGE;
        if (const char* env = std::getenv("FRCHECK_MAX_SEND_SGE")) {
            try {
                requested_max_send_sge = std::max(1, std::stoi(env));
            } catch (...) {
                throw std::runtime_error("FRCheck RDMA: invalid FRCHECK_MAX_SEND_SGE");
            }
        }
        ibv_device_attr device_attr{};
        if (ibv_query_device(ctx_, &device_attr) == 0 && device_attr.max_sge > 0)
            requested_max_send_sge = std::min(requested_max_send_sge, device_attr.max_sge);
        attr.cap.max_send_sge = requested_max_send_sge;
        attr.cap.max_recv_sge = requested_max_send_sge;
        qp_ = ibv_create_qp(pd_, &attr);
        if (!qp_)
            throw std::runtime_error("FRCheck RDMA: failed to create QP");
        max_send_sge_ = std::max(1, std::min(requested_max_send_sge,
                                             static_cast<int>(attr.cap.max_send_sge)));
        max_recv_sge_ = std::max(1, std::min(requested_max_send_sge,
                                             static_cast<int>(attr.cap.max_recv_sge)));
    }

    ~FRCheckRdmaChannel() {
        stop_tag_receiver();
        if (qp_) ibv_destroy_qp(qp_);
        if (recv_cq_) { ibv_destroy_cq(recv_cq_); recv_cq_ = nullptr; }
        if (send_cq_) { ibv_destroy_cq(send_cq_); send_cq_ = nullptr; }
        if (tcp_sock_ >= 0) close(tcp_sock_);
    }

    int peer_rank() const { return peer_rank_; }
    bool is_connected() const { return connected_; }
    int max_send_sge() const { return max_send_sge_; }
    int max_recv_sge() const { return max_recv_sge_; }

    // Break blocking send/recv during shutdown (ECLATIN-style conn cleanup).
    void abort_connection() {
        connected_ = false;
        tag_stop_.store(true, std::memory_order_release);
        tag_cv_.notify_all();
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

        uint64_t local_segment_caps[2] = {
            htobe64(static_cast<uint64_t>(std::min(max_send_sge_, max_recv_sge_))),
            htobe64(static_cast<uint64_t>(frcheck_rdma_chunk_size())),
        };
        uint64_t remote_segment_caps[2]{};
        if (send(tcp_sock_, local_segment_caps, sizeof(local_segment_caps), 0) !=
                static_cast<ssize_t>(sizeof(local_segment_caps)))
            throw std::runtime_error("FRCheck RDMA: failed to send segment capabilities");
        if (recv(tcp_sock_, remote_segment_caps, sizeof(remote_segment_caps), MSG_WAITALL) !=
                static_cast<ssize_t>(sizeof(remote_segment_caps)))
            throw std::runtime_error("FRCheck RDMA: failed to recv segment capabilities");
        const uint64_t remote_sge = be64toh(remote_segment_caps[0]);
        const uint64_t remote_chunk = be64toh(remote_segment_caps[1]);
        if (remote_sge == 0 || remote_chunk == 0)
            throw std::runtime_error("FRCheck RDMA: invalid peer segment capabilities");
        segment_sge_limit_ = std::max(
            1, std::min(std::min(max_send_sge_, max_recv_sge_),
                        static_cast<int>(std::min<uint64_t>(
                            remote_sge, static_cast<uint64_t>(std::numeric_limits<int>::max())))));
        segment_chunk_size_ = std::min(
            frcheck_rdma_chunk_size(), static_cast<size_t>(remote_chunk));
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


    void send_segments(const std::vector<std::pair<uintptr_t, size_t>>& segments,
                       size_t logical_size,
                       const std::function<void()>& wait_cb = nullptr,
                       const std::function<void()>& done_cb = nullptr) {
        std::lock_guard<std::mutex> lock(send_mtx_);
        if (!connected_)
            throw std::runtime_error("FRCheck RDMA: channel not connected");
        size_t segment_total = 0;
        for (const auto& segment : segments) segment_total += segment.second;
        if (segment_total != logical_size)
            throw std::runtime_error("FRCheck RDMA: scatter logical size mismatch");
        if (wait_cb) wait_cb();
        uint64_t net_sz = htobe64(logical_size);
        if (send(tcp_sock_, &net_sz, sizeof(net_sz), 0) != sizeof(net_sz))
            throw std::runtime_error("FRCheck RDMA: failed to send scatter size");
        uint8_t ack;
        if (recv(tcp_sock_, &ack, 1, MSG_WAITALL) != 1)
            throw std::runtime_error("FRCheck RDMA: failed to recv scatter ack");
        if (done_cb) done_cb();
        send_segments_chunked_(segments, logical_size, wait_cb, done_cb);
    }

    size_t recv_segments(
            const std::vector<std::pair<uintptr_t, size_t>>& segments,
            size_t logical_size,
            const std::function<void()>& wait_cb = nullptr,
            const std::function<void()>& done_cb = nullptr) {
        std::lock_guard<std::mutex> lock(recv_mtx_);
        if (!connected_)
            throw std::runtime_error("FRCheck RDMA: channel not connected");
        validate_segments_(segments, logical_size, "receive");
        if (wait_cb) wait_cb();
        uint64_t net_sz = 0;
        if (recv(tcp_sock_, &net_sz, sizeof(net_sz), MSG_WAITALL) != sizeof(net_sz))
            throw std::runtime_error("FRCheck RDMA: failed to recv scatter size");
        const size_t size = be64toh(net_sz);
        if (size != logical_size)
            throw std::runtime_error("FRCheck RDMA: scatter receive size mismatch");
        uint8_t ack = 1;
        if (send(tcp_sock_, &ack, 1, 0) != 1)
            throw std::runtime_error("FRCheck RDMA: failed to send scatter ack");
        if (done_cb) done_cb();
        recv_segments_chunked_(segments, logical_size, wait_cb, done_cb);
        return size;
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

    void send_segments_chunked_(
            const std::vector<std::pair<uintptr_t, size_t>>& segments,
            size_t total,
            const std::function<void()>& wait_cb,
            const std::function<void()>& done_cb) {
        size_t segment_idx = 0;
        size_t segment_offset = 0;
        size_t remaining = total;
        while (remaining > 0) {
            if (wait_cb) wait_cb();
            size_t message_remaining = std::min(remaining, segment_chunk_size_);
            std::vector<ibv_sge> sges;
            while (message_remaining > 0 &&
                   sges.size() < static_cast<size_t>(segment_sge_limit_)) {
                while (segment_idx < segments.size() &&
                       segment_offset == segments[segment_idx].second) {
                    ++segment_idx;
                    segment_offset = 0;
                }
                if (segment_idx >= segments.size())
                    throw std::runtime_error("FRCheck RDMA: scatter segments ended early");
                const auto& segment = segments[segment_idx];
                size_t length = std::min(message_remaining, segment.second - segment_offset);
                ibv_mr* mr = find_mr(segment.first + segment_offset, length);
                if (!mr)
                    throw std::runtime_error(
                        "FRCheck RDMA: unregistered scatter segment addr=" +
                        std::to_string(segment.first + segment_offset) +
                        " size=" + std::to_string(length));
                ibv_sge sge{};
                sge.addr = static_cast<uint64_t>(segment.first + segment_offset);
                sge.length = static_cast<uint32_t>(length);
                sge.lkey = mr->lkey;
                sges.push_back(sge);
                segment_offset += length;
                message_remaining -= length;
                remaining -= length;
            }
            ibv_send_wr wr{};
            wr.sg_list = sges.data();
            wr.num_sge = static_cast<int>(sges.size());
            wr.opcode = IBV_WR_SEND;
            wr.send_flags = IBV_SEND_SIGNALED;
            ibv_send_wr* bad = nullptr;
            if (ibv_post_send(qp_, &wr, &bad))
                throw std::runtime_error("FRCheck RDMA: scatter post_send failed");
            poll_cq(send_cq_, 1);
            if (done_cb) done_cb();
        }
    }

    void validate_segments_(
            const std::vector<std::pair<uintptr_t, size_t>>& segments,
            size_t logical_size,
            const char* operation) {
        size_t total = 0;
        for (const auto& segment : segments) {
            if (segment.first == 0 || segment.second == 0)
                throw std::runtime_error(std::string("FRCheck RDMA: empty ") + operation + " segment");
            if (segment.second > std::numeric_limits<size_t>::max() - total)
                throw std::runtime_error("FRCheck RDMA: segment size overflow");
            total += segment.second;
            if (!find_mr(segment.first, segment.second))
                throw std::runtime_error(
                    std::string("FRCheck RDMA: unregistered ") + operation +
                    " segment addr=" + std::to_string(segment.first) +
                    " size=" + std::to_string(segment.second));
        }
        if (segments.empty() || total != logical_size)
            throw std::runtime_error(std::string("FRCheck RDMA: ") + operation +
                                     " segment logical size mismatch");
    }

    void recv_segments_chunked_(
            const std::vector<std::pair<uintptr_t, size_t>>& segments,
            size_t total,
            const std::function<void()>& wait_cb,
            const std::function<void()>& done_cb) {
        size_t segment_idx = 0;
        size_t segment_offset = 0;
        size_t remaining = total;
        while (remaining > 0) {
            if (wait_cb) wait_cb();
            size_t message_remaining = std::min(remaining, segment_chunk_size_);
            std::vector<ibv_sge> sges;
            while (message_remaining > 0 &&
                   sges.size() < static_cast<size_t>(segment_sge_limit_)) {
                while (segment_idx < segments.size() &&
                       segment_offset == segments[segment_idx].second) {
                    ++segment_idx;
                    segment_offset = 0;
                }
                if (segment_idx >= segments.size())
                    throw std::runtime_error("FRCheck RDMA: receive segments ended early");
                const auto& segment = segments[segment_idx];
                const size_t length = std::min(message_remaining,
                                               segment.second - segment_offset);
                ibv_mr* mr = find_mr(segment.first + segment_offset, length);
                if (!mr)
                    throw std::runtime_error("FRCheck RDMA: receive segment MR disappeared");
                ibv_sge sge{};
                sge.addr = static_cast<uint64_t>(segment.first + segment_offset);
                sge.length = static_cast<uint32_t>(length);
                sge.lkey = mr->lkey;
                sges.push_back(sge);
                segment_offset += length;
                message_remaining -= length;
                remaining -= length;
            }
            ibv_recv_wr wr{};
            wr.sg_list = sges.data();
            wr.num_sge = static_cast<int>(sges.size());
            ibv_recv_wr* bad = nullptr;
            if (ibv_post_recv(qp_, &wr, &bad))
                throw std::runtime_error("FRCheck RDMA: scatter post_recv failed");
            poll_cq(recv_cq_, 1);
            if (done_cb) done_cb();
        }
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
            size_t nchunks = (std::min(remaining, frcheck_rdma_chunk_size()) + frcheck_rdma_chunk_size() - 1) / frcheck_rdma_chunk_size();
            std::vector<ibv_sge> sge(nchunks);
            std::vector<ibv_send_wr> wr(nchunks);
            for (size_t i = 0; i < nchunks; ++i) {
                size_t cur = std::min(frcheck_rdma_chunk_size(), remaining);
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

    int post_recv_chunked_(uint8_t* buf, size_t total, ibv_mr* mr) {
        size_t remaining = total, offset = 0;
        int posted = 0;
        while (remaining > 0) {
            size_t group_bytes = std::min(remaining, frcheck_rdma_chunk_size());
            size_t nchunks = (group_bytes + frcheck_rdma_chunk_size() - 1) /
                             frcheck_rdma_chunk_size();
            std::vector<ibv_sge> sge(nchunks);
            std::vector<ibv_recv_wr> wr(nchunks);
            for (size_t i = 0; i < nchunks; ++i) {
                size_t cur = std::min(frcheck_rdma_chunk_size(), remaining);
                sge[i].addr = reinterpret_cast<uint64_t>(buf + offset);
                sge[i].length = static_cast<uint32_t>(cur);
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
                throw std::runtime_error("FRCheck RDMA: prepost recv failed");
            posted += static_cast<int>(nchunks);
        }
        return posted;
    }

    void _recv_chunked(uint8_t* buf, size_t total, ibv_mr* mr,
                       const std::function<void()>& wait_cb,
                       const std::function<void()>& done_cb) {
        size_t remaining = total, offset = 0;
        while (remaining > 0) {
            if (wait_cb) wait_cb();
            size_t nchunks = (std::min(remaining, frcheck_rdma_chunk_size()) + frcheck_rdma_chunk_size() - 1) / frcheck_rdma_chunk_size();
            std::vector<ibv_sge> sge(nchunks);
            std::vector<ibv_recv_wr> wr(nchunks);
            for (size_t i = 0; i < nchunks; ++i) {
                size_t cur = std::min(frcheck_rdma_chunk_size(), remaining);
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
    int max_send_sge_;
    int max_recv_sge_;
    int segment_sge_limit_;
    size_t segment_chunk_size_;
    bool connected_;
    std::mutex send_mtx_;
    std::mutex recv_mtx_;

    // ---- Tagged multiplexing (shared-lane mode) ----
    // A shared lane is bidirectional and carries interleaved messages for many
    // stripes.  A single dedicated reader thread owns ALL TCP control bytes and
    // demultiplexes two control record types:
    //   DATA: cache the incoming payload size for `tag`; recv_tagged() sends the
    //         ACK only after its RDMA receive buffer is ready.
    //   ACK : the peer accepted our earlier DATA for `tag`; wake the sender so
    //         it can RDMA send the payload.
    // Senders never read the socket directly (that would steal ACK/DATA bytes);
    // they wait for the reader thread to deliver the ACK.
    struct TaggedCtl { uint32_t type; uint32_t pad; uint64_t tag; uint64_t size; };
    static constexpr uint32_t kTagData = 0;
    static constexpr uint32_t kTagAck  = 1;

    std::mutex tag_mtx_;
    std::condition_variable tag_cv_;
    std::map<uint64_t, size_t> pending_data_sizes_;
    struct PreparedTaggedRecv {
        uint8_t* buffer = nullptr;
        size_t capacity = 0;
        size_t size = 0;
        int completions = 0;
        bool posted = false;
        std::string error;
    };
    std::map<uint64_t, bool> ack_ready_;
    std::map<uint64_t, PreparedTaggedRecv> prepared_tagged_recvs_;
    std::atomic<uint64_t> tagged_ack_wait_total_us_{0};
    std::atomic<uint64_t> tagged_ack_wait_max_us_{0};
    std::atomic<uint64_t> tagged_ack_wait_count_{0};
    std::mutex sock_write_mtx_;
    std::thread tag_receiver_thread_;
    std::atomic<bool> tag_stop_{false};
    bool tag_mode_ = false;

    bool read_full_(void* p, size_t n) {
        uint8_t* c = (uint8_t*)p;
        size_t got = 0;
        while (got < n) {
            ssize_t r = ::recv(tcp_sock_, c + got, n - got, MSG_WAITALL);
            if (r <= 0) return false;
            got += (size_t)r;
        }
        return true;
    }

    bool write_full_locked_(const void* p, size_t n) {
        std::lock_guard<std::mutex> lk(sock_write_mtx_);
        const uint8_t* c = (const uint8_t*)p;
        size_t sent = 0;
        while (sent < n) {
            ssize_t w = ::send(tcp_sock_, c + sent, n - sent, 0);
            if (w <= 0) return false;
            sent += (size_t)w;
        }
        return true;
    }

    void tag_receiver_loop_() {
        while (!tag_stop_.load(std::memory_order_acquire)) {
            TaggedCtl c;
            if (!read_full_(&c, sizeof(c))) break;
            uint32_t type = be32toh(c.type);
            uint64_t tag = be64toh(c.tag);
            size_t size = (size_t)be64toh(c.size);

            if (type == kTagAck) {
                std::lock_guard<std::mutex> lk(tag_mtx_);
                ack_ready_[tag] = true;
                tag_cv_.notify_all();
                continue;
            }

            bool prepared = false;
            {
                std::lock_guard<std::mutex> lk(tag_mtx_);
                prepared = prepared_tagged_recvs_.count(tag) != 0;
                if (!prepared) pending_data_sizes_[tag] = size;
            }
            if (!prepared) {
                tag_cv_.notify_all();
                continue;
            }

            std::string error;
            int completions = 0;
            {
                std::lock_guard<std::mutex> recv_lock(recv_mtx_);
                uint8_t* buffer = nullptr;
                size_t capacity = 0;
                {
                    std::lock_guard<std::mutex> lk(tag_mtx_);
                    auto& target = prepared_tagged_recvs_.at(tag);
                    buffer = target.buffer;
                    capacity = target.capacity;
                }
                try {
                    if (size > capacity)
                        throw std::runtime_error("prepared tagged receive exceeds capacity");
                    ibv_mr* mr = find_mr(reinterpret_cast<uintptr_t>(buffer), size);
                    if (!mr) throw std::runtime_error("unregistered prepared tagged receive buffer");
                    completions = post_recv_chunked_(buffer, size, mr);
                } catch (const std::exception& e) {
                    error = e.what();
                }
            }
            {
                std::lock_guard<std::mutex> lk(tag_mtx_);
                auto& target = prepared_tagged_recvs_.at(tag);
                target.size = size;
                target.completions = completions;
                target.error = error;
                target.posted = true;
            }
            tag_cv_.notify_all();
            if (!error.empty()) {
                tag_stop_.store(true, std::memory_order_release);
                tag_cv_.notify_all();
                break;
            }
            TaggedCtl ack{htobe32(kTagAck), 0, htobe64(tag), 0};
            if (!write_full_locked_(&ack, sizeof(ack))) {
                tag_stop_.store(true, std::memory_order_release);
                tag_cv_.notify_all();
                break;
            }
        }
        tag_stop_.store(true, std::memory_order_release);
        tag_cv_.notify_all();
    }

public:
    void start_tag_receiver() {
        if (tag_mode_) return;
        tag_mode_ = true;
        tag_stop_.store(false, std::memory_order_release);
        tag_receiver_thread_ = std::thread(&FRCheckRdmaChannel::tag_receiver_loop_, this);
    }

    void stop_tag_receiver() {
        if (!tag_mode_) return;
        tag_stop_.store(true, std::memory_order_release);
        tag_cv_.notify_all();
        if (tcp_sock_ >= 0) shutdown(tcp_sock_, SHUT_RDWR);
        if (tag_receiver_thread_.joinable()) tag_receiver_thread_.join();
        tag_mode_ = false;
    }

    // Serialized tagged send: one message in flight per channel.  Writes a DATA
    // control record, waits (via the reader thread) for the peer ACK, then
    // RDMA sends the payload.
    void send_tagged(uint64_t tag, const uint8_t* data, size_t size,
                     const std::function<void()>& wait_cb = nullptr,
                     const std::function<void()>& done_cb = nullptr) {
        std::lock_guard<std::mutex> lock(send_mtx_);
        if (!connected_)
            throw std::runtime_error("FRCheck RDMA: channel not connected");
        if (wait_cb) wait_cb();
        {
            std::lock_guard<std::mutex> lk(tag_mtx_);
            ack_ready_[tag] = false;
        }
        TaggedCtl c{htobe32(kTagData), 0, htobe64(tag), htobe64(size)};
        if (!write_full_locked_(&c, sizeof(c)))
            throw std::runtime_error("FRCheck RDMA: failed to send tagged header");
        const uint64_t ack_wait_t0 = frcheck_now_us();
        {
            std::unique_lock<std::mutex> lk(tag_mtx_);
            tag_cv_.wait(lk, [&]{
                auto it = ack_ready_.find(tag);
                return (it != ack_ready_.end() && it->second) ||
                       tag_stop_.load(std::memory_order_acquire);
            });
            ack_ready_.erase(tag);
            if (tag_stop_.load(std::memory_order_acquire)) {
                if (done_cb) done_cb();
                return;
            }
        }
        const uint64_t ack_wait_us = frcheck_now_us() - ack_wait_t0;
        tagged_ack_wait_total_us_.fetch_add(ack_wait_us, std::memory_order_relaxed);
        tagged_ack_wait_count_.fetch_add(1, std::memory_order_relaxed);
        uint64_t ack_max = tagged_ack_wait_max_us_.load(std::memory_order_relaxed);
        while (ack_max < ack_wait_us &&
               !tagged_ack_wait_max_us_.compare_exchange_weak(
                   ack_max, ack_wait_us, std::memory_order_relaxed)) {}
        if (done_cb) done_cb();
        ibv_mr* mr = find_mr((uintptr_t)data, size);
        if (!mr)
            throw std::runtime_error("FRCheck RDMA: unregistered tagged send buffer");
        send_chunked(data, size, mr, wait_cb, done_cb);
    }

    void send_tagged_segments(
            uint64_t tag,
            const std::vector<std::pair<uintptr_t, size_t>>& segments,
            size_t logical_size,
            const std::function<void()>& wait_cb = nullptr,
            const std::function<void()>& done_cb = nullptr) {
        std::lock_guard<std::mutex> lock(send_mtx_);
        if (!connected_)
            throw std::runtime_error("FRCheck RDMA: channel not connected");
        size_t segment_total = 0;
        for (const auto& segment : segments) segment_total += segment.second;
        if (segment_total != logical_size)
            throw std::runtime_error("FRCheck RDMA: tagged scatter logical size mismatch");
        if (wait_cb) wait_cb();
        {
            std::lock_guard<std::mutex> lk(tag_mtx_);
            ack_ready_[tag] = false;
        }
        TaggedCtl c{htobe32(kTagData), 0, htobe64(tag), htobe64(logical_size)};
        if (!write_full_locked_(&c, sizeof(c)))
            throw std::runtime_error("FRCheck RDMA: failed to send tagged scatter header");
        const uint64_t ack_wait_t0 = frcheck_now_us();
        {
            std::unique_lock<std::mutex> lk(tag_mtx_);
            tag_cv_.wait(lk, [&]{
                auto it = ack_ready_.find(tag);
                return (it != ack_ready_.end() && it->second) ||
                       tag_stop_.load(std::memory_order_acquire);
            });
            ack_ready_.erase(tag);
            if (tag_stop_.load(std::memory_order_acquire)) {
                if (done_cb) done_cb();
                return;
            }
        }
        const uint64_t ack_wait_us = frcheck_now_us() - ack_wait_t0;
        tagged_ack_wait_total_us_.fetch_add(ack_wait_us, std::memory_order_relaxed);
        tagged_ack_wait_count_.fetch_add(1, std::memory_order_relaxed);
        uint64_t ack_max = tagged_ack_wait_max_us_.load(std::memory_order_relaxed);
        while (ack_max < ack_wait_us &&
               !tagged_ack_wait_max_us_.compare_exchange_weak(
                   ack_max, ack_wait_us, std::memory_order_relaxed)) {}
        if (done_cb) done_cb();
        send_segments_chunked_(segments, logical_size, wait_cb, done_cb);
    }

    void prepare_tagged_recv(uint64_t tag, uint8_t* buf, size_t capacity) {
        if (!buf || capacity == 0)
            throw std::runtime_error("FRCheck prepared tagged receive: empty target");
        if (!find_mr(reinterpret_cast<uintptr_t>(buf), capacity))
            throw std::runtime_error("FRCheck prepared tagged receive: unregistered target");
        std::lock_guard<std::mutex> lk(tag_mtx_);
        if (prepared_tagged_recvs_.count(tag) || pending_data_sizes_.count(tag))
            throw std::runtime_error("FRCheck prepared tagged receive: duplicate or late tag");
        PreparedTaggedRecv target;
        target.buffer = buf;
        target.capacity = capacity;
        prepared_tagged_recvs_.emplace(tag, std::move(target));
    }

    size_t wait_prepared_tagged_recv(uint64_t tag) {
        int completions = 0;
        size_t size = 0;
        std::string error;
        {
            std::unique_lock<std::mutex> lk(tag_mtx_);
            tag_cv_.wait(lk, [&] {
                auto it = prepared_tagged_recvs_.find(tag);
                return (it != prepared_tagged_recvs_.end() && it->second.posted) ||
                       tag_stop_.load(std::memory_order_acquire);
            });
            auto it = prepared_tagged_recvs_.find(tag);
            if (it == prepared_tagged_recvs_.end())
                throw std::runtime_error("FRCheck prepared tagged receive: unknown tag");
            completions = it->second.completions;
            size = it->second.size;
            error = it->second.error;
        }
        if (!error.empty())
            throw std::runtime_error("FRCheck prepared tagged receive: " + error);
        if (tag_stop_.load(std::memory_order_acquire) && completions == 0)
            throw std::runtime_error("FRCheck prepared tagged receive: channel stopped");
        {
            std::lock_guard<std::mutex> recv_lock(recv_mtx_);
            poll_cq(recv_cq_, completions);
        }
        {
            std::lock_guard<std::mutex> lk(tag_mtx_);
            prepared_tagged_recvs_.erase(tag);
        }
        return size;
    }

    // Register a tagged recv target and block until its DATA header arrives.
    // The ACK is sent from this thread after the buffer/MR is known, then the
    // matching RDMA recv is posted.  The TCP reader remains free to process
    // unrelated tags on the same shared lane.
    size_t recv_tagged(uint64_t tag, uint8_t* buf, size_t buf_size,
                       const std::function<void()>& wait_cb = nullptr,
                       const std::function<void()>& done_cb = nullptr) {
        if (wait_cb) wait_cb();
        size_t size = 0;
        {
            std::unique_lock<std::mutex> lk(tag_mtx_);
            tag_cv_.wait(lk, [&]{
                return pending_data_sizes_.count(tag) ||
                       tag_stop_.load(std::memory_order_acquire);
            });
            if (tag_stop_.load(std::memory_order_acquire)) {
                if (done_cb) done_cb();
                return 0;
            }
            size = pending_data_sizes_[tag];
            pending_data_sizes_.erase(tag);
        }
        if (size > buf_size) {
            throw std::runtime_error("FRCheck tagged recv: size " + std::to_string(size) +
                                     " exceeds buffer " + std::to_string(buf_size) +
                                     " (tag=" + std::to_string(tag) + ")");
        }
        ibv_mr* mr = find_mr((uintptr_t)buf, size);
        if (!mr) {
            throw std::runtime_error("FRCheck RDMA: unregistered tagged recv buffer");
        }
        std::lock_guard<std::mutex> lock(recv_mtx_);
        TaggedCtl ack{htobe32(kTagAck), 0, htobe64(tag), 0};
        if (!write_full_locked_(&ack, sizeof(ack))) {
            throw std::runtime_error("FRCheck RDMA: failed to send tagged ack");
        }
        recv_chunked(buf, size, mr, wait_cb, done_cb);
        return size;
    }

    size_t recv_tagged_segments(
            uint64_t tag,
            const std::vector<std::pair<uintptr_t, size_t>>& segments,
            size_t logical_size,
            const std::function<void()>& wait_cb = nullptr,
            const std::function<void()>& done_cb = nullptr) {
        validate_segments_(segments, logical_size, "tagged receive");
        if (wait_cb) wait_cb();
        size_t size = 0;
        {
            std::unique_lock<std::mutex> lk(tag_mtx_);
            tag_cv_.wait(lk, [&] {
                return pending_data_sizes_.count(tag) ||
                       tag_stop_.load(std::memory_order_acquire);
            });
            if (tag_stop_.load(std::memory_order_acquire)) {
                if (done_cb) done_cb();
                return 0;
            }
            size = pending_data_sizes_[tag];
            pending_data_sizes_.erase(tag);
        }
        if (size != logical_size)
            throw std::runtime_error("FRCheck tagged scatter receive size mismatch");
        std::lock_guard<std::mutex> lock(recv_mtx_);
        TaggedCtl ack{htobe32(kTagAck), 0, htobe64(tag), 0};
        if (!write_full_locked_(&ack, sizeof(ack)))
            throw std::runtime_error("FRCheck RDMA: failed to send tagged scatter ack");
        recv_segments_chunked_(segments, logical_size, wait_cb, done_cb);
        return size;
    }

    void reset_tagged_ack_timing() {
        tagged_ack_wait_total_us_.store(0, std::memory_order_relaxed);
        tagged_ack_wait_max_us_.store(0, std::memory_order_relaxed);
        tagged_ack_wait_count_.store(0, std::memory_order_relaxed);
    }
    uint64_t tagged_ack_wait_total_us() const {
        return tagged_ack_wait_total_us_.load(std::memory_order_relaxed);
    }
    uint64_t tagged_ack_wait_max_us() const {
        return tagged_ack_wait_max_us_.load(std::memory_order_relaxed);
    }
    uint64_t tagged_ack_wait_count() const {
        return tagged_ack_wait_count_.load(std::memory_order_relaxed);
    }
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
        if (n == 2) {
            // Simple two-rank microbench mode only needs RDMA channels, not POA/RS coding.
            n_ = n;
            poa_path_ = "(generated n=2 simple)";
            int lanes = 1;
            const char* send_lanes_env = std::getenv("FRCHECK_SEND_LANES_PER_PEER");
            const char* recv_lanes_env = std::getenv("FRCHECK_RECV_LANES_PER_PEER");
            if (send_lanes_env && send_lanes_env[0] != '\0' &&
                recv_lanes_env && recv_lanes_env[0] != '\0') {
                lanes = std::max(1, std::atoi(send_lanes_env)) +
                        std::max(1, std::atoi(recv_lanes_env));
            } else {
                const char* lanes_env = std::getenv("FRCHECK_RDMA_LANES_PER_PEER");
                if (lanes_env && lanes_env[0] != '\0') {
                    lanes = std::max(1, std::atoi(lanes_env));
                }
            }
            table_.assign((size_t)lanes, std::vector<int>{1, 2});
        } else {
            generate_poa_(n);
            init_encode_tables_();
        }
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
        decoder_send_cv_.notify_all();
        failed_cv_.notify_all();
        recovery_batch_cv_.notify_all();
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

        num_stripes_ = (int)table_.size();
        if (num_stripes_ <= 0)
            throw std::runtime_error("FRCheck RDMA: POA table has no stripes");
        lane_direction_split_ = false;
        save_forward_lanes_ = 0;
        save_reverse_lanes_ = 0;
        const char* send_lanes_env = std::getenv("FRCHECK_SEND_LANES_PER_PEER");
        const char* recv_lanes_env = std::getenv("FRCHECK_RECV_LANES_PER_PEER");
        if (send_lanes_env && send_lanes_env[0] != '\0' &&
            recv_lanes_env && recv_lanes_env[0] != '\0') {
            save_forward_lanes_ = std::max(1, std::atoi(send_lanes_env));
            save_reverse_lanes_ = std::max(1, std::atoi(recv_lanes_env));
            num_lanes_ = save_forward_lanes_ + save_reverse_lanes_;
            lane_direction_split_ = true;
        } else {
            num_lanes_ = num_stripes_;
            const char* lanes_env = std::getenv("FRCHECK_RDMA_LANES_PER_PEER");
            if (lanes_env && lanes_env[0] != '\0') {
                int requested_lanes = std::atoi(lanes_env);
                if (requested_lanes > 0) {
                    int bounded_lanes = std::min(num_stripes_, requested_lanes);
                    if (bounded_lanes < num_stripes_) {
                        // Shared lanes are bidirectional and carry interleaved stripe
                        // messages; correctness relies on the tagged multiplexing
                        // protocol (per-channel reader thread + DATA/ACK records).
                        // Keep it opt-in while the path is experimental.
                        const char* unsafe_env = std::getenv("FRCHECK_ALLOW_UNSAFE_LANE_SHARING");
                        if (!unsafe_env || std::atoi(unsafe_env) == 0) {
                            throw std::runtime_error(
                                "FRCheck RDMA: FRCHECK_RDMA_LANES_PER_PEER < num_stripes uses the "
                                "experimental tagged shared-lane path; set "
                                "FRCHECK_ALLOW_UNSAFE_LANE_SHARING=1 to enable it");
                        }
                    }
                    num_lanes_ = bounded_lanes;
                }
            }
        }
        // n=2 simple microbench can issue multiple concurrent layer tasks per peer.
        // Use the tagged control path so parallel recv/send threads cannot steal FIFO headers.
        shared_lane_ = (num_lanes_ < num_stripes_) || (n_ == 2) || lane_direction_split_;
        if (debug_)
            std::cout << "[FRCheck RDMA] rank=" << rank_in_group
                      << " num_stripes=" << num_stripes_
                      << " num_lanes=" << num_lanes_
                      << " shared_lane=" << (shared_lane_ ? 1 : 0)
                      << " split_lanes=" << (lane_direction_split_ ? 1 : 0)
                      << " forward_lanes=" << save_forward_lanes_
                      << " reverse_lanes=" << save_reverse_lanes_
                      << " (connections per peer=" << num_lanes_ << ")" << std::endl;

        // Launch accept thread
        acceptor_thread_stop_ = false;
        accept_thread_ = std::thread([this]() { accept_loop_(); });

        // channels_[peer_rg][lane_id], where lane_id is stripe_id % num_lanes_.
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

        // Connect to lower ranks: one TCP+QP per (peer, lane).
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

        // Shared-lane mode: start a per-channel tagged receiver thread so that
        // out-of-order stripe messages on a shared channel are dispatched to
        // the correct destination buffer by tag instead of by FIFO position.
        if (shared_lane_) {
            for (auto& peer_row : channels_) {
                for (auto* ch : peer_row) {
                    if (ch) ch->start_tag_receiver();
                }
            }
        }

        if (debug_)
            std::cout << "[FRCheck RDMA] rank=" << rank_in_group
                      << " all " << group_size_ << " peers x " << num_lanes_
                  << " lanes connected (per-channel CQ pairs="
                  << (group_size_ - 1) * num_lanes_ << ")" << std::endl;

        // Init RS encode thread pool
        rs_pool_init();

        // Pre-compile stripe plans only for real FRCheck POA mode.
        if (n_ >= 3) {
            compile_stripe_plans_();
        }

        // Start per-stripe workers (needs stripe_plans_ populated)
        mirror_worker_init();
        init_stripe_workers();
    }

    int group_size() const { return group_size_; }
    int rank_in_group() const { return rank_in_group_; }

    // ---- Buffer registration ----
    void register_buffer(uintptr_t addr, size_t size) {
        register_buffer_(addr, size, false);
    }

    void register_recovery_buffer(uintptr_t addr, size_t size) {
        register_buffer_(addr, size, true);
    }

    void unregister_buffer(uintptr_t addr) {
        std::lock_guard<std::mutex> lk(buf_mtx_);
        auto it = registered_bufs_.find(addr);
        if (it != registered_bufs_.end()) {
            if (it->second.mr) ibv_dereg_mr(it->second.mr);
            registered_bufs_.erase(it);
        }
        recovery_buffer_addrs_.erase(addr);
    }

    void clear_recovery_buffers() {
        const int pending_chunks = pending_recovery_chunks_.load(std::memory_order_acquire);
        if (pending_chunks != 0) {
            throw std::runtime_error(
                "FRCheck: clear_recovery_buffers called with pending recovery chunks: " +
                std::to_string(pending_chunks));
        }
        std::lock_guard<std::mutex> lk(buf_mtx_);
        for (uintptr_t addr : recovery_buffer_addrs_) {
            auto it = registered_bufs_.find(addr);
            if (it == registered_bufs_.end()) continue;
            if (it->second.mr) ibv_dereg_mr(it->second.mr);
            registered_bufs_.erase(it);
        }
        recovery_buffer_addrs_.clear();
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
    void submit_stripe_decode_batch(
        int k,
        const std::vector<int>& survivor_positions,
        const std::vector<int>& lost_positions,
        const std::vector<uintptr_t>& survivor_addrs,
        const std::vector<uintptr_t>& recovered_addrs,
        size_t block_size)
    {
        if (stopped_)
            throw std::runtime_error("FRCheck decode: native runtime is stopped");
        if (k <= 0 || k > 253 || block_size == 0 ||
            block_size > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("FRCheck decode: invalid decode arguments");
        if (lost_positions.empty() || lost_positions.size() > 2)
            throw std::runtime_error("FRCheck decode: invalid lost position count");
        if (recovered_addrs.size() != lost_positions.size())
            throw std::runtime_error("FRCheck decode: lost position/output count mismatch");
        if (survivor_positions.size() != static_cast<size_t>(k) ||
            survivor_addrs.size() != static_cast<size_t>(k)) {
            throw std::runtime_error(
                "FRCheck decode: survivor count/address mismatch: positions=" +
                std::to_string(survivor_positions.size()) + " addrs=" +
                std::to_string(survivor_addrs.size()) + " expected=" +
                std::to_string(k));
        }
        std::set<uintptr_t> unique_survivor_addrs;
        for (uintptr_t addr : survivor_addrs) {
            if (addr == 0)
                throw std::runtime_error("FRCheck decode: null survivor address");
            if (!unique_survivor_addrs.insert(addr).second)
                throw std::runtime_error("FRCheck decode: duplicate survivor address");
        }
        std::set<uintptr_t> unique_outputs;
        for (uintptr_t addr : recovered_addrs) {
            if (addr == 0)
                throw std::runtime_error("FRCheck decode: null recovered address");
            if (!unique_outputs.insert(addr).second)
                throw std::runtime_error("FRCheck decode: duplicate recovered address");
            if (unique_survivor_addrs.count(addr) != 0)
                throw std::runtime_error("FRCheck decode: recovered address aliases survivor");
        }

        std::lock_guard<std::mutex> lk(decode_mtx_);
        init_decode_tables_(k, survivor_positions, lost_positions);
        if (!decode_tbls_)
            throw std::runtime_error("FRCheck decode: failed to initialize decode tables");

        const int output_count = static_cast<int>(lost_positions.size());
        std::vector<unsigned char*> data_ptrs(static_cast<size_t>(k));
        for (int i = 0; i < k; ++i)
            data_ptrs[static_cast<size_t>(i)] =
                reinterpret_cast<unsigned char*>(survivor_addrs[static_cast<size_t>(i)]);
        std::vector<unsigned char*> output_ptrs(static_cast<size_t>(output_count));
        for (int i = 0; i < output_count; ++i)
            output_ptrs[static_cast<size_t>(i)] =
                reinterpret_cast<unsigned char*>(recovered_addrs[static_cast<size_t>(i)]);

        if (rs_pool_inited_.load(std::memory_order_acquire)) {
            RsEncodeJob job;
            job.len = static_cast<int>(block_size);
            job.k = k;
            job.m = output_count;
            job.g_tbls = decode_tbls_;
            job.data_ptrs = data_ptrs.data();
            job.parity_ptrs = output_ptrs.data();
            rs_pool_run_parallel_encode(job);
        } else {
            ec_encode_data(static_cast<int>(block_size), k, output_count,
                           decode_tbls_, data_ptrs.data(), output_ptrs.data());
        }
    }

    void submit_stripe_decode(
        int k,
        const std::vector<int>& survivor_positions,
        int lost_position,
        const std::vector<uintptr_t>& survivor_addrs,
        uintptr_t recovered_addr,
        size_t block_size)
    {
        submit_stripe_decode_batch(
            k, survivor_positions, std::vector<int>{lost_position}, survivor_addrs,
            std::vector<uintptr_t>{recovered_addr}, block_size);
    }

    // ---- Point-to-point RDMA (for recovery) ----
    void send_to_peer(int peer_rig, int stripe_id, uintptr_t addr, size_t size,
                      uint64_t batch_id = 0, int tag_kind = 3) {
        int lane_id = (num_lanes_ > 0) ? (stripe_id % num_lanes_) : stripe_id;
        FRCheckRdmaChannel* ch = get_channel_by_lane_(peer_rig, lane_id);
        if (!ch) {
            throw std::runtime_error(
                "FRCheck send_to_peer: no channel to rig " + std::to_string(peer_rig) +
                " lane " + std::to_string(lane_id));
        }
        if (shared_lane_) {
            ch->send_tagged(make_channel_tag_(tag_kind, stripe_id, batch_id),
                            (const uint8_t*)addr, size);
        } else {
            ch->send_data((const uint8_t*)addr, size);
        }
    }

    void recv_from_peer(int peer_rig, int stripe_id, uintptr_t addr, size_t size,
                        uint64_t batch_id = 0, int tag_kind = 3) {
        int lane_id = (num_lanes_ > 0) ? (stripe_id % num_lanes_) : stripe_id;
        FRCheckRdmaChannel* ch = get_channel_by_lane_(peer_rig, lane_id);
        if (!ch) {
            throw std::runtime_error(
                "FRCheck recv_from_peer: no channel from rig " + std::to_string(peer_rig) +
                " lane " + std::to_string(lane_id));
        }
        if (shared_lane_) {
            ch->recv_tagged(make_channel_tag_(tag_kind, stripe_id, batch_id),
                            (uint8_t*)addr, size);
        } else {
            ch->recv_data((uint8_t*)addr, size);
        }
    }

    void send_layer_to_peer(int peer_rig, uintptr_t addr, size_t size, uint64_t batch_id, int lane_id = 0) {
        int mapped_lane_id = map_save_send_lane_(peer_rig, lane_id);
        FRCheckRdmaChannel* ch = get_channel_by_lane_(peer_rig, mapped_lane_id);
        if (!ch) {
            throw std::runtime_error(
                "FRCheck layer send: no channel to rig " + std::to_string(peer_rig) +
                " lane " + std::to_string(mapped_lane_id));
        }
        uint64_t net_t0 = frcheck_now_us();
        record_save_net_start_(net_t0);
        if (shared_lane_) {
            ch->send_tagged(make_channel_tag_(2, mapped_lane_id, batch_id), (const uint8_t*)addr, size);
        } else {
            ch->send_data((const uint8_t*)addr, size);
        }
        uint64_t net_t1 = frcheck_now_us();
        record_save_net_end_(net_t1);
        uint64_t elapsed = net_t1 - net_t0;
        save_source_send_total_us_.fetch_add(elapsed, std::memory_order_relaxed);
        save_source_send_tasks_.fetch_add(1, std::memory_order_relaxed);
        save_source_send_bytes_.fetch_add(size, std::memory_order_relaxed);
        record_atomic_max_(save_source_send_max_us_, elapsed);
    }

    int get_max_send_sge() const {
        int result = FRCHECK_DEFAULT_MAX_SEND_SGE;
        bool found = false;
        for (const auto& channel : channel_owners_) {
            if (!channel) continue;
            result = found ? std::min(result, channel->max_send_sge()) : channel->max_send_sge();
            found = true;
        }
        return found ? result : 1;
    }

    void send_layer_blocks_to_peer(
            int peer_rig, uintptr_t mirror_base,
            const std::vector<size_t>& block_indices, size_t block_size,
            uint64_t batch_id, int lane_id = 0) {
        if (block_size == 0 || block_indices.empty())
            throw std::runtime_error("FRCheck layer scatter send: empty block list or block size");
        if (block_indices.size() > SIZE_MAX / block_size)
            throw std::runtime_error("FRCheck layer scatter send: logical size overflow");
        int mapped_lane_id = map_save_send_lane_(peer_rig, lane_id);
        FRCheckRdmaChannel* ch = get_channel_by_lane_(peer_rig, mapped_lane_id);
        if (!ch)
            throw std::runtime_error("FRCheck layer scatter send: no channel to rig " +
                                     std::to_string(peer_rig));
        std::vector<std::pair<uintptr_t, size_t>> segments;
        segments.reserve(block_indices.size());
        for (size_t block_idx : block_indices) {
            if (block_idx > (SIZE_MAX - mirror_base) / block_size)
                throw std::runtime_error("FRCheck layer scatter send: block address overflow");
            segments.emplace_back(mirror_base + block_idx * block_size, block_size);
        }
        size_t logical_size = block_indices.size() * block_size;
        uint64_t net_t0 = frcheck_now_us();
        record_save_net_start_(net_t0);
        if (shared_lane_)
            ch->send_tagged_segments(make_channel_tag_(2, mapped_lane_id, batch_id),
                                     segments, logical_size);
        else
            ch->send_segments(segments, logical_size);
        uint64_t net_t1 = frcheck_now_us();
        record_save_net_end_(net_t1);
        uint64_t elapsed = net_t1 - net_t0;
        save_source_send_total_us_.fetch_add(elapsed, std::memory_order_relaxed);
        save_source_send_tasks_.fetch_add(1, std::memory_order_relaxed);
        save_source_send_bytes_.fetch_add(logical_size, std::memory_order_relaxed);
        size_t wr_count = 0;
        size_t sge_count = 0;
        size_t max_sge = 0;
        size_t segment_idx = 0;
        size_t segment_offset = 0;
        size_t stats_remaining = logical_size;
        while (stats_remaining > 0) {
            size_t message_remaining = std::min(stats_remaining, frcheck_rdma_chunk_size());
            size_t message_sge = 0;
            while (message_remaining > 0) {
                while (segment_offset == segments[segment_idx].second) {
                    ++segment_idx;
                    segment_offset = 0;
                }
                size_t length = std::min(
                    message_remaining, segments[segment_idx].second - segment_offset);
                segment_offset += length;
                message_remaining -= length;
                stats_remaining -= length;
                ++message_sge;
            }
            ++wr_count;
            sge_count += message_sge;
            max_sge = std::max(max_sge, message_sge);
        }
        save_source_send_wr_count_.fetch_add(wr_count, std::memory_order_relaxed);
        save_source_send_sge_count_.fetch_add(sge_count, std::memory_order_relaxed);
        record_atomic_max_(save_source_send_max_sge_, max_sge);
        record_atomic_max_(save_source_send_max_us_, elapsed);
    }

    void prepare_layer_recv_from_peer(
            int peer_rig, uintptr_t addr, size_t capacity,
            uint64_t batch_id, int lane_id = 0) {
        int mapped_lane_id = map_save_recv_lane_(peer_rig, lane_id);
        FRCheckRdmaChannel* ch = get_channel_by_lane_(peer_rig, mapped_lane_id);
        if (!ch) throw std::runtime_error("FRCheck prepared layer recv: no channel");
        if (!shared_lane_)
            throw std::runtime_error("FRCheck prepared layer recv requires shared lanes");
        record_save_net_start_(frcheck_now_us());
        ch->prepare_tagged_recv(
            make_channel_tag_(2, mapped_lane_id, batch_id),
            reinterpret_cast<uint8_t*>(addr), capacity);
    }

    size_t wait_prepared_layer_recv_from_peer(
            int peer_rig, uint64_t batch_id, int lane_id = 0) {
        int mapped_lane_id = map_save_recv_lane_(peer_rig, lane_id);
        FRCheckRdmaChannel* ch = get_channel_by_lane_(peer_rig, mapped_lane_id);
        if (!ch) throw std::runtime_error("FRCheck prepared layer recv wait: no channel");
        uint64_t net_t0 = frcheck_now_us();
        record_save_net_start_(net_t0);
        size_t got = ch->wait_prepared_tagged_recv(
            make_channel_tag_(2, mapped_lane_id, batch_id));
        uint64_t net_t1 = frcheck_now_us();
        record_save_net_end_(net_t1);
        uint64_t elapsed = net_t1 - net_t0;
        save_enc_recv_total_us_.fetch_add(elapsed, std::memory_order_relaxed);
        save_enc_recv_tasks_.fetch_add(1, std::memory_order_relaxed);
        save_enc_recv_bytes_.fetch_add(got, std::memory_order_relaxed);
        record_atomic_max_(save_enc_recv_max_us_, elapsed);
        return got;
    }

    size_t recv_layer_from_peer(int peer_rig, uintptr_t addr, size_t capacity, uint64_t batch_id, int lane_id = 0) {
        int mapped_lane_id = map_save_recv_lane_(peer_rig, lane_id);
        FRCheckRdmaChannel* ch = get_channel_by_lane_(peer_rig, mapped_lane_id);
        if (!ch) {
            throw std::runtime_error(
                "FRCheck layer recv: no channel from rig " + std::to_string(peer_rig) +
                " lane " + std::to_string(mapped_lane_id));
        }
        uint64_t net_t0 = frcheck_now_us();
        record_save_net_start_(net_t0);
        size_t got = 0;
        if (shared_lane_) {
            got = ch->recv_tagged(make_channel_tag_(2, mapped_lane_id, batch_id), (uint8_t*)addr, capacity);
        } else {
            got = ch->recv_data((uint8_t*)addr, capacity);
        }
        uint64_t net_t1 = frcheck_now_us();
        record_save_net_end_(net_t1);
        uint64_t elapsed = net_t1 - net_t0;
        save_enc_recv_total_us_.fetch_add(elapsed, std::memory_order_relaxed);
        save_enc_recv_tasks_.fetch_add(1, std::memory_order_relaxed);
        save_enc_recv_bytes_.fetch_add(got, std::memory_order_relaxed);
        record_atomic_max_(save_enc_recv_max_us_, elapsed);
        return got;
    }

    double simple_exchange_with_peer(
        int peer_rig,
        uintptr_t send_addr,
        uintptr_t recv_addr,
        size_t total_size,
        int lanes,
        uint64_t send_batch_id,
        uint64_t recv_batch_id) {
        lanes = std::max(1, lanes);
        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> errors((size_t)lanes * 2);
        threads.reserve((size_t)lanes * 2);
        auto t0 = std::chrono::steady_clock::now();

        for (int lane = 0; lane < lanes; ++lane) {
            size_t start = (total_size * (size_t)lane) / (size_t)lanes;
            size_t end = (total_size * (size_t)(lane + 1)) / (size_t)lanes;
            size_t part = end - start;
            if (part == 0) continue;
            uint64_t recv_tag = recv_batch_id + (uint64_t)lane * 104729ULL;
            uint64_t send_tag = send_batch_id + (uint64_t)lane * 104729ULL;
            size_t recv_err_idx = (size_t)lane * 2;
            size_t send_err_idx = recv_err_idx + 1;
            threads.emplace_back([this, peer_rig, recv_addr, start, part, recv_tag, lane, recv_err_idx, &errors]() {
                try {
                    this->recv_layer_from_peer(peer_rig, recv_addr + start, part, recv_tag, lane);
                } catch (...) {
                    errors[recv_err_idx] = std::current_exception();
                }
            });
            threads.emplace_back([this, peer_rig, send_addr, start, part, send_tag, lane, send_err_idx, &errors]() {
                try {
                    this->send_layer_to_peer(peer_rig, send_addr + start, part, send_tag, lane);
                } catch (...) {
                    errors[send_err_idx] = std::current_exception();
                }
            });
        }

        for (auto& th : threads) {
            if (th.joinable()) th.join();
        }
        for (auto& ep : errors) {
            if (ep) std::rethrow_exception(ep);
        }
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    }

    void set_require_registered_mr(bool require) { require_registered_mr_ = require; }
    void set_debug(bool d) { debug_ = d; }

    // ---- Hardware recovery batch pipeline ----
    void reset_recovery_generation() {
        const int pending_chunks = pending_recovery_chunks_.load(std::memory_order_acquire);
        const int active_workers =
            helper_active_.load(std::memory_order_acquire) +
            decoder_active_.load(std::memory_order_acquire) +
            decoder_send_active_.load(std::memory_order_acquire) +
            failed_active_.load(std::memory_order_acquire);
        if (pending_chunks != 0 || active_workers != 0)
            throw std::runtime_error(
                "FRCheck: recovery generation reset before drain: pending=" +
                std::to_string(pending_chunks) + " active=" +
                std::to_string(active_workers));
        reset_recovery_completion_();
        {
            std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
            recovery_batches_.clear();
            if (recovery_generation_ >= kRecoveryGenerationMax)
                throw std::runtime_error("FRCheck: recovery generation exhausted");
            ++recovery_generation_;
            next_recovery_batch_id_ =
                (recovery_generation_ << kRecoveryBatchCycleBits) + 1;
            legacy_recovery_batch_active_ = false;
            legacy_recovery_batch_id_ = 0;
        }
    }

    void init_recovery_plans(const std::vector<int>& failed_nodes_1based) {
        reset_recovery_generation();
        clear_recovery_buffers();

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

    uint64_t begin_recovery_batch(bool low_priority = false) {
        if (encoding_batch_active_)
            throw std::runtime_error("FRCheck: encode batch active, cannot start recovery batch");
        ensure_recovery_workers_();
        reset_recovery_batch_profile_();
        uint64_t batch_id;
        {
            std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
            const uint64_t generation_end =
                (recovery_generation_ + 1) << kRecoveryBatchCycleBits;
            if (recovery_generation_ == 0 || next_recovery_batch_id_ >= generation_end)
                throw std::runtime_error("FRCheck: recovery batch ID exhausted for generation");
            batch_id = next_recovery_batch_id_++;
        }
        std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
        RecoveryBatchState state;
        state.batch_start_us = frcheck_now_us();
        state.low_priority = low_priority;
        recovery_batches_[batch_id] = state;
        return batch_id;
    }

    void reset_recovery_batch() {
        if (legacy_recovery_batch_active_)
            throw std::runtime_error("FRCheck: recovery batch already active");
        legacy_recovery_batch_id_ = begin_recovery_batch();
        legacy_recovery_batch_active_ = true;
    }

    void submit_recovery_stripe_to_batch(
        uint64_t batch_id,
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
        bool store_to_layer_buf,
        bool active,
        const std::vector<int>& target_roles = {})
    {
        if (stopped_)
            throw std::runtime_error("FRCheck recovery submit: native runtime is stopped");
        ensure_recovery_batch_exists_(batch_id);
        if (stripe_id < 0 || stripe_id >= (int)recovery_plans_.size())
            throw std::runtime_error("FRCheck: invalid recovery stripe_id");

        const RecoveryStripePlan& plan = recovery_plans_[stripe_id];
        int my_node = rank_in_group_ + 1;

        if (!active) {
            recovery_skipped_stripes_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        std::set<int> requested_roles;
        for (int role : target_roles) {
            if (role < (int)StripeRole::SOURCE || role > (int)StripeRole::PARITY_TARGET)
                throw std::runtime_error("FRCheck recovery submit: invalid target role");
            if (!requested_roles.insert(role).second)
                throw std::runtime_error("FRCheck recovery submit: duplicate target role");
        }
        auto role_selected = [&](int role) {
            return requested_roles.empty() || requested_roles.count(role) != 0;
        };

        std::vector<const RecoveryFailedTarget*> selected_targets;
        RecoveryFailedTarget single_target;
        if (plan.dual_failure) {
            for (const auto& ft : plan.failed_targets) {
                if (role_selected(ft.original_role))
                    selected_targets.push_back(&ft);
            }
        } else if (role_selected(plan.original_role)) {
            single_target.failed_node = plan.failed_node;
            single_target.failed_pos = plan.failed_pos;
            single_target.original_role = plan.original_role;
            selected_targets.push_back(&single_target);
        }
        if (selected_targets.empty())
            return;
        if (selected_targets.size() > 2)
            throw std::runtime_error("FRCheck recovery submit: too many selected targets");

        auto is_helper = [&]() {
            return std::find(plan.helper_nodes.begin(), plan.helper_nodes.end(), my_node)
                   != plan.helper_nodes.end();
        };

        if (is_helper() && helper_block_addr == 0)
            throw std::runtime_error("FRCheck recovery submit: missing helper block");
        if (is_helper()) {
            RecoveryHelperTask task;
            task.batch_id = batch_id;
            task.stripe_id = stripe_id;
            task.block_size = block_size;
            task.helper_block = helper_block_addr;
            task.decoder_rig = plan.decoder_node - 1;
            recovery_batch_add_expected_(batch_id, 1);
            pending_recovery_chunks_.fetch_add(1, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lk(helper_mtx_);
                helper_q_.push(std::move(task));
            }
            helper_cv_.notify_all();
        }

        if (my_node == plan.decoder_node && decoder_self_block_addr == 0)
            throw std::runtime_error("FRCheck recovery submit: missing decoder self block");
        if (my_node == plan.decoder_node) {
            if (decoder_helper_recv_addrs.size() != plan.helper_nodes.size())
                throw std::runtime_error("FRCheck recovery submit: helper buffer count mismatch");
            if (decoder_recovered_addrs.size() < selected_targets.size())
                throw std::runtime_error("FRCheck recovery submit: recovered buffer count mismatch");
            RecoveryDecoderTask task;
            task.batch_id = batch_id;
            task.stripe_id = stripe_id;
            task.block_size = block_size;
            task.self_block = decoder_self_block_addr;
            task.helper_recv_bufs = decoder_helper_recv_addrs;
            for (int hn : plan.helper_nodes)
                task.helper_rigs.push_back(hn - 1);
            task.dual_failure = plan.dual_failure;
            task.survivor_positions = plan.survivor_positions;
            for (size_t i = 0; i < selected_targets.size(); ++i) {
                const auto& ft = *selected_targets[i];
                task.failed_positions.push_back(ft.failed_pos);
                task.failed_rigs.push_back(ft.failed_node - 1);
                task.recovered_bufs.push_back(decoder_recovered_addrs[i]);
            }
            task.failed_pos = selected_targets[0]->failed_pos;
            int decoder_outputs = (int)selected_targets.size();
            task.completion_count = decoder_outputs;
            recovery_batch_add_expected_(batch_id, decoder_outputs);
            pending_recovery_chunks_.fetch_add(decoder_outputs, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lk(decoder_mtx_);
                decoder_q_.push(std::move(task));
            }
            decoder_cv_.notify_all();
        }

        for (const RecoveryFailedTarget* ft : selected_targets) {
            if (my_node != ft->failed_node)
                continue;
            if (failed_recv_buf_addr == 0)
                throw std::runtime_error("FRCheck recovery submit: missing failed receive buffer");
            RecoveryFailedTask task;
            task.batch_id = batch_id;
            task.stripe_id = stripe_id;
            task.block_size = block_size;
            task.recv_buf = failed_recv_buf_addr;
            task.decoder_rig = plan.decoder_node - 1;
            task.layer_buf = failed_layer_buf_addr;
            task.layer_offset = failed_layer_offset;
            task.ncopy = failed_ncopy;
            task.store_to_layer = store_to_layer_buf;
            recovery_batch_add_expected_(batch_id, 1);
            pending_recovery_chunks_.fetch_add(1, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lk(failed_mtx_);
                failed_q_.push(std::move(task));
            }
            failed_cv_.notify_all();
            break;
        }
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
        bool store_to_layer_buf,
        bool active)
    {
        if (!legacy_recovery_batch_active_)
            throw std::runtime_error("FRCheck: submit_recovery_stripe without reset_recovery_batch");
        submit_recovery_stripe_to_batch(
            legacy_recovery_batch_id_, stripe_id, block_size, helper_block_addr,
            decoder_self_block_addr, decoder_helper_recv_addrs, decoder_recovered_addrs,
            failed_recv_buf_addr, failed_layer_buf_addr, failed_layer_offset,
            failed_ncopy, store_to_layer_buf, active);
    }

    void end_recovery_batch(uint64_t batch_id) {
        {
            std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
            auto it = recovery_batches_.find(batch_id);
            if (it == recovery_batches_.end())
                throw std::runtime_error("FRCheck: unknown recovery batch id");
            it->second.closed = true;
        }
        recovery_batch_cv_.notify_all();
    }

    void wait_recovery_batch_id(uint64_t batch_id) {
        std::string error;
        {
            std::unique_lock<std::mutex> lk(recovery_batch_mtx_);
            recovery_batch_cv_.wait(lk, [&] {
                if (stopped_.load()) return true;
                auto it = recovery_batches_.find(batch_id);
                return it != recovery_batches_.end() && it->second.closed &&
                       it->second.done >= it->second.expected;
            });
            auto it = recovery_batches_.find(batch_id);
            if (it != recovery_batches_.end()) {
                error = it->second.error;
            }
        }
        print_recovery_batch_profile_();
        if (!error.empty())
            throw std::runtime_error(error);
    }

    void submit_recovery_sentinel() {
        if (!legacy_recovery_batch_active_)
            throw std::runtime_error("FRCheck: submit_recovery_sentinel without reset_recovery_batch");
        end_recovery_batch(legacy_recovery_batch_id_);
    }

    void wait_recovery_batch() {
        if (!legacy_recovery_batch_active_)
            throw std::runtime_error("FRCheck: wait_recovery_batch without reset_recovery_batch");
        wait_recovery_batch_id(legacy_recovery_batch_id_);
        discard_recovery_batch_milestones_(legacy_recovery_batch_id_);
        legacy_recovery_batch_active_ = false;
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
    void register_buffer_(uintptr_t addr, size_t size, bool recovery_buffer) {
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
        auto existing = registered_bufs_.find(addr);
        if (existing != registered_bufs_.end()) {
            if (existing->second.mr) ibv_dereg_mr(existing->second.mr);
            registered_bufs_.erase(existing);
        }
        registered_bufs_.emplace(addr, RdmaBuffer{mr, addr, size});
        if (recovery_buffer)
            recovery_buffer_addrs_.insert(addr);
        else
            recovery_buffer_addrs_.erase(addr);
    }

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
        if (n_ < 3 || n_ > kRsMaxDataBlocks + kRsMaxParityBlocks) {
            throw std::runtime_error(
                "FRCheck: RS coding requires n in [3," +
                std::to_string(kRsMaxDataBlocks + kRsMaxParityBlocks) + "]");
        }
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
    // Builds output rows for one or more losses from the same survivor matrix.
    void init_decode_tables_(int k,
                             const std::vector<int>& survivor_positions,
                             const std::vector<int>& lost_positions) {
        const int m_parity = 2;
        const int full_rows = k + m_parity;
        if (k <= 0 || survivor_positions.size() != static_cast<size_t>(k))
            throw std::runtime_error("FRCheck decode table: survivor count mismatch");
        if (lost_positions.empty() || lost_positions.size() > static_cast<size_t>(m_parity))
            throw std::runtime_error("FRCheck decode table: invalid lost position count");

        std::set<int> unique_survivors;
        for (int pos : survivor_positions) {
            if (pos < 0 || pos >= full_rows)
                throw std::runtime_error("FRCheck decode table: survivor position out of range");
            if (!unique_survivors.insert(pos).second)
                throw std::runtime_error("FRCheck decode table: duplicate survivor position");
        }
        std::set<int> unique_losses;
        for (int pos : lost_positions) {
            if (pos < 0 || pos >= full_rows)
                throw std::runtime_error("FRCheck decode table: lost position out of range");
            if (!unique_losses.insert(pos).second)
                throw std::runtime_error("FRCheck decode table: duplicate lost position");
            if (unique_survivors.count(pos) != 0)
                throw std::runtime_error("FRCheck decode table: lost position listed as survivor");
        }

        std::vector<unsigned char> encode_mat(
            static_cast<size_t>(k) * static_cast<size_t>(full_rows));
        gf_gen_rs_matrix(encode_mat.data(), full_rows, k);

        std::vector<unsigned char> survivor_mat(
            static_cast<size_t>(k) * static_cast<size_t>(k), 0);
        for (int row = 0; row < k; ++row) {
            const int pos = survivor_positions[static_cast<size_t>(row)];
            for (int col = 0; col < k; ++col)
                survivor_mat[static_cast<size_t>(row) * k + col] =
                    encode_mat[static_cast<size_t>(pos) * k + col];
        }

        std::vector<unsigned char> inverse(
            static_cast<size_t>(k) * static_cast<size_t>(k));
        const int ret = gf_invert_matrix(survivor_mat.data(), inverse.data(), k);
        if (ret != 0) {
            throw std::runtime_error(
                "FRCheck decode table: gf_invert_matrix failed, ret=" +
                std::to_string(ret));
        }

        const int output_count = static_cast<int>(lost_positions.size());
        std::vector<unsigned char> decode_mat(
            static_cast<size_t>(output_count) * static_cast<size_t>(k), 0);
        for (int output = 0; output < output_count; ++output) {
            const int lost_pos = lost_positions[static_cast<size_t>(output)];
            for (int survivor = 0; survivor < k; ++survivor) {
                unsigned char coeff = 0;
                for (int data = 0; data < k; ++data) {
                    coeff ^= gf_mul(
                        encode_mat[static_cast<size_t>(lost_pos) * k + data],
                        inverse[static_cast<size_t>(data) * k + survivor]);
                }
                decode_mat[static_cast<size_t>(output) * k + survivor] = coeff;
            }
        }

        const size_t tbl_size = 32 * static_cast<size_t>(k) *
                                static_cast<size_t>(output_count);
        void* tmp = nullptr;
        if (posix_memalign(&tmp, 32, tbl_size) != 0) tmp = nullptr;
        if (tmp == nullptr) tmp = malloc(tbl_size);
        if (tmp == nullptr)
            throw std::runtime_error("FRCheck decode table: allocation failed");
        unsigned char* new_decode_tbls = static_cast<unsigned char*>(tmp);
        ec_init_tables(k, output_count, decode_mat.data(), new_decode_tbls);
        if (decode_tbls_) free(decode_tbls_);
        decode_tbls_ = new_decode_tbls;
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
        rs_pool_plan_spans_.reserve(kRsInitialPlanSpans);
        for (auto& e : rs_pool_last_epoch_) e = 0;
        for (auto& elapsed : rs_pool_worker_elapsed_us_)
            elapsed.store(0, std::memory_order_relaxed);
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
            const std::vector<RsEncodeJob>* jobs = rs_pool_shared_jobs_;
            lk.unlock();

            if (jobs != nullptr) {
                const uint64_t compute_t0 = frcheck_now_us();
                rs_pool_execute_batch_slice(*jobs, wid);
                rs_pool_worker_elapsed_us_[wid].store(
                    frcheck_now_us() - compute_t0, std::memory_order_release);
            } else {
                rs_pool_execute_slice(job, wid);
            }

            {
                std::lock_guard<std::mutex> guard(rs_pool_mutex_);
                rs_pool_last_epoch_[wid] = e;
            }
            int left = rs_pool_remaining_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            if (left == 0) rs_pool_coordinator_cv_.notify_one();
        }
    }

    void rs_pool_execute_slice(const RsEncodeJob& job, int wid) {
        if (job.k <= 0 || job.m <= 0 || job.g_tbls == nullptr ||
            job.data_ptrs == nullptr || job.parity_ptrs == nullptr)
            return;
        for (int i = 0; i < job.k; ++i) {
            if (job.data_ptrs[i] == nullptr) return;
        }
        for (int i = 0; i < job.m; ++i) {
            if (job.parity_ptrs[i] == nullptr) return;
        }

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
        std::vector<unsigned char*> dest((size_t)job.m);
        for (int i = 0; i < job.m; ++i)
            dest[i] = job.parity_ptrs[i] + off;

        ec_encode_data(len, job.k, job.m, job.g_tbls, src.data(), dest.data());
    }

    void rs_pool_execute_batch_slice(const std::vector<RsEncodeJob>& jobs, int wid) {
        const size_t begin = rs_pool_worker_plan_begin_[static_cast<size_t>(wid)];
        const size_t count = rs_pool_worker_plan_count_[static_cast<size_t>(wid)];
        for (size_t i = 0; i < count; ++i) {
            const RsPoolSpan& span = rs_pool_plan_spans_[begin + i];
            rs_pool_execute_range(
                jobs[static_cast<size_t>(span.job_index)], span.offset, span.length);
        }
    }

    void rs_pool_execute_range(const RsEncodeJob& job, int off, int len) {
        if (len <= 0 || off < 0 || off > job.len || len > job.len - off ||
            job.k <= 0 || job.k > kRsMaxDataBlocks ||
            job.m <= 0 || job.m > kRsMaxParityBlocks ||
            job.g_tbls == nullptr || job.data_ptrs == nullptr ||
            job.parity_ptrs == nullptr)
            return;

        std::array<unsigned char*, kRsMaxDataBlocks> src;
        std::array<unsigned char*, kRsMaxParityBlocks> dest;
        for (int i = 0; i < job.k; ++i) {
            if (job.data_ptrs[i] == nullptr) return;
            src[static_cast<size_t>(i)] = job.data_ptrs[i] + off;
        }
        for (int i = 0; i < job.m; ++i) {
            if (job.parity_ptrs[i] == nullptr) return;
            dest[static_cast<size_t>(i)] = job.parity_ptrs[i] + off;
        }
        ec_encode_data(len, job.k, job.m, job.g_tbls, src.data(), dest.data());
        save_encode_ec_calls_.fetch_add(1, std::memory_order_relaxed);
    }

    void rs_pool_build_batch_plan(const std::vector<RsEncodeJob>& jobs) {
        uint64_t total_bytes = 0;
        for (const auto& job : jobs) {
            if (job.len < 0 || job.k <= 0 || job.k > kRsMaxDataBlocks ||
                job.m <= 0 || job.m > kRsMaxParityBlocks ||
                job.g_tbls == nullptr || job.data_ptrs == nullptr ||
                job.parity_ptrs == nullptr) {
                throw std::runtime_error("FRCheck: invalid RS batch job");
            }
            for (int i = 0; i < job.k; ++i) {
                if (job.data_ptrs[i] == nullptr)
                    throw std::runtime_error("FRCheck: null RS batch source pointer");
            }
            for (int i = 0; i < job.m; ++i) {
                if (job.parity_ptrs[i] == nullptr)
                    throw std::runtime_error("FRCheck: null RS batch parity pointer");
            }
            const uint64_t len = static_cast<uint64_t>(job.len);
            if (total_bytes > std::numeric_limits<uint64_t>::max() - len)
                throw std::runtime_error("FRCheck: RS batch byte count overflow");
            total_bytes += len;
        }

        rs_pool_plan_spans_.clear();
        if (jobs.size() <= std::numeric_limits<size_t>::max() - kRsPoolSize &&
            rs_pool_plan_spans_.capacity() < jobs.size() + kRsPoolSize) {
            rs_pool_plan_spans_.reserve(jobs.size() + kRsPoolSize);
        }

        const uint64_t base = total_bytes / static_cast<uint64_t>(kRsPoolSize);
        const uint64_t rem = total_bytes % static_cast<uint64_t>(kRsPoolSize);
        size_t job_index = 0;
        uint64_t job_begin = 0;
        for (int wid = 0; wid < kRsPoolSize; ++wid) {
            const uint64_t worker_begin = static_cast<uint64_t>(wid) * base +
                std::min<uint64_t>(static_cast<uint64_t>(wid), rem);
            const uint64_t worker_len = base +
                (static_cast<uint64_t>(wid) < rem ? 1 : 0);
            const uint64_t worker_end = worker_begin + worker_len;
            rs_pool_worker_plan_begin_[static_cast<size_t>(wid)] =
                rs_pool_plan_spans_.size();

            while (job_index < jobs.size() &&
                   job_begin + static_cast<uint64_t>(jobs[job_index].len) <= worker_begin) {
                job_begin += static_cast<uint64_t>(jobs[job_index].len);
                ++job_index;
            }
            size_t scan_index = job_index;
            uint64_t scan_begin = job_begin;
            while (scan_index < jobs.size() && scan_begin < worker_end) {
                const uint64_t scan_end =
                    scan_begin + static_cast<uint64_t>(jobs[scan_index].len);
                const uint64_t begin = std::max(worker_begin, scan_begin);
                const uint64_t end = std::min(worker_end, scan_end);
                if (begin < end) {
                    const uint64_t offset = begin - scan_begin;
                    const uint64_t length = end - begin;
                    if (scan_index > std::numeric_limits<uint32_t>::max() ||
                        offset > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                        length > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                        throw std::runtime_error("FRCheck: RS batch span exceeds supported range");
                    }
                    rs_pool_plan_spans_.push_back(RsPoolSpan{
                        static_cast<uint32_t>(scan_index), static_cast<int>(offset),
                        static_cast<int>(length)});
                }
                scan_begin = scan_end;
                ++scan_index;
            }
            rs_pool_worker_plan_count_[static_cast<size_t>(wid)] =
                rs_pool_plan_spans_.size() -
                rs_pool_worker_plan_begin_[static_cast<size_t>(wid)];
        }
    }

    void rs_pool_run_parallel_encode(const RsEncodeJob& job) {
        {
            std::lock_guard<std::mutex> publish(rs_pool_mutex_);
            if (stopped_) return;
            rs_pool_shared_job_ = job;
            rs_pool_shared_jobs_ = nullptr;
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

    // Encode multiple stripes in one pool dispatch. Workers divide the combined
    // logical byte range, so each job is touched only by intersecting workers.
    // `jobs` must stay alive until this call returns.
    void rs_pool_run_parallel_encode_batch(const std::vector<RsEncodeJob>& jobs) {
        if (jobs.empty()) return;
        const uint64_t plan_t0 = frcheck_now_us();
        rs_pool_build_batch_plan(jobs);
        save_encode_pool_plan_total_us_.fetch_add(
            frcheck_now_us() - plan_t0, std::memory_order_relaxed);
        for (auto& elapsed : rs_pool_worker_elapsed_us_)
            elapsed.store(0, std::memory_order_relaxed);
        const uint64_t barrier_t0 = frcheck_now_us();
        {
            std::lock_guard<std::mutex> publish(rs_pool_mutex_);
            if (stopped_) return;
            rs_pool_shared_jobs_ = &jobs;
            rs_pool_epoch_.fetch_add(1, std::memory_order_acq_rel);
            rs_pool_remaining_.store(kRsPoolSize, std::memory_order_release);
        }
        rs_pool_worker_cv_.notify_all();
        {
            std::unique_lock<std::mutex> lk(rs_pool_mutex_);
            rs_pool_coordinator_cv_.wait(lk, [&] {
                return rs_pool_remaining_.load(std::memory_order_acquire) == 0 ||
                       stopped_.load() || rs_pool_stop_.load(std::memory_order_acquire);
            });
            rs_pool_shared_jobs_ = nullptr;
        }
        const uint64_t barrier_us = frcheck_now_us() - barrier_t0;
        uint64_t worker_max_us = 0;
        for (const auto& elapsed : rs_pool_worker_elapsed_us_) {
            worker_max_us = std::max(
                worker_max_us, elapsed.load(std::memory_order_acquire));
        }
        save_encode_pool_barrier_total_us_.fetch_add(
            barrier_us, std::memory_order_relaxed);
        save_encode_worker_compute_max_total_us_.fetch_add(
            worker_max_us, std::memory_order_relaxed);
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
    std::atomic<bool> mirror_done_{false};
    std::atomic<bool> mirror_idle_{true};
    cudaStream_t d2h_stream_ = nullptr;
    std::atomic<uint64_t> mirror_d2h_busy_total_us_{0};
    std::atomic<uint64_t> mirror_tasks_submitted_{0};
    std::atomic<uint64_t> mirror_bytes_submitted_{0};
    std::atomic<uint64_t> mirror_tasks_completed_{0};
    std::atomic<uint64_t> mirror_bytes_completed_{0};
    std::atomic<uint64_t> mirror_tasks_failed_{0};
    std::atomic<uint64_t> mirror_bytes_failed_{0};

    struct MirrorCopyTiming {
        cudaEvent_t start{};
        cudaEvent_t end{};
        bool valid{false};
    };
    std::vector<MirrorCopyTiming> mirror_copy_timings_;
    std::mutex mirror_timing_mutex_;
    std::atomic<uint64_t> save_net_start_us_{0};
    std::atomic<uint64_t> save_net_end_us_{0};
    std::atomic<uint64_t> save_encode_total_us_{0};
    std::atomic<uint64_t> save_encode_wait_total_us_{0};
    std::atomic<uint64_t> save_encode_batch_dispatches_{0};
    std::atomic<uint64_t> save_encode_batch_jobs_{0};
    std::atomic<uint64_t> save_encode_ec_calls_{0};
    std::atomic<uint64_t> save_encode_pool_plan_total_us_{0};
    std::atomic<uint64_t> save_encode_pool_barrier_total_us_{0};
    std::atomic<uint64_t> save_encode_worker_compute_max_total_us_{0};
    std::atomic<uint64_t> save_source_send_total_us_{0};
    std::atomic<uint64_t> save_source_send_max_us_{0};
    std::atomic<uint64_t> save_enc_recv_total_us_{0};
    std::atomic<uint64_t> save_enc_recv_max_us_{0};
    std::atomic<uint64_t> save_source_send_tasks_{0};
    std::atomic<uint64_t> save_enc_recv_tasks_{0};
    std::atomic<uint64_t> save_source_send_bytes_{0};
    std::atomic<uint64_t> save_source_send_wr_count_{0};
    std::atomic<uint64_t> save_source_send_sge_count_{0};
    std::atomic<uint64_t> save_source_send_max_sge_{0};
    std::atomic<uint64_t> save_enc_recv_bytes_{0};

    // ---- Role-based queues: n workers each, encoder split into RECV→encode+SEND ----
    struct StripeInfo {
        std::vector<int> src_peer_rigs;
        int enc_peer_rig = -1;
        int par_peer_rig = -1;
    };
    std::vector<StripeInfo> stripe_info_;

    struct SourceTask   { int sid; uintptr_t data, mirror; size_t bs; uint64_t encode_batch = 0; bool skip = false; };
    struct EncRecvTask  { int sid; uintptr_t recv, p1, p2; size_t bs; std::vector<uint8_t> mask; uint64_t encode_batch = 0; };
    struct EncRecvSubTask { int sid; int src_idx; uintptr_t dst; size_t bs; uint64_t batch; std::shared_ptr<std::atomic<int>> remaining; std::shared_ptr<std::mutex> done_mtx; std::shared_ptr<std::condition_variable> done_cv; };
    struct EncSendTask  { int sid; uintptr_t recv, p1, p2; size_t bs; int n_src; std::vector<int> src_peer_rigs; int par_peer_rig; bool parity_send_only = false; uint64_t async_order = 0; uint64_t async_batch = 0; };
    struct ParityTask   { int sid; uintptr_t parity_in; size_t bs; bool async_p2_only = false; uint64_t async_order = 0; uint64_t async_batch = 0; };

    std::queue<SourceTask>   source_q_;   std::mutex source_mtx_;   std::condition_variable source_cv_;
    std::queue<EncRecvTask>  enc_recv_q_; std::mutex enc_recv_mtx_; std::condition_variable enc_recv_cv_;
    std::queue<EncRecvSubTask> enc_recv_part_q_; std::mutex enc_recv_part_mtx_; std::condition_variable enc_recv_part_cv_;
    std::queue<EncSendTask>  enc_send_q_; std::mutex enc_send_mtx_; std::condition_variable enc_send_cv_;
    std::queue<ParityTask>   parity_q_;   std::mutex parity_mtx_;   std::condition_variable parity_cv_;

    std::vector<std::thread> source_workers_, enc_recv_workers_, enc_recv_part_workers_, enc_send_workers_, parity_workers_;
    std::atomic<bool> all_stop_{false};
    std::atomic<int> task_total_{0}, task_done_{0};
    std::atomic<int> task_encode_total_{0}, task_encode_done_{0};
    std::atomic<int> task_async_total_{0}, task_async_done_{0};
    std::vector<std::thread> aggregate_p2_threads_;
    std::mutex aggregate_p2_mtx_;
    std::condition_variable aggregate_p2_cv_;
    std::exception_ptr aggregate_p2_error_;
    std::atomic<int> aggregate_p2_total_{0};
    std::atomic<int> aggregate_p2_done_{0};
    uint64_t aggregate_p2_generation_ = 0;
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
    std::vector<uint64_t> async_p2_send_batch_run_order_;
    std::vector<uint64_t> async_p2_recv_batch_run_order_;
    std::vector<uint64_t> encode_source_batch_run_order_;
    std::vector<uint64_t> encode_recv_batch_run_order_;
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

    void wait_async_p2_send_turn_(int sid, uint64_t order, uint64_t batch = 0) {
        std::unique_lock<std::mutex> lk(async_p2_order_mtx_);
        while (!all_stop_.load(std::memory_order_acquire) &&
               ((batch == 0 && sid >= 0 && sid < (int)async_p2_send_run_order_.size() &&
                 async_p2_send_run_order_[(size_t)sid] != order) ||
                (batch != 0 && sid >= 0 && sid < (int)async_p2_send_batch_run_order_.size() &&
                 async_p2_send_batch_run_order_[(size_t)sid] != batch))) {
            if (debug_) {
                std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                          << " async_p2_send wait_turn sid=" << sid
                          << " order=" << order
                          << " batch=" << batch
                          << " expect=" << (sid >= 0 && sid < (int)async_p2_send_run_order_.size() ? async_p2_send_run_order_[(size_t)sid] : 0)
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

    void advance_async_p2_send_turn_(int sid, uint64_t batch = 0) {
        {
            std::lock_guard<std::mutex> lk(async_p2_order_mtx_);
            if (batch == 0) {
                if (sid >= 0 && sid < (int)async_p2_send_run_order_.size())
                    ++async_p2_send_run_order_[(size_t)sid];
            } else {
                if (sid >= 0 && sid < (int)async_p2_send_batch_run_order_.size())
                    ++async_p2_send_batch_run_order_[(size_t)sid];
            }
        }
        async_p2_order_cv_.notify_all();
    }

    void wait_async_p2_recv_turn_(int sid, uint64_t order, uint64_t batch = 0) {
        std::unique_lock<std::mutex> lk(async_p2_order_mtx_);
        while (!all_stop_.load(std::memory_order_acquire) &&
               ((batch == 0 && sid >= 0 && sid < (int)async_p2_recv_run_order_.size() &&
                 async_p2_recv_run_order_[(size_t)sid] != order) ||
                (batch != 0 && sid >= 0 && sid < (int)async_p2_recv_batch_run_order_.size() &&
                 async_p2_recv_batch_run_order_[(size_t)sid] != batch))) {
            if (debug_) {
                std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                          << " async_p2_recv wait_turn sid=" << sid
                          << " order=" << order
                          << " batch=" << batch
                          << " expect=" << (sid >= 0 && sid < (int)async_p2_recv_run_order_.size() ? async_p2_recv_run_order_[(size_t)sid] : 0)
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

    void advance_async_p2_recv_turn_(int sid, uint64_t batch = 0) {
        {
            std::lock_guard<std::mutex> lk(async_p2_order_mtx_);
            if (batch == 0) {
                if (sid >= 0 && sid < (int)async_p2_recv_run_order_.size())
                    ++async_p2_recv_run_order_[(size_t)sid];
            } else {
                if (sid >= 0 && sid < (int)async_p2_recv_batch_run_order_.size())
                    ++async_p2_recv_batch_run_order_[(size_t)sid];
            }
        }
        async_p2_order_cv_.notify_all();
    }

    void wait_encode_source_turn_(int sid, uint64_t batch) {
        if (batch == 0) return;
        std::unique_lock<std::mutex> lk(async_p2_order_mtx_);
        while (!all_stop_.load(std::memory_order_acquire) &&
               sid >= 0 && sid < (int)encode_source_batch_run_order_.size() &&
               encode_source_batch_run_order_[(size_t)sid] != batch) {
            async_p2_order_cv_.wait_for(lk, std::chrono::seconds(1));
        }
    }

    void advance_encode_source_turn_(int sid, uint64_t batch) {
        if (batch == 0) return;
        {
            std::lock_guard<std::mutex> lk(async_p2_order_mtx_);
            if (sid >= 0 && sid < (int)encode_source_batch_run_order_.size())
                ++encode_source_batch_run_order_[(size_t)sid];
        }
        async_p2_order_cv_.notify_all();
    }

    void wait_encode_recv_turn_(int sid, uint64_t batch) {
        if (batch == 0) return;
        std::unique_lock<std::mutex> lk(async_p2_order_mtx_);
        while (!all_stop_.load(std::memory_order_acquire) &&
               sid >= 0 && sid < (int)encode_recv_batch_run_order_.size() &&
               encode_recv_batch_run_order_[(size_t)sid] != batch) {
            async_p2_order_cv_.wait_for(lk, std::chrono::seconds(1));
        }
    }

    void advance_encode_recv_turn_(int sid, uint64_t batch) {
        if (batch == 0) return;
        {
            std::lock_guard<std::mutex> lk(async_p2_order_mtx_);
            if (sid >= 0 && sid < (int)encode_recv_batch_run_order_.size())
                ++encode_recv_batch_run_order_[(size_t)sid];
        }
        async_p2_order_cv_.notify_all();
    }

    void record_save_net_start_(uint64_t t) {
        uint64_t old = save_net_start_us_.load(std::memory_order_relaxed);
        while ((old == 0 || t < old) &&
               !save_net_start_us_.compare_exchange_weak(
                   old, t, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
    }

    void record_save_net_end_(uint64_t t) {
        uint64_t old = save_net_end_us_.load(std::memory_order_relaxed);
        while (t > old &&
               !save_net_end_us_.compare_exchange_weak(
                   old, t, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
    }

    void record_atomic_max_(std::atomic<uint64_t>& target, uint64_t value) {
        uint64_t old = target.load(std::memory_order_relaxed);
        while (value > old &&
               !target.compare_exchange_weak(
                   old, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
    }

    void record_atomic_min_nonzero_(std::atomic<uint64_t>& target, uint64_t value) {
        uint64_t old = target.load(std::memory_order_relaxed);
        while ((old == 0 || value < old) &&
               !target.compare_exchange_weak(
                   old, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
    }

    void record_recovery_net_start_(uint64_t t) {
        record_atomic_min_nonzero_(recovery_net_start_us_, t);
    }

    void record_recovery_net_end_(uint64_t t) {
        record_atomic_max_(recovery_net_end_us_, t);
    }

    void record_recovery_decode_start_(uint64_t t) {
        record_atomic_min_nonzero_(recovery_decode_start_us_, t);
    }

    void record_recovery_decode_end_(uint64_t t) {
        record_atomic_max_(recovery_decode_end_us_, t);
    }

    void source_worker_() {
        while (!all_stop_) {
            SourceTask t;
            { std::unique_lock<std::mutex> lk(source_mtx_); source_cv_.wait(lk, [&]{ return all_stop_ || !source_q_.empty(); });
              if (all_stop_ && source_q_.empty()) break; t = source_q_.front(); source_q_.pop(); }
            auto& si = stripe_info_[(size_t)t.sid];
            wait_encode_source_turn_(t.sid, t.encode_batch);
            if (!t.skip) {
                auto* ch = get_save_send_channel_(si.enc_peer_rig, t.sid);
                if (!ch) { std::cerr << "FRCheck source " << t.sid << ": no channel\n"; continue; }
                uint64_t net_t0 = frcheck_now_us();
                record_save_net_start_(net_t0);
                if (shared_lane_) {
                    ch->send_tagged(make_channel_tag_(0, t.sid, t.encode_batch),
                                    (const uint8_t*)t.data, t.bs);
                } else {
                    ch->send_data((const uint8_t*)t.data, t.bs);
                }
                uint64_t net_t1 = frcheck_now_us();
                record_save_net_end_(net_t1);
                uint64_t elapsed = net_t1 - net_t0;
                save_source_send_total_us_.fetch_add(elapsed, std::memory_order_relaxed);
                save_source_send_tasks_.fetch_add(1, std::memory_order_relaxed);
                save_source_send_bytes_.fetch_add(t.bs, std::memory_order_relaxed);
                record_atomic_max_(save_source_send_max_us_, elapsed);
                if (t.mirror) push_mirror_task_(t.data, t.mirror, t.bs);
            }
            advance_encode_source_turn_(t.sid, t.encode_batch);
            check_encode_done_();
            check_done_();
        }
    }

    void enc_recv_part_worker_() {
        while (!all_stop_) {
            EncRecvSubTask t;
            {
                std::unique_lock<std::mutex> lk(enc_recv_part_mtx_);
                enc_recv_part_cv_.wait(lk, [&]{ return all_stop_ || !enc_recv_part_q_.empty(); });
                if (all_stop_ && enc_recv_part_q_.empty()) break;
                t = enc_recv_part_q_.front();
                enc_recv_part_q_.pop();
            }
            auto& si = stripe_info_[(size_t)t.sid];
            if (t.src_idx >= 0 && t.src_idx < (int)si.src_peer_rigs.size() &&
                si.src_peer_rigs[(size_t)t.src_idx] != rank_in_group_) {
                auto* ch = get_save_recv_channel_(si.src_peer_rigs[(size_t)t.src_idx], t.sid);
                if (ch) {
                    uint64_t net_t0 = frcheck_now_us();
                    record_save_net_start_(net_t0);
                    if (shared_lane_) {
                        ch->recv_tagged(make_channel_tag_(0, t.sid, t.batch),
                                        (uint8_t*)t.dst, t.bs);
                    } else {
                        ch->recv_data((uint8_t*)t.dst, t.bs);
                    }
                    uint64_t net_t1 = frcheck_now_us();
                    record_save_net_end_(net_t1);
                    uint64_t elapsed = net_t1 - net_t0;
                    save_enc_recv_total_us_.fetch_add(elapsed, std::memory_order_relaxed);
                    save_enc_recv_tasks_.fetch_add(1, std::memory_order_relaxed);
                    save_enc_recv_bytes_.fetch_add(t.bs, std::memory_order_relaxed);
                    record_atomic_max_(save_enc_recv_max_us_, elapsed);
                }
            }
            if (t.remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(*t.done_mtx);
                t.done_cv->notify_one();
            }
        }
    }

    void enc_recv_worker_() {
        while (!all_stop_) {
            EncRecvTask t;
            { std::unique_lock<std::mutex> lk(enc_recv_mtx_); enc_recv_cv_.wait(lk, [&]{ return all_stop_ || !enc_recv_q_.empty(); });
              if (all_stop_ && enc_recv_q_.empty()) break; t = std::move(enc_recv_q_.front()); enc_recv_q_.pop(); }
            auto& si = stripe_info_[(size_t)t.sid];
            wait_encode_recv_turn_(t.sid, t.encode_batch);
            int n_src = (int)si.src_peer_rigs.size();
            int active_remote = 0;
            auto remaining = std::make_shared<std::atomic<int>>(0);
            auto done_mtx = std::make_shared<std::mutex>();
            auto done_cv = std::make_shared<std::condition_variable>();

            for (int i = 0; i < n_src; ++i) {
                if (!t.mask.empty() && i < (int)t.mask.size() && t.mask[(size_t)i] == 0) continue;
                if (si.src_peer_rigs[(size_t)i] == rank_in_group_) continue;
                ++active_remote;
            }
            remaining->store(active_remote, std::memory_order_release);
            if (active_remote > 0) {
                {
                    std::lock_guard<std::mutex> lk(enc_recv_part_mtx_);
                    for (int i = 0; i < n_src; ++i) {
                        if (!t.mask.empty() && i < (int)t.mask.size() && t.mask[(size_t)i] == 0) continue;
                        if (si.src_peer_rigs[(size_t)i] == rank_in_group_) continue;
                        uintptr_t dst = t.recv + (uintptr_t)i * t.bs;
                        enc_recv_part_q_.push({t.sid, i, dst, t.bs, t.encode_batch, remaining, done_mtx, done_cv});
                    }
                }
                enc_recv_part_cv_.notify_all();
                std::unique_lock<std::mutex> lk(*done_mtx);
                done_cv->wait(lk, [&]{ return remaining->load(std::memory_order_acquire) == 0 || all_stop_.load(std::memory_order_acquire); });
            }
            advance_encode_recv_turn_(t.sid, t.encode_batch);
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
                wait_async_p2_send_turn_(t.sid, t.async_order, t.async_batch);
                _wait_if_paused();
                auto wait_cb = [this]() { this->_async_rdma_begin(); };
                auto done_cb = [this]() { this->_async_rdma_end(); };
                {
                    auto* ch = get_save_send_channel_(t.par_peer_rig, t.sid);
                    if (debug_) {
                        std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                                  << " async_p2_send begin sid=" << t.sid
                                  << " order=" << t.async_order
                                  << " batch=" << t.async_batch
                                  << " peer=" << t.par_peer_rig
                                  << " bytes=" << t.bs
                                  << " done=" << task_async_done_.load(std::memory_order_acquire)
                                  << "/" << task_async_total_.load(std::memory_order_acquire)
                                  << " ch=" << (ch ? 1 : 0) << std::endl;
                    }
                    if (ch) {
                        uint64_t net_t0 = frcheck_now_us();
                        record_save_net_start_(net_t0);
                        if (shared_lane_) {
                            ch->send_tagged(make_channel_tag_(1, t.sid, t.async_batch),
                                            (const uint8_t*)t.p1, t.bs, wait_cb, done_cb);
                        } else {
                            ch->send_data((const uint8_t*)t.p1, t.bs, wait_cb, done_cb);
                        }
                        record_save_net_end_(frcheck_now_us());
                    }
                }
                advance_async_p2_send_turn_(t.sid, t.async_batch);
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
                uint64_t encode_t0 = frcheck_now_us();
                { std::lock_guard<std::mutex> lk(encoder_encode_mtx_); rs_pool_run_parallel_encode(rs); }
                save_encode_total_us_.fetch_add(
                    frcheck_now_us() - encode_t0, std::memory_order_relaxed);
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
            auto* ch = get_save_recv_channel_(si.enc_peer_rig, t.sid);
            if (t.async_p2_only) wait_async_p2_recv_turn_(t.sid, t.async_order, t.async_batch);
            if (debug_ && t.async_p2_only) {
                std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                          << " async_p2_recv begin sid=" << t.sid
                          << " order=" << t.async_order
                          << " batch=" << t.async_batch
                          << " peer=" << si.enc_peer_rig
                          << " bytes=" << t.bs
                          << " done=" << task_async_done_.load(std::memory_order_acquire)
                          << "/" << task_async_total_.load(std::memory_order_acquire)
                          << " ch=" << (ch ? 1 : 0) << std::endl;
            }
            // Do not pause receive-side posts: a sender that already entered RDMA
            // needs the matching recv to make progress and drain before PP starts.
            if (ch) {
                uint64_t net_t0 = frcheck_now_us();
                record_save_net_start_(net_t0);
                if (shared_lane_ && t.async_p2_only) {
                    ch->recv_tagged(make_channel_tag_(1, t.sid, t.async_batch),
                                    (uint8_t*)t.parity_in, t.bs);
                } else {
                    ch->recv_data((uint8_t*)t.parity_in, t.bs);
                }
                record_save_net_end_(frcheck_now_us());
            }
            if (t.async_p2_only) advance_async_p2_recv_turn_(t.sid, t.async_batch);
            check_async_done_();
            if (debug_ && t.async_p2_only) {
                std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                          << " async_p2_recv done sid=" << t.sid
                          << " order=" << t.async_order
                          << " batch=" << t.async_batch
                          << " peer=" << si.enc_peer_rig
                          << " bytes=" << t.bs
                          << " done=" << task_async_done_.load(std::memory_order_acquire)
                          << "/" << task_async_total_.load(std::memory_order_acquire)
                          << std::endl;
            }
            check_done_();
        }
    }

    void reset_mirror_d2h_timing_() {
        mirror_d2h_busy_total_us_.store(0, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(mirror_timing_mutex_);
        for (auto& timing : mirror_copy_timings_) {
            if (!timing.valid) {
                continue;
            }
            cudaEventDestroy(timing.start);
            cudaEventDestroy(timing.end);
        }
        mirror_copy_timings_.clear();
    }

    void finalize_mirror_d2h_timing_() {
        uint64_t total_us = 0;
        std::lock_guard<std::mutex> lk(mirror_timing_mutex_);
        for (auto& timing : mirror_copy_timings_) {
            if (!timing.valid) {
                continue;
            }
            float elapsed_ms = 0.0f;
            if (cudaEventElapsedTime(&elapsed_ms, timing.start, timing.end) == cudaSuccess) {
                total_us += static_cast<uint64_t>(elapsed_ms * 1000.0);
            }
            cudaEventDestroy(timing.start);
            cudaEventDestroy(timing.end);
            timing.valid = false;
        }
        mirror_copy_timings_.clear();
        mirror_d2h_busy_total_us_.store(total_us, std::memory_order_relaxed);
    }

    void push_mirror_task_(uintptr_t gpu_addr, uintptr_t cpu_addr, size_t block_size) {
        MirrorTask mt{gpu_addr, cpu_addr, block_size};
        {
            std::lock_guard<std::mutex> lk(mirror_mtx_);
            mirror_q_.push(mt);
            mirror_idle_ = false;
        }
        mirror_tasks_submitted_.fetch_add(1, std::memory_order_relaxed);
        mirror_bytes_submitted_.fetch_add(block_size, std::memory_order_relaxed);
        mirror_cv_.notify_one();
    }

    void mirror_worker_func() {
        cudaSetDevice(resolve_cuda_device());
        while (true) {
            MirrorTask task;
            {
                std::unique_lock<std::mutex> lk(mirror_mtx_);
                mirror_cv_.wait(lk, [this] { return !mirror_q_.empty(); });
                task = mirror_q_.front();
                mirror_q_.pop();
            }
            if (task.gpu_addr == 0 && task.cpu_addr == 0) {
                mirror_done_ = true;
                break;
            }
            MirrorCopyTiming timing{};
            const bool start_ok = cudaEventCreate(&timing.start) == cudaSuccess;
            const bool end_ok = cudaEventCreate(&timing.end) == cudaSuccess;
            if (start_ok && end_ok) {
                cudaEventRecord(timing.start, d2h_stream_);
            } else {
                if (start_ok) {
                    cudaEventDestroy(timing.start);
                }
                if (end_ok) {
                    cudaEventDestroy(timing.end);
                }
            }
            cudaError_t err = cudaMemcpyAsync(
                reinterpret_cast<void*>(task.cpu_addr),
                reinterpret_cast<const void*>(task.gpu_addr),
                task.block_size,
                cudaMemcpyDeviceToHost,
                d2h_stream_);
            if (err != cudaSuccess) {
                mirror_tasks_failed_.fetch_add(1, std::memory_order_relaxed);
                mirror_bytes_failed_.fetch_add(task.block_size, std::memory_order_relaxed);
                std::cerr << "FRCheck mirror_worker: async D2H failed: "
                          << cudaGetErrorString(err) << std::endl;
            } else {
                mirror_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
                mirror_bytes_completed_.fetch_add(task.block_size, std::memory_order_relaxed);
            }
            if (start_ok && end_ok && err == cudaSuccess) {
                cudaEventRecord(timing.end, d2h_stream_);
                timing.valid = true;
                std::lock_guard<std::mutex> lk(mirror_timing_mutex_);
                mirror_copy_timings_.push_back(timing);
            }
        }
    }

public:
    void mirror_layer(uintptr_t gpu_addr, uintptr_t cpu_addr, size_t size) {
        if (gpu_addr == 0 || cpu_addr == 0 || size == 0) return;
        push_mirror_task_(gpu_addr, cpu_addr, size);
    }

    void encode_layer_stripes_batch(
        const std::vector<int>& stripe_ids,
        const std::vector<uintptr_t>& data_addrs,
        const std::vector<uintptr_t>& p1_addrs,
        const std::vector<uintptr_t>& p2_addrs,
        const std::vector<size_t>& block_sizes)
    {
        const int n_src = n_ - 2;
        const size_t n_jobs = stripe_ids.size();
        if (n_src <= 0 || n_jobs == 0) return;
        if (n_jobs > std::numeric_limits<size_t>::max() / static_cast<size_t>(n_src) ||
            data_addrs.size() != n_jobs * static_cast<size_t>(n_src) ||
            p1_addrs.size() != n_jobs || p2_addrs.size() != n_jobs ||
            block_sizes.size() != n_jobs) {
            throw std::runtime_error("FRCheck layer batch encode: invalid argument sizes");
        }

        std::vector<std::vector<unsigned char*>> src_ptrs(n_jobs);
        std::vector<std::array<unsigned char*, 2>> par_ptrs(n_jobs);
        std::vector<RsEncodeJob> jobs(n_jobs);
        for (size_t job_idx = 0; job_idx < n_jobs; ++job_idx) {
            const size_t block_size = block_sizes[job_idx];
            if (block_size == 0 || block_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("FRCheck layer batch encode: invalid block size");
            }
            if (p1_addrs[job_idx] == 0 || p2_addrs[job_idx] == 0) {
                throw std::runtime_error("FRCheck layer batch encode: missing parity output buffer");
            }
            src_ptrs[job_idx].resize(static_cast<size_t>(n_src));
            for (int src_idx = 0; src_idx < n_src; ++src_idx) {
                const size_t flat_idx = job_idx * static_cast<size_t>(n_src) +
                    static_cast<size_t>(src_idx);
                const uintptr_t addr = data_addrs[flat_idx];
                if (addr == 0) {
                    throw std::runtime_error("FRCheck layer batch encode: missing source block buffer");
                }
                src_ptrs[job_idx][static_cast<size_t>(src_idx)] =
                    reinterpret_cast<unsigned char*>(addr);
            }
            par_ptrs[job_idx][0] = reinterpret_cast<unsigned char*>(p1_addrs[job_idx]);
            par_ptrs[job_idx][1] = reinterpret_cast<unsigned char*>(p2_addrs[job_idx]);
            jobs[job_idx] = RsEncodeJob{
                static_cast<int>(block_size), n_src, 2, g_tbls_,
                src_ptrs[job_idx].data(), par_ptrs[job_idx].data(),
            };
        }

        const uint64_t wait_t0 = frcheck_now_us();
        std::lock_guard<std::mutex> lk(encoder_encode_mtx_);
        const uint64_t encode_t0 = frcheck_now_us();
        save_encode_wait_total_us_.fetch_add(encode_t0 - wait_t0, std::memory_order_relaxed);
        save_encode_batch_dispatches_.fetch_add(1, std::memory_order_relaxed);
        save_encode_batch_jobs_.fetch_add(n_jobs, std::memory_order_relaxed);
        rs_pool_run_parallel_encode_batch(jobs);
        save_encode_total_us_.fetch_add(frcheck_now_us() - encode_t0, std::memory_order_relaxed);
    }

    void encode_layer_stripes(
        const std::vector<int>& stripe_ids,
        const std::vector<uintptr_t>& data_addrs,
        const std::vector<uintptr_t>& p1_addrs,
        const std::vector<uintptr_t>& p2_addrs,
        size_t block_size)
    {
        std::vector<size_t> block_sizes(stripe_ids.size(), block_size);
        encode_layer_stripes_batch(
            stripe_ids, data_addrs, p1_addrs, p2_addrs, block_sizes);
    }

    void reset_ft_timing_stats() {
        for (auto& peer_row : channels_)
            for (auto* ch : peer_row)
                if (ch) ch->reset_tagged_ack_timing();
        save_net_start_us_.store(0, std::memory_order_relaxed);
        save_net_end_us_.store(0, std::memory_order_relaxed);
        save_encode_total_us_.store(0, std::memory_order_relaxed);
        save_encode_wait_total_us_.store(0, std::memory_order_relaxed);
        save_encode_batch_dispatches_.store(0, std::memory_order_relaxed);
        save_encode_batch_jobs_.store(0, std::memory_order_relaxed);
        save_encode_ec_calls_.store(0, std::memory_order_relaxed);
        save_encode_pool_plan_total_us_.store(0, std::memory_order_relaxed);
        save_encode_pool_barrier_total_us_.store(0, std::memory_order_relaxed);
        save_encode_worker_compute_max_total_us_.store(0, std::memory_order_relaxed);
        save_source_send_total_us_.store(0, std::memory_order_relaxed);
        save_source_send_max_us_.store(0, std::memory_order_relaxed);
        save_enc_recv_total_us_.store(0, std::memory_order_relaxed);
        save_enc_recv_max_us_.store(0, std::memory_order_relaxed);
        save_source_send_tasks_.store(0, std::memory_order_relaxed);
        save_enc_recv_tasks_.store(0, std::memory_order_relaxed);
        save_source_send_bytes_.store(0, std::memory_order_relaxed);
        save_source_send_wr_count_.store(0, std::memory_order_relaxed);
        save_source_send_sge_count_.store(0, std::memory_order_relaxed);
        save_source_send_max_sge_.store(0, std::memory_order_relaxed);
        save_enc_recv_bytes_.store(0, std::memory_order_relaxed);
        mirror_d2h_busy_total_us_.store(0, std::memory_order_relaxed);
        mirror_tasks_submitted_.store(0, std::memory_order_relaxed);
        mirror_bytes_submitted_.store(0, std::memory_order_relaxed);
        mirror_tasks_completed_.store(0, std::memory_order_relaxed);
        mirror_bytes_completed_.store(0, std::memory_order_relaxed);
        mirror_tasks_failed_.store(0, std::memory_order_relaxed);
        mirror_bytes_failed_.store(0, std::memory_order_relaxed);
    }

public:
    py::dict get_ft_timing_stats() const {
        const uint64_t net_start = save_net_start_us_.load(std::memory_order_relaxed);
        const uint64_t net_end = save_net_end_us_.load(std::memory_order_relaxed);
        const double net_s = (net_start > 0 && net_end > net_start)
            ? static_cast<double>(net_end - net_start) / 1e6
            : 0.0;
        py::dict result;
        result["net_s"] = net_s;
        result["encode_s"] = static_cast<double>(
            save_encode_total_us_.load(std::memory_order_relaxed)) / 1e6;
        result["encode_wait_s"] = static_cast<double>(
            save_encode_wait_total_us_.load(std::memory_order_relaxed)) / 1e6;
        result["encode_batch_dispatches"] = static_cast<double>(
            save_encode_batch_dispatches_.load(std::memory_order_relaxed));
        result["encode_batch_jobs"] = static_cast<double>(
            save_encode_batch_jobs_.load(std::memory_order_relaxed));
        result["encode_ec_calls"] = static_cast<double>(
            save_encode_ec_calls_.load(std::memory_order_relaxed));
        result["encode_pool_plan_s"] = static_cast<double>(
            save_encode_pool_plan_total_us_.load(std::memory_order_relaxed)) / 1e6;
        result["encode_pool_barrier_s"] = static_cast<double>(
            save_encode_pool_barrier_total_us_.load(std::memory_order_relaxed)) / 1e6;
        result["encode_worker_compute_max_sum_s"] = static_cast<double>(
            save_encode_worker_compute_max_total_us_.load(std::memory_order_relaxed)) / 1e6;
        result["d2h_s"] = static_cast<double>(
            mirror_d2h_busy_total_us_.load(std::memory_order_relaxed)) / 1e6;
        result["mirror_tasks_submitted"] = static_cast<double>(
            mirror_tasks_submitted_.load(std::memory_order_relaxed));
        result["mirror_bytes_submitted"] = static_cast<double>(
            mirror_bytes_submitted_.load(std::memory_order_relaxed));
        result["mirror_tasks_completed"] = static_cast<double>(
            mirror_tasks_completed_.load(std::memory_order_relaxed));
        result["mirror_bytes_completed"] = static_cast<double>(
            mirror_bytes_completed_.load(std::memory_order_relaxed));
        result["mirror_tasks_failed"] = static_cast<double>(
            mirror_tasks_failed_.load(std::memory_order_relaxed));
        result["mirror_bytes_failed"] = static_cast<double>(
            mirror_bytes_failed_.load(std::memory_order_relaxed));
        uint64_t tagged_ack_wait_total_us = 0;
        uint64_t tagged_ack_wait_max_us = 0;
        uint64_t tagged_ack_wait_count = 0;
        for (const auto& peer_row : channels_) {
            for (const auto* ch : peer_row) {
                if (!ch) continue;
                tagged_ack_wait_total_us += ch->tagged_ack_wait_total_us();
                tagged_ack_wait_max_us = std::max(
                    tagged_ack_wait_max_us, ch->tagged_ack_wait_max_us());
                tagged_ack_wait_count += ch->tagged_ack_wait_count();
            }
        }
        result["tagged_ack_wait_sum_s"] = static_cast<double>(tagged_ack_wait_total_us) / 1e6;
        result["tagged_ack_wait_max_s"] = static_cast<double>(tagged_ack_wait_max_us) / 1e6;
        result["tagged_ack_wait_count"] = static_cast<double>(tagged_ack_wait_count);
        result["source_send_sum_s"] = static_cast<double>(
            save_source_send_total_us_.load(std::memory_order_relaxed)) / 1e6;
        result["source_send_max_s"] = static_cast<double>(
            save_source_send_max_us_.load(std::memory_order_relaxed)) / 1e6;
        result["enc_recv_sum_s"] = static_cast<double>(
            save_enc_recv_total_us_.load(std::memory_order_relaxed)) / 1e6;
        result["enc_recv_max_s"] = static_cast<double>(
            save_enc_recv_max_us_.load(std::memory_order_relaxed)) / 1e6;
        result["source_send_tasks"] = static_cast<double>(
            save_source_send_tasks_.load(std::memory_order_relaxed));
        result["enc_recv_tasks"] = static_cast<double>(
            save_enc_recv_tasks_.load(std::memory_order_relaxed));
        result["source_send_bytes"] = static_cast<double>(
            save_source_send_bytes_.load(std::memory_order_relaxed));
        result["source_send_wr_count"] = static_cast<double>(
            save_source_send_wr_count_.load(std::memory_order_relaxed));
        result["source_send_sge_count"] = static_cast<double>(
            save_source_send_sge_count_.load(std::memory_order_relaxed));
        result["source_send_max_sge"] = static_cast<double>(
            save_source_send_max_sge_.load(std::memory_order_relaxed));
        result["enc_recv_bytes"] = static_cast<double>(
            save_enc_recv_bytes_.load(std::memory_order_relaxed));
        return result;
    }

private:
    void finalize_mirror_stream_() {
        if (d2h_stream_ != nullptr) {
            cudaSetDevice(resolve_cuda_device());
            cudaError_t err = cudaStreamSynchronize(d2h_stream_);
            if (err != cudaSuccess) {
                mirror_tasks_failed_.fetch_add(1, std::memory_order_relaxed);
                finalize_mirror_d2h_timing_();
                cudaStreamDestroy(d2h_stream_);
                d2h_stream_ = nullptr;
                throw std::runtime_error(
                    std::string("FRCheck mirror: d2h_stream sync failed: ") +
                    cudaGetErrorString(err));
            }
            finalize_mirror_d2h_timing_();
            cudaStreamDestroy(d2h_stream_);
            d2h_stream_ = nullptr;
        }
    }

    void stop_mirror_worker_() {
        if (mirror_thread_.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mirror_mtx_);
                mirror_q_.push({0, 0, 0});
            }
            mirror_cv_.notify_one();
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(300);
            while (!mirror_done_.load()) {
                if (std::chrono::steady_clock::now() > deadline) {
                    std::cerr << "FRCheck mirror: worker drain timed out on rank "
                              << rank_in_group_ << std::endl;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (mirror_thread_.joinable()) {
                mirror_thread_.join();
            }
        } else {
            std::lock_guard<std::mutex> lk(mirror_mtx_);
            while (!mirror_q_.empty()) {
                mirror_q_.pop();
            }
        }
        finalize_mirror_stream_();
        mirror_done_ = false;
    }

    void mirror_worker_init() {
        start_mirror_worker();
    }

    void mirror_worker_shutdown() {
        stop_mirror_worker_();
        reset_mirror_d2h_timing_();
        mirror_idle_ = true;
    }

public:
    void start_mirror_worker() {
        if (mirror_thread_.joinable()) {
            stop_mirror_worker_();
        } else {
            finalize_mirror_stream_();
            {
                std::lock_guard<std::mutex> lk(mirror_mtx_);
                while (!mirror_q_.empty()) {
                    mirror_q_.pop();
                }
            }
            mirror_done_ = false;
        }
        cudaSetDevice(resolve_cuda_device());
        cudaError_t err = cudaStreamCreate(&d2h_stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("FRCheck: cudaStreamCreate(d2h_stream) failed: ") +
                cudaGetErrorString(err));
        }
        mirror_thread_ = std::thread(&FRCheckNative::mirror_worker_func, this);
    }

private:
    void init_stripe_workers() {
        int ns = (int)stripe_plans_.size();
        stripe_info_.resize((size_t)ns);
        async_p2_send_batch_run_order_.assign((size_t)ns, 1);
        async_p2_recv_batch_run_order_.assign((size_t)ns, 1);
        encode_source_batch_run_order_.assign((size_t)ns, 1);
        encode_recv_batch_run_order_.assign((size_t)ns, 1);
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
        // Each encoder stripe may recv from (n-2) remote sources in parallel.
        // Use a dedicated pool large enough for concurrent stripes * sources.
        int enc_recv_part_nw = nw * std::max(1, nw - 2);
        const char* part_env = std::getenv("FRCHECK_ENC_RECV_PART_WORKERS");
        if (part_env && part_env[0] != '\0') {
            int env_nw = std::atoi(part_env);
            if (env_nw > 0) enc_recv_part_nw = env_nw;
        }
        for (int w = 0; w < nw; ++w) {
            source_workers_.emplace_back(&FRCheckNative::source_worker_, this);
            enc_recv_workers_.emplace_back(&FRCheckNative::enc_recv_worker_, this);
            enc_send_workers_.emplace_back(&FRCheckNative::enc_send_worker_, this);
            parity_workers_.emplace_back(&FRCheckNative::parity_worker_, this);
        }
        for (int w = 0; w < enc_recv_part_nw; ++w) {
            enc_recv_part_workers_.emplace_back(&FRCheckNative::enc_recv_part_worker_, this);
        }
        // Reuse enc_send workers for deferred P2 sends (n parallel workers,
        // same as sync path).  P2 send tasks are pushed to enc_send_q_ so
        // the existing n workers pick them up in FIFO order.
        if (debug_)
            std::cout << "FRCheck: " << nw << " workers/role, "
                      << enc_recv_part_nw << " enc_recv_part workers for "
                      << ns << " stripes" << std::endl;
    }

    void shutdown_stripe_workers() {
        all_stop_.store(true, std::memory_order_release);
        source_cv_.notify_all(); enc_recv_cv_.notify_all();
        enc_recv_part_cv_.notify_all(); enc_send_cv_.notify_all(); parity_cv_.notify_all();
        async_p2_order_cv_.notify_all();
        for (auto* v : {&source_workers_, &enc_recv_workers_, &enc_recv_part_workers_, &enc_send_workers_, &parity_workers_})
            for (auto& t : *v) if (t.joinable()) t.join();
        source_workers_.clear(); enc_recv_workers_.clear(); enc_recv_part_workers_.clear();
        enc_send_workers_.clear(); parity_workers_.clear();
    }

public:
    void skip_source_batch(int sid, uint64_t batch) {
        task_total_.fetch_add(1, std::memory_order_acq_rel);
        task_encode_total_.fetch_add(1, std::memory_order_acq_rel);
        { std::lock_guard<std::mutex> lk(source_mtx_); source_q_.push({sid, 0, 0, 0, batch, true}); }
        source_cv_.notify_one();
    }

    void submit_source_with_batch(int sid, uintptr_t data, uintptr_t mirror, size_t bs, uint64_t batch) {
        task_total_.fetch_add(1, std::memory_order_acq_rel);
        task_encode_total_.fetch_add(1, std::memory_order_acq_rel);
        { std::lock_guard<std::mutex> lk(source_mtx_); source_q_.push({sid, data, mirror, bs, batch, false}); }
        source_cv_.notify_one();
    }

    void submit_source(int sid, uintptr_t data, uintptr_t mirror, size_t bs) {
        submit_source_with_batch(sid, data, mirror, bs, 0);
    }

    void submit_enc_recv_with_batch(int sid, uintptr_t recv, uintptr_t p1, uintptr_t p2, size_t bs,
                                    const std::vector<uint8_t>& mask, uint64_t batch) {
        task_total_.fetch_add(1, std::memory_order_acq_rel);
        task_encode_total_.fetch_add(1, std::memory_order_acq_rel);
        { std::lock_guard<std::mutex> lk(enc_recv_mtx_); enc_recv_q_.push({sid, recv, p1, p2, bs, mask, batch}); }
        enc_recv_cv_.notify_one();
    }

    void submit_enc_recv(int sid, uintptr_t recv, uintptr_t p1, uintptr_t p2, size_t bs,
                         const std::vector<uint8_t>& mask) {
        submit_enc_recv_with_batch(sid, recv, p1, p2, bs, mask, 0);
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

    std::vector<int> submit_async_p2_layer_with_batch(
        const std::vector<uintptr_t>& p2_addrs,
        size_t bs,
        uint64_t batch) {
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
                    t.async_batch = batch;
                    {
                        std::lock_guard<std::mutex> order_lk(async_p2_order_mtx_);
                        if (batch == 0) {
                            if (sid >= 0 && sid < (int)async_p2_recv_enq_order_.size())
                                t.async_order = async_p2_recv_enq_order_[(size_t)sid]++;
                        } else {
                            t.async_order = batch;
                        }
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
                    t.async_batch = batch;
                    {
                        std::lock_guard<std::mutex> order_lk(async_p2_order_mtx_);
                        if (batch == 0) {
                            if (sid >= 0 && sid < (int)async_p2_send_enq_order_.size())
                                t.async_order = async_p2_send_enq_order_[(size_t)sid]++;
                        } else {
                            t.async_order = batch;
                        }
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

    std::vector<int> submit_async_p2_layer(
        const std::vector<uintptr_t>& p2_addrs,
        size_t bs) {
        return submit_async_p2_layer_with_batch(p2_addrs, bs, 0);
    }

    std::vector<int> get_p2_route(int sid) const {
        if (sid < 0 || sid >= (int)stripe_info_.size())
            throw std::runtime_error("FRCheck P2 route: invalid stripe id");
        const int role = get_role_for_stripe(sid);
        if (role == (int)StripeRole::ENCODER) {
            const int peer = stripe_info_[(size_t)sid].par_peer_rig;
            return {role, peer, map_save_send_lane_(peer, sid)};
        }
        if (role == (int)StripeRole::PARITY_TARGET) {
            const int peer = stripe_info_[(size_t)sid].enc_peer_rig;
            return {role, peer, map_save_recv_lane_(peer, sid)};
        }
        return {role, -1, -1};
    }

    using AggregateP2Task = std::tuple<
        int, int, std::vector<std::pair<uintptr_t, size_t>>>;

    std::vector<int> submit_aggregate_p2(
        const std::vector<AggregateP2Task>& send_tasks,
        const std::vector<AggregateP2Task>& recv_tasks,
        uint64_t generation) {
        if (!aggregate_p2_threads_.empty() ||
            aggregate_p2_done_.load(std::memory_order_acquire) <
                aggregate_p2_total_.load(std::memory_order_acquire)) {
            throw std::runtime_error("FRCheck aggregate P2: previous generation is still active");
        }
        if (generation == 0 || generation <= aggregate_p2_generation_)
            throw std::runtime_error("FRCheck aggregate P2: generation must increase");
        aggregate_p2_generation_ = generation;
        aggregate_p2_error_ = nullptr;
        aggregate_p2_done_.store(0, std::memory_order_release);
        const int total = static_cast<int>(send_tasks.size() + recv_tasks.size());
        aggregate_p2_total_.store(total, std::memory_order_release);

        auto launch = [this, generation](
                bool is_send, int peer, int lane,
                std::vector<std::pair<uintptr_t, size_t>> segments) {
            aggregate_p2_threads_.emplace_back([
                    this, generation, is_send, peer, lane,
                    segments = std::move(segments)]() {
                try {
                    FRCheckRdmaChannel* ch = get_channel_by_lane_(peer, lane);
                    if (!ch)
                        throw std::runtime_error(
                            "FRCheck aggregate P2: missing channel peer=" +
                            std::to_string(peer) + " lane=" + std::to_string(lane));
                    size_t size = 0;
                    for (const auto& segment : segments) {
                        if (segment.second > std::numeric_limits<size_t>::max() - size)
                            throw std::runtime_error("FRCheck aggregate P2: segment size overflow");
                        size += segment.second;
                    }
                    const uint64_t tag = make_channel_tag_(5, lane, generation);
                    record_save_net_start_(frcheck_now_us());
                    if (is_send) {
                        _wait_if_paused();
                        auto wait_cb = [this]() { this->_async_rdma_begin(); };
                        auto done_cb = [this]() { this->_async_rdma_end(); };
                        if (shared_lane_)
                            ch->send_tagged_segments(tag, segments, size, wait_cb, done_cb);
                        else
                            ch->send_segments(segments, size, wait_cb, done_cb);
                    } else if (shared_lane_) {
                        const size_t got = ch->recv_tagged_segments(tag, segments, size);
                        if (got != size)
                            throw std::runtime_error("FRCheck aggregate P2: short tagged receive");
                    } else {
                        const size_t got = ch->recv_segments(segments, size);
                        if (got != size)
                            throw std::runtime_error("FRCheck aggregate P2: short receive");
                    }
                    record_save_net_end_(frcheck_now_us());
                } catch (...) {
                    std::lock_guard<std::mutex> lk(aggregate_p2_mtx_);
                    if (!aggregate_p2_error_) aggregate_p2_error_ = std::current_exception();
                }
                aggregate_p2_done_.fetch_add(1, std::memory_order_acq_rel);
                aggregate_p2_cv_.notify_all();
            });
        };
        try {
            for (const auto& task : recv_tasks)
                launch(false, std::get<0>(task), std::get<1>(task), std::get<2>(task));
            for (const auto& task : send_tasks)
                launch(true, std::get<0>(task), std::get<1>(task), std::get<2>(task));
        } catch (...) {
            abort_all_channels_();
            for (auto& thread : aggregate_p2_threads_)
                if (thread.joinable()) thread.join();
            aggregate_p2_threads_.clear();
            aggregate_p2_total_.store(0, std::memory_order_release);
            aggregate_p2_done_.store(0, std::memory_order_release);
            throw;
        }
        return {static_cast<int>(send_tasks.size()), static_cast<int>(recv_tasks.size())};
    }

    void wait_aggregate_p2(int timeout_seconds = 300) {
        const int total = aggregate_p2_total_.load(std::memory_order_acquire);
        if (total > 0) {
            std::unique_lock<std::mutex> lk(aggregate_p2_mtx_);
            if (!aggregate_p2_cv_.wait_for(
                    lk, std::chrono::seconds(std::max(1, timeout_seconds)), [this, total] {
                        return aggregate_p2_done_.load(std::memory_order_acquire) >= total;
                    })) {
                const int done = aggregate_p2_done_.load(std::memory_order_acquire);
                lk.unlock();
                abort_all_channels_();
                for (auto& thread : aggregate_p2_threads_)
                    if (thread.joinable()) thread.join();
                aggregate_p2_threads_.clear();
                throw std::runtime_error(
                    "FRCheck aggregate P2 timed out: done=" +
                    std::to_string(done) + "/" + std::to_string(total));
            }
        }
        for (auto& thread : aggregate_p2_threads_)
            if (thread.joinable()) thread.join();
        aggregate_p2_threads_.clear();
        std::exception_ptr error;
        {
            std::lock_guard<std::mutex> lk(aggregate_p2_mtx_);
            error = aggregate_p2_error_;
            aggregate_p2_error_ = nullptr;
        }
        if (error) std::rethrow_exception(error);
        if (aggregate_p2_done_.load(std::memory_order_acquire) != total)
            throw std::runtime_error("FRCheck aggregate P2 completion count mismatch");
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
            std::fill(async_p2_send_batch_run_order_.begin(), async_p2_send_batch_run_order_.end(), 1);
            std::fill(async_p2_recv_batch_run_order_.begin(), async_p2_recv_batch_run_order_.end(), 1);
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
        encode_source_batch_run_order_.assign(encode_source_batch_run_order_.size(), 1);
        encode_recv_batch_run_order_.assign(encode_recv_batch_run_order_.size(), 1);
        { std::lock_guard<std::mutex> lk(source_mtx_);   while (!source_q_.empty()) source_q_.pop(); }
        { std::lock_guard<std::mutex> lk(enc_recv_mtx_); while (!enc_recv_q_.empty()) enc_recv_q_.pop(); }
        { std::lock_guard<std::mutex> lk(enc_recv_part_mtx_); while (!enc_recv_part_q_.empty()) enc_recv_part_q_.pop(); }
        { std::lock_guard<std::mutex> lk(enc_send_mtx_); while (!enc_send_q_.empty()) enc_send_q_.pop(); }
        { std::lock_guard<std::mutex> lk(parity_mtx_);   while (!parity_q_.empty()) parity_q_.pop(); }
    }

    void reset_encode_layer() {
        task_done_.store(0, std::memory_order_release);
        task_total_.store(0, std::memory_order_release);
        task_encode_done_.store(0, std::memory_order_release);
        task_encode_total_.store(0, std::memory_order_release);
        encode_source_batch_run_order_.assign(encode_source_batch_run_order_.size(), 1);
        encode_recv_batch_run_order_.assign(encode_recv_batch_run_order_.size(), 1);
        { std::lock_guard<std::mutex> lk(source_mtx_);   while (!source_q_.empty()) source_q_.pop(); }
        { std::lock_guard<std::mutex> lk(enc_recv_mtx_); while (!enc_recv_q_.empty()) enc_recv_q_.pop(); }
        {
            std::lock_guard<std::mutex> lk(enc_send_mtx_);
            std::queue<EncSendTask> keep;
            while (!enc_send_q_.empty()) {
                EncSendTask t = std::move(enc_send_q_.front());
                enc_send_q_.pop();
                if (t.parity_send_only) keep.push(std::move(t));
            }
            enc_send_q_.swap(keep);
        }
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
        // Block new async P2 RDMA before PP communication starts.
        // Already-started RDMA is allowed to drain in the background so PP does
        // not wait at the communication boundary.
        int prev = async_p2p_pause_count_.fetch_add(1, std::memory_order_acq_rel);
        if (debug_)
            std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                      << " inc_pause_async_p2p prev=" << prev
                      << " -> paused new async RDMA" << std::endl;
    }

    void dec_pause_async_p2p() {
        int current = async_p2p_pause_count_.load(std::memory_order_acquire);
        while (current > 0) {
            if (async_p2p_pause_count_.compare_exchange_weak(
                    current, current - 1,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                if (debug_)
                    std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                              << " dec_pause_async_p2p prev=" << current << std::endl;
                return;
            }
        }
        if (debug_)
            std::cerr << "[FRCHECK-DEBUG] rank " << rank_in_group_
                      << " dec_pause_async_p2p ignored underflow" << std::endl;
    }

    void wait_mirror_completion() {
        stop_mirror_worker_();
        mirror_idle_ = true;
    }

    // ---- Recovery worker tasks ----
    struct RecoveryBatchState {
        int expected = 0;
        int done = 0;
        bool closed = false;
        bool low_priority = false;
        std::string error;
        uint64_t batch_start_us = 0;
        uint64_t decoder_decode_done_us = 0;
        uint64_t decoder_send_done_us = 0;
        uint64_t failed_delivered_us = 0;
    };

    struct RecoveryHelperTask {
        uint64_t batch_id = 0;
        int stripe_id = 0;
        size_t block_size = 0;
        uintptr_t helper_block = 0;
        int decoder_rig = -1;
    };

    struct RecoveryDecoderTask {
        uint64_t batch_id = 0;
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
        int completion_count = 1;
    };

    struct RecoveryDecoderSendTask {
        uint64_t batch_id = 0;
        int stripe_id = 0;
        size_t block_size = 0;
        uintptr_t recovered_buf = 0;
        int failed_rig = -1;
    };

    struct RecoveryFailedTask {
        uint64_t batch_id = 0;
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

    bool recovery_batch_is_low_priority_(uint64_t batch_id) {
        std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
        auto it = recovery_batches_.find(batch_id);
        return it != recovery_batches_.end() && it->second.low_priority;
    }

    void send_recovery_to_peer_(int peer_rig, int stripe_id, uintptr_t addr,
                                size_t size, uint64_t batch_id, int tag_kind) {
        if (!recovery_batch_is_low_priority_(batch_id)) {
            send_to_peer(peer_rig, stripe_id, addr, size, batch_id, tag_kind);
            return;
        }

        struct AsyncRdmaScope {
            FRCheckNative* owner;
            explicit AsyncRdmaScope(FRCheckNative* native) : owner(native) {
                const uint64_t wait_start = frcheck_now_us();
                owner->_wait_if_paused();
                owner->_async_rdma_begin();
                const uint64_t waited = frcheck_now_us() - wait_start;
                owner->recovery_low_priority_pause_wait_us_.fetch_add(
                    waited, std::memory_order_relaxed);
                owner->recovery_low_priority_send_tasks_.fetch_add(
                    1, std::memory_order_relaxed);
                if (owner->debug_ && waited > 0) {
                    std::cerr << "[FRCHECK-DEBUG] rank " << owner->rank_in_group_
                              << " recovery parity send pause_wait_us=" << waited
                              << std::endl;
                }
            }
            ~AsyncRdmaScope() { owner->_async_rdma_end(); }
        } scope(this);
        send_to_peer(peer_rig, stripe_id, addr, size, batch_id, tag_kind);
    }

    void execute_recovery_helper_(const RecoveryHelperTask& task) {
        uint64_t t0 = frcheck_now_us();
        record_recovery_net_start_(t0);
        send_recovery_to_peer_(task.decoder_rig, task.stripe_id,
                               task.helper_block, task.block_size, task.batch_id, 3);
        uint64_t t1 = frcheck_now_us();
        record_recovery_net_end_(t1);
        recovery_helper_send_us_.fetch_add(t1 - t0, std::memory_order_relaxed);
        recovery_helper_tasks_.fetch_add(1, std::memory_order_relaxed);
    }

    void execute_recovery_decoder_(const RecoveryDecoderTask& task) {
        uint64_t t_recv = frcheck_now_us();
        record_recovery_net_start_(t_recv);
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
                        task.helper_recv_bufs[hi], task.block_size, task.batch_id, 3);
                } catch (...) {
                    recv_errors[hi] = std::current_exception();
                }
            });
        }
        for (auto& t : recv_threads) t.join();
        for (size_t hi = 0; hi < recv_errors.size(); ++hi) {
            if (recv_errors[hi]) std::rethrow_exception(recv_errors[hi]);
        }
        uint64_t t_recv_done = frcheck_now_us();
        record_recovery_net_end_(t_recv_done);
        recovery_decoder_recv_us_.fetch_add(t_recv_done - t_recv, std::memory_order_relaxed);

        int k = n_ - 2;
        std::vector<uintptr_t> survivor_addrs;
        survivor_addrs.push_back(task.self_block);
        for (uintptr_t addr : task.helper_recv_bufs)
            survivor_addrs.push_back(addr);

        if (!task.dual_failure) {
            if (task.recovered_bufs.empty())
                throw std::runtime_error("FRCheck recovery: missing recovered buffer");
            uint64_t t_decode = frcheck_now_us();
            record_recovery_decode_start_(t_decode);
            submit_stripe_decode(
                k, task.survivor_positions, task.failed_pos,
                survivor_addrs, task.recovered_bufs[0], task.block_size);
            uint64_t t_decode_done = frcheck_now_us();
            recovery_batch_record_event_(task.batch_id, RecoveryBatchEvent::DECODER_DECODE, t_decode_done);
            record_recovery_decode_end_(t_decode_done);
            recovery_decoder_decode_us_.fetch_add(t_decode_done - t_decode, std::memory_order_relaxed);
            if (task.failed_rigs.empty())
                throw std::runtime_error("FRCheck recovery: missing failed rig");
            enqueue_recovery_decoder_send_(
                task.batch_id, task.stripe_id, task.block_size,
                task.recovered_bufs[0], task.failed_rigs[0]);
        } else {
            if (task.failed_positions.empty() || task.failed_positions.size() > 2 ||
                task.recovered_bufs.size() != task.failed_positions.size() ||
                task.failed_rigs.size() != task.failed_positions.size()) {
                throw std::runtime_error("FRCheck recovery: invalid filtered dual outputs");
            }
            uint64_t t_decode = frcheck_now_us();
            record_recovery_decode_start_(t_decode);
            submit_stripe_decode_batch(
                k, task.survivor_positions, task.failed_positions,
                survivor_addrs, task.recovered_bufs, task.block_size);
            uint64_t t_decode_done = frcheck_now_us();
            recovery_batch_record_event_(
                task.batch_id, RecoveryBatchEvent::DECODER_DECODE, t_decode_done);
            record_recovery_decode_end_(t_decode_done);
            recovery_decoder_decode_us_.fetch_add(
                t_decode_done - t_decode, std::memory_order_relaxed);
            for (size_t slot = 0; slot < task.failed_positions.size(); ++slot) {
                enqueue_recovery_decoder_send_(
                    task.batch_id, task.stripe_id, task.block_size,
                    task.recovered_bufs[slot], task.failed_rigs[slot]);
            }
        }
        recovery_decoder_tasks_.fetch_add(1, std::memory_order_relaxed);
    }

    void enqueue_recovery_decoder_send_(
        uint64_t batch_id,
        int stripe_id,
        size_t block_size,
        uintptr_t recovered_buf,
        int failed_rig) {
        RecoveryDecoderSendTask task;
        task.batch_id = batch_id;
        task.stripe_id = stripe_id;
        task.block_size = block_size;
        task.recovered_buf = recovered_buf;
        task.failed_rig = failed_rig;
        {
            std::lock_guard<std::mutex> lk(decoder_send_mtx_);
            decoder_send_q_.push(std::move(task));
        }
        decoder_send_cv_.notify_one();
    }

    void execute_recovery_decoder_send_(const RecoveryDecoderSendTask& task) {
        uint64_t t_send = frcheck_now_us();
        record_recovery_net_start_(t_send);
        send_recovery_to_peer_(task.failed_rig, task.stripe_id,
                               task.recovered_buf, task.block_size, task.batch_id, 4);
        uint64_t t_send_done = frcheck_now_us();
        recovery_batch_record_event_(task.batch_id, RecoveryBatchEvent::DECODER_SEND, t_send_done);
        record_recovery_net_end_(t_send_done);
        recovery_decoder_send_us_.fetch_add(t_send_done - t_send, std::memory_order_relaxed);
    }

    void execute_recovery_failed_(const RecoveryFailedTask& task) {
        uint64_t t_recv = frcheck_now_us();
        record_recovery_net_start_(t_recv);
        recv_from_peer(task.decoder_rig, task.stripe_id,
                       task.recv_buf, task.block_size, task.batch_id, 4);
        uint64_t t_recv_done = frcheck_now_us();
        record_recovery_net_end_(t_recv_done);
        recovery_failed_recv_us_.fetch_add(t_recv_done - t_recv, std::memory_order_relaxed);
        if (task.store_to_layer && task.layer_buf != 0 && task.ncopy > 0) {
            uint64_t t_copy = frcheck_now_us();
            std::memcpy(
                reinterpret_cast<void*>(task.layer_buf + task.layer_offset),
                reinterpret_cast<const void*>(task.recv_buf),
                task.ncopy);
            recovery_failed_copy_us_.fetch_add(frcheck_now_us() - t_copy, std::memory_order_relaxed);
        }
        recovery_batch_record_event_(
            task.batch_id, RecoveryBatchEvent::FAILED_DELIVERED, frcheck_now_us());
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
            helper_active_.fetch_add(1, std::memory_order_acq_rel);
            try {
                execute_recovery_helper_(task);
            } catch (const std::exception& e) {
                helper_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                std::cerr << "FRCheck helper_worker: " << e.what() << std::endl;
                recovery_batch_mark_error_(task.batch_id, "FRCheck helper_worker: " + std::string(e.what()));
                continue;
            } catch (...) {
                helper_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                recovery_batch_mark_error_(task.batch_id, "FRCheck helper_worker: unknown exception");
                continue;
            }
            helper_active_.fetch_sub(1, std::memory_order_acq_rel);
            pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
            recovery_batch_mark_done_(task.batch_id);
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
            decoder_active_.fetch_add(1, std::memory_order_acq_rel);
            try {
                execute_recovery_decoder_(task);
            } catch (const std::exception& e) {
                decoder_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(task.completion_count, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                std::cerr << "FRCheck decoder_worker: " << e.what() << std::endl;
                recovery_batch_mark_error_(task.batch_id, "FRCheck decoder_worker: " + std::string(e.what()), task.completion_count);
                continue;
            } catch (...) {
                decoder_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(task.completion_count, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                recovery_batch_mark_error_(task.batch_id, "FRCheck decoder_worker: unknown exception", task.completion_count);
                continue;
            }
            decoder_active_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    void decoder_send_worker_loop_() {
        while (!recovery_workers_stop_) {
            RecoveryDecoderSendTask task;
            {
                std::unique_lock<std::mutex> lk(decoder_send_mtx_);
                decoder_send_cv_.wait(lk, [this] {
                    return recovery_workers_stop_ || !decoder_send_q_.empty();
                });
                if (recovery_workers_stop_) break;
                task = decoder_send_q_.front();
                decoder_send_q_.pop();
            }
            decoder_send_active_.fetch_add(1, std::memory_order_acq_rel);
            try {
                execute_recovery_decoder_send_(task);
            } catch (const std::exception& e) {
                decoder_send_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                std::cerr << "FRCheck decoder_send_worker: " << e.what() << std::endl;
                recovery_batch_mark_error_(task.batch_id, "FRCheck decoder_send_worker: " + std::string(e.what()));
                continue;
            } catch (...) {
                decoder_send_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                recovery_batch_mark_error_(task.batch_id, "FRCheck decoder_send_worker: unknown exception");
                continue;
            }
            decoder_send_active_.fetch_sub(1, std::memory_order_acq_rel);
            pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
            recovery_batch_mark_done_(task.batch_id);
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
            failed_active_.fetch_add(1, std::memory_order_acq_rel);
            try {
                execute_recovery_failed_(task);
            } catch (const std::exception& e) {
                failed_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                std::cerr << "FRCheck failed_worker: " << e.what() << std::endl;
                recovery_batch_mark_error_(task.batch_id, "FRCheck failed_worker: " + std::string(e.what()));
                continue;
            } catch (...) {
                failed_active_.fetch_sub(1, std::memory_order_acq_rel);
                pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
                if (recovery_workers_stop_) return;
                recovery_batch_mark_error_(task.batch_id, "FRCheck failed_worker: unknown exception");
                continue;
            }
            failed_active_.fetch_sub(1, std::memory_order_acq_rel);
            pending_recovery_chunks_.fetch_sub(1, std::memory_order_acq_rel);
            recovery_batch_mark_done_(task.batch_id);
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
            decoder_send_threads_.emplace_back(&FRCheckNative::decoder_send_worker_loop_, this);
            failed_threads_.emplace_back(&FRCheckNative::failed_worker_loop_, this);
        }
        recovery_workers_inited_ = true;
    }

    void recovery_workers_join_() {
        for (auto& t : helper_threads_) if (t.joinable()) t.join();
        for (auto& t : decoder_threads_) if (t.joinable()) t.join();
        for (auto& t : decoder_send_threads_) if (t.joinable()) t.join();
        for (auto& t : failed_threads_) if (t.joinable()) t.join();
        helper_threads_.clear();
        decoder_threads_.clear();
        decoder_send_threads_.clear();
        failed_threads_.clear();
        recovery_workers_inited_ = false;
    }

    void reset_recovery_batch_profile_() {
        recovery_helper_send_us_.store(0, std::memory_order_relaxed);
        recovery_decoder_recv_us_.store(0, std::memory_order_relaxed);
        recovery_decoder_decode_us_.store(0, std::memory_order_relaxed);
        recovery_decoder_send_us_.store(0, std::memory_order_relaxed);
        recovery_failed_recv_us_.store(0, std::memory_order_relaxed);
        recovery_failed_copy_us_.store(0, std::memory_order_relaxed);
        recovery_net_start_us_.store(0, std::memory_order_relaxed);
        recovery_net_end_us_.store(0, std::memory_order_relaxed);
        recovery_decode_start_us_.store(0, std::memory_order_relaxed);
        recovery_decode_end_us_.store(0, std::memory_order_relaxed);
        recovery_skipped_stripes_.store(0, std::memory_order_relaxed);
        recovery_helper_tasks_.store(0, std::memory_order_relaxed);
        recovery_decoder_tasks_.store(0, std::memory_order_relaxed);
        recovery_failed_tasks_.store(0, std::memory_order_relaxed);
    }

public:
    py::dict get_recovery_batch_milestones(uint64_t batch_id) {
        RecoveryBatchState state;
        {
            std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
            auto it = recovery_batches_.find(batch_id);
            if (it == recovery_batches_.end())
                throw std::runtime_error("FRCheck: unknown recovery batch id");
            if (!it->second.closed || it->second.done < it->second.expected)
                throw std::runtime_error("FRCheck: recovery batch is not complete");
            state = it->second;
            recovery_batches_.erase(it);
        }
        const uint64_t now_us = frcheck_now_us();
        py::dict result;
        result["batch_start_us"] = static_cast<double>(state.batch_start_us);
        result["now_us"] = static_cast<double>(now_us);
        result["decoder_decode_done_us"] = static_cast<double>(state.decoder_decode_done_us);
        result["decoder_send_done_us"] = static_cast<double>(state.decoder_send_done_us);
        result["failed_delivered_us"] = static_cast<double>(state.failed_delivered_us);
        result["decoder_decode_offset_s"] = event_offset_s_(
            state.batch_start_us, state.decoder_decode_done_us);
        result["decoder_send_offset_s"] = event_offset_s_(
            state.batch_start_us, state.decoder_send_done_us);
        result["failed_delivered_offset_s"] = event_offset_s_(
            state.batch_start_us, state.failed_delivered_us);
        return result;
    }

    py::dict get_recovery_batch_timing_stats() const {
        py::dict result;
        const double helper_send_s = static_cast<double>(
            recovery_helper_send_us_.load(std::memory_order_relaxed)) / 1e6;
        const double decoder_recv_s = static_cast<double>(
            recovery_decoder_recv_us_.load(std::memory_order_relaxed)) / 1e6;
        const double decoder_send_s = static_cast<double>(
            recovery_decoder_send_us_.load(std::memory_order_relaxed)) / 1e6;
        const double failed_recv_s = static_cast<double>(
            recovery_failed_recv_us_.load(std::memory_order_relaxed)) / 1e6;
        const double failed_copy_s = static_cast<double>(
            recovery_failed_copy_us_.load(std::memory_order_relaxed)) / 1e6;
        const uint64_t net_start = recovery_net_start_us_.load(std::memory_order_relaxed);
        const uint64_t net_end = recovery_net_end_us_.load(std::memory_order_relaxed);
        const uint64_t decode_start = recovery_decode_start_us_.load(std::memory_order_relaxed);
        const uint64_t decode_end = recovery_decode_end_us_.load(std::memory_order_relaxed);
        result["helper_send_s"] = helper_send_s;
        result["decoder_recv_s"] = decoder_recv_s;
        result["decoder_decode_sum_s"] = static_cast<double>(
            recovery_decoder_decode_us_.load(std::memory_order_relaxed)) / 1e6;
        result["decoder_send_s"] = decoder_send_s;
        result["failed_recv_s"] = failed_recv_s;
        result["failed_copy_s"] = failed_copy_s;
        result["net_s"] = (net_start > 0 && net_end > net_start)
            ? static_cast<double>(net_end - net_start) / 1e6
            : 0.0;
        result["decode_s"] = (decode_start > 0 && decode_end > decode_start)
            ? static_cast<double>(decode_end - decode_start) / 1e6
            : 0.0;
        result["net_start_us"] = static_cast<double>(net_start);
        result["net_end_us"] = static_cast<double>(net_end);
        result["decode_start_us"] = static_cast<double>(decode_start);
        result["decode_end_us"] = static_cast<double>(decode_end);
        result["copy_s"] = failed_copy_s;
        result["helper_tasks"] = static_cast<double>(
            recovery_helper_tasks_.load(std::memory_order_relaxed));
        result["decoder_tasks"] = static_cast<double>(
            recovery_decoder_tasks_.load(std::memory_order_relaxed));
        result["failed_tasks"] = static_cast<double>(
            recovery_failed_tasks_.load(std::memory_order_relaxed));
        result["skipped_stripes"] = static_cast<double>(
            recovery_skipped_stripes_.load(std::memory_order_relaxed));
        result["low_priority_pause_wait_s"] = static_cast<double>(
            recovery_low_priority_pause_wait_us_.load(std::memory_order_relaxed)) / 1e6;
        result["low_priority_send_tasks"] = static_cast<double>(
            recovery_low_priority_send_tasks_.load(std::memory_order_relaxed));
        return result;
    }

private:
    enum class RecoveryBatchEvent {
        DECODER_DECODE,
        DECODER_SEND,
        FAILED_DELIVERED,
    };

    static double event_offset_s_(uint64_t start_us, uint64_t event_us) {
        return (start_us > 0 && event_us >= start_us)
            ? static_cast<double>(event_us - start_us) / 1e6
            : 0.0;
    }

    void recovery_batch_record_event_(
        uint64_t batch_id, RecoveryBatchEvent event, uint64_t timestamp_us) {
        std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
        auto it = recovery_batches_.find(batch_id);
        if (it == recovery_batches_.end()) return;
        uint64_t* target = nullptr;
        if (event == RecoveryBatchEvent::DECODER_DECODE)
            target = &it->second.decoder_decode_done_us;
        else if (event == RecoveryBatchEvent::DECODER_SEND)
            target = &it->second.decoder_send_done_us;
        else
            target = &it->second.failed_delivered_us;
        *target = std::max(*target, timestamp_us);
    }

    void discard_recovery_batch_milestones_(uint64_t batch_id) {
        std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
        recovery_batches_.erase(batch_id);
    }

    void print_recovery_batch_profile_() {}

    void ensure_recovery_batch_exists_(uint64_t batch_id) {
        std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
        if (recovery_batches_.find(batch_id) == recovery_batches_.end())
            throw std::runtime_error("FRCheck: unknown recovery batch id");
    }

    void recovery_batch_add_expected_(uint64_t batch_id, int count) {
        if (count <= 0) return;
        {
            std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
            auto it = recovery_batches_.find(batch_id);
            if (it == recovery_batches_.end())
                throw std::runtime_error("FRCheck: unknown recovery batch id");
            it->second.expected += count;
        }
        recovery_batch_cv_.notify_all();
    }

    void recovery_batch_mark_error_(
        uint64_t batch_id, const std::string& error, int completion_count = 1) {
        {
            std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
            auto it = recovery_batches_.find(batch_id);
            if (it != recovery_batches_.end()) {
                if (it->second.error.empty())
                    it->second.error = error;
                it->second.done += std::max(1, completion_count);
            }
        }
        recovery_batch_cv_.notify_all();
    }

    void recovery_batch_mark_done_(uint64_t batch_id) {
        {
            std::lock_guard<std::mutex> lk(recovery_batch_mtx_);
            auto it = recovery_batches_.find(batch_id);
            if (it != recovery_batches_.end())
                it->second.done += 1;
        }
        recovery_batch_cv_.notify_all();
    }

    void reset_recovery_completion_() {
        pending_recovery_chunks_.store(0, std::memory_order_release);
        helper_active_.store(0, std::memory_order_release);
        decoder_active_.store(0, std::memory_order_release);
        failed_active_.store(0, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(helper_mtx_);
            while (!helper_q_.empty()) helper_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lk(decoder_mtx_);
            while (!decoder_q_.empty()) decoder_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lk(decoder_send_mtx_);
            while (!decoder_send_q_.empty()) decoder_send_q_.pop();
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
        rdma_ctx_ = ibv_open_device(find_rdma_device_by_ip(
            my_ip_, devs, ndev, {"FRCHECK", "ECLATIN"}));
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

        {
            std::lock_guard<std::mutex> lk(buf_mtx_);
            for (auto& kv : registered_bufs_) {
                if (kv.second.mr) ibv_dereg_mr(kv.second.mr);
            }
            registered_bufs_.clear();
            recovery_buffer_addrs_.clear();
        }

        if (rdma_pd_) { ibv_dealloc_pd(rdma_pd_); rdma_pd_ = nullptr; }
        if (rdma_ctx_) { ibv_close_device(rdma_ctx_); rdma_ctx_ = nullptr; }
        n_connected_ = 0;
    }

    // ---- TCP helpers ----
    static void configure_control_socket_(int fd) {
        int enabled = 1;
        if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) != 0) {
            throw std::runtime_error(
                "FRCheck: failed to enable TCP_NODELAY: " + std::string(std::strerror(errno)));
        }
    }

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
            if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                configure_control_socket_(fd);
                return fd;
            }
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
                try {
                    configure_control_socket_(fd);
                } catch (const std::exception& e) {
                    std::cerr << "[FRCheck RDMA] accept: " << e.what() << std::endl;
                    close(fd);
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
    FRCheckRdmaChannel* get_channel_by_lane_(int peer_rg, int lane_id) {
        if (peer_rg < 0 || peer_rg >= group_size_ || peer_rg == rank_in_group_)
            return nullptr;
        if (lane_id < 0 || lane_id >= num_lanes_ || num_lanes_ <= 0)
            return nullptr;
        return channels_[peer_rg][lane_id];
    }

    FRCheckRdmaChannel* get_channel_(int peer_rg, int stripe_id) {
        if (stripe_id < 0 || stripe_id >= num_stripes_ || num_lanes_ <= 0)
            return nullptr;
        return get_channel_by_lane_(peer_rg, stripe_id % num_lanes_);
    }

    int map_save_direction_lane_(bool forward, int logical_lane_id) const {
        if (num_lanes_ <= 0 || logical_lane_id < 0)
            return -1;
        if (!lane_direction_split_)
            return logical_lane_id % num_lanes_;
        if (forward)
            return logical_lane_id % std::max(1, save_forward_lanes_);
        return save_forward_lanes_ + (logical_lane_id % std::max(1, save_reverse_lanes_));
    }

    int map_save_send_lane_(int peer_rg, int logical_lane_id) const {
        return map_save_direction_lane_(rank_in_group_ < peer_rg, logical_lane_id);
    }

    int map_save_recv_lane_(int peer_rg, int logical_lane_id) const {
        return map_save_direction_lane_(peer_rg < rank_in_group_, logical_lane_id);
    }

    FRCheckRdmaChannel* get_save_send_channel_(int peer_rg, int logical_lane_id) {
        return get_channel_by_lane_(peer_rg, map_save_send_lane_(peer_rg, logical_lane_id));
    }

    FRCheckRdmaChannel* get_save_recv_channel_(int peer_rg, int logical_lane_id) {
        return get_channel_by_lane_(peer_rg, map_save_recv_lane_(peer_rg, logical_lane_id));
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

    // Channels: index by peer rank_in_group and lane (lane_id = stripe_id % num_lanes_).
    int num_stripes_ = 0;
    int num_lanes_ = 0;
    int save_forward_lanes_ = 0;
    int save_reverse_lanes_ = 0;
    bool shared_lane_ = false;
    bool lane_direction_split_ = false;

    // Tag layout for shared-lane multiplexing: kind|sid|batch.
    //   kind: 0 = sync source->encoder, 1 = async P2 encoder->parity target,
    //         2 = layer-level source payload.
    static uint64_t make_channel_tag_(int kind, int sid, uint64_t batch) {
        return ((uint64_t)(kind & 0xF) << 60)
             | ((uint64_t)(uint32_t)sid << 32)
             | (uint64_t)(uint32_t)batch;
    }
    std::vector<std::vector<FRCheckRdmaChannel*>> channels_;
    std::vector<std::unique_ptr<FRCheckRdmaChannel>> channel_owners_;
    bool require_registered_mr_ = true;

    // Registered buffers (shared across channels)
    std::map<uintptr_t, RdmaBuffer> registered_bufs_;
    std::set<uintptr_t> recovery_buffer_addrs_;
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
    bool legacy_recovery_batch_active_ = false;
    uint64_t legacy_recovery_batch_id_ = 0;
    static constexpr uint64_t kRecoveryBatchCycleBits = 20;
    static constexpr uint64_t kRecoveryGenerationMax = (1ULL << 12) - 1;
    uint64_t recovery_generation_ = 0;
    uint64_t next_recovery_batch_id_ = 1;
    std::unordered_map<uint64_t, RecoveryBatchState> recovery_batches_;
    std::mutex recovery_batch_mtx_;
    std::condition_variable recovery_batch_cv_;
    bool encoding_batch_active_ = false;
    bool recovery_workers_inited_ = false;
    std::atomic<bool> recovery_workers_stop_{false};
    std::queue<RecoveryHelperTask> helper_q_;
    std::mutex helper_mtx_;
    std::condition_variable helper_cv_;
    std::queue<RecoveryDecoderTask> decoder_q_;
    std::mutex decoder_mtx_;
    std::condition_variable decoder_cv_;
    std::queue<RecoveryDecoderSendTask> decoder_send_q_;
    std::mutex decoder_send_mtx_;
    std::condition_variable decoder_send_cv_;
    std::queue<RecoveryFailedTask> failed_q_;
    std::mutex failed_mtx_;
    std::condition_variable failed_cv_;
    std::vector<std::thread> helper_threads_;
    std::vector<std::thread> decoder_threads_;
    std::vector<std::thread> decoder_send_threads_;
    std::vector<std::thread> failed_threads_;
    std::atomic<int> pending_recovery_chunks_{0};
    std::atomic<int> helper_active_{0};
    std::atomic<int> decoder_active_{0};
    std::atomic<int> decoder_send_active_{0};
    std::atomic<int> failed_active_{0};
    std::atomic<uint64_t> recovery_helper_send_us_{0};
    std::atomic<uint64_t> recovery_decoder_recv_us_{0};
    std::atomic<uint64_t> recovery_decoder_decode_us_{0};
    std::atomic<uint64_t> recovery_decoder_send_us_{0};
    std::atomic<uint64_t> recovery_failed_recv_us_{0};
    std::atomic<uint64_t> recovery_failed_copy_us_{0};
    std::atomic<uint64_t> recovery_net_start_us_{0};
    std::atomic<uint64_t> recovery_net_end_us_{0};
    std::atomic<uint64_t> recovery_decode_start_us_{0};
    std::atomic<uint64_t> recovery_decode_end_us_{0};
    std::atomic<int> recovery_skipped_stripes_{0};
    std::atomic<int> recovery_helper_tasks_{0};
    std::atomic<int> recovery_decoder_tasks_{0};
    std::atomic<int> recovery_failed_tasks_{0};
    std::atomic<uint64_t> recovery_low_priority_pause_wait_us_{0};
    std::atomic<int> recovery_low_priority_send_tasks_{0};

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
    const std::vector<RsEncodeJob>* rs_pool_shared_jobs_ = nullptr;
    std::vector<RsPoolSpan> rs_pool_plan_spans_;
    std::array<size_t, kRsPoolWorkers> rs_pool_worker_plan_begin_{};
    std::array<size_t, kRsPoolWorkers> rs_pool_worker_plan_count_{};
    std::array<std::atomic<uint64_t>, kRsPoolWorkers> rs_pool_worker_elapsed_us_{};
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
        .def("register_recovery_buffer", &FRCheckNative::register_recovery_buffer,
             py::arg("addr"), py::arg("size"))
        .def("unregister_buffer", &FRCheckNative::unregister_buffer,
             py::arg("addr"))
        .def("clear_recovery_buffers", &FRCheckNative::clear_recovery_buffers)

        // GDR capability
        .def_static("gdr_available", &FRCheckNative::gdr_available)

        // Stripe decode (hardware recovery)
        .def("submit_stripe_decode_batch", &FRCheckNative::submit_stripe_decode_batch,
             py::arg("k"),
             py::arg("survivor_positions"),
             py::arg("lost_positions"),
             py::arg("survivor_addrs"),
             py::arg("recovered_addrs"),
             py::arg("block_size"))
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
             py::arg("batch_id") = 0, py::arg("tag_kind") = 3,
             py::call_guard<py::gil_scoped_release>())
        .def("recv_from_peer", &FRCheckNative::recv_from_peer,
             py::arg("peer_rig"), py::arg("stripe_id"),
             py::arg("addr"), py::arg("size"),
             py::arg("batch_id") = 0, py::arg("tag_kind") = 3,
             py::call_guard<py::gil_scoped_release>())
        .def("send_layer_to_peer", &FRCheckNative::send_layer_to_peer,
             py::arg("peer_rig"), py::arg("addr"), py::arg("size"), py::arg("batch_id"),
             py::arg("lane_id") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("send_layer_blocks_to_peer", &FRCheckNative::send_layer_blocks_to_peer,
             py::arg("peer_rig"), py::arg("mirror_base"), py::arg("block_indices"),
             py::arg("block_size"), py::arg("batch_id"), py::arg("lane_id") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("get_max_send_sge", &FRCheckNative::get_max_send_sge)
        .def("prepare_layer_recv_from_peer", &FRCheckNative::prepare_layer_recv_from_peer,
             py::arg("peer_rig"), py::arg("addr"), py::arg("capacity"), py::arg("batch_id"),
             py::arg("lane_id") = 0)
        .def("wait_prepared_layer_recv_from_peer", &FRCheckNative::wait_prepared_layer_recv_from_peer,
             py::arg("peer_rig"), py::arg("batch_id"), py::arg("lane_id") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("recv_layer_from_peer", &FRCheckNative::recv_layer_from_peer,
             py::arg("peer_rig"), py::arg("addr"), py::arg("capacity"), py::arg("batch_id"),
             py::arg("lane_id") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("simple_exchange_with_peer", &FRCheckNative::simple_exchange_with_peer,
             py::arg("peer_rig"), py::arg("send_addr"), py::arg("recv_addr"),
             py::arg("total_size"), py::arg("lanes"), py::arg("send_batch_id"),
             py::arg("recv_batch_id"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_require_registered_mr", &FRCheckNative::set_require_registered_mr,
             py::arg("require"))
        .def("set_debug", &FRCheckNative::set_debug,
             py::arg("debug"))
        .def("mirror_layer", &FRCheckNative::mirror_layer,
             py::arg("gpu_addr"), py::arg("cpu_addr"), py::arg("size"))
        .def("encode_layer_stripes", &FRCheckNative::encode_layer_stripes,
             py::arg("stripe_ids"), py::arg("data_addrs"), py::arg("p1_addrs"),
             py::arg("p2_addrs"), py::arg("block_size"),
             py::call_guard<py::gil_scoped_release>())
        .def("encode_layer_stripes_batch", &FRCheckNative::encode_layer_stripes_batch,
             py::arg("stripe_ids"), py::arg("data_addrs"), py::arg("p1_addrs"),
             py::arg("p2_addrs"), py::arg("block_sizes"),
             py::call_guard<py::gil_scoped_release>())
        .def("reset_ft_timing_stats", &FRCheckNative::reset_ft_timing_stats)
        .def("get_ft_timing_stats", &FRCheckNative::get_ft_timing_stats,
             "Return per-rank FRCheck timing counters including mirror D2H busy time")

        // Hardware recovery batch pipeline
        .def("init_recovery_plans", &FRCheckNative::init_recovery_plans,
             py::arg("failed_nodes_1based"))
        .def("reset_recovery_generation", &FRCheckNative::reset_recovery_generation,
             "Start a drained recovery generation while preserving plans and registered buffers")
        .def("begin_recovery_batch", &FRCheckNative::begin_recovery_batch,
             py::arg("low_priority") = false)
        .def("end_recovery_batch", &FRCheckNative::end_recovery_batch,
             py::arg("batch_id"))
        .def("wait_recovery_batch_id", &FRCheckNative::wait_recovery_batch_id,
             py::arg("batch_id"), py::call_guard<py::gil_scoped_release>())
        .def("get_recovery_batch_milestones", &FRCheckNative::get_recovery_batch_milestones,
             py::arg("batch_id"),
             "Return and release exact milestones for a completed recovery batch")
        .def("get_recovery_batch_timing_stats", &FRCheckNative::get_recovery_batch_timing_stats,
             "Return timing counters for the most recently completed recovery batch")
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
             py::arg("store_to_layer_buf"),
             py::arg("active") = true)
        .def("submit_recovery_stripe_to_batch", &FRCheckNative::submit_recovery_stripe_to_batch,
             py::arg("batch_id"),
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
             py::arg("store_to_layer_buf"),
             py::arg("active") = true,
             py::arg("target_roles") = std::vector<int>{})
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
        .def("reset_encode_layer", &FRCheckNative::reset_encode_layer)
        .def("submit_source", &FRCheckNative::submit_source,
             py::arg("stripe_id"), py::arg("data_addr"), py::arg("mirror_addr"), py::arg("block_size"))
        .def("submit_source_with_batch", &FRCheckNative::submit_source_with_batch,
             py::arg("stripe_id"), py::arg("data_addr"), py::arg("mirror_addr"),
             py::arg("block_size"), py::arg("batch_id"))
        .def("skip_source_batch", &FRCheckNative::skip_source_batch,
             py::arg("stripe_id"), py::arg("batch_id"))
        .def("submit_enc_recv", &FRCheckNative::submit_enc_recv,
             py::arg("stripe_id"), py::arg("recv_addr"), py::arg("p1_addr"),
             py::arg("p2_addr"), py::arg("block_size"),
             py::arg("source_active_mask") = std::vector<uint8_t>())
        .def("submit_enc_recv_with_batch", &FRCheckNative::submit_enc_recv_with_batch,
             py::arg("stripe_id"), py::arg("recv_addr"), py::arg("p1_addr"),
             py::arg("p2_addr"), py::arg("block_size"),
             py::arg("source_active_mask"), py::arg("batch_id"))
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
        .def("submit_async_p2_layer_with_batch", &FRCheckNative::submit_async_p2_layer_with_batch,
             py::arg("p2_addrs"), py::arg("block_size"), py::arg("batch_id"))
        .def("get_p2_route", &FRCheckNative::get_p2_route, py::arg("stripe_id"))
        .def("submit_aggregate_p2", &FRCheckNative::submit_aggregate_p2,
             py::arg("send_tasks"), py::arg("recv_tasks"), py::arg("generation"))
        .def("wait_aggregate_p2", &FRCheckNative::wait_aggregate_p2,
             py::arg("timeout_seconds") = 300, py::call_guard<py::gil_scoped_release>())
        .def("reset_async_parity", &FRCheckNative::reset_async_parity)
        .def("wait_layer", &FRCheckNative::wait_layer,
             py::call_guard<py::gil_scoped_release>())
        .def("wait_encode_only", &FRCheckNative::wait_encode_only,
             py::call_guard<py::gil_scoped_release>())
        .def("wait_parity_flush", &FRCheckNative::wait_parity_flush,
             py::call_guard<py::gil_scoped_release>())
        .def("inc_pause_async_p2p", &FRCheckNative::inc_pause_async_p2p)
        .def("dec_pause_async_p2p", &FRCheckNative::dec_pause_async_p2p)
        .def("start_mirror_worker", &FRCheckNative::start_mirror_worker)
        .def("wait_mirror_completion", &FRCheckNative::wait_mirror_completion);
}
