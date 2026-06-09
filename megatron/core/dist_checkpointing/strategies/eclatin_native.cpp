#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <boost/asio.hpp>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>

// 64-bit network byte order conversion functions (for large data transfers > 4GB)
inline uint64_t htonll(uint64_t value) {
    // Check if system is little-endian
    static const int num = 1;
    if (*reinterpret_cast<const char*>(&num) == 1) {
        // Little-endian: swap bytes
        return ((static_cast<uint64_t>(htonl(value & 0xFFFFFFFF)) << 32) | 
                htonl(value >> 32));
    } else {
        // Big-endian: no swap needed
        return value;
    }
}

inline uint64_t ntohll(uint64_t value) {
    // ntohll is the same as htonll (symmetric operation)
    return htonll(value);
}

// RDMA headers (ibverbs) - only if RDMA libraries are available
#if RDMA_AVAILABLE
#include <infiniband/verbs.h>
#include "rdma_device_utils.h"
#include <endian.h>
#endif

#include <array>
#include <atomic>
#include <memory>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <map>
#include <sstream>

#include <isa-l/erasure_code.h>
#include <isa-l/raid.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif


namespace {

// Configuration: Enable/disable async CUDA transfers
#ifndef ECLATIN_USE_ASYNC_CUDA
#define ECLATIN_USE_ASYNC_CUDA 1  // 1 = async (default), 0 = sync (fallback)
#endif

#ifndef ECLATIN_NUM_CUDA_STREAMS
#define ECLATIN_NUM_CUDA_STREAMS 4  // Default number of CUDA streams for async transfers
#endif

#ifndef ECLATIN_RANKS_PER_GROUP
#define ECLATIN_RANKS_PER_GROUP 4  // Ranks per ECLATIN group (multi-rank support)
#endif

// ============================================================================
// Connection Manager Interface
// ============================================================================

class IConnectionManager {
public:
    virtual ~IConnectionManager() = default;
    
    // Connection initialization
    virtual void init_connections() = 0;
    
    // Send/receive operations for parity 1
    virtual void send_parity1_send1(const uint8_t* data, size_t size) = 0;
    virtual void send_parity1_send2(const uint8_t* data, size_t size) = 0;
    virtual bool recv_parity1_recv1(uint8_t* buffer, size_t size) = 0;
    virtual bool recv_parity1_recv2(uint8_t* buffer, size_t size) = 0;
    
    // Send/receive operations for parity 2
    virtual void send_parity2_send1(const uint8_t* data, size_t size) = 0;
    virtual void send_parity2_send2(const uint8_t* data, size_t size) = 0;
    virtual bool recv_parity2_recv1(uint8_t* buffer, size_t size) = 0;
    virtual bool recv_parity2_recv2(uint8_t* buffer, size_t size) = 0;
    
    // Load mode operations
    virtual bool recv_load_data(const std::string& socket_name, uint8_t* buffer, size_t size) = 0;
    virtual void send_load_data(const std::string& socket_name, const uint8_t* data, size_t size) = 0;
    
    // Connection status
    virtual bool is_connected() const = 0;
    
    // RDMA-specific methods (no-op for ASIO)
    virtual void register_buffer(uintptr_t addr, size_t size) {}
    virtual void unregister_buffer(uintptr_t addr) {}
    
    // Cleanup
    virtual void cleanup() = 0;
};

// ============================================================================
// RDMA Connection Info (for ibverbs)
// ============================================================================

#if RDMA_AVAILABLE
// RDMA connection info exchanged via TCP (same as gemini_native.cpp)
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

// RDMA connection channel: one QP per connection, control over TCP socket
class RdmaConnectionChannel {
private:
    ibv_context* context_;
    ibv_pd* pd_;
    ibv_cq* send_cq_;
    ibv_cq* recv_cq_;
    ibv_qp* qp_;
    int control_sock_send_;
    int control_sock_recv_;
    std::map<uintptr_t, RdmaBuffer>* registered_buffers_;
    std::mutex* buffer_mutex_;
    std::vector<uint8_t> temp_send_buffer_;
    std::vector<uint8_t> temp_recv_buffer_;
    ibv_mr* temp_send_mr_;
    ibv_mr* temp_recv_mr_;
    int rank_;
    int peer_rank_;
    bool connected_;
    std::mutex send_mutex_;
    std::mutex recv_mutex_;
    std::mutex* shared_control_mutex_;  // optional: serializes TCP handshake when multiple QPs share one socket
    static const size_t TEMP_BUFFER_SIZE = 128ULL * 1024 * 1024;  // 128 MB (reduced from 1GB to avoid RDMA memory limits, matching ecnaive)
    static const size_t CHUNK_SIZE = 64 * 1024 * 1024;
    static const int MAX_WR = 64;
    static const int MAX_BATCH_WR = 32;

    void connect_qp(const RdmaConnInfo& remote_info) {
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = 1;
        attr.pkey_index = 0;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            throw std::runtime_error("ECLATIN RDMA: Failed to transition QP to INIT");
        }
        attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_4096;
        attr.dest_qp_num = remote_info.qp_num;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = 1;
        attr.ah_attr.sl = 0;
        attr.ah_attr.dlid = remote_info.lid;
        memcpy(&attr.ah_attr.grh.dgid, remote_info.gid, 16);
        attr.ah_attr.grh.sgid_index = 1; // GID index 1 for erdma (RoCE v2)
        attr.ah_attr.grh.hop_limit = 64;
        if (ibv_modify_qp(qp_, &attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
            throw std::runtime_error("ECLATIN RDMA: Failed to transition QP to RTR");
        }
        attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        if (ibv_modify_qp(qp_, &attr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
            throw std::runtime_error("ECLATIN RDMA: Failed to transition QP to RTS");
        }
        connected_ = true;
    }

    RdmaConnInfo get_local_conn_info() {
        RdmaConnInfo info{};
        info.qp_num = qp_->qp_num;
        ibv_port_attr port_attr;
        if (ibv_query_port(context_, 1, &port_attr)) throw std::runtime_error("ECLATIN RDMA: Failed to query port");
        info.lid = port_attr.lid;
        ibv_gid gid;
        if (ibv_query_gid(context_, 1, 1, &gid)) throw std::runtime_error("ECLATIN RDMA: Failed to query GID");
        memcpy(info.gid, &gid, 16);
        return info;
    }

    ibv_mr* find_registered_mr(uintptr_t addr, size_t size) {
        std::lock_guard<std::mutex> lock(*buffer_mutex_);
        for (auto& [reg_addr, buf] : *registered_buffers_) {
            if (addr >= reg_addr && (addr + size) <= (reg_addr + buf.size)) return buf.mr;
        }
        return nullptr;
    }

    void poll_completion(ibv_cq* cq, int num_completions) {
        int completed = 0;
        while (completed < num_completions) {
            ibv_wc wc;
            int ret = ibv_poll_cq(cq, 1, &wc);
            if (ret < 0) throw std::runtime_error("ECLATIN RDMA: Failed to poll CQ");
            if (ret > 0) {
                if (wc.status != IBV_WC_SUCCESS) throw std::runtime_error("ECLATIN RDMA: Work completion failed");
                completed++;
            }
        }
    }

    void send_data_chunked(const uint8_t* data, size_t total_size, ibv_mr* mr) {
        size_t remaining = total_size;
        size_t offset = 0;
        while (remaining > 0) {
            size_t chunk_size = std::min(remaining, CHUNK_SIZE);
            size_t chunk_count = (chunk_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
            std::vector<ibv_sge> sges(chunk_count);
            std::vector<ibv_send_wr> wrs(chunk_count);
            for (size_t i = 0; i < chunk_count; ++i) {
                size_t current_size = std::min(CHUNK_SIZE, remaining);
                sges[i].addr = reinterpret_cast<uint64_t>(data + offset);
                sges[i].length = current_size;
                sges[i].lkey = mr->lkey;
                wrs[i].wr_id = static_cast<uint64_t>(i);
                wrs[i].sg_list = &sges[i];
                wrs[i].num_sge = 1;
                wrs[i].opcode = IBV_WR_SEND;
                wrs[i].send_flags = IBV_SEND_SIGNALED;
                wrs[i].next = (i < chunk_count - 1) ? &wrs[i + 1] : nullptr;
                offset += current_size;
                remaining -= current_size;
            }
            ibv_send_wr* bad_wr = nullptr;
            if (ibv_post_send(qp_, &wrs[0], &bad_wr)) throw std::runtime_error("ECLATIN RDMA: Failed to post send");
            poll_completion(send_cq_, static_cast<int>(chunk_count));
        }
    }

    void receive_data_chunked(uint8_t* buffer, size_t total_size, ibv_mr* mr) {
        size_t remaining = total_size;
        size_t offset = 0;
        while (remaining > 0) {
            size_t chunk_count = std::min(remaining, CHUNK_SIZE * MAX_BATCH_WR) / CHUNK_SIZE;
            if (chunk_count == 0) chunk_count = 1;
            std::vector<ibv_sge> sges(chunk_count);
            std::vector<ibv_recv_wr> wrs(chunk_count);
            for (size_t i = 0; i < chunk_count; ++i) {
                size_t current_size = std::min(CHUNK_SIZE, remaining);
                sges[i].addr = reinterpret_cast<uint64_t>(buffer + offset);
                sges[i].length = current_size;
                sges[i].lkey = mr->lkey;
                wrs[i].wr_id = static_cast<uint64_t>(i);
                wrs[i].sg_list = &sges[i];
                wrs[i].num_sge = 1;
                wrs[i].next = (i < chunk_count - 1) ? &wrs[i + 1] : nullptr;
                offset += current_size;
                remaining -= current_size;
            }
            ibv_recv_wr* bad_wr = nullptr;
            if (ibv_post_recv(qp_, &wrs[0], &bad_wr)) throw std::runtime_error("ECLATIN RDMA: Failed to post recv");
            poll_completion(recv_cq_, static_cast<int>(chunk_count));
        }
    }

public:
    RdmaConnectionChannel(ibv_context* context, ibv_pd* pd, ibv_cq* send_cq, ibv_cq* recv_cq,
                           int control_sock_send, int control_sock_recv,
                           std::map<uintptr_t, RdmaBuffer>* registered_buffers, std::mutex* buffer_mutex,
                           int rank, int peer_rank,
                           std::mutex* shared_control_mutex = nullptr)
        : context_(context), pd_(pd), send_cq_(send_cq), recv_cq_(recv_cq), qp_(nullptr),
          control_sock_send_(control_sock_send), control_sock_recv_(control_sock_recv),
          registered_buffers_(registered_buffers), buffer_mutex_(buffer_mutex),
          temp_send_mr_(nullptr), temp_recv_mr_(nullptr), rank_(rank), peer_rank_(peer_rank),
          connected_(false), shared_control_mutex_(shared_control_mutex) {
        ibv_qp_init_attr qp_attr{};
        qp_attr.send_cq = send_cq_;
        qp_attr.recv_cq = recv_cq_;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr = MAX_WR;
        qp_attr.cap.max_recv_wr = MAX_WR;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_ = ibv_create_qp(pd_, &qp_attr);
        if (!qp_) throw std::runtime_error("ECLATIN RDMA: Failed to create QP");
        temp_send_buffer_.resize(TEMP_BUFFER_SIZE);
        temp_recv_buffer_.resize(TEMP_BUFFER_SIZE);
        temp_send_mr_ = ibv_reg_mr(pd_, temp_send_buffer_.data(), TEMP_BUFFER_SIZE,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        temp_recv_mr_ = ibv_reg_mr(pd_, temp_recv_buffer_.data(), TEMP_BUFFER_SIZE,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!temp_send_mr_ || !temp_recv_mr_) throw std::runtime_error("ECLATIN RDMA: Failed to register temp buffers");
    }

    ~RdmaConnectionChannel() {
        if (temp_send_mr_) ibv_dereg_mr(temp_send_mr_);
        if (temp_recv_mr_) ibv_dereg_mr(temp_recv_mr_);
        if (qp_) ibv_destroy_qp(qp_);
    }

    // Helper: receive exactly n bytes, handling partial reads from non-blocking sockets.
    static void recv_all(int sock, void* buf, size_t n) {
        size_t total = 0;
        int retries = 0;
        while (total < n) {
            ssize_t r = recv(sock, static_cast<char*>(buf) + total, n - total, 0);
            if (r > 0) {
                total += r;
                retries = 0;
            } else if (r == 0) {
                throw std::runtime_error("ECLATIN RDMA: connection closed during recv (got "
                    + std::to_string(total) + " of " + std::to_string(n) + " bytes)");
            } else if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                if (++retries > 1000)
                    throw std::runtime_error("ECLATIN RDMA: recv timed out after 1000 retries");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                throw std::runtime_error("ECLATIN RDMA: recv error: " + std::string(std::strerror(errno)));
            }
        }
    }

    void exchange_and_connect(bool we_send_first) {
        RdmaConnInfo local_info = get_local_conn_info();
        RdmaConnInfo remote_info;
        std::memset(&remote_info, 0, sizeof(remote_info));
        int sock = control_sock_send_;
        if (we_send_first) {
            if (send(sock, &local_info, sizeof(local_info), 0) != static_cast<ssize_t>(sizeof(local_info)))
                throw std::runtime_error("ECLATIN RDMA: failed to send local RdmaConnInfo");
            recv_all(sock, &remote_info, sizeof(remote_info));
        } else {
            recv_all(sock, &remote_info, sizeof(remote_info));
            if (send(sock, &local_info, sizeof(local_info), 0) != static_cast<ssize_t>(sizeof(local_info)))
                throw std::runtime_error("ECLATIN RDMA: failed to send local RdmaConnInfo");
        }
        connect_qp(remote_info);
    }

    void send_data(const uint8_t* data, size_t size) {
        std::lock_guard<std::mutex> lock(send_mutex_);
        if (!connected_) throw std::runtime_error("ECLATIN RDMA: channel not connected");
        ibv_mr* mr = find_registered_mr(reinterpret_cast<uintptr_t>(data), size);
        const uint8_t* send_ptr = data;
        if (!mr) {
            if (size > TEMP_BUFFER_SIZE) throw std::runtime_error("ECLATIN RDMA: data exceeds temp buffer");
            memcpy(temp_send_buffer_.data(), data, size);
            mr = temp_send_mr_;
            send_ptr = temp_send_buffer_.data();
        }
        {
            std::unique_lock<std::mutex> ctrl_lock;
            if (shared_control_mutex_) {
                ctrl_lock = std::unique_lock<std::mutex>(*shared_control_mutex_);
            }
            uint64_t size_network = htobe64(size);
            if (send(control_sock_send_, &size_network, sizeof(size_network), 0) != sizeof(size_network))
                throw std::runtime_error("ECLATIN RDMA: failed to send size");
            uint8_t ack;
            if (recv(control_sock_send_, &ack, sizeof(ack), MSG_WAITALL) != sizeof(ack))
                throw std::runtime_error("ECLATIN RDMA: failed to receive ACK");
        }
        send_data_chunked(send_ptr, size, mr);
    }

    size_t receive_data(uint8_t* buffer, size_t buffer_size) {
        std::lock_guard<std::mutex> lock(recv_mutex_);
        if (!connected_) throw std::runtime_error("ECLATIN RDMA: channel not connected");
        size_t size = 0;
        {
            std::unique_lock<std::mutex> ctrl_lock;
            if (shared_control_mutex_) {
                ctrl_lock = std::unique_lock<std::mutex>(*shared_control_mutex_);
            }
            uint64_t size_network;
            if (recv(control_sock_recv_, &size_network, sizeof(size_network), MSG_WAITALL) != sizeof(size_network))
                throw std::runtime_error("ECLATIN RDMA: failed to receive size");
            size = be64toh(size_network);
            if (size > buffer_size) throw std::runtime_error("ECLATIN RDMA: received size exceeds buffer");
            uint8_t ack = 1;
            if (send(control_sock_recv_, &ack, sizeof(ack), 0) != sizeof(ack))
                throw std::runtime_error("ECLATIN RDMA: failed to send ACK");
        }
        ibv_mr* mr = find_registered_mr(reinterpret_cast<uintptr_t>(buffer), size);
        bool use_temp = !mr;
        if (!mr) {
            if (size > TEMP_BUFFER_SIZE) throw std::runtime_error("ECLATIN RDMA: data exceeds temp buffer");
            mr = temp_recv_mr_;
        }
        uint8_t* recv_ptr = use_temp ? temp_recv_buffer_.data() : buffer;
        receive_data_chunked(recv_ptr, size, mr);
        if (use_temp) memcpy(buffer, temp_recv_buffer_.data(), size);
        return size;
    }

    bool is_connected() const { return connected_; }
};
#endif

// ASIO connection manager (pattern from eccheck_native)
static constexpr int NS_LOAD_NUM_CHANNELS = 8;

class AsioConnectionManager {
private:
    boost::asio::io_context io_context_;
    
    // Parity 1 sockets
    boost::asio::ip::tcp::socket parity1_send1_socket_;
    boost::asio::ip::tcp::socket parity1_send2_socket_;
    boost::asio::ip::tcp::socket parity1_recv1_socket_;
    boost::asio::ip::tcp::socket parity1_recv2_socket_;
    boost::asio::ip::tcp::acceptor parity1_recv1_acceptor_;
    boost::asio::ip::tcp::acceptor parity1_recv2_acceptor_;
    
    // Parity 2 sockets
    boost::asio::ip::tcp::socket parity2_send1_socket_;
    boost::asio::ip::tcp::socket parity2_send2_socket_;
    boost::asio::ip::tcp::socket parity2_recv1_socket_;
    boost::asio::ip::tcp::socket parity2_recv2_socket_;
    boost::asio::ip::tcp::acceptor parity2_recv1_acceptor_;
    boost::asio::ip::tcp::acceptor parity2_recv2_acceptor_;

    // Load mode sockets (rank2 as receiver)
    boost::asio::ip::tcp::socket load_recv_rank0_data2_socket_;
    boost::asio::ip::tcp::socket load_recv_rank0_parity2_socket_;
    boost::asio::ip::tcp::socket load_recv_rank1_data1_socket_;
    boost::asio::ip::tcp::socket load_recv_rank1_parity1_socket_;
    boost::asio::ip::tcp::socket load_recv_rank3_data1_socket_;
    boost::asio::ip::tcp::socket load_recv_rank3_data2_socket_;
    boost::asio::ip::tcp::acceptor load_recv_rank0_data2_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank0_parity2_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank1_data1_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank1_parity1_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank3_data1_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank3_data2_acceptor_;
    
    // Load mode sockets (other ranks as senders)
    boost::asio::ip::tcp::socket load_send_rank0_data2_socket_;
    boost::asio::ip::tcp::socket load_send_rank0_parity2_socket_;
    boost::asio::ip::tcp::socket load_send_rank1_data1_socket_;
    boost::asio::ip::tcp::socket load_send_rank1_parity1_socket_;
    boost::asio::ip::tcp::socket load_send_rank3_data1_socket_;
    boost::asio::ip::tcp::socket load_send_rank3_data2_socket_;

    // Two-failures load mode (2 connections per rank: peer0 and peer1) [old scheme]
    boost::asio::ip::tcp::socket load_twofail_peer0_socket_;
    boost::asio::ip::tcp::socket load_twofail_peer1_socket_;
    boost::asio::ip::tcp::acceptor load_twofail_peer0_acceptor_;
    boost::asio::ip::tcp::acceptor load_twofail_peer1_acceptor_;
    std::atomic<bool> load_twofail_peer0_connected_{false};
    std::atomic<bool> load_twofail_peer1_connected_{false};

    // New two-failures v2: 5 connections per group
    //   surv_exch:  survivor↔survivor (rig2 bind, rig3 connect)
    //   n1_from_n3: Node3→Node1 (rig0 bind, rig2 connect)
    //   n1_from_n4: Node4→Node1 (rig0 bind, rig3 connect)
    //   n2_from_n3: Node3→Node2 (rig1 bind, rig2 connect)
    //   n2_from_n4: Node4→Node2 (rig1 bind, rig3 connect)
    boost::asio::ip::tcp::socket twf_surv_exch_socket_;
    boost::asio::ip::tcp::acceptor twf_surv_exch_acceptor_;
    std::atomic<bool> twf_surv_exch_connected_{false};

    boost::asio::ip::tcp::socket twf_n1_from_n3_socket_;
    boost::asio::ip::tcp::socket twf_n1_from_n4_socket_;
    boost::asio::ip::tcp::socket twf_n2_from_n3_socket_;
    boost::asio::ip::tcp::socket twf_n2_from_n4_socket_;
    boost::asio::ip::tcp::acceptor twf_n1_from_n3_acceptor_;
    boost::asio::ip::tcp::acceptor twf_n1_from_n4_acceptor_;
    boost::asio::ip::tcp::acceptor twf_n2_from_n3_acceptor_;
    boost::asio::ip::tcp::acceptor twf_n2_from_n4_acceptor_;
    std::atomic<bool> twf_n1_from_n3_connected_{false};
    std::atomic<bool> twf_n1_from_n4_connected_{false};
    std::atomic<bool> twf_n2_from_n3_connected_{false};
    std::atomic<bool> twf_n2_from_n4_connected_{false};

    std::atomic<bool> parity1_send1_connected_;
    std::atomic<bool> parity1_send2_connected_;
    std::atomic<bool> parity1_recv1_connected_;
    std::atomic<bool> parity1_recv2_connected_;
    std::atomic<bool> parity2_send1_connected_;
    std::atomic<bool> parity2_send2_connected_;
    std::atomic<bool> parity2_recv1_connected_;
    std::atomic<bool> parity2_recv2_connected_;
    
    // Load mode connection flags (rank2 receiver)
    std::atomic<bool> load_recv_rank0_data2_connected_{false};
    std::atomic<bool> load_recv_rank0_parity2_connected_{false};
    std::atomic<bool> load_recv_rank1_data1_connected_{false};
    std::atomic<bool> load_recv_rank1_parity1_connected_{false};
    std::atomic<bool> load_recv_rank3_data1_connected_{false};
    std::atomic<bool> load_recv_rank3_data2_connected_{false};
    
    // Load mode connection flags (other ranks sender)
    std::atomic<bool> load_send_rank0_data2_connected_{false};
    std::atomic<bool> load_send_rank0_parity2_connected_{false};
    std::atomic<bool> load_send_rank1_data1_connected_{false};
    std::atomic<bool> load_send_rank1_parity1_connected_{false};
    std::atomic<bool> load_send_rank3_data1_connected_{false};
    std::atomic<bool> load_send_rank3_data2_connected_{false};

    // No-shared-block HW1 load (8 independent channels, 0-based indexing)
    std::vector<boost::asio::ip::tcp::socket> ns_load_recv_sockets_;
    std::vector<boost::asio::ip::tcp::acceptor> ns_load_recv_acceptors_;
    std::vector<boost::asio::ip::tcp::socket> ns_load_send_sockets_;
    std::array<std::atomic<bool>, NS_LOAD_NUM_CHANNELS> ns_load_recv_connected_;
    std::array<std::atomic<bool>, NS_LOAD_NUM_CHANNELS> ns_load_send_connected_;
    std::atomic<bool> ns_load_active_{false};

    std::mutex connection_mutex_;
    std::condition_variable connection_cv_;

public:
    AsioConnectionManager()
        : io_context_(),
          parity1_send1_socket_(io_context_),
          parity1_send2_socket_(io_context_),
          parity1_recv1_socket_(io_context_),
          parity1_recv2_socket_(io_context_),
          parity1_recv1_acceptor_(io_context_),
          parity1_recv2_acceptor_(io_context_),
          parity2_send1_socket_(io_context_),
          parity2_send2_socket_(io_context_),
          parity2_recv1_socket_(io_context_),
          parity2_recv2_socket_(io_context_),
          parity2_recv1_acceptor_(io_context_),
          parity2_recv2_acceptor_(io_context_),
          load_recv_rank0_data2_socket_(io_context_),
          load_recv_rank0_parity2_socket_(io_context_),
          load_recv_rank1_data1_socket_(io_context_),
          load_recv_rank1_parity1_socket_(io_context_),
          load_recv_rank3_data1_socket_(io_context_),
          load_recv_rank3_data2_socket_(io_context_),
          load_recv_rank0_data2_acceptor_(io_context_),
          load_recv_rank0_parity2_acceptor_(io_context_),
          load_recv_rank1_data1_acceptor_(io_context_),
          load_recv_rank1_parity1_acceptor_(io_context_),
          load_recv_rank3_data1_acceptor_(io_context_),
          load_recv_rank3_data2_acceptor_(io_context_),
          load_send_rank0_data2_socket_(io_context_),
          load_send_rank0_parity2_socket_(io_context_),
          load_send_rank1_data1_socket_(io_context_),
          load_send_rank1_parity1_socket_(io_context_),
          load_send_rank3_data1_socket_(io_context_),
          load_send_rank3_data2_socket_(io_context_),
          load_twofail_peer0_socket_(io_context_),
          load_twofail_peer1_socket_(io_context_),
          load_twofail_peer0_acceptor_(io_context_),
          load_twofail_peer1_acceptor_(io_context_),
          twf_surv_exch_socket_(io_context_),
          twf_surv_exch_acceptor_(io_context_),
          twf_n1_from_n3_socket_(io_context_),
          twf_n1_from_n4_socket_(io_context_),
          twf_n2_from_n3_socket_(io_context_),
          twf_n2_from_n4_socket_(io_context_),
          twf_n1_from_n3_acceptor_(io_context_),
          twf_n1_from_n4_acceptor_(io_context_),
          twf_n2_from_n3_acceptor_(io_context_),
          twf_n2_from_n4_acceptor_(io_context_),
          parity1_send1_connected_(false),
          parity1_send2_connected_(false),
          parity1_recv1_connected_(false),
          parity1_recv2_connected_(false),
          parity2_send1_connected_(false),
          parity2_send2_connected_(false),
          parity2_recv1_connected_(false),
          parity2_recv2_connected_(false) {
        for (int i = 0; i < NS_LOAD_NUM_CHANNELS; ++i) {
            ns_load_recv_sockets_.emplace_back(io_context_);
            ns_load_recv_acceptors_.emplace_back(io_context_);
            ns_load_send_sockets_.emplace_back(io_context_);
            ns_load_recv_connected_[i] = false;
            ns_load_send_connected_[i] = false;
        }
    }

    // Parity 1 getters
    boost::asio::ip::tcp::socket& get_parity1_send1_socket() { return parity1_send1_socket_; }
    boost::asio::ip::tcp::socket& get_parity1_send2_socket() { return parity1_send2_socket_; }
    boost::asio::ip::tcp::socket& get_parity1_recv1_socket() { return parity1_recv1_socket_; }
    boost::asio::ip::tcp::socket& get_parity1_recv2_socket() { return parity1_recv2_socket_; }
    
    // Parity 2 getters
    boost::asio::ip::tcp::socket& get_parity2_send1_socket() { return parity2_send1_socket_; }
    boost::asio::ip::tcp::socket& get_parity2_send2_socket() { return parity2_send2_socket_; }
    boost::asio::ip::tcp::socket& get_parity2_recv1_socket() { return parity2_recv1_socket_; }
    boost::asio::ip::tcp::socket& get_parity2_recv2_socket() { return parity2_recv2_socket_; }
    
    // Load mode getters (rank2 receiver)
    boost::asio::ip::tcp::socket& get_load_recv_rank0_data2_socket() { return load_recv_rank0_data2_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank0_parity2_socket() { return load_recv_rank0_parity2_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank1_data1_socket() { return load_recv_rank1_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank1_parity1_socket() { return load_recv_rank1_parity1_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank3_data1_socket() { return load_recv_rank3_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank3_data2_socket() { return load_recv_rank3_data2_socket_; }
    
    // Load mode getters (other ranks sender)
    boost::asio::ip::tcp::socket& get_load_send_rank0_data2_socket() { return load_send_rank0_data2_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank0_parity2_socket() { return load_send_rank0_parity2_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank1_data1_socket() { return load_send_rank1_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank1_parity1_socket() { return load_send_rank1_parity1_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank3_data1_socket() { return load_send_rank3_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank3_data2_socket() { return load_send_rank3_data2_socket_; }

    // Two-failures load getters [old scheme]
    boost::asio::ip::tcp::socket& get_load_twofail_peer0_socket() { return load_twofail_peer0_socket_; }
    boost::asio::ip::tcp::socket& get_load_twofail_peer1_socket() { return load_twofail_peer1_socket_; }
    bool is_load_twofail_peer0_connected() const { return load_twofail_peer0_connected_; }
    bool is_load_twofail_peer1_connected() const { return load_twofail_peer1_connected_; }

    // Two-failures v2 getters
    boost::asio::ip::tcp::socket& get_twf_surv_exch_socket() { return twf_surv_exch_socket_; }
    bool is_twf_surv_exch_connected() const { return twf_surv_exch_connected_; }
    boost::asio::ip::tcp::socket& get_twf_n1_n3_socket() { return twf_n1_from_n3_socket_; }
    boost::asio::ip::tcp::socket& get_twf_n1_n4_socket() { return twf_n1_from_n4_socket_; }
    boost::asio::ip::tcp::socket& get_twf_n2_n3_socket() { return twf_n2_from_n3_socket_; }
    boost::asio::ip::tcp::socket& get_twf_n2_n4_socket() { return twf_n2_from_n4_socket_; }
    bool is_twf_v2_connected() const {
        return twf_surv_exch_connected_ ||
               twf_n1_from_n3_connected_ || twf_n1_from_n4_connected_ ||
               twf_n2_from_n3_connected_ || twf_n2_from_n4_connected_;
    }

    // Two-failures v2: bind+listen helpers (for failed nodes as acceptors, rig2 for surv_exch)
    void bind_listen_twf_surv_exch(const std::string& listen_ip, uint16_t port);
    void accept_twf_surv_exch();
    void init_twf_surv_exch_send(const std::string& partner_ip, uint16_t port);

    void bind_listen_twf_n1_from_n3(const std::string& listen_ip, uint16_t port);
    void bind_listen_twf_n1_from_n4(const std::string& listen_ip, uint16_t port);
    void bind_listen_twf_n2_from_n3(const std::string& listen_ip, uint16_t port);
    void bind_listen_twf_n2_from_n4(const std::string& listen_ip, uint16_t port);

    void accept_twf_n1_from_n3();
    void accept_twf_n1_from_n4();
    void accept_twf_n2_from_n3();
    void accept_twf_n2_from_n4();

    void init_twf_send_n1_from_n3(const std::string& partner_ip, uint16_t port);
    void init_twf_send_n1_from_n4(const std::string& partner_ip, uint16_t port);
    void init_twf_send_n2_from_n3(const std::string& partner_ip, uint16_t port);
    void init_twf_send_n2_from_n4(const std::string& partner_ip, uint16_t port);

    void wait_for_twf_v2_connections(int timeout_seconds);

    // Parity 1 connection checks
    bool is_parity1_send1_connected() const { return parity1_send1_connected_; }
    bool is_parity1_send2_connected() const { return parity1_send2_connected_; }
    bool is_parity1_recv1_connected() const { return parity1_recv1_connected_; }
    bool is_parity1_recv2_connected() const { return parity1_recv2_connected_; }
    
    // Parity 2 connection checks
    bool is_parity2_send1_connected() const { return parity2_send1_connected_; }
    bool is_parity2_send2_connected() const { return parity2_send2_connected_; }
    bool is_parity2_recv1_connected() const { return parity2_recv1_connected_; }
    bool is_parity2_recv2_connected() const { return parity2_recv2_connected_; }

    // Parity 1 init functions
    void init_parity1_send1(const std::string& partner_ip, uint16_t port);
    void init_parity1_send2(const std::string& partner_ip, uint16_t port);
    void init_parity1_recv1(const std::string& listen_ip, uint16_t port);
    void init_parity1_recv2(const std::string& listen_ip, uint16_t port);
    
    // Parity 2 init functions
    void init_parity2_send1(const std::string& partner_ip, uint16_t port);
    void init_parity2_send2(const std::string& partner_ip, uint16_t port);
    void init_parity2_recv1(const std::string& listen_ip, uint16_t port);
    void init_parity2_recv2(const std::string& listen_ip, uint16_t port);
    
    // Load mode init functions (rank2 as receiver)
    void init_load_recv_rank0_data2(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank0_parity2(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank1_parity1(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank3_data2(const std::string& listen_ip, uint16_t port);
    
    // Load mode init functions (other ranks as senders)
    void init_load_send_rank0_data2(const std::string& rank2_ip, uint16_t port);
    void init_load_send_rank0_parity2(const std::string& rank2_ip, uint16_t port);
    void init_load_send_rank1_data1(const std::string& rank2_ip, uint16_t port);
    void init_load_send_rank1_parity1(const std::string& rank2_ip, uint16_t port);
    void init_load_send_rank3_data1(const std::string& rank2_ip, uint16_t port);
    void init_load_send_rank3_data2(const std::string& rank2_ip, uint16_t port);
    
    // Load mode bind+listen helpers (for rank2, before accept)
    void bind_listen_load_recv_rank0_data2(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank0_parity2(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank1_parity1(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank3_data2(const std::string& listen_ip, uint16_t port);
    
    // Load mode accept helpers (for rank2, after bind+listen)
    void accept_load_recv_rank0_data2();
    void accept_load_recv_rank0_parity2();
    void accept_load_recv_rank1_data1();
    void accept_load_recv_rank1_parity1();
    void accept_load_recv_rank3_data1();
    void accept_load_recv_rank3_data2();

    // No-shared-block HW1 load (8 channels)
    void set_ns_load_active(bool active) { ns_load_active_ = active; }
    bool is_ns_load_active() const { return ns_load_active_; }
    void bind_listen_ns_load_recv(int channel, const std::string& listen_ip, uint16_t port);
    void accept_ns_load_recv(int channel);
    void init_ns_load_send(int channel, const std::string& rank2_ip, uint16_t port);
    boost::asio::ip::tcp::socket& get_ns_load_recv_socket(int channel);
    boost::asio::ip::tcp::socket& get_ns_load_send_socket(int channel);
    bool is_ns_load_recv_connected(int channel) const;
    bool is_ns_load_send_connected(int channel) const;
    bool is_ns_load_recv_acceptor_open(int channel) const;
    void wait_for_ns_load_connections(int rank_in_group, int timeout_seconds = 30);

    // Two-failures load mode methods
    void bind_listen_load_twofail_peer0(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_twofail_peer1(const std::string& listen_ip, uint16_t port);
    void accept_load_twofail_peer0();
    void accept_load_twofail_peer1();
    void init_load_send_twofail_peer0(const std::string& partner_ip, uint16_t port);
    void init_load_send_twofail_peer1(const std::string& partner_ip, uint16_t port);

    void wait_for_connections(int timeout_seconds = 30);
    void wait_for_load_connections(int timeout_seconds = 30);
    void cleanup();
};

// Parity 1 init functions
void AsioConnectionManager::init_parity1_send1(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(parity1_send1_socket_, endpoints);
        parity1_send1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity1_send1 init error: " << e.what() << std::endl;
        parity1_send1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity1_send2(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(parity1_send2_socket_, endpoints);
        parity1_send2_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity1_send2 init error: " << e.what() << std::endl;
        parity1_send2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity1_recv1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        parity1_recv1_acceptor_.open(endpoint.protocol());
        parity1_recv1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        parity1_recv1_acceptor_.bind(endpoint);
        parity1_recv1_acceptor_.listen();
        parity1_recv1_acceptor_.accept(parity1_recv1_socket_);
        parity1_recv1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity1_recv1 init error: " << e.what() << std::endl;
        parity1_recv1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity1_recv2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        parity1_recv2_acceptor_.open(endpoint.protocol());
        parity1_recv2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        parity1_recv2_acceptor_.bind(endpoint);
        parity1_recv2_acceptor_.listen();
        parity1_recv2_acceptor_.accept(parity1_recv2_socket_);
        parity1_recv2_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity1_recv2 init error: " << e.what() << std::endl;
        parity1_recv2_connected_ = false;
        connection_cv_.notify_all();
    }
}

// Parity 2 init functions
void AsioConnectionManager::init_parity2_send1(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(parity2_send1_socket_, endpoints);
        parity2_send1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity2_send1 init error: " << e.what() << std::endl;
        parity2_send1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity2_send2(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(parity2_send2_socket_, endpoints);
        parity2_send2_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity2_send2 init error: " << e.what() << std::endl;
        parity2_send2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity2_recv1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        parity2_recv1_acceptor_.open(endpoint.protocol());
        parity2_recv1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        parity2_recv1_acceptor_.bind(endpoint);
        parity2_recv1_acceptor_.listen();
        parity2_recv1_acceptor_.accept(parity2_recv1_socket_);
        parity2_recv1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity2_recv1 init error: " << e.what() << std::endl;
        parity2_recv1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity2_recv2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        parity2_recv2_acceptor_.open(endpoint.protocol());
        parity2_recv2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        parity2_recv2_acceptor_.bind(endpoint);
        parity2_recv2_acceptor_.listen();
        parity2_recv2_acceptor_.accept(parity2_recv2_socket_);
        parity2_recv2_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity2_recv2 init error: " << e.what() << std::endl;
        parity2_recv2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::wait_for_connections(int timeout_seconds) {
    std::unique_lock<std::mutex> lock(connection_mutex_);
    connection_cv_.wait_for(
        lock,
        std::chrono::seconds(timeout_seconds),
        [this]() {
            return parity1_send1_connected_ && parity1_send2_connected_ && 
                   parity1_recv1_connected_ && parity1_recv2_connected_ &&
                   parity2_send1_connected_ && parity2_send2_connected_ && 
                   parity2_recv1_connected_ && parity2_recv2_connected_;
        }
    );
}

void AsioConnectionManager::wait_for_load_connections(int timeout_seconds) {
    if (ns_load_active_) {
        std::cerr << "ECLATIN: wait_for_load_connections called in ns_load mode; "
                  << "use wait_for_ns_load_connections instead" << std::endl;
        return;
    }
    // Single-failure HW recovery: rank_in_group 2 is receiver (6 acceptors).
    // Use acceptor.is_open() to detect the receiver role even before the first
    // accept completes (mirrors wait_for_twf_v2_connections; fixes multi-node races).
    const bool is_rank2_receiver =
        load_recv_rank0_data2_acceptor_.is_open() ||
        load_recv_rank0_parity2_acceptor_.is_open() ||
        load_recv_rank1_data1_acceptor_.is_open() ||
        load_recv_rank1_parity1_acceptor_.is_open() ||
        load_recv_rank3_data1_acceptor_.is_open() ||
        load_recv_rank3_data2_acceptor_.is_open();

    if (is_rank2_receiver) {
        int wait_count = 0;
        while (true) {
            bool done = true;
            if ((load_recv_rank0_data2_acceptor_.is_open() || load_recv_rank0_data2_connected_) &&
                !load_recv_rank0_data2_connected_) {
                done = false;
            }
            if ((load_recv_rank0_parity2_acceptor_.is_open() || load_recv_rank0_parity2_connected_) &&
                !load_recv_rank0_parity2_connected_) {
                done = false;
            }
            if ((load_recv_rank1_data1_acceptor_.is_open() || load_recv_rank1_data1_connected_) &&
                !load_recv_rank1_data1_connected_) {
                done = false;
            }
            if ((load_recv_rank1_parity1_acceptor_.is_open() || load_recv_rank1_parity1_connected_) &&
                !load_recv_rank1_parity1_connected_) {
                done = false;
            }
            if ((load_recv_rank3_data1_acceptor_.is_open() || load_recv_rank3_data1_connected_) &&
                !load_recv_rank3_data1_connected_) {
                done = false;
            }
            if ((load_recv_rank3_data2_acceptor_.is_open() || load_recv_rank3_data2_connected_) &&
                !load_recv_rank3_data2_connected_) {
                done = false;
            }
            if (done) {
                break;
            }
            if (wait_count % 100 == 0) {
                std::cout << "ECLATIN: [Rank 2] Waiting for load connections: "
                          << "r0_d2=" << (load_recv_rank0_data2_connected_ ? "true" : "false")
                          << ", r0_p2=" << (load_recv_rank0_parity2_connected_ ? "true" : "false")
                          << ", r1_d1=" << (load_recv_rank1_data1_connected_ ? "true" : "false")
                          << ", r1_p1=" << (load_recv_rank1_parity1_connected_ ? "true" : "false")
                          << ", r3_d1=" << (load_recv_rank3_data1_connected_ ? "true" : "false")
                          << ", r3_d2=" << (load_recv_rank3_data2_connected_ ? "true" : "false") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Rank 2] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    } else if (load_send_rank0_data2_connected_ || load_send_rank0_parity2_connected_) {
        // rank0: wait for 2 send connections
        int wait_count = 0;
        while (!(load_send_rank0_data2_connected_ && load_send_rank0_parity2_connected_)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Rank 0] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    } else if (load_send_rank1_data1_connected_ || load_send_rank1_parity1_connected_) {
        // rank1: wait for 2 send connections
        int wait_count = 0;
        while (!(load_send_rank1_data1_connected_ && load_send_rank1_parity1_connected_)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Rank 1] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    } else if (load_send_rank3_data1_connected_ || load_send_rank3_data2_connected_) {
        // rank3: wait for 2 send connections
        int wait_count = 0;
        while (!(load_send_rank3_data1_connected_ && load_send_rank3_data2_connected_)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Rank 3] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    }
    // Two-failures load: wait for peer0 and peer1 connections (recv or send)
    if (load_twofail_peer0_connected_ || load_twofail_peer1_connected_) {
        int wait_count = 0;
        while (!(load_twofail_peer0_connected_ && load_twofail_peer1_connected_)) {
            if (wait_count % 100 == 0) {
                std::cout << "ECLATIN: [Two-fail] Waiting for load connections: "
                          << "peer0=" << (load_twofail_peer0_connected_ ? "true" : "false")
                          << ", peer1=" << (load_twofail_peer1_connected_ ? "true" : "false") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Two-fail] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    }
}

void AsioConnectionManager::bind_listen_ns_load_recv(
    int channel, const std::string& listen_ip, uint16_t port
) {
    if (channel < 0 || channel >= NS_LOAD_NUM_CHANNELS) {
        throw std::runtime_error("ASIO: bind_listen_ns_load_recv invalid channel");
    }
    boost::asio::ip::tcp::endpoint endpoint(
        boost::asio::ip::address::from_string(listen_ip), port);
    ns_load_recv_acceptors_[channel].open(endpoint.protocol());
    ns_load_recv_acceptors_[channel].set_option(
        boost::asio::ip::tcp::acceptor::reuse_address(true));
    ns_load_recv_acceptors_[channel].bind(endpoint);
    ns_load_recv_acceptors_[channel].listen();
}

void AsioConnectionManager::accept_ns_load_recv(int channel) {
    if (channel < 0 || channel >= NS_LOAD_NUM_CHANNELS) {
        throw std::runtime_error("ASIO: accept_ns_load_recv invalid channel");
    }
    try {
        ns_load_recv_acceptors_[channel].accept(ns_load_recv_sockets_[channel]);
        ns_load_recv_connected_[channel] = true;
        std::cout << "ASIO: ns_load_recv ch" << channel << " connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: ns_load_recv ch" << channel << " accept error: " << e.what() << std::endl;
        ns_load_recv_connected_[channel] = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_ns_load_send(
    int channel, const std::string& rank2_ip, uint16_t port
) {
    if (channel < 0 || channel >= NS_LOAD_NUM_CHANNELS) {
        throw std::runtime_error("ASIO: init_ns_load_send invalid channel");
    }
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(ns_load_send_sockets_[channel], endpoints);
        ns_load_send_connected_[channel] = true;
        std::cout << "ASIO: ns_load_send ch" << channel << " connected to rank2" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: ns_load_send ch" << channel << " init error: " << e.what() << std::endl;
        ns_load_send_connected_[channel] = false;
        throw;
    }
}

boost::asio::ip::tcp::socket& AsioConnectionManager::get_ns_load_recv_socket(int channel) {
    return ns_load_recv_sockets_.at(channel);
}

boost::asio::ip::tcp::socket& AsioConnectionManager::get_ns_load_send_socket(int channel) {
    return ns_load_send_sockets_.at(channel);
}

bool AsioConnectionManager::is_ns_load_recv_connected(int channel) const {
    return ns_load_recv_connected_.at(channel);
}

bool AsioConnectionManager::is_ns_load_send_connected(int channel) const {
    return ns_load_send_connected_.at(channel);
}

bool AsioConnectionManager::is_ns_load_recv_acceptor_open(int channel) const {
    return ns_load_recv_acceptors_.at(channel).is_open();
}

void AsioConnectionManager::wait_for_ns_load_connections(int rank_in_group, int timeout_seconds) {
    auto wait_channels = [&](const std::vector<int>& channels, bool recv_side) {
        int wait_count = 0;
        while (true) {
            bool done = true;
            for (int ch : channels) {
                bool connected = recv_side
                    ? ns_load_recv_connected_[ch].load()
                    : ns_load_send_connected_[ch].load();
                if (!connected) {
                    done = false;
                    break;
                }
            }
            if (done) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: Timeout waiting for ns_load connections (rig"
                          << rank_in_group << ")" << std::endl;
                break;
            }
        }
    };

    if (rank_in_group == 2) {
        std::vector<int> all_channels(NS_LOAD_NUM_CHANNELS);
        for (int i = 0; i < NS_LOAD_NUM_CHANNELS; ++i) all_channels[i] = i;
        wait_channels(all_channels, true);
    } else if (rank_in_group == 0) {
        wait_channels({2, 6}, false);
    } else if (rank_in_group == 1) {
        wait_channels({0, 4}, false);
    } else if (rank_in_group == 3) {
        wait_channels({1, 3, 5, 7}, false);
    }
}

void AsioConnectionManager::init_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank1_data1_acceptor_.open(endpoint.protocol());
        load_recv_rank1_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank1_data1_acceptor_.bind(endpoint);
        load_recv_rank1_data1_acceptor_.listen();
        load_recv_rank1_data1_acceptor_.accept(load_recv_rank1_data1_socket_);
        load_recv_rank1_data1_connected_ = true;
        std::cout << "ASIO: load_recv_rank1_data1 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank1_data1 init error: " << e.what() << std::endl;
        load_recv_rank1_data1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank1_parity1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank1_parity1_acceptor_.open(endpoint.protocol());
        load_recv_rank1_parity1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank1_parity1_acceptor_.bind(endpoint);
        load_recv_rank1_parity1_acceptor_.listen();
        load_recv_rank1_parity1_acceptor_.accept(load_recv_rank1_parity1_socket_);
        load_recv_rank1_parity1_connected_ = true;
        std::cout << "ASIO: load_recv_rank1_parity1 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank1_parity1 init error: " << e.what() << std::endl;
        load_recv_rank1_parity1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank0_data2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank0_data2_acceptor_.open(endpoint.protocol());
        load_recv_rank0_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank0_data2_acceptor_.bind(endpoint);
        load_recv_rank0_data2_acceptor_.listen();
        load_recv_rank0_data2_acceptor_.accept(load_recv_rank0_data2_socket_);
        load_recv_rank0_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank0_data2 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank0_data2 init error: " << e.what() << std::endl;
        load_recv_rank0_data2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank0_parity2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank0_parity2_acceptor_.open(endpoint.protocol());
        load_recv_rank0_parity2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank0_parity2_acceptor_.bind(endpoint);
        load_recv_rank0_parity2_acceptor_.listen();
        load_recv_rank0_parity2_acceptor_.accept(load_recv_rank0_parity2_socket_);
        load_recv_rank0_parity2_connected_ = true;
        std::cout << "ASIO: load_recv_rank0_parity2 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank0_parity2 init error: " << e.what() << std::endl;
        load_recv_rank0_parity2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank3_data1_acceptor_.open(endpoint.protocol());
        load_recv_rank3_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank3_data1_acceptor_.bind(endpoint);
        load_recv_rank3_data1_acceptor_.listen();
        load_recv_rank3_data1_acceptor_.accept(load_recv_rank3_data1_socket_);
        load_recv_rank3_data1_connected_ = true;
        std::cout << "ASIO: load_recv_rank3_data1 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank3_data1 init error: " << e.what() << std::endl;
        load_recv_rank3_data1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank3_data2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank3_data2_acceptor_.open(endpoint.protocol());
        load_recv_rank3_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank3_data2_acceptor_.bind(endpoint);
        load_recv_rank3_data2_acceptor_.listen();
        load_recv_rank3_data2_acceptor_.accept(load_recv_rank3_data2_socket_);
        load_recv_rank3_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank3_data2 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank3_data2 init error: " << e.what() << std::endl;
        load_recv_rank3_data2_connected_ = false;
        throw;
    }
}

// Load mode init functions (other ranks as senders)
void AsioConnectionManager::init_load_send_rank0_data2(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank0_data2_socket_, endpoints);
        load_send_rank0_data2_connected_ = true;
        std::cout << "ASIO: load_send_rank0_data2 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank0_data2 init error: " << e.what() << std::endl;
        load_send_rank0_data2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank0_parity2(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank0_parity2_socket_, endpoints);
        load_send_rank0_parity2_connected_ = true;
        std::cout << "ASIO: load_send_rank0_parity2 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank0_parity2 init error: " << e.what() << std::endl;
        load_send_rank0_parity2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank1_data1(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank1_data1_socket_, endpoints);
        load_send_rank1_data1_connected_ = true;
        std::cout << "ASIO: load_send_rank1_data1 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank1_data1 init error: " << e.what() << std::endl;
        load_send_rank1_data1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank1_parity1(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank1_parity1_socket_, endpoints);
        load_send_rank1_parity1_connected_ = true;
        std::cout << "ASIO: load_send_rank1_parity1 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank1_parity1 init error: " << e.what() << std::endl;
        load_send_rank1_parity1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank3_data1(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank3_data1_socket_, endpoints);
        load_send_rank3_data1_connected_ = true;
        std::cout << "ASIO: load_send_rank3_data1 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank3_data1 init error: " << e.what() << std::endl;
        load_send_rank3_data1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank3_data2(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank3_data2_socket_, endpoints);
        load_send_rank3_data2_connected_ = true;
        std::cout << "ASIO: load_send_rank3_data2 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank3_data2 init error: " << e.what() << std::endl;
        load_send_rank3_data2_connected_ = false;
        throw;
    }
}

// Load mode bind+listen helpers (for rank2, before accept)
void AsioConnectionManager::bind_listen_load_recv_rank0_data2(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank0_data2_acceptor_.open(endpoint.protocol());
    load_recv_rank0_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank0_data2_acceptor_.bind(endpoint);
    load_recv_rank0_data2_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank0_parity2(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank0_parity2_acceptor_.open(endpoint.protocol());
    load_recv_rank0_parity2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank0_parity2_acceptor_.bind(endpoint);
    load_recv_rank0_parity2_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank1_data1_acceptor_.open(endpoint.protocol());
    load_recv_rank1_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank1_data1_acceptor_.bind(endpoint);
    load_recv_rank1_data1_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank1_parity1(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank1_parity1_acceptor_.open(endpoint.protocol());
    load_recv_rank1_parity1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank1_parity1_acceptor_.bind(endpoint);
    load_recv_rank1_parity1_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank3_data1_acceptor_.open(endpoint.protocol());
    load_recv_rank3_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank3_data1_acceptor_.bind(endpoint);
    load_recv_rank3_data1_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank3_data2(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank3_data2_acceptor_.open(endpoint.protocol());
    load_recv_rank3_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank3_data2_acceptor_.bind(endpoint);
    load_recv_rank3_data2_acceptor_.listen();
}

// Load mode accept helpers (for rank2, after bind+listen)
void AsioConnectionManager::accept_load_recv_rank0_data2() {
    try {
        load_recv_rank0_data2_acceptor_.accept(load_recv_rank0_data2_socket_);
        load_recv_rank0_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank0_data2 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank0_data2 accept error: " << e.what() << std::endl;
        load_recv_rank0_data2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank0_parity2() {
    try {
        load_recv_rank0_parity2_acceptor_.accept(load_recv_rank0_parity2_socket_);
        load_recv_rank0_parity2_connected_ = true;
        std::cout << "ASIO: load_recv_rank0_parity2 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank0_parity2 accept error: " << e.what() << std::endl;
        load_recv_rank0_parity2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank1_data1() {
    try {
        load_recv_rank1_data1_acceptor_.accept(load_recv_rank1_data1_socket_);
        load_recv_rank1_data1_connected_ = true;
        std::cout << "ASIO: load_recv_rank1_data1 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank1_data1 accept error: " << e.what() << std::endl;
        load_recv_rank1_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank1_parity1() {
    try {
        load_recv_rank1_parity1_acceptor_.accept(load_recv_rank1_parity1_socket_);
        load_recv_rank1_parity1_connected_ = true;
        std::cout << "ASIO: load_recv_rank1_parity1 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank1_parity1 accept error: " << e.what() << std::endl;
        load_recv_rank1_parity1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank3_data1() {
    try {
        load_recv_rank3_data1_acceptor_.accept(load_recv_rank3_data1_socket_);
        load_recv_rank3_data1_connected_ = true;
        std::cout << "ASIO: load_recv_rank3_data1 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank3_data1 accept error: " << e.what() << std::endl;
        load_recv_rank3_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank3_data2() {
    try {
        load_recv_rank3_data2_acceptor_.accept(load_recv_rank3_data2_socket_);
        load_recv_rank3_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank3_data2 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank3_data2 accept error: " << e.what() << std::endl;
        load_recv_rank3_data2_connected_ = false;
        connection_cv_.notify_all();
    }
}

// ── Two-failures load mode: bind+listen helpers ────────────────────────────

void AsioConnectionManager::bind_listen_load_twofail_peer0(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_twofail_peer0_acceptor_.open(endpoint.protocol());
    load_twofail_peer0_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_twofail_peer0_acceptor_.bind(endpoint);
    load_twofail_peer0_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_twofail_peer1(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_twofail_peer1_acceptor_.open(endpoint.protocol());
    load_twofail_peer1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_twofail_peer1_acceptor_.bind(endpoint);
    load_twofail_peer1_acceptor_.listen();
}

// ── Two-failures load mode: accept helpers ─────────────────────────────────

void AsioConnectionManager::accept_load_twofail_peer0() {
    try {
        load_twofail_peer0_acceptor_.accept(load_twofail_peer0_socket_);
        load_twofail_peer0_connected_ = true;
        std::cout << "ASIO: load_twofail_peer0 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_twofail_peer0 accept error: " << e.what() << std::endl;
        load_twofail_peer0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_twofail_peer1() {
    try {
        load_twofail_peer1_acceptor_.accept(load_twofail_peer1_socket_);
        load_twofail_peer1_connected_ = true;
        std::cout << "ASIO: load_twofail_peer1 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_twofail_peer1 accept error: " << e.what() << std::endl;
        load_twofail_peer1_connected_ = false;
        connection_cv_.notify_all();
    }
}

// ── Two-failures load mode: connect helpers ────────────────────────────────

void AsioConnectionManager::init_load_send_twofail_peer0(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(load_twofail_peer0_socket_, endpoints);
        load_twofail_peer0_connected_ = true;
        std::cout << "ASIO: load_twofail_peer0 send connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_twofail_peer0 send init error: " << e.what() << std::endl;
        load_twofail_peer0_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_twofail_peer1(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(load_twofail_peer1_socket_, endpoints);
        load_twofail_peer1_connected_ = true;
        std::cout << "ASIO: load_twofail_peer1 send connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_twofail_peer1 send init error: " << e.what() << std::endl;
        load_twofail_peer1_connected_ = false;
        throw;
    }
}

// ── Two-failures v2: survivor exchange bind+listen+accept ─────────────────

void AsioConnectionManager::bind_listen_twf_surv_exch(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    twf_surv_exch_acceptor_.open(endpoint.protocol());
    twf_surv_exch_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    twf_surv_exch_acceptor_.bind(endpoint);
    twf_surv_exch_acceptor_.listen();
}

void AsioConnectionManager::accept_twf_surv_exch() {
    try {
        twf_surv_exch_acceptor_.accept(twf_surv_exch_socket_);
        twf_surv_exch_connected_ = true;
        std::cout << "ASIO: twf_surv_exch connected (accepted)" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: twf_surv_exch accept error: " << e.what() << std::endl;
        twf_surv_exch_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_twf_surv_exch_send(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(twf_surv_exch_socket_, endpoints);
        twf_surv_exch_connected_ = true;
        std::cout << "ASIO: twf_surv_exch connected (connect)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: twf_surv_exch connect error: " << e.what() << std::endl;
        twf_surv_exch_connected_ = false;
        throw;
    }
}

// ── Two-failures v2: failed node bind+listen helpers ──────────────────────

void AsioConnectionManager::bind_listen_twf_n1_from_n3(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    twf_n1_from_n3_acceptor_.open(endpoint.protocol());
    twf_n1_from_n3_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    twf_n1_from_n3_acceptor_.bind(endpoint);
    twf_n1_from_n3_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_twf_n1_from_n4(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    twf_n1_from_n4_acceptor_.open(endpoint.protocol());
    twf_n1_from_n4_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    twf_n1_from_n4_acceptor_.bind(endpoint);
    twf_n1_from_n4_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_twf_n2_from_n3(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    twf_n2_from_n3_acceptor_.open(endpoint.protocol());
    twf_n2_from_n3_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    twf_n2_from_n3_acceptor_.bind(endpoint);
    twf_n2_from_n3_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_twf_n2_from_n4(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    twf_n2_from_n4_acceptor_.open(endpoint.protocol());
    twf_n2_from_n4_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    twf_n2_from_n4_acceptor_.bind(endpoint);
    twf_n2_from_n4_acceptor_.listen();
}

// ── Two-failures v2: failed node accept helpers ───────────────────────────

void AsioConnectionManager::accept_twf_n1_from_n3() {
    try {
        twf_n1_from_n3_acceptor_.accept(twf_n1_from_n3_socket_);
        twf_n1_from_n3_connected_ = true;
        std::cout << "ASIO: twf_n1_from_n3 connected (accepted)" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: twf_n1_from_n3 accept error: " << e.what() << std::endl;
        twf_n1_from_n3_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_twf_n1_from_n4() {
    try {
        twf_n1_from_n4_acceptor_.accept(twf_n1_from_n4_socket_);
        twf_n1_from_n4_connected_ = true;
        std::cout << "ASIO: twf_n1_from_n4 connected (accepted)" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: twf_n1_from_n4 accept error: " << e.what() << std::endl;
        twf_n1_from_n4_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_twf_n2_from_n3() {
    try {
        twf_n2_from_n3_acceptor_.accept(twf_n2_from_n3_socket_);
        twf_n2_from_n3_connected_ = true;
        std::cout << "ASIO: twf_n2_from_n3 connected (accepted)" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: twf_n2_from_n3 accept error: " << e.what() << std::endl;
        twf_n2_from_n3_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_twf_n2_from_n4() {
    try {
        twf_n2_from_n4_acceptor_.accept(twf_n2_from_n4_socket_);
        twf_n2_from_n4_connected_ = true;
        std::cout << "ASIO: twf_n2_from_n4 connected (accepted)" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: twf_n2_from_n4 accept error: " << e.what() << std::endl;
        twf_n2_from_n4_connected_ = false;
        connection_cv_.notify_all();
    }
}

// ── Two-failures v2: survivor connect helpers ─────────────────────────────

void AsioConnectionManager::init_twf_send_n1_from_n3(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(twf_n1_from_n3_socket_, endpoints);
        twf_n1_from_n3_connected_ = true;
        std::cout << "ASIO: twf_send_n1_from_n3 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: twf_send_n1_from_n3 connect error: " << e.what() << std::endl;
        twf_n1_from_n3_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_twf_send_n1_from_n4(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(twf_n1_from_n4_socket_, endpoints);
        twf_n1_from_n4_connected_ = true;
        std::cout << "ASIO: twf_send_n1_from_n4 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: twf_send_n1_from_n4 connect error: " << e.what() << std::endl;
        twf_n1_from_n4_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_twf_send_n2_from_n3(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(twf_n2_from_n3_socket_, endpoints);
        twf_n2_from_n3_connected_ = true;
        std::cout << "ASIO: twf_send_n2_from_n3 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: twf_send_n2_from_n3 connect error: " << e.what() << std::endl;
        twf_n2_from_n3_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_twf_send_n2_from_n4(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(twf_n2_from_n4_socket_, endpoints);
        twf_n2_from_n4_connected_ = true;
        std::cout << "ASIO: twf_send_n2_from_n4 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: twf_send_n2_from_n4 connect error: " << e.what() << std::endl;
        twf_n2_from_n4_connected_ = false;
        throw;
    }
}

// ── Two-failures v2: wait for all connections ─────────────────────────────

void AsioConnectionManager::wait_for_twf_v2_connections(int timeout_seconds) {
    if (!is_twf_v2_connected()) return;

    // Each rank waits for its subset of connections
    // rig0: n1_from_n3 + n1_from_n4
    // rig1: n2_from_n3 + n2_from_n4
    // rig2: surv_exch + n1_from_n3 + n2_from_n3
    // rig3: surv_exch + n1_from_n4 + n2_from_n4
    //
    // We wait until all connections that were initialized are established.
    int wait_count = 0;
    while (true) {
        bool done = true;

        // Check survivor exchange (rig2+rig3)
        if (twf_surv_exch_acceptor_.is_open() || twf_surv_exch_connected_) {
            if (!twf_surv_exch_connected_) done = false;
        }

        // Check survivor→failed connections
        if (twf_n1_from_n3_acceptor_.is_open() || twf_n1_from_n3_connected_) {
            if (!twf_n1_from_n3_connected_) done = false;
        }
        if (twf_n1_from_n4_acceptor_.is_open() || twf_n1_from_n4_connected_) {
            if (!twf_n1_from_n4_connected_) done = false;
        }
        if (twf_n2_from_n3_acceptor_.is_open() || twf_n2_from_n3_connected_) {
            if (!twf_n2_from_n3_connected_) done = false;
        }
        if (twf_n2_from_n4_acceptor_.is_open() || twf_n2_from_n4_connected_) {
            if (!twf_n2_from_n4_connected_) done = false;
        }

        if (done) break;

        if (wait_count % 100 == 0) {
            std::cout << "ECLATIN: [Two-fail v2] Waiting for connections: "
                      << "surv_exch=" << twf_surv_exch_connected_
                      << " n1_n3=" << twf_n1_from_n3_connected_
                      << " n1_n4=" << twf_n1_from_n4_connected_
                      << " n2_n3=" << twf_n2_from_n3_connected_
                      << " n2_n4=" << twf_n2_from_n4_connected_ << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
        if (wait_count * 10 > timeout_seconds * 1000) {
            std::cerr << "ECLATIN: [Two-fail v2] Timeout waiting for connections" << std::endl;
            break;
        }
    }
}

void AsioConnectionManager::cleanup() {
    // Parity 1 sockets
    if (parity1_send1_socket_.is_open()) parity1_send1_socket_.close();
    if (parity1_send2_socket_.is_open()) parity1_send2_socket_.close();
    if (parity1_recv1_socket_.is_open()) parity1_recv1_socket_.close();
    if (parity1_recv2_socket_.is_open()) parity1_recv2_socket_.close();
    if (parity1_recv1_acceptor_.is_open()) parity1_recv1_acceptor_.close();
    if (parity1_recv2_acceptor_.is_open()) parity1_recv2_acceptor_.close();
    
    // Parity 2 sockets
    if (parity2_send1_socket_.is_open()) parity2_send1_socket_.close();
    if (parity2_send2_socket_.is_open()) parity2_send2_socket_.close();
    if (parity2_recv1_socket_.is_open()) parity2_recv1_socket_.close();
    if (parity2_recv2_socket_.is_open()) parity2_recv2_socket_.close();
    if (parity2_recv1_acceptor_.is_open()) parity2_recv1_acceptor_.close();
    if (parity2_recv2_acceptor_.is_open()) parity2_recv2_acceptor_.close();
    
    // Load mode sockets (rank2 receiver)
    if (load_recv_rank0_data2_socket_.is_open()) load_recv_rank0_data2_socket_.close();
    if (load_recv_rank0_parity2_socket_.is_open()) load_recv_rank0_parity2_socket_.close();
    if (load_recv_rank1_data1_socket_.is_open()) load_recv_rank1_data1_socket_.close();
    if (load_recv_rank1_parity1_socket_.is_open()) load_recv_rank1_parity1_socket_.close();
    if (load_recv_rank3_data1_socket_.is_open()) load_recv_rank3_data1_socket_.close();
    if (load_recv_rank3_data2_socket_.is_open()) load_recv_rank3_data2_socket_.close();
    if (load_recv_rank0_data2_acceptor_.is_open()) load_recv_rank0_data2_acceptor_.close();
    if (load_recv_rank0_parity2_acceptor_.is_open()) load_recv_rank0_parity2_acceptor_.close();
    if (load_recv_rank1_data1_acceptor_.is_open()) load_recv_rank1_data1_acceptor_.close();
    if (load_recv_rank1_parity1_acceptor_.is_open()) load_recv_rank1_parity1_acceptor_.close();
    if (load_recv_rank3_data1_acceptor_.is_open()) load_recv_rank3_data1_acceptor_.close();
    if (load_recv_rank3_data2_acceptor_.is_open()) load_recv_rank3_data2_acceptor_.close();
    
    // Load mode sockets (other ranks sender)
    if (load_send_rank0_data2_socket_.is_open()) load_send_rank0_data2_socket_.close();
    if (load_send_rank0_parity2_socket_.is_open()) load_send_rank0_parity2_socket_.close();
    if (load_send_rank1_data1_socket_.is_open()) load_send_rank1_data1_socket_.close();
    if (load_send_rank1_parity1_socket_.is_open()) load_send_rank1_parity1_socket_.close();
    if (load_send_rank3_data1_socket_.is_open()) load_send_rank3_data1_socket_.close();
    if (load_send_rank3_data2_socket_.is_open()) load_send_rank3_data2_socket_.close();

    // Two-failures load sockets [old scheme]
    if (load_twofail_peer0_socket_.is_open()) load_twofail_peer0_socket_.close();
    if (load_twofail_peer1_socket_.is_open()) load_twofail_peer1_socket_.close();
    if (load_twofail_peer0_acceptor_.is_open()) load_twofail_peer0_acceptor_.close();
    if (load_twofail_peer1_acceptor_.is_open()) load_twofail_peer1_acceptor_.close();

    // Two-failures v2 load sockets
    if (twf_surv_exch_socket_.is_open()) twf_surv_exch_socket_.close();
    if (twf_surv_exch_acceptor_.is_open()) twf_surv_exch_acceptor_.close();
    if (twf_n1_from_n3_socket_.is_open()) twf_n1_from_n3_socket_.close();
    if (twf_n1_from_n4_socket_.is_open()) twf_n1_from_n4_socket_.close();
    if (twf_n2_from_n3_socket_.is_open()) twf_n2_from_n3_socket_.close();
    if (twf_n2_from_n4_socket_.is_open()) twf_n2_from_n4_socket_.close();
    if (twf_n1_from_n3_acceptor_.is_open()) twf_n1_from_n3_acceptor_.close();
    if (twf_n1_from_n4_acceptor_.is_open()) twf_n1_from_n4_acceptor_.close();
    if (twf_n2_from_n3_acceptor_.is_open()) twf_n2_from_n3_acceptor_.close();
    if (twf_n2_from_n4_acceptor_.is_open()) twf_n2_from_n4_acceptor_.close();
}


bool send_with_size(boost::asio::ip::tcp::socket& sock, uintptr_t addr, size_t size) {
    try {
        // Use uint64_t to support data transfers > 4GB
        uint64_t sz_net = htonll(static_cast<uint64_t>(size));
        boost::asio::write(sock, boost::asio::buffer(&sz_net, sizeof(uint64_t)));
        boost::asio::write(sock, boost::asio::buffer(reinterpret_cast<void*>(addr), size));
        return true;
    } catch (...) {
        return false;
    }
}

bool recv_with_size_bool(boost::asio::ip::tcp::socket& sock, void* buf, size_t size) {
    try {
        // Use uint64_t to support data transfers > 4GB
        uint64_t sz_net = 0;
        boost::asio::read(sock, boost::asio::buffer(&sz_net, sizeof(uint64_t)));
        if (ntohll(sz_net) != static_cast<uint64_t>(size)) {
            return false;
        }
        boost::asio::read(sock, boost::asio::buffer(buf, size));
        return true;
    } catch (...) {
        return false;
    }
}


struct SendTask {
    uintptr_t addr{0};
    size_t size{0};
};

struct RecvXorTask {
    uintptr_t recv1_addr{0};
    uintptr_t recv2_addr{0};
    uintptr_t parity_addr{0};
    size_t size{0};
};

// Two-fail v2 chunked recovery pipeline task (size==0 is sentinel).
struct TwofailPipelineTask {
    size_t size{0};
    int chunk_index{0};
    uintptr_t own_d1{0};
    uintptr_t own_d2{0};
    uintptr_t own_p1{0};
    uintptr_t own_p2{0};
    uintptr_t peer_d1{0};
    uintptr_t peer_d2{0};
    uintptr_t out1{0};
    uintptr_t out2{0};
    uintptr_t out3{0};
    uintptr_t out4{0};
    uintptr_t recv_a1{0};
    uintptr_t recv_a2{0};
    uintptr_t recv_a3{0};
    uintptr_t recv_a4{0};
    uintptr_t release_addrs[6]{};
    int num_release{0};

    bool is_sentinel() const { return size == 0; }

    static TwofailPipelineTask make_sentinel() { return TwofailPipelineTask{}; }
};

enum class OnefailTaskKind : uint8_t { Send = 0, Recv = 1 };

// Single-failure (rig2) chunked load pipeline task (size==0 is sentinel).
struct OnefailPipelineTask {
    size_t size{0};
    int chunk_index{0};
    OnefailTaskKind kind{OnefailTaskKind::Recv};
    std::string send_b1_name;
    std::string send_b2_name;
    uintptr_t send_b1_addr{0};
    uintptr_t send_b2_addr{0};
    uintptr_t r0d2{0};
    uintptr_t r0p2{0};
    uintptr_t r1d1{0};
    uintptr_t r1p1{0};
    uintptr_t r3d1{0};
    uintptr_t r3d2{0};
    uintptr_t out_d1{0};
    uintptr_t out_d2{0};
    uintptr_t out_p1{0};
    uintptr_t out_p2{0};
    uintptr_t ns_recv[NS_LOAD_NUM_CHANNELS]{};
    bool noshared{false};
    uintptr_t release_addrs[NS_LOAD_NUM_CHANNELS]{};
    int num_release{0};

    bool is_sentinel() const { return size == 0; }

    static OnefailPipelineTask make_sentinel() { return OnefailPipelineTask{}; }
};

struct TensorTransferInfo {
    uintptr_t gpu_data_ptr{0};
    size_t cpu_offset{0};
    size_t size_bytes{0};
    std::vector<int64_t> shape;
    std::string name;
};

struct LayerWiseTask {
    int layer_id{0};
    std::vector<TensorTransferInfo> gpu_tensors;
    uintptr_t cpu_buffer_addr{0};  // Direct send from this address, no data buffer needed
    size_t aligned_layer_size{0};  // Aligned size for network transmission
    size_t actual_layer_size{0};   // Actual data size for writing to data_block
    uintptr_t data_block_1_addr{0};
    uintptr_t data_block_2_addr{0};
    uintptr_t parity_block_1_addr{0};
    uintptr_t parity_block_2_addr{0};
    // Recv buffer addresses (from continuous buffer pool allocated in strategy)
    uintptr_t recv1_parity1_addr{0};
    uintptr_t recv2_parity1_addr{0};
    uintptr_t recv1_parity2_addr{0};
    uintptr_t recv2_parity2_addr{0};
};

struct LayerWiseLoadTask {
    int layer_id{0};
    // For rank2 recovery: receive from other ranks
    uintptr_t recv_rank0_data2_addr{0};      // Receive rank0's data2
    uintptr_t recv_rank0_parity2_addr{0};    // Receive rank0's parity2
    uintptr_t recv_rank1_data1_addr{0};      // Receive rank1's data1
    uintptr_t recv_rank1_parity1_addr{0};    // Receive rank1's parity1
    uintptr_t recv_rank3_data1_addr{0};      // Receive rank3's data1
    uintptr_t recv_rank3_data2_addr{0};      // Receive rank3's data2
    // Recovery output buffers
    uintptr_t recovered_data1_addr{0};       // Recovered data1 block
    uintptr_t recovered_data2_addr{0};       // Recovered data2 block
    uintptr_t recovered_parity1_addr{0};     // Recovered parity1 block
    uintptr_t recovered_parity2_addr{0};     // Recovered parity2 block
    size_t layer_size{0};                    // Size of this layer
    // For H2D transfer (CPU→GPU)
    std::vector<TensorTransferInfo> gpu_tensors;
};

// ── XOR thread pool (16 workers + CPU affinity, matches ecnaive xor_pool) ────
static constexpr int kEclatinXorPoolSize = 16;
static constexpr const char* kEclatinXorCpuListEnv = "ECLATIN_XOR_CPU_LIST";

// Out-of-place 2-input XOR: dst[i] = src0[i] ^ src1[i] (no memcpy; AVX2 when available)
static void xor_two_out_of_place(uint8_t* dst, const uint8_t* src0, const uint8_t* src1, size_t len) {
#if defined(__AVX2__)
    size_t i = 0;
    const size_t simd_end = len & ~static_cast<size_t>(31);
    for (; i < simd_end; i += 32) {
        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src0 + i));
        const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src1 + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_xor_si256(v0, v1));
    }
    for (; i < len; ++i) {
        dst[i] = static_cast<uint8_t>(src0[i] ^ src1[i]);
    }
#else
    for (size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<uint8_t>(src0[i] ^ src1[i]);
    }
#endif
}

struct EclatinXorStripeOp {
    uintptr_t dst = 0;
    uintptr_t src0 = 0;
    uintptr_t src1 = 0;
};

enum class EclatinXorPoolJobKind : uint8_t {
    ThreeInput = 0,
    TwoStripe = 1,
};

struct EclatinXorPoolJob {
    EclatinXorPoolJobKind kind = EclatinXorPoolJobKind::ThreeInput;
    int len = 0;
    uintptr_t dst = 0;
    uintptr_t src1 = 0;
    uintptr_t src2 = 0;
    EclatinXorStripeOp stripe_op[2]{};
};

struct EclatinXorPoolCtx {
    class ECLATINNative* self = nullptr;
    int wid = 0;
};

class ECLATINNative {
public:
    ECLATINNative(const std::string& parity1_send1_ip, uint16_t parity1_send1_port,
                  const std::string& parity1_send2_ip, uint16_t parity1_send2_port,
                  const std::string& parity1_recv1_ip, uint16_t parity1_recv1_port,
                  const std::string& parity1_recv2_ip, uint16_t parity1_recv2_port,
                  const std::string& parity2_send1_ip, uint16_t parity2_send1_port,
                  const std::string& parity2_send2_ip, uint16_t parity2_send2_port,
                  const std::string& parity2_recv1_ip, uint16_t parity2_recv1_port,
                  const std::string& parity2_recv2_ip, uint16_t parity2_recv2_port,
                  int num_cuda_streams = ECLATIN_NUM_CUDA_STREAMS,
                  bool use_rdma = false,
                  int rank = -1,
                  int world_size = -1,
                  int rank_in_group = -1)
        : stop_(false),
          parity1_send1_ip_(parity1_send1_ip),
          parity1_send1_port_(parity1_send1_port),
          parity1_send2_ip_(parity1_send2_ip),
          parity1_send2_port_(parity1_send2_port),
          parity1_recv1_ip_(parity1_recv1_ip),
          parity1_recv1_port_(parity1_recv1_port),
          parity1_recv2_ip_(parity1_recv2_ip),
          parity1_recv2_port_(parity1_recv2_port),
          parity2_send1_ip_(parity2_send1_ip),
          parity2_send1_port_(parity2_send1_port),
          parity2_send2_ip_(parity2_send2_ip),
          parity2_send2_port_(parity2_send2_port),
          parity2_recv1_ip_(parity2_recv1_ip),
          parity2_recv1_port_(parity2_recv1_port),
          parity2_recv2_ip_(parity2_recv2_ip),
          parity2_recv2_port_(parity2_recv2_port),
          parity1_send1_completed_(false),
          parity1_send2_completed_(false),
          parity1_recv_xor_completed_(false),
          parity2_send1_completed_(false),
          parity2_send2_completed_(false),
          parity2_recv_xor_completed_(false),
          parity1_send1_sentinel_received_(false),
          parity1_send2_sentinel_received_(false),
          parity1_recv_xor_sentinel_received_(false),
          parity2_send1_sentinel_received_(false),
          parity2_send2_sentinel_received_(false),
          parity2_recv_xor_sentinel_received_(false),
          total_encoding_time_ms_(0.0),
          total_send_time_ms_(0.0),
          total_recv_time_ms_(0.0),
          total_xor_time_ms_(0.0),
          encoding_count_(0),
          send_count_(0),
          recv_count_(0),
          xor_count_(0),
          num_cuda_streams_(num_cuda_streams),
          use_async_cuda_(ECLATIN_USE_ASYNC_CUDA),
          use_rdma_(use_rdma),
          rank_(rank),
          world_size_(world_size),
          rank_in_group_(rank_in_group)
#if RDMA_AVAILABLE
        , rdma_context_(nullptr)
        , rdma_pd_(nullptr)
        , rdma_send_cq_{}
        , rdma_recv_cq_{}
        , rdma_load_send_cq_{}
        , rdma_load_recv_cq_{}
        , rdma_load_send_cq_noshared_{}
        , rdma_load_recv_cq_noshared_{}
        , rdma_load_send_cq_two_fail_{}
        , rdma_load_recv_cq_two_fail_{}
        , rdma_twf_v2_sx_send_cq_{}
        , rdma_twf_v2_sx_recv_cq_{}
        , rdma_twf_v2_n1n3_send_cq_{}
        , rdma_twf_v2_n1n3_recv_cq_{}
        , rdma_twf_v2_n1n4_send_cq_{}
        , rdma_twf_v2_n1n4_recv_cq_{}
        , rdma_twf_v2_n2n3_send_cq_{}
        , rdma_twf_v2_n2n3_recv_cq_{}
        , rdma_twf_v2_n2n4_send_cq_{}
        , rdma_twf_v2_n2n4_recv_cq_{}
#endif
    {
        const char* mode_str = use_rdma_ ? "RDMA" : "ASIO";
        std::cout << "ECLATIN: Initializing connections (mode: " << mode_str << ")..." << std::endl;
        
        #ifdef USE_CUDA
        // Initialize CUDA streams for async transfers
        if (use_async_cuda_ && num_cuda_streams_ > 0) {
            cuda_streams_.resize(num_cuda_streams_);
            for (int i = 0; i < num_cuda_streams_; ++i) {
                cudaError_t err = cudaStreamCreate(&cuda_streams_[i]);
                if (err != cudaSuccess) {
                    std::cerr << "ECLATIN: Failed to create CUDA stream " << i 
                              << ": " << cudaGetErrorString(err) << std::endl;
                    // Fallback to sync mode
                    use_async_cuda_ = false;
                    cuda_streams_.clear();
                    break;
                }
            }
            if (use_async_cuda_) {
                std::cout << "ECLATIN: Async CUDA mode enabled with " << num_cuda_streams_ 
                          << " streams" << std::endl;
            } else {
                std::cout << "ECLATIN: Falling back to sync CUDA mode" << std::endl;
            }
        } else {
            std::cout << "ECLATIN: Sync CUDA mode enabled" << std::endl;
        }
        #endif
        
        init_connections();
#if RDMA_AVAILABLE
        if (use_rdma_) {
            try {
                init_rdma_resources();
                // Initialize RDMA save channels after resources are ready
                if (rdma_pd_) {
                    init_rdma_save_channels();
                }
            } catch (const std::exception& e) {
                std::cerr << "ECLATIN: RDMA initialization failed: " << e.what() << std::endl;
                std::cerr << "ECLATIN: Falling back to ASIO" << std::endl;
                use_rdma_ = false;
                // Clear any partially initialized channels
                for (int i = 0; i < RDMA_NUM_SAVE_CHANNELS; ++i) rdma_save_channels_[i].reset();
            }
        }
#endif
        // Lazy-init zero buffer for 2-input XOR via xor_pool (resized on first use)
        twofail_zero_buffer_.clear();
        start_threads();
        // std::cout << "ECLATIN: Pipeline started successfully" << std::endl;
    }

    ~ECLATINNative() {
        stop();
        
        #ifdef USE_CUDA
        // Destroy CUDA streams
        for (auto stream : cuda_streams_) {
            cudaStreamDestroy(stream);
        }
        cuda_streams_.clear();
        #endif
    }

    // Parity 1 pipelines
    void submit_parity1_send1(uintptr_t send_addr, size_t size) {
        // std::cout << "ECLATIN: Submitting parity1_send1 task: send_addr=" << send_addr
                //   << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_send1_mutex_);
            parity1_send1_q_.push({send_addr, size});
        }
        parity1_send1_cv_.notify_one();
    }

    void submit_parity1_send2(uintptr_t send_addr, size_t size) {
        // std::cout << "ECLATIN: Submitting parity1_send2 task: send_addr=" << send_addr
                //   << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_send2_mutex_);
            parity1_send2_q_.push({send_addr, size});
        }
        parity1_send2_cv_.notify_one();
    }

    void submit_parity1_recv_xor(uintptr_t recv1_addr,
                                  uintptr_t recv2_addr,
                                  uintptr_t parity_addr,
                                  size_t size) {
        // std::cout << "ECLATIN: Submitting parity1_recv_xor task: recv1_addr=" << recv1_addr
        //           << ", recv2_addr=" << recv2_addr
        //           << ", parity_addr=" << parity_addr
        //           << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_recv_xor_mutex_);
            parity1_recv_xor_q_.push({recv1_addr, recv2_addr, parity_addr, size});
        }
        parity1_recv_xor_cv_.notify_one();
    }

    // Parity 2 pipelines
    void submit_parity2_send1(uintptr_t send_addr, size_t size) {
        // std::cout << "ECLATIN: Submitting parity2_send1 task: send_addr=" << send_addr
        //           << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_send1_mutex_);
            parity2_send1_q_.push({send_addr, size});
        }
        parity2_send1_cv_.notify_one();
    }

    void submit_parity2_send2(uintptr_t send_addr, size_t size) {
        // std::cout << "ECLATIN: Submitting parity2_send2 task: send_addr=" << send_addr
        //           << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_send2_mutex_);
            parity2_send2_q_.push({send_addr, size});
        }
        parity2_send2_cv_.notify_one();
    }

    void submit_parity2_recv_xor(uintptr_t recv1_addr,
                                  uintptr_t recv2_addr,
                                  uintptr_t parity_addr,
                                  size_t size) {
        // std::cout << "ECLATIN: Submitting parity2_recv_xor task: recv1_addr=" << recv1_addr
        //           << ", recv2_addr=" << recv2_addr
        //           << ", parity_addr=" << parity_addr
        //           << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_recv_xor_mutex_);
            parity2_recv_xor_q_.push({recv1_addr, recv2_addr, parity_addr, size});
        }
        parity2_recv_xor_cv_.notify_one();
    }

    // Submit sentinels to signal pipeline completion
    void submit_parity1_send1_sentinel() {
        // std::cout << "ECLATIN: Submitting sentinel to parity1_send1 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_send1_mutex_);
            parity1_send1_q_.push({0, 0});
        }
        parity1_send1_cv_.notify_one();
    }

    void submit_parity1_send2_sentinel() {
        // std::cout << "ECLATIN: Submitting sentinel to parity1_send2 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_send2_mutex_);
            parity1_send2_q_.push({0, 0});
        }
        parity1_send2_cv_.notify_one();
    }

    void submit_parity1_recv_xor_sentinel() {
        // std::cout << "ECLATIN: Submitting sentinel to parity1_recv_xor pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_recv_xor_mutex_);
            parity1_recv_xor_q_.push({0, 0, 0, 0});
        }
        parity1_recv_xor_cv_.notify_one();
    }

    void submit_parity2_send1_sentinel() {
        // std::cout << "ECLATIN: Submitting sentinel to parity2_send1 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_send1_mutex_);
            parity2_send1_q_.push({0, 0});
        }
        parity2_send1_cv_.notify_one();
    }

    void submit_parity2_send2_sentinel() {
        // std::cout << "ECLATIN: Submitting sentinel to parity2_send2 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_send2_mutex_);
            parity2_send2_q_.push({0, 0});
        }
        parity2_send2_cv_.notify_one();
    }

    void submit_parity2_recv_xor_sentinel() {
        // std::cout << "ECLATIN: Submitting sentinel to parity2_recv_xor pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_recv_xor_mutex_);
            parity2_recv_xor_q_.push({0, 0, 0, 0});
        }
        parity2_recv_xor_cv_.notify_one();
    }

    // Release helpers: Python can poll these to free buffers.
    // Data buffers are released after send operations complete
    std::vector<uintptr_t> get_data_buffers_to_release() {
        std::vector<uintptr_t> buffers;
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        while (!data_buffers_to_release_.empty()) {
            buffers.push_back(data_buffers_to_release_.front());
            data_buffers_to_release_.pop();
        }
        return buffers;
    }
    
    // Recv buffers are released after recv_xor operations complete (XOR done)
    std::vector<uintptr_t> get_recv_buffers_to_release() {
        std::vector<uintptr_t> buffers;
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        while (!recv_buffers_to_release_.empty()) {
            buffers.push_back(recv_buffers_to_release_.front());
            recv_buffers_to_release_.pop();
        }
        return buffers;
    }

    void reset_encoding_completion_flags() {
        // Parity 1 flags
        parity1_send1_completed_ = false;
        parity1_send2_completed_ = false;
        parity1_recv_xor_completed_ = false;
        
        parity1_send1_sentinel_received_ = false;
        parity1_send2_sentinel_received_ = false;
        parity1_recv_xor_sentinel_received_ = false;
        
        // Parity 2 flags
        parity2_send1_completed_ = false;
        parity2_send2_completed_ = false;
        parity2_recv_xor_completed_ = false;
        parity2_send1_sentinel_received_ = false;
        parity2_send2_sentinel_received_ = false;
        parity2_recv_xor_sentinel_received_ = false;

        // Reset time statistics at the start of each checkpoint
        reset_time_statistics();
        
        // Start pipeline timing
        pipeline_start_time_ = std::chrono::high_resolution_clock::now();
        pipeline_timing_started_ = true;

        // Clear queues
        {
            std::lock_guard<std::mutex> lock(parity1_send1_mutex_);
            while (!parity1_send1_q_.empty()) parity1_send1_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(parity1_send2_mutex_);
            while (!parity1_send2_q_.empty()) parity1_send2_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(parity1_recv_xor_mutex_);
            while (!parity1_recv_xor_q_.empty()) parity1_recv_xor_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(parity2_send1_mutex_);
            while (!parity2_send1_q_.empty()) parity2_send1_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(parity2_send2_mutex_);
            while (!parity2_send2_q_.empty()) parity2_send2_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(parity2_recv_xor_mutex_);
            while (!parity2_recv_xor_q_.empty()) parity2_recv_xor_q_.pop();
        }

        std::cout << "ECLATIN: Reset encoding completion flags and cleared all queues" << std::endl;
    }

    void wait_for_encoding_completion() {
        // Wait for all 6 workers
        int wait_count = 0;
        while (!parity1_send1_completed_ || !parity1_send2_completed_ || !parity1_recv_xor_completed_ ||
               !parity2_send1_completed_ || !parity2_send2_completed_ || !parity2_recv_xor_completed_) {
            if (wait_count % 100 == 0) {
                std::cout << "ECLATIN: Waiting for workers: "
                          << "p1_s1=" << (parity1_send1_completed_ ? "true" : "false")
                          << ", p1_s2=" << (parity1_send2_completed_ ? "true" : "false")
                          << ", p1_rx=" << (parity1_recv_xor_completed_ ? "true" : "false")
                          << ", p2_s1=" << (parity2_send1_completed_ ? "true" : "false")
                          << ", p2_s2=" << (parity2_send2_completed_ ? "true" : "false")
                          << ", p2_rx=" << (parity2_recv_xor_completed_ ? "true" : "false") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
        }
        std::cout << "ECLATIN: All workers completed" << std::endl;
        
        // Calculate pipeline wall-clock time
        double pipeline_wall_time_ms = 0.0;
        if (pipeline_timing_started_) {
            auto pipeline_end_time = std::chrono::high_resolution_clock::now();
            pipeline_wall_time_ms = std::chrono::duration<double, std::milli>(pipeline_end_time - pipeline_start_time_).count();
        }
        
        // Print time statistics after all operations complete
        print_time_statistics(pipeline_wall_time_ms);
    }

    void stop() {
        bool expected = false;
        if (!stop_.compare_exchange_strong(expected, true)) {
            return;  // already stopped
        }
        parity1_send1_cv_.notify_all();
        parity1_send2_cv_.notify_all();
        parity1_recv_xor_cv_.notify_all();
        parity2_send1_cv_.notify_all();
        parity2_send2_cv_.notify_all();
        parity2_recv_xor_cv_.notify_all();
        layerwise_cv_.notify_all();
        layerwise_load_cv_.notify_all();
        twofail_exch_cv_.notify_all();
        twofail_xor_cv_.notify_all();
        twofail_fwd_cv_.notify_all();
        onefail_net_cv_.notify_all();
        onefail_xor_cv_.notify_all();
        if (parity1_recv_xor_thread_.joinable()) parity1_recv_xor_thread_.join();
        if (parity1_send1_thread_.joinable()) parity1_send1_thread_.join();
        if (parity1_send2_thread_.joinable()) parity1_send2_thread_.join();
        if (parity2_recv_xor_thread_.joinable()) parity2_recv_xor_thread_.join();
        if (parity2_send1_thread_.joinable()) parity2_send1_thread_.join();
        if (parity2_send2_thread_.joinable()) parity2_send2_thread_.join();
        if (layerwise_worker_thread_.joinable()) layerwise_worker_thread_.join();
        if (layerwise_load_worker_thread_.joinable()) layerwise_load_worker_thread_.join();
        if (twofail_exch_thread_.joinable()) twofail_exch_thread_.join();
        if (twofail_xor_thread_.joinable()) twofail_xor_thread_.join();
        if (twofail_fwd_thread_.joinable()) twofail_fwd_thread_.join();
        if (onefail_net_thread_.joinable()) onefail_net_thread_.join();
        if (onefail_xor_thread_.joinable()) onefail_xor_thread_.join();
        xor_pool_shutdown();
#if RDMA_AVAILABLE
        cleanup_rdma_resources();
#endif
        conn_.cleanup();
    }

    void register_buffer(uintptr_t buffer_addr, size_t buffer_size) {
#if RDMA_AVAILABLE
        if (use_rdma_ && rdma_pd_) {
            std::lock_guard<std::mutex> lock(rdma_buffer_mutex_);
            if (rdma_registered_buffers_.count(buffer_addr)) return;
            void* ptr = reinterpret_cast<void*>(buffer_addr);
            ibv_mr* mr = ibv_reg_mr(rdma_pd_, ptr, buffer_size,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
            if (!mr) throw std::runtime_error("ECLATIN RDMA: ibv_reg_mr failed");
            rdma_registered_buffers_[buffer_addr] = RdmaBuffer{mr, buffer_addr, buffer_size};
        }
#endif
    }

    void unregister_buffer(uintptr_t buffer_addr) {
#if RDMA_AVAILABLE
        if (!use_rdma_) return;
        std::lock_guard<std::mutex> lock(rdma_buffer_mutex_);
        auto it = rdma_registered_buffers_.find(buffer_addr);
        if (it == rdma_registered_buffers_.end()) return;
        if (it->second.mr) ibv_dereg_mr(it->second.mr);
        rdma_registered_buffers_.erase(it);
#endif
    }

    // Load mode functions
    void set_load_mode(bool is_load, int failed_rank) {
        is_load_mode_ = is_load;
        failed_rank_ = failed_rank;
        failed_rank_in_group_ = (failed_rank >= 0)
            ? (failed_rank % ECLATIN_RANKS_PER_GROUP)
            : -1;
        std::cout << "ECLATIN: Set load mode: "
                  << (is_load ? "true" : "false") << ", failed_rank=" << failed_rank
                  << ", failed_rank_in_group=" << failed_rank_in_group_ << std::endl;
    }

    void set_no_shared_block(bool enabled) {
        use_no_shared_block_ = enabled;
        std::cout << "ECLATIN: Set no-shared-block mode: "
                  << (enabled ? "true" : "false") << std::endl;
    }

    void init_load_connections(
        int rank_in_group,
        const std::string& rank2_ip,
        uint16_t load_recv_rank0_data2_port,
        uint16_t load_recv_rank0_parity2_port,
        uint16_t load_recv_rank1_data1_port,
        uint16_t load_recv_rank1_parity1_port,
        uint16_t load_recv_rank3_data1_port,
        uint16_t load_recv_rank3_data2_port
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: init_load_connections called but not in load mode" << std::endl;
            return;
        }
        
        std::cout << "ECLATIN: [Rank_in_group " << rank_in_group << "] Initializing load connections..." << std::endl;
        
        if (rank_in_group == 2) {
            // rank_in_group 2 (receiver): bind+listen 6 acceptors, then each accept
            // thread does accept() + QP exchange inline (same thread → no boost::asio
            // io_context race on socket native_handle).
            std::cout << "ECLATIN: [Rank_in_group 2] Binding and listening all acceptors..." << std::endl;
            try {
                conn_.bind_listen_load_recv_rank0_data2(rank2_ip, load_recv_rank0_data2_port);
                conn_.bind_listen_load_recv_rank0_parity2(rank2_ip, load_recv_rank0_parity2_port);
                conn_.bind_listen_load_recv_rank1_data1(rank2_ip, load_recv_rank1_data1_port);
                conn_.bind_listen_load_recv_rank1_parity1(rank2_ip, load_recv_rank1_parity1_port);
                conn_.bind_listen_load_recv_rank3_data1(rank2_ip, load_recv_rank3_data1_port);
                conn_.bind_listen_load_recv_rank3_data2(rank2_ip, load_recv_rank3_data2_port);
                std::cout << "ECLATIN: [Rank_in_group 2] All acceptors bound and listening" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "ECLATIN: [Rank_in_group 2] Failed to bind/listen acceptors: " << e.what() << std::endl;
                throw;
            }

            // Init RDMA resources now (accept threads will use them for QP exchange).
            if (use_rdma_ && rdma_pd_ && rdma_load_send_cq_[0] == nullptr) {
                init_rdma_load_resources();
            }

            // Each accept thread: accept() → create channel → QP exchange → done.
            // Overlaps accept+QP across 6 sockets; no detached thread → no race.
            std::thread r0_d2([this, rank_in_group]() {
                conn_.accept_load_recv_rank0_data2();
                if (use_rdma_ && rdma_pd_) {
                    rdma_load_channels_[0] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[0], rdma_load_recv_cq_[0],
                        conn_.get_load_recv_rank0_data2_socket().native_handle(),
                        conn_.get_load_recv_rank0_data2_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 0);
                    rdma_load_channels_[0]->exchange_and_connect(false);
                }
            });
            std::thread r0_p2([this, rank_in_group]() {
                conn_.accept_load_recv_rank0_parity2();
                if (use_rdma_ && rdma_pd_) {
                    rdma_load_channels_[1] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[1], rdma_load_recv_cq_[1],
                        conn_.get_load_recv_rank0_parity2_socket().native_handle(),
                        conn_.get_load_recv_rank0_parity2_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 0);
                    rdma_load_channels_[1]->exchange_and_connect(false);
                }
            });
            std::thread r1_d1([this, rank_in_group]() {
                conn_.accept_load_recv_rank1_data1();
                if (use_rdma_ && rdma_pd_) {
                    rdma_load_channels_[2] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[2], rdma_load_recv_cq_[2],
                        conn_.get_load_recv_rank1_data1_socket().native_handle(),
                        conn_.get_load_recv_rank1_data1_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 0);
                    rdma_load_channels_[2]->exchange_and_connect(false);
                }
            });
            std::thread r1_p1([this, rank_in_group]() {
                conn_.accept_load_recv_rank1_parity1();
                if (use_rdma_ && rdma_pd_) {
                    rdma_load_channels_[3] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[3], rdma_load_recv_cq_[3],
                        conn_.get_load_recv_rank1_parity1_socket().native_handle(),
                        conn_.get_load_recv_rank1_parity1_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 0);
                    rdma_load_channels_[3]->exchange_and_connect(false);
                }
            });
            std::thread r3_d1([this, rank_in_group]() {
                conn_.accept_load_recv_rank3_data1();
                if (use_rdma_ && rdma_pd_) {
                    rdma_load_channels_[4] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[4], rdma_load_recv_cq_[4],
                        conn_.get_load_recv_rank3_data1_socket().native_handle(),
                        conn_.get_load_recv_rank3_data1_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 0);
                    rdma_load_channels_[4]->exchange_and_connect(false);
                }
            });
            std::thread r3_d2([this, rank_in_group]() {
                conn_.accept_load_recv_rank3_data2();
                if (use_rdma_ && rdma_pd_) {
                    rdma_load_channels_[5] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[5], rdma_load_recv_cq_[5],
                        conn_.get_load_recv_rank3_data2_socket().native_handle(),
                        conn_.get_load_recv_rank3_data2_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 0);
                    rdma_load_channels_[5]->exchange_and_connect(false);
                }
            });
            r0_d2.detach(); r0_p2.detach(); r1_d1.detach();
            r1_p1.detach(); r3_d1.detach(); r3_d2.detach();
            std::cout << "ECLATIN: [Rank_in_group 2] 6 accept+QP threads started (detached)" << std::endl;
        } else {
            // rank_in_group 0/1/3: TCP connect + QP exchange inline (same thread).
            if (use_rdma_ && rdma_pd_ && rdma_load_send_cq_[0] == nullptr) {
                init_rdma_load_resources();
            }
            if (rank_in_group == 0) {
                std::cout << "ECLATIN: [Rank_in_group 0] Connecting + QP exchange..." << std::endl;
                conn_.init_load_send_rank0_data2(rank2_ip, load_recv_rank0_data2_port);
                conn_.init_load_send_rank0_parity2(rank2_ip, load_recv_rank0_parity2_port);
                if (use_rdma_ && rdma_pd_) {
                    rdma_load_channels_[0] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[0], rdma_load_recv_cq_[0],
                        conn_.get_load_send_rank0_data2_socket().native_handle(),
                        conn_.get_load_send_rank0_data2_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 2);
                    rdma_load_channels_[0]->exchange_and_connect(true);
                    rdma_load_channels_[1] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[1], rdma_load_recv_cq_[1],
                        conn_.get_load_send_rank0_parity2_socket().native_handle(),
                        conn_.get_load_send_rank0_parity2_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 2);
                    rdma_load_channels_[1]->exchange_and_connect(true);
                }
            } else if (rank_in_group == 1) {
                std::cout << "ECLATIN: [Rank_in_group 1] Connecting + QP exchange..." << std::endl;
                conn_.init_load_send_rank1_data1(rank2_ip, load_recv_rank1_data1_port);
                conn_.init_load_send_rank1_parity1(rank2_ip, load_recv_rank1_parity1_port);
                if (use_rdma_ && rdma_pd_) {
                    rdma_load_channels_[2] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[2], rdma_load_recv_cq_[2],
                        conn_.get_load_send_rank1_data1_socket().native_handle(),
                        conn_.get_load_send_rank1_data1_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 2);
                    rdma_load_channels_[2]->exchange_and_connect(true);
                    rdma_load_channels_[3] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[3], rdma_load_recv_cq_[3],
                        conn_.get_load_send_rank1_parity1_socket().native_handle(),
                        conn_.get_load_send_rank1_parity1_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 2);
                    rdma_load_channels_[3]->exchange_and_connect(true);
                }
            } else if (rank_in_group == 3) {
                std::cout << "ECLATIN: [Rank_in_group 3] Connecting + QP exchange..." << std::endl;
                conn_.init_load_send_rank3_data1(rank2_ip, load_recv_rank3_data1_port);
                conn_.init_load_send_rank3_data2(rank2_ip, load_recv_rank3_data2_port);
                if (use_rdma_ && rdma_pd_) {
                    rdma_load_channels_[4] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[4], rdma_load_recv_cq_[4],
                        conn_.get_load_send_rank3_data1_socket().native_handle(),
                        conn_.get_load_send_rank3_data1_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 2);
                    rdma_load_channels_[4]->exchange_and_connect(true);
                    rdma_load_channels_[5] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_, rdma_load_send_cq_[5], rdma_load_recv_cq_[5],
                        conn_.get_load_send_rank3_data2_socket().native_handle(),
                        conn_.get_load_send_rank3_data2_socket().native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group, 2);
                    rdma_load_channels_[5]->exchange_and_connect(true);
                }
            }
        }
        
        std::cout << "ECLATIN: [Rank_in_group " << rank_in_group << "] Load connections initialized" << std::endl;
    }

    void init_load_connections_noshared(
        int rank_in_group,
        const std::string& rank2_ip,
        uint16_t port_n1_d1,
        uint16_t port_n3_p1,
        uint16_t port_n0_d0,
        uint16_t port_n3_p0,
        uint16_t port_n1_d0,
        uint16_t port_n3_d1,
        uint16_t port_n0_d1,
        uint16_t port_n3_d0
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: init_load_connections_noshared called but not in load mode" << std::endl;
            return;
        }
        if (!use_no_shared_block_) {
            throw std::runtime_error(
                "ECLATIN: init_load_connections_noshared requires no-shared-block mode");
        }

        const uint16_t ports[NS_LOAD_NUM_CHANNELS] = {
            port_n1_d1, port_n3_p1, port_n0_d0, port_n3_p0,
            port_n1_d0, port_n3_d1, port_n0_d1, port_n3_d0
        };

        conn_.set_ns_load_active(true);
        std::cout << "ECLATIN: [Rank_in_group " << rank_in_group
                  << "] Initializing no-shared load connections (8 channels)..." << std::endl;

        if (rank_in_group == 2) {
            const std::string listen_ip = rank2_ip;
            for (int ch = 0; ch < NS_LOAD_NUM_CHANNELS; ++ch) {
                conn_.bind_listen_ns_load_recv(ch, listen_ip, ports[ch]);
            }
            if (use_rdma_ && rdma_pd_ && rdma_load_send_cq_noshared_[0] == nullptr) {
                init_rdma_load_resources_noshared();
            }
            std::vector<std::thread> accept_threads;
            accept_threads.reserve(NS_LOAD_NUM_CHANNELS);
            for (int ch = 0; ch < NS_LOAD_NUM_CHANNELS; ++ch) {
                accept_threads.emplace_back([this, ch]() {
                    conn_.accept_ns_load_recv(ch);
                    setup_ns_rdma_load_channel(ch, false, 0);
                });
            }
            for (auto& t : accept_threads) {
                t.detach();
            }
        } else {
            if (use_rdma_ && rdma_pd_ && rdma_load_send_cq_noshared_[0] == nullptr) {
                init_rdma_load_resources_noshared();
            }
            if (rank_in_group == 0) {
                conn_.init_ns_load_send(2, rank2_ip, ports[2]);
                conn_.init_ns_load_send(6, rank2_ip, ports[6]);
                setup_ns_rdma_load_channel(2, true, 2);
                setup_ns_rdma_load_channel(6, true, 2);
            } else if (rank_in_group == 1) {
                conn_.init_ns_load_send(0, rank2_ip, ports[0]);
                conn_.init_ns_load_send(4, rank2_ip, ports[4]);
                setup_ns_rdma_load_channel(0, true, 2);
                setup_ns_rdma_load_channel(4, true, 2);
            } else if (rank_in_group == 3) {
                conn_.init_ns_load_send(1, rank2_ip, ports[1]);
                conn_.init_ns_load_send(3, rank2_ip, ports[3]);
                conn_.init_ns_load_send(5, rank2_ip, ports[5]);
                conn_.init_ns_load_send(7, rank2_ip, ports[7]);
                setup_ns_rdma_load_channel(1, true, 2);
                setup_ns_rdma_load_channel(3, true, 2);
                setup_ns_rdma_load_channel(5, true, 2);
                setup_ns_rdma_load_channel(7, true, 2);
            } else {
                throw std::runtime_error(
                    "ECLATIN: init_load_connections_noshared unexpected rank_in_group");
            }
        }

        std::cout << "ECLATIN: [Rank_in_group " << rank_in_group
                  << "] No-shared load connections initialized" << std::endl;
    }

    void init_two_failures_load_connections(
        int rank_in_group,
        const std::string& peer0_ip, uint16_t peer0_port,
        const std::string& peer1_ip, uint16_t peer1_port
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: init_two_failures_load_connections called but not in load mode" << std::endl;
            return;
        }

        std::cout << "ECLATIN: [Two-fail] rank_in_group " << rank_in_group
                  << " initializing two-failures load connections..." << std::endl;

        if (rank_in_group == 0 || rank_in_group == 1) {
            // Failed ranks (0, 1): bind+listen on both peer ports, then accept in threads
            std::cout << "ECLATIN: [Two-fail] Failed rank " << rank_in_group
                      << " binding/listening peer0=" << peer0_ip << ":" << peer0_port
                      << " peer1=" << peer1_ip << ":" << peer1_port << std::endl;
            conn_.bind_listen_load_twofail_peer0(peer0_ip, peer0_port);
            conn_.bind_listen_load_twofail_peer1(peer1_ip, peer1_port);

            std::thread recv_init_thread([this]() {
                std::thread t0([this]() { conn_.accept_load_twofail_peer0(); });
                std::thread t1([this]() { conn_.accept_load_twofail_peer1(); });
                t0.join();
                t1.join();
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            recv_init_thread.detach();
            std::cout << "ECLATIN: [Two-fail] Failed rank " << rank_in_group
                      << " accept threads started" << std::endl;
        } else if (rank_in_group == 2 || rank_in_group == 3) {
            // Surviving ranks (2, 3): connect send sockets to both failed ranks
            std::cout << "ECLATIN: [Two-fail] Surviving rank " << rank_in_group
                      << " connecting to failed ranks..." << std::endl;
            conn_.init_load_send_twofail_peer0(peer0_ip, peer0_port);
            conn_.init_load_send_twofail_peer1(peer1_ip, peer1_port);
            std::cout << "ECLATIN: [Two-fail] Surviving rank " << rank_in_group
                      << " both send sockets connected" << std::endl;
        } else {
            std::cerr << "ECLATIN: [Two-fail] unexpected rank_in_group=" << rank_in_group << std::endl;
            throw std::runtime_error("ECLATIN: init_two_failures_load_connections: unexpected rank_in_group");
        }

        std::cout << "ECLATIN: [Two-fail] rank_in_group " << rank_in_group
                  << " load connections initialized" << std::endl;
    }

    void wait_for_load_connections(int timeout_seconds = 30) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: wait_for_load_connections called but not in load mode" << std::endl;
            return;
        }
        if (use_no_shared_block_) {
            conn_.wait_for_ns_load_connections(rank_in_group_, timeout_seconds);
#if RDMA_AVAILABLE
            if (use_rdma_ && rdma_pd_ && rdma_load_send_cq_noshared_[0] != nullptr) {
                std::vector<int> channels;
                if (rank_in_group_ == 2) {
                    for (int i = 0; i < NS_LOAD_NUM_CHANNELS; ++i) channels.push_back(i);
                } else if (rank_in_group_ == 0) {
                    channels = {2, 6};
                } else if (rank_in_group_ == 1) {
                    channels = {0, 4};
                } else if (rank_in_group_ == 3) {
                    channels = {1, 3, 5, 7};
                }
                int spin = 0;
                for (int ch : channels) {
                    while (rdma_load_channels_noshared_[ch] == nullptr) {
                        if (++spin % 100 == 0) {
                            std::cout << "[ECLATIN RDMA] Waiting for ns load channel "
                                      << ch << "..." << std::endl;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                }
            }
#endif
            return;
        }
        conn_.wait_for_load_connections(timeout_seconds);
        // If RDMA load channels were started asynchronously (single-failure rank2),
        // spin until the detached accept+QP threads have finished creating them.
        if (use_rdma_ && rdma_pd_ && rdma_load_send_cq_[0] != nullptr) {
            int spin = 0;
            // Each rank only creates a subset: rig2 creates [0..5], others create 2 each
            int start_ch = 0, end_ch = RDMA_NUM_LOAD_CHANNELS;
            if (rank_in_group_ == 0)      end_ch = 2;
            else if (rank_in_group_ == 1) { start_ch = 2; end_ch = 4; }
            else if (rank_in_group_ == 3) { start_ch = 4; end_ch = 6; }
            for (int i = start_ch; i < end_ch; ++i) {
                while (rdma_load_channels_[i] == nullptr) {
                    if (++spin % 100 == 0)
                        std::cout << "[ECLATIN RDMA] Waiting for load channel " << i << "..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        }
#if RDMA_AVAILABLE
        // Two-failures v2 RDMA init
        bool is_twf_v2 = conn_.is_twf_v2_connected();

        // v2 two-fail: init dedicated RDMA resources and channels
        if (is_twf_v2 && use_rdma_ && rdma_pd_ && rdma_twf_v2_sx_send_cq_[0] == nullptr) {
            try {
                init_rdma_twf_v2_resources();
                init_rdma_twf_v2_channels();
            } catch (const std::exception& e) {
                std::cerr << "ECLATIN: Two-fail v2 RDMA channels init failed: " << e.what() << std::endl;
                throw;
            }
        }

        // Two-failures RDMA load channels (old scheme only)
        bool is_twofail_old = (!is_twf_v2) && (
            failed_rank_in_group_ == 10 ||
            conn_.is_load_twofail_peer0_connected() ||
            conn_.is_load_twofail_peer1_connected());
        if (use_rdma_ && rdma_pd_ && rdma_load_send_cq_two_fail_[0] == nullptr && is_twofail_old) {
            try {
                init_rdma_load_resources_two_fail();
                init_rdma_load_channels_two_fail();
            } catch (const std::exception& e) {
                std::cerr << "ECLATIN: Two-fail load RDMA channels init failed: " << e.what() << std::endl;
                throw;
            }
        }
        // Normal load RDMA channels (single-failure / software recovery).
        // init_load_connections already did accept+QP exchange inline for all ranks,
        // so the channels should already exist.  Only init if guard says they don't.
        else if (!is_twf_v2 && use_rdma_ && rdma_pd_ && rdma_load_send_cq_[0] == nullptr) {
            try {
                init_rdma_load_resources();
                init_rdma_load_channels();
            } catch (const std::exception& e) {
                std::cerr << "ECLATIN: Load RDMA channels init failed: " << e.what() << std::endl;
                throw;
            }
        }
#endif
    }

    // ── One-fail (rig2) chunk-level helpers ──────────────────────────────────

    void onefail_recv_chunk_(
        uintptr_t rank0_data2_addr,
        uintptr_t rank0_parity2_addr,
        uintptr_t rank1_data1_addr,
        uintptr_t rank1_parity1_addr,
        uintptr_t rank3_data1_addr,
        uintptr_t rank3_data2_addr,
        size_t size
    ) {
        std::vector<std::exception_ptr> recv_exceptions(6);
        std::vector<std::thread> recv_threads;

        recv_threads.emplace_back([&]() {
            try {
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_load_channels_[0]) {
                    rdma_load_channels_[0]->receive_data(
                        reinterpret_cast<uint8_t*>(rank0_data2_addr), size);
                } else
#endif
                if (!recv_with_size_bool(conn_.get_load_recv_rank0_data2_socket(),
                                        reinterpret_cast<void*>(rank0_data2_addr), size)) {
                    throw std::runtime_error("Failed to receive rank0_data2");
                }
            } catch (...) {
                recv_exceptions[0] = std::current_exception();
            }
        });

        recv_threads.emplace_back([&]() {
            try {
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_load_channels_[1]) {
                    rdma_load_channels_[1]->receive_data(
                        reinterpret_cast<uint8_t*>(rank0_parity2_addr), size);
                } else
#endif
                if (!recv_with_size_bool(conn_.get_load_recv_rank0_parity2_socket(),
                                        reinterpret_cast<void*>(rank0_parity2_addr), size)) {
                    throw std::runtime_error("Failed to receive rank0_parity2");
                }
            } catch (...) {
                recv_exceptions[1] = std::current_exception();
            }
        });

        recv_threads.emplace_back([&]() {
            try {
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_load_channels_[2]) {
                    rdma_load_channels_[2]->receive_data(
                        reinterpret_cast<uint8_t*>(rank1_data1_addr), size);
                } else
#endif
                if (!recv_with_size_bool(conn_.get_load_recv_rank1_data1_socket(),
                                        reinterpret_cast<void*>(rank1_data1_addr), size)) {
                    throw std::runtime_error("Failed to receive rank1_data1");
                }
            } catch (...) {
                recv_exceptions[2] = std::current_exception();
            }
        });

        recv_threads.emplace_back([&]() {
            try {
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_load_channels_[3]) {
                    rdma_load_channels_[3]->receive_data(
                        reinterpret_cast<uint8_t*>(rank1_parity1_addr), size);
                } else
#endif
                if (!recv_with_size_bool(conn_.get_load_recv_rank1_parity1_socket(),
                                        reinterpret_cast<void*>(rank1_parity1_addr), size)) {
                    throw std::runtime_error("Failed to receive rank1_parity1");
                }
            } catch (...) {
                recv_exceptions[3] = std::current_exception();
            }
        });

        recv_threads.emplace_back([&]() {
            try {
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_load_channels_[4]) {
                    rdma_load_channels_[4]->receive_data(
                        reinterpret_cast<uint8_t*>(rank3_data1_addr), size);
                } else
#endif
                if (!recv_with_size_bool(conn_.get_load_recv_rank3_data1_socket(),
                                        reinterpret_cast<void*>(rank3_data1_addr), size)) {
                    throw std::runtime_error("Failed to receive rank3_data1");
                }
            } catch (...) {
                recv_exceptions[4] = std::current_exception();
            }
        });

        recv_threads.emplace_back([&]() {
            try {
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_load_channels_[5]) {
                    rdma_load_channels_[5]->receive_data(
                        reinterpret_cast<uint8_t*>(rank3_data2_addr), size);
                } else
#endif
                if (!recv_with_size_bool(conn_.get_load_recv_rank3_data2_socket(),
                                        reinterpret_cast<void*>(rank3_data2_addr), size)) {
                    throw std::runtime_error("Failed to receive rank3_data2");
                }
            } catch (...) {
                recv_exceptions[5] = std::current_exception();
            }
        });

        for (auto& t : recv_threads) {
            t.join();
        }
        for (size_t i = 0; i < recv_exceptions.size(); ++i) {
            if (recv_exceptions[i]) {
                std::rethrow_exception(recv_exceptions[i]);
            }
        }
    }

    void onefail_xor_chunk_(
        uintptr_t rank0_data2_addr,
        uintptr_t rank0_parity2_addr,
        uintptr_t rank1_data1_addr,
        uintptr_t rank1_parity1_addr,
        uintptr_t rank3_data1_addr,
        uintptr_t rank3_data2_addr,
        uintptr_t recovered_data1_addr,
        uintptr_t recovered_data2_addr,
        uintptr_t recovered_parity1_addr,
        uintptr_t recovered_parity2_addr,
        size_t size
    ) {
        std::vector<std::exception_ptr> xor_exceptions(4);
        std::vector<std::thread> xor_threads;

        xor_threads.emplace_back([&]() {
            try {
                xor_two_out_of_place(
                    reinterpret_cast<uint8_t*>(recovered_data1_addr),
                    reinterpret_cast<const uint8_t*>(rank0_data2_addr),
                    reinterpret_cast<const uint8_t*>(rank1_parity1_addr),
                    size);
            } catch (...) {
                xor_exceptions[0] = std::current_exception();
            }
        });

        xor_threads.emplace_back([&]() {
            try {
                xor_two_out_of_place(
                    reinterpret_cast<uint8_t*>(recovered_data2_addr),
                    reinterpret_cast<const uint8_t*>(rank0_parity2_addr),
                    reinterpret_cast<const uint8_t*>(rank1_data1_addr),
                    size);
            } catch (...) {
                xor_exceptions[1] = std::current_exception();
            }
        });

        xor_threads.emplace_back([&]() {
            try {
                xor_two_out_of_place(
                    reinterpret_cast<uint8_t*>(recovered_parity1_addr),
                    reinterpret_cast<const uint8_t*>(rank1_data1_addr),
                    reinterpret_cast<const uint8_t*>(rank3_data2_addr),
                    size);
            } catch (...) {
                xor_exceptions[2] = std::current_exception();
            }
        });

        xor_threads.emplace_back([&]() {
            try {
                xor_two_out_of_place(
                    reinterpret_cast<uint8_t*>(recovered_parity2_addr),
                    reinterpret_cast<const uint8_t*>(rank0_data2_addr),
                    reinterpret_cast<const uint8_t*>(rank3_data1_addr),
                    size);
            } catch (...) {
                xor_exceptions[3] = std::current_exception();
            }
        });

        for (auto& t : xor_threads) {
            t.join();
        }
        for (size_t i = 0; i < xor_exceptions.size(); ++i) {
            if (xor_exceptions[i]) {
                std::rethrow_exception(xor_exceptions[i]);
            }
        }
    }

    static int ns_load_channel_from_name(const std::string& block_name) {
        if (block_name == "ns_n1_d1") return 0;
        if (block_name == "ns_n3_p1") return 1;
        if (block_name == "ns_n0_d0") return 2;
        if (block_name == "ns_n3_p0") return 3;
        if (block_name == "ns_n1_d0") return 4;
        if (block_name == "ns_n3_d1") return 5;
        if (block_name == "ns_n0_d1") return 6;
        if (block_name == "ns_n3_d0") return 7;
        return -1;
    }

    void onefail_noshared_recv_chunk_(const uintptr_t recv_addrs[NS_LOAD_NUM_CHANNELS], size_t size) {
        std::vector<std::exception_ptr> recv_exceptions(NS_LOAD_NUM_CHANNELS);
        std::vector<std::thread> recv_threads;
        recv_threads.reserve(NS_LOAD_NUM_CHANNELS);

        for (int ch = 0; ch < NS_LOAD_NUM_CHANNELS; ++ch) {
            recv_threads.emplace_back([&, ch]() {
                try {
#if RDMA_AVAILABLE
                    if (use_rdma_ && rdma_load_channels_noshared_[ch]) {
                        rdma_load_channels_noshared_[ch]->receive_data(
                            reinterpret_cast<uint8_t*>(recv_addrs[ch]), size);
                    } else
#endif
                    if (!recv_with_size_bool(conn_.get_ns_load_recv_socket(ch),
                                            reinterpret_cast<void*>(recv_addrs[ch]), size)) {
                        throw std::runtime_error(
                            "Failed to receive ns_load channel " + std::to_string(ch));
                    }
                } catch (...) {
                    recv_exceptions[ch] = std::current_exception();
                }
            });
        }

        for (auto& t : recv_threads) {
            t.join();
        }
        for (int i = 0; i < NS_LOAD_NUM_CHANNELS; ++i) {
            if (recv_exceptions[i]) {
                std::rethrow_exception(recv_exceptions[i]);
            }
        }
    }

    void onefail_noshared_xor_chunk_(
        const uintptr_t recv_addrs[NS_LOAD_NUM_CHANNELS],
        uintptr_t out_d0,
        uintptr_t out_d1,
        uintptr_t out_p0,
        uintptr_t out_p1,
        size_t size
    ) {
        // node2_data0   = node1_data1  XOR node3_parity1  -> ch0 XOR ch1
        // node2_data1   = node0_data0  XOR node3_parity0  -> ch2 XOR ch3
        // node2_parity0 = node1_data0  XOR node3_data1    -> ch4 XOR ch5
        // node2_parity1 = node0_data1  XOR node3_data0    -> ch6 XOR ch7
        std::vector<std::exception_ptr> xor_exceptions(4);
        std::vector<std::thread> xor_threads;

        xor_threads.emplace_back([&]() {
            try {
                xor_two_out_of_place(
                    reinterpret_cast<uint8_t*>(out_d0),
                    reinterpret_cast<const uint8_t*>(recv_addrs[0]),
                    reinterpret_cast<const uint8_t*>(recv_addrs[1]),
                    size);
            } catch (...) {
                xor_exceptions[0] = std::current_exception();
            }
        });
        xor_threads.emplace_back([&]() {
            try {
                xor_two_out_of_place(
                    reinterpret_cast<uint8_t*>(out_d1),
                    reinterpret_cast<const uint8_t*>(recv_addrs[2]),
                    reinterpret_cast<const uint8_t*>(recv_addrs[3]),
                    size);
            } catch (...) {
                xor_exceptions[1] = std::current_exception();
            }
        });
        xor_threads.emplace_back([&]() {
            try {
                xor_two_out_of_place(
                    reinterpret_cast<uint8_t*>(out_p0),
                    reinterpret_cast<const uint8_t*>(recv_addrs[4]),
                    reinterpret_cast<const uint8_t*>(recv_addrs[5]),
                    size);
            } catch (...) {
                xor_exceptions[2] = std::current_exception();
            }
        });
        xor_threads.emplace_back([&]() {
            try {
                xor_two_out_of_place(
                    reinterpret_cast<uint8_t*>(out_p1),
                    reinterpret_cast<const uint8_t*>(recv_addrs[6]),
                    reinterpret_cast<const uint8_t*>(recv_addrs[7]),
                    size);
            } catch (...) {
                xor_exceptions[3] = std::current_exception();
            }
        });

        for (auto& t : xor_threads) {
            t.join();
        }
        for (size_t i = 0; i < xor_exceptions.size(); ++i) {
            if (xor_exceptions[i]) {
                std::rethrow_exception(xor_exceptions[i]);
            }
        }
    }

    // Unified recovery interface for rank2 (parallel recv + parallel XOR)
    void load_recover(
        // Receive buffers (6 blocks from other ranks)
        uintptr_t rank0_data2_addr,
        uintptr_t rank0_parity2_addr,
        uintptr_t rank1_data1_addr,
        uintptr_t rank1_parity1_addr,
        uintptr_t rank3_data1_addr,
        uintptr_t rank3_data2_addr,
        // Recovered buffers (4 blocks to write results)
        uintptr_t recovered_data1_addr,
        uintptr_t recovered_data2_addr,
        uintptr_t recovered_parity1_addr,
        uintptr_t recovered_parity2_addr,
        size_t size
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: [Rank 2] load_recover called but not in load mode" << std::endl;
            return;
        }

        std::cout << "ECLATIN: [Rank 2] Starting recovery (size=" << size << ")" << std::endl;

        onefail_recv_chunk_(
            rank0_data2_addr, rank0_parity2_addr,
            rank1_data1_addr, rank1_parity1_addr,
            rank3_data1_addr, rank3_data2_addr,
            size);

        std::cout << "ECLATIN: [Rank 2] All 6 blocks received" << std::endl;

        onefail_xor_chunk_(
            rank0_data2_addr, rank0_parity2_addr,
            rank1_data1_addr, rank1_parity1_addr,
            rank3_data1_addr, rank3_data2_addr,
            recovered_data1_addr, recovered_data2_addr,
            recovered_parity1_addr, recovered_parity2_addr,
            size);

        std::cout << "ECLATIN: [Rank 2] Recovery completed successfully" << std::endl;
    }

    // ── Two-failures load recovery ────────────────────────────────────────────
    void load_recover_two_failures(
        // 8 recv buffer addrs (4 from rank 2, 4 from rank 3)
        uintptr_t r2_d2, uintptr_t r2_D2, uintptr_t r2_p2, uintptr_t r2_P2,
        uintptr_t r3_d3, uintptr_t r3_D3, uintptr_t r3_p3, uintptr_t r3_P3,
        // 2 recovered output addrs
        uintptr_t recovered_data1,   // data_block_1 of this failed rank
        uintptr_t recovered_data2,   // data_block_2 of this failed rank
        int rank_in_group,           // 0 or 1 (determines recovery formula)
        size_t size
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: [Two-fail] load_recover_two_failures called but not in load mode" << std::endl;
            return;
        }

        std::cout << "ECLATIN: [Two-fail] rank_in_group " << rank_in_group
                  << " starting recovery (size=" << size << ")" << std::endl;

        // Stage 1: Parallel receive 8 blocks (4 from peer0=rank2, 4 from peer1=rank3)
        // Peer0 (rank 2) blocks: r2_d2, r2_D2, r2_p2, r2_P2
        // Peer1 (rank 3) blocks: r3_d3, r3_D3, r3_p3, r3_P3
        std::vector<std::exception_ptr> recv_exceptions(8);
        std::vector<std::thread> recv_threads;

        // ── Recv from peer0 (rank 2): 4 blocks ──
        recv_threads.emplace_back([&]() {
            try {
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_load_channels_two_fail_[0]) {
                    rdma_load_channels_two_fail_[0]->receive_data(reinterpret_cast<uint8_t*>(r2_d2), size);
                } else
#endif
                if (!recv_with_size_bool(conn_.get_load_twofail_peer0_socket(),
                                        reinterpret_cast<void*>(r2_d2), size))
                    throw std::runtime_error("Failed to receive r2_d2");
            } catch (...) { recv_exceptions[0] = std::current_exception(); }
        });
        recv_threads.emplace_back([&]() {
            try {
                if (use_rdma_ && rdma_load_channels_two_fail_[1])
                    rdma_load_channels_two_fail_[1]->receive_data(reinterpret_cast<uint8_t*>(r2_D2), size);
                else if (!recv_with_size_bool(conn_.get_load_twofail_peer0_socket(),
                                             reinterpret_cast<void*>(r2_D2), size))
                    throw std::runtime_error("Failed to receive r2_D2");
            } catch (...) { recv_exceptions[1] = std::current_exception(); }
        });
        recv_threads.emplace_back([&]() {
            try {
                if (use_rdma_ && rdma_load_channels_two_fail_[2])
                    rdma_load_channels_two_fail_[2]->receive_data(reinterpret_cast<uint8_t*>(r2_p2), size);
                else if (!recv_with_size_bool(conn_.get_load_twofail_peer0_socket(),
                                             reinterpret_cast<void*>(r2_p2), size))
                    throw std::runtime_error("Failed to receive r2_p2");
            } catch (...) { recv_exceptions[2] = std::current_exception(); }
        });
        recv_threads.emplace_back([&]() {
            try {
                if (use_rdma_ && rdma_load_channels_two_fail_[3])
                    rdma_load_channels_two_fail_[3]->receive_data(reinterpret_cast<uint8_t*>(r2_P2), size);
                else if (!recv_with_size_bool(conn_.get_load_twofail_peer0_socket(),
                                             reinterpret_cast<void*>(r2_P2), size))
                    throw std::runtime_error("Failed to receive r2_P2");
            } catch (...) { recv_exceptions[3] = std::current_exception(); }
        });

        // ── Recv from peer1 (rank 3): 4 blocks ──
        recv_threads.emplace_back([&]() {
            try {
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_load_channels_two_fail_[4]) {
                    rdma_load_channels_two_fail_[4]->receive_data(reinterpret_cast<uint8_t*>(r3_d3), size);
                } else
#endif
                if (!recv_with_size_bool(conn_.get_load_twofail_peer1_socket(),
                                        reinterpret_cast<void*>(r3_d3), size))
                    throw std::runtime_error("Failed to receive r3_d3");
            } catch (...) { recv_exceptions[4] = std::current_exception(); }
        });
        recv_threads.emplace_back([&]() {
            try {
                if (use_rdma_ && rdma_load_channels_two_fail_[5])
                    rdma_load_channels_two_fail_[5]->receive_data(reinterpret_cast<uint8_t*>(r3_D3), size);
                else if (!recv_with_size_bool(conn_.get_load_twofail_peer1_socket(),
                                             reinterpret_cast<void*>(r3_D3), size))
                    throw std::runtime_error("Failed to receive r3_D3");
            } catch (...) { recv_exceptions[5] = std::current_exception(); }
        });
        recv_threads.emplace_back([&]() {
            try {
                if (use_rdma_ && rdma_load_channels_two_fail_[6])
                    rdma_load_channels_two_fail_[6]->receive_data(reinterpret_cast<uint8_t*>(r3_p3), size);
                else if (!recv_with_size_bool(conn_.get_load_twofail_peer1_socket(),
                                             reinterpret_cast<void*>(r3_p3), size))
                    throw std::runtime_error("Failed to receive r3_p3");
            } catch (...) { recv_exceptions[6] = std::current_exception(); }
        });
        recv_threads.emplace_back([&]() {
            try {
                if (use_rdma_ && rdma_load_channels_two_fail_[7])
                    rdma_load_channels_two_fail_[7]->receive_data(reinterpret_cast<uint8_t*>(r3_P3), size);
                else if (!recv_with_size_bool(conn_.get_load_twofail_peer1_socket(),
                                             reinterpret_cast<void*>(r3_P3), size))
                    throw std::runtime_error("Failed to receive r3_P3");
            } catch (...) { recv_exceptions[7] = std::current_exception(); }
        });

        // Join all recv threads
        for (auto& t : recv_threads) t.join();
        for (size_t i = 0; i < recv_exceptions.size(); ++i)
            if (recv_exceptions[i]) std::rethrow_exception(recv_exceptions[i]);

        std::cout << "ECLATIN: [Two-fail] All 8 blocks received, starting XOR pool recovery" << std::endl;

        // Lazy-init zero buffer for 2-input XOR via 16-thread pool
        if (twofail_zero_buffer_.size() < size) {
            twofail_zero_buffer_.resize(size, 0);
        }

        // Stage 2: XOR pool recovery using 16-thread pool with CPU affinity
        // d0 = p2 XOR D3 (2-input via xor_pool with zero buffer)
        // d1 = p3 XOR D2 (2-input via xor_pool with zero buffer)
        // Allocate temporary buffers for d0 and d1
        auto tmp_d0 = std::make_unique<uint8_t[]>(size);
        auto tmp_d1 = std::make_unique<uint8_t[]>(size);

        // d0 = p2 XOR D3
        std::memcpy(tmp_d0.get(), reinterpret_cast<void*>(r2_p2), size);
        xor_pool_run_parallel(
            reinterpret_cast<uintptr_t>(tmp_d0.get()),
            r3_D3,
            reinterpret_cast<uintptr_t>(twofail_zero_buffer_.data()),
            static_cast<int>(size));

        // d1 = p3 XOR D2
        std::memcpy(tmp_d1.get(), reinterpret_cast<void*>(r3_p3), size);
        xor_pool_run_parallel(
            reinterpret_cast<uintptr_t>(tmp_d1.get()),
            r2_D2,
            reinterpret_cast<uintptr_t>(twofail_zero_buffer_.data()),
            static_cast<int>(size));

        // Rank-specific data block recovery
        if (rank_in_group == 0) {
            // data_block_1 = d0, data_block_2 = D0 = P2 XOR d1
            std::memcpy(reinterpret_cast<void*>(recovered_data1), tmp_d0.get(), size);
            std::memcpy(reinterpret_cast<void*>(recovered_data2), reinterpret_cast<void*>(r2_P2), size);
            xor_pool_run_parallel(
                recovered_data2,
                reinterpret_cast<uintptr_t>(tmp_d1.get()),
                reinterpret_cast<uintptr_t>(twofail_zero_buffer_.data()),
                static_cast<int>(size));
        } else {  // rank_in_group == 1
            // data_block_1 = d1, data_block_2 = D1 = P3 XOR d0
            std::memcpy(reinterpret_cast<void*>(recovered_data1), tmp_d1.get(), size);
            std::memcpy(reinterpret_cast<void*>(recovered_data2), reinterpret_cast<void*>(r3_P3), size);
            xor_pool_run_parallel(
                recovered_data2,
                reinterpret_cast<uintptr_t>(tmp_d0.get()),
                reinterpret_cast<uintptr_t>(twofail_zero_buffer_.data()),
                static_cast<int>(size));
        }

        std::cout << "ECLATIN: [Two-fail] rank_in_group " << rank_in_group
                  << " recovery completed successfully" << std::endl;
    }

    void onefail_send_chunk_(
        const std::string& block1_name,
        uintptr_t block1_addr,
        const std::string& block2_name,
        uintptr_t block2_addr,
        size_t size
    ) {
        auto get_socket = [this](const std::string& block_name) -> boost::asio::ip::tcp::socket* {
            if (use_no_shared_block_) {
                int ch = ns_load_channel_from_name(block_name);
                if (ch >= 0) {
                    return &conn_.get_ns_load_send_socket(ch);
                }
                return nullptr;
            }
            if (block_name == "rank0_data2") {
                return &conn_.get_load_send_rank0_data2_socket();
            } else if (block_name == "rank0_parity2") {
                return &conn_.get_load_send_rank0_parity2_socket();
            } else if (block_name == "rank1_data1") {
                return &conn_.get_load_send_rank1_data1_socket();
            } else if (block_name == "rank1_parity1") {
                return &conn_.get_load_send_rank1_parity1_socket();
            } else if (block_name == "rank3_data1") {
                return &conn_.get_load_send_rank3_data1_socket();
            } else if (block_name == "rank3_data2") {
                return &conn_.get_load_send_rank3_data2_socket();
            }
            return nullptr;
        };
#if RDMA_AVAILABLE
        auto get_load_rdma_ch = [this](const std::string& block_name) -> int {
            if (use_no_shared_block_) {
                return ns_load_channel_from_name(block_name);
            }
            if (block_name == "rank0_data2") return 0;
            if (block_name == "rank0_parity2") return 1;
            if (block_name == "rank1_data1") return 2;
            if (block_name == "rank1_parity1") return 3;
            if (block_name == "rank3_data1") return 4;
            if (block_name == "rank3_data2") return 5;
            return -1;
        };
        auto get_rdma_channel = [this](int ch) -> RdmaConnectionChannel* {
            if (use_no_shared_block_) {
                return (ch >= 0 && ch < NS_LOAD_NUM_CHANNELS)
                    ? rdma_load_channels_noshared_[ch].get() : nullptr;
            }
            return (ch >= 0 && ch < RDMA_NUM_LOAD_CHANNELS)
                ? rdma_load_channels_[ch].get() : nullptr;
        };
#endif

        std::exception_ptr thread1_exception = nullptr;
        std::exception_ptr thread2_exception = nullptr;

        std::thread thread1([&]() {
            try {
#if RDMA_AVAILABLE
                int ch1 = get_load_rdma_ch(block1_name);
                if (use_rdma_ && ch1 >= 0 && get_rdma_channel(ch1)) {
                    get_rdma_channel(ch1)->send_data(
                        reinterpret_cast<const uint8_t*>(block1_addr), size);
                } else
#endif
                {
                    boost::asio::ip::tcp::socket* sock = get_socket(block1_name);
                    if (sock == nullptr || !sock->is_open()) {
                        throw std::runtime_error(
                            "ECLATIN: onefail_send_chunk socket not available for " + block1_name);
                    }
                    if (!send_with_size(*sock, block1_addr, size)) {
                        throw std::runtime_error(
                            "ECLATIN: onefail_send_chunk send failed for " + block1_name);
                    }
                }
            } catch (...) {
                thread1_exception = std::current_exception();
            }
        });

        std::thread thread2([&]() {
            try {
#if RDMA_AVAILABLE
                int ch2 = get_load_rdma_ch(block2_name);
                if (use_rdma_ && ch2 >= 0 && get_rdma_channel(ch2)) {
                    get_rdma_channel(ch2)->send_data(
                        reinterpret_cast<const uint8_t*>(block2_addr), size);
                } else
#endif
                {
                    boost::asio::ip::tcp::socket* sock = get_socket(block2_name);
                    if (sock == nullptr || !sock->is_open()) {
                        throw std::runtime_error(
                            "ECLATIN: onefail_send_chunk socket not available for " + block2_name);
                    }
                    if (!send_with_size(*sock, block2_addr, size)) {
                        throw std::runtime_error(
                            "ECLATIN: onefail_send_chunk send failed for " + block2_name);
                    }
                }
            } catch (...) {
                thread2_exception = std::current_exception();
            }
        });

        thread1.join();
        thread2.join();
        if (thread1_exception) std::rethrow_exception(thread1_exception);
        if (thread2_exception) std::rethrow_exception(thread2_exception);
    }

    // Unified send interface for other ranks (rank1, rank2, rank3) - parallel send two blocks
    void load_send_blocks(
        const std::string& block1_name,
        uintptr_t block1_addr,
        const std::string& block2_name,
        uintptr_t block2_addr,
        size_t size
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: load_send_blocks called but not in load mode" << std::endl;
            return;
        }

        std::cout << "ECLATIN: Starting parallel send of " << block1_name
                  << " and " << block2_name << " to rank2 (size=" << size << ")" << std::endl;
        onefail_send_chunk_(block1_name, block1_addr, block2_name, block2_addr, size);
        std::cout << "ECLATIN: Both blocks sent successfully" << std::endl;
    }

    // ── Two-failures load: send 4 blocks to a failed rank ─────────────────────
    void load_send_all_blocks_two_fail(
        const std::string& target_rank_in_group,  // "0" or "1" — which failed rank to send to
        uintptr_t data1_addr, uintptr_t data2_addr,
        uintptr_t parity1_addr, uintptr_t parity2_addr,
        size_t size
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: load_send_all_blocks_two_fail called but not in load mode" << std::endl;
            return;
        }

        // Select peer socket: peer0 → rank_in_group 0, peer1 → rank_in_group 1
        boost::asio::ip::tcp::socket* peer_sock =
            (target_rank_in_group == "0")
                ? &conn_.get_load_twofail_peer0_socket()
                : &conn_.get_load_twofail_peer1_socket();

        int rdma_ch_base = (target_rank_in_group == "0") ? 0 : 4;

        std::cout << "ECLATIN: [Two-fail] Sending 4 blocks to failed rank "
                  << target_rank_in_group << " (size=" << size << ")" << std::endl;

        // Parallel send 4 blocks
        std::array<std::exception_ptr, 4> send_exceptions;
        std::array<std::thread, 4> send_threads;
        std::array<uintptr_t, 4> addrs = {data1_addr, data2_addr, parity1_addr, parity2_addr};
        std::array<std::string, 4> names = {"data1", "data2", "parity1", "parity2"};

        for (int i = 0; i < 4; ++i) {
            send_threads[static_cast<size_t>(i)] = std::thread([this, &addrs, &names, &send_exceptions,
                                                                  peer_sock, rdma_ch_base, size, i]() {
                try {
#if RDMA_AVAILABLE
                    int ch = rdma_ch_base + i;
                    if (use_rdma_ && rdma_load_channels_two_fail_[ch]) {
                        rdma_load_channels_two_fail_[ch]->send_data(
                            reinterpret_cast<const uint8_t*>(addrs[static_cast<size_t>(i)]), size);
                    } else
#endif
                    {
                        if (peer_sock == nullptr || !peer_sock->is_open()) {
                            throw std::runtime_error("ECLATIN: load_send_all_blocks_two_fail socket not available");
                        }
                        if (!send_with_size(*peer_sock, addrs[static_cast<size_t>(i)], size)) {
                            throw std::runtime_error("ECLATIN: send failed for " + names[static_cast<size_t>(i)]);
                        }
                    }
                    // std::cout << "ECLATIN: [Two-fail] Sent " << names[i] << " successfully" << std::endl;
                } catch (...) {
                    send_exceptions[static_cast<size_t>(i)] = std::current_exception();
                }
            });
        }

        for (int i = 0; i < 4; ++i) send_threads[static_cast<size_t>(i)].join();
        for (int i = 0; i < 4; ++i)
            if (send_exceptions[static_cast<size_t>(i)])
                std::rethrow_exception(send_exceptions[static_cast<size_t>(i)]);

        std::cout << "ECLATIN: [Two-fail] All 4 blocks sent to failed rank "
                  << target_rank_in_group << " successfully" << std::endl;
    }

    // ── Two-failures v2: Phase 0a - bind+listen only (no blocking ops) ─────────

    void init_twofail_bind_phase(
        int rank_in_group,
        const std::string& surv_exch_ip, uint16_t surv_exch_port,
        const std::string& n1_ip, uint16_t n1_n3_port,
        const std::string& n2_ip, uint16_t n2_n3_port,
        const std::string& n1_ip2, uint16_t n1_n4_port,
        const std::string& n2_ip2, uint16_t n2_n4_port
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: init_twofail_bind_phase called but not in load mode" << std::endl;
            return;
        }

        is_twofail_v2_ = true;

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " bind phase..." << std::endl;

        // All bind+listen calls are synchronous and return immediately after listen().
        // This ensures the kernel TCP backlog is ready before any connect() from Phase B.
        if (rank_in_group == 2) {
            // Node3: bind survivor exchange acceptor
            conn_.bind_listen_twf_surv_exch(surv_exch_ip, surv_exch_port);
        } else if (rank_in_group == 0) {
            // Node1: bind acceptors for Node3 and Node4
            conn_.bind_listen_twf_n1_from_n3(n1_ip, n1_n3_port);
            conn_.bind_listen_twf_n1_from_n4(n1_ip2, n1_n4_port);
        } else if (rank_in_group == 1) {
            // Node2: bind acceptors for Node3 and Node4
            conn_.bind_listen_twf_n2_from_n3(n2_ip, n2_n3_port);
            conn_.bind_listen_twf_n2_from_n4(n2_ip2, n2_n4_port);
        }
        // rig3: nothing to bind (connects to all others)

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " bind phase complete" << std::endl;
    }

    // ── Two-failures v2: Phase 0b - connect + accept ──────────────────────────

    void init_twofail_connect_phase(
        int rank_in_group,
        const std::string& surv_exch_ip, uint16_t surv_exch_port,
        const std::string& n1_ip, uint16_t n1_n3_port,
        const std::string& n2_ip, uint16_t n2_n3_port,
        const std::string& n1_ip2, uint16_t n1_n4_port,
        const std::string& n2_ip2, uint16_t n2_n4_port
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: init_twofail_connect_phase called but not in load mode" << std::endl;
            return;
        }

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " connect phase..." << std::endl;

        if (rank_in_group == 2) {
            // Node3: start accept thread + connect to Node1 and Node2
            std::thread surv_accept_thread([this]() {
                conn_.accept_twf_surv_exch();
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            surv_accept_thread.detach();

            conn_.init_twf_send_n1_from_n3(n1_ip, n1_n3_port);
            conn_.init_twf_send_n2_from_n3(n2_ip, n2_n3_port);

        } else if (rank_in_group == 3) {
            // Node4: connect to survivor exchange + Node1 + Node2
            conn_.init_twf_surv_exch_send(surv_exch_ip, surv_exch_port);
            conn_.init_twf_send_n1_from_n4(n1_ip2, n1_n4_port);
            conn_.init_twf_send_n2_from_n4(n2_ip2, n2_n4_port);

        } else if (rank_in_group == 0) {
            // Node1: start accept threads for Node3 and Node4
            std::thread accept_thread([this]() {
                std::thread t3([this]() { conn_.accept_twf_n1_from_n3(); });
                std::thread t4([this]() { conn_.accept_twf_n1_from_n4(); });
                t3.join();
                t4.join();
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            accept_thread.detach();

        } else if (rank_in_group == 1) {
            // Node2: start accept threads for Node3 and Node4
            std::thread accept_thread([this]() {
                std::thread t3([this]() { conn_.accept_twf_n2_from_n3(); });
                std::thread t4([this]() { conn_.accept_twf_n2_from_n4(); });
                t3.join();
                t4.join();
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            accept_thread.detach();

        } else {
            throw std::runtime_error(
                "ECLATIN: init_twofail_connect_phase: unexpected rank_in_group=" +
                std::to_string(rank_in_group));
        }

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " connect phase complete" << std::endl;
    }

    // ── Two-failures v2: Step 1 - survivor data exchange ─────────────────────

    void survivor_exchange_data(
        int rank_in_group,
        uintptr_t send_d1, uintptr_t send_d2,
        uintptr_t recv_d1, uintptr_t recv_d2,
        size_t size
    ) {
        if (!is_twofail_v2_) {
            std::cerr << "ECLATIN: survivor_exchange_data called but not in two-fail v2 mode"
                      << std::endl;
            return;
        }

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " exchanging data blocks (size=" << size << ")" << std::endl;

        // RDMA path: parallel d1/d2 transfer on paired channels (separate QPs, shared control socket)
        // Channel mapping: rig3 sends on [2,3] and recvs on [0,1]; rig2 recvs on [2,3] and sends on [0,1]
#if RDMA_AVAILABLE
        if (use_rdma_ && rdma_twf_v2_survexch_channels_[0] && rdma_twf_v2_survexch_channels_[1] &&
            rdma_twf_v2_survexch_channels_[2] && rdma_twf_v2_survexch_channels_[3]) {
            const int send_lo = (rank_in_group == 3) ? 2 : 0;
            const int recv_lo = (rank_in_group == 3) ? 0 : 2;

            auto run_parallel_pair = [&](int ch_a, uintptr_t addr_a, int ch_b, uintptr_t addr_b,
                                         bool is_send) {
                std::exception_ptr ex_a;
                std::exception_ptr ex_b;
                std::thread t_a([&]() {
                    try {
                        if (is_send) {
                            rdma_twf_v2_survexch_channels_[ch_a]->send_data(
                                reinterpret_cast<const uint8_t*>(addr_a), size);
                        } else {
                            rdma_twf_v2_survexch_channels_[ch_a]->receive_data(
                                reinterpret_cast<uint8_t*>(addr_a), size);
                        }
                    } catch (...) {
                        ex_a = std::current_exception();
                    }
                });
                std::thread t_b([&]() {
                    try {
                        if (is_send) {
                            rdma_twf_v2_survexch_channels_[ch_b]->send_data(
                                reinterpret_cast<const uint8_t*>(addr_b), size);
                        } else {
                            rdma_twf_v2_survexch_channels_[ch_b]->receive_data(
                                reinterpret_cast<uint8_t*>(addr_b), size);
                        }
                    } catch (...) {
                        ex_b = std::current_exception();
                    }
                });
                t_a.join();
                t_b.join();
                if (ex_a) std::rethrow_exception(ex_a);
                if (ex_b) std::rethrow_exception(ex_b);
            };

            if (rank_in_group == 3) {
                // Phase 1: send own d1/d2 in parallel (paired with rig2 recv on ch 2,3)
                run_parallel_pair(send_lo, send_d1, send_lo + 1, send_d2, true);
                // Phase 2: recv peer d1/d2 in parallel (paired with rig2 send on ch 0,1)
                run_parallel_pair(recv_lo, recv_d1, recv_lo + 1, recv_d2, false);
            } else {
                // Phase 1: recv peer d1/d2 in parallel
                run_parallel_pair(recv_lo, recv_d1, recv_lo + 1, recv_d2, false);
                // Phase 2: send own d1/d2 in parallel
                run_parallel_pair(send_lo, send_d1, send_lo + 1, send_d2, true);
            }

            std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                      << " data exchange complete (RDMA, parallel d1/d2)" << std::endl;
            return;
        }
#endif

        // TCP fallback: serialized to avoid interleaving on single socket
        // rig3 (connector) sends first; rig2 (acceptor) recvs first
        auto& sock = conn_.get_twf_surv_exch_socket();
        if (rank_in_group == 3) {
            if (!send_with_size(sock, send_d1, size))
                throw std::runtime_error("TCP: Failed to send d1");
            if (!send_with_size(sock, send_d2, size))
                throw std::runtime_error("TCP: Failed to send d2");
            if (!recv_with_size_bool(sock, reinterpret_cast<void*>(recv_d1), size))
                throw std::runtime_error("TCP: Failed to recv peer d1");
            if (!recv_with_size_bool(sock, reinterpret_cast<void*>(recv_d2), size))
                throw std::runtime_error("TCP: Failed to recv peer d2");
        } else {
            if (!recv_with_size_bool(sock, reinterpret_cast<void*>(recv_d1), size))
                throw std::runtime_error("TCP: Failed to recv peer d1");
            if (!recv_with_size_bool(sock, reinterpret_cast<void*>(recv_d2), size))
                throw std::runtime_error("TCP: Failed to recv peer d2");
            if (!send_with_size(sock, send_d1, size))
                throw std::runtime_error("TCP: Failed to send d1");
            if (!send_with_size(sock, send_d2, size))
                throw std::runtime_error("TCP: Failed to send d2");
        }

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " data exchange complete (TCP)" << std::endl;
    }

    // ── Two-failures v2: Step 2 - XOR decode on survivor ─────────────────────

    void survivor_xor_decode(
        int rank_in_group,
        uintptr_t out1, uintptr_t out2, uintptr_t out3, uintptr_t out4,
        uintptr_t own_d1, uintptr_t own_d2, uintptr_t own_p1, uintptr_t own_p2,
        uintptr_t peer_d1, uintptr_t peer_d2,
        size_t size
    ) {
        if (!is_twofail_v2_) {
            std::cerr << "ECLATIN: survivor_xor_decode called but not in two-fail v2 mode"
                      << std::endl;
            return;
        }

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " starting XOR decode (size=" << size << ")" << std::endl;

        if (rank_in_group != 2 && rank_in_group != 3) {
            throw std::runtime_error(
                "ECLATIN: survivor_xor_decode: unexpected rank_in_group=" +
                std::to_string(rank_in_group));
        }

        const int xor_len = static_cast<int>(size);

        // Pass 1 (parallel): out1 = peer_d2 XOR own_p1, out3 = own_p2 XOR peer_d1
        xor_pool_run_parallel_two_stripe(
            EclatinXorStripeOp{out1, peer_d2, own_p1},
            EclatinXorStripeOp{out3, own_p2, peer_d1},
            xor_len);

        // Pass 2 (parallel): out2 = out1 XOR own_d2, out4 = out3 XOR own_d1
        xor_pool_run_parallel_two_stripe(
            EclatinXorStripeOp{out2, out1, own_d2},
            EclatinXorStripeOp{out4, out3, own_d1},
            xor_len);

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " XOR decode complete" << std::endl;
    }

    // ── Two-failures v2: Step 3 - send 2 blocks to one failed rank ───────────

    void send_two_blocks(
        int rank_in_group,
        const std::string& target_rig,
        uintptr_t addr1, uintptr_t addr2,
        size_t size
    ) {
        if (!is_twofail_v2_) {
            std::cerr << "ECLATIN: send_two_blocks called but not in two-fail v2 mode"
                      << std::endl;
            return;
        }

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " sending 2 blocks to rig" << target_rig
                  << " (size=" << size << ")" << std::endl;

        // Select channel array and TCP socket
        auto* channels = static_cast<decltype(&rdma_twf_v2_n1n3_channels_)>(nullptr);
        boost::asio::ip::tcp::socket* sock = nullptr;
        if (rank_in_group == 2) {
            if (target_rig == "0") { channels = &rdma_twf_v2_n1n3_channels_; sock = &conn_.get_twf_n1_n3_socket(); }
            else                  { channels = &rdma_twf_v2_n2n3_channels_; sock = &conn_.get_twf_n2_n3_socket(); }
        } else if (rank_in_group == 3) {
            if (target_rig == "0") { channels = &rdma_twf_v2_n1n4_channels_; sock = &conn_.get_twf_n1_n4_socket(); }
            else                  { channels = &rdma_twf_v2_n2n4_channels_; sock = &conn_.get_twf_n2_n4_socket(); }
        } else {
            throw std::runtime_error("ECLATIN: send_two_blocks: unexpected rank_in_group=" +
                                     std::to_string(rank_in_group));
        }

        // RDMA path: parallel send on 2 channels (shared control mutex serializes handshake only)
#if RDMA_AVAILABLE
        if (use_rdma_ && (*channels)[0] && (*channels)[1]) {
            std::exception_ptr ex_a;
            std::exception_ptr ex_b;
            std::thread t_a([&]() {
                try {
                    (*channels)[0]->send_data(reinterpret_cast<const uint8_t*>(addr1), size);
                } catch (...) {
                    ex_a = std::current_exception();
                }
            });
            std::thread t_b([&]() {
                try {
                    (*channels)[1]->send_data(reinterpret_cast<const uint8_t*>(addr2), size);
                } catch (...) {
                    ex_b = std::current_exception();
                }
            });
            t_a.join();
            t_b.join();
            if (ex_a) std::rethrow_exception(ex_a);
            if (ex_b) std::rethrow_exception(ex_b);

            std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                      << " sent 2 blocks to rig" << target_rig << " (RDMA, parallel)" << std::endl;
            return;
        }
#endif

        // TCP fallback: serialized over single socket
        if (!send_with_size(*sock, addr1, size))
            throw std::runtime_error("TCP: Failed to send block 1");
        if (!send_with_size(*sock, addr2, size))
            throw std::runtime_error("TCP: Failed to send block 2");

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " sent 2 blocks to rig" << target_rig << " (TCP)" << std::endl;
    }

    // Step 3: survivor sends 2 blocks to rig0 and rig1 in parallel (distinct peer sockets)
    void send_to_both_failed_ranks(
        int rank_in_group,
        uintptr_t rig0_addr1, uintptr_t rig0_addr2,
        uintptr_t rig1_addr1, uintptr_t rig1_addr2,
        size_t size
    ) {
        if (!is_twofail_v2_) {
            std::cerr << "ECLATIN: send_to_both_failed_ranks called but not in two-fail v2 mode"
                      << std::endl;
            return;
        }
        if (rank_in_group != 2 && rank_in_group != 3) {
            throw std::runtime_error(
                "ECLATIN: send_to_both_failed_ranks: unexpected rank_in_group=" +
                std::to_string(rank_in_group));
        }

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " sending to rig0 and rig1 in parallel (size=" << size << ")" << std::endl;

        std::exception_ptr ex0;
        std::exception_ptr ex1;
        std::thread t0([&]() {
            try {
                send_two_blocks(rank_in_group, "0", rig0_addr1, rig0_addr2, size);
            } catch (...) {
                ex0 = std::current_exception();
            }
        });
        std::thread t1([&]() {
            try {
                send_two_blocks(rank_in_group, "1", rig1_addr1, rig1_addr2, size);
            } catch (...) {
                ex1 = std::current_exception();
            }
        });
        t0.join();
        t1.join();
        if (ex0) std::rethrow_exception(ex0);
        if (ex1) std::rethrow_exception(ex1);

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " sent to both failed ranks" << std::endl;
    }

    // ── Two-failures v2: Step 3 - failed node recv 4 blocks from 2 sockets ───

    void recv_four_blocks(
        int rank_in_group,
        uintptr_t addr1, uintptr_t addr2, uintptr_t addr3, uintptr_t addr4,
        size_t size
    ) {
        if (!is_twofail_v2_) {
            std::cerr << "ECLATIN: recv_four_blocks called but not in two-fail v2 mode"
                      << std::endl;
            return;
        }

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " receiving 4 blocks (size=" << size << ")" << std::endl;

        // Select channel arrays and TCP sockets
        // rig0: ch_n3 = n1n3 (addr2,addr4), ch_n4 = n1n4 (addr1,addr3)
        // rig1: ch_n3 = n2n3 (addr1,addr3), ch_n4 = n2n4 (addr2,addr4)
        auto* ch_n3 = static_cast<decltype(&rdma_twf_v2_n1n3_channels_)>(nullptr);
        auto* ch_n4 = static_cast<decltype(&rdma_twf_v2_n1n4_channels_)>(nullptr);
        boost::asio::ip::tcp::socket* sock_n3;
        boost::asio::ip::tcp::socket* sock_n4;
        uintptr_t a_n3[2], a_n4[2];

        if (rank_in_group == 0) {
            ch_n3 = &rdma_twf_v2_n1n3_channels_; ch_n4 = &rdma_twf_v2_n1n4_channels_;
            sock_n3 = &conn_.get_twf_n1_n3_socket(); sock_n4 = &conn_.get_twf_n1_n4_socket();
            a_n3[0] = addr2; a_n3[1] = addr4;
            a_n4[0] = addr1; a_n4[1] = addr3;
        } else if (rank_in_group == 1) {
            ch_n3 = &rdma_twf_v2_n2n3_channels_; ch_n4 = &rdma_twf_v2_n2n4_channels_;
            sock_n3 = &conn_.get_twf_n2_n3_socket(); sock_n4 = &conn_.get_twf_n2_n4_socket();
            a_n3[0] = addr1; a_n3[1] = addr3;
            a_n4[0] = addr2; a_n4[1] = addr4;
        } else {
            throw std::runtime_error("ECLATIN: recv_four_blocks: unexpected rank_in_group=" +
                                     std::to_string(rank_in_group));
        }

        // RDMA path: parallel recv on n3/n4 peers and parallel within each peer's 2 channels
#if RDMA_AVAILABLE
        if (use_rdma_ && (*ch_n3)[0] && (*ch_n3)[1] && (*ch_n4)[0] && (*ch_n4)[1]) {
            std::exception_ptr recv_ex[4];
            std::thread recv_threads[4];
            recv_threads[0] = std::thread([&]() {
                try {
                    (*ch_n3)[0]->receive_data(reinterpret_cast<uint8_t*>(a_n3[0]), size);
                } catch (...) {
                    recv_ex[0] = std::current_exception();
                }
            });
            recv_threads[1] = std::thread([&]() {
                try {
                    (*ch_n3)[1]->receive_data(reinterpret_cast<uint8_t*>(a_n3[1]), size);
                } catch (...) {
                    recv_ex[1] = std::current_exception();
                }
            });
            recv_threads[2] = std::thread([&]() {
                try {
                    (*ch_n4)[0]->receive_data(reinterpret_cast<uint8_t*>(a_n4[0]), size);
                } catch (...) {
                    recv_ex[2] = std::current_exception();
                }
            });
            recv_threads[3] = std::thread([&]() {
                try {
                    (*ch_n4)[1]->receive_data(reinterpret_cast<uint8_t*>(a_n4[1]), size);
                } catch (...) {
                    recv_ex[3] = std::current_exception();
                }
            });
            for (auto& t : recv_threads) {
                t.join();
            }
            for (const auto& ex : recv_ex) {
                if (ex) std::rethrow_exception(ex);
            }

            std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                      << " received all 4 blocks (RDMA, parallel n3/n4 + parallel per peer)" << std::endl;
            return;
        }
#endif

        // TCP fallback: parallel recv on n3 and n4 sockets
        {
            std::exception_ptr n3_ex;
            std::exception_ptr n4_ex;
            std::thread t_n3([&]() {
                try {
                    if (!recv_with_size_bool(*sock_n3, reinterpret_cast<void*>(a_n3[0]), size))
                        throw std::runtime_error("TCP: Failed to recv n3 block 1");
                    if (!recv_with_size_bool(*sock_n3, reinterpret_cast<void*>(a_n3[1]), size))
                        throw std::runtime_error("TCP: Failed to recv n3 block 2");
                } catch (...) {
                    n3_ex = std::current_exception();
                }
            });
            std::thread t_n4([&]() {
                try {
                    if (!recv_with_size_bool(*sock_n4, reinterpret_cast<void*>(a_n4[0]), size))
                        throw std::runtime_error("TCP: Failed to recv n4 block 1");
                    if (!recv_with_size_bool(*sock_n4, reinterpret_cast<void*>(a_n4[1]), size))
                        throw std::runtime_error("TCP: Failed to recv n4 block 2");
                } catch (...) {
                    n4_ex = std::current_exception();
                }
            });
            t_n3.join();
            t_n4.join();
            if (n3_ex) std::rethrow_exception(n3_ex);
            if (n4_ex) std::rethrow_exception(n4_ex);
        }

        std::cout << "ECLATIN: [Two-fail v2] rig" << rank_in_group
                  << " received all 4 blocks (TCP)" << std::endl;
    }

    // ── One-fail (rig2) chunked load pipeline ────────────────────────────────

    void onefail_set_pipeline_error(std::exception_ptr ex) {
        if (!ex) return;
        std::lock_guard<std::mutex> lock(onefail_net_mutex_);
        if (!onefail_pipeline_error_) {
            onefail_pipeline_error_ = ex;
        }
    }

    void onefail_rethrow_if_error() {
        if (onefail_pipeline_error_) {
            std::rethrow_exception(onefail_pipeline_error_);
        }
    }

    void onefail_release_pool_buffers(const OnefailPipelineTask& task) {
        if (task.num_release <= 0) return;
        std::lock_guard<std::mutex> lock(onefail_release_mutex_);
        for (int i = 0; i < task.num_release; ++i) {
            if (task.release_addrs[i] != 0) {
                onefail_buffers_to_release_.push(task.release_addrs[i]);
            }
        }
    }

    void reset_onefail_pipeline() {
        onefail_pipeline_error_ = nullptr;
        onefail_net_done_ = false;
        onefail_xor_done_ = false;
        {
            std::lock_guard<std::mutex> lk(onefail_net_mutex_);
            while (!onefail_net_q_.empty()) onefail_net_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lk(onefail_xor_mutex_);
            while (!onefail_xor_q_.empty()) onefail_xor_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lk(onefail_release_mutex_);
            while (!onefail_buffers_to_release_.empty()) onefail_buffers_to_release_.pop();
        }
        std::cout << "ECLATIN: [One-fail] pipeline reset (rig" << rank_in_group_ << ")" << std::endl;
    }

    void submit_onefail_send_chunk(
        int /*rank_in_group*/,
        const std::string& block1_name,
        uintptr_t block1_addr,
        const std::string& block2_name,
        uintptr_t block2_addr,
        size_t size,
        int chunk_index
    ) {
        if (!is_load_mode_) {
            throw std::runtime_error("ECLATIN: submit_onefail_send_chunk called but not in load mode");
        }
        OnefailPipelineTask task;
        task.kind = OnefailTaskKind::Send;
        task.size = size;
        task.chunk_index = chunk_index;
        task.send_b1_name = block1_name;
        task.send_b2_name = block2_name;
        task.send_b1_addr = block1_addr;
        task.send_b2_addr = block2_addr;
        task.num_release = 0;
        {
            std::lock_guard<std::mutex> lk(onefail_net_mutex_);
            onefail_net_q_.push(task);
        }
        onefail_net_cv_.notify_one();
    }

    void submit_onefail_recv_chunk(
        uintptr_t r0d2, uintptr_t r0p2,
        uintptr_t r1d1, uintptr_t r1p1,
        uintptr_t r3d1, uintptr_t r3d2,
        uintptr_t out_d1, uintptr_t out_d2,
        uintptr_t out_p1, uintptr_t out_p2,
        size_t size, int chunk_index,
        uintptr_t rel0, uintptr_t rel1, uintptr_t rel2,
        uintptr_t rel3, uintptr_t rel4, uintptr_t rel5
    ) {
        if (!is_load_mode_) {
            throw std::runtime_error("ECLATIN: submit_onefail_recv_chunk called but not in load mode");
        }
        OnefailPipelineTask task;
        task.kind = OnefailTaskKind::Recv;
        task.size = size;
        task.chunk_index = chunk_index;
        task.r0d2 = r0d2;
        task.r0p2 = r0p2;
        task.r1d1 = r1d1;
        task.r1p1 = r1p1;
        task.r3d1 = r3d1;
        task.r3d2 = r3d2;
        task.out_d1 = out_d1;
        task.out_d2 = out_d2;
        task.out_p1 = out_p1;
        task.out_p2 = out_p2;
        task.release_addrs[0] = rel0;
        task.release_addrs[1] = rel1;
        task.release_addrs[2] = rel2;
        task.release_addrs[3] = rel3;
        task.release_addrs[4] = rel4;
        task.release_addrs[5] = rel5;
        task.num_release = 6;
        {
            std::lock_guard<std::mutex> lk(onefail_net_mutex_);
            onefail_net_q_.push(task);
        }
        onefail_net_cv_.notify_one();
    }

    void submit_onefail_noshared_recv_chunk(
        uintptr_t recv_n1_d1, uintptr_t recv_n3_p1,
        uintptr_t recv_n0_d0, uintptr_t recv_n3_p0,
        uintptr_t recv_n1_d0, uintptr_t recv_n3_d1,
        uintptr_t recv_n0_d1, uintptr_t recv_n3_d0,
        uintptr_t out_d0, uintptr_t out_d1,
        uintptr_t out_p0, uintptr_t out_p1,
        size_t size, int chunk_index,
        uintptr_t rel0, uintptr_t rel1, uintptr_t rel2, uintptr_t rel3,
        uintptr_t rel4, uintptr_t rel5, uintptr_t rel6, uintptr_t rel7
    ) {
        if (!is_load_mode_) {
            throw std::runtime_error(
                "ECLATIN: submit_onefail_noshared_recv_chunk called but not in load mode");
        }
        OnefailPipelineTask task;
        task.kind = OnefailTaskKind::Recv;
        task.noshared = true;
        task.size = size;
        task.chunk_index = chunk_index;
        task.ns_recv[0] = recv_n1_d1;
        task.ns_recv[1] = recv_n3_p1;
        task.ns_recv[2] = recv_n0_d0;
        task.ns_recv[3] = recv_n3_p0;
        task.ns_recv[4] = recv_n1_d0;
        task.ns_recv[5] = recv_n3_d1;
        task.ns_recv[6] = recv_n0_d1;
        task.ns_recv[7] = recv_n3_d0;
        task.out_d1 = out_d0;
        task.out_d2 = out_d1;
        task.out_p1 = out_p0;
        task.out_p2 = out_p1;
        task.release_addrs[0] = rel0;
        task.release_addrs[1] = rel1;
        task.release_addrs[2] = rel2;
        task.release_addrs[3] = rel3;
        task.release_addrs[4] = rel4;
        task.release_addrs[5] = rel5;
        task.release_addrs[6] = rel6;
        task.release_addrs[7] = rel7;
        task.num_release = NS_LOAD_NUM_CHANNELS;
        {
            std::lock_guard<std::mutex> lk(onefail_net_mutex_);
            onefail_net_q_.push(task);
        }
        onefail_net_cv_.notify_one();
    }

    void submit_onefail_pipeline_sentinels() {
        OnefailPipelineTask sentinel = OnefailPipelineTask::make_sentinel();
        {
            std::lock_guard<std::mutex> lk(onefail_net_mutex_);
            onefail_net_q_.push(sentinel);
        }
        onefail_net_cv_.notify_one();
    }

    std::vector<uintptr_t> get_onefail_buffers_to_release() {
        std::vector<uintptr_t> out;
        std::lock_guard<std::mutex> lk(onefail_release_mutex_);
        while (!onefail_buffers_to_release_.empty()) {
            out.push_back(onefail_buffers_to_release_.front());
            onefail_buffers_to_release_.pop();
        }
        return out;
    }

    void wait_for_onefail_pipeline_completion() {
        int spin = 0;
        while (!onefail_net_done_ || !onefail_xor_done_) {
            if (spin++ % 100 == 0) {
                std::cout << "ECLATIN: [One-fail] waiting for pipeline workers: net="
                          << onefail_net_done_.load() << " xor=" << onefail_xor_done_.load()
                          << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        onefail_rethrow_if_error();
        std::cout << "ECLATIN: [One-fail] pipeline completed (rig" << rank_in_group_ << ")" << std::endl;
    }

    void onefail_net_worker() {
        std::cout << "ECLATIN: [One-fail] net worker started (rig" << rank_in_group_ << ")" << std::endl;
        while (!stop_.load()) {
            OnefailPipelineTask task;
            {
                std::unique_lock<std::mutex> lk(onefail_net_mutex_);
                onefail_net_cv_.wait(lk, [this] {
                    return stop_.load() || !onefail_net_q_.empty();
                });
                if (stop_.load() && onefail_net_q_.empty()) break;
                if (onefail_net_q_.empty()) continue;
                task = onefail_net_q_.front();
                onefail_net_q_.pop();
            }
            if (task.is_sentinel()) {
                {
                    std::lock_guard<std::mutex> lk(onefail_xor_mutex_);
                    onefail_xor_q_.push(task);
                }
                onefail_xor_cv_.notify_one();
                onefail_net_done_ = true;
                continue;
            }
            try {
                if (task.kind == OnefailTaskKind::Send) {
                    onefail_send_chunk_(
                        task.send_b1_name, task.send_b1_addr,
                        task.send_b2_name, task.send_b2_addr,
                        task.size);
                } else if (rank_in_group_ == 2) {
                    if (task.noshared) {
                        onefail_noshared_recv_chunk_(task.ns_recv, task.size);
                    } else {
                        onefail_recv_chunk_(
                            task.r0d2, task.r0p2, task.r1d1, task.r1p1,
                            task.r3d1, task.r3d2, task.size);
                    }
                    {
                        std::lock_guard<std::mutex> lk(onefail_xor_mutex_);
                        onefail_xor_q_.push(task);
                    }
                    onefail_xor_cv_.notify_one();
                }
            } catch (...) {
                onefail_set_pipeline_error(std::current_exception());
                onefail_net_done_ = true;
                onefail_xor_done_ = true;
                onefail_net_cv_.notify_all();
                onefail_xor_cv_.notify_all();
                continue;
            }
        }
        std::cout << "ECLATIN: [One-fail] net worker done (rig" << rank_in_group_ << ")" << std::endl;
    }

    void onefail_xor_worker() {
        std::cout << "ECLATIN: [One-fail] XOR worker started (rig" << rank_in_group_ << ")" << std::endl;
        while (!stop_.load()) {
            OnefailPipelineTask task;
            {
                std::unique_lock<std::mutex> lk(onefail_xor_mutex_);
                onefail_xor_cv_.wait(lk, [this] {
                    return stop_.load() || !onefail_xor_q_.empty();
                });
                if (stop_.load() && onefail_xor_q_.empty()) break;
                if (onefail_xor_q_.empty()) continue;
                task = onefail_xor_q_.front();
                onefail_xor_q_.pop();
            }
            if (task.is_sentinel()) {
                onefail_xor_done_ = true;
                continue;
            }
            try {
                if (rank_in_group_ == 2 && task.kind == OnefailTaskKind::Recv) {
                    std::lock_guard<std::mutex> pool_lk(xor_pool_work_mutex_);
                    if (task.noshared) {
                        onefail_noshared_xor_chunk_(
                            task.ns_recv,
                            task.out_d1, task.out_d2, task.out_p1, task.out_p2,
                            task.size);
                    } else {
                        onefail_xor_chunk_(
                            task.r0d2, task.r0p2, task.r1d1, task.r1p1,
                            task.r3d1, task.r3d2,
                            task.out_d1, task.out_d2, task.out_p1, task.out_p2,
                            task.size);
                    }
                    onefail_release_pool_buffers(task);
                }
            } catch (...) {
                onefail_set_pipeline_error(std::current_exception());
                onefail_net_done_ = true;
                onefail_xor_done_ = true;
                onefail_net_cv_.notify_all();
                onefail_xor_cv_.notify_all();
                continue;
            }
        }
        std::cout << "ECLATIN: [One-fail] XOR worker done (rig" << rank_in_group_ << ")" << std::endl;
    }

    // ── Two-fail v2: chunk-level helpers (pipeline workers + bulk pybind) ────

    void twofail_exchange_chunk_(
        int rank_in_group,
        uintptr_t send_d1, uintptr_t send_d2,
        uintptr_t recv_d1, uintptr_t recv_d2,
        size_t size
    ) {
        survivor_exchange_data(rank_in_group, send_d1, send_d2, recv_d1, recv_d2, size);
    }

    void twofail_xor_chunk_(
        int rank_in_group,
        uintptr_t out1, uintptr_t out2, uintptr_t out3, uintptr_t out4,
        uintptr_t own_d1, uintptr_t own_d2, uintptr_t own_p1, uintptr_t own_p2,
        uintptr_t peer_d1, uintptr_t peer_d2,
        size_t size
    ) {
        survivor_xor_decode(
            rank_in_group, out1, out2, out3, out4,
            own_d1, own_d2, own_p1, own_p2, peer_d1, peer_d2, size);
    }

    void twofail_send_chunk_(
        int rank_in_group,
        uintptr_t rig0_a1, uintptr_t rig0_a2,
        uintptr_t rig1_a1, uintptr_t rig1_a2,
        size_t size
    ) {
        send_to_both_failed_ranks(rank_in_group, rig0_a1, rig0_a2, rig1_a1, rig1_a2, size);
    }

    void twofail_recv_chunk_(
        int rank_in_group,
        uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4,
        size_t size
    ) {
        recv_four_blocks(rank_in_group, a1, a2, a3, a4, size);
    }

    // ── Two-fail v2: chunked recovery pipeline ───────────────────────────────

    void twofail_set_pipeline_error(std::exception_ptr ex) {
        if (!ex) return;
        std::lock_guard<std::mutex> lock(twofail_exch_mutex_);
        if (!twofail_pipeline_error_) {
            twofail_pipeline_error_ = ex;
        }
    }

    void twofail_rethrow_if_error() {
        if (twofail_pipeline_error_) {
            std::rethrow_exception(twofail_pipeline_error_);
        }
    }

    void twofail_release_pool_buffers(const TwofailPipelineTask& task) {
        if (task.num_release <= 0) return;
        std::lock_guard<std::mutex> lock(twofail_release_mutex_);
        for (int i = 0; i < task.num_release; ++i) {
            if (task.release_addrs[i] != 0) {
                twofail_buffers_to_release_.push(task.release_addrs[i]);
            }
        }
    }

    void reset_twofail_pipeline() {
        twofail_pipeline_error_ = nullptr;
        twofail_exch_done_ = false;
        twofail_xor_done_ = false;
        twofail_fwd_done_ = false;
        {
            std::lock_guard<std::mutex> lk(twofail_exch_mutex_);
            while (!twofail_exch_q_.empty()) twofail_exch_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lk(twofail_xor_mutex_);
            while (!twofail_xor_q_.empty()) twofail_xor_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lk(twofail_fwd_mutex_);
            while (!twofail_fwd_q_.empty()) twofail_fwd_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lk(twofail_release_mutex_);
            while (!twofail_buffers_to_release_.empty()) twofail_buffers_to_release_.pop();
        }
        std::cout << "ECLATIN: [Two-fail v2] pipeline reset (rig" << rank_in_group_ << ")" << std::endl;
    }

    void submit_twofail_survivor_chunk(
        int rank_in_group,
        uintptr_t own_d1, uintptr_t own_d2,
        uintptr_t own_p1, uintptr_t own_p2,
        uintptr_t peer_d1, uintptr_t peer_d2,
        uintptr_t out1, uintptr_t out2, uintptr_t out3, uintptr_t out4,
        size_t size, int chunk_index,
        uintptr_t rel0, uintptr_t rel1, uintptr_t rel2,
        uintptr_t rel3, uintptr_t rel4, uintptr_t rel5
    ) {
        if (!is_twofail_v2_) {
            throw std::runtime_error("ECLATIN: submit_twofail_survivor_chunk called but not in two-fail v2 mode");
        }
        TwofailPipelineTask task;
        task.size = size;
        task.chunk_index = chunk_index;
        task.own_d1 = own_d1;
        task.own_d2 = own_d2;
        task.own_p1 = own_p1;
        task.own_p2 = own_p2;
        task.peer_d1 = peer_d1;
        task.peer_d2 = peer_d2;
        task.out1 = out1;
        task.out2 = out2;
        task.out3 = out3;
        task.out4 = out4;
        task.release_addrs[0] = rel0;
        task.release_addrs[1] = rel1;
        task.release_addrs[2] = rel2;
        task.release_addrs[3] = rel3;
        task.release_addrs[4] = rel4;
        task.release_addrs[5] = rel5;
        task.num_release = 6;
        {
            std::lock_guard<std::mutex> lk(twofail_exch_mutex_);
            twofail_exch_q_.push(task);
        }
        twofail_exch_cv_.notify_one();
    }

    void submit_twofail_failed_chunk(
        int rank_in_group,
        uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4,
        size_t size, int chunk_index
    ) {
        if (!is_twofail_v2_) {
            throw std::runtime_error("ECLATIN: submit_twofail_failed_chunk called but not in two-fail v2 mode");
        }
        TwofailPipelineTask task;
        task.size = size;
        task.chunk_index = chunk_index;
        task.recv_a1 = a1;
        task.recv_a2 = a2;
        task.recv_a3 = a3;
        task.recv_a4 = a4;
        task.num_release = 0;
        {
            std::lock_guard<std::mutex> lk(twofail_fwd_mutex_);
            twofail_fwd_q_.push(task);
        }
        twofail_fwd_cv_.notify_one();
    }

    void submit_twofail_pipeline_sentinels() {
        TwofailPipelineTask sentinel = TwofailPipelineTask::make_sentinel();
        {
            std::lock_guard<std::mutex> lk(twofail_exch_mutex_);
            twofail_exch_q_.push(sentinel);
        }
        twofail_exch_cv_.notify_one();
    }

    std::vector<uintptr_t> get_twofail_buffers_to_release() {
        std::vector<uintptr_t> out;
        std::lock_guard<std::mutex> lk(twofail_release_mutex_);
        while (!twofail_buffers_to_release_.empty()) {
            out.push_back(twofail_buffers_to_release_.front());
            twofail_buffers_to_release_.pop();
        }
        return out;
    }

    void wait_for_twofail_pipeline_completion() {
        int spin = 0;
        while (!twofail_exch_done_ || !twofail_xor_done_ || !twofail_fwd_done_) {
            if (spin++ % 100 == 0) {
                std::cout << "ECLATIN: [Two-fail v2] waiting for pipeline workers: exch="
                          << twofail_exch_done_.load() << " xor=" << twofail_xor_done_.load()
                          << " fwd=" << twofail_fwd_done_.load() << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        twofail_rethrow_if_error();
        std::cout << "ECLATIN: [Two-fail v2] pipeline completed (rig" << rank_in_group_ << ")" << std::endl;
    }

    void twofail_exch_worker() {
        std::cout << "ECLATIN: [Two-fail v2] exchange worker started (rig" << rank_in_group_ << ")" << std::endl;
        while (!stop_.load()) {
            TwofailPipelineTask task;
            {
                std::unique_lock<std::mutex> lk(twofail_exch_mutex_);
                twofail_exch_cv_.wait(lk, [this] {
                    return stop_.load() || !twofail_exch_q_.empty();
                });
                if (stop_.load() && twofail_exch_q_.empty()) break;
                if (twofail_exch_q_.empty()) continue;
                task = twofail_exch_q_.front();
                twofail_exch_q_.pop();
            }
            if (task.is_sentinel()) {
                {
                    std::lock_guard<std::mutex> lk(twofail_xor_mutex_);
                    twofail_xor_q_.push(task);
                }
                twofail_xor_cv_.notify_one();
                twofail_exch_done_ = true;
                continue;
            }
            try {
                if (rank_in_group_ == 2 || rank_in_group_ == 3) {
                    twofail_exchange_chunk_(
                        rank_in_group_, task.own_d1, task.own_d2,
                        task.peer_d1, task.peer_d2, task.size);
                }
                {
                    std::lock_guard<std::mutex> lk(twofail_xor_mutex_);
                    twofail_xor_q_.push(task);
                }
                twofail_xor_cv_.notify_one();
            } catch (...) {
                twofail_set_pipeline_error(std::current_exception());
                twofail_exch_done_ = true;
                twofail_xor_done_ = true;
                twofail_fwd_done_ = true;
                twofail_exch_cv_.notify_all();
                twofail_xor_cv_.notify_all();
                twofail_fwd_cv_.notify_all();
                continue;
            }
        }
        std::cout << "ECLATIN: [Two-fail v2] exchange worker done (rig" << rank_in_group_ << ")" << std::endl;
    }

    void twofail_xor_worker() {
        std::cout << "ECLATIN: [Two-fail v2] XOR worker started (rig" << rank_in_group_ << ")" << std::endl;
        while (!stop_.load()) {
            TwofailPipelineTask task;
            {
                std::unique_lock<std::mutex> lk(twofail_xor_mutex_);
                twofail_xor_cv_.wait(lk, [this] {
                    return stop_.load() || !twofail_xor_q_.empty();
                });
                if (stop_.load() && twofail_xor_q_.empty()) break;
                if (twofail_xor_q_.empty()) continue;
                task = twofail_xor_q_.front();
                twofail_xor_q_.pop();
            }
            if (task.is_sentinel()) {
                {
                    std::lock_guard<std::mutex> lk(twofail_fwd_mutex_);
                    twofail_fwd_q_.push(task);
                }
                twofail_fwd_cv_.notify_one();
                twofail_xor_done_ = true;
                continue;
            }
            try {
                if (rank_in_group_ == 2 || rank_in_group_ == 3) {
                    std::lock_guard<std::mutex> pool_lk(xor_pool_work_mutex_);
                    twofail_xor_chunk_(
                        rank_in_group_, task.out1, task.out2, task.out3, task.out4,
                        task.own_d1, task.own_d2, task.own_p1, task.own_p2,
                        task.peer_d1, task.peer_d2, task.size);
                }
                {
                    std::lock_guard<std::mutex> lk(twofail_fwd_mutex_);
                    twofail_fwd_q_.push(task);
                }
                twofail_fwd_cv_.notify_one();
            } catch (...) {
                twofail_set_pipeline_error(std::current_exception());
                twofail_exch_done_ = true;
                twofail_xor_done_ = true;
                twofail_fwd_done_ = true;
                twofail_exch_cv_.notify_all();
                twofail_xor_cv_.notify_all();
                twofail_fwd_cv_.notify_all();
                continue;
            }
        }
        std::cout << "ECLATIN: [Two-fail v2] XOR worker done (rig" << rank_in_group_ << ")" << std::endl;
    }

    void twofail_fwd_worker() {
        std::cout << "ECLATIN: [Two-fail v2] forward worker started (rig" << rank_in_group_ << ")" << std::endl;
        while (!stop_.load()) {
            TwofailPipelineTask task;
            {
                std::unique_lock<std::mutex> lk(twofail_fwd_mutex_);
                twofail_fwd_cv_.wait(lk, [this] {
                    return stop_.load() || !twofail_fwd_q_.empty();
                });
                if (stop_.load() && twofail_fwd_q_.empty()) break;
                if (twofail_fwd_q_.empty()) continue;
                task = twofail_fwd_q_.front();
                twofail_fwd_q_.pop();
            }
            if (task.is_sentinel()) {
                twofail_fwd_done_ = true;
                continue;
            }
            try {
                if (rank_in_group_ == 2) {
                    twofail_send_chunk_(
                        rank_in_group_, task.out3, task.out2, task.out1, task.out4, task.size);
                    twofail_release_pool_buffers(task);
                } else if (rank_in_group_ == 3) {
                    twofail_send_chunk_(
                        rank_in_group_, task.out1, task.out4, task.out3, task.out2, task.size);
                    twofail_release_pool_buffers(task);
                } else if (rank_in_group_ == 0 || rank_in_group_ == 1) {
                    twofail_recv_chunk_(
                        rank_in_group_, task.recv_a1, task.recv_a2,
                        task.recv_a3, task.recv_a4, task.size);
                }
            } catch (...) {
                twofail_set_pipeline_error(std::current_exception());
                twofail_exch_done_ = true;
                twofail_xor_done_ = true;
                twofail_fwd_done_ = true;
                twofail_exch_cv_.notify_all();
                twofail_xor_cv_.notify_all();
                twofail_fwd_cv_.notify_all();
                continue;
            }
        }
        std::cout << "ECLATIN: [Two-fail v2] forward worker done (rig" << rank_in_group_ << ")" << std::endl;
    }

    // Layer-wise processing functions
    void submit_layer_wise(
        int layer_id,
        pybind11::list gpu_tensors_info,
        uintptr_t cpu_buffer_addr,
        size_t aligned_layer_size,
        size_t actual_layer_size,
        uintptr_t data_block_1_base,
        uintptr_t data_block_2_base,
        uintptr_t parity_block_1_base,
        uintptr_t parity_block_2_base,
        size_t data_block_1_offset,
        size_t data_block_2_offset,
        size_t parity_block_1_offset,
        size_t parity_block_2_offset,
        uintptr_t recv1_parity1_addr,
        uintptr_t recv2_parity1_addr,
        uintptr_t recv1_parity2_addr,
        uintptr_t recv2_parity2_addr
    ) {
        LayerWiseTask task;
        task.layer_id = layer_id;
        task.cpu_buffer_addr = cpu_buffer_addr;
        task.aligned_layer_size = aligned_layer_size;
        task.actual_layer_size = actual_layer_size;
        task.data_block_1_addr = data_block_1_base + data_block_1_offset;
        task.data_block_2_addr = data_block_2_base + data_block_2_offset;
        task.parity_block_1_addr = parity_block_1_base + parity_block_1_offset;
        task.parity_block_2_addr = parity_block_2_base + parity_block_2_offset;
        task.recv1_parity1_addr = recv1_parity1_addr;
        task.recv2_parity1_addr = recv2_parity1_addr;
        task.recv1_parity2_addr = recv1_parity2_addr;
        task.recv2_parity2_addr = recv2_parity2_addr;
        
        // Parse GPU tensor info
        for (auto item : gpu_tensors_info) {
            pybind11::tuple tensor_tuple = item.cast<pybind11::tuple>();
            if (tensor_tuple.size() != 5) {
                throw std::runtime_error("Each tensor info must be (gpu_ptr, cpu_offset, size, shape, name)");
            }
            
            TensorTransferInfo info;
            info.gpu_data_ptr = tensor_tuple[0].cast<uintptr_t>();
            info.cpu_offset = tensor_tuple[1].cast<size_t>();
            info.size_bytes = tensor_tuple[2].cast<size_t>();
            info.shape = tensor_tuple[3].cast<std::vector<int64_t>>();
            info.name = tensor_tuple[4].cast<std::string>();
            task.gpu_tensors.push_back(info);
        }
        
        // Add to queue
        {
            std::lock_guard<std::mutex> lock(layerwise_mutex_);
            layerwise_queue_.push(task);
            layers_submitted_++;
        }
        layerwise_cv_.notify_one();
    }
    
    void wait_all_layers_complete() {
        std::unique_lock<std::mutex> lock(completion_mutex_);
        completion_cv_.wait(lock, [this] {
            return layers_completed_ >= layers_submitted_ && layerwise_queue_.empty();
        });
    }
    
    // Layerwise load methods
    void submit_layer_wise_load(
        int layer_id,
        pybind11::list gpu_tensors_info,
        uintptr_t recv_rank0_data2_addr,
        uintptr_t recv_rank0_parity2_addr,
        uintptr_t recv_rank1_data1_addr,
        uintptr_t recv_rank1_parity1_addr,
        uintptr_t recv_rank3_data1_addr,
        uintptr_t recv_rank3_data2_addr,
        uintptr_t recovered_data1_addr,
        uintptr_t recovered_data2_addr,
        uintptr_t recovered_parity1_addr,
        uintptr_t recovered_parity2_addr,
        size_t layer_size
    ) {
        LayerWiseLoadTask task;
        task.layer_id = layer_id;
        task.recv_rank0_data2_addr = recv_rank0_data2_addr;
        task.recv_rank0_parity2_addr = recv_rank0_parity2_addr;
        task.recv_rank1_data1_addr = recv_rank1_data1_addr;
        task.recv_rank1_parity1_addr = recv_rank1_parity1_addr;
        task.recv_rank3_data1_addr = recv_rank3_data1_addr;
        task.recv_rank3_data2_addr = recv_rank3_data2_addr;
        task.recovered_data1_addr = recovered_data1_addr;
        task.recovered_data2_addr = recovered_data2_addr;
        task.recovered_parity1_addr = recovered_parity1_addr;
        task.recovered_parity2_addr = recovered_parity2_addr;
        task.layer_size = layer_size;
        
        // Parse GPU tensor info
        for (auto tensor_obj : gpu_tensors_info) {
            auto tensor_tuple = tensor_obj.cast<pybind11::tuple>();
            if (tensor_tuple.size() != 5) {
                throw std::runtime_error("ECLATIN Load: Expected tuple of (gpu_ptr, cpu_offset, size, shape, name)");
            }
            
            TensorTransferInfo info;
            info.gpu_data_ptr = tensor_tuple[0].cast<uintptr_t>();
            info.cpu_offset = tensor_tuple[1].cast<size_t>();
            info.size_bytes = tensor_tuple[2].cast<size_t>();
            info.shape = tensor_tuple[3].cast<std::vector<int64_t>>();
            info.name = tensor_tuple[4].cast<std::string>();
            task.gpu_tensors.push_back(info);
        }
        
        // Add to load queue
        {
            std::lock_guard<std::mutex> lock(layerwise_load_mutex_);
            layerwise_load_queue_.push(task);
            layers_load_submitted_++;
        }
        layerwise_load_cv_.notify_one();
        
        std::cout << "ECLATIN Load: Submitted layer " << layer_id 
                  << " for load pipeline (size=" << layer_size << ")" << std::endl;
    }
    
    void wait_all_load_layers_complete() {
        std::unique_lock<std::mutex> lock(load_completion_mutex_);
        load_completion_cv_.wait(lock, [this] {
            return layers_load_completed_ >= layers_load_submitted_ && layerwise_load_queue_.empty();
        });
        std::cout << "ECLATIN Load: All " << layers_load_completed_ << " layers completed" << std::endl;
    }
    
    // Reset layerwise load statistics
    void reset_layerwise_load_statistics() {
        std::lock_guard<std::mutex> lock(layerwise_stats_mutex_);
        per_layer_recovery_time_ms_.clear();
        per_layer_h2d_time_ms_.clear();
        per_layer_total_time_ms_.clear();
        total_layerwise_recovery_time_ms_ = 0.0;
        total_layerwise_h2d_time_ms_ = 0.0;
        layerwise_critical_path_time_ms_ = 0.0;
        layerwise_first_layer_started_ = false;
        std::cout << "ECLATIN: Reset layerwise load statistics" << std::endl;
    }
    
    // Get layerwise load statistics (return as map for Python)
    std::map<std::string, pybind11::object> get_layerwise_load_statistics() {
        std::lock_guard<std::mutex> lock(layerwise_stats_mutex_);
        std::map<std::string, pybind11::object> stats;
        
        // Per-layer statistics
        pybind11::dict per_layer_recovery;
        pybind11::dict per_layer_h2d;
        pybind11::dict per_layer_total;
        
        for (const auto& [layer_id, time_ms] : per_layer_recovery_time_ms_) {
            per_layer_recovery[pybind11::str(std::to_string(layer_id))] = time_ms;
        }
        for (const auto& [layer_id, time_ms] : per_layer_h2d_time_ms_) {
            per_layer_h2d[pybind11::str(std::to_string(layer_id))] = time_ms;
        }
        for (const auto& [layer_id, time_ms] : per_layer_total_time_ms_) {
            per_layer_total[pybind11::str(std::to_string(layer_id))] = time_ms;
        }
        
        stats["per_layer_recovery_ms"] = per_layer_recovery;
        stats["per_layer_h2d_ms"] = per_layer_h2d;
        stats["per_layer_total_ms"] = per_layer_total;
        stats["total_recovery_ms"] = pybind11::cast(total_layerwise_recovery_time_ms_.load());
        stats["total_h2d_ms"] = pybind11::cast(total_layerwise_h2d_time_ms_.load());
        stats["critical_path_ms"] = pybind11::cast(layerwise_critical_path_time_ms_.load());
        
        return stats;
    }
    
    // Print layerwise load statistics
    void print_layerwise_load_statistics() {
        std::lock_guard<std::mutex> lock(layerwise_stats_mutex_);
        
        std::cout << "ECLATIN Layerwise Load: Time Statistics" << std::endl;
        
        if (!per_layer_total_time_ms_.empty()) {
            std::cout << "  Per-Layer Breakdown:" << std::endl;
            for (const auto& [layer_id, total_time] : per_layer_total_time_ms_) {
                std::cout << "    Layer " << layer_id << ":" << std::endl;
                std::cout << "      Total Time: " << total_time << " ms (" << (total_time / 1000.0) << " s)" << std::endl;
                
                auto recovery_it = per_layer_recovery_time_ms_.find(layer_id);
                if (recovery_it != per_layer_recovery_time_ms_.end()) {
                    double recovery_time = recovery_it->second;
                    std::cout << "      Recovery Time: " << recovery_time << " ms (" << (recovery_time / 1000.0) << " s)" << std::endl;
                }
                
                auto h2d_it = per_layer_h2d_time_ms_.find(layer_id);
                if (h2d_it != per_layer_h2d_time_ms_.end()) {
                    double h2d_time = h2d_it->second;
                    std::cout << "      H2D Time: " << h2d_time << " ms (" << (h2d_time / 1000.0) << " s)" << std::endl;
                }
            }
        }
        
        std::cout << "  Accumulated Statistics:" << std::endl;
        if (total_layerwise_recovery_time_ms_.load() > 0.0) {
            std::cout << "    Total Recovery Time: " << total_layerwise_recovery_time_ms_.load() 
                      << " ms (" << (total_layerwise_recovery_time_ms_.load() / 1000.0) << " s)" << std::endl;
        }
        std::cout << "    Total H2D Time: " << total_layerwise_h2d_time_ms_.load() 
                  << " ms (" << (total_layerwise_h2d_time_ms_.load() / 1000.0) << " s)" << std::endl;
        std::cout << "    Critical Path Time: " << layerwise_critical_path_time_ms_.load() 
                  << " ms (" << (layerwise_critical_path_time_ms_.load() / 1000.0) << " s)" << std::endl;
    }

private:
    std::atomic<bool> stop_;

    // ASIO connections for pipelines
    AsioConnectionManager conn_;
    
    // Parity 1 network config
    std::string parity1_send1_ip_;
    uint16_t parity1_send1_port_;
    std::string parity1_send2_ip_;
    uint16_t parity1_send2_port_;
    std::string parity1_recv1_ip_;
    uint16_t parity1_recv1_port_;
    std::string parity1_recv2_ip_;
    uint16_t parity1_recv2_port_;
    
    // Parity 2 network config
    std::string parity2_send1_ip_;
    uint16_t parity2_send1_port_;
    std::string parity2_send2_ip_;
    uint16_t parity2_send2_port_;
    std::string parity2_recv1_ip_;
    uint16_t parity2_recv1_port_;
    std::string parity2_recv2_ip_;
    uint16_t parity2_recv2_port_;

    // Parity 1 pipelines
    std::queue<SendTask> parity1_send1_q_;
    std::mutex parity1_send1_mutex_;
    std::condition_variable parity1_send1_cv_;
    std::queue<SendTask> parity1_send2_q_;
    std::mutex parity1_send2_mutex_;
    std::condition_variable parity1_send2_cv_;
    std::queue<RecvXorTask> parity1_recv_xor_q_;
    std::mutex parity1_recv_xor_mutex_;
    std::condition_variable parity1_recv_xor_cv_;

    // Parity 2 pipelines
    std::queue<SendTask> parity2_send1_q_;
    std::mutex parity2_send1_mutex_;
    std::condition_variable parity2_send1_cv_;
    std::queue<SendTask> parity2_send2_q_;
    std::mutex parity2_send2_mutex_;
    std::condition_variable parity2_send2_cv_;
    std::queue<RecvXorTask> parity2_recv_xor_q_;
    std::mutex parity2_recv_xor_mutex_;
    std::condition_variable parity2_recv_xor_cv_;

    // Separate release queues for data and recv buffers
    std::queue<uintptr_t> data_buffers_to_release_;
    std::queue<uintptr_t> recv_buffers_to_release_;
    std::mutex release_queue_mutex_;

    // Completion flags
    std::atomic<bool> parity1_send1_completed_;
    std::atomic<bool> parity1_send2_completed_;
    std::atomic<bool> parity1_recv_xor_completed_;
    std::atomic<bool> parity2_send1_completed_;
    std::atomic<bool> parity2_send2_completed_;
    std::atomic<bool> parity2_recv_xor_completed_;

    // Sentinel received flags
    std::atomic<bool> parity1_send1_sentinel_received_;
    std::atomic<bool> parity1_send2_sentinel_received_;
    std::atomic<bool> parity1_recv_xor_sentinel_received_;
    std::atomic<bool> parity2_send1_sentinel_received_;
    std::atomic<bool> parity2_send2_sentinel_received_;
    std::atomic<bool> parity2_recv_xor_sentinel_received_;
    
    // Time statistics (accumulated time from all operations)
    std::atomic<double> total_encoding_time_ms_;
    std::atomic<double> total_send_time_ms_;
    std::atomic<double> total_recv_time_ms_;
    std::atomic<double> total_xor_time_ms_;
    std::atomic<int> encoding_count_;
    std::atomic<int> send_count_;
    std::atomic<int> recv_count_;
    std::atomic<int> xor_count_;
    
    // Per-worker execution time (from first task to last task completion)
    std::atomic<double> parity1_send1_total_time_ms_{0.0};
    std::atomic<double> parity1_send2_total_time_ms_{0.0};
    std::atomic<double> parity1_recv_xor_total_time_ms_{0.0};
    std::atomic<double> parity2_send1_total_time_ms_{0.0};
    std::atomic<double> parity2_send2_total_time_ms_{0.0};
    std::atomic<double> parity2_recv_xor_total_time_ms_{0.0};
    
    // Per-worker operation time breakdown (for bottleneck analysis)
    std::atomic<double> parity1_send1_ops_time_ms_{0.0};
    std::atomic<double> parity1_send2_ops_time_ms_{0.0};
    std::atomic<double> parity1_recv_xor_recv_time_ms_{0.0};
    std::atomic<double> parity1_recv_xor_xor_time_ms_{0.0};
    std::atomic<double> parity2_send1_ops_time_ms_{0.0};
    std::atomic<double> parity2_send2_ops_time_ms_{0.0};
    std::atomic<double> parity2_recv_xor_recv_time_ms_{0.0};
    std::atomic<double> parity2_recv_xor_xor_time_ms_{0.0};
    
    // Pipeline wall-clock time
    std::chrono::high_resolution_clock::time_point pipeline_start_time_;
    std::atomic<bool> pipeline_timing_started_{false};

    std::thread parity1_send1_thread_;
    std::thread parity1_send2_thread_;
    std::thread parity1_recv_xor_thread_;
    std::thread parity2_send1_thread_;
    std::thread parity2_send2_thread_;
    std::thread parity2_recv_xor_thread_;

    // Two-fail v2 chunked recovery pipeline (3-stage: exchange -> xor -> forward)
    std::queue<TwofailPipelineTask> twofail_exch_q_;
    std::mutex twofail_exch_mutex_;
    std::condition_variable twofail_exch_cv_;
    std::queue<TwofailPipelineTask> twofail_xor_q_;
    std::mutex twofail_xor_mutex_;
    std::condition_variable twofail_xor_cv_;
    std::queue<TwofailPipelineTask> twofail_fwd_q_;
    std::mutex twofail_fwd_mutex_;
    std::condition_variable twofail_fwd_cv_;
    std::thread twofail_exch_thread_;
    std::thread twofail_xor_thread_;
    std::thread twofail_fwd_thread_;
    std::atomic<bool> twofail_exch_done_{false};
    std::atomic<bool> twofail_xor_done_{false};
    std::atomic<bool> twofail_fwd_done_{false};
    std::queue<uintptr_t> twofail_buffers_to_release_;
    std::mutex twofail_release_mutex_;
    std::exception_ptr twofail_pipeline_error_{nullptr};

    // One-fail (rig2) chunked load pipeline (2-stage: net -> xor)
    std::queue<OnefailPipelineTask> onefail_net_q_;
    std::mutex onefail_net_mutex_;
    std::condition_variable onefail_net_cv_;
    std::queue<OnefailPipelineTask> onefail_xor_q_;
    std::mutex onefail_xor_mutex_;
    std::condition_variable onefail_xor_cv_;
    std::thread onefail_net_thread_;
    std::thread onefail_xor_thread_;
    std::atomic<bool> onefail_net_done_{false};
    std::atomic<bool> onefail_xor_done_{false};
    std::queue<uintptr_t> onefail_buffers_to_release_;
    std::mutex onefail_release_mutex_;
    std::exception_ptr onefail_pipeline_error_{nullptr};
    
    // Layer-wise processing (save mode)
    std::queue<LayerWiseTask> layerwise_queue_;
    std::mutex layerwise_mutex_;
    std::condition_variable layerwise_cv_;
    std::thread layerwise_worker_thread_;
    std::atomic<int> layers_submitted_{0};
    std::atomic<int> layers_completed_{0};
    std::mutex completion_mutex_;
    std::condition_variable completion_cv_;
    
    // Layer-wise load processing
    std::queue<LayerWiseLoadTask> layerwise_load_queue_;
    std::mutex layerwise_load_mutex_;
    std::condition_variable layerwise_load_cv_;
    std::thread layerwise_load_worker_thread_;
    std::atomic<int> layers_load_submitted_{0};
    std::atomic<int> layers_load_completed_{0};
    std::mutex load_completion_mutex_;
    std::condition_variable load_completion_cv_;
    
    // Load mode flags
    std::atomic<bool> is_load_mode_{false};
    bool use_no_shared_block_{false};
    int failed_rank_{-1};
    int failed_rank_in_group_{-1};  // failed rank within 4-rank group (for multi-group support)
    bool is_twofail_v2_{false};     // two-failures v2 mode (survivor-side XOR decode)

    // Multi-rank: global rank, world size, rank within group (0..3 per group)
    int rank_{-1};
    int world_size_{-1};
    int rank_in_group_{-1};

    // Layerwise load time statistics
    std::mutex layerwise_stats_mutex_;
    std::map<int, double> per_layer_recovery_time_ms_;      // layer_id -> recovery time (Rank 2 only)
    std::map<int, double> per_layer_h2d_time_ms_;            // layer_id -> H2D time (all ranks)
    std::map<int, double> per_layer_total_time_ms_;          // layer_id -> total time (all ranks)
    std::atomic<double> total_layerwise_recovery_time_ms_{0.0};  // Accumulated (Rank 2 only)
    std::atomic<double> total_layerwise_h2d_time_ms_{0.0};       // Accumulated (all ranks)
    std::atomic<double> layerwise_critical_path_time_ms_{0.0};   // From first layer start to last layer end
    std::chrono::high_resolution_clock::time_point layerwise_first_layer_start_;
    std::chrono::high_resolution_clock::time_point layerwise_last_layer_end_;
    std::atomic<bool> layerwise_first_layer_started_{false};
    
    // CUDA async transfer configuration
    int num_cuda_streams_;
    bool use_async_cuda_;
    #ifdef USE_CUDA
    std::vector<cudaStream_t> cuda_streams_;
    #endif
    
    // RDMA configuration
    bool use_rdma_;
#if RDMA_AVAILABLE
    static const int RDMA_NUM_SAVE_CHANNELS = 8;  // parity1 send1/send2, recv1/recv2; parity2 send1/send2, recv1/recv2
    static const int RDMA_NUM_LOAD_CHANNELS = 6;   // rank2 recv: rank0_data2, rank0_parity2, rank1_data1, rank1_parity1, rank3_data1, rank3_data2
    static const int RDMA_NUM_LOAD_CHANNELS_TWO_FAIL = 8;  // two-fail: 4 blocks from each of 2 peers
    static const int RDMA_NUM_TWF_V2_SURVEXCH = 4;  // v2 survivor exchange: 2 send + 2 recv
    static const int RDMA_NUM_TWF_V2_PEER = 2;      // v2 per peer: 2 blocks per direction
    ibv_context* rdma_context_;
    ibv_pd* rdma_pd_;
    ibv_cq* rdma_send_cq_[RDMA_NUM_SAVE_CHANNELS];
    ibv_cq* rdma_recv_cq_[RDMA_NUM_SAVE_CHANNELS];
    ibv_cq* rdma_load_send_cq_[RDMA_NUM_LOAD_CHANNELS];
    ibv_cq* rdma_load_recv_cq_[RDMA_NUM_LOAD_CHANNELS];
    ibv_cq* rdma_load_send_cq_noshared_[NS_LOAD_NUM_CHANNELS];
    ibv_cq* rdma_load_recv_cq_noshared_[NS_LOAD_NUM_CHANNELS];
    ibv_cq* rdma_load_send_cq_two_fail_[RDMA_NUM_LOAD_CHANNELS_TWO_FAIL];
    ibv_cq* rdma_load_recv_cq_two_fail_[RDMA_NUM_LOAD_CHANNELS_TWO_FAIL];
    // v2 two-fail: survivor exchange (4 ch over twf_surv_exch)
    ibv_cq* rdma_twf_v2_sx_send_cq_[RDMA_NUM_TWF_V2_SURVEXCH];
    ibv_cq* rdma_twf_v2_sx_recv_cq_[RDMA_NUM_TWF_V2_SURVEXCH];
    // v2 two-fail: peer connections (2 ch each over twf_n1_n3/n1_n4/n2_n3/n2_n4)
    ibv_cq* rdma_twf_v2_n1n3_send_cq_[RDMA_NUM_TWF_V2_PEER];
    ibv_cq* rdma_twf_v2_n1n3_recv_cq_[RDMA_NUM_TWF_V2_PEER];
    ibv_cq* rdma_twf_v2_n1n4_send_cq_[RDMA_NUM_TWF_V2_PEER];
    ibv_cq* rdma_twf_v2_n1n4_recv_cq_[RDMA_NUM_TWF_V2_PEER];
    ibv_cq* rdma_twf_v2_n2n3_send_cq_[RDMA_NUM_TWF_V2_PEER];
    ibv_cq* rdma_twf_v2_n2n3_recv_cq_[RDMA_NUM_TWF_V2_PEER];
    ibv_cq* rdma_twf_v2_n2n4_send_cq_[RDMA_NUM_TWF_V2_PEER];
    ibv_cq* rdma_twf_v2_n2n4_recv_cq_[RDMA_NUM_TWF_V2_PEER];
    std::map<uintptr_t, RdmaBuffer> rdma_registered_buffers_;
    std::mutex rdma_buffer_mutex_;
    std::array<std::unique_ptr<RdmaConnectionChannel>, RDMA_NUM_SAVE_CHANNELS> rdma_save_channels_;
    std::array<std::unique_ptr<RdmaConnectionChannel>, RDMA_NUM_LOAD_CHANNELS> rdma_load_channels_;
    std::array<std::unique_ptr<RdmaConnectionChannel>, NS_LOAD_NUM_CHANNELS> rdma_load_channels_noshared_;
    std::array<std::unique_ptr<RdmaConnectionChannel>, RDMA_NUM_LOAD_CHANNELS_TWO_FAIL> rdma_load_channels_two_fail_;
    // v2 two-fail RDMA channels
    std::array<std::unique_ptr<RdmaConnectionChannel>, RDMA_NUM_TWF_V2_SURVEXCH> rdma_twf_v2_survexch_channels_;
    std::mutex twf_survexch_control_mutex_;  // serializes TCP handshake across 4 QPs on one socket
    std::mutex twf_n1n3_control_mutex_;
    std::mutex twf_n1n4_control_mutex_;
    std::mutex twf_n2n3_control_mutex_;
    std::mutex twf_n2n4_control_mutex_;
    std::array<std::unique_ptr<RdmaConnectionChannel>, RDMA_NUM_TWF_V2_PEER> rdma_twf_v2_n1n3_channels_;
    std::array<std::unique_ptr<RdmaConnectionChannel>, RDMA_NUM_TWF_V2_PEER> rdma_twf_v2_n1n4_channels_;
    std::array<std::unique_ptr<RdmaConnectionChannel>, RDMA_NUM_TWF_V2_PEER> rdma_twf_v2_n2n3_channels_;
    std::array<std::unique_ptr<RdmaConnectionChannel>, RDMA_NUM_TWF_V2_PEER> rdma_twf_v2_n2n4_channels_;

    // Zero buffer for 2-input XOR via 16-thread pool (d_i = p_X XOR D_Y)
    std::vector<uint8_t> twofail_zero_buffer_;
#endif

    // ── XOR thread pool (matches ecnaive xor_pool) ──────
    std::array<pthread_t, kEclatinXorPoolSize> xor_pool_threads_{};
    std::array<EclatinXorPoolCtx, kEclatinXorPoolSize> xor_pool_ctx_{};
    std::array<int, kEclatinXorPoolSize> xor_pool_cpus_{};
    std::atomic<bool> xor_pool_inited_{false};
    std::atomic<bool> xor_pool_stop_{false};
    std::mutex xor_pool_mutex_;
    std::condition_variable xor_pool_worker_cv_;
    std::condition_variable xor_pool_coordinator_cv_;
    std::atomic<uint64_t> xor_pool_epoch_{0};
    std::array<uint64_t, kEclatinXorPoolSize> xor_pool_last_epoch_{};
    std::atomic<int> xor_pool_remaining_{0};
    EclatinXorPoolJob xor_pool_shared_job_{};
    std::mutex xor_pool_work_mutex_;  // serialize parity1/parity2 pool dispatch

    void start_threads() {
        xor_pool_init();
        std::cout << "ECLATIN: Starting worker threads..." << std::endl;
        parity1_send1_thread_ = std::thread(&ECLATINNative::parity1_send1_worker, this);
        parity1_send2_thread_ = std::thread(&ECLATINNative::parity1_send2_worker, this);
        parity1_recv_xor_thread_ = std::thread(&ECLATINNative::parity1_recv_xor_worker, this);
        parity2_send1_thread_ = std::thread(&ECLATINNative::parity2_send1_worker, this);
        parity2_send2_thread_ = std::thread(&ECLATINNative::parity2_send2_worker, this);
        parity2_recv_xor_thread_ = std::thread(&ECLATINNative::parity2_recv_xor_worker, this);
        layerwise_worker_thread_ = std::thread(&ECLATINNative::layerwise_worker, this);
        layerwise_load_worker_thread_ = std::thread(&ECLATINNative::layerwise_load_worker, this);
        twofail_exch_thread_ = std::thread(&ECLATINNative::twofail_exch_worker, this);
        twofail_xor_thread_ = std::thread(&ECLATINNative::twofail_xor_worker, this);
        twofail_fwd_thread_ = std::thread(&ECLATINNative::twofail_fwd_worker, this);
        onefail_net_thread_ = std::thread(&ECLATINNative::onefail_net_worker, this);
        onefail_xor_thread_ = std::thread(&ECLATINNative::onefail_xor_worker, this);
        std::cout << "ECLATIN: All worker threads started (including layerwise load, twofail and onefail pipelines)" << std::endl;
    }

    // ── XOR pool methods ──────────────────────────────────────────────

    static std::array<int, kEclatinXorPoolSize> xor_parse_cpus() {
        std::array<int, kEclatinXorPoolSize> cpus{};
        const char* env = std::getenv(kEclatinXorCpuListEnv);
        if (!env || !*env) {
            for (int i = 0; i < kEclatinXorPoolSize; ++i)
                cpus[static_cast<size_t>(i)] = i;
            std::cout << "ECLATIN: " << kEclatinXorCpuListEnv
                      << " not set; XOR pool binds workers to CPUs 0.."
                      << (kEclatinXorPoolSize - 1) << std::endl;
            return cpus;
        }
        std::vector<int> parsed;
        const char* p = env;
        while (*p) {
            while (*p && (std::isspace(static_cast<unsigned char>(*p)) || *p == ','))
                ++p;
            if (!*p) break;
            char* end = nullptr;
            long v = std::strtol(p, &end, 10);
            if (end == p || v < 0 || v > 65535)
                throw std::runtime_error(std::string(kEclatinXorCpuListEnv) + ": invalid CPU id token");
            parsed.push_back(static_cast<int>(v));
            p = end;
        }
        if (parsed.size() != static_cast<size_t>(kEclatinXorPoolSize))
            throw std::runtime_error(
                std::string(kEclatinXorCpuListEnv) +
                " must contain exactly 16 comma-separated CPU ids (or unset to use 0..15)");
        for (size_t i = 0; i < cpus.size(); ++i)
            cpus[i] = parsed[i];
        return cpus;
    }

    void xor_pool_init() {
        if (xor_pool_inited_.load(std::memory_order_acquire)) return;
        xor_pool_cpus_ = xor_parse_cpus();
        xor_pool_stop_.store(false, std::memory_order_release);
        xor_pool_epoch_.store(0, std::memory_order_release);
        xor_pool_remaining_.store(0, std::memory_order_release);
        for (auto& e : xor_pool_last_epoch_) e = 0;
        for (int i = 0; i < kEclatinXorPoolSize; ++i) {
            xor_pool_ctx_[static_cast<size_t>(i)].self = this;
            xor_pool_ctx_[static_cast<size_t>(i)].wid = i;
            int rc = pthread_create(&xor_pool_threads_[static_cast<size_t>(i)], nullptr,
                                    &ECLATINNative::xor_pool_pthread_entry,
                                    &xor_pool_ctx_[static_cast<size_t>(i)]);
            if (rc != 0) {
                xor_pool_stop_.store(true, std::memory_order_release);
                xor_pool_worker_cv_.notify_all();
                for (int j = 0; j < i; ++j)
                    pthread_join(xor_pool_threads_[static_cast<size_t>(j)], nullptr);
                throw std::runtime_error("ECLATIN: pthread_create for XOR pool failed: " +
                                         std::string(std::strerror(rc)));
            }
        }
        xor_pool_inited_.store(true, std::memory_order_release);
        std::cout << "ECLATIN: XOR pthread pool (" << kEclatinXorPoolSize
                  << " workers) initialized" << std::endl;
    }

    void xor_pool_shutdown() {
        if (!xor_pool_inited_.load(std::memory_order_acquire)) return;
        xor_pool_stop_.store(true, std::memory_order_release);
        xor_pool_worker_cv_.notify_all();
        for (int i = 0; i < kEclatinXorPoolSize; ++i)
            pthread_join(xor_pool_threads_[static_cast<size_t>(i)], nullptr);
        xor_pool_stop_.store(false, std::memory_order_release);
        xor_pool_inited_.store(false, std::memory_order_release);
        std::cout << "ECLATIN: XOR pthread pool shut down" << std::endl;
    }

    static void* xor_pool_pthread_entry(void* arg) {
        auto* ctx = static_cast<EclatinXorPoolCtx*>(arg);
        ctx->self->xor_pool_worker_loop(ctx->wid);
        return nullptr;
    }

    static void xor_pool_chunk_range(int total_len, int wid, size_t& off, size_t& len) {
        const size_t total = static_cast<size_t>(total_len);
        const size_t base = total / static_cast<size_t>(kEclatinXorPoolSize);
        const size_t rem = total % static_cast<size_t>(kEclatinXorPoolSize);
        if (wid < kEclatinXorPoolSize - 1) {
            off = static_cast<size_t>(wid) * base;
            len = base;
        } else {
            off = static_cast<size_t>(kEclatinXorPoolSize - 1) * base;
            len = base + rem;
        }
    }

    void xor_pool_execute_chunk(const EclatinXorPoolJob& job, int wid) {
        size_t off = 0;
        size_t len = 0;
        xor_pool_chunk_range(job.len, wid, off, len);
        if (len == 0) return;

        uint8_t* dst = reinterpret_cast<uint8_t*>(job.dst) + off;
        uint8_t* s1  = reinterpret_cast<uint8_t*>(job.src1) + off;
        uint8_t* s2  = reinterpret_cast<uint8_t*>(job.src2) + off;

        void* xa[3] = {dst, s1, s2};
        xor_gen(3, static_cast<int>(len), xa);
    }

    // dst = src0 XOR src1; out-of-place per stripe chunk inside each worker (no memcpy)
    void xor_pool_execute_two_stripe_chunk(const EclatinXorPoolJob& job, int wid) {
        size_t off = 0;
        size_t len = 0;
        xor_pool_chunk_range(job.len, wid, off, len);
        if (len == 0) return;

        for (int k = 0; k < 2; ++k) {
            const auto& op = job.stripe_op[static_cast<size_t>(k)];
            xor_two_out_of_place(
                reinterpret_cast<uint8_t*>(op.dst) + off,
                reinterpret_cast<const uint8_t*>(op.src0) + off,
                reinterpret_cast<const uint8_t*>(op.src1) + off,
                len);
        }
    }

    void xor_pool_worker_loop(int wid) {
        const int cpu = xor_pool_cpus_[static_cast<size_t>(wid)];
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (cpu >= 0 && static_cast<unsigned>(cpu) < CPU_SETSIZE) {
            CPU_SET(static_cast<unsigned>(cpu), &cpuset);
            int af = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
            if (af != 0)
                std::cerr << "ECLATIN: xor_pool worker " << wid
                          << " pthread_setaffinity_np failed: " << af << std::endl;
        }

        while (true) {
            std::unique_lock<std::mutex> lk(xor_pool_mutex_);
            xor_pool_worker_cv_.wait(lk, [&] {
                return xor_pool_stop_.load(std::memory_order_acquire) ||
                       (xor_pool_last_epoch_[static_cast<size_t>(wid)] <
                        xor_pool_epoch_.load(std::memory_order_acquire));
            });
            if (xor_pool_stop_.load(std::memory_order_acquire)) break;
            uint64_t e = xor_pool_epoch_.load(std::memory_order_acquire);
            EclatinXorPoolJob local_copy = xor_pool_shared_job_;
            lk.unlock();

            if (local_copy.kind == EclatinXorPoolJobKind::TwoStripe) {
                xor_pool_execute_two_stripe_chunk(local_copy, wid);
            } else {
                xor_pool_execute_chunk(local_copy, wid);
            }

            {
                std::lock_guard<std::mutex> guard(xor_pool_mutex_);
                xor_pool_last_epoch_[static_cast<size_t>(wid)] = e;
            }
            int left = xor_pool_remaining_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            if (left == 0)
                xor_pool_coordinator_cv_.notify_one();
        }
    }

    void xor_pool_dispatch_and_wait(const EclatinXorPoolJob& job) {
        {
            std::lock_guard<std::mutex> publish(xor_pool_mutex_);
            if (stop_.load(std::memory_order_acquire)) return;
            xor_pool_shared_job_ = job;
            xor_pool_epoch_.fetch_add(1, std::memory_order_acq_rel);
            xor_pool_remaining_.store(kEclatinXorPoolSize, std::memory_order_release);
        }
        xor_pool_worker_cv_.notify_all();
        std::unique_lock<std::mutex> lk(xor_pool_mutex_);
        xor_pool_coordinator_cv_.wait(lk, [&] {
            return xor_pool_remaining_.load(std::memory_order_acquire) == 0 ||
                   stop_.load(std::memory_order_acquire);
        });
    }

    void xor_pool_run_parallel(uintptr_t dst, uintptr_t src1, uintptr_t src2, int len) {
        EclatinXorPoolJob job;
        job.kind = EclatinXorPoolJobKind::ThreeInput;
        job.len = len;
        job.dst = dst;
        job.src1 = src1;
        job.src2 = src2;
        xor_pool_dispatch_and_wait(job);
    }

    // Two independent dst=src0 XOR src1 ops in one pool round (out-of-place per chunk in workers)
    void xor_pool_run_parallel_two_stripe(
        const EclatinXorStripeOp& op0,
        const EclatinXorStripeOp& op1,
        int len
    ) {
        EclatinXorPoolJob job;
        job.kind = EclatinXorPoolJobKind::TwoStripe;
        job.len = len;
        job.stripe_op[0] = op0;
        job.stripe_op[1] = op1;
        xor_pool_dispatch_and_wait(job);
    }

    // ── Connection initialization ─────────────────────────────────────

    void init_connections() {
        std::cout << "ECLATIN: Initializing connections..." << std::endl;
        // Start acceptors in separate threads to avoid deadlock (mirror eccheck pattern)
        std::thread recv_init_thread([this]() {
            std::thread r1_1([this]() { conn_.init_parity1_recv1(parity1_recv1_ip_, parity1_recv1_port_); });
            std::thread r2_1([this]() { conn_.init_parity1_recv2(parity1_recv2_ip_, parity1_recv2_port_); });
            std::thread r1_2([this]() { conn_.init_parity2_recv1(parity2_recv1_ip_, parity2_recv1_port_); });
            std::thread r2_2([this]() { conn_.init_parity2_recv2(parity2_recv2_ip_, parity2_recv2_port_); });
            r1_1.join();
            r2_1.join();
            r1_2.join();
            r2_2.join();
        });

        // Small delay to ensure acceptors are listening
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Connect send sockets (blocking)
        std::cout << "ECLATIN: Connecting parity1_send1 socket..." << std::endl;
        conn_.init_parity1_send1(parity1_send1_ip_, parity1_send1_port_);
        std::cout << "ECLATIN: Connecting parity1_send2 socket..." << std::endl;
        conn_.init_parity1_send2(parity1_send2_ip_, parity1_send2_port_);
        std::cout << "ECLATIN: Connecting parity2_send1 socket..." << std::endl;
        conn_.init_parity2_send1(parity2_send1_ip_, parity2_send1_port_);
        std::cout << "ECLATIN: Connecting parity2_send2 socket..." << std::endl;
        conn_.init_parity2_send2(parity2_send2_ip_, parity2_send2_port_);

        recv_init_thread.join();
        std::cout << "ECLATIN: Waiting for all connections..." << std::endl;
        conn_.wait_for_connections();
        std::cout << "ECLATIN: All connections established" << std::endl;
        // Note: init_rdma_save_channels() is called after init_rdma_resources() in constructor
    }

#if RDMA_AVAILABLE
    void init_rdma_save_channels() {
        if (!use_rdma_ || !rdma_pd_) return;
        std::cout << "[ECLATIN RDMA] Creating 8 RDMA save channels..." << std::endl;
        int rank_for_log = (rank_in_group_ >= 0) ? rank_in_group_ : 0;
        auto& c = conn_;
        // Create all 8 channel objects (no exchange yet)
        rdma_save_channels_[0] = std::make_unique<RdmaConnectionChannel>(
            rdma_context_, rdma_pd_, rdma_send_cq_[0], rdma_recv_cq_[0],
            c.get_parity1_send1_socket().native_handle(), c.get_parity1_send1_socket().native_handle(),
            &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
        rdma_save_channels_[1] = std::make_unique<RdmaConnectionChannel>(
            rdma_context_, rdma_pd_, rdma_send_cq_[1], rdma_recv_cq_[1],
            c.get_parity1_send2_socket().native_handle(), c.get_parity1_send2_socket().native_handle(),
            &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
        rdma_save_channels_[2] = std::make_unique<RdmaConnectionChannel>(
            rdma_context_, rdma_pd_, rdma_send_cq_[2], rdma_recv_cq_[2],
            c.get_parity1_recv1_socket().native_handle(), c.get_parity1_recv1_socket().native_handle(),
            &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
        rdma_save_channels_[3] = std::make_unique<RdmaConnectionChannel>(
            rdma_context_, rdma_pd_, rdma_send_cq_[3], rdma_recv_cq_[3],
            c.get_parity1_recv2_socket().native_handle(), c.get_parity1_recv2_socket().native_handle(),
            &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
        rdma_save_channels_[4] = std::make_unique<RdmaConnectionChannel>(
            rdma_context_, rdma_pd_, rdma_send_cq_[4], rdma_recv_cq_[4],
            c.get_parity2_send1_socket().native_handle(), c.get_parity2_send1_socket().native_handle(),
            &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
        rdma_save_channels_[5] = std::make_unique<RdmaConnectionChannel>(
            rdma_context_, rdma_pd_, rdma_send_cq_[5], rdma_recv_cq_[5],
            c.get_parity2_send2_socket().native_handle(), c.get_parity2_send2_socket().native_handle(),
            &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
        rdma_save_channels_[6] = std::make_unique<RdmaConnectionChannel>(
            rdma_context_, rdma_pd_, rdma_send_cq_[6], rdma_recv_cq_[6],
            c.get_parity2_recv1_socket().native_handle(), c.get_parity2_recv1_socket().native_handle(),
            &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
        rdma_save_channels_[7] = std::make_unique<RdmaConnectionChannel>(
            rdma_context_, rdma_pd_, rdma_send_cq_[7], rdma_recv_cq_[7],
            c.get_parity2_recv2_socket().native_handle(), c.get_parity2_recv2_socket().native_handle(),
            &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);

        // Run exchanges by connection-pair (like eccheck): on each TCP connection exactly one side
        // sends first and one recvs first to avoid deadlock.
        if (rank_in_group_ < 0) {
            // Fallback: original order (may deadlock)
            rdma_save_channels_[0]->exchange_and_connect(true);
            rdma_save_channels_[1]->exchange_and_connect(true);
            rdma_save_channels_[2]->exchange_and_connect(false);
            rdma_save_channels_[3]->exchange_and_connect(false);
            rdma_save_channels_[4]->exchange_and_connect(true);
            rdma_save_channels_[5]->exchange_and_connect(true);
            rdma_save_channels_[6]->exchange_and_connect(false);
            rdma_save_channels_[7]->exchange_and_connect(false);
        } else {
            const int r = rank_in_group_;
            // Round 1: parity1 (0 send1, 2 recv2)
            if (r == 0) rdma_save_channels_[0]->exchange_and_connect(true);
            if (r == 2) rdma_save_channels_[3]->exchange_and_connect(false);
            // Round 2: parity1 (1 send1, 3 recv2)
            if (r == 1) rdma_save_channels_[0]->exchange_and_connect(true);
            if (r == 3) rdma_save_channels_[3]->exchange_and_connect(false);
            // Round 3: parity1 (0 send2, 1 recv1)
            if (r == 0) rdma_save_channels_[1]->exchange_and_connect(true);
            if (r == 1) rdma_save_channels_[2]->exchange_and_connect(false);
            // Round 4: parity1 (2 send2, 3 recv1)
            if (r == 2) rdma_save_channels_[1]->exchange_and_connect(true);
            if (r == 3) rdma_save_channels_[2]->exchange_and_connect(false);
            // Round 5: parity2 (0 send1, 3 recv2)
            if (r == 0) rdma_save_channels_[4]->exchange_and_connect(true);
            if (r == 3) rdma_save_channels_[7]->exchange_and_connect(false);
            // Round 6: parity2 (1 send1, 2 recv2)
            if (r == 1) rdma_save_channels_[4]->exchange_and_connect(true);
            if (r == 2) rdma_save_channels_[7]->exchange_and_connect(false);
            // Round 7: parity2 (0 send2, 2 recv1)
            if (r == 0) rdma_save_channels_[5]->exchange_and_connect(true);
            if (r == 2) rdma_save_channels_[6]->exchange_and_connect(false);
            // Round 8: parity2 (1 send2, 3 recv1)
            if (r == 1) rdma_save_channels_[5]->exchange_and_connect(true);
            if (r == 3) rdma_save_channels_[6]->exchange_and_connect(false);
            // === Reverse direction exchanges (8 missing rounds) ===
            // Round 9: parity1 (2 send1, 0 recv2)
            if (r == 2) rdma_save_channels_[0]->exchange_and_connect(true);
            if (r == 0) rdma_save_channels_[3]->exchange_and_connect(false);
            // Round 10: parity1 (3 send1, 1 recv2)
            if (r == 3) rdma_save_channels_[0]->exchange_and_connect(true);
            if (r == 1) rdma_save_channels_[3]->exchange_and_connect(false);
            // Round 11: parity1 (1 send2, 0 recv1)
            if (r == 1) rdma_save_channels_[1]->exchange_and_connect(true);
            if (r == 0) rdma_save_channels_[2]->exchange_and_connect(false);
            // Round 12: parity1 (3 send2, 2 recv1)
            if (r == 3) rdma_save_channels_[1]->exchange_and_connect(true);
            if (r == 2) rdma_save_channels_[2]->exchange_and_connect(false);
            // Round 13: parity2 (3 send1, 0 recv2)
            if (r == 3) rdma_save_channels_[4]->exchange_and_connect(true);
            if (r == 0) rdma_save_channels_[7]->exchange_and_connect(false);
            // Round 14: parity2 (2 send1, 1 recv2)
            if (r == 2) rdma_save_channels_[4]->exchange_and_connect(true);
            if (r == 1) rdma_save_channels_[7]->exchange_and_connect(false);
            // Round 15: parity2 (2 send2, 0 recv1)
            if (r == 2) rdma_save_channels_[5]->exchange_and_connect(true);
            if (r == 0) rdma_save_channels_[6]->exchange_and_connect(false);
            // Round 16: parity2 (3 send2, 1 recv1)
            if (r == 3) rdma_save_channels_[5]->exchange_and_connect(true);
            if (r == 1) rdma_save_channels_[6]->exchange_and_connect(false);
        }
        std::cout << "[ECLATIN RDMA] All 8 save channels connected" << std::endl;
    }

    void init_rdma_resources() {
        if (!use_rdma_) return;
        std::cout << "[ECLATIN RDMA] Initializing RDMA resources (8 CQ pairs for save)..." << std::endl;

        if (ibv_fork_init() != 0) {
            std::cerr << "[ECLATIN RDMA] WARNING: ibv_fork_init() failed. Forked processes may get Bad address."
                      << std::endl;
        }

        int num_devices;
        ibv_device** device_list = ibv_get_device_list(&num_devices);
        if (!device_list || num_devices == 0) {
            throw std::runtime_error("ECLATIN RDMA: No RDMA devices found");
        }

        rdma_context_ = ibv_open_device(
            find_rdma_device_by_ip(parity1_recv1_ip_, device_list, num_devices));
        if (!rdma_context_) {
            ibv_free_device_list(device_list);
            throw std::runtime_error("ECLATIN RDMA: Failed to open RDMA device");
        }
        ibv_free_device_list(device_list);

        rdma_pd_ = ibv_alloc_pd(rdma_context_);
        if (!rdma_pd_) {
            ibv_close_device(rdma_context_);
            rdma_context_ = nullptr;
            throw std::runtime_error("ECLATIN RDMA: Failed to allocate protection domain");
        }

        for (int i = 0; i < RDMA_NUM_SAVE_CHANNELS; ++i) {
            rdma_send_cq_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            rdma_recv_cq_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            if (!rdma_send_cq_[i] || !rdma_recv_cq_[i]) {
                for (int j = 0; j < i; ++j) {
                    if (rdma_send_cq_[j]) { ibv_destroy_cq(rdma_send_cq_[j]); rdma_send_cq_[j] = nullptr; }
                    if (rdma_recv_cq_[j]) { ibv_destroy_cq(rdma_recv_cq_[j]); rdma_recv_cq_[j] = nullptr; }
                }
                ibv_dealloc_pd(rdma_pd_);
                ibv_close_device(rdma_context_);
                rdma_pd_ = nullptr;
                rdma_context_ = nullptr;
                throw std::runtime_error("ECLATIN RDMA: Failed to create completion queues for channel " + std::to_string(i));
            }
        }
        std::cout << "[ECLATIN RDMA] RDMA resources initialized (8 CQ pairs for save)" << std::endl;
    }

    void init_rdma_load_resources() {
        if (!use_rdma_ || !rdma_pd_) return;
        std::cout << "[ECLATIN RDMA] Initializing load RDMA resources (6 CQ pairs)..." << std::endl;
        for (int i = 0; i < RDMA_NUM_LOAD_CHANNELS; ++i) {
            rdma_load_send_cq_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            rdma_load_recv_cq_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            if (!rdma_load_send_cq_[i] || !rdma_load_recv_cq_[i]) {
                for (int j = 0; j < i; ++j) {
                    if (rdma_load_send_cq_[j]) { ibv_destroy_cq(rdma_load_send_cq_[j]); rdma_load_send_cq_[j] = nullptr; }
                    if (rdma_load_recv_cq_[j]) { ibv_destroy_cq(rdma_load_recv_cq_[j]); rdma_load_recv_cq_[j] = nullptr; }
                }
                throw std::runtime_error("ECLATIN RDMA: Failed to create load CQs for channel " + std::to_string(i));
            }
        }
        std::cout << "[ECLATIN RDMA] Load RDMA resources initialized (6 CQ pairs)" << std::endl;
    }

    void init_rdma_load_resources_noshared() {
        if (!use_rdma_ || !rdma_pd_) return;
        if (rdma_load_send_cq_noshared_[0] != nullptr) return;
        if (rdma_context_ == nullptr) {
            init_rdma_resources();
        }
        std::cout << "[ECLATIN RDMA] Initializing no-shared load RDMA resources ("
                  << NS_LOAD_NUM_CHANNELS << " CQ pairs)..." << std::endl;
        for (int i = 0; i < NS_LOAD_NUM_CHANNELS; ++i) {
            rdma_load_send_cq_noshared_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            rdma_load_recv_cq_noshared_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            if (!rdma_load_send_cq_noshared_[i] || !rdma_load_recv_cq_noshared_[i]) {
                for (int j = 0; j < i; ++j) {
                    if (rdma_load_send_cq_noshared_[j]) {
                        ibv_destroy_cq(rdma_load_send_cq_noshared_[j]);
                        rdma_load_send_cq_noshared_[j] = nullptr;
                    }
                    if (rdma_load_recv_cq_noshared_[j]) {
                        ibv_destroy_cq(rdma_load_recv_cq_noshared_[j]);
                        rdma_load_recv_cq_noshared_[j] = nullptr;
                    }
                }
                throw std::runtime_error(
                    "ECLATIN RDMA: Failed to create no-shared load CQs for channel "
                    + std::to_string(i));
            }
        }
        std::cout << "[ECLATIN RDMA] No-shared load RDMA resources initialized ("
                  << NS_LOAD_NUM_CHANNELS << " CQ pairs)" << std::endl;
    }

    void setup_ns_rdma_load_channel(int ch, bool is_sender, int peer_rank) {
#if RDMA_AVAILABLE
        if (!use_rdma_ || !rdma_pd_) return;
        auto& sock = is_sender
            ? conn_.get_ns_load_send_socket(ch)
            : conn_.get_ns_load_recv_socket(ch);
        rdma_load_channels_noshared_[ch] = std::make_unique<RdmaConnectionChannel>(
            rdma_context_, rdma_pd_,
            rdma_load_send_cq_noshared_[ch], rdma_load_recv_cq_noshared_[ch],
            sock.native_handle(), sock.native_handle(),
            &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_in_group_, peer_rank);
        rdma_load_channels_noshared_[ch]->exchange_and_connect(is_sender);
#else
        (void)ch; (void)is_sender; (void)peer_rank;
#endif
    }

    void init_rdma_load_channels() {
        if (!use_rdma_ || !rdma_pd_) return;
        int rank_for_log = (rank_in_group_ >= 0) ? rank_in_group_ : 0;
        auto& c = conn_;
        try {
            if (rank_in_group_ == 2) {
                std::cout << "[ECLATIN RDMA] Creating 6 RDMA load channels (rank2 recv)..." << std::endl;
                rdma_load_channels_[0] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[0], rdma_load_recv_cq_[0],
                    c.get_load_recv_rank0_data2_socket().native_handle(), c.get_load_recv_rank0_data2_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
                rdma_load_channels_[0]->exchange_and_connect(false);
                rdma_load_channels_[1] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[1], rdma_load_recv_cq_[1],
                    c.get_load_recv_rank0_parity2_socket().native_handle(), c.get_load_recv_rank0_parity2_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
                rdma_load_channels_[1]->exchange_and_connect(false);
                rdma_load_channels_[2] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[2], rdma_load_recv_cq_[2],
                    c.get_load_recv_rank1_data1_socket().native_handle(), c.get_load_recv_rank1_data1_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
                rdma_load_channels_[2]->exchange_and_connect(false);
                rdma_load_channels_[3] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[3], rdma_load_recv_cq_[3],
                    c.get_load_recv_rank1_parity1_socket().native_handle(), c.get_load_recv_rank1_parity1_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
                rdma_load_channels_[3]->exchange_and_connect(false);
                rdma_load_channels_[4] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[4], rdma_load_recv_cq_[4],
                    c.get_load_recv_rank3_data1_socket().native_handle(), c.get_load_recv_rank3_data1_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
                rdma_load_channels_[4]->exchange_and_connect(false);
                rdma_load_channels_[5] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[5], rdma_load_recv_cq_[5],
                    c.get_load_recv_rank3_data2_socket().native_handle(), c.get_load_recv_rank3_data2_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 0);
                rdma_load_channels_[5]->exchange_and_connect(false);
                std::cout << "[ECLATIN RDMA] All 6 load channels connected (rank2)" << std::endl;
            } else if (rank_in_group_ == 0) {
                std::cout << "[ECLATIN RDMA] Creating 2 RDMA load channels (rank0 send)..." << std::endl;
                rdma_load_channels_[0] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[0], rdma_load_recv_cq_[0],
                    c.get_load_send_rank0_data2_socket().native_handle(), c.get_load_send_rank0_data2_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 2);
                rdma_load_channels_[0]->exchange_and_connect(true);
                rdma_load_channels_[1] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[1], rdma_load_recv_cq_[1],
                    c.get_load_send_rank0_parity2_socket().native_handle(), c.get_load_send_rank0_parity2_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 2);
                rdma_load_channels_[1]->exchange_and_connect(true);
                std::cout << "[ECLATIN RDMA] All 2 load channels connected (rank0)" << std::endl;
            } else if (rank_in_group_ == 1) {
                std::cout << "[ECLATIN RDMA] Creating 2 RDMA load channels (rank1 send)..." << std::endl;
                rdma_load_channels_[2] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[2], rdma_load_recv_cq_[2],
                    c.get_load_send_rank1_data1_socket().native_handle(), c.get_load_send_rank1_data1_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 2);
                rdma_load_channels_[2]->exchange_and_connect(true);
                rdma_load_channels_[3] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[3], rdma_load_recv_cq_[3],
                    c.get_load_send_rank1_parity1_socket().native_handle(), c.get_load_send_rank1_parity1_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 2);
                rdma_load_channels_[3]->exchange_and_connect(true);
                std::cout << "[ECLATIN RDMA] All 2 load channels connected (rank1)" << std::endl;
            } else if (rank_in_group_ == 3) {
                std::cout << "[ECLATIN RDMA] Creating 2 RDMA load channels (rank3 send)..." << std::endl;
                rdma_load_channels_[4] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[4], rdma_load_recv_cq_[4],
                    c.get_load_send_rank3_data1_socket().native_handle(), c.get_load_send_rank3_data1_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 2);
                rdma_load_channels_[4]->exchange_and_connect(true);
                rdma_load_channels_[5] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_load_send_cq_[5], rdma_load_recv_cq_[5],
                    c.get_load_send_rank3_data2_socket().native_handle(), c.get_load_send_rank3_data2_socket().native_handle(),
                    &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 2);
                rdma_load_channels_[5]->exchange_and_connect(true);
                std::cout << "[ECLATIN RDMA] All 2 load channels connected (rank3)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[ECLATIN RDMA] Failed to init load channels: " << e.what() << std::endl;
            for (int i = 0; i < RDMA_NUM_LOAD_CHANNELS; ++i) rdma_load_channels_[i].reset();
            throw;
        }
    }

    // ── Two-failures RDMA load resource init ─────────────────────────────────

    void init_rdma_load_resources_two_fail() {
        if (!use_rdma_ || !rdma_pd_) return;
        std::cout << "[ECLATIN RDMA] Initializing two-fail load RDMA resources (8 CQ pairs)..." << std::endl;
        for (int i = 0; i < RDMA_NUM_LOAD_CHANNELS_TWO_FAIL; ++i) {
            rdma_load_send_cq_two_fail_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            rdma_load_recv_cq_two_fail_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            if (!rdma_load_send_cq_two_fail_[i] || !rdma_load_recv_cq_two_fail_[i]) {
                for (int j = 0; j < i; ++j) {
                    if (rdma_load_send_cq_two_fail_[j]) { ibv_destroy_cq(rdma_load_send_cq_two_fail_[j]); rdma_load_send_cq_two_fail_[j] = nullptr; }
                    if (rdma_load_recv_cq_two_fail_[j]) { ibv_destroy_cq(rdma_load_recv_cq_two_fail_[j]); rdma_load_recv_cq_two_fail_[j] = nullptr; }
                }
                throw std::runtime_error("ECLATIN RDMA: Failed to create two-fail load CQs for channel " + std::to_string(i));
            }
        }
        std::cout << "[ECLATIN RDMA] Two-fail load RDMA resources initialized (8 CQ pairs)" << std::endl;
    }

    void init_rdma_load_channels_two_fail() {
        if (!use_rdma_ || !rdma_pd_) return;
        int rank_for_log = (rank_in_group_ >= 0) ? rank_in_group_ : 0;
        auto& c = conn_;

        std::cout << "[ECLATIN RDMA] Creating 8 two-fail RDMA load channels (rank_in_group "
                  << rank_in_group_ << ")..." << std::endl;

        try {
            if (rank_in_group_ == 0 || rank_in_group_ == 1) {
                // Failed ranks: 8 recv channels (4 from rank2 peer0, 4 from rank3 peer1)
                // Channels 0-3: peer0 (rank 2), Channels 4-7: peer1 (rank 3)
                for (int i = 0; i < 8; ++i) {
                    boost::asio::ip::tcp::socket& sock = (i < 4)
                        ? c.get_load_twofail_peer0_socket()
                        : c.get_load_twofail_peer1_socket();
                    rdma_load_channels_two_fail_[i] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_,
                        rdma_load_send_cq_two_fail_[i], rdma_load_recv_cq_two_fail_[i],
                        sock.native_handle(), sock.native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log,
                        (i < 4) ? 2 : 3);  // peer_rank: 2 for rank2, 3 for rank3
                    rdma_load_channels_two_fail_[i]->exchange_and_connect(false);  // recv first
                }
                std::cout << "[ECLATIN RDMA] All 8 two-fail load channels connected (failed rank "
                          << rank_in_group_ << ")" << std::endl;
            } else if (rank_in_group_ == 2 || rank_in_group_ == 3) {
                // Surviving ranks: 8 send channels (4 to rank0 peer0, 4 to rank1 peer1)
                for (int i = 0; i < 8; ++i) {
                    boost::asio::ip::tcp::socket& sock = (i < 4)
                        ? c.get_load_twofail_peer0_socket()
                        : c.get_load_twofail_peer1_socket();
                    int target_rig = (i < 4) ? 0 : 1;  // peer0 → rank 0, peer1 → rank 1
                    rdma_load_channels_two_fail_[i] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_,
                        rdma_load_send_cq_two_fail_[i], rdma_load_recv_cq_two_fail_[i],
                        sock.native_handle(), sock.native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, target_rig);
                    rdma_load_channels_two_fail_[i]->exchange_and_connect(true);  // send first
                }
                std::cout << "[ECLATIN RDMA] All 8 two-fail load channels connected (surviving rank "
                          << rank_in_group_ << ")" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[ECLATIN RDMA] Failed to init two-fail load channels: " << e.what() << std::endl;
            for (int i = 0; i < RDMA_NUM_LOAD_CHANNELS_TWO_FAIL; ++i) rdma_load_channels_two_fail_[i].reset();
            throw;
        }
    }

    // ── Two-failures v2 RDMA resources ────────────────────────────────────────

    void init_rdma_twf_v2_resources() {
        if (!use_rdma_ || !rdma_pd_) return;
        std::cout << "[ECLATIN RDMA] Initializing two-fail v2 RDMA resources "
                  << "(survexch=" << RDMA_NUM_TWF_V2_SURVEXCH
                  << ", peers=" << (RDMA_NUM_TWF_V2_PEER * 4) << " CQ pairs)..." << std::endl;

        // Survivor exchange: 4 CQ pairs
        for (int i = 0; i < RDMA_NUM_TWF_V2_SURVEXCH; ++i) {
            rdma_twf_v2_sx_send_cq_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            rdma_twf_v2_sx_recv_cq_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            if (!rdma_twf_v2_sx_send_cq_[i] || !rdma_twf_v2_sx_recv_cq_[i]) {
                for (int j = 0; j <= i; ++j) {
                    if (rdma_twf_v2_sx_send_cq_[j]) { ibv_destroy_cq(rdma_twf_v2_sx_send_cq_[j]); rdma_twf_v2_sx_send_cq_[j] = nullptr; }
                    if (rdma_twf_v2_sx_recv_cq_[j]) { ibv_destroy_cq(rdma_twf_v2_sx_recv_cq_[j]); rdma_twf_v2_sx_recv_cq_[j] = nullptr; }
                }
                throw std::runtime_error("ECLATIN RDMA: Failed to create v2 surv exch CQs");
            }
        }
        // Peer connections: 2 CQ pairs each × 4 peers
        auto init_peer_cqs = [this](ibv_cq** send_cqs, ibv_cq** recv_cqs, const char* name) {
            for (int i = 0; i < RDMA_NUM_TWF_V2_PEER; ++i) {
                send_cqs[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
                recv_cqs[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
                if (!send_cqs[i] || !recv_cqs[i]) {
                    for (int j = 0; j <= i; ++j) {
                        if (send_cqs[j]) { ibv_destroy_cq(send_cqs[j]); send_cqs[j] = nullptr; }
                        if (recv_cqs[j]) { ibv_destroy_cq(recv_cqs[j]); recv_cqs[j] = nullptr; }
                    }
                    throw std::runtime_error(std::string("ECLATIN RDMA: Failed to create v2 ") + name + " CQs");
                }
            }
        };
        init_peer_cqs(rdma_twf_v2_n1n3_send_cq_, rdma_twf_v2_n1n3_recv_cq_, "n1n3");
        init_peer_cqs(rdma_twf_v2_n1n4_send_cq_, rdma_twf_v2_n1n4_recv_cq_, "n1n4");
        init_peer_cqs(rdma_twf_v2_n2n3_send_cq_, rdma_twf_v2_n2n3_recv_cq_, "n2n3");
        init_peer_cqs(rdma_twf_v2_n2n4_send_cq_, rdma_twf_v2_n2n4_recv_cq_, "n2n4");

        std::cout << "[ECLATIN RDMA] Two-fail v2 RDMA resources initialized ("
                  << RDMA_NUM_TWF_V2_SURVEXCH << "+" << (RDMA_NUM_TWF_V2_PEER * 4)
                  << " CQ pairs)" << std::endl;
    }

    void init_rdma_twf_v2_channels() {
        if (!use_rdma_ || !rdma_pd_) return;
        AsioConnectionManager& c = conn_;
        int rank_for_log = rank_in_group_;

        std::cout << "[ECLATIN RDMA] Creating two-fail v2 RDMA channels (rig "
                  << rank_in_group_ << ")..." << std::endl;

        try {
            // ── Survivor exchange channels (4) ────────────────────────────
            // Only rig2 and rig3 have the surv_exch socket connected.
            if (rank_in_group_ == 2 || rank_in_group_ == 3) {
                bool surv_send_first = (rank_in_group_ == 3);
                for (int i = 0; i < RDMA_NUM_TWF_V2_SURVEXCH; ++i) {
                    auto& sock = c.get_twf_surv_exch_socket();
                    rdma_twf_v2_survexch_channels_[i] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_,
                        rdma_twf_v2_sx_send_cq_[i], rdma_twf_v2_sx_recv_cq_[i],
                        sock.native_handle(), sock.native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log,
                        (rank_in_group_ == 2) ? 3 : 2,
                        &twf_survexch_control_mutex_);
                    rdma_twf_v2_survexch_channels_[i]->exchange_and_connect(surv_send_first);
                }
                std::cout << "[ECLATIN RDMA] v2 surv exch channels connected (rig"
                          << rank_in_group_ << ")" << std::endl;
            }

            // ── Peer channels (2 each, 4 peers) ───────────────────────────
            // Helper: init peer channels
            auto init_peer_channels = [&](
                std::array<std::unique_ptr<RdmaConnectionChannel>, RDMA_NUM_TWF_V2_PEER>& channels,
                ibv_cq** send_cqs, ibv_cq** recv_cqs,
                boost::asio::ip::tcp::socket& sock,
                bool send_first, int peer_rig, const char* name,
                std::mutex* control_mutex)
            {
                for (int i = 0; i < RDMA_NUM_TWF_V2_PEER; ++i) {
                    channels[i] = std::make_unique<RdmaConnectionChannel>(
                        rdma_context_, rdma_pd_,
                        send_cqs[i], recv_cqs[i],
                        sock.native_handle(), sock.native_handle(),
                        &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, peer_rig,
                        control_mutex);
                    channels[i]->exchange_and_connect(send_first);
                }
                std::cout << "[ECLATIN RDMA] v2 " << name << " channels connected (rig"
                          << rank_in_group_ << ")" << std::endl;
            };

            // n1_n3: rig0 recv-first, rig2 send-first; peer_rig=0 for rig2, peer_rig=2 for rig0
            if (rank_in_group_ == 0 || rank_in_group_ == 2) {
                bool sf = (rank_in_group_ == 2);
                int pr = (rank_in_group_ == 2) ? 0 : 2;
                init_peer_channels(rdma_twf_v2_n1n3_channels_, rdma_twf_v2_n1n3_send_cq_,
                                   rdma_twf_v2_n1n3_recv_cq_, c.get_twf_n1_n3_socket(),
                                   sf, pr, "n1n3", &twf_n1n3_control_mutex_);
            }
            // n1_n4: rig0 recv-first, rig3 send-first
            if (rank_in_group_ == 0 || rank_in_group_ == 3) {
                bool sf = (rank_in_group_ == 3);
                int pr = (rank_in_group_ == 3) ? 0 : 3;
                init_peer_channels(rdma_twf_v2_n1n4_channels_, rdma_twf_v2_n1n4_send_cq_,
                                   rdma_twf_v2_n1n4_recv_cq_, c.get_twf_n1_n4_socket(),
                                   sf, pr, "n1n4", &twf_n1n4_control_mutex_);
            }
            // n2_n3: rig1 recv-first, rig2 send-first
            if (rank_in_group_ == 1 || rank_in_group_ == 2) {
                bool sf = (rank_in_group_ == 2);
                int pr = (rank_in_group_ == 2) ? 1 : 2;
                init_peer_channels(rdma_twf_v2_n2n3_channels_, rdma_twf_v2_n2n3_send_cq_,
                                   rdma_twf_v2_n2n3_recv_cq_, c.get_twf_n2_n3_socket(),
                                   sf, pr, "n2n3", &twf_n2n3_control_mutex_);
            }
            // n2_n4: rig1 recv-first, rig3 send-first
            if (rank_in_group_ == 1 || rank_in_group_ == 3) {
                bool sf = (rank_in_group_ == 3);
                int pr = (rank_in_group_ == 3) ? 1 : 3;
                init_peer_channels(rdma_twf_v2_n2n4_channels_, rdma_twf_v2_n2n4_send_cq_,
                                   rdma_twf_v2_n2n4_recv_cq_, c.get_twf_n2_n4_socket(),
                                   sf, pr, "n2n4", &twf_n2n4_control_mutex_);
            }

            std::cout << "[ECLATIN RDMA] All two-fail v2 channels connected (rig"
                      << rank_in_group_ << ")" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[ECLATIN RDMA] Failed to init two-fail v2 channels: " << e.what() << std::endl;
            for (int i = 0; i < RDMA_NUM_TWF_V2_SURVEXCH; ++i) rdma_twf_v2_survexch_channels_[i].reset();
            for (int i = 0; i < RDMA_NUM_TWF_V2_PEER; ++i) {
                rdma_twf_v2_n1n3_channels_[i].reset();
                rdma_twf_v2_n1n4_channels_[i].reset();
                rdma_twf_v2_n2n3_channels_[i].reset();
                rdma_twf_v2_n2n4_channels_[i].reset();
            }
            throw;
        }
    }

    void cleanup_rdma_resources() {
        if (!use_rdma_) return;
        std::cout << "[ECLATIN RDMA] Cleaning up RDMA resources..." << std::endl;
        for (int i = 0; i < RDMA_NUM_SAVE_CHANNELS; ++i) {
            rdma_save_channels_[i].reset();
        }
        for (int i = 0; i < RDMA_NUM_LOAD_CHANNELS; ++i) {
            rdma_load_channels_[i].reset();
        }
        for (int i = 0; i < NS_LOAD_NUM_CHANNELS; ++i) {
            rdma_load_channels_noshared_[i].reset();
        }
        for (int i = 0; i < RDMA_NUM_LOAD_CHANNELS_TWO_FAIL; ++i) {
            rdma_load_channels_two_fail_[i].reset();
        }
        for (int i = 0; i < RDMA_NUM_SAVE_CHANNELS; ++i) {
            if (rdma_send_cq_[i]) { ibv_destroy_cq(rdma_send_cq_[i]); rdma_send_cq_[i] = nullptr; }
            if (rdma_recv_cq_[i]) { ibv_destroy_cq(rdma_recv_cq_[i]); rdma_recv_cq_[i] = nullptr; }
        }
        for (int i = 0; i < RDMA_NUM_LOAD_CHANNELS; ++i) {
            if (rdma_load_send_cq_[i]) { ibv_destroy_cq(rdma_load_send_cq_[i]); rdma_load_send_cq_[i] = nullptr; }
            if (rdma_load_recv_cq_[i]) { ibv_destroy_cq(rdma_load_recv_cq_[i]); rdma_load_recv_cq_[i] = nullptr; }
        }
        for (int i = 0; i < NS_LOAD_NUM_CHANNELS; ++i) {
            if (rdma_load_send_cq_noshared_[i]) {
                ibv_destroy_cq(rdma_load_send_cq_noshared_[i]);
                rdma_load_send_cq_noshared_[i] = nullptr;
            }
            if (rdma_load_recv_cq_noshared_[i]) {
                ibv_destroy_cq(rdma_load_recv_cq_noshared_[i]);
                rdma_load_recv_cq_noshared_[i] = nullptr;
            }
        }
        for (int i = 0; i < RDMA_NUM_LOAD_CHANNELS_TWO_FAIL; ++i) {
            if (rdma_load_send_cq_two_fail_[i]) { ibv_destroy_cq(rdma_load_send_cq_two_fail_[i]); rdma_load_send_cq_two_fail_[i] = nullptr; }
            if (rdma_load_recv_cq_two_fail_[i]) { ibv_destroy_cq(rdma_load_recv_cq_two_fail_[i]); rdma_load_recv_cq_two_fail_[i] = nullptr; }
        }
        // v2 two-fail channels
        for (int i = 0; i < RDMA_NUM_TWF_V2_SURVEXCH; ++i) {
            rdma_twf_v2_survexch_channels_[i].reset();
            if (rdma_twf_v2_sx_send_cq_[i]) { ibv_destroy_cq(rdma_twf_v2_sx_send_cq_[i]); rdma_twf_v2_sx_send_cq_[i] = nullptr; }
            if (rdma_twf_v2_sx_recv_cq_[i]) { ibv_destroy_cq(rdma_twf_v2_sx_recv_cq_[i]); rdma_twf_v2_sx_recv_cq_[i] = nullptr; }
        }
        auto cleanup_peer = [](auto& channels, ibv_cq** scqs, ibv_cq** rcqs) {
            for (int i = 0; i < RDMA_NUM_TWF_V2_PEER; ++i) {
                channels[i].reset();
                if (scqs[i]) { ibv_destroy_cq(scqs[i]); scqs[i] = nullptr; }
                if (rcqs[i]) { ibv_destroy_cq(rcqs[i]); rcqs[i] = nullptr; }
            }
        };
        cleanup_peer(rdma_twf_v2_n1n3_channels_, rdma_twf_v2_n1n3_send_cq_, rdma_twf_v2_n1n3_recv_cq_);
        cleanup_peer(rdma_twf_v2_n1n4_channels_, rdma_twf_v2_n1n4_send_cq_, rdma_twf_v2_n1n4_recv_cq_);
        cleanup_peer(rdma_twf_v2_n2n3_channels_, rdma_twf_v2_n2n3_send_cq_, rdma_twf_v2_n2n3_recv_cq_);
        cleanup_peer(rdma_twf_v2_n2n4_channels_, rdma_twf_v2_n2n4_send_cq_, rdma_twf_v2_n2n4_recv_cq_);
        {
            std::lock_guard<std::mutex> lock(rdma_buffer_mutex_);
            for (auto& [addr, buf] : rdma_registered_buffers_) {
                if (buf.mr) ibv_dereg_mr(buf.mr);
            }
            rdma_registered_buffers_.clear();
        }
        if (rdma_pd_) {
            ibv_dealloc_pd(rdma_pd_);
            rdma_pd_ = nullptr;
        }
        if (rdma_context_) {
            ibv_close_device(rdma_context_);
            rdma_context_ = nullptr;
        }
        std::cout << "[ECLATIN RDMA] RDMA resources cleaned up" << std::endl;
    }
#endif

    // Parity 1 workers
    void parity1_recv_xor_worker() {
        std::cout << "ECLATIN: Parity1_RecvXor worker started" << std::endl;
        std::chrono::high_resolution_clock::time_point worker_start_time;
        bool worker_start_time_set = false;
        
        while (!stop_) {
            RecvXorTask task;
            {
                std::unique_lock<std::mutex> lk(parity1_recv_xor_mutex_);
                parity1_recv_xor_cv_.wait(lk, [this] { return stop_ || !parity1_recv_xor_q_.empty(); });
                if (stop_) break;
                task = parity1_recv_xor_q_.front();
                parity1_recv_xor_q_.pop();
            }
            // Check for sentinel
            if (task.recv1_addr == 0 && task.recv2_addr == 0 &&
                task.parity_addr == 0 && task.size == 0) {
                parity1_recv_xor_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity1_recv_xor_mutex_);
                    if (parity1_recv_xor_q_.empty()) {
                        if (worker_start_time_set) {
                            auto worker_end_time = std::chrono::high_resolution_clock::now();
                            double worker_total_time_ms = std::chrono::duration<double, std::milli>(worker_end_time - worker_start_time).count();
                            parity1_recv_xor_total_time_ms_.store(worker_total_time_ms);
                        }
                        parity1_recv_xor_completed_ = true;
                        std::cout << "ECLATIN: Parity1_RecvXor worker completed" << std::endl;
                        parity1_recv_xor_sentinel_received_ = false;
                        // Reset for next round
                        worker_start_time_set = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.recv1_addr == 0 ||
                task.recv2_addr == 0 || task.parity_addr == 0) {
                continue;
            }

            // Ensure connections are ready
            if (!conn_.is_parity1_recv1_connected()) {
                std::cerr << "ECLATIN: ERROR: parity1_recv1 socket not connected!" << std::endl;
                throw std::runtime_error("ECLATIN: parity1_recv1 socket not connected");
            }
            if (!conn_.is_parity1_recv2_connected()) {
                std::cerr << "ECLATIN: ERROR: parity1_recv2 socket not connected!" << std::endl;
                throw std::runtime_error("ECLATIN: parity1_recv2 socket not connected");
            }

            // Record start time when first real task begins
            if (!worker_start_time_set) {
                worker_start_time = std::chrono::high_resolution_clock::now();
                worker_start_time_set = true;
            }

            auto recv_start = std::chrono::high_resolution_clock::now();
            bool recv1_success = false;
            bool recv2_success = false;
            std::string recv1_error_msg;
            std::string recv2_error_msg;
            std::exception_ptr recv1_exception = nullptr;
            std::exception_ptr recv2_exception = nullptr;

            std::thread recv1_thread([&]() {
                try {
#if RDMA_AVAILABLE
                    if (use_rdma_ && rdma_save_channels_[2]) {
                        // std::cout << "[ECLATIN RDMA] Parity1_Recv1: Receiving " << task.size << " bytes via RDMA" << std::endl;
                        rdma_save_channels_[2]->receive_data(reinterpret_cast<uint8_t*>(task.recv1_addr), task.size);
                        recv1_success = true;
                    } else
#endif
                    {
                        std::cout << "[ECLATIN ASIO] Parity1_Recv1: Receiving " << task.size << " bytes via ASIO" << std::endl;
                        if (!recv_with_size_bool(
                                conn_.get_parity1_recv1_socket(),
                                reinterpret_cast<void*>(task.recv1_addr),
                                task.size)) {
                            recv1_error_msg = "ECLATIN: parity1_recv1_with_size_bool returned false";
                            recv1_success = false;
                        } else {
                            recv1_success = true;
                        }
                    }
                } catch (const std::exception& e) {
                    recv1_exception = std::current_exception();
                    recv1_error_msg = std::string("ECLATIN: parity1_recv1 exception: ") + e.what();
                } catch (...) {
                    recv1_exception = std::current_exception();
                    recv1_error_msg = "ECLATIN: parity1_recv1 unknown exception";
                }
            });

            std::thread recv2_thread([&]() {
                try {
#if RDMA_AVAILABLE
                    if (use_rdma_ && rdma_save_channels_[3]) {
                        // std::cout << "[ECLATIN RDMA] Parity1_Recv2: Receiving " << task.size << " bytes via RDMA" << std::endl;
                        rdma_save_channels_[3]->receive_data(reinterpret_cast<uint8_t*>(task.recv2_addr), task.size);
                        recv2_success = true;
                    } else
#endif
                    {
                        std::cout << "[ECLATIN ASIO] Parity1_Recv2: Receiving " << task.size << " bytes via ASIO" << std::endl;
                        if (!recv_with_size_bool(
                                conn_.get_parity1_recv2_socket(),
                                reinterpret_cast<void*>(task.recv2_addr),
                                task.size)) {
                            recv2_error_msg = "ECLATIN: parity1_recv2_with_size_bool returned false";
                            recv2_success = false;
                        } else {
                            recv2_success = true;
                        }
                    }
                } catch (const std::exception& e) {
                    recv2_exception = std::current_exception();
                    recv2_error_msg = std::string("ECLATIN: parity1_recv2 exception: ") + e.what();
                } catch (...) {
                    recv2_exception = std::current_exception();
                    recv2_error_msg = "ECLATIN: parity1_recv2 unknown exception";
                }
            });

            recv1_thread.join();
            recv2_thread.join();
            
            auto recv_end = std::chrono::high_resolution_clock::now();
            double recv_time_ms = std::chrono::duration<double, std::milli>(recv_end - recv_start).count();
            total_recv_time_ms_.store(total_recv_time_ms_.load() + recv_time_ms);
            parity1_recv_xor_recv_time_ms_.store(parity1_recv_xor_recv_time_ms_.load() + recv_time_ms);
            recv_count_++;

            // Bubble up failures
            if (recv1_exception) {
                std::cerr << recv1_error_msg << std::endl;
                std::rethrow_exception(recv1_exception);
            }
            if (recv2_exception) {
                std::cerr << recv2_error_msg << std::endl;
                std::rethrow_exception(recv2_exception);
            }
            if (!recv1_success) {
                std::cerr << recv1_error_msg << std::endl;
                throw std::runtime_error(recv1_error_msg);
            }
            if (!recv2_success) {
                std::cerr << recv2_error_msg << std::endl;
                throw std::runtime_error(recv2_error_msg);
            }

            // XOR after both recvs succeed → dispatch to 16-worker pool
            auto xor_start = std::chrono::high_resolution_clock::now();
            // XOR: parity = parity ⊕ recv1 ⊕ recv2  (parity pre-zeroed, result → parity)
            {
                std::lock_guard<std::mutex> pool_lk(xor_pool_work_mutex_);
                xor_pool_run_parallel(task.parity_addr, task.recv1_addr, task.recv2_addr,
                                       static_cast<int>(task.size));
            }
            auto xor_end = std::chrono::high_resolution_clock::now();
            double xor_time_ms = std::chrono::duration<double, std::milli>(xor_end - xor_start).count();
            total_xor_time_ms_.store(total_xor_time_ms_.load() + xor_time_ms);
            parity1_recv_xor_xor_time_ms_.store(parity1_recv_xor_xor_time_ms_.load() + xor_time_ms);
            xor_count_++;

            // Release recv buffers after XOR operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                recv_buffers_to_release_.push(task.recv1_addr);
                recv_buffers_to_release_.push(task.recv2_addr);
            }

            if (parity1_recv_xor_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity1_recv_xor_mutex_);
                if (parity1_recv_xor_q_.empty()) {
                    parity1_recv_xor_completed_ = true;
                    std::cout << "ECLATIN: Parity1_RecvXor worker completed" << std::endl;
                    parity1_recv_xor_sentinel_received_ = false;
                }
            }
        }
    }

    void parity1_send1_worker() {
        std::cout << "ECLATIN: Parity1_Send1 worker started" << std::endl;
        std::chrono::high_resolution_clock::time_point worker_start_time;
        bool worker_start_time_set = false;
        
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(parity1_send1_mutex_);
                parity1_send1_cv_.wait(lk, [this] { return stop_ || !parity1_send1_q_.empty(); });
                if (stop_) break;
                task = parity1_send1_q_.front();
                parity1_send1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                parity1_send1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity1_send1_mutex_);
                    if (parity1_send1_q_.empty()) {
                        if (worker_start_time_set) {
                            auto worker_end_time = std::chrono::high_resolution_clock::now();
                            double worker_total_time_ms = std::chrono::duration<double, std::milli>(worker_end_time - worker_start_time).count();
                            parity1_send1_total_time_ms_.store(worker_total_time_ms);
                        }
                        parity1_send1_completed_ = true;
                        std::cout << "ECLATIN: Parity1_Send1 worker completed" << std::endl;
                        parity1_send1_sentinel_received_ = false;
                        // Reset for next round
                        worker_start_time_set = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            
            // Record start time when first real task begins
            if (!worker_start_time_set) {
                worker_start_time = std::chrono::high_resolution_clock::now();
                worker_start_time_set = true;
            }
            
            if (conn_.is_parity1_send1_connected()) {
                auto send_start = std::chrono::high_resolution_clock::now();
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_save_channels_[0]) {
                    // std::cout << "[ECLATIN RDMA] Parity1_Send1: Sending " << task.size << " bytes via RDMA" << std::endl;
                    rdma_save_channels_[0]->send_data(reinterpret_cast<const uint8_t*>(task.addr), task.size);
                } else
#endif
                {
                    // std::cout << "[ECLATIN ASIO] Parity1_Send1: Sending " << task.size << " bytes via ASIO" << std::endl;
                    send_with_size(conn_.get_parity1_send1_socket(), task.addr, task.size);
                }
                auto send_end = std::chrono::high_resolution_clock::now();
                double send_time_ms = std::chrono::duration<double, std::milli>(send_end - send_start).count();
                total_send_time_ms_.store(total_send_time_ms_.load() + send_time_ms);
                parity1_send1_ops_time_ms_.store(parity1_send1_ops_time_ms_.load() + send_time_ms);
                send_count_++;
            }
            // Release data buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                data_buffers_to_release_.push(task.addr);
            }

            if (parity1_send1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity1_send1_mutex_);
                if (parity1_send1_q_.empty()) {
                    parity1_send1_completed_ = true;
                    std::cout << "ECLATIN: Parity1_Send1 worker completed" << std::endl;
                    parity1_send1_sentinel_received_ = false;
                }
            }
        }
    }

    void parity1_send2_worker() {
        std::cout << "ECLATIN: Parity1_Send2 worker started" << std::endl;
        std::chrono::high_resolution_clock::time_point worker_start_time;
        bool worker_start_time_set = false;
        
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(parity1_send2_mutex_);
                parity1_send2_cv_.wait(lk, [this] { return stop_ || !parity1_send2_q_.empty(); });
                if (stop_) break;
                task = parity1_send2_q_.front();
                parity1_send2_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                parity1_send2_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity1_send2_mutex_);
                    if (parity1_send2_q_.empty()) {
                        if (worker_start_time_set) {
                            auto worker_end_time = std::chrono::high_resolution_clock::now();
                            double worker_total_time_ms = std::chrono::duration<double, std::milli>(worker_end_time - worker_start_time).count();
                            parity1_send2_total_time_ms_.store(worker_total_time_ms);
                        }
                        parity1_send2_completed_ = true;
                        std::cout << "ECLATIN: Parity1_Send2 worker completed" << std::endl;
                        parity1_send2_sentinel_received_ = false;
                        // Reset for next round
                        worker_start_time_set = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            
            // Record start time when first real task begins
            if (!worker_start_time_set) {
                worker_start_time = std::chrono::high_resolution_clock::now();
                worker_start_time_set = true;
            }
            
            if (conn_.is_parity1_send2_connected()) {
                auto send_start = std::chrono::high_resolution_clock::now();
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_save_channels_[1]) {
                    // std::cout << "[ECLATIN RDMA] Parity1_Send2: Sending " << task.size << " bytes via RDMA" << std::endl;
                    rdma_save_channels_[1]->send_data(reinterpret_cast<const uint8_t*>(task.addr), task.size);
                } else
#endif
                {
                    std::cout << "[ECLATIN ASIO] Parity1_Send2: Sending " << task.size << " bytes via ASIO" << std::endl;
                    send_with_size(conn_.get_parity1_send2_socket(), task.addr, task.size);
                }
                auto send_end = std::chrono::high_resolution_clock::now();
                double send_time_ms = std::chrono::duration<double, std::milli>(send_end - send_start).count();
                total_send_time_ms_.store(total_send_time_ms_.load() + send_time_ms);
                parity1_send2_ops_time_ms_.store(parity1_send2_ops_time_ms_.load() + send_time_ms);
                send_count_++;
            }
            // Release data buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                data_buffers_to_release_.push(task.addr);
            }

            if (parity1_send2_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity1_send2_mutex_);
                if (parity1_send2_q_.empty()) {
                    parity1_send2_completed_ = true;
                    std::cout << "ECLATIN: Parity1_Send2 worker completed" << std::endl;
                    parity1_send2_sentinel_received_ = false;
                }
            }
        }
    }

    // Parity 2 workers (same logic as parity1)
    void parity2_recv_xor_worker() {
        std::cout << "ECLATIN: Parity2_RecvXor worker started" << std::endl;
        std::chrono::high_resolution_clock::time_point worker_start_time;
        bool worker_start_time_set = false;
        
        while (!stop_) {
            RecvXorTask task;
            {
                std::unique_lock<std::mutex> lk(parity2_recv_xor_mutex_);
                parity2_recv_xor_cv_.wait(lk, [this] { return stop_ || !parity2_recv_xor_q_.empty(); });
                if (stop_) break;
                task = parity2_recv_xor_q_.front();
                parity2_recv_xor_q_.pop();
            }
            // Check for sentinel
            if (task.recv1_addr == 0 && task.recv2_addr == 0 &&
                task.parity_addr == 0 && task.size == 0) {
                parity2_recv_xor_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity2_recv_xor_mutex_);
                    if (parity2_recv_xor_q_.empty()) {
                        if (worker_start_time_set) {
                            auto worker_end_time = std::chrono::high_resolution_clock::now();
                            double worker_total_time_ms = std::chrono::duration<double, std::milli>(worker_end_time - worker_start_time).count();
                            parity2_recv_xor_total_time_ms_.store(worker_total_time_ms);
                        }
                        parity2_recv_xor_completed_ = true;
                        // std::cout << "ECLATIN: Parity2_RecvXor worker completed" << std::endl;
                        parity2_recv_xor_sentinel_received_ = false;
                        // Reset for next round
                        worker_start_time_set = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.recv1_addr == 0 ||
                task.recv2_addr == 0 || task.parity_addr == 0) {
                continue;
            }

            // Ensure connections are ready
            if (!conn_.is_parity2_recv1_connected()) {
                std::cerr << "ECLATIN: ERROR: parity2_recv1 socket not connected!" << std::endl;
                throw std::runtime_error("ECLATIN: parity2_recv1 socket not connected");
            }
            if (!conn_.is_parity2_recv2_connected()) {
                std::cerr << "ECLATIN: ERROR: parity2_recv2 socket not connected!" << std::endl;
                throw std::runtime_error("ECLATIN: parity2_recv2 socket not connected");
            }

            // Record start time when first real task begins
            if (!worker_start_time_set) {
                worker_start_time = std::chrono::high_resolution_clock::now();
                worker_start_time_set = true;
            }

            auto recv_start = std::chrono::high_resolution_clock::now();
            bool recv1_success = false;
            bool recv2_success = false;
            std::string recv1_error_msg;
            std::string recv2_error_msg;
            std::exception_ptr recv1_exception = nullptr;
            std::exception_ptr recv2_exception = nullptr;

            std::thread recv1_thread([&]() {
                try {
#if RDMA_AVAILABLE
                    if (use_rdma_ && rdma_save_channels_[6]) {
                        // std::cout << "[ECLATIN RDMA] Parity2_Recv1: Receiving " << task.size << " bytes via RDMA" << std::endl;
                        rdma_save_channels_[6]->receive_data(reinterpret_cast<uint8_t*>(task.recv1_addr), task.size);
                        recv1_success = true;
                    } else
#endif
                    {
                        std::cout << "[ECLATIN ASIO] Parity2_Recv1: Receiving " << task.size << " bytes via ASIO" << std::endl;
                        if (!recv_with_size_bool(
                                conn_.get_parity2_recv1_socket(),
                                reinterpret_cast<void*>(task.recv1_addr),
                                task.size)) {
                            recv1_error_msg = "ECLATIN: parity2_recv1_with_size_bool returned false";
                            recv1_success = false;
                        } else {
                            recv1_success = true;
                        }
                    }
                } catch (const std::exception& e) {
                    recv1_exception = std::current_exception();
                    recv1_error_msg = std::string("ECLATIN: parity2_recv1 exception: ") + e.what();
                } catch (...) {
                    recv1_exception = std::current_exception();
                    recv1_error_msg = "ECLATIN: parity2_recv1 unknown exception";
                }
            });

            std::thread recv2_thread([&]() {
                try {
#if RDMA_AVAILABLE
                    if (use_rdma_ && rdma_save_channels_[7]) {
                        // std::cout << "[ECLATIN RDMA] Parity2_Recv2: Receiving " << task.size << " bytes via RDMA" << std::endl;
                        rdma_save_channels_[7]->receive_data(reinterpret_cast<uint8_t*>(task.recv2_addr), task.size);
                        recv2_success = true;
                    } else
#endif
                    {
                        std::cout << "[ECLATIN ASIO] Parity2_Recv2: Receiving " << task.size << " bytes via ASIO" << std::endl;
                        if (!recv_with_size_bool(
                                conn_.get_parity2_recv2_socket(),
                                reinterpret_cast<void*>(task.recv2_addr),
                                task.size)) {
                            recv2_error_msg = "ECLATIN: parity2_recv2_with_size_bool returned false";
                            recv2_success = false;
                        } else {
                            recv2_success = true;
                        }
                    }
                } catch (const std::exception& e) {
                    recv2_exception = std::current_exception();
                    recv2_error_msg = std::string("ECLATIN: parity2_recv2 exception: ") + e.what();
                } catch (...) {
                    recv2_exception = std::current_exception();
                    recv2_error_msg = "ECLATIN: parity2_recv2 unknown exception";
                }
            });

            recv1_thread.join();
            recv2_thread.join();
            
            auto recv_end = std::chrono::high_resolution_clock::now();
            double recv_time_ms = std::chrono::duration<double, std::milli>(recv_end - recv_start).count();
            total_recv_time_ms_.store(total_recv_time_ms_.load() + recv_time_ms);
            parity2_recv_xor_recv_time_ms_.store(parity2_recv_xor_recv_time_ms_.load() + recv_time_ms);
            recv_count_++;

            // Bubble up failures
            if (recv1_exception) {
                std::cerr << recv1_error_msg << std::endl;
                std::rethrow_exception(recv1_exception);
            }
            if (recv2_exception) {
                std::cerr << recv2_error_msg << std::endl;
                std::rethrow_exception(recv2_exception);
            }
            if (!recv1_success) {
                std::cerr << recv1_error_msg << std::endl;
                throw std::runtime_error(recv1_error_msg);
            }
            if (!recv2_success) {
                std::cerr << recv2_error_msg << std::endl;
                throw std::runtime_error(recv2_error_msg);
            }

            // XOR after both recvs succeed → dispatch to 16-worker pool
            auto xor_start = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> pool_lk(xor_pool_work_mutex_);
                xor_pool_run_parallel(task.parity_addr, task.recv1_addr, task.recv2_addr,
                                       static_cast<int>(task.size));
            }
            auto xor_end = std::chrono::high_resolution_clock::now();
            double xor_time_ms = std::chrono::duration<double, std::milli>(xor_end - xor_start).count();
            total_xor_time_ms_.store(total_xor_time_ms_.load() + xor_time_ms);
            parity2_recv_xor_xor_time_ms_.store(parity2_recv_xor_xor_time_ms_.load() + xor_time_ms);
            xor_count_++;

            // Release recv buffers after XOR operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                recv_buffers_to_release_.push(task.recv1_addr);
                recv_buffers_to_release_.push(task.recv2_addr);
            }

            if (parity2_recv_xor_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity2_recv_xor_mutex_);
                if (parity2_recv_xor_q_.empty()) {
                    parity2_recv_xor_completed_ = true;
                    // std::cout << "ECLATIN: Parity2_RecvXor worker completed" << std::endl;
                    parity2_recv_xor_sentinel_received_ = false;
                }
            }
        }
    }

    void parity2_send1_worker() {
        std::cout << "ECLATIN: Parity2_Send1 worker started" << std::endl;
        std::chrono::high_resolution_clock::time_point worker_start_time;
        bool worker_start_time_set = false;
        
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(parity2_send1_mutex_);
                parity2_send1_cv_.wait(lk, [this] { return stop_ || !parity2_send1_q_.empty(); });
                if (stop_) break;
                task = parity2_send1_q_.front();
                parity2_send1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                parity2_send1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity2_send1_mutex_);
                    if (parity2_send1_q_.empty()) {
                        if (worker_start_time_set) {
                            auto worker_end_time = std::chrono::high_resolution_clock::now();
                            double worker_total_time_ms = std::chrono::duration<double, std::milli>(worker_end_time - worker_start_time).count();
                            parity2_send1_total_time_ms_.store(worker_total_time_ms);
                        }
                        parity2_send1_completed_ = true;
                        std::cout << "ECLATIN: Parity2_Send1 worker completed" << std::endl;
                        parity2_send1_sentinel_received_ = false;
                        // Reset for next round
                        worker_start_time_set = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            
            // Record start time when first real task begins
            if (!worker_start_time_set) {
                worker_start_time = std::chrono::high_resolution_clock::now();
                worker_start_time_set = true;
            }
            
            if (conn_.is_parity2_send1_connected()) {
                auto send_start = std::chrono::high_resolution_clock::now();
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_save_channels_[4]) {
                    // std::cout << "[ECLATIN RDMA] Parity2_Send1: Sending " << task.size << " bytes via RDMA" << std::endl;
                    rdma_save_channels_[4]->send_data(reinterpret_cast<const uint8_t*>(task.addr), task.size);
                } else
#endif
                {
                    std::cout << "[ECLATIN ASIO] Parity2_Send1: Sending " << task.size << " bytes via ASIO" << std::endl;
                    send_with_size(conn_.get_parity2_send1_socket(), task.addr, task.size);
                }
                auto send_end = std::chrono::high_resolution_clock::now();
                double send_time_ms = std::chrono::duration<double, std::milli>(send_end - send_start).count();
                total_send_time_ms_.store(total_send_time_ms_.load() + send_time_ms);
                parity2_send1_ops_time_ms_.store(parity2_send1_ops_time_ms_.load() + send_time_ms);
                send_count_++;
            }
            // Release data buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                data_buffers_to_release_.push(task.addr);
            }

            if (parity2_send1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity2_send1_mutex_);
                if (parity2_send1_q_.empty()) {
                    parity2_send1_completed_ = true;
                    std::cout << "ECLATIN: Parity2_Send1 worker completed" << std::endl;
                    parity2_send1_sentinel_received_ = false;
                }
            }
        }
    }

    void parity2_send2_worker() {
        std::cout << "ECLATIN: Parity2_Send2 worker started" << std::endl;
        std::chrono::high_resolution_clock::time_point worker_start_time;
        bool worker_start_time_set = false;
        
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(parity2_send2_mutex_);
                parity2_send2_cv_.wait(lk, [this] { return stop_ || !parity2_send2_q_.empty(); });
                if (stop_) break;
                task = parity2_send2_q_.front();
                parity2_send2_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                parity2_send2_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity2_send2_mutex_);
                    if (parity2_send2_q_.empty()) {
                        if (worker_start_time_set) {
                            auto worker_end_time = std::chrono::high_resolution_clock::now();
                            double worker_total_time_ms = std::chrono::duration<double, std::milli>(worker_end_time - worker_start_time).count();
                            parity2_send2_total_time_ms_.store(worker_total_time_ms);
                        }
                        parity2_send2_completed_ = true;
                        std::cout << "ECLATIN: Parity2_Send2 worker completed" << std::endl;
                        parity2_send2_sentinel_received_ = false;
                        // Reset for next round
                        worker_start_time_set = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            
            // Record start time when first real task begins
            if (!worker_start_time_set) {
                worker_start_time = std::chrono::high_resolution_clock::now();
                worker_start_time_set = true;
            }
            
            if (conn_.is_parity2_send2_connected()) {
                auto send_start = std::chrono::high_resolution_clock::now();
#if RDMA_AVAILABLE
                if (use_rdma_ && rdma_save_channels_[5]) {
                    // std::cout << "[ECLATIN RDMA] Parity2_Send2: Sending " << task.size << " bytes via RDMA" << std::endl;
                    rdma_save_channels_[5]->send_data(reinterpret_cast<const uint8_t*>(task.addr), task.size);
                } else
#endif
                {
                    // std::cout << "[ECLATIN ASIO] Parity2_Send2: Sending " << task.size << " bytes via ASIO" << std::endl;
                    send_with_size(conn_.get_parity2_send2_socket(), task.addr, task.size);
                }
                auto send_end = std::chrono::high_resolution_clock::now();
                double send_time_ms = std::chrono::duration<double, std::milli>(send_end - send_start).count();
                total_send_time_ms_.store(total_send_time_ms_.load() + send_time_ms);
                parity2_send2_ops_time_ms_.store(parity2_send2_ops_time_ms_.load() + send_time_ms);
                send_count_++;
            }
            // Release data buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                data_buffers_to_release_.push(task.addr);
            }

            if (parity2_send2_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity2_send2_mutex_);
                if (parity2_send2_q_.empty()) {
                    parity2_send2_completed_ = true;
                    std::cout << "ECLATIN: Parity2_Send2 worker completed" << std::endl;
                    parity2_send2_sentinel_received_ = false;
                }
            }
        }
    }
    
    // void reset_time_statistics() {
    //     total_encoding_time_ms_ = 0.0;
    //     total_send_time_ms_ = 0.0;
    //     total_recv_time_ms_ = 0.0;
    //     total_xor_time_ms_ = 0.0;
    //     encoding_count_ = 0;
    //     send_count_ = 0;
    //     recv_count_ = 0;
    //     xor_count_ = 0;
        
    //     // Reset per-worker time statistics
    //     parity1_send1_total_time_ms_ = 0.0;
    //     parity1_send2_total_time_ms_ = 0.0;
    //     parity1_recv_xor_total_time_ms_ = 0.0;
    //     parity2_send1_total_time_ms_ = 0.0;
    //     parity2_send2_total_time_ms_ = 0.0;
    //     parity2_recv_xor_total_time_ms_ = 0.0;
        
    //     parity1_send1_ops_time_ms_ = 0.0;
    //     parity1_send2_ops_time_ms_ = 0.0;
    //     parity1_recv_xor_recv_time_ms_ = 0.0;
    //     parity1_recv_xor_xor_time_ms_ = 0.0;
    //     parity2_send1_ops_time_ms_ = 0.0;
    //     parity2_send2_ops_time_ms_ = 0.0;
    //     parity2_recv_xor_recv_time_ms_ = 0.0;
    //     parity2_recv_xor_xor_time_ms_ = 0.0;
        
    //     pipeline_timing_started_ = false;
    //     std::cout << "ECLATIN: Reset time statistics" << std::endl;
    // }
    
    // void print_time_statistics(double pipeline_wall_time_ms = 0.0) {
    //     std::cout << "ECLATIN: Time Statistics:" << std::endl;
        
    //     // Find the bottleneck worker (the slowest one)
    //     struct WorkerTime {
    //         std::string name;
    //         double total_time_ms;
    //         double ops_time_ms;
    //         std::string ops_type;
    //     };
        
    //     std::vector<WorkerTime> worker_times;
    //     worker_times.push_back({"parity1_send1", parity1_send1_total_time_ms_.load(), 
    //                             parity1_send1_ops_time_ms_.load(), "send"});
    //     worker_times.push_back({"parity1_send2", parity1_send2_total_time_ms_.load(), 
    //                             parity1_send2_ops_time_ms_.load(), "send"});
    //     worker_times.push_back({"parity1_recv_xor", parity1_recv_xor_total_time_ms_.load(), 
    //                             parity1_recv_xor_recv_time_ms_.load() + parity1_recv_xor_xor_time_ms_.load(), "recv+xor"});
    //     worker_times.push_back({"parity2_send1", parity2_send1_total_time_ms_.load(), 
    //                             parity2_send1_ops_time_ms_.load(), "send"});
    //     worker_times.push_back({"parity2_send2", parity2_send2_total_time_ms_.load(), 
    //                             parity2_send2_ops_time_ms_.load(), "send"});
    //     worker_times.push_back({"parity2_recv_xor", parity2_recv_xor_total_time_ms_.load(), 
    //                             parity2_recv_xor_recv_time_ms_.load() + parity2_recv_xor_xor_time_ms_.load(), "recv+xor"});
        
    //     WorkerTime* bottleneck = nullptr;
    //     double max_time = 0.0;
    //     for (auto& wt : worker_times) {
    //         if (wt.total_time_ms > max_time) {
    //             max_time = wt.total_time_ms;
    //             bottleneck = &wt;
    //         }
    //     }
        
    //     // Display bottleneck worker information
    //     if (bottleneck && max_time > 0.0) {
    //         std::cout << "  Bottleneck Worker: " << bottleneck->name 
    //                   << " (wall-clock time=" << max_time << " ms, " << (max_time / 1000.0) << " s)" << std::endl;
            
    //         // Calculate task count for this worker
    //         int task_count = 0;
    //         if (bottleneck->name == "parity1_send1" || bottleneck->name == "parity1_send2" ||
    //             bottleneck->name == "parity2_send1" || bottleneck->name == "parity2_send2") {
    //             // For send workers, we can estimate task count from accumulated time vs avg time
    //             // But we don't have per-worker count, so we'll just show the accumulated ops time
    //             std::cout << "    Accumulated Operations (" << bottleneck->ops_type << "): " 
    //                       << bottleneck->ops_time_ms << " ms (sum of all tasks)" << std::endl;
    //             if (bottleneck->ops_time_ms > max_time) {
    //                 std::cout << "    Note: Accumulated time > wall-clock time indicates operations may include overhead" << std::endl;
    //             }
    //         } else if (bottleneck->name.find("recv_xor") != std::string::npos) {
    //             if (bottleneck->name == "parity1_recv_xor") {
    //                 std::cout << "    Accumulated Operations (recv+xor): " << bottleneck->ops_time_ms << " ms (sum of all tasks)" << std::endl;
    //                 std::cout << "      Recv (accumulated): " << parity1_recv_xor_recv_time_ms_.load() << " ms" << std::endl;
    //                 std::cout << "      XOR (accumulated): " << parity1_recv_xor_xor_time_ms_.load() << " ms" << std::endl;
    //             } else {
    //                 std::cout << "    Accumulated Operations (recv+xor): " << bottleneck->ops_time_ms << " ms (sum of all tasks)" << std::endl;
    //                 std::cout << "      Recv (accumulated): " << parity2_recv_xor_recv_time_ms_.load() << " ms" << std::endl;
    //                 std::cout << "      XOR (accumulated): " << parity2_recv_xor_xor_time_ms_.load() << " ms" << std::endl;
    //             }
    //             if (bottleneck->ops_time_ms > max_time) {
    //                 std::cout << "    Note: Accumulated time > wall-clock time indicates operations may include overhead" << std::endl;
    //             }
    //         }
    //     }
        
    //     if (pipeline_wall_time_ms > 0.0) {
    //         std::cout << "  Pipeline Wall-Clock Time: " << pipeline_wall_time_ms << " ms (" 
    //                   << (pipeline_wall_time_ms / 1000.0) << " s)" << std::endl;
    //     }
        
    //     std::cout << "  Send (accumulated): total=" << total_send_time_ms_.load() << " ms, "
    //               << "count=" << send_count_.load() << ", "
    //               << "avg=" << (send_count_.load() > 0 ? total_send_time_ms_.load() / send_count_.load() : 0.0) << " ms" << std::endl;
    //     std::cout << "  Recv (accumulated): total=" << total_recv_time_ms_.load() << " ms, "
    //               << "count=" << recv_count_.load() << ", "
    //               << "avg=" << (recv_count_.load() > 0 ? total_recv_time_ms_.load() / recv_count_.load() : 0.0) << " ms" << std::endl;
    //     std::cout << "  XOR (accumulated): total=" << total_xor_time_ms_.load() << " ms, "
    //               << "count=" << xor_count_.load() << ", "
    //               << "avg=" << (xor_count_.load() > 0 ? total_xor_time_ms_.load() / xor_count_.load() : 0.0) << " ms" << std::endl;
    // }
    
    // std::map<std::string, double> get_time_statistics() {
    //     std::map<std::string, double> stats;
    //     stats["send_total_ms"] = total_send_time_ms_.load();
    //     stats["send_count"] = static_cast<double>(send_count_.load());
    //     stats["send_avg_ms"] = send_count_.load() > 0 ? total_send_time_ms_.load() / send_count_.load() : 0.0;
    //     stats["recv_total_ms"] = total_recv_time_ms_.load();
    //     stats["recv_count"] = static_cast<double>(recv_count_.load());
    //     stats["recv_avg_ms"] = recv_count_.load() > 0 ? total_recv_time_ms_.load() / recv_count_.load() : 0.0;
    //     stats["xor_total_ms"] = total_xor_time_ms_.load();
    //     stats["xor_count"] = static_cast<double>(xor_count_.load());
    //     stats["xor_avg_ms"] = xor_count_.load() > 0 ? total_xor_time_ms_.load() / xor_count_.load() : 0.0;
    //     return stats;
    // }
    
    void layerwise_worker() {
        std::cout << "ECLATIN: LayerWise worker started" << std::endl;
        while (!stop_) {
            LayerWiseTask task;
            {
                std::unique_lock<std::mutex> lk(layerwise_mutex_);
                layerwise_cv_.wait(lk, [this] { return stop_ || !layerwise_queue_.empty(); });
                if (stop_) break;
                task = layerwise_queue_.front();
                layerwise_queue_.pop();
            }
            
            std::cout << "ECLATIN: Processing layer " << task.layer_id 
                      << " (actual_size=" << task.actual_layer_size 
                      << ", aligned_size=" << task.aligned_layer_size << ")" << std::endl;
            
            // Validate base addresses before processing
            if (task.cpu_buffer_addr == 0) {
                std::cerr << "ECLATIN: ERROR: cpu_buffer_addr is 0 for layer " << task.layer_id << std::endl;
                {
                    std::lock_guard<std::mutex> lock(completion_mutex_);
                    layers_completed_++;
                }
                completion_cv_.notify_all();
                continue;
            }
            if (task.data_block_1_addr == 0 || task.data_block_2_addr == 0) {
                std::cerr << "ECLATIN: ERROR: data_block address is 0 for layer " << task.layer_id << std::endl;
                {
                    std::lock_guard<std::mutex> lock(completion_mutex_);
                    layers_completed_++;
                }
                completion_cv_.notify_all();
                continue;
            }
            
            // Step 1: D2H transfer (CUDA mode only - ECLATIN layerwise requires CUDA)
            #ifdef USE_CUDA
            auto d2h_start = std::chrono::high_resolution_clock::now();
            
            if (use_async_cuda_ && !cuda_streams_.empty()) {
                // Async CUDA transfer path
                std::vector<std::vector<const TensorTransferInfo*>> stream_tensors(num_cuda_streams_);
                
                // Distribute tensors across streams (round-robin)
                for (size_t i = 0; i < task.gpu_tensors.size(); ++i) {
                    int stream_idx = i % num_cuda_streams_;
                    stream_tensors[stream_idx].push_back(&task.gpu_tensors[i]);
                }
                
                // Launch async D2H transfers on all streams
                for (int stream_idx = 0; stream_idx < num_cuda_streams_; ++stream_idx) {
                    for (const auto* tensor_info : stream_tensors[stream_idx]) {
                        uintptr_t gpu_ptr = tensor_info->gpu_data_ptr;
                        uintptr_t cpu_ptr = task.cpu_buffer_addr + tensor_info->cpu_offset;
                        size_t size = tensor_info->size_bytes;
                        
                        if (gpu_ptr == 0 || cpu_ptr == 0 || size == 0) {
                            std::cerr << "ECLATIN: ERROR: Invalid tensor info for layer " << task.layer_id 
                                      << " (gpu_ptr=" << gpu_ptr << ", cpu_ptr=" << cpu_ptr 
                                      << ", size=" << size << ")" << std::endl;
                            throw std::runtime_error("Invalid tensor info for D2H transfer");
                        }
                        
                        cudaError_t err = cudaMemcpyAsync(
                            reinterpret_cast<void*>(cpu_ptr), 
                            reinterpret_cast<void*>(gpu_ptr), 
                            size, 
                            cudaMemcpyDeviceToHost,
                            cuda_streams_[stream_idx]
                        );
                        
                        if (err != cudaSuccess) {
                            std::cerr << "ECLATIN: ERROR: cudaMemcpyAsync failed for layer " << task.layer_id 
                                      << " stream " << stream_idx << ": " << cudaGetErrorString(err) << std::endl;
                            throw std::runtime_error(
                                std::string("CUDA memcpy async failed: ") + cudaGetErrorString(err)
                            );
                        }
                    }
                }
                
                // Synchronize all streams before network send
                for (int stream_idx = 0; stream_idx < num_cuda_streams_; ++stream_idx) {
                    cudaError_t err = cudaStreamSynchronize(cuda_streams_[stream_idx]);
                    if (err != cudaSuccess) {
                        std::cerr << "ECLATIN: ERROR: cudaStreamSynchronize failed for layer " << task.layer_id 
                                  << " stream " << stream_idx << ": " << cudaGetErrorString(err) << std::endl;
                        throw std::runtime_error(
                            std::string("CUDA stream synchronize failed: ") + cudaGetErrorString(err)
                        );
                    }
                }
                
                auto d2h_end = std::chrono::high_resolution_clock::now();
                double d2h_time_ms = std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
                std::cout << "ECLATIN: Layer " << task.layer_id << " D2H transfer completed (async, " 
                          << num_cuda_streams_ << " streams, " << task.gpu_tensors.size() 
                          << " tensors, " << d2h_time_ms << " ms)" << std::endl;
            } else {
                // Sync CUDA transfer path (fallback)
                for (const auto& tensor_info : task.gpu_tensors) {
                    uintptr_t gpu_ptr = tensor_info.gpu_data_ptr;
                    uintptr_t cpu_ptr = task.cpu_buffer_addr + tensor_info.cpu_offset;
                    size_t size = tensor_info.size_bytes;
                    
                    if (gpu_ptr == 0 || cpu_ptr == 0 || size == 0) {
                        std::cerr << "ECLATIN: ERROR: Invalid tensor info for layer " << task.layer_id 
                                  << " (gpu_ptr=" << gpu_ptr << ", cpu_ptr=" << cpu_ptr 
                                  << ", size=" << size << ")" << std::endl;
                        throw std::runtime_error("Invalid tensor info for D2H transfer");
                    }
                    
                    cudaError_t err = cudaMemcpy(
                        reinterpret_cast<void*>(cpu_ptr), 
                        reinterpret_cast<void*>(gpu_ptr), 
                        size, 
                        cudaMemcpyDeviceToHost
                    );
                    
                    if (err != cudaSuccess) {
                        std::cerr << "ECLATIN: ERROR: cudaMemcpy failed for layer " << task.layer_id 
                                  << ": " << cudaGetErrorString(err) << std::endl;
                        throw std::runtime_error(
                            std::string("CUDA memcpy failed: ") + cudaGetErrorString(err)
                        );
                    }
                }
                
                cudaError_t sync_err = cudaDeviceSynchronize();
                if (sync_err != cudaSuccess) {
                    std::cerr << "ECLATIN: ERROR: cudaDeviceSynchronize failed for layer " << task.layer_id 
                              << ": " << cudaGetErrorString(sync_err) << std::endl;
                    throw std::runtime_error(
                        std::string("CUDA synchronize failed: ") + cudaGetErrorString(sync_err)
                    );
                }
                
                auto d2h_end = std::chrono::high_resolution_clock::now();
                double d2h_time_ms = std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
                std::cout << "ECLATIN: Layer " << task.layer_id << " D2H transfer completed (sync, " 
                          << task.gpu_tensors.size() << " tensors, " << d2h_time_ms << " ms)" << std::endl;
            }
            #else
            // ECLATIN layerwise requires CUDA - this should not be reached
            throw std::runtime_error(
                "ECLATIN layerwise requires CUDA support. Please compile with USE_CUDA defined."
            );
            #endif
            
            // Step 2: Data splitting - write to data blocks (only actual data, no padding)
            size_t half_actual = task.actual_layer_size / 2;
            uintptr_t cpu_data = task.cpu_buffer_addr;
            
            // Write first half to data_block_1
            if (half_actual > 0) {
                std::memcpy(reinterpret_cast<void*>(task.data_block_1_addr),
                           reinterpret_cast<void*>(cpu_data),
                           half_actual);
            }
            
            // Write second half to data_block_2
            size_t second_half_actual = task.actual_layer_size - half_actual;
            if (second_half_actual > 0) {
                std::memcpy(reinterpret_cast<void*>(task.data_block_2_addr),
                           reinterpret_cast<void*>(cpu_data + half_actual),
                           second_half_actual);
            }
            
            // Step 3: Submit to network pipelines (directly from layer_cpu_buffer, no data buffer copy)
            // All ranks use the same aligned_layer_size, so half_aligned is consistent across ranks
            size_t half_aligned = task.aligned_layer_size / 2;
            
            // Directly send from layer_cpu_buffer (no need to copy to data buffer)
            // First half: send to parity1_send1 and parity2_send1
            submit_parity1_send1(task.cpu_buffer_addr, half_aligned);
            submit_parity2_send1(task.cpu_buffer_addr, half_aligned);
            
            // Second half: send to parity1_send2 and parity2_send2
            submit_parity1_send2(task.cpu_buffer_addr + half_aligned, half_aligned);
            submit_parity2_send2(task.cpu_buffer_addr + half_aligned, half_aligned);
            
            // Submit recv_xor tasks (recv buffers are from continuous buffer pool with offset)
            submit_parity1_recv_xor(task.recv1_parity1_addr, task.recv2_parity1_addr, 
                                   task.parity_block_1_addr, half_aligned);
            submit_parity2_recv_xor(task.recv1_parity2_addr, task.recv2_parity2_addr, 
                                   task.parity_block_2_addr, half_aligned);
            
            // Step 6: Update completion count
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                layers_completed_++;
            }
            completion_cv_.notify_all();
            
            std::cout << "ECLATIN: Layer " << task.layer_id << " processing completed" << std::endl;
        }
        std::cout << "ECLATIN: LayerWise worker stopped" << std::endl;
    }
    
    void layerwise_load_worker() {
        std::cout << "ECLATIN Load: LayerWise load worker started" << std::endl;
        
        while (!stop_) {
            LayerWiseLoadTask task;
            {
                std::unique_lock<std::mutex> lk(layerwise_load_mutex_);
                layerwise_load_cv_.wait(lk, [this] { 
                    return stop_ || !layerwise_load_queue_.empty(); 
                });
                if (stop_) break;
                task = layerwise_load_queue_.front();
                layerwise_load_queue_.pop();
            }
            
            // Record first layer start time for critical path calculation
            bool is_first_layer = !layerwise_first_layer_started_.exchange(true);
            auto layer_start = std::chrono::high_resolution_clock::now();
            if (is_first_layer) {
                layerwise_first_layer_start_ = layer_start;
            }
            
            std::cout << "ECLATIN Load: Processing layer " << task.layer_id 
                      << " (size=" << task.layer_size << ")" << std::endl;
            
            // Three-stage pipeline:
            // Stage 1: Network reception (for rank2 recovery)
            // Stage 2: Recovery computation (XOR operations)
            // Stage 3: H2D transfer (CPU→GPU model initialization)
            
            double recovery_time_ms = 0.0;
            double h2d_time_ms = 0.0;
            
            // Stage 1 & 2: If rank_in_group 2 (receiver) needs recovery, receive then XOR
            if (is_load_mode_ && failed_rank_in_group_ == 2) {
                std::cout << "ECLATIN Load: Performing rank_in_group 2 recovery for layer " << task.layer_id << std::endl;
                
                // Stage 1: Parallel receive 6 blocks for this layer (same as load_recover but per-layer)
                std::vector<std::exception_ptr> recv_exceptions(6);
                std::vector<std::thread> recv_threads;
                size_t recv_size = task.layer_size;
                
                recv_threads.emplace_back([&]() {
                    try {
#if RDMA_AVAILABLE
                        if (use_rdma_ && rdma_load_channels_[0]) {
                            rdma_load_channels_[0]->receive_data(reinterpret_cast<uint8_t*>(task.recv_rank0_data2_addr), recv_size);
                        } else
#endif
                        if (!recv_with_size_bool(conn_.get_load_recv_rank0_data2_socket(),
                                                reinterpret_cast<void*>(task.recv_rank0_data2_addr), recv_size)) {
                            throw std::runtime_error("Failed to receive rank0_data2");
                        }
                    } catch (...) {
                        recv_exceptions[0] = std::current_exception();
                    }
                });
                recv_threads.emplace_back([&]() {
                    try {
#if RDMA_AVAILABLE
                        if (use_rdma_ && rdma_load_channels_[1]) {
                            rdma_load_channels_[1]->receive_data(reinterpret_cast<uint8_t*>(task.recv_rank0_parity2_addr), recv_size);
                        } else
#endif
                        if (!recv_with_size_bool(conn_.get_load_recv_rank0_parity2_socket(),
                                                reinterpret_cast<void*>(task.recv_rank0_parity2_addr), recv_size)) {
                            throw std::runtime_error("Failed to receive rank0_parity2");
                        }
                    } catch (...) {
                        recv_exceptions[1] = std::current_exception();
                    }
                });
                recv_threads.emplace_back([&]() {
                    try {
#if RDMA_AVAILABLE
                        if (use_rdma_ && rdma_load_channels_[2]) {
                            rdma_load_channels_[2]->receive_data(reinterpret_cast<uint8_t*>(task.recv_rank1_data1_addr), recv_size);
                        } else
#endif
                        if (!recv_with_size_bool(conn_.get_load_recv_rank1_data1_socket(),
                                                reinterpret_cast<void*>(task.recv_rank1_data1_addr), recv_size)) {
                            throw std::runtime_error("Failed to receive rank1_data1");
                        }
                    } catch (...) {
                        recv_exceptions[2] = std::current_exception();
                    }
                });
                recv_threads.emplace_back([&]() {
                    try {
#if RDMA_AVAILABLE
                        if (use_rdma_ && rdma_load_channels_[3]) {
                            rdma_load_channels_[3]->receive_data(reinterpret_cast<uint8_t*>(task.recv_rank1_parity1_addr), recv_size);
                        } else
#endif
                        if (!recv_with_size_bool(conn_.get_load_recv_rank1_parity1_socket(),
                                                reinterpret_cast<void*>(task.recv_rank1_parity1_addr), recv_size)) {
                            throw std::runtime_error("Failed to receive rank1_parity1");
                        }
                    } catch (...) {
                        recv_exceptions[3] = std::current_exception();
                    }
                });
                recv_threads.emplace_back([&]() {
                    try {
#if RDMA_AVAILABLE
                        if (use_rdma_ && rdma_load_channels_[4]) {
                            rdma_load_channels_[4]->receive_data(reinterpret_cast<uint8_t*>(task.recv_rank3_data1_addr), recv_size);
                        } else
#endif
                        if (!recv_with_size_bool(conn_.get_load_recv_rank3_data1_socket(),
                                                reinterpret_cast<void*>(task.recv_rank3_data1_addr), recv_size)) {
                            throw std::runtime_error("Failed to receive rank3_data1");
                        }
                    } catch (...) {
                        recv_exceptions[4] = std::current_exception();
                    }
                });
                recv_threads.emplace_back([&]() {
                    try {
#if RDMA_AVAILABLE
                        if (use_rdma_ && rdma_load_channels_[5]) {
                            rdma_load_channels_[5]->receive_data(reinterpret_cast<uint8_t*>(task.recv_rank3_data2_addr), recv_size);
                        } else
#endif
                        if (!recv_with_size_bool(conn_.get_load_recv_rank3_data2_socket(),
                                                reinterpret_cast<void*>(task.recv_rank3_data2_addr), recv_size)) {
                            throw std::runtime_error("Failed to receive rank3_data2");
                        }
                    } catch (...) {
                        recv_exceptions[5] = std::current_exception();
                    }
                });
                
                for (auto& t : recv_threads) {
                    t.join();
                }
                for (size_t i = 0; i < recv_exceptions.size(); ++i) {
                    if (recv_exceptions[i]) {
                        std::rethrow_exception(recv_exceptions[i]);
                    }
                }
                std::cout << "ECLATIN Load: Layer " << task.layer_id << " received 6 blocks" << std::endl;
                
                // Stage 2: XOR recovery (same formula as standard ECLATIN)
                auto recovery_start = std::chrono::high_resolution_clock::now();
                
                // Recovery formula for rank2 (similar to standard ECLATIN recovery):
                // data1 = rank1_data1 XOR rank3_data1
                // data2 = rank0_data2 XOR rank3_data2  
                // parity1 = rank1_parity1 XOR rank3_data1
                // parity2 = rank0_parity2 XOR rank3_data2
                
                size_t half_size = task.layer_size / 2;
                
                // Recover data1: rank0.data2 XOR rank1.parity1 (same as batch mode)
                std::memcpy(reinterpret_cast<void*>(task.recovered_data1_addr), 
                           reinterpret_cast<void*>(task.recv_rank0_data2_addr), half_size);
                void* xor_array_data1[2] = {reinterpret_cast<void*>(task.recovered_data1_addr), 
                                            reinterpret_cast<void*>(task.recv_rank1_parity1_addr)};
                xor_gen(2, static_cast<int>(half_size), xor_array_data1);
                
                // Recover data2: rank0.parity2 XOR rank1.data1 (same as batch mode)
                std::memcpy(reinterpret_cast<void*>(task.recovered_data2_addr), 
                           reinterpret_cast<void*>(task.recv_rank0_parity2_addr), task.layer_size - half_size);
                void* xor_array_data2[2] = {reinterpret_cast<void*>(task.recovered_data2_addr), 
                                            reinterpret_cast<void*>(task.recv_rank1_data1_addr)};
                xor_gen(2, static_cast<int>(task.layer_size - half_size), xor_array_data2);
                
                // Recover parity1: rank1.data1 XOR rank3.data2 (same as batch mode)
                std::memcpy(reinterpret_cast<void*>(task.recovered_parity1_addr), 
                           reinterpret_cast<void*>(task.recv_rank1_data1_addr), task.layer_size - half_size);
                void* xor_array_parity1[2] = {reinterpret_cast<void*>(task.recovered_parity1_addr), 
                                              reinterpret_cast<void*>(task.recv_rank3_data2_addr)};
                xor_gen(2, static_cast<int>(task.layer_size - half_size), xor_array_parity1);
                
                // Recover parity2: rank0.data2 XOR rank3.data1 (same as batch mode)
                std::memcpy(reinterpret_cast<void*>(task.recovered_parity2_addr), 
                           reinterpret_cast<void*>(task.recv_rank0_data2_addr), half_size);
                void* xor_array_parity2[2] = {reinterpret_cast<void*>(task.recovered_parity2_addr), 
                                              reinterpret_cast<void*>(task.recv_rank3_data1_addr)};
                xor_gen(2, static_cast<int>(half_size), xor_array_parity2);
                
                auto recovery_end = std::chrono::high_resolution_clock::now();
                recovery_time_ms = std::chrono::duration<double, std::milli>(recovery_end - recovery_start).count();
                
                // Record recovery time statistics
                {
                    std::lock_guard<std::mutex> lock(layerwise_stats_mutex_);
                    per_layer_recovery_time_ms_[task.layer_id] = recovery_time_ms;
                }
                double old_recovery = total_layerwise_recovery_time_ms_.load();
                total_layerwise_recovery_time_ms_.store(old_recovery + recovery_time_ms);
                
                std::cout << "ECLATIN Load: Layer " << task.layer_id << " recovery completed (" 
                          << recovery_time_ms << " ms)" << std::endl;
            }
            
            // Stage 3: H2D transfer (CPU→GPU) for model initialization
            // For rank2 layerwise: data is split - first half in recovered_data1, second in recovered_data2
            #ifdef USE_CUDA
            if (!task.gpu_tensors.empty()) {
                auto h2d_start = std::chrono::high_resolution_clock::now();
                size_t half_layer = task.layer_size / 2;
                uintptr_t base1 = task.recovered_data1_addr;
                uintptr_t base2 = task.recovered_data2_addr;
                bool use_two_blocks = (is_load_mode_ && failed_rank_in_group_ == 2);
                
                if (use_async_cuda_ && !cuda_streams_.empty()) {
                    // Async CUDA transfer path
                    std::vector<std::vector<const TensorTransferInfo*>> stream_tensors(num_cuda_streams_);
                    
                    // Distribute tensors across streams (round-robin)
                    for (size_t i = 0; i < task.gpu_tensors.size(); ++i) {
                        int stream_idx = i % num_cuda_streams_;
                        stream_tensors[stream_idx].push_back(&task.gpu_tensors[i]);
                    }
                    
                    // Launch async H2D transfers on all streams
                    for (int stream_idx = 0; stream_idx < num_cuda_streams_; ++stream_idx) {
                        for (const auto* tensor_info : stream_tensors[stream_idx]) {
                            uintptr_t gpu_ptr = tensor_info->gpu_data_ptr;
                            size_t cpu_offset = tensor_info->cpu_offset;
                            size_t size = tensor_info->size_bytes;
                            
                            if (gpu_ptr == 0 || size == 0) {
                                std::cerr << "ECLATIN Load: ERROR: Invalid tensor info for layer " << task.layer_id 
                                          << " (gpu_ptr=" << gpu_ptr << ", size=" << size << ")" << std::endl;
                                continue;
                            }
                            
                            if (use_two_blocks) {
                                // Rank2: copy from recovered_data1 (first half) and/or recovered_data2 (second half)
                                if (cpu_offset + size <= half_layer) {
                                    uintptr_t cpu_ptr = base1 + cpu_offset;
                                    cudaError_t err = cudaMemcpyAsync(
                                        reinterpret_cast<void*>(gpu_ptr),
                                        reinterpret_cast<void*>(cpu_ptr),
                                        size, cudaMemcpyHostToDevice, cuda_streams_[stream_idx]);
                                    if (err != cudaSuccess) {
                                        std::cerr << "ECLATIN Load: ERROR: cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err) << std::endl;
                                        throw std::runtime_error("ECLATIN Load: H2D async transfer failed");
                                    }
                                } else if (cpu_offset >= half_layer) {
                                    uintptr_t cpu_ptr = base2 + (cpu_offset - half_layer);
                                    cudaError_t err = cudaMemcpyAsync(
                                        reinterpret_cast<void*>(gpu_ptr),
                                        reinterpret_cast<void*>(cpu_ptr),
                                        size, cudaMemcpyHostToDevice, cuda_streams_[stream_idx]);
                                    if (err != cudaSuccess) {
                                        std::cerr << "ECLATIN Load: ERROR: cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err) << std::endl;
                                        throw std::runtime_error("ECLATIN Load: H2D async transfer failed");
                                    }
                                } else {
                                    size_t first_len = half_layer - cpu_offset;
                                    size_t second_len = size - first_len;
                                    cudaError_t err = cudaMemcpyAsync(
                                        reinterpret_cast<void*>(gpu_ptr),
                                        reinterpret_cast<void*>(base1 + cpu_offset),
                                        first_len, cudaMemcpyHostToDevice, cuda_streams_[stream_idx]);
                                    if (err != cudaSuccess) {
                                        std::cerr << "ECLATIN Load: ERROR: cudaMemcpyAsync H2D (first segment) failed: " << cudaGetErrorString(err) << std::endl;
                                        throw std::runtime_error("ECLATIN Load: H2D async transfer failed");
                                    }
                                    err = cudaMemcpyAsync(
                                        reinterpret_cast<void*>(gpu_ptr + first_len),
                                        reinterpret_cast<void*>(base2),
                                        second_len, cudaMemcpyHostToDevice, cuda_streams_[stream_idx]);
                                    if (err != cudaSuccess) {
                                        std::cerr << "ECLATIN Load: ERROR: cudaMemcpyAsync H2D (second segment) failed: " << cudaGetErrorString(err) << std::endl;
                                        throw std::runtime_error("ECLATIN Load: H2D async transfer failed");
                                    }
                                }
                            } else {
                                uintptr_t cpu_base = task.recv_rank0_data2_addr;
                                uintptr_t cpu_ptr = cpu_base + cpu_offset;
                                cudaError_t err = cudaMemcpyAsync(
                                    reinterpret_cast<void*>(gpu_ptr),
                                    reinterpret_cast<void*>(cpu_ptr),
                                    size, cudaMemcpyHostToDevice, cuda_streams_[stream_idx]);
                                if (err != cudaSuccess) {
                                    std::cerr << "ECLATIN Load: ERROR: cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err) << std::endl;
                                    throw std::runtime_error("ECLATIN Load: H2D async transfer failed");
                                }
                            }
                        }
                    }
                    
                    // Synchronize all streams before moving to next layer
                    for (int stream_idx = 0; stream_idx < num_cuda_streams_; ++stream_idx) {
                        cudaError_t err = cudaStreamSynchronize(cuda_streams_[stream_idx]);
                        if (err != cudaSuccess) {
                            std::cerr << "ECLATIN Load: ERROR: cudaStreamSynchronize failed for layer " 
                                      << task.layer_id << " stream " << stream_idx 
                                      << ": " << cudaGetErrorString(err) << std::endl;
                            throw std::runtime_error("ECLATIN Load: Stream synchronize failed");
                        }
                    }
                    
                    auto h2d_end = std::chrono::high_resolution_clock::now();
                    h2d_time_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
                    
                    // Record H2D time statistics
                    {
                        std::lock_guard<std::mutex> lock(layerwise_stats_mutex_);
                        per_layer_h2d_time_ms_[task.layer_id] = h2d_time_ms;
                    }
                    double old_h2d = total_layerwise_h2d_time_ms_.load();
                    total_layerwise_h2d_time_ms_.store(old_h2d + h2d_time_ms);
                    
                    std::cout << "ECLATIN Load: Layer " << task.layer_id << " H2D transfer completed (async, " 
                              << num_cuda_streams_ << " streams, " << task.gpu_tensors.size() 
                              << " tensors, " << h2d_time_ms << " ms)" << std::endl;
                } else {
                    // Sync CUDA transfer path (fallback)
                    std::cout << "ECLATIN Load: Transferring layer " << task.layer_id 
                              << " from CPU to GPU (" << task.gpu_tensors.size() << " tensors, sync mode)" << std::endl;
                    
                    for (const auto& tensor_info : task.gpu_tensors) {
                        uintptr_t gpu_ptr = tensor_info.gpu_data_ptr;
                        size_t cpu_offset = tensor_info.cpu_offset;
                        size_t size = tensor_info.size_bytes;
                        
                        if (gpu_ptr == 0 || size == 0) {
                            std::cerr << "ECLATIN Load: ERROR: Invalid tensor info for layer " << task.layer_id 
                                      << " (gpu_ptr=" << gpu_ptr << ", size=" << size << ")" << std::endl;
                            continue;
                        }
                        
                        if (use_two_blocks) {
                            if (cpu_offset + size <= half_layer) {
                                uintptr_t cpu_ptr = base1 + cpu_offset;
                                cudaError_t err = cudaMemcpy(
                                    reinterpret_cast<void*>(gpu_ptr), reinterpret_cast<void*>(cpu_ptr),
                                    size, cudaMemcpyHostToDevice);
                                if (err != cudaSuccess) {
                                    std::cerr << "ECLATIN Load: ERROR: cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
                                    throw std::runtime_error("ECLATIN Load: H2D transfer failed");
                                }
                            } else if (cpu_offset >= half_layer) {
                                uintptr_t cpu_ptr = base2 + (cpu_offset - half_layer);
                                cudaError_t err = cudaMemcpy(
                                    reinterpret_cast<void*>(gpu_ptr), reinterpret_cast<void*>(cpu_ptr),
                                    size, cudaMemcpyHostToDevice);
                                if (err != cudaSuccess) {
                                    std::cerr << "ECLATIN Load: ERROR: cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
                                    throw std::runtime_error("ECLATIN Load: H2D transfer failed");
                                }
                            } else {
                                size_t first_len = half_layer - cpu_offset;
                                size_t second_len = size - first_len;
                                cudaError_t err = cudaMemcpy(
                                    reinterpret_cast<void*>(gpu_ptr),
                                    reinterpret_cast<void*>(base1 + cpu_offset),
                                    first_len, cudaMemcpyHostToDevice);
                                if (err != cudaSuccess) {
                                    std::cerr << "ECLATIN Load: ERROR: cudaMemcpy H2D (first segment) failed: " << cudaGetErrorString(err) << std::endl;
                                    throw std::runtime_error("ECLATIN Load: H2D transfer failed");
                                }
                                err = cudaMemcpy(
                                    reinterpret_cast<void*>(gpu_ptr + first_len),
                                    reinterpret_cast<void*>(base2),
                                    second_len, cudaMemcpyHostToDevice);
                                if (err != cudaSuccess) {
                                    std::cerr << "ECLATIN Load: ERROR: cudaMemcpy H2D (second segment) failed: " << cudaGetErrorString(err) << std::endl;
                                    throw std::runtime_error("ECLATIN Load: H2D transfer failed");
                                }
                            }
                        } else {
                            uintptr_t cpu_ptr = task.recv_rank0_data2_addr + cpu_offset;
                            cudaError_t err = cudaMemcpy(
                                reinterpret_cast<void*>(gpu_ptr), reinterpret_cast<void*>(cpu_ptr),
                                size, cudaMemcpyHostToDevice);
                            if (err != cudaSuccess) {
                                std::cerr << "ECLATIN Load: ERROR: cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
                                throw std::runtime_error("ECLATIN Load: H2D transfer failed");
                            }
                        }
                    }
                    
                    // Synchronize to ensure transfer completes before moving to next layer
                    cudaDeviceSynchronize();
                    
                    auto h2d_end = std::chrono::high_resolution_clock::now();
                    h2d_time_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
                    
                    // Record H2D time statistics
                    {
                        std::lock_guard<std::mutex> lock(layerwise_stats_mutex_);
                        per_layer_h2d_time_ms_[task.layer_id] = h2d_time_ms;
                    }
                    double old_h2d = total_layerwise_h2d_time_ms_.load();
                    total_layerwise_h2d_time_ms_.store(old_h2d + h2d_time_ms);
                    
                    std::cout << "ECLATIN Load: Layer " << task.layer_id << " H2D transfer completed (sync, " 
                              << task.gpu_tensors.size() << " tensors, " << h2d_time_ms << " ms)" << std::endl;
                }
            }
            #else
            std::cout << "ECLATIN Load: WARNING: CUDA not available, skipping H2D transfer for layer " 
                      << task.layer_id << std::endl;
            #endif
            
            // Record layer total time and update critical path
            auto layer_end = std::chrono::high_resolution_clock::now();
            double layer_total_time_ms = std::chrono::duration<double, std::milli>(layer_end - layer_start).count();
            
            {
                std::lock_guard<std::mutex> lock(layerwise_stats_mutex_);
                per_layer_total_time_ms_[task.layer_id] = layer_total_time_ms;
                layerwise_last_layer_end_ = layer_end;
            }
            
            // Calculate critical path time (from first layer start to last layer end)
            double critical_path_ms = std::chrono::duration<double, std::milli>(
                layerwise_last_layer_end_ - layerwise_first_layer_start_).count();
            layerwise_critical_path_time_ms_ = critical_path_ms;
            
            // Update completion count
            {
                std::lock_guard<std::mutex> lock(load_completion_mutex_);
                layers_load_completed_++;
            }
            load_completion_cv_.notify_all();
            
            std::cout << "ECLATIN Load: Layer " << task.layer_id << " processing completed "
                      << "(total: " << layer_total_time_ms << " ms)" << std::endl;
        }
        
        std::cout << "ECLATIN Load: LayerWise load worker stopped" << std::endl;
    }
    
    void reset_time_statistics() {
        total_encoding_time_ms_ = 0.0;
        total_send_time_ms_ = 0.0;
        total_recv_time_ms_ = 0.0;
        total_xor_time_ms_ = 0.0;
        encoding_count_ = 0;
        send_count_ = 0;
        recv_count_ = 0;
        xor_count_ = 0;
        
        // Reset per-worker time statistics
        parity1_send1_total_time_ms_ = 0.0;
        parity1_send2_total_time_ms_ = 0.0;
        parity1_recv_xor_total_time_ms_ = 0.0;
        parity2_send1_total_time_ms_ = 0.0;
        parity2_send2_total_time_ms_ = 0.0;
        parity2_recv_xor_total_time_ms_ = 0.0;
        
        parity1_send1_ops_time_ms_ = 0.0;
        parity1_send2_ops_time_ms_ = 0.0;
        parity1_recv_xor_recv_time_ms_ = 0.0;
        parity1_recv_xor_xor_time_ms_ = 0.0;
        parity2_send1_ops_time_ms_ = 0.0;
        parity2_send2_ops_time_ms_ = 0.0;
        parity2_recv_xor_recv_time_ms_ = 0.0;
        parity2_recv_xor_xor_time_ms_ = 0.0;
        
        pipeline_timing_started_ = false;
        std::cout << "ECLATIN: Reset time statistics" << std::endl;
    }
    
    void print_time_statistics(double pipeline_wall_time_ms = 0.0) {
        std::cout << "ECLATIN: Time Statistics:" << std::endl;
        
        // Find the bottleneck worker (the slowest one)
        struct WorkerTime {
            std::string name;
            double total_time_ms;
            double ops_time_ms;
            std::string ops_type;
        };
        
        std::vector<WorkerTime> worker_times;
        worker_times.push_back({"parity1_send1", parity1_send1_total_time_ms_.load(), 
                                parity1_send1_ops_time_ms_.load(), "send"});
        worker_times.push_back({"parity1_send2", parity1_send2_total_time_ms_.load(), 
                                parity1_send2_ops_time_ms_.load(), "send"});
        worker_times.push_back({"parity1_recv_xor", parity1_recv_xor_total_time_ms_.load(), 
                                parity1_recv_xor_recv_time_ms_.load() + parity1_recv_xor_xor_time_ms_.load(), "recv+xor"});
        worker_times.push_back({"parity2_send1", parity2_send1_total_time_ms_.load(), 
                                parity2_send1_ops_time_ms_.load(), "send"});
        worker_times.push_back({"parity2_send2", parity2_send2_total_time_ms_.load(), 
                                parity2_send2_ops_time_ms_.load(), "send"});
        worker_times.push_back({"parity2_recv_xor", parity2_recv_xor_total_time_ms_.load(), 
                                parity2_recv_xor_recv_time_ms_.load() + parity2_recv_xor_xor_time_ms_.load(), "recv+xor"});
        
        WorkerTime* bottleneck = nullptr;
        double max_time = 0.0;
        for (auto& wt : worker_times) {
            if (wt.total_time_ms > max_time) {
                max_time = wt.total_time_ms;
                bottleneck = &wt;
            }
        }
        
        // Display bottleneck worker information
        if (bottleneck && max_time > 0.0) {
            std::cout << "  Bottleneck Worker: " << bottleneck->name 
                      << " (wall-clock time=" << max_time << " ms, " << (max_time / 1000.0) << " s)" << std::endl;
            
            // Calculate task count for this worker
            int task_count = 0;
            if (bottleneck->name == "parity1_send1" || bottleneck->name == "parity1_send2" ||
                bottleneck->name == "parity2_send1" || bottleneck->name == "parity2_send2") {
                // For send workers, we can estimate task count from accumulated time vs avg time
                // But we don't have per-worker count, so we'll just show the accumulated ops time
                std::cout << "    Accumulated Operations (" << bottleneck->ops_type << "): " 
                          << bottleneck->ops_time_ms << " ms (sum of all tasks)" << std::endl;
                if (bottleneck->ops_time_ms > max_time) {
                    std::cout << "    Note: Accumulated time > wall-clock time indicates operations may include overhead" << std::endl;
                }
            } else if (bottleneck->name.find("recv_xor") != std::string::npos) {
                if (bottleneck->name == "parity1_recv_xor") {
                    std::cout << "    Accumulated Operations (recv+xor): " << bottleneck->ops_time_ms << " ms (sum of all tasks)" << std::endl;
                    std::cout << "      Recv (accumulated): " << parity1_recv_xor_recv_time_ms_.load() << " ms" << std::endl;
                    std::cout << "      XOR (accumulated): " << parity1_recv_xor_xor_time_ms_.load() << " ms" << std::endl;
                } else {
                    std::cout << "    Accumulated Operations (recv+xor): " << bottleneck->ops_time_ms << " ms (sum of all tasks)" << std::endl;
                    std::cout << "      Recv (accumulated): " << parity2_recv_xor_recv_time_ms_.load() << " ms" << std::endl;
                    std::cout << "      XOR (accumulated): " << parity2_recv_xor_xor_time_ms_.load() << " ms" << std::endl;
                }
                if (bottleneck->ops_time_ms > max_time) {
                    std::cout << "    Note: Accumulated time > wall-clock time indicates operations may include overhead" << std::endl;
                }
            }
        }
        
        if (pipeline_wall_time_ms > 0.0) {
            std::cout << "  Pipeline Wall-Clock Time: " << pipeline_wall_time_ms << " ms (" 
                      << (pipeline_wall_time_ms / 1000.0) << " s)" << std::endl;
        }
        
        std::cout << "  Send (accumulated): total=" << total_send_time_ms_.load() << " ms, "
                  << "count=" << send_count_.load() << ", "
                  << "avg=" << (send_count_.load() > 0 ? total_send_time_ms_.load() / send_count_.load() : 0.0) << " ms" << std::endl;
        std::cout << "  Recv (accumulated): total=" << total_recv_time_ms_.load() << " ms, "
                  << "count=" << recv_count_.load() << ", "
                  << "avg=" << (recv_count_.load() > 0 ? total_recv_time_ms_.load() / recv_count_.load() : 0.0) << " ms" << std::endl;
        std::cout << "  XOR (accumulated): total=" << total_xor_time_ms_.load() << " ms, "
                  << "count=" << xor_count_.load() << ", "
                  << "avg=" << (xor_count_.load() > 0 ? total_xor_time_ms_.load() / xor_count_.load() : 0.0) << " ms" << std::endl;
    }
    
    std::map<std::string, double> get_time_statistics() {
        std::map<std::string, double> stats;
        stats["send_total_ms"] = total_send_time_ms_.load();
        stats["send_count"] = static_cast<double>(send_count_.load());
        stats["send_avg_ms"] = send_count_.load() > 0 ? total_send_time_ms_.load() / send_count_.load() : 0.0;
        stats["recv_total_ms"] = total_recv_time_ms_.load();
        stats["recv_count"] = static_cast<double>(recv_count_.load());
        stats["recv_avg_ms"] = recv_count_.load() > 0 ? total_recv_time_ms_.load() / recv_count_.load() : 0.0;
        stats["xor_total_ms"] = total_xor_time_ms_.load();
        stats["xor_count"] = static_cast<double>(xor_count_.load());
        stats["xor_avg_ms"] = xor_count_.load() > 0 ? total_xor_time_ms_.load() / xor_count_.load() : 0.0;
        return stats;
    }
};

}  // namespace

PYBIND11_MODULE(eclatin_native, m) {
    m.doc() = "ECLATIN Native C++ Module for erasure coding with ASIO or RDMA";
    
    // Static utility functions
    m.def("is_rdma_available", []() {
#if RDMA_AVAILABLE
        // Check for RDMA devices using ibverbs
        int num_devices;
        ibv_device** device_list = ibv_get_device_list(&num_devices);
        if (!device_list || num_devices == 0) {
            if (device_list) {
                ibv_free_device_list(device_list);
            }
            return false;
        }
        ibv_free_device_list(device_list);
        return true;
#else
        // RDMA libraries not available at compile time
        return false;
#endif
    }, "Check if RDMA is available on the system");
    
    pybind11::class_<ECLATINNative>(m, "ECLATINNative")
        .def(pybind11::init<const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            int,
                            bool,
                            int,
                            int,
                            int>(),
             pybind11::arg("parity1_send1_ip"),
             pybind11::arg("parity1_send1_port"),
             pybind11::arg("parity1_send2_ip"),
             pybind11::arg("parity1_send2_port"),
             pybind11::arg("parity1_recv1_ip"),
             pybind11::arg("parity1_recv1_port"),
             pybind11::arg("parity1_recv2_ip"),
             pybind11::arg("parity1_recv2_port"),
             pybind11::arg("parity2_send1_ip"),
             pybind11::arg("parity2_send1_port"),
             pybind11::arg("parity2_send2_ip"),
             pybind11::arg("parity2_send2_port"),
             pybind11::arg("parity2_recv1_ip"),
             pybind11::arg("parity2_recv1_port"),
             pybind11::arg("parity2_recv2_ip"),
             pybind11::arg("parity2_recv2_port"),
             pybind11::arg("num_cuda_streams") = ECLATIN_NUM_CUDA_STREAMS,
             pybind11::arg("use_rdma") = false,
             pybind11::arg("rank") = -1,
             pybind11::arg("world_size") = -1,
             pybind11::arg("rank_in_group") = -1,
             "Initialize ECLATIN native module with ASIO or RDMA transport.\n\n"
             "Args:\n"
             "    parity1_send1_ip, parity1_send1_port: Parity 1 send1 connection\n"
             "    parity1_send2_ip, parity1_send2_port: Parity 1 send2 connection\n"
             "    parity1_recv1_ip, parity1_recv1_port: Parity 1 recv1 connection\n"
             "    parity1_recv2_ip, parity1_recv2_port: Parity 1 recv2 connection\n"
             "    parity2_send1_ip, parity2_send1_port: Parity 2 send1 connection\n"
             "    parity2_send2_ip, parity2_send2_port: Parity 2 send2 connection\n"
             "    parity2_recv1_ip, parity2_recv1_port: Parity 2 recv1 connection\n"
             "    parity2_recv2_ip, parity2_recv2_port: Parity 2 recv2 connection\n"
             "    num_cuda_streams: Number of CUDA streams for async transfers\n"
             "    use_rdma: Use RDMA transport (default: False, uses ASIO)\n"
             "    rank: Global rank (for multi-rank support)\n"
             "    world_size: World size (for multi-rank support)\n"
             "    rank_in_group: Rank within 4-rank group (0..3)\n")
        // RDMA buffer registration (no-op for ASIO mode)
        .def("register_buffer", &ECLATINNative::register_buffer,
             pybind11::arg("buffer_addr"),
             pybind11::arg("buffer_size"),
             "Register buffer for RDMA operations (no-op for ASIO).\n\n"
             "For RDMA: Register buffer during first allocation in save phase.\n"
             "Args:\n"
             "    buffer_addr: Memory address of the buffer (uintptr_t)\n"
             "    buffer_size: Size of the buffer in bytes\n")
        .def("unregister_buffer", &ECLATINNative::unregister_buffer,
             pybind11::arg("buffer_addr"),
             "Unregister buffer (no-op for ASIO).\n\n"
             "Args:\n"
             "    buffer_addr: Memory address of the buffer (uintptr_t)\n")
        // Parity 1 submit functions
        .def("submit_parity1_send1", &ECLATINNative::submit_parity1_send1,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_parity1_send2", &ECLATINNative::submit_parity1_send2,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_parity1_recv_xor", &ECLATINNative::submit_parity1_recv_xor,
             pybind11::arg("recv1_addr"),
             pybind11::arg("recv2_addr"),
             pybind11::arg("parity_addr"),
             pybind11::arg("size"))
        // Parity 2 submit functions
        .def("submit_parity2_send1", &ECLATINNative::submit_parity2_send1,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_parity2_send2", &ECLATINNative::submit_parity2_send2,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_parity2_recv_xor", &ECLATINNative::submit_parity2_recv_xor,
             pybind11::arg("recv1_addr"),
             pybind11::arg("recv2_addr"),
             pybind11::arg("parity_addr"),
             pybind11::arg("size"))
        // Common functions
        .def("get_data_buffers_to_release", &ECLATINNative::get_data_buffers_to_release)
        .def("get_recv_buffers_to_release", &ECLATINNative::get_recv_buffers_to_release)
        .def("reset_encoding_completion_flags", &ECLATINNative::reset_encoding_completion_flags)
        .def("wait_for_encoding_completion", &ECLATINNative::wait_for_encoding_completion)
        // Layer-wise processing functions
        .def("submit_layer_wise", &ECLATINNative::submit_layer_wise,
             "Submit a layer for layer-wise processing",
             pybind11::arg("layer_id"),
             pybind11::arg("gpu_tensors_info"),
             pybind11::arg("cpu_buffer_addr"),
             pybind11::arg("aligned_layer_size"),
             pybind11::arg("actual_layer_size"),
             pybind11::arg("data_block_1_base"),
             pybind11::arg("data_block_2_base"),
             pybind11::arg("parity_block_1_base"),
             pybind11::arg("parity_block_2_base"),
             pybind11::arg("data_block_1_offset"),
             pybind11::arg("data_block_2_offset"),
             pybind11::arg("parity_block_1_offset"),
             pybind11::arg("parity_block_2_offset"),
             pybind11::arg("recv1_parity1_addr"),
             pybind11::arg("recv2_parity1_addr"),
             pybind11::arg("recv1_parity2_addr"),
             pybind11::arg("recv2_parity2_addr"))
        .def("wait_all_layers_complete", &ECLATINNative::wait_all_layers_complete,
             "Wait for all layer-wise tasks to complete")
        // Layer-wise load processing functions
        .def("submit_layer_wise_load", &ECLATINNative::submit_layer_wise_load,
             "Submit a layer for layer-wise load processing (recovery + H2D)",
             pybind11::arg("layer_id"),
             pybind11::arg("gpu_tensors_info"),
             pybind11::arg("recv_rank0_data2_addr"),
             pybind11::arg("recv_rank0_parity2_addr"),
             pybind11::arg("recv_rank1_data1_addr"),
             pybind11::arg("recv_rank1_parity1_addr"),
             pybind11::arg("recv_rank3_data1_addr"),
             pybind11::arg("recv_rank3_data2_addr"),
             pybind11::arg("recovered_data1_addr"),
             pybind11::arg("recovered_data2_addr"),
             pybind11::arg("recovered_parity1_addr"),
             pybind11::arg("recovered_parity2_addr"),
             pybind11::arg("layer_size"))
        .def("wait_all_load_layers_complete", &ECLATINNative::wait_all_load_layers_complete,
             "Wait for all layer-wise load tasks to complete")
        .def("reset_layerwise_load_statistics", &ECLATINNative::reset_layerwise_load_statistics,
             "Reset layerwise load time statistics")
        .def("get_layerwise_load_statistics", &ECLATINNative::get_layerwise_load_statistics,
             "Get layerwise load time statistics as a dictionary")
        .def("print_layerwise_load_statistics", &ECLATINNative::print_layerwise_load_statistics,
             "Print layerwise load time statistics")
        // Parity 1 sentinels
        .def("submit_parity1_send1_sentinel", &ECLATINNative::submit_parity1_send1_sentinel)
        .def("submit_parity1_send2_sentinel", &ECLATINNative::submit_parity1_send2_sentinel)
        .def("submit_parity1_recv_xor_sentinel", &ECLATINNative::submit_parity1_recv_xor_sentinel)
        // Parity 2 sentinels
        .def("submit_parity2_send1_sentinel", &ECLATINNative::submit_parity2_send1_sentinel)
        .def("submit_parity2_send2_sentinel", &ECLATINNative::submit_parity2_send2_sentinel)
        .def("submit_parity2_recv_xor_sentinel", &ECLATINNative::submit_parity2_recv_xor_sentinel)
        // Load mode functions
        .def("set_load_mode", &ECLATINNative::set_load_mode,
             "Set load mode for recovery",
             pybind11::arg("is_load"),
             pybind11::arg("failed_rank") = -1)
        .def("set_no_shared_block", &ECLATINNative::set_no_shared_block,
             "Enable HW1 no-shared-block recovery breakdown mode",
             pybind11::arg("enabled"))
        .def("init_load_connections", &ECLATINNative::init_load_connections,
             "Initialize load mode connections (rank_in_group 2 recv, 0/1/3 send)",
             pybind11::arg("rank_in_group"),
             pybind11::arg("rank2_ip"),
             pybind11::arg("load_recv_rank0_data2_port"),
             pybind11::arg("load_recv_rank0_parity2_port"),
             pybind11::arg("load_recv_rank1_data1_port"),
             pybind11::arg("load_recv_rank1_parity1_port"),
             pybind11::arg("load_recv_rank3_data1_port"),
             pybind11::arg("load_recv_rank3_data2_port"))
        .def("init_load_connections_noshared", &ECLATINNative::init_load_connections_noshared,
             "Initialize no-shared-block HW1 load connections (8 channels)",
             pybind11::arg("rank_in_group"),
             pybind11::arg("rank2_ip"),
             pybind11::arg("port_n1_d1"),
             pybind11::arg("port_n3_p1"),
             pybind11::arg("port_n0_d0"),
             pybind11::arg("port_n3_p0"),
             pybind11::arg("port_n1_d0"),
             pybind11::arg("port_n3_d1"),
             pybind11::arg("port_n0_d1"),
             pybind11::arg("port_n3_d0"))
        .def("wait_for_load_connections", &ECLATINNative::wait_for_load_connections,
             "Wait for load mode connections to be established",
             pybind11::arg("timeout_seconds") = 30)
        .def("load_recover", &ECLATINNative::load_recover,
             "Unified recovery interface for rank0 (parallel recv + parallel XOR)",
             pybind11::arg("rank1_data1_addr"),
             pybind11::arg("rank1_data2_addr"),
             pybind11::arg("rank2_data2_addr"),
             pybind11::arg("rank2_parity2_addr"),
             pybind11::arg("rank3_data1_addr"),
             pybind11::arg("rank3_parity1_addr"),
             pybind11::arg("recovered_data1_addr"),
             pybind11::arg("recovered_data2_addr"),
             pybind11::arg("recovered_parity1_addr"),
             pybind11::arg("recovered_parity2_addr"),
             pybind11::arg("size"))
        .def("load_send_blocks", &ECLATINNative::load_send_blocks,
             "Send two blocks to rank2 in parallel (for rank0, rank1, rank3)",
             pybind11::arg("block1_name"),
             pybind11::arg("block1_addr"),
             pybind11::arg("block2_name"),
             pybind11::arg("block2_addr"),
             pybind11::arg("size"))
        // Two-failures load functions
        .def("init_two_failures_load_connections", &ECLATINNative::init_two_failures_load_connections,
             "Initialize two-failures load connections (rank0/1 recv, rank2/3 send)",
             pybind11::arg("rank_in_group"),
             pybind11::arg("peer0_ip"),
             pybind11::arg("peer0_port"),
             pybind11::arg("peer1_ip"),
             pybind11::arg("peer1_port"))
        .def("load_recover_two_failures", &ECLATINNative::load_recover_two_failures,
             "Two-failures recovery: recv 8 blocks + XOR pool recovery",
             pybind11::arg("r2_d2"), pybind11::arg("r2_D2"),
             pybind11::arg("r2_p2"), pybind11::arg("r2_P2"),
             pybind11::arg("r3_d3"), pybind11::arg("r3_D3"),
             pybind11::arg("r3_p3"), pybind11::arg("r3_P3"),
             pybind11::arg("recovered_data1"),
             pybind11::arg("recovered_data2"),
             pybind11::arg("rank_in_group"),
             pybind11::arg("size"))
        .def("load_send_all_blocks_two_fail", &ECLATINNative::load_send_all_blocks_two_fail,
             "Send 4 blocks to a failed rank (for surviving ranks in two-failures)",
             pybind11::arg("target_rank_in_group"),
             pybind11::arg("data1_addr"),
             pybind11::arg("data2_addr"),
             pybind11::arg("parity1_addr"),
             pybind11::arg("parity2_addr"),
             pybind11::arg("size"))
        // Two-failures v2: new 3-step protocol (2-phase connection)
        .def("init_twofail_bind_phase", &ECLATINNative::init_twofail_bind_phase,
             "Phase 0a: bind+listen only (synchronous, no blocking)",
             pybind11::arg("rank_in_group"),
             pybind11::arg("surv_exch_ip"), pybind11::arg("surv_exch_port"),
             pybind11::arg("n1_ip"), pybind11::arg("n1_n3_port"),
             pybind11::arg("n2_ip"), pybind11::arg("n2_n3_port"),
             pybind11::arg("n1_ip2"), pybind11::arg("n1_n4_port"),
             pybind11::arg("n2_ip2"), pybind11::arg("n2_n4_port"))
        .def("init_twofail_connect_phase", &ECLATINNative::init_twofail_connect_phase,
             "Phase 0b: connect + accept (after barrier, all listeners ready)",
             pybind11::arg("rank_in_group"),
             pybind11::arg("surv_exch_ip"), pybind11::arg("surv_exch_port"),
             pybind11::arg("n1_ip"), pybind11::arg("n1_n3_port"),
             pybind11::arg("n2_ip"), pybind11::arg("n2_n3_port"),
             pybind11::arg("n1_ip2"), pybind11::arg("n1_n4_port"),
             pybind11::arg("n2_ip2"), pybind11::arg("n2_n4_port"))
        .def("survivor_exchange_data", &ECLATINNative::survivor_exchange_data,
             "Step 1: Survivor↔Survivor data block exchange",
             pybind11::arg("rank_in_group"),
             pybind11::arg("send_d1"), pybind11::arg("send_d2"),
             pybind11::arg("recv_d1"), pybind11::arg("recv_d2"),
             pybind11::arg("size"))
        .def("survivor_xor_decode", &ECLATINNative::survivor_xor_decode,
             "Step 2: XOR decode on survivor (2-pass, 2-input, 16-thread pool)",
             pybind11::arg("rank_in_group"),
             pybind11::arg("out1"), pybind11::arg("out2"),
             pybind11::arg("out3"), pybind11::arg("out4"),
             pybind11::arg("own_d1"), pybind11::arg("own_d2"),
             pybind11::arg("own_p1"), pybind11::arg("own_p2"),
             pybind11::arg("peer_d1"), pybind11::arg("peer_d2"),
             pybind11::arg("size"))
        .def("send_two_blocks", &ECLATINNative::send_two_blocks,
             "Step 3: Survivor sends 2 blocks to a failed rank",
             pybind11::arg("rank_in_group"),
             pybind11::arg("target_rig"),
             pybind11::arg("addr1"), pybind11::arg("addr2"),
             pybind11::arg("size"))
        .def("send_to_both_failed_ranks", &ECLATINNative::send_to_both_failed_ranks,
             "Step 3: Survivor sends 2 blocks each to rig0 and rig1 in parallel",
             pybind11::arg("rank_in_group"),
             pybind11::arg("rig0_addr1"), pybind11::arg("rig0_addr2"),
             pybind11::arg("rig1_addr1"), pybind11::arg("rig1_addr2"),
             pybind11::arg("size"))
        .def("recv_four_blocks", &ECLATINNative::recv_four_blocks,
             "Step 3: Failed node receives 4 blocks from 2 survivors (parallel)",
             pybind11::arg("rank_in_group"),
             pybind11::arg("addr1"), pybind11::arg("addr2"),
             pybind11::arg("addr3"), pybind11::arg("addr4"),
             pybind11::arg("size"))
        .def("reset_twofail_pipeline", &ECLATINNative::reset_twofail_pipeline,
             "Reset two-fail v2 chunked recovery pipeline state")
        .def("submit_twofail_survivor_chunk", &ECLATINNative::submit_twofail_survivor_chunk,
             "Submit one survivor recovery chunk to the pipeline",
             pybind11::arg("rank_in_group"),
             pybind11::arg("own_d1"), pybind11::arg("own_d2"),
             pybind11::arg("own_p1"), pybind11::arg("own_p2"),
             pybind11::arg("peer_d1"), pybind11::arg("peer_d2"),
             pybind11::arg("out1"), pybind11::arg("out2"),
             pybind11::arg("out3"), pybind11::arg("out4"),
             pybind11::arg("size"), pybind11::arg("chunk_index"),
             pybind11::arg("rel0"), pybind11::arg("rel1"), pybind11::arg("rel2"),
             pybind11::arg("rel3"), pybind11::arg("rel4"), pybind11::arg("rel5"))
        .def("submit_twofail_failed_chunk", &ECLATINNative::submit_twofail_failed_chunk,
             "Submit one failed-rank recv chunk to the pipeline",
             pybind11::arg("rank_in_group"),
             pybind11::arg("a1"), pybind11::arg("a2"),
             pybind11::arg("a3"), pybind11::arg("a4"),
             pybind11::arg("size"), pybind11::arg("chunk_index"))
        .def("submit_twofail_pipeline_sentinels", &ECLATINNative::submit_twofail_pipeline_sentinels,
             "Submit pipeline sentinels to drain all twofail workers")
        .def("wait_for_twofail_pipeline_completion", &ECLATINNative::wait_for_twofail_pipeline_completion,
             "Block until all twofail pipeline workers finish")
        .def("get_twofail_buffers_to_release", &ECLATINNative::get_twofail_buffers_to_release,
             "Poll pool buffer addresses ready for release after twofail pipeline")
        .def("reset_onefail_pipeline", &ECLATINNative::reset_onefail_pipeline,
             "Reset one-fail chunked load pipeline state")
        .def("submit_onefail_send_chunk", &ECLATINNative::submit_onefail_send_chunk,
             "Submit one sender load chunk to the one-fail pipeline",
             pybind11::arg("rank_in_group"),
             pybind11::arg("block1_name"), pybind11::arg("block1_addr"),
             pybind11::arg("block2_name"), pybind11::arg("block2_addr"),
             pybind11::arg("size"), pybind11::arg("chunk_index"))
        .def("submit_onefail_recv_chunk", &ECLATINNative::submit_onefail_recv_chunk,
             "Submit one rig2 recv+xor load chunk to the one-fail pipeline",
             pybind11::arg("r0d2"), pybind11::arg("r0p2"),
             pybind11::arg("r1d1"), pybind11::arg("r1p1"),
             pybind11::arg("r3d1"), pybind11::arg("r3d2"),
             pybind11::arg("out_d1"), pybind11::arg("out_d2"),
             pybind11::arg("out_p1"), pybind11::arg("out_p2"),
             pybind11::arg("size"), pybind11::arg("chunk_index"),
             pybind11::arg("rel0"), pybind11::arg("rel1"), pybind11::arg("rel2"),
             pybind11::arg("rel3"), pybind11::arg("rel4"), pybind11::arg("rel5"))
        .def("submit_onefail_noshared_recv_chunk", &ECLATINNative::submit_onefail_noshared_recv_chunk,
             "Submit one rig2 no-shared recv+xor load chunk (8 independent blocks)",
             pybind11::arg("recv_n1_d1"), pybind11::arg("recv_n3_p1"),
             pybind11::arg("recv_n0_d0"), pybind11::arg("recv_n3_p0"),
             pybind11::arg("recv_n1_d0"), pybind11::arg("recv_n3_d1"),
             pybind11::arg("recv_n0_d1"), pybind11::arg("recv_n3_d0"),
             pybind11::arg("out_d0"), pybind11::arg("out_d1"),
             pybind11::arg("out_p0"), pybind11::arg("out_p1"),
             pybind11::arg("size"), pybind11::arg("chunk_index"),
             pybind11::arg("rel0"), pybind11::arg("rel1"), pybind11::arg("rel2"),
             pybind11::arg("rel3"), pybind11::arg("rel4"), pybind11::arg("rel5"),
             pybind11::arg("rel6"), pybind11::arg("rel7"))
        .def("submit_onefail_pipeline_sentinels", &ECLATINNative::submit_onefail_pipeline_sentinels,
             "Submit pipeline sentinels to drain one-fail load workers")
        .def("wait_for_onefail_pipeline_completion", &ECLATINNative::wait_for_onefail_pipeline_completion,
             "Block until all one-fail load pipeline workers finish")
        .def("get_onefail_buffers_to_release", &ECLATINNative::get_onefail_buffers_to_release,
             "Poll recv pool buffer addresses ready for release after one-fail pipeline")
        .def("stop", &ECLATINNative::stop);
}


