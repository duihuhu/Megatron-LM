#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/asio.hpp>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>

#include <array>
#include <cctype>
#include <chrono>

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

#include <cstdlib>

#include <atomic>
#include <cerrno>
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
#include <map>
#include <memory>

#include <isa-l/erasure_code.h>
#include <isa-l/raid.h>

// RDMA headers
#include <infiniband/verbs.h>

#include "rdma_device_utils.h"


namespace {

#ifndef ECNAIVE_RANKS_PER_GROUP
#define ECNAIVE_RANKS_PER_GROUP 4  // Ranks per EC-NAIVE group (multi-rank support)
#endif

// RDMA structures (similar to Gemini)
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

// Connection interface for abstraction
class IConnectionChannel {
public:
    virtual ~IConnectionChannel() = default;
    virtual void send_data(const uint8_t* data, size_t size) = 0;
    virtual size_t receive_data(uint8_t* buffer, size_t buffer_size) = 0;
    virtual bool is_connected() const = 0;
    // RDMA-only: exchange QP info and connect; no-op for ASIO channels
    virtual void exchange_and_connect(bool we_send_first) { (void)we_send_first; }
};

// Forward declarations
class AsioConnectionChannel;
class RdmaConnectionChannel;

// ASIO Connection Channel Implementation
class AsioConnectionChannel : public IConnectionChannel {
private:
    boost::asio::ip::tcp::socket& socket_;
    std::mutex send_mutex_;
    std::mutex recv_mutex_;

public:
    explicit AsioConnectionChannel(boost::asio::ip::tcp::socket& socket)
        : socket_(socket) {}

    void send_data(const uint8_t* data, size_t size) override {
        std::lock_guard<std::mutex> lock(send_mutex_);
        // Use uint64_t to support data transfers > 4GB
        uint64_t sz_net = htonll(static_cast<uint64_t>(size));
        boost::asio::write(socket_, boost::asio::buffer(&sz_net, sizeof(uint64_t)));
        boost::asio::write(socket_, boost::asio::buffer(data, size));
    }

    size_t receive_data(uint8_t* buffer, size_t buffer_size) override {
        std::lock_guard<std::mutex> lock(recv_mutex_);
        // Use uint64_t to support data transfers > 4GB
        uint64_t sz_net;
        boost::asio::read(socket_, boost::asio::buffer(&sz_net, sizeof(uint64_t)));
        uint64_t size = ntohll(sz_net);
        if (size > static_cast<uint64_t>(buffer_size)) {
            throw std::runtime_error("Received size exceeds buffer size");
        }
        boost::asio::read(socket_, boost::asio::buffer(buffer, size));
        return static_cast<size_t>(size);
    }

    bool is_connected() const override {
        return socket_.is_open();
    }
};

// RDMA Connection Channel Implementation
class RdmaConnectionChannel : public IConnectionChannel {
private:
    // RDMA resources (shared across channels)
    ibv_context* context_;
    ibv_pd* pd_;
    ibv_cq* send_cq_;
    ibv_cq* recv_cq_;
    ibv_qp* qp_;
    
    // TCP control sockets for coordination
    int control_sock_send_;  // For sending size notifications
    int control_sock_recv_;  // For receiving size notifications
    
    // Buffer registry (shared)
    std::map<uintptr_t, RdmaBuffer>* registered_buffers_;
    std::mutex* buffer_mutex_;
    
    // Temporary buffers for unregistered data
    std::vector<uint8_t> temp_send_buffer_;
    std::vector<uint8_t> temp_recv_buffer_;
    ibv_mr* temp_send_mr_;
    ibv_mr* temp_recv_mr_;
    
    int rank_;
    int peer_rank_;
    bool connected_;
    std::mutex send_mutex_;
    std::mutex recv_mutex_;
    
    static const size_t TEMP_BUFFER_SIZE = 128ULL * 1024 * 1024;  // 128 MB
    static const size_t CHUNK_SIZE = 64 * 1024 * 1024;  // 64 MB per RDMA operation
    static const int MAX_WR = 64;
    static const int MAX_BATCH_WR = 32;

public:
    RdmaConnectionChannel(
        ibv_context* context,
        ibv_pd* pd,
        ibv_cq* send_cq,
        ibv_cq* recv_cq,
        int control_sock_send,
        int control_sock_recv,
        std::map<uintptr_t, RdmaBuffer>* registered_buffers,
        std::mutex* buffer_mutex,
        int rank,
        int peer_rank
    )
        : context_(context),
          pd_(pd),
          send_cq_(send_cq),
          recv_cq_(recv_cq),
          qp_(nullptr),
          control_sock_send_(control_sock_send),
          control_sock_recv_(control_sock_recv),
          registered_buffers_(registered_buffers),
          buffer_mutex_(buffer_mutex),
          temp_send_mr_(nullptr),
          temp_recv_mr_(nullptr),
          rank_(rank),
          peer_rank_(peer_rank),
          connected_(false)
    {
        // Create QP
        ibv_qp_init_attr qp_attr{};
        qp_attr.send_cq = send_cq_;
        qp_attr.recv_cq = recv_cq_;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr = MAX_WR;
        qp_attr.cap.max_recv_wr = MAX_WR;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        
        qp_ = ibv_create_qp(pd_, &qp_attr);
        if (!qp_) {
            throw std::runtime_error("Failed to create QP for RDMA channel");
        }
        
        // Allocate temporary buffers
        temp_send_buffer_.resize(TEMP_BUFFER_SIZE);
        temp_recv_buffer_.resize(TEMP_BUFFER_SIZE);
        
        temp_send_mr_ = ibv_reg_mr(pd_, temp_send_buffer_.data(), TEMP_BUFFER_SIZE,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        temp_recv_mr_ = ibv_reg_mr(pd_, temp_recv_buffer_.data(), TEMP_BUFFER_SIZE,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        
        if (!temp_send_mr_ || !temp_recv_mr_) {
            throw std::runtime_error("Failed to register temporary buffers");
        }
    }
    
    ~RdmaConnectionChannel() {
        if (temp_send_mr_) ibv_dereg_mr(temp_send_mr_);
        if (temp_recv_mr_) ibv_dereg_mr(temp_recv_mr_);
        if (qp_) ibv_destroy_qp(qp_);
    }
    
    void connect_qp(const RdmaConnInfo& remote_info) {
        // Transition QP to INIT
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = 1;
        attr.pkey_index = 0;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            throw std::runtime_error("Failed to transition QP to INIT");
        }
        // Query port for active MTU (aligned with Gemini)
        ibv_port_attr port_attr;
        if (ibv_query_port(context_, 1, &port_attr) != 0) {
            throw std::runtime_error("Failed to query port for RTR");
        }
        ibv_mtu mtu = port_attr.active_mtu;
        // Transition QP to RTR: use active_mtu and GID/LID like Gemini
        bool use_gid = (remote_info.lid == 0);
        attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = mtu;
        attr.dest_qp_num = remote_info.qp_num;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = use_gid ? 1 : 0;
        attr.ah_attr.dlid = remote_info.lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = 1;
        if (use_gid) {
            std::memcpy(&attr.ah_attr.grh.dgid, remote_info.gid, 16);
            attr.ah_attr.grh.flow_label = 0;
            attr.ah_attr.grh.sgid_index = 1;
            attr.ah_attr.grh.hop_limit = 255;
            attr.ah_attr.grh.traffic_class = 0;
        }
        if (ibv_modify_qp(qp_, &attr,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
            throw std::runtime_error("Failed to transition QP to RTR");
        }
        // Transition QP to RTS
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
            throw std::runtime_error("Failed to transition QP to RTS");
        }
        connected_ = true;
    }
    
    // Aligned with Gemini: do not throw on port/GID query failure; leave LID/GID zero on error.
    RdmaConnInfo get_local_conn_info() {
        RdmaConnInfo info;
        std::memset(&info, 0, sizeof(info));
        info.qp_num = qp_->qp_num;
        ibv_port_attr port_attr;
        if (ibv_query_port(context_, 1, &port_attr) == 0) {
            info.lid = port_attr.lid;
        }
        ibv_gid gid;
        if (ibv_query_gid(context_, 1, 1, &gid) == 0) {
            std::memcpy(info.gid, &gid, 16);
        }
        return info;
    }
    
    // Exchange RdmaConnInfo with peer over control socket and connect QP (aligned with Gemini: always send then recv).
    void exchange_and_connect(bool we_send_first) override {
        (void)we_send_first;
        // std::cout << "[ECNAIVE RDMA] rank=" << rank_ << " exchange_and_connect: get_local_conn_info start" << std::endl;
        RdmaConnInfo local_info = get_local_conn_info();
        // std::cout << "[ECNAIVE RDMA] rank=" << rank_ << " exchange_and_connect: get_local_conn_info done qp_num=" << local_info.qp_num << " lid=" << local_info.lid << std::endl;
        RdmaConnInfo remote_info;
        std::memset(&remote_info, 0, sizeof(remote_info));
        int sock = control_sock_send_;
        // std::cout << "[ECNAIVE RDMA] rank=" << rank_ << " fd=" << sock
        //           << " exchange_and_connect: send local RdmaConnInfo start" << std::endl;
        ssize_t n_sent = send(sock, &local_info, sizeof(local_info), 0);
        if (n_sent != static_cast<ssize_t>(sizeof(local_info))) {
            int err = errno;
            throw std::runtime_error(std::string("RdmaConnectionChannel: failed to send local RdmaConnInfo (ret=") +
                std::to_string(n_sent) + ", errno=" + std::to_string(err) + ": " + std::strerror(err) + ")");
        }
        // std::cout << "[ECNAIVE RDMA] rank=" << rank_ << " exchange_and_connect: send local RdmaConnInfo done" << std::endl;
        // std::cout << "[ECNAIVE RDMA] rank=" << rank_ << " fd=" << sock
        //           << " exchange_and_connect: recv remote RdmaConnInfo start" << std::endl;
        ssize_t n_recv = recv(sock, &remote_info, sizeof(remote_info), MSG_WAITALL);
        if (n_recv != static_cast<ssize_t>(sizeof(remote_info))) {
            int err = errno;
            throw std::runtime_error(std::string("RdmaConnectionChannel: failed to receive remote RdmaConnInfo (ret=") +
                std::to_string(n_recv) + ", errno=" + std::to_string(err) + ": " + std::strerror(err) + ")");
        }
        // std::cout << "[ECNAIVE RDMA] rank=" << rank_ << " exchange_and_connect: recv remote RdmaConnInfo done remote_qp=" << remote_info.qp_num << std::endl;
        // std::cout << "[ECNAIVE RDMA] rank=" << rank_ << " exchange_and_connect: connect_qp start" << std::endl;
        connect_qp(remote_info);
        // std::cout << "[ECNAIVE RDMA] rank=" << rank_ << " exchange_and_connect: connect_qp done" << std::endl;
    }
    
    void send_data(const uint8_t* data, size_t size) override {
        std::lock_guard<std::mutex> lock(send_mutex_);
        
        if (!connected_) {
            throw std::runtime_error("RDMA channel not connected");
        }
        
        // Send size via control socket
        uint64_t size_network = htobe64(size);
        if (send(control_sock_send_, &size_network, sizeof(size_network), 0) != sizeof(size_network)) {
            throw std::runtime_error("Failed to send size via control socket");
        }
        
        // Wait for ACK
        uint8_t ack;
        if (recv(control_sock_send_, &ack, sizeof(ack), MSG_WAITALL) != sizeof(ack)) {
            throw std::runtime_error("Failed to receive ACK");
        }
        
        // Find registered MR or use temp buffer
        ibv_mr* mr = find_registered_mr(reinterpret_cast<uintptr_t>(data), size);
        bool use_temp = false;
        
        if (!mr) {
            if (size > TEMP_BUFFER_SIZE) {
                throw std::runtime_error("Data size exceeds temporary buffer size");
            }
            memcpy(temp_send_buffer_.data(), data, size);
            mr = temp_send_mr_;
            data = temp_send_buffer_.data();
            use_temp = true;
        }
        
        // Send data in chunks
        send_data_chunked(data, size, mr);
    }
    
    size_t receive_data(uint8_t* buffer, size_t buffer_size) override {
        std::lock_guard<std::mutex> lock(recv_mutex_);
        
        if (!connected_) {
            throw std::runtime_error("RDMA channel not connected");
        }
        
        // Receive size via control socket
        uint64_t size_network;
        if (recv(control_sock_recv_, &size_network, sizeof(size_network), MSG_WAITALL) != sizeof(size_network)) {
            throw std::runtime_error("Failed to receive size via control socket");
        }
        size_t size = be64toh(size_network);
        
        if (size > buffer_size) {
            throw std::runtime_error("Received size exceeds buffer size");
        }
        
        // Send immediate ACK
        uint8_t ack = 1;
        if (send(control_sock_recv_, &ack, sizeof(ack), 0) != sizeof(ack)) {
            throw std::runtime_error("Failed to send ACK");
        }
        
        // Find registered MR or use temp buffer
        ibv_mr* mr = find_registered_mr(reinterpret_cast<uintptr_t>(buffer), size);
        bool use_temp = false;
        
        if (!mr) {
            if (size > TEMP_BUFFER_SIZE) {
                throw std::runtime_error("Data size exceeds temporary buffer size");
            }
            mr = temp_recv_mr_;
            use_temp = true;
        }
        
        // Receive data in chunks
        uint8_t* recv_ptr = use_temp ? temp_recv_buffer_.data() : buffer;
        receive_data_chunked(recv_ptr, size, mr);
        
        // Copy from temp buffer if needed
        if (use_temp) {
            memcpy(buffer, temp_recv_buffer_.data(), size);
        }
        
        return size;
    }
    
    bool is_connected() const override {
        return connected_;
    }

private:
    ibv_mr* find_registered_mr(uintptr_t addr, size_t size) {
        std::lock_guard<std::mutex> lock(*buffer_mutex_);
        
        for (auto& [reg_addr, buf] : *registered_buffers_) {
            if (addr >= reg_addr && (addr + size) <= (reg_addr + buf.size)) {
                return buf.mr;
            }
        }
        return nullptr;
    }
    
    void send_data_chunked(const uint8_t* data, size_t total_size, ibv_mr* mr) {
        size_t remaining = total_size;
        size_t offset = 0;
        
        while (remaining > 0) {
            size_t chunk_size = std::min(remaining, CHUNK_SIZE);
            size_t chunk_count = (chunk_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
            
            // Resize first so &sges[i] and &wrs[i+1] stay valid for the whole post_send call.
            std::vector<ibv_sge> sges(chunk_count);
            std::vector<ibv_send_wr> wrs(chunk_count);
            
            for (size_t i = 0; i < chunk_count; ++i) {
                size_t current_size = std::min(CHUNK_SIZE, remaining);
                
                sges[i].addr = reinterpret_cast<uint64_t>(data + offset);
                sges[i].length = static_cast<uint32_t>(current_size);
                sges[i].lkey = mr->lkey;
                
                std::memset(&wrs[i], 0, sizeof(wrs[i]));
                wrs[i].wr_id = i;
                wrs[i].sg_list = &sges[i];
                wrs[i].num_sge = 1;
                wrs[i].opcode = IBV_WR_SEND;
                wrs[i].send_flags = IBV_SEND_SIGNALED;
                wrs[i].next = (i + 1 < chunk_count) ? &wrs[i + 1] : nullptr;
                
                offset += current_size;
                remaining -= current_size;
            }
            
            ibv_send_wr* bad_wr = nullptr;
            if (ibv_post_send(qp_, &wrs[0], &bad_wr)) {
                throw std::runtime_error("Failed to post send work request");
            }
            
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
                sges[i].length = static_cast<uint32_t>(current_size);
                sges[i].lkey = mr->lkey;
                
                std::memset(&wrs[i], 0, sizeof(wrs[i]));
                wrs[i].wr_id = i;
                wrs[i].sg_list = &sges[i];
                wrs[i].num_sge = 1;
                wrs[i].next = (i + 1 < chunk_count) ? &wrs[i + 1] : nullptr;
                
                offset += current_size;
                remaining -= current_size;
            }
            
            ibv_recv_wr* bad_wr = nullptr;
            if (ibv_post_recv(qp_, &wrs[0], &bad_wr)) {
                throw std::runtime_error("Failed to post receive work request");
            }
            
            poll_completion(recv_cq_, static_cast<int>(chunk_count));
        }
    }
    
    void poll_completion(ibv_cq* cq, int num_completions) {
        int completed = 0;
        while (completed < num_completions) {
            ibv_wc wc;
            int ret = ibv_poll_cq(cq, 1, &wc);
            if (ret < 0) {
                throw std::runtime_error("Failed to poll CQ");
            }
            if (ret > 0) {
                if (wc.status != IBV_WC_SUCCESS) {
                    throw std::runtime_error("Work completion failed");
                }
                completed++;
            }
        }
    }
};

// ASIO connection manager (pattern from eccheck_native)
class AsioConnectionManager {
private:
    boost::asio::io_context io_context_;
    
    // Save mode sockets: generalized vectors (size = num_channels = k+1)
    std::vector<boost::asio::ip::tcp::socket> send_sockets_;
    std::vector<boost::asio::ip::tcp::socket> recv_sockets_;
    std::vector<boost::asio::ip::tcp::acceptor> recv_acceptors_;
    std::deque<std::atomic<bool>> send_connected_;
    std::deque<std::atomic<bool>> recv_connected_;

    // Legacy named save-mode sockets (referenced by init_/get_/cleanup methods)
    boost::asio::ip::tcp::socket send_data1_socket_;
    boost::asio::ip::tcp::socket send_parity0_socket_;
    boost::asio::ip::tcp::socket send_parity1_socket_;
    boost::asio::ip::tcp::socket recv_parity1_socket_;
    boost::asio::ip::tcp::socket recv_parity0_socket_;
    boost::asio::ip::tcp::socket recv_data1_socket_;
    boost::asio::ip::tcp::acceptor recv_parity1_acceptor_;
    boost::asio::ip::tcp::acceptor recv_parity0_acceptor_;
    boost::asio::ip::tcp::acceptor recv_data1_acceptor_;

    // Software-only SW recovery sockets (k=2, rank3 -> rank2)
    boost::asio::ip::tcp::socket ecnaive_load_recv_rank3_data1_socket_;
    boost::asio::ip::tcp::acceptor ecnaive_load_recv_rank3_data1_acceptor_;
    boost::asio::ip::tcp::socket ecnaive_load_send_rank3_data1_socket_;

    // Save mode connection flags
    std::atomic<bool> send_data1_connected_{false};
    std::atomic<bool> send_parity0_connected_{false};
    std::atomic<bool> send_parity1_connected_{false};
    std::atomic<bool> recv_parity1_connected_{false};
    std::atomic<bool> recv_parity0_connected_{false};
    std::atomic<bool> recv_data1_connected_{false};

    std::atomic<bool> ecnaive_load_recv_rank3_data1_connected_{false};
    std::atomic<bool> ecnaive_load_send_rank3_data1_connected_{false};

    std::mutex connection_mutex_;
    std::condition_variable connection_cv_;

public:
    AsioConnectionManager()
        : io_context_(),
          send_data1_socket_(io_context_),
          send_parity0_socket_(io_context_),
          send_parity1_socket_(io_context_),
          recv_parity1_socket_(io_context_),
          recv_parity0_socket_(io_context_),
          recv_data1_socket_(io_context_),
          recv_parity1_acceptor_(io_context_),
          recv_parity0_acceptor_(io_context_),
          recv_data1_acceptor_(io_context_),
          ecnaive_load_recv_rank3_data1_socket_(io_context_),
          ecnaive_load_recv_rank3_data1_acceptor_(io_context_),
          ecnaive_load_send_rank3_data1_socket_(io_context_) {}

    // Save mode getters
    boost::asio::ip::tcp::socket& get_send_data1_socket() { return send_data1_socket_; }
    boost::asio::ip::tcp::socket& get_send_parity0_socket() { return send_parity0_socket_; }
    boost::asio::ip::tcp::socket& get_send_parity1_socket() { return send_parity1_socket_; }
    boost::asio::ip::tcp::socket& get_recv_parity1_socket() { return recv_parity1_socket_; }
    boost::asio::ip::tcp::socket& get_recv_parity0_socket() { return recv_parity0_socket_; }
    boost::asio::ip::tcp::socket& get_recv_data1_socket() { return recv_data1_socket_; }

    boost::asio::ip::tcp::socket& get_ecnaive_load_recv_rank3_data1_socket() {
        return ecnaive_load_recv_rank3_data1_socket_;
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_send_rank3_data1_socket() {
        return ecnaive_load_send_rank3_data1_socket_;
    }

    // Save mode connection checks
    bool is_send_data1_connected() const { return send_data1_connected_; }
    bool is_send_parity0_connected() const { return send_parity0_connected_; }
    bool is_send_parity1_connected() const { return send_parity1_connected_; }
    bool is_recv_parity1_connected() const { return recv_parity1_connected_; }
    bool is_recv_parity0_connected() const { return recv_parity0_connected_; }
    bool is_recv_data1_connected() const { return recv_data1_connected_; }

    // Save mode init functions
    void init_send_data1(const std::string& partner_ip, uint16_t port);
    void init_send_parity0(const std::string& partner_ip, uint16_t port);
    void init_send_parity1(const std::string& partner_ip, uint16_t port);
    void init_recv_parity1(const std::string& listen_ip, uint16_t port);
    void init_recv_parity0(const std::string& listen_ip, uint16_t port);
    void init_recv_data1(const std::string& listen_ip, uint16_t port);

    // Generalized init methods (for k+2 schemes)
    void init_send_channels(const std::vector<std::string>& ips,
                            const std::vector<uint16_t>& ports);
    void init_recv_channels(const std::vector<std::string>& ips,
                            const std::vector<uint16_t>& ports);
    int num_save_channels() const { return static_cast<int>(send_sockets_.size()); }
    boost::asio::ip::tcp::socket& send_socket(int idx) { return send_sockets_[idx]; }
    boost::asio::ip::tcp::socket& recv_socket(int idx) { return recv_sockets_[idx]; }
    
    // Software-only SW recovery (k=2)
    void bind_listen_ecnaive_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port);
    void accept_ecnaive_load_recv_rank3_data1();
    void init_ecnaive_load_send_rank3_data1(const std::string& rank2_ip, uint16_t port);
    bool is_ecnaive_load_recv_rank3_data1_connected() const {
        return ecnaive_load_recv_rank3_data1_connected_;
    }
    bool is_ecnaive_load_send_rank3_data1_connected() const {
        return ecnaive_load_send_rank3_data1_connected_;
    }

    void wait_for_connections(int timeout_seconds = 30);
    void cleanup();
};

// Save mode init functions
void AsioConnectionManager::init_send_data1(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(send_data1_socket_, endpoints);
        send_data1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: send_data1 init error: " << e.what() << std::endl;
        send_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_send_parity0(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(send_parity0_socket_, endpoints);
        send_parity0_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: send_parity0 init error: " << e.what() << std::endl;
        send_parity0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_send_parity1(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(send_parity1_socket_, endpoints);
        send_parity1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: send_parity1 init error: " << e.what() << std::endl;
        send_parity1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_recv_parity1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        recv_parity1_acceptor_.open(endpoint.protocol());
        recv_parity1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        recv_parity1_acceptor_.bind(endpoint);
        recv_parity1_acceptor_.listen();
        recv_parity1_acceptor_.accept(recv_parity1_socket_);
        recv_parity1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: recv_parity1 init error: " << e.what() << std::endl;
        recv_parity1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_recv_parity0(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        recv_parity0_acceptor_.open(endpoint.protocol());
        recv_parity0_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        recv_parity0_acceptor_.bind(endpoint);
        recv_parity0_acceptor_.listen();
        recv_parity0_acceptor_.accept(recv_parity0_socket_);
        recv_parity0_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: recv_parity0 init error: " << e.what() << std::endl;
        recv_parity0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_recv_data1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        recv_data1_acceptor_.open(endpoint.protocol());
        recv_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        recv_data1_acceptor_.bind(endpoint);
        recv_data1_acceptor_.listen();
        recv_data1_acceptor_.accept(recv_data1_socket_);
        recv_data1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: recv_data1 init error: " << e.what() << std::endl;
        recv_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

// Generalized init: initialize all send channels from vectors
void AsioConnectionManager::init_send_channels(
    const std::vector<std::string>& ips,
    const std::vector<uint16_t>& ports)
{
    send_sockets_.clear();
    send_connected_.clear();
    send_sockets_.reserve(ips.size());
    for (size_t i = 0; i < ips.size(); ++i) {
        send_sockets_.emplace_back(io_context_);
        send_connected_.emplace_back(false);
    }
    for (size_t i = 0; i < ips.size(); ++i) {
        try {
            boost::asio::ip::tcp::resolver resolver(io_context_);
            auto endpoints = resolver.resolve(ips[i], std::to_string(ports[i]));
            boost::asio::connect(send_sockets_[i], endpoints);
            send_connected_[i] = true;
        } catch (const std::exception& e) {
            std::cerr << "ASIO: send channel " << i << " init error: " << e.what() << std::endl;
            send_connected_[i] = false;
        }
    }
    connection_cv_.notify_all();
}

// Generalized init: initialize all recv channels from vectors
void AsioConnectionManager::init_recv_channels(
    const std::vector<std::string>& ips,
    const std::vector<uint16_t>& ports)
{
    recv_sockets_.clear();
    recv_acceptors_.clear();
    recv_connected_.clear();
    recv_sockets_.reserve(ips.size());
    recv_acceptors_.reserve(ips.size());
    for (size_t i = 0; i < ips.size(); ++i) {
        recv_sockets_.emplace_back(io_context_);
        recv_acceptors_.emplace_back(io_context_);
        recv_connected_.emplace_back(false);
    }
    // Start all acceptors in parallel
    std::vector<std::thread> accept_threads;
    for (size_t i = 0; i < ips.size(); ++i) {
        accept_threads.emplace_back([this, i, &ips, &ports]() {
            try {
                boost::asio::ip::tcp::endpoint endpoint(
                    boost::asio::ip::address::from_string(ips[i]), ports[i]);
                recv_acceptors_[i].open(endpoint.protocol());
                recv_acceptors_[i].set_option(
                    boost::asio::ip::tcp::acceptor::reuse_address(true));
                recv_acceptors_[i].bind(endpoint);
                recv_acceptors_[i].listen();
                recv_acceptors_[i].accept(recv_sockets_[i]);
                recv_connected_[i] = true;
            } catch (const std::exception& e) {
                std::cerr << "ASIO: recv channel " << i << " init error: " << e.what() << std::endl;
                recv_connected_[i] = false;
            }
        });
    }
    for (auto& t : accept_threads) t.join();
    connection_cv_.notify_all();
}

void AsioConnectionManager::wait_for_connections(int timeout_seconds) {
    std::unique_lock<std::mutex> lock(connection_mutex_);
    connection_cv_.wait_for(
        lock,
        std::chrono::seconds(timeout_seconds),
        [this]() {
            // Check generalized channels if they exist
            bool all_ok = true;
            if (!send_connected_.empty()) {
                for (size_t i = 0; i < send_connected_.size(); ++i)
                    if (!send_connected_[i]) { all_ok = false; break; }
                for (size_t i = 0; i < recv_connected_.size(); ++i)
                    if (!recv_connected_[i]) { all_ok = false; break; }
            } else {
                // Fallback to named flags (backward compat)
                all_ok = send_data1_connected_ && send_parity0_connected_ && send_parity1_connected_ &&
                         recv_parity1_connected_ && recv_parity0_connected_ && recv_data1_connected_;
            }
            return all_ok;
        }
    );
}

// ========== EC-NAIVE Load Mode Connection Methods ==========

// EC-NAIVE load mode bind+listen helpers (for rank2, before accept)
void AsioConnectionManager::bind_listen_ecnaive_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        ecnaive_load_recv_rank3_data1_acceptor_.open(endpoint.protocol());
        ecnaive_load_recv_rank3_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        ecnaive_load_recv_rank3_data1_acceptor_.bind(endpoint);
        ecnaive_load_recv_rank3_data1_acceptor_.listen();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: bind_listen_ecnaive_load_recv_rank3_data1 error: " << e.what() << std::endl;
        throw;
    }
}

// EC-NAIVE load mode accept helpers (for rank2, after bind+listen)
void AsioConnectionManager::accept_ecnaive_load_recv_rank3_data1() {
    try {
        ecnaive_load_recv_rank3_data1_acceptor_.accept(ecnaive_load_recv_rank3_data1_socket_);
        ecnaive_load_recv_rank3_data1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: accept_ecnaive_load_recv_rank3_data1 error: " << e.what() << std::endl;
        ecnaive_load_recv_rank3_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

// EC-NAIVE load mode init functions (rank0/1/3 sender)
void AsioConnectionManager::init_ecnaive_load_send_rank3_data1(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(ecnaive_load_send_rank3_data1_socket_, endpoints);
        ecnaive_load_send_rank3_data1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: init_ecnaive_load_send_rank3_data1 error: " << e.what() << std::endl;
        ecnaive_load_send_rank3_data1_connected_ = false;
        connection_cv_.notify_all();
        throw;
    }
}

void AsioConnectionManager::cleanup() {
    // Save mode sockets
    if (send_data1_socket_.is_open()) send_data1_socket_.close();
    if (send_parity0_socket_.is_open()) send_parity0_socket_.close();
    if (send_parity1_socket_.is_open()) send_parity1_socket_.close();
    if (recv_parity1_socket_.is_open()) recv_parity1_socket_.close();
    if (recv_parity0_socket_.is_open()) recv_parity0_socket_.close();
    if (recv_data1_socket_.is_open()) recv_data1_socket_.close();
    if (recv_parity1_acceptor_.is_open()) recv_parity1_acceptor_.close();
    if (recv_parity0_acceptor_.is_open()) recv_parity0_acceptor_.close();
    if (recv_data1_acceptor_.is_open()) recv_data1_acceptor_.close();
    
    // Software-only SW recovery sockets
    if (ecnaive_load_recv_rank3_data1_socket_.is_open()) ecnaive_load_recv_rank3_data1_socket_.close();
    if (ecnaive_load_recv_rank3_data1_acceptor_.is_open()) ecnaive_load_recv_rank3_data1_acceptor_.close();
    if (ecnaive_load_send_rank3_data1_socket_.is_open()) ecnaive_load_send_rank3_data1_socket_.close();
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

struct RecvTask {
    uintptr_t addr{0};
    size_t size{0};
};

// RS decode pthread pool: shared job + per-worker stripe execution
struct EcRsJob {
    size_t size;
    int k;       // number of surviving blocks (input to ec_encode_data)
    int m;       // number of recovered blocks (output from ec_encode_data)
    const uintptr_t* surviving_addrs;  // [k] input block addresses
    const uintptr_t* recovered_addrs;  // [m] output block addresses
    const unsigned char* decode_tbls;  // decode tables (32 * k * m bytes)
};

class ECNaiveNative;

struct EcRsPoolWorkerCtx {
    ECNaiveNative* self{nullptr};
    int wid{0};
};

class ECNaiveNative {
public:
    ECNaiveNative(const std::vector<std::string>& send_ips,
                  const std::vector<uint16_t>& send_ports,
                  const std::vector<std::string>& recv_ips,
                  const std::vector<uint16_t>& recv_ports,
                  const std::vector<uint16_t>& rdma_send_ports,
                  const std::vector<uint16_t>& rdma_recv_ports,
                  int k = 2,
                  bool use_rdma = false,
                  int rank_in_group = -1)
        : stop_(false),
          use_rdma_(use_rdma),
          rdma_context_(nullptr),
          rdma_pd_(nullptr),
          rdma_software_load_send_cq_(nullptr),
          rdma_software_load_recv_cq_(nullptr),
          send_ips_(send_ips),
          send_ports_(send_ports),
          recv_ips_(recv_ips),
          recv_ports_(recv_ports),
          rdma_send_ports_(rdma_send_ports),
          rdma_recv_ports_(rdma_recv_ports),
          k_(k),
          rows_(2),
          n_(k + 2),
          num_channels_(static_cast<int>(send_ips.size())),
          a_mat_(nullptr),
          g_tbls_(nullptr),
          rank_(-1),
          rank_in_group_(rank_in_group) {

        // Validate input sizes
        if (static_cast<int>(send_ips.size()) != num_channels_ ||
            static_cast<int>(recv_ips.size()) != num_channels_ ||
            static_cast<int>(recv_ports.size()) != num_channels_) {
            throw std::runtime_error("ECNaiveNative: Mismatched channel counts");
        }
        if (num_channels_ != n_ - 1) {
            throw std::runtime_error("ECNaiveNative: Expected n-1 channels, got " +
                                     std::to_string(num_channels_));
        }
        if (k_ < 2) {
            throw std::runtime_error("ECNaiveNative: k must be >= 2");
        }

        // Initialize EC encoding tables
        init_ec_encoding();

        // Fill vectors via emplace_back (avoids resize on non-copyable types)
        auto reserve_all = [&](int n) {
            send_queues_.reserve(n);
            recv_queues_.reserve(n);
            send_channels_.reserve(n);
            recv_channels_.reserve(n);
            rdma_send_fds_.reserve(n);
            rdma_recv_fds_.reserve(n);
            rdma_send_cqs_.reserve(n);
            rdma_recv_cqs_.reserve(n);
        };
        reserve_all(num_channels_);

        for (int i = 0; i < num_channels_; ++i) {
            send_queues_.emplace_back();
            send_mutexes_.emplace_back();
            send_cvs_.emplace_back();
            recv_queues_.emplace_back();
            recv_mutexes_.emplace_back();
            recv_cvs_.emplace_back();
            send_channels_.emplace_back();
            recv_channels_.emplace_back();
            send_threads_.emplace_back();
            recv_threads_.emplace_back();
            send_completed_.emplace_back(false);
            recv_completed_.emplace_back(false);
            send_sentinel_received_.emplace_back(false);
            recv_sentinel_received_.emplace_back(false);
            rdma_send_fds_.emplace_back(-1);
            rdma_recv_fds_.emplace_back(-1);
            rdma_send_cqs_.emplace_back(nullptr);
            rdma_recv_cqs_.emplace_back(nullptr);
        }


        // Initialize RDMA resources if enabled
        if (use_rdma_) {
            try {
                init_rdma_resources();
            } catch (const std::exception& e) {
                std::cerr << "ECNAIVE: RDMA initialization failed: " << e.what() << std::endl;
                std::cerr << "ECNAIVE: Falling back to ASIO" << std::endl;
                use_rdma_ = false;
            }
        }

        init_connections();
        if (use_rdma_) {
            init_rdma_exchange_sockets();
        }
        if (use_rdma_) {
            init_rdma_save_channels();
        }
        start_threads();
    }

    ~ECNaiveNative() {
        stop();
        
        // Clean up RDMA resources
        cleanup_rdma_resources();
        
        // Free EC encoding tables
        if (g_tbls_ != nullptr) {
            free(g_tbls_);
            g_tbls_ = nullptr;
        }
        if (a_mat_ != nullptr) {
            free(a_mat_);
            a_mat_ = nullptr;
        }
    }
    
    // Buffer registration methods for RDMA
    void register_buffer(uintptr_t addr, size_t size) {
        if (!use_rdma_ || !rdma_pd_) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(rdma_buffer_mutex_);
        
        // Check if already registered
        if (rdma_registered_buffers_.find(addr) != rdma_registered_buffers_.end()) {
            // std::cout << "[ECNAIVE RDMA] Buffer already registered at 0x" << std::hex << addr << std::dec << std::endl;
            return;
        }
        
        // std::cout << "[ECNAIVE RDMA] Registering buffer at 0x" << std::hex << addr << std::dec 
                //   << ", size: " << (size / (1024.0 * 1024.0)) << " MB" << std::endl;
        
        ibv_mr* mr = ibv_reg_mr(rdma_pd_, reinterpret_cast<void*>(addr), size,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        
        if (!mr) {
            throw std::runtime_error("Failed to register memory region for RDMA");
        }
        
        rdma_registered_buffers_[addr] = {mr, addr, size};
        // std::cout << "[ECNAIVE RDMA] Buffer registered successfully (total: " << rdma_registered_buffers_.size() << ")" << std::endl;
    }
    
    void unregister_buffer(uintptr_t addr) {
        if (!use_rdma_) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(rdma_buffer_mutex_);
        
        auto it = rdma_registered_buffers_.find(addr);
        if (it != rdma_registered_buffers_.end()) {
            ibv_dereg_mr(it->second.mr);
            rdma_registered_buffers_.erase(it);
            // std::cout << "[ECNAIVE RDMA] Buffer unregistered at 0x" << std::hex << addr << std::dec << std::endl;
        }
    }

    // Save mode pipelines: 3 sends + 3 receives
    void submit_send_data1(uintptr_t send_addr, size_t size) {
        // std::cout << "ECNAIVE: Submitting send_data1 task: send_addr=" << send_addr
        //           << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(send_data1_mutex_);
            send_data1_q_.push({send_addr, size});
        }
        send_data1_cv_.notify_one();
    }

    void submit_send_parity0(uintptr_t send_addr, size_t size) {
        // std::cout << "ECNAIVE: Submitting send_parity0 task: send_addr=" << send_addr
        //           << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(send_parity0_mutex_);
            send_parity0_q_.push({send_addr, size});
        }
        send_parity0_cv_.notify_one();
    }

    void submit_send_parity1(uintptr_t send_addr, size_t size) {
        // std::cout << "ECNAIVE: Submitting send_parity1 task: send_addr=" << send_addr << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(send_parity1_mutex_);
            send_parity1_q_.push({send_addr, size});
        }
        send_parity1_cv_.notify_one();
    }

    void submit_recv_parity1(uintptr_t recv_addr, size_t size) {
        // std::cout << "ECNAIVE: Submitting recv_parity1 task: recv_addr=" << recv_addr
        //           << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_parity1_mutex_);
            recv_parity1_q_.push({recv_addr, size});
        }
        recv_parity1_cv_.notify_one();
    }

    void submit_recv_parity0(uintptr_t recv_addr, size_t size) {
        // std::cout << "ECNAIVE: Submitting recv_parity0 task: recv_addr=" << recv_addr
        //           << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_parity0_mutex_);
            recv_parity0_q_.push({recv_addr, size});
        }
        recv_parity0_cv_.notify_one();
    }

    void submit_recv_data1(uintptr_t recv_addr, size_t size) {
        // std::cout << "ECNAIVE: Submitting recv_data1 task: recv_addr=" << recv_addr
        //           << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_data1_mutex_);
            recv_data1_q_.push({recv_addr, size});
        }
        recv_data1_cv_.notify_one();
    }

    // Unified save mode submit function: encode and submit send/recv tasks
    // rank i: keep d_{i0}, send d_{i1} to rank i+1, p_{i0} to rank i+2, p_{i1} to rank i+3
    // rank i: recv p_{i+1,1} from rank i+1, p_{i+2,0} from rank i+2, d_{i+3,1} from rank i+3
    void submit_ecnaive_save(uintptr_t data0_addr,      // d_{i0} - keep, not sent
                             uintptr_t data1_addr,      // d_{i1} - send to rank i+1
                             uintptr_t parity0_addr,    // p_{i0} - send to rank i+2
                             uintptr_t parity1_addr,    // p_{i1} - send to rank i+3
                             uintptr_t recv_parity1_addr, // recv p_{i+1,1} from rank i+1
                             uintptr_t recv_parity0_addr, // recv p_{i+2,0} from rank i+2
                             uintptr_t recv_data1_addr,   // recv d_{i+3,1} from rank i+3
                             size_t size) {
        // std::cout << "ECNAIVE: Submitting save task: data0=" << data0_addr
        //           << ", data1=" << data1_addr
        //           << ", parity0=" << parity0_addr
        //           << ", parity1=" << parity1_addr
        //           << ", recv_p1=" << recv_parity1_addr
        //           << ", recv_p0=" << recv_parity0_addr
        //           << ", recv_d1=" << recv_data1_addr
        //           << ", size=" << size << std::endl;
        
        // Step 1: Encode data blocks to get parity blocks (legacy 2-data-block call)
        {
            std::vector<uintptr_t> addrs = {data0_addr, data1_addr};
            encode_ec_blocks(addrs, parity0_addr, parity1_addr, size);
        }
        
        // Step 2: Submit send tasks (data1, parity0, parity1)
        submit_send_data1(data1_addr, size);
        submit_send_parity0(parity0_addr, size);
        submit_send_parity1(parity1_addr, size);
        
        // Step 3: Submit receive tasks (recv_parity1, recv_parity0, recv_data1)
        submit_recv_parity1(recv_parity1_addr, size);
        submit_recv_parity0(recv_parity0_addr, size);
        submit_recv_data1(recv_data1_addr, size);
    }

    // Submit sentinels to signal pipeline completion
    void submit_send_data1_sentinel() {
        {
            std::lock_guard<std::mutex> lk(send_data1_mutex_);
            send_data1_q_.push({0, 0});
        }
        send_data1_cv_.notify_one();
    }

    void submit_send_parity0_sentinel() {
        // std::cout << "ECNAIVE: Submitting sentinel to send_parity0 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(send_parity0_mutex_);
            send_parity0_q_.push({0, 0});
        }
        send_parity0_cv_.notify_one();
    }

    void submit_send_parity1_sentinel() {
        // std::cout << "ECNAIVE: Submitting sentinel to send_parity1 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(send_parity1_mutex_);
            send_parity1_q_.push({0, 0});
        }
        send_parity1_cv_.notify_one();
    }

    void submit_recv_parity1_sentinel() {
        // std::cout << "ECNAIVE: Submitting sentinel to recv_parity1 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_parity1_mutex_);
            recv_parity1_q_.push({0, 0});
        }
        recv_parity1_cv_.notify_one();
    }

    void submit_recv_parity0_sentinel() {
        // std::cout << "ECNAIVE: Submitting sentinel to recv_parity0 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_parity0_mutex_);
            recv_parity0_q_.push({0, 0});
        }
        recv_parity0_cv_.notify_one();
    }

    void submit_recv_data1_sentinel() {
        // std::cout << "ECNAIVE: Submitting sentinel to recv_data1 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_data1_mutex_);
            recv_data1_q_.push({0, 0});
        }
        recv_data1_cv_.notify_one();
    }

    // ========== RS Decode Recovery (synchronous, with 16-worker pool) ==========

    // Recover m lost data blocks from k_orig data blocks using surviving blocks.
    // k_orig: number of original data blocks (e.g. 2 or 6)
    // m: number of lost data blocks (1-2)
    // lost_positions: indices of lost data blocks (0..k_orig-1)
    // surviving_addrs: addresses of surviving blocks
    //   - surviving data blocks in original position order
    //   - surviving parity blocks at the end (1 or 2)
    //   Size = (k_orig - m) + num_surviving_parity, which may be != k_orig
    void submit_ecnaive_decode_recovery(
            int k_orig, int m,
            const std::vector<int>& lost_positions,
            const std::vector<uintptr_t>& surviving_addrs,
            const std::vector<uintptr_t>& recovered_addrs,
            size_t size)
    {
        if (k_orig < 2 || m < 1 || m > 2) {
            std::cerr << "ECNAIVE: decode_recovery invalid params k_orig=" << k_orig << " m=" << m << std::endl;
            return;
        }
        int surviving_count = static_cast<int>(surviving_addrs.size());
        int num_surviving_parity = surviving_count - (k_orig - m);
        if (num_surviving_parity < 1 || num_surviving_parity > 2) {
            std::cerr << "ECNAIVE: decode_recovery invalid parity count " << num_surviving_parity << std::endl;
            return;
        }
        if (static_cast<int>(recovered_addrs.size()) != m) {
            std::cerr << "ECNAIVE: decode_recovery need exactly m recovered buffers" << std::endl;
            return;
        }

        // Step 1: Build full (k_orig+2) x k_orig Vandermonde encoding matrix
        int full_rows = k_orig + 2;
        std::vector<unsigned char> encode_mat(k_orig * full_rows);
        gf_gen_rs_matrix(encode_mat.data(), full_rows, k_orig);

        // Step 2: Build k_orig x k_orig survivor matrix A
        std::vector<unsigned char> A(k_orig * k_orig, 0);
        int data_row = 0;
        for (int pos = 0; pos < k_orig; ++pos) {
            bool is_lost = false;
            for (int lp : lost_positions) {
                if (lp == pos) { is_lost = true; break; }
            }
            if (!is_lost) {
                A[data_row * k_orig + pos] = 1;
                ++data_row;
            }
        }
        // Fill in surviving parity rows
        for (int parity_idx = 0; parity_idx < num_surviving_parity; ++parity_idx) {
            int src_row = k_orig + parity_idx;
            for (int col = 0; col < k_orig; ++col) {
                A[data_row * k_orig + col] = encode_mat[src_row * k_orig + col];
            }
            ++data_row;
        }

        // Step 3: Invert A in GF(2^8)
        std::vector<unsigned char> inv_workspace(k_orig * 2 * k_orig);
        std::vector<unsigned char> A_inv(k_orig * k_orig);
        for (int i = 0; i < k_orig * k_orig; ++i) inv_workspace[i] = A[i];
        int ret = gf_invert_matrix(inv_workspace.data(), A_inv.data(), k_orig);
        if (ret != 0) {
            std::cerr << "ECNAIVE: gf_invert_matrix failed (singular matrix), ret=" << ret << std::endl;
            for (int i = 0; i < m; ++i)
                std::memset(reinterpret_cast<void*>(recovered_addrs[i]), 0, size);
            return;
        }

        // Step 4: Extract decode coefficients (surviving_count columns per lost position)
        std::vector<unsigned char> decode_mat(m * surviving_count);
        for (int i = 0; i < m; ++i) {
            int lost_pos = lost_positions[i];
            int col = 0;
            // Coefficients for surviving data blocks (at their original positions)
            for (int pos = 0; pos < k_orig; ++pos) {
                bool is_lost = false;
                for (int lp : lost_positions) { if (lp == pos) { is_lost = true; break; } }
                if (!is_lost) {
                    decode_mat[i * surviving_count + col] = A_inv[lost_pos * k_orig + pos];
                    ++col;
                }
            }
            // Coefficients for surviving parity blocks
            for (int pi = 0; pi < num_surviving_parity; ++pi) {
                decode_mat[i * surviving_count + col] = A_inv[lost_pos * k_orig + k_orig + pi];
                ++col;
            }
        }

        // Step 5: Generate decode tables
        size_t decode_tbls_size = 32 * (size_t)surviving_count * (size_t)m;
        ec_decode_tbls_.resize(decode_tbls_size);
        ec_init_tables(surviving_count, m, decode_mat.data(), ec_decode_tbls_.data());
        ec_rs_pool_have_tbls_ = true;
        ec_decode_tbls_k_ = surviving_count;
        ec_decode_tbls_m_ = m;

        // Step 6: Lazy-init RS pool on first use
        if (!ec_rs_pool_inited_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lk(ec_rs_pool_mutex_);
            if (!ec_rs_pool_inited_.load(std::memory_order_acquire)) {
                ec_rs_pool_init();
            }
        }

        // Step 7: Dispatch to 16-worker pthread pool
        EcRsJob job;
        job.size = size;
        job.k = surviving_count;
        job.m = m;
        job.surviving_addrs = surviving_addrs.data();
        job.recovered_addrs = recovered_addrs.data();
        job.decode_tbls = ec_decode_tbls_.data();
        ec_rs_pool_run_parallel(job);

    }

    // Software failure mode interfaces (direct send/recv without pipeline)

    // rank3 software failure mode send d21 (once complete transmission)
    // ---- Generalized software failure send/recv (supports any k) ----

    // Legacy k=2 aliases for backward compat
    void software_send_rank3_data1(uintptr_t send_addr, size_t size) {
        sw_send_data(0, send_addr, size);
    }
    void software_recv_data1(uintptr_t recv_addr, size_t size) {
        sw_recv_data(0, recv_addr, size);
    }

    void sw_send_data(int block_idx, uintptr_t send_addr, size_t size) {
        LoadNetScopeTimer net_timer(this);
        // Forward to RANK2 (failed rank): try RDMA channel first, fall back to ASIO socket
        if (use_rdma_ && block_idx >= 0 && static_cast<size_t>(block_idx) < rdma_sw_recovery_channels_.size()
            && rdma_sw_recovery_channels_[block_idx]) {
            rdma_sw_recovery_channels_[block_idx]->send_data(
                reinterpret_cast<const uint8_t*>(send_addr), size);
            return;
        }

        if (use_rdma_ && rdma_software_load_channel_) {
            // std::cout << "[ECNAIVE RDMA] Load: Sending " << size << " bytes via RDMA" << std::endl;
            rdma_software_load_channel_->send_data(reinterpret_cast<const uint8_t*>(send_addr), size);
            return;
        }
        // Legacy k=2 path: dedicated named socket
        if (block_idx == 0 && conn_.is_ecnaive_load_send_rank3_data1_connected()) {
            send_with_size(conn_.get_ecnaive_load_send_rank3_data1_socket(), send_addr, size);
            return;
        }
        std::cerr << "EC-NAIVE: sw_send_data(" << block_idx << ", " << size << "): no channel" << std::endl;
    }

    void sw_recv_data(int block_idx, uintptr_t recv_addr, size_t size) {
        LoadRecvNetScopeTimer net_timer(this);
        if (use_rdma_ && block_idx >= 0 && static_cast<size_t>(block_idx) < rdma_sw_recovery_channels_.size()
            && rdma_sw_recovery_channels_[block_idx]) {
            rdma_sw_recovery_channels_[block_idx]->receive_data(
                reinterpret_cast<uint8_t*>(recv_addr), size);
            return;
        }

        if (use_rdma_ && rdma_software_load_channel_) {
            // std::cout << "[ECNAIVE RDMA] Load: Receiving " << size << " bytes via RDMA" << std::endl;
            rdma_software_load_channel_->receive_data(reinterpret_cast<uint8_t*>(recv_addr), size);
            return;
        }
        // Legacy k=2 path
        if (block_idx == 0 && conn_.is_ecnaive_load_recv_rank3_data1_connected()) {
            if (!recv_with_size_bool(conn_.get_ecnaive_load_recv_rank3_data1_socket(),
                                     reinterpret_cast<void*>(recv_addr), size)) {
                std::cerr << "EC-NAIVE: sw_recv_data(0) legacy recv failed" << std::endl;
            }
            return;
        }
        std::cerr << "EC-NAIVE: sw_recv_data(" << block_idx << ", " << size << "): no channel" << std::endl;
    }

    // Release helpers: Python can poll these to free buffers.
    pybind11::dict get_ft_timing_stats() const {
        double net_s = 0.0;
        double encode_s = 0.0;
        if (is_load_mode_) {
            const uint64_t net_ns =
                load_recv_total_ns_.load(std::memory_order_relaxed) +
                load_send_total_ns_.load(std::memory_order_relaxed);
            net_s = static_cast<double>(net_ns) / 1e9;
            encode_s = 0.0;  // RS decode timing is merged into Python network_encode_s
        } else {
            net_s = static_cast<double>(
                save_net_wall_span_ns_.load(std::memory_order_relaxed)) / 1e9;
            encode_s = static_cast<double>(
                save_encode_total_ns_.load(std::memory_order_relaxed)) / 1e9;
        }
        pybind11::dict result;
        result["net_s"] = net_s;
        result["encode_s"] = encode_s;
        return result;
    }

    // Data buffers (data1_addr) are released after send operations complete
    std::vector<uintptr_t> get_data_buffers_to_release() {
        std::vector<uintptr_t> buffers;
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        while (!data_buffers_to_release_.empty()) {
            buffers.push_back(data_buffers_to_release_.front());
            data_buffers_to_release_.pop();
        }
        return buffers;
    }
    
    // Parity buffers (parity0_addr, parity1_addr) are released after send operations complete
    std::vector<uintptr_t> get_parity_buffers_to_release() {
        std::vector<uintptr_t> buffers;
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        while (!parity_buffers_to_release_.empty()) {
            buffers.push_back(parity_buffers_to_release_.front());
            parity_buffers_to_release_.pop();
        }
        return buffers;
    }

    void reset_encoding_completion_flags() {
        // Reset generalized flags
        for (int i = 0; i < num_channels_; ++i) {
            send_completed_[i] = false;
            recv_completed_[i] = false;
            send_sentinel_received_[i] = false;
            recv_sentinel_received_[i] = false;
            {
                std::lock_guard<std::mutex> lock(send_mutexes_[i]);
                while (!send_queues_[i].empty()) send_queues_[i].pop();
            }
            {
                std::lock_guard<std::mutex> lock(recv_mutexes_[i]);
                while (!recv_queues_[i].empty()) recv_queues_[i].pop();
            }
        }
        // Also reset legacy flags for backward compat
        send_data1_completed_ = false;
        send_parity0_completed_ = false;
        send_parity1_completed_ = false;
        recv_parity1_completed_ = false;
        recv_parity0_completed_ = false;
        recv_data1_completed_ = false;
        send_data1_sentinel_received_ = false;
        send_parity0_sentinel_received_ = false;
        send_parity1_sentinel_received_ = false;
        recv_parity1_sentinel_received_ = false;
        recv_parity0_sentinel_received_ = false;
        recv_data1_sentinel_received_ = false;
        save_encode_total_ns_.store(0, std::memory_order_relaxed);
        save_encode_op_count_.store(0, std::memory_order_relaxed);
        save_net_wall_span_ns_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(save_net_wall_mu_);
            save_net_wall_have_any_ = false;
        }
        load_send_total_ns_.store(0, std::memory_order_relaxed);
        network_tasks_inflight_.store(0, std::memory_order_relaxed);
        // Clear legacy queues
        {
            std::lock_guard<std::mutex> lock(send_data1_mutex_);
            while (!send_data1_q_.empty()) send_data1_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(send_parity0_mutex_);
            while (!send_parity0_q_.empty()) send_parity0_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(send_parity1_mutex_);
            while (!send_parity1_q_.empty()) send_parity1_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(recv_parity1_mutex_);
            while (!recv_parity1_q_.empty()) recv_parity1_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(recv_parity0_mutex_);
            while (!recv_parity0_q_.empty()) recv_parity0_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(recv_data1_mutex_);
            while (!recv_data1_q_.empty()) recv_data1_q_.pop();
        }

    }

    void wait_for_encoding_completion() {
        // Check generalized workers
        int wait_count = 0;
        while (true) {
            bool all_done = true;
            for (int i = 0; i < num_channels_; ++i) {
                if (!send_completed_[i] || !recv_completed_[i]) { all_done = false; break; }
            }
            // Also check legacy flags if num_channels_ == 0 (backward compat)
            if (num_channels_ == 0) {
                all_done = send_data1_completed_ && send_parity0_completed_ && send_parity1_completed_ &&
                           recv_parity1_completed_ && recv_parity0_completed_ && recv_data1_completed_;
            }
            if (all_done) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
        }
    }

    // Wait until all queued/in-flight save-path send/recv tasks finish (no sentinels required).
    void wait_for_pending_network_tasks() {
        while (true) {
            if (network_tasks_inflight_.load(std::memory_order_acquire) == 0) {
                bool queues_empty = true;
                for (int i = 0; i < num_channels_; ++i) {
                    std::lock_guard<std::mutex> lk_send(send_mutexes_[i]);
                    if (!send_queues_[i].empty()) {
                        queues_empty = false;
                        break;
                    }
                    std::lock_guard<std::mutex> lk_recv(recv_mutexes_[i]);
                    if (!recv_queues_[i].empty()) {
                        queues_empty = false;
                        break;
                    }
                }
                if (queues_empty) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    void stop() {
        bool expected = false;
        if (!stop_.compare_exchange_strong(expected, true)) {
            return;  // already stopped
        }
        // Notify generalized workers
        for (int i = 0; i < num_channels_; ++i) {
            send_cvs_[i].notify_all();
            recv_cvs_[i].notify_all();
        }
        // Notify legacy workers
        send_data1_cv_.notify_all();
        send_parity0_cv_.notify_all();
        send_parity1_cv_.notify_all();
        recv_parity1_cv_.notify_all();
        recv_parity0_cv_.notify_all();
        recv_data1_cv_.notify_all();
        
        // Join generalized worker threads
        for (int i = 0; i < num_channels_; ++i) {
            if (send_threads_[i].joinable()) send_threads_[i].join();
            if (recv_threads_[i].joinable()) recv_threads_[i].join();
        }
        // Join legacy worker threads (backward compat)
        if (send_data1_thread_.joinable()) send_data1_thread_.join();
        if (send_parity0_thread_.joinable()) send_parity0_thread_.join();
        if (send_parity1_thread_.joinable()) send_parity1_thread_.join();
        if (recv_parity1_thread_.joinable()) recv_parity1_thread_.join();
        if (recv_parity0_thread_.joinable()) recv_parity0_thread_.join();
        if (recv_data1_thread_.joinable()) recv_data1_thread_.join();
        
        // Shut down EC RS pool if initialized
        ec_rs_pool_shutdown();

        conn_.cleanup();
    }

    // Load mode functions
    void set_load_mode(bool is_load, int failed_rank, int rank = -1, bool is_software_only = false) {
        is_load_mode_ = is_load;
        failed_rank_ = failed_rank;
        failed_rank_in_group_ = (failed_rank >= 0)
            ? (failed_rank % ECNAIVE_RANKS_PER_GROUP)
            : -1;
        if (rank >= 0) {
            rank_ = rank;
        }
        
        if (is_load) {
            load_recv_total_ns_.store(0, std::memory_order_relaxed);
            load_send_total_ns_.store(0, std::memory_order_relaxed);
        }
    }

    // Phase 1: set rank_, create load RDMA CQs (all ranks), rank2 bind+listen on 8 ports.
    // Software failure only: 1 port (rank3_data1), no workers; rank_ = rank_in_group
    void init_ecnaive_load_connections_software_only(
        int rank_in_group,
        const std::string& rank2_ip,
        uint16_t load_recv_rank3_data1_port
    ) {
        if (!is_load_mode_) {
            std::cerr << "EC-NAIVE: init_ecnaive_load_connections_software_only called but not in load mode" << std::endl;
            return;
        }
        rank_ = rank_in_group;
        if (rank_in_group == 2) {
            conn_.bind_listen_ecnaive_load_recv_rank3_data1(rank2_ip, load_recv_rank3_data1_port);
            conn_.accept_ecnaive_load_recv_rank3_data1();
        } else if (rank_in_group == 3) {
            conn_.init_ecnaive_load_send_rank3_data1(rank2_ip, load_recv_rank3_data1_port);
        }
        if (use_rdma_ && (rank_ == 2 || rank_ == 3)) {
            try {
                init_rdma_software_load_resources();
                init_rdma_software_load_channel();
            } catch (const std::exception& e) {
                std::cerr << "EC-NAIVE: Software-only load RDMA init failed: " << e.what() << std::endl;
                throw;
            }
        }
    }

    // ===== Generalized SW recovery connections (k-1 ports, any k >= 2) =====

    void init_ecnaive_load_sw_bind_listen(
        int rank_in_group, const std::string& receiver_ip,
        int num_blocks, const std::vector<uint16_t>& ports) {
        sw_recovery_num_blocks_ = num_blocks;
        sw_asio_acceptors_.resize(num_blocks);
        sw_asio_recv_sockets_.resize(num_blocks);
        sw_asio_send_sockets_.resize(num_blocks);
        rdma_sw_recovery_channels_.resize(num_blocks);
        if (rank_in_group != 2) return;  // only the receiver binds/listens

        for (int i = 0; i < num_blocks; ++i) {
            sw_asio_acceptors_[i] = std::make_unique<boost::asio::ip::tcp::acceptor>(
                sw_recovery_io_);
            sw_asio_acceptors_[i]->open(boost::asio::ip::tcp::v4());
            sw_asio_acceptors_[i]->set_option(
                boost::asio::ip::tcp::acceptor::reuse_address(true));
            sw_asio_acceptors_[i]->bind(
                boost::asio::ip::tcp::endpoint(
                    boost::asio::ip::address::from_string(receiver_ip), ports[i]));
            sw_asio_acceptors_[i]->listen();
        }
    }

    // Phase 2a: receiver (rig=2) accepts on all k-1 ports, sets up RDMA channels
    void init_ecnaive_load_sw_accept(
        int rank_in_group, int num_blocks) {
        if (rank_in_group != 2) return;
        if (num_blocks != sw_recovery_num_blocks_) {
            std::cerr << "EC-NAIVE: SW recovery port count mismatch "
                      << num_blocks << " vs " << sw_recovery_num_blocks_ << std::endl;
            return;
        }
        for (int i = 0; i < num_blocks; ++i) {
            sw_asio_recv_sockets_[i] = std::make_unique<boost::asio::ip::tcp::socket>(
                sw_recovery_io_);
            sw_asio_acceptors_[i]->accept(*sw_asio_recv_sockets_[i]);
        }

        // RDMA channels on accepted sockets
        if (use_rdma_) {
            for (int i = 0; i < num_blocks; ++i) {
                if (!sw_asio_recv_sockets_[i]) continue;
                int sock_fd = sw_asio_recv_sockets_[i]->native_handle();
                auto send_cq = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
                auto recv_cq = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
                rdma_sw_recovery_channels_[i] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, send_cq, recv_cq,
                    sock_fd, sock_fd, &rdma_registered_buffers_, &rdma_buffer_mutex_,
                    rank_in_group, 2);
                rdma_sw_recovery_channels_[i]->exchange_and_connect(true);
            }
        }
    }

    // Phase 2b: one sender connects to exactly one port (its block_idx)
    void init_ecnaive_load_sw_connect_one(
        int rank_in_group, const std::string& receiver_ip,
        int block_idx, uint16_t port) {
        if (rank_in_group == 2) return;  // receivers use init_ecnaive_load_sw_accept instead
        if (block_idx < 0 || block_idx >= sw_recovery_num_blocks_) {
            std::cerr << "EC-NAIVE: SW recovery connect_one: block_idx " << block_idx
                      << " out of range [0, " << sw_recovery_num_blocks_ << ")" << std::endl;
            return;
        }
        sw_asio_send_sockets_[block_idx] = std::make_unique<boost::asio::ip::tcp::socket>(
            sw_recovery_io_);
        int max_retries = 10;
        for (int retry = 0; retry < max_retries; ++retry) {
            try {
                sw_asio_send_sockets_[block_idx]->connect(
                    boost::asio::ip::tcp::endpoint(
                        boost::asio::ip::address::from_string(receiver_ip), port));
                break;
            } catch (const boost::system::system_error&) {
                if (retry == max_retries - 1) throw;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        // RDMA channel for this single connection
        if (use_rdma_) {
            if (!rdma_pd_) {
                if (!rdma_context_) {
                    if (ibv_fork_init() != 0) {
                        std::cerr << "[ECNAIVE RDMA] WARNING: ibv_fork_init() failed." << std::endl;
                    }
                    int num_devices;
                    ibv_device** device_list = ibv_get_device_list(&num_devices);
                    if (device_list && num_devices > 0) {
                        rdma_context_ = ibv_open_device(
                            find_rdma_device_by_ip(recv_ips_[0], device_list, num_devices));
                        ibv_free_device_list(device_list);
                    }
                }
                if (rdma_context_) {
                    rdma_pd_ = ibv_alloc_pd(rdma_context_);
                }
            }
            int sock_fd = sw_asio_send_sockets_[block_idx]->native_handle();
            auto send_cq = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            auto recv_cq = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            rdma_sw_recovery_channels_[block_idx] = std::make_unique<RdmaConnectionChannel>(
                rdma_context_, rdma_pd_, send_cq, recv_cq,
                sock_fd, sock_fd, &rdma_registered_buffers_, &rdma_buffer_mutex_,
                rank_in_group, 2);
            rdma_sw_recovery_channels_[block_idx]->exchange_and_connect(false);
        }
    }

private:
    std::atomic<bool> stop_;

    // ASIO connections for pipelines
    AsioConnectionManager conn_;
    
    // RDMA resources
    bool use_rdma_;
    ibv_context* rdma_context_;
    ibv_pd* rdma_pd_;
    // Legacy fixed-size RDMA arrays (referenced by named constants)
    std::vector<ibv_cq*> rdma_send_cqs_;           // size = num_channels
    std::vector<ibv_cq*> rdma_recv_cqs_;           // size = num_channels
    std::map<uintptr_t, RdmaBuffer> rdma_registered_buffers_;
    std::mutex rdma_buffer_mutex_;

    // Connection channels: send_channel[i] sends our data/parity block i to partner
    // recv_channel[i] receives peer's block for our recv slot i
    std::vector<std::unique_ptr<IConnectionChannel>> send_channels_;
    std::vector<std::unique_ptr<IConnectionChannel>> recv_channels_;

    // Software-only load RDMA: 1 channel (legacy k=2, kept for backward compat)
    ibv_cq* rdma_software_load_send_cq_;
    ibv_cq* rdma_software_load_recv_cq_;
    std::unique_ptr<RdmaConnectionChannel> rdma_software_load_channel_;
    // Generalized SW recovery: k-1 channels (one per non-local data block)
    boost::asio::io_context sw_recovery_io_;
    int sw_recovery_num_blocks_{0};
    std::vector<std::unique_ptr<boost::asio::ip::tcp::acceptor>> sw_asio_acceptors_;
    std::vector<std::unique_ptr<boost::asio::ip::tcp::socket>>   sw_asio_recv_sockets_;
    std::vector<std::unique_ptr<boost::asio::ip::tcp::socket>>   sw_asio_send_sockets_;
    std::vector<std::unique_ptr<RdmaConnectionChannel>> rdma_sw_recovery_channels_;

    // Save mode network config (generalized vectors)
    std::vector<std::string> send_ips_;
    std::vector<uint16_t> send_ports_;
    std::vector<std::string> recv_ips_;
    std::vector<uint16_t> recv_ports_;

    // RDMA exchange: dedicated TCP ports and fds
    std::vector<uint16_t> rdma_send_ports_;
    std::vector<uint16_t> rdma_recv_ports_;
    std::vector<int> rdma_send_fds_;
    std::vector<int> rdma_recv_fds_;

    // EC encoding parameters
    int k_;    // number of data blocks
    int rows_; // number of parity blocks (always 2)
    int n_;    // ranks per group = k + 2
    int num_channels_; // n-1 = k+1 send channels = k+1 recv channels
    unsigned char* a_mat_;    // RS matrix (k * m, where m = k + rows)
    unsigned char* g_tbls_;   // EC encoding tables (32 * k * rows)

    // Save mode pipelines: num_channels sends + num_channels receives
    std::vector<std::queue<SendTask>> send_queues_;
    std::deque<std::mutex> send_mutexes_;
    std::deque<std::condition_variable> send_cvs_;
    std::vector<std::queue<RecvTask>> recv_queues_;
    std::deque<std::mutex> recv_mutexes_;
    std::deque<std::condition_variable> recv_cvs_;

    // Separate release queues for data and parity buffers
    std::queue<uintptr_t> data_buffers_to_release_;
    std::queue<uintptr_t> parity_buffers_to_release_;
    std::mutex release_queue_mutex_;
    std::atomic<int> network_tasks_inflight_{0};

    // Completion flags
    std::deque<std::atomic<bool>> send_completed_;
    std::deque<std::atomic<bool>> recv_completed_;
    // Sentinel received flags
    std::deque<std::atomic<bool>> send_sentinel_received_;
    std::deque<std::atomic<bool>> recv_sentinel_received_;

    // Worker threads
    std::deque<std::thread> send_threads_;
    std::deque<std::thread> recv_threads_;

    // Legacy backward-compat members (kept so old named worker functions compile)
    std::queue<SendTask> send_data1_q_;
    std::mutex send_data1_mutex_;
    std::condition_variable send_data1_cv_;
    std::queue<SendTask> send_parity0_q_;
    std::mutex send_parity0_mutex_;
    std::condition_variable send_parity0_cv_;
    std::queue<SendTask> send_parity1_q_;
    std::mutex send_parity1_mutex_;
    std::condition_variable send_parity1_cv_;
    std::queue<RecvTask> recv_parity1_q_;
    std::mutex recv_parity1_mutex_;
    std::condition_variable recv_parity1_cv_;
    std::queue<RecvTask> recv_parity0_q_;
    std::mutex recv_parity0_mutex_;
    std::condition_variable recv_parity0_cv_;
    std::queue<RecvTask> recv_data1_q_;
    std::mutex recv_data1_mutex_;
    std::condition_variable recv_data1_cv_;
    std::atomic<bool> send_data1_completed_{false};
    std::atomic<bool> send_parity0_completed_{false};
    std::atomic<bool> send_parity1_completed_{false};
    std::atomic<bool> recv_parity1_completed_{false};
    std::atomic<bool> recv_parity0_completed_{false};
    std::atomic<bool> recv_data1_completed_{false};
    std::atomic<bool> send_data1_sentinel_received_{false};
    std::atomic<bool> send_parity0_sentinel_received_{false};
    std::atomic<bool> send_parity1_sentinel_received_{false};
    std::atomic<bool> recv_parity1_sentinel_received_{false};
    std::atomic<bool> recv_parity0_sentinel_received_{false};
    std::atomic<bool> recv_data1_sentinel_received_{false};
    std::thread send_data1_thread_;
    std::thread send_parity0_thread_;
    std::thread send_parity1_thread_;
    std::thread recv_parity1_thread_;
    std::thread recv_parity0_thread_;
    std::thread recv_data1_thread_;
    std::unique_ptr<IConnectionChannel> send_data1_channel_;
    std::unique_ptr<IConnectionChannel> send_parity0_channel_;
    std::unique_ptr<IConnectionChannel> send_parity1_channel_;
    std::unique_ptr<IConnectionChannel> recv_parity1_channel_;
    std::unique_ptr<IConnectionChannel> recv_parity0_channel_;
    std::unique_ptr<IConnectionChannel> recv_data1_channel_;
    // RDMA save CQs are stored in rdma_send_cqs_ / rdma_recv_cqs_ (sized to 2*num_channels_)
    
    // Load mode flags
    std::atomic<bool> is_load_mode_{false};
    int failed_rank_{-1};
    int failed_rank_in_group_{-1};  // failed rank within 4-rank group (for multi-group support)
    int rank_;  // Current rank_in_group (0..3) for load mode, set in init_ecnaive_load_connections
    int rank_in_group_;  // rank_in_group for save mode (0..3), set in constructor

    // EC RS pthread pool (encode + decode, same pattern as FRCheck rs_pool)
    static constexpr int kEcRsPoolSize = 16;
    std::array<pthread_t, kEcRsPoolSize> ec_rs_pool_threads_{};
    std::array<EcRsPoolWorkerCtx, kEcRsPoolSize> ec_rs_pool_ctx_{};
    std::array<int, kEcRsPoolSize> ec_rs_pool_cpus_{};
    std::atomic<bool> ec_rs_pool_inited_{false};
    std::atomic<bool> ec_rs_pool_stop_{false};
    std::mutex ec_rs_pool_mutex_;
    std::condition_variable ec_rs_pool_worker_cv_;
    std::condition_variable ec_rs_pool_coordinator_cv_;
    std::atomic<uint64_t> ec_rs_pool_epoch_{0};
    std::array<uint64_t, kEcRsPoolSize> ec_rs_pool_last_epoch_{};
    std::atomic<int> ec_rs_pool_remaining_{0};
    EcRsJob ec_rs_pool_shared_job_{};
    // Per-worker decode table storage: each worker has its own tables
    // to avoid concurrent access. Tables are initialized before dispatch.
    bool ec_rs_pool_have_tbls_{false};
    int ec_decode_tbls_k_{0};
    int ec_decode_tbls_m_{0};
    std::vector<unsigned char> ec_decode_tbls_;  // stable storage for decode tables

    std::atomic<uint64_t> save_encode_total_ns_{0};
    std::atomic<size_t> save_encode_op_count_{0};
    std::atomic<uint64_t> load_recv_total_ns_{0};
    std::atomic<uint64_t> load_send_total_ns_{0};

    // Save net: wall from earliest net op start to latest net op end (overlapping workers).
    std::mutex save_net_wall_mu_;
    bool save_net_wall_have_any_{false};
    std::chrono::steady_clock::time_point save_net_wall_first_{};
    std::chrono::steady_clock::time_point save_net_wall_last_{};
    std::atomic<uint64_t> save_net_wall_span_ns_{0};

    void record_save_encode_op_(uint64_t ns) {
        save_encode_total_ns_.fetch_add(ns, std::memory_order_relaxed);
        save_encode_op_count_.fetch_add(1, std::memory_order_relaxed);
    }

    void touch_save_net_wall_(
        std::chrono::steady_clock::time_point t0,
        std::chrono::steady_clock::time_point t1) {
        std::lock_guard<std::mutex> lk(save_net_wall_mu_);
        if (!save_net_wall_have_any_) {
            save_net_wall_first_ = t0;
            save_net_wall_last_ = t1;
            save_net_wall_have_any_ = true;
        } else {
            if (t0 < save_net_wall_first_) {
                save_net_wall_first_ = t0;
            }
            if (t1 > save_net_wall_last_) {
                save_net_wall_last_ = t1;
            }
        }
        const uint64_t span_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                save_net_wall_last_ - save_net_wall_first_).count());
        save_net_wall_span_ns_.store(span_ns, std::memory_order_relaxed);
    }

    struct SaveEncodeScopeTimer {
        ECNaiveNative* owner;
        std::chrono::steady_clock::time_point t0;
        explicit SaveEncodeScopeTimer(ECNaiveNative* o)
            : owner(o), t0(std::chrono::steady_clock::now()) {}
        ~SaveEncodeScopeTimer() {
            if (owner != nullptr && !owner->is_load_mode_) {
                const auto t1 = std::chrono::steady_clock::now();
                const uint64_t ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                owner->record_save_encode_op_(ns);
            }
        }
    };

    struct SaveNetScopeTimer {
        ECNaiveNative* owner;
        std::chrono::steady_clock::time_point t0;
        explicit SaveNetScopeTimer(ECNaiveNative* o)
            : owner(o), t0(std::chrono::steady_clock::now()) {}
        ~SaveNetScopeTimer() {
            if (owner != nullptr && !owner->is_load_mode_) {
                const auto t1 = std::chrono::steady_clock::now();
                owner->touch_save_net_wall_(t0, t1);
            }
        }
    };

    struct LoadNetScopeTimer {
        ECNaiveNative* owner;
        std::chrono::steady_clock::time_point t0;
        explicit LoadNetScopeTimer(ECNaiveNative* o)
            : owner(o), t0(std::chrono::steady_clock::now()) {}
        ~LoadNetScopeTimer() {
            if (owner != nullptr && owner->is_load_mode_) {
                const auto t1 = std::chrono::steady_clock::now();
                const uint64_t ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                owner->load_send_total_ns_.fetch_add(ns, std::memory_order_relaxed);
            }
        }
    };

    struct LoadRecvNetScopeTimer {
        ECNaiveNative* owner;
        std::chrono::steady_clock::time_point t0;
        explicit LoadRecvNetScopeTimer(ECNaiveNative* o)
            : owner(o), t0(std::chrono::steady_clock::now()) {}
        ~LoadRecvNetScopeTimer() {
            if (owner != nullptr && owner->is_load_mode_) {
                const auto t1 = std::chrono::steady_clock::now();
                const uint64_t ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                owner->load_recv_total_ns_.fetch_add(ns, std::memory_order_relaxed);
            }
        }
    };

    // EC-NAIVE load mode sentinel flags
    // EC encoding initialization
    void init_ec_encoding() {
        int m = k_ + rows_;  // m = 2 + 2 = 4
        
        // Allocate RS matrix (k * m = 2 * 4)
        a_mat_ = (unsigned char*)malloc((size_t)k_ * (size_t)m);
        if (a_mat_ == nullptr) {
            std::cerr << "ECNAIVE: Failed to allocate EC encoding matrix" << std::endl;
            throw std::runtime_error("ECNAIVE: Failed to allocate EC encoding matrix");
        }
        
        // Generate RS matrix
        gf_gen_rs_matrix(a_mat_, m, k_);
        
        // Allocate encoding tables: 32 * k * rows = 32 * 2 * 2
        size_t gtbls_size = 32 * (size_t)k_ * (size_t)rows_;
        void* tmp = nullptr;
        if (posix_memalign(&tmp, 32, gtbls_size) != 0) tmp = nullptr;
        if (tmp == nullptr) tmp = malloc(gtbls_size);
        g_tbls_ = reinterpret_cast<unsigned char*>(tmp);
        if (g_tbls_ == nullptr) {
            std::cerr << "ECNAIVE: Failed to allocate EC encoding tables" << std::endl;
            free(a_mat_);
            a_mat_ = nullptr;
            throw std::runtime_error("ECNAIVE: Failed to allocate EC encoding tables");
        }
        
        // Initialize tables using isa-l — only pass the parity rows (rows k..k+rows-1),
        // not the full Vandermonde matrix (which includes data rows at positions 0..k-1).
        // ec_init_tables expects a k×rows matrix; a_mat_ is k×(k+rows), so offset by k*k.
        ec_init_tables(k_, rows_, a_mat_ + k_ * k_, g_tbls_);
    }
    
    // EC encoding function: encode k_ data blocks to rows_ (2) parity blocks
    void encode_ec_blocks(const std::vector<uintptr_t>& data_addrs,
                          uintptr_t parity0_addr, uintptr_t parity1_addr,
                          size_t size) {
        SaveEncodeScopeTimer encode_timer(this);
        if (g_tbls_ == nullptr || a_mat_ == nullptr) {
            std::cerr << "ECNAIVE: ERROR: EC encoding tables not initialized!" << std::endl;
            throw std::runtime_error("ECNAIVE: EC encoding tables not initialized");
        }
        if (static_cast<int>(data_addrs.size()) != k_) {
            throw std::runtime_error("ECNAIVE: data_addrs size must equal k_");
        }

        // Use 16-pthread pool when available (aligns with FRCheck rs_pool / ECLATIN xor_pool)
        if (ec_rs_pool_inited_.load(std::memory_order_acquire)) {
            ec_rs_pool_run_encode(data_addrs, parity0_addr, parity1_addr, size);
            return;
        }

        // Fallback: inline ISA-L (single-threaded)
        std::vector<unsigned char*> srcs(k_);
        for (int i = 0; i < k_; ++i) {
            srcs[i] = reinterpret_cast<unsigned char*>(data_addrs[i]);
        }
        unsigned char* dests[2];
        dests[0] = reinterpret_cast<unsigned char*>(parity0_addr);
        dests[1] = reinterpret_cast<unsigned char*>(parity1_addr);

        ec_encode_data((int)size, k_, rows_, g_tbls_, srcs.data(), dests);
    }
    
    // RDMA initialization
    void init_rdma_resources() {

        // Must be called before any RDMA memory registration when the process may fork (e.g. multiprocessing
        // write workers). Without this, child processes get EFAULT (Bad address) when accessing RDMA-registered
        // memory or when writing from buffers that were registered in the parent.
        if (ibv_fork_init() != 0) {
            std::cerr << "[ECNAIVE RDMA] WARNING: ibv_fork_init() failed. Forked processes may get Bad address."
                      << std::endl;
        }

        // Get device list
        int num_devices;
        ibv_device** device_list = ibv_get_device_list(&num_devices);
        if (!device_list || num_devices == 0) {
            throw std::runtime_error("No RDMA devices found");
        }
        
        // Select RDMA device by local IP (recv_ips_[0])
        rdma_context_ = ibv_open_device(
            find_rdma_device_by_ip(recv_ips_[0], device_list, num_devices));
        if (!rdma_context_) {
            ibv_free_device_list(device_list);
            throw std::runtime_error("Failed to open RDMA device");
        }
        
        ibv_free_device_list(device_list);
        
        // Allocate protection domain
        rdma_pd_ = ibv_alloc_pd(rdma_context_);
        if (!rdma_pd_) {
            throw std::runtime_error("Failed to allocate protection domain");
        }
        
        // Create 2 * num_channels_ CQ pairs: one send_cq + one recv_cq per save channel
        int total_cqs = 2 * num_channels_;
        rdma_send_cqs_.resize(total_cqs, nullptr);
        rdma_recv_cqs_.resize(total_cqs, nullptr);
        for (int i = 0; i < total_cqs; ++i) {
            rdma_send_cqs_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            rdma_recv_cqs_[i] = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            if (!rdma_send_cqs_[i] || !rdma_recv_cqs_[i]) {
                for (int j = 0; j < i; ++j) {
                    if (rdma_send_cqs_[j]) { ibv_destroy_cq(rdma_send_cqs_[j]); rdma_send_cqs_[j] = nullptr; }
                    if (rdma_recv_cqs_[j]) { ibv_destroy_cq(rdma_recv_cqs_[j]); rdma_recv_cqs_[j] = nullptr; }
                }
                throw std::runtime_error("Failed to create completion queues for RDMA channel " + std::to_string(i));
            }
        }

    }

    // Software-only load RDMA: 1 CQ pair and 1 channel (rank3_data1). Call after init_ecnaive_load_connections_software_only.
    void init_rdma_software_load_resources() {
        if (!use_rdma_) return;
        if (rdma_software_load_send_cq_ || rdma_software_load_recv_cq_) return;  // already inited
        if (!rdma_context_) {
            if (ibv_fork_init() != 0) {
                std::cerr << "[ECNAIVE RDMA] WARNING: ibv_fork_init() failed." << std::endl;
            }
            int num_devices;
            ibv_device** device_list = ibv_get_device_list(&num_devices);
            if (!device_list || num_devices == 0) {
                throw std::runtime_error("No RDMA devices found");
            }
            rdma_context_ = ibv_open_device(
                find_rdma_device_by_ip(recv_ips_[0], device_list, num_devices));
            if (!rdma_context_) {
                ibv_free_device_list(device_list);
                throw std::runtime_error("Failed to open RDMA device");
            }
            ibv_free_device_list(device_list);
        }
        if (!rdma_pd_) {
            rdma_pd_ = ibv_alloc_pd(rdma_context_);
            if (!rdma_pd_) {
                throw std::runtime_error("Failed to allocate protection domain");
            }
        }
        rdma_software_load_send_cq_ = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
        rdma_software_load_recv_cq_ = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
        if (!rdma_software_load_send_cq_ || !rdma_software_load_recv_cq_) {
            if (rdma_software_load_send_cq_) {
                ibv_destroy_cq(rdma_software_load_send_cq_);
                rdma_software_load_send_cq_ = nullptr;
            }
            if (rdma_software_load_recv_cq_) {
                ibv_destroy_cq(rdma_software_load_recv_cq_);
                rdma_software_load_recv_cq_ = nullptr;
            }
            throw std::runtime_error("Failed to create software-only load completion queues");
        }
    }

    void init_rdma_software_load_channel() {
        if (!use_rdma_ || !rdma_pd_ || !rdma_software_load_send_cq_ || !rdma_software_load_recv_cq_) return;
        if (rdma_software_load_channel_) return;  // already inited
        int rank_for_log = (rank_ >= 0) ? rank_ : 0;
        try {
            if (rank_ == 2) {
                int sock = conn_.get_ecnaive_load_recv_rank3_data1_socket().native_handle();
                rdma_software_load_channel_ = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_software_load_send_cq_, rdma_software_load_recv_cq_,
                    sock, sock, &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 3);
                rdma_software_load_channel_->exchange_and_connect(false);
            } else if (rank_ == 3) {
                int sock = conn_.get_ecnaive_load_send_rank3_data1_socket().native_handle();
                rdma_software_load_channel_ = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_, rdma_software_load_send_cq_, rdma_software_load_recv_cq_,
                    sock, sock, &rdma_registered_buffers_, &rdma_buffer_mutex_, rank_for_log, 2);
                rdma_software_load_channel_->exchange_and_connect(true);
            }
        } catch (const std::exception& e) {
            std::cerr << "[ECNAIVE RDMA] Failed to init software-only load channel: " << e.what() << std::endl;
            rdma_software_load_channel_.reset();
            throw;
        }
    }
    
    void cleanup_rdma_resources() {
        if (!use_rdma_) {
            return;
        }
        
        
        // Destroy RDMA channels first (each channel destroys its QP)
        send_data1_channel_.reset();
        send_parity0_channel_.reset();
        send_parity1_channel_.reset();
        recv_parity1_channel_.reset();
        recv_parity0_channel_.reset();
        recv_data1_channel_.reset();
        close_rdma_exchange_sockets();
        rdma_software_load_channel_.reset();
        if (rdma_software_load_send_cq_) {
            ibv_destroy_cq(rdma_software_load_send_cq_);
            rdma_software_load_send_cq_ = nullptr;
        }
        if (rdma_software_load_recv_cq_) {
            ibv_destroy_cq(rdma_software_load_recv_cq_);
            rdma_software_load_recv_cq_ = nullptr;
        }
        
        // Unregister all buffers
        {
            std::lock_guard<std::mutex> lock(rdma_buffer_mutex_);
            for (auto& [addr, buf] : rdma_registered_buffers_) {
                if (buf.mr) {
                    ibv_dereg_mr(buf.mr);
                }
            }
            rdma_registered_buffers_.clear();
        }
        
        // Destroy save CQs from vectors (sized to 2 * num_channels_)
        for (auto& cq : rdma_send_cqs_) {
            if (cq) { ibv_destroy_cq(cq); cq = nullptr; }
        }
        for (auto& cq : rdma_recv_cqs_) {
            if (cq) { ibv_destroy_cq(cq); cq = nullptr; }
        }
        rdma_send_cqs_.clear();
        rdma_recv_cqs_.clear();
        
        // Dealloc PD
        if (rdma_pd_) {
            ibv_dealloc_pd(rdma_pd_);
            rdma_pd_ = nullptr;
        }
        
        // Close device
        if (rdma_context_) {
            ibv_close_device(rdma_context_);
            rdma_context_ = nullptr;
        }
        
    }

    void start_threads() {
        ec_rs_pool_init();
        for (int i = 0; i < num_channels_; ++i) {
            send_threads_[i] = std::thread(&ECNaiveNative::send_worker, this, i);
            recv_threads_[i] = std::thread(&ECNaiveNative::recv_worker, this, i);
        }
    }

    void init_connections() {

        // Start acceptors in separate thread (parallel listen/accept for all recv channels)
        std::thread recv_init_thread([this]() {
            conn_.init_recv_channels(recv_ips_, recv_ports_);
        });

        // Small delay to ensure acceptors are listening
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Connect all send sockets (blocking)
        conn_.init_send_channels(send_ips_, send_ports_);

        recv_init_thread.join();
        conn_.wait_for_connections();
    }

    // Establish 6 dedicated raw TCP sockets for RDMA RdmaConnInfo exchange only (like Gemini).
    // Uses blocking socket()/bind()/listen()/accept() and connect() - no ASIO.
    void init_rdma_exchange_sockets() {
        if (!use_rdma_) {
            return;
        }
        // Accept side: same order as ASIO recv roles (recv_parity1, recv_parity0, recv_data1).
        std::thread accept_thread([this]() {
            auto do_listen_accept = [this](const std::string& ip, uint16_t port, int& out_fd, const std::string& name) {
                int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
                if (listen_fd < 0) {
                    throw std::runtime_error(std::string("[ECNAIVE RDMA] socket() failed for ") + name + ": " + std::strerror(errno));
                }
                int opt = 1;
                setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
                sockaddr_in addr = {};
                addr.sin_family = AF_INET;
                addr.sin_port = htons(port);
                if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
                    close(listen_fd);
                    throw std::runtime_error(std::string("[ECNAIVE RDMA] inet_pton failed for ") + name);
                }
                if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
                    close(listen_fd);
                    throw std::runtime_error(std::string("[ECNAIVE RDMA] bind failed for ") + name + ": " + std::strerror(errno));
                }
                if (listen(listen_fd, 1) < 0) {
                    close(listen_fd);
                    throw std::runtime_error(std::string("[ECNAIVE RDMA] listen failed for ") + name + ": " + std::strerror(errno));
                }
                sockaddr_in client_addr = {};
                socklen_t client_len = sizeof(client_addr);
                out_fd = accept(listen_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
                close(listen_fd);
                if (out_fd < 0) {
                    throw std::runtime_error(std::string("[ECNAIVE RDMA] accept failed for ") + name + ": " + std::strerror(errno));
                }
            };
            for (int i = 0; i < num_channels_; ++i) {
                do_listen_accept(recv_ips_[i], rdma_recv_ports_[i], rdma_recv_fds_[i],
                                 "rdma_recv_" + std::to_string(i));
            }
        });
        auto do_connect = [this](const std::string& ip, uint16_t port, int& out_fd, const std::string& name) {
            out_fd = socket(AF_INET, SOCK_STREAM, 0);
            if (out_fd < 0) {
                throw std::runtime_error(std::string("[ECNAIVE RDMA] socket() failed for ") + name + ": " + std::strerror(errno));
            }
            sockaddr_in server_addr = {};
            server_addr.sin_family = AF_INET;
            server_addr.sin_port = htons(port);
            if (inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr) <= 0) {
                close(out_fd);
                out_fd = -1;
                throw std::runtime_error(std::string("[ECNAIVE RDMA] inet_pton failed for ") + name);
            }
            const int max_retries = 100;
            for (int attempt = 0; attempt < max_retries; ++attempt) {
                if (connect(out_fd, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) == 0) {
                    break;
                }
                if (attempt == max_retries - 1) {
                    close(out_fd);
                    out_fd = -1;
                    throw std::runtime_error(std::string("[ECNAIVE RDMA] connect failed for ") + name + ": " + std::strerror(errno));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        };
        try {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            for (int i = 0; i < num_channels_; ++i) {
                do_connect(send_ips_[i], rdma_send_ports_[i], rdma_send_fds_[i],
                           "rdma_send_" + std::to_string(i));
            }
        } catch (...) {
            // Avoid std::terminate: a joinable std::thread must be joined before destruction.
            if (accept_thread.joinable()) {
                accept_thread.join();
            }
            throw;
        }
        if (accept_thread.joinable()) {
            accept_thread.join();
        }
    }

    void close_rdma_exchange_sockets() {
        auto close_fd = [](int& fd, const std::string& name) {
            if (fd >= 0) {
                close(fd);
                fd = -1;
            }
        };
        // Close generalized RDMA sockets
        for (size_t i = 0; i < rdma_send_fds_.size(); ++i) {
            close_fd(rdma_send_fds_[i], ("rdma_send_" + std::to_string(i)).c_str());
        }
        for (size_t i = 0; i < rdma_recv_fds_.size(); ++i) {
            close_fd(rdma_recv_fds_[i], ("rdma_recv_" + std::to_string(i)).c_str());
        }
    }

    // Create RDMA channels and connect QPs for save (after ASIO connections are up).
    // Create RDMA save channels using ASIO socket fds for control
    // (same pattern as eclatin init_rdma_save_channels).
    void init_rdma_save_channels() {
        if (!use_rdma_ || !rdma_pd_) return;

        try {

            // Create all channels first (no exchange yet).
            // Send channels: rdma_send_fds_[i] is the "connect" side.
            for (int i = 0; i < num_channels_; ++i) {
                send_channels_[i] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_,
                    rdma_send_cqs_[i], rdma_recv_cqs_[i],
                    rdma_send_fds_[i], rdma_send_fds_[i],
                    &rdma_registered_buffers_, &rdma_buffer_mutex_,
                    rank_in_group_, 0);
            }
            // Recv channels: rdma_recv_fds_[i] is the "accept" side.
            for (int i = 0; i < num_channels_; ++i) {
                recv_channels_[i] = std::make_unique<RdmaConnectionChannel>(
                    rdma_context_, rdma_pd_,
                    rdma_send_cqs_[i + num_channels_], rdma_recv_cqs_[i + num_channels_],
                    rdma_recv_fds_[i], rdma_recv_fds_[i],
                    &rdma_registered_buffers_, &rdma_buffer_mutex_,
                    rank_in_group_, 0);
            }

            // Run all 2*num_channels_ exchanges concurrently in threads.
            // Each send channel does exchange(true), each recv channel
            // does exchange(false). Threads avoid ordering deadlocks.
            {
                std::vector<std::thread> ex_threads;
                ex_threads.reserve(2 * num_channels_);
                for (int i = 0; i < num_channels_; ++i) {
                    ex_threads.emplace_back([this, i]() {
                        send_channels_[i]->exchange_and_connect(true);
                    });
                    ex_threads.emplace_back([this, i]() {
                        recv_channels_[i]->exchange_and_connect(false);
                    });
                }
                for (auto& t : ex_threads) t.join();
            }


        } catch (const std::exception& e) {
            std::cerr << "[ECNAIVE RDMA] Save channel init failed: " << e.what()
                      << " — falling back to ASIO" << std::endl;
            use_rdma_ = false;
            for (int i = 0; i < num_channels_; ++i) {
                send_channels_[i].reset();
                recv_channels_[i].reset();
            }
        }
    }

    // Save mode workers: 3 sends + 3 receives
    void recv_parity1_worker() {
        while (!stop_) {
            RecvTask task;
            {
                std::unique_lock<std::mutex> lk(recv_parity1_mutex_);
                recv_parity1_cv_.wait(lk, [this] { return stop_ || !recv_parity1_q_.empty(); });
                if (stop_) break;
                task = recv_parity1_q_.front();
                recv_parity1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                recv_parity1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(recv_parity1_mutex_);
                    if (recv_parity1_q_.empty()) {
                        recv_parity1_completed_ = true;
                        recv_parity1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }

            // Ensure connection is ready
            if (!conn_.is_recv_parity1_connected()) {
                std::cerr << "ECNAIVE: ERROR: recv_parity1 socket not connected!" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_parity1 socket not connected");
            }

            if (use_rdma_ && recv_parity1_channel_) {
                // std::cout << "[ECNAIVE RDMA] Recv_Parity1: Receiving " << task.size << " bytes via RDMA" << std::endl;
                recv_parity1_channel_->receive_data(reinterpret_cast<uint8_t*>(task.addr), task.size);
            } else {
                // std::cout << "[ECNAIVE ASIO] Recv_Parity1: Receiving " << task.size << " bytes via ASIO" << std::endl;
                if (!recv_with_size_bool(
                        conn_.get_recv_parity1_socket(),
                        reinterpret_cast<void*>(task.addr),
                        task.size)) {
                    std::cerr << "ECNAIVE: recv_parity1_with_size_bool returned false" << std::endl;
                    throw std::runtime_error("ECNAIVE: recv_parity1_with_size_bool returned false");
                }
            }

            if (recv_parity1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_parity1_mutex_);
                if (recv_parity1_q_.empty()) {
                    recv_parity1_completed_ = true;
                    recv_parity1_sentinel_received_ = false;
                }
            }
        }
    }

    void recv_parity0_worker() {
        while (!stop_) {
            RecvTask task;
            {
                std::unique_lock<std::mutex> lk(recv_parity0_mutex_);
                recv_parity0_cv_.wait(lk, [this] { return stop_ || !recv_parity0_q_.empty(); });
                if (stop_) break;
                task = recv_parity0_q_.front();
                recv_parity0_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                recv_parity0_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(recv_parity0_mutex_);
                    if (recv_parity0_q_.empty()) {
                        recv_parity0_completed_ = true;
                        recv_parity0_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }

            // Ensure connection is ready
            if (!conn_.is_recv_parity0_connected()) {
                std::cerr << "ECNAIVE: ERROR: recv_parity0 socket not connected!" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_parity0 socket not connected");
            }

            if (use_rdma_ && recv_parity0_channel_) {
                // std::cout << "[ECNAIVE RDMA] Recv_Parity0: Receiving " << task.size << " bytes via RDMA" << std::endl;
                recv_parity0_channel_->receive_data(reinterpret_cast<uint8_t*>(task.addr), task.size);
            } else {
                if (!recv_with_size_bool(
                        conn_.get_recv_parity0_socket(),
                        reinterpret_cast<void*>(task.addr),
                        task.size)) {
                    std::cerr << "ECNAIVE: recv_parity0_with_size_bool returned false" << std::endl;
                    throw std::runtime_error("ECNAIVE: recv_parity0_with_size_bool returned false");
                }
            }

            if (recv_parity0_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_parity0_mutex_);
                if (recv_parity0_q_.empty()) {
                    recv_parity0_completed_ = true;
                    recv_parity0_sentinel_received_ = false;
                }
            }
        }
    }

    void recv_data1_worker() {
        while (!stop_) {
            RecvTask task;
            {
                std::unique_lock<std::mutex> lk(recv_data1_mutex_);
                recv_data1_cv_.wait(lk, [this] { return stop_ || !recv_data1_q_.empty(); });
                if (stop_) break;
                task = recv_data1_q_.front();
                recv_data1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                recv_data1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(recv_data1_mutex_);
                    if (recv_data1_q_.empty()) {
                        recv_data1_completed_ = true;
                        recv_data1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }

            // Ensure connection is ready
            if (!conn_.is_recv_data1_connected()) {
                std::cerr << "ECNAIVE: ERROR: recv_data1 socket not connected!" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_data1 socket not connected");
            }

            if (use_rdma_ && recv_data1_channel_) {
                recv_data1_channel_->receive_data(reinterpret_cast<uint8_t*>(task.addr), task.size);
            } else {
                if (!recv_with_size_bool(
                        conn_.get_recv_data1_socket(),
                        reinterpret_cast<void*>(task.addr),
                        task.size)) {
                    std::cerr << "ECNAIVE: recv_data1_with_size_bool returned false" << std::endl;
                    throw std::runtime_error("ECNAIVE: recv_data1_with_size_bool returned false");
                }
            }

            if (recv_data1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_data1_mutex_);
                if (recv_data1_q_.empty()) {
                    recv_data1_completed_ = true;
                    recv_data1_sentinel_received_ = false;
                }
            }
        }
    }

    void send_data1_worker() {
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(send_data1_mutex_);
                send_data1_cv_.wait(lk, [this] { return stop_ || !send_data1_q_.empty(); });
                if (stop_) break;
                task = send_data1_q_.front();
                send_data1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                send_data1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(send_data1_mutex_);
                    if (send_data1_q_.empty()) {
                        send_data1_completed_ = true;
                        send_data1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_send_data1_connected()) {
                if (use_rdma_ && send_data1_channel_) {
                    // std::cout << "[ECNAIVE RDMA] Send_Data1: Sending " << task.size << " bytes via RDMA" << std::endl;
                    send_data1_channel_->send_data(reinterpret_cast<const uint8_t*>(task.addr), task.size);
                } else {
                    send_with_size(conn_.get_send_data1_socket(), task.addr, task.size);
                }
            }
            // Release data buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                data_buffers_to_release_.push(task.addr);
            }

            if (send_data1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(send_data1_mutex_);
                if (send_data1_q_.empty()) {
                    send_data1_completed_ = true;
                    send_data1_sentinel_received_ = false;
                }
            }
        }
    }

    void send_parity0_worker() {
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(send_parity0_mutex_);
                send_parity0_cv_.wait(lk, [this] { return stop_ || !send_parity0_q_.empty(); });
                if (stop_) break;
                task = send_parity0_q_.front();
                send_parity0_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                send_parity0_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(send_parity0_mutex_);
                    if (send_parity0_q_.empty()) {
                        send_parity0_completed_ = true;
                        send_parity0_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_send_parity0_connected()) {
                if (use_rdma_ && send_parity0_channel_) {
                    // std::cout << "[ECNAIVE RDMA] Send_Parity0: Sending " << task.size << " bytes via RDMA" << std::endl;
                    send_parity0_channel_->send_data(reinterpret_cast<const uint8_t*>(task.addr), task.size);
                } else {
                    send_with_size(conn_.get_send_parity0_socket(), task.addr, task.size);
                }
            }
            // Release parity buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                parity_buffers_to_release_.push(task.addr);
            }

            if (send_parity0_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(send_parity0_mutex_);
                if (send_parity0_q_.empty()) {
                    send_parity0_completed_ = true;
                    send_parity0_sentinel_received_ = false;
                }
            }
        }
    }

    void send_parity1_worker() {
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(send_parity1_mutex_);
                send_parity1_cv_.wait(lk, [this] { return stop_ || !send_parity1_q_.empty(); });
                if (stop_) break;
                task = send_parity1_q_.front();
                send_parity1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                send_parity1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(send_parity1_mutex_);
                    if (send_parity1_q_.empty()) {
                        send_parity1_completed_ = true;
                        send_parity1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_send_parity1_connected()) {
                if (use_rdma_ && send_parity1_channel_) {
                    // std::cout << "[ECNAIVE RDMA] Send_Parity1: Sending " << task.size << " bytes via RDMA" << std::endl;
                    send_parity1_channel_->send_data(reinterpret_cast<const uint8_t*>(task.addr), task.size);
                } else {
                    send_with_size(conn_.get_send_parity1_socket(), task.addr, task.size);
                }
            }
            // Release parity buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                parity_buffers_to_release_.push(task.addr);
            }

            if (send_parity1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(send_parity1_mutex_);
                if (send_parity1_q_.empty()) {
                    send_parity1_completed_ = true;
                    send_parity1_sentinel_received_ = false;
                }
            }
        }
    }

    // ========== Generalized save workers (for k+2 scheme) ==========

    void send_worker(int idx) {
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(send_mutexes_[idx]);
                send_cvs_[idx].wait(lk, [this, idx] {
                    return stop_ || !send_queues_[idx].empty();
                });
                if (stop_) break;
                task = send_queues_[idx].front();
                send_queues_[idx].pop();
            }
            if (task.addr == 0 && task.size == 0) {
                send_sentinel_received_[idx] = true;
                {
                    std::lock_guard<std::mutex> lock(send_mutexes_[idx]);
                    if (send_queues_[idx].empty()) {
                        send_completed_[idx] = true;
                        send_sentinel_received_[idx] = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) continue;
            network_tasks_inflight_.fetch_add(1, std::memory_order_relaxed);
            SaveNetScopeTimer save_net_timer(this);
            if (use_rdma_ && send_channels_[idx] && send_channels_[idx]->is_connected()) {
                // std::cout << "[ECNAIVE RDMA] SendWorker[" << idx << "] RDMA send "
                //           << (task.size / (1024.0*1024.0)) << " MB" << std::endl;
                send_channels_[idx]->send_data(
                    reinterpret_cast<const uint8_t*>(task.addr), task.size);
            } else if (conn_.send_socket(idx).is_open()) {
                send_with_size(conn_.send_socket(idx), task.addr, task.size);
            }
            network_tasks_inflight_.fetch_sub(1, std::memory_order_relaxed);
            // Release buffer after send: data channels 0..k-2, parity channels k-1..k
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                if (idx >= k_ - 1) {
                    parity_buffers_to_release_.push(task.addr);
                } else {
                    data_buffers_to_release_.push(task.addr);
                }
            }
        }
    }

    void recv_worker(int idx) {
        while (!stop_) {
            RecvTask task;
            {
                std::unique_lock<std::mutex> lk(recv_mutexes_[idx]);
                recv_cvs_[idx].wait(lk, [this, idx] {
                    return stop_ || !recv_queues_[idx].empty();
                });
                if (stop_) break;
                task = recv_queues_[idx].front();
                recv_queues_[idx].pop();
            }
            if (task.addr == 0 && task.size == 0) {
                recv_sentinel_received_[idx] = true;
                {
                    std::lock_guard<std::mutex> lock(recv_mutexes_[idx]);
                    if (recv_queues_[idx].empty()) {
                        recv_completed_[idx] = true;
                        recv_sentinel_received_[idx] = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) continue;
            network_tasks_inflight_.fetch_add(1, std::memory_order_relaxed);
            SaveNetScopeTimer save_net_timer(this);
            if (use_rdma_ && recv_channels_[idx] && recv_channels_[idx]->is_connected()) {
                // std::cout << "[ECNAIVE RDMA] RecvWorker[" << idx << "] RDMA recv "
                //           << (task.size / (1024.0*1024.0)) << " MB" << std::endl;
                recv_channels_[idx]->receive_data(
                    reinterpret_cast<uint8_t*>(task.addr), task.size);
            } else if (conn_.recv_socket(idx).is_open()) {
                if (!recv_with_size_bool(conn_.recv_socket(idx),
                                         reinterpret_cast<void*>(task.addr), task.size)) {
                    std::cerr << "ECNAIVE: RecvWorker[" << idx << "] recv failed" << std::endl;
                }
            }
            network_tasks_inflight_.fetch_sub(1, std::memory_order_relaxed);
        }
    }

public:
    // ========== Generalized save submit interface ==========

    // Expose ISA-L RS encode for HW recovery (callable from Python)
    // Encodes k data blocks into 2 parity blocks using the ISA-L RS encoder
    void encode_ec_blocks_public(
            const std::vector<uintptr_t>& data_addrs,
            uintptr_t parity0_addr,
            uintptr_t parity1_addr,
            size_t size)
    {
        encode_ec_blocks(data_addrs, parity0_addr, parity1_addr, size);
    }

    // Submit one encoding + distribution operation for a chunk
    void submit_ecnaive_save_general(
            const std::vector<uintptr_t>& data_addrs,  // k data block addresses
            uintptr_t parity0_addr,
            uintptr_t parity1_addr,
            const std::vector<uintptr_t>& recv_addrs,  // n-1 recv block addresses
            size_t size)
    {
        // Encode k data blocks to 2 parity blocks
        encode_ec_blocks(data_addrs, parity0_addr, parity1_addr, size);

        // Steps:
        // 1. Submit recv tasks for all n-1 channels (write to recv_addrs)
        // 2. Submit send tasks:
        //    - data blocks d_1..d_{k-1} via send channels 0..k-2
        //    - parity0 via send channel k-1
        //    - parity1 via send channel k

        for (int i = 0; i < num_channels_; ++i) {
            {
                std::lock_guard<std::mutex> lk(recv_mutexes_[i]);
                recv_queues_[i].push({recv_addrs[i], size});
            }
            recv_cvs_[i].notify_one();
        }

        // Send data blocks d_1..d_{k-1}
        for (int j = 1; j < k_; ++j) {
            int send_idx = j - 1;  // send channel 0..k-2
            {
                std::lock_guard<std::mutex> lk(send_mutexes_[send_idx]);
                send_queues_[send_idx].push({data_addrs[j], size});
            }
            send_cvs_[send_idx].notify_one();
        }

        // Send parity blocks
        {
            std::lock_guard<std::mutex> lk(send_mutexes_[k_ - 1]);  // parity0
            send_queues_[k_ - 1].push({parity0_addr, size});
        }
        send_cvs_[k_ - 1].notify_one();

        {
            std::lock_guard<std::mutex> lk(send_mutexes_[k_]);  // parity1
            send_queues_[k_].push({parity1_addr, size});
        }
        send_cvs_[k_].notify_one();
    }

    void submit_send_sentinels(int num_sends) {
        for (int i = 0; i < num_sends; ++i) {
            std::lock_guard<std::mutex> lk(send_mutexes_[i]);
            send_queues_[i].push({0, 0});
            send_cvs_[i].notify_one();
        }
    }

    void submit_recv_sentinels(int num_recvs) {
        for (int i = 0; i < num_recvs; ++i) {
            std::lock_guard<std::mutex> lk(recv_mutexes_[i]);
            recv_queues_[i].push({0, 0});
            recv_cvs_[i].notify_one();
        }
    }

    // Generalized single-task submit: push one send/recv task to a channel (for recovery)
    void submit_send_task(int channel_idx, uintptr_t addr, size_t size) {
        if (channel_idx < 0 || channel_idx >= num_channels_) return;
        {
            std::lock_guard<std::mutex> lk(send_mutexes_[channel_idx]);
            send_queues_[channel_idx].push({addr, size});
        }
        send_cvs_[channel_idx].notify_one();
    }

    void submit_recv_task(int channel_idx, uintptr_t addr, size_t size) {
        if (channel_idx < 0 || channel_idx >= num_channels_) return;
        {
            std::lock_guard<std::mutex> lk(recv_mutexes_[channel_idx]);
            recv_queues_[channel_idx].push({addr, size});
        }
        recv_cvs_[channel_idx].notify_one();
    }

private:
    // ========== EC Decode pthread pool (RS decode, 16 workers) ==========

    static std::array<int, kEcRsPoolSize> parse_xor_pool_cpus_or_throw() {
        std::array<int, kEcRsPoolSize> cpus{};
        const char* env = std::getenv("ECNAIVE_XOR_CPU_LIST");
        if (!env || !*env) {
            for (int i = 0; i < kEcRsPoolSize; ++i) {
                cpus[static_cast<size_t>(i)] = i;
            }
            return cpus;
        }
        std::vector<int> parsed;
        const char* p = env;
        while (*p) {
            while (*p && (std::isspace(static_cast<unsigned char>(*p)) || *p == ',')) {
                ++p;
            }
            if (!*p) {
                break;
            }
            char* end = nullptr;
            long v = std::strtol(p, &end, 10);
            if (end == p || v < 0 || v > 65535) {
                throw std::runtime_error("ECNAIVE_XOR_CPU_LIST: invalid CPU id token");
            }
            parsed.push_back(static_cast<int>(v));
            p = end;
        }
        if (parsed.size() != static_cast<size_t>(kEcRsPoolSize)) {
            throw std::runtime_error(
                "ECNAIVE_XOR_CPU_LIST must contain exactly 16 comma-separated CPU ids "
                "(or unset to use 0..15)");
        }
        for (size_t i = 0; i < cpus.size(); ++i) {
            cpus[i] = parsed[i];
        }
        return cpus;
    }

    void ec_rs_pool_init() {
        if (ec_rs_pool_inited_.load(std::memory_order_acquire)) {
            return;
        }
        ec_rs_pool_cpus_ = parse_xor_pool_cpus_or_throw();  // reuse same CPU list
        ec_rs_pool_stop_.store(false, std::memory_order_release);
        ec_rs_pool_epoch_.store(0, std::memory_order_release);
        ec_rs_pool_remaining_.store(0, std::memory_order_release);
        ec_rs_pool_have_tbls_ = false;
        for (auto& e : ec_rs_pool_last_epoch_) e = 0;

        for (int i = 0; i < kEcRsPoolSize; ++i) {
            ec_rs_pool_ctx_[static_cast<size_t>(i)].self = this;
            ec_rs_pool_ctx_[static_cast<size_t>(i)].wid = i;
            int rc = pthread_create(
                &ec_rs_pool_threads_[static_cast<size_t>(i)],
                nullptr,
                &ECNaiveNative::ec_rs_pool_pthread_entry,
                &ec_rs_pool_ctx_[static_cast<size_t>(i)]);
            if (rc != 0) {
                ec_rs_pool_stop_.store(true, std::memory_order_release);
                ec_rs_pool_worker_cv_.notify_all();
                for (int j = 0; j < i; ++j) {
                    pthread_join(ec_rs_pool_threads_[static_cast<size_t>(j)], nullptr);
                }
                throw std::runtime_error("ECNAIVE: pthread_create for EC RS pool failed: " +
                                         std::string(std::strerror(rc)));
            }
        }
        ec_rs_pool_inited_.store(true, std::memory_order_release);
    }

    void ec_rs_pool_shutdown() {
        if (!ec_rs_pool_inited_.load(std::memory_order_acquire)) return;
        ec_rs_pool_stop_.store(true, std::memory_order_release);
        ec_rs_pool_worker_cv_.notify_all();
        for (int i = 0; i < kEcRsPoolSize; ++i) {
            pthread_join(ec_rs_pool_threads_[static_cast<size_t>(i)], nullptr);
        }
        ec_rs_pool_stop_.store(false, std::memory_order_release);
        ec_rs_pool_inited_.store(false, std::memory_order_release);
    }

    static void* ec_rs_pool_pthread_entry(void* arg) {
        auto* ctx = static_cast<EcRsPoolWorkerCtx*>(arg);
        ctx->self->ec_rs_pool_worker_loop(ctx->wid);
        return nullptr;
    }

    void ec_rs_pool_execute_slice(const EcRsJob& job, int wid) {
        const size_t total = job.size;
        const size_t base = total / static_cast<size_t>(kEcRsPoolSize);
        const size_t rem  = total % static_cast<size_t>(kEcRsPoolSize);
        size_t off, len;
        if (wid < kEcRsPoolSize - 1) {
            off = static_cast<size_t>(wid) * base;
            len = base;
        } else {
            off = static_cast<size_t>(kEcRsPoolSize - 1) * base;
            len = base + rem;
        }
        if (len == 0) return;

        // Build per-stripe address arrays
        std::vector<unsigned char*> srcs(static_cast<size_t>(job.k));
        std::vector<unsigned char*> dests(static_cast<size_t>(job.m));
        for (int i = 0; i < job.k; ++i) {
            srcs[static_cast<size_t>(i)] =
                reinterpret_cast<unsigned char*>(job.surviving_addrs[i] + off);
        }
        for (int i = 0; i < job.m; ++i) {
            dests[static_cast<size_t>(i)] =
                reinterpret_cast<unsigned char*>(job.recovered_addrs[i] + off);
        }
        ec_encode_data(static_cast<int>(len), job.k, job.m,
                       const_cast<unsigned char*>(job.decode_tbls),
                       srcs.data(), dests.data());
    }

    void ec_rs_pool_worker_loop(int wid) {
        const int cpu = ec_rs_pool_cpus_[static_cast<size_t>(wid)];
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (cpu >= 0 && static_cast<unsigned>(cpu) < CPU_SETSIZE) {
            CPU_SET(static_cast<unsigned>(cpu), &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        }
        while (true) {
            std::unique_lock<std::mutex> lk(ec_rs_pool_mutex_);
            ec_rs_pool_worker_cv_.wait(lk, [&] {
                return ec_rs_pool_stop_.load(std::memory_order_acquire) ||
                       ec_rs_pool_last_epoch_[static_cast<size_t>(wid)] <
                           ec_rs_pool_epoch_.load(std::memory_order_acquire);
            });
            if (ec_rs_pool_stop_.load(std::memory_order_acquire)) break;
            uint64_t e = ec_rs_pool_epoch_.load(std::memory_order_acquire);
            EcRsJob local_copy = ec_rs_pool_shared_job_;
            lk.unlock();

            ec_rs_pool_execute_slice(local_copy, wid);

            {
                std::lock_guard<std::mutex> guard(ec_rs_pool_mutex_);
                ec_rs_pool_last_epoch_[static_cast<size_t>(wid)] = e;
                int rem = ec_rs_pool_remaining_.fetch_sub(1, std::memory_order_acq_rel) - 1;
                if (rem == 0) ec_rs_pool_coordinator_cv_.notify_one();
            }
        }
    }

    void ec_rs_pool_run_parallel(const EcRsJob& job) {
        {
            std::lock_guard<std::mutex> publish(ec_rs_pool_mutex_);
            if (stop_.load(std::memory_order_acquire)) return;
            ec_rs_pool_shared_job_ = job;
            ec_rs_pool_epoch_.fetch_add(1, std::memory_order_acq_rel);
            ec_rs_pool_remaining_.store(kEcRsPoolSize, std::memory_order_release);
        }
        ec_rs_pool_worker_cv_.notify_all();
        std::unique_lock<std::mutex> lk(ec_rs_pool_mutex_);
        ec_rs_pool_coordinator_cv_.wait(lk, [&] {
            return ec_rs_pool_remaining_.load(std::memory_order_acquire) == 0 ||
                   stop_.load(std::memory_order_acquire);
        });
    }

    // Encode dispatch via ec_rs_pool (ISA-L ec_encode_data — encode with
    // g_tbls_, decode with ec_decode_tbls_).
    void ec_rs_pool_run_encode(const std::vector<uintptr_t>& data_addrs,
                                uintptr_t parity0_addr, uintptr_t parity1_addr,
                                size_t size) {
        uintptr_t parity_out[2] = {parity0_addr, parity1_addr};
        EcRsJob job{};
        job.size = size;
        job.k = k_;
        job.m = rows_;
        job.surviving_addrs = data_addrs.data();
        job.recovered_addrs = parity_out;
        job.decode_tbls = g_tbls_;
        ec_rs_pool_run_parallel(job);
    }
};

}  // namespace

PYBIND11_MODULE(ecnaive_native, m) {
    pybind11::class_<ECNaiveNative>(m, "ECNaiveNative")
        // Generalized constructor: vectors for variable number of channels
        .def(pybind11::init<const std::vector<std::string>&,
                            const std::vector<uint16_t>&,
                            const std::vector<std::string>&,
                            const std::vector<uint16_t>&,
                            const std::vector<uint16_t>&,
                            const std::vector<uint16_t>&,
                            int, bool, int>(),
             pybind11::arg("send_ips"),
             pybind11::arg("send_ports"),
             pybind11::arg("recv_ips"),
             pybind11::arg("recv_ports"),
             pybind11::arg("rdma_send_ports"),
             pybind11::arg("rdma_recv_ports"),
             pybind11::arg("k") = 2,
             pybind11::arg("use_rdma") = false,
             pybind11::arg("rank_in_group") = -1)
        // RDMA buffer management
        .def("register_buffer", &ECNaiveNative::register_buffer,
             "Register buffer for RDMA operations",
             pybind11::arg("addr"),
             pybind11::arg("size"))
        .def("unregister_buffer", &ECNaiveNative::unregister_buffer,
             "Unregister buffer from RDMA",
             pybind11::arg("addr"))
        // Generalized save mode submit function (k+2 scheme)
        .def("submit_ecnaive_save_general", &ECNaiveNative::submit_ecnaive_save_general,
             "Generalized save: encode k data blocks to 2 parity and distribute",
             pybind11::arg("data_addrs"),
             pybind11::arg("parity0_addr"),
             pybind11::arg("parity1_addr"),
             pybind11::arg("recv_addrs"),
             pybind11::arg("size"))
        // Legacy unified save mode (backward compat for k=2)
        .def("submit_ecnaive_save", &ECNaiveNative::submit_ecnaive_save,
             "Unified save mode function: encode and submit send/recv tasks",
             pybind11::arg("data0_addr"),
             pybind11::arg("data1_addr"),
             pybind11::arg("parity0_addr"),
             pybind11::arg("parity1_addr"),
             pybind11::arg("recv_parity1_addr"),
             pybind11::arg("recv_parity0_addr"),
             pybind11::arg("recv_data1_addr"),
             pybind11::arg("size"))
        // Generalized sentinel methods
        .def("submit_send_sentinels", &ECNaiveNative::submit_send_sentinels,
             pybind11::arg("num_sends"))
        .def("submit_recv_sentinels", &ECNaiveNative::submit_recv_sentinels,
             pybind11::arg("num_recvs"))
        // Generalized single-task submit for recovery
        .def("submit_send_task", &ECNaiveNative::submit_send_task,
             pybind11::arg("channel_idx"),
             pybind11::arg("addr"),
             pybind11::arg("size"))
        .def("submit_recv_task", &ECNaiveNative::submit_recv_task,
             pybind11::arg("channel_idx"),
             pybind11::arg("addr"),
             pybind11::arg("size"))
        // Legacy submit functions
        .def("submit_send_data1", &ECNaiveNative::submit_send_data1,
             pybind11::arg("send_addr"), pybind11::arg("size"))
        .def("submit_send_parity0", &ECNaiveNative::submit_send_parity0,
             pybind11::arg("send_addr"), pybind11::arg("size"))
        .def("submit_send_parity1", &ECNaiveNative::submit_send_parity1,
             pybind11::arg("send_addr"), pybind11::arg("size"))
        .def("submit_recv_parity1", &ECNaiveNative::submit_recv_parity1,
             pybind11::arg("recv_addr"), pybind11::arg("size"))
        .def("submit_recv_parity0", &ECNaiveNative::submit_recv_parity0,
             pybind11::arg("recv_addr"), pybind11::arg("size"))
        .def("submit_recv_data1", &ECNaiveNative::submit_recv_data1,
             pybind11::arg("recv_addr"), pybind11::arg("size"))
        // Common functions
        .def("get_data_buffers_to_release", &ECNaiveNative::get_data_buffers_to_release)
        .def("get_parity_buffers_to_release", &ECNaiveNative::get_parity_buffers_to_release)
        .def("reset_encoding_completion_flags", &ECNaiveNative::reset_encoding_completion_flags)
        .def("get_ft_timing_stats", &ECNaiveNative::get_ft_timing_stats,
             "Return per-rank timing: net_s wall-span; encode_s serial-equivalent encode CPU sum")
        .def("wait_for_encoding_completion", &ECNaiveNative::wait_for_encoding_completion)
        .def("wait_for_pending_network_tasks", &ECNaiveNative::wait_for_pending_network_tasks,
             "Wait until all in-flight save-path send/recv tasks complete (no sentinels)")
        // Legacy save mode sentinels
        .def("submit_send_data1_sentinel", &ECNaiveNative::submit_send_data1_sentinel)
        .def("submit_send_parity0_sentinel", &ECNaiveNative::submit_send_parity0_sentinel)
        .def("submit_send_parity1_sentinel", &ECNaiveNative::submit_send_parity1_sentinel)
        .def("submit_recv_parity1_sentinel", &ECNaiveNative::submit_recv_parity1_sentinel)
        .def("submit_recv_parity0_sentinel", &ECNaiveNative::submit_recv_parity0_sentinel)
        .def("submit_recv_data1_sentinel", &ECNaiveNative::submit_recv_data1_sentinel)
        // Load mode functions
        .def("set_load_mode", &ECNaiveNative::set_load_mode,
             "Set load mode for recovery",
             pybind11::arg("is_load"),
             pybind11::arg("failed_rank") = -1,
             pybind11::arg("rank") = -1,
             pybind11::arg("is_software_only") = false)
        // Software failure mode functions
        .def("software_send_rank3_data1", &ECNaiveNative::software_send_rank3_data1,
             "Software failure mode: rank3 send d21 to rank2",
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("software_recv_data1", &ECNaiveNative::software_recv_data1,
             "Software failure mode: rank2 receive d21 from rank3",
             pybind11::arg("recv_addr"),
             pybind11::arg("size"))
        // Generalized SW recovery (k-1 ports, supports any k >= 2)
        .def("init_ecnaive_load_sw_bind_listen",
             &ECNaiveNative::init_ecnaive_load_sw_bind_listen,
             "Generalized SW recovery phase 1: bind/listen on k-1 ports",
             pybind11::arg("rank_in_group"), pybind11::arg("receiver_ip"),
             pybind11::arg("num_blocks"), pybind11::arg("ports"))
        .def("init_ecnaive_load_sw_accept",
             &ECNaiveNative::init_ecnaive_load_sw_accept,
             "SW recovery phase 2a: receiver accepts on all k-1 ports + RDMA channels",
             pybind11::arg("rank_in_group"), pybind11::arg("num_blocks"))
        .def("init_ecnaive_load_sw_connect_one",
             &ECNaiveNative::init_ecnaive_load_sw_connect_one,
             "SW recovery phase 2b: one sender connects to exactly one port for its block",
             pybind11::arg("rank_in_group"), pybind11::arg("receiver_ip"),
             pybind11::arg("block_idx"), pybind11::arg("port"))
        .def("sw_send_data", &ECNaiveNative::sw_send_data,
             "Generalized SW recovery: send data block by index",
             pybind11::arg("block_idx"), pybind11::arg("addr"), pybind11::arg("size"))
        .def("sw_recv_data", &ECNaiveNative::sw_recv_data,
             "Generalized SW recovery: receive data block by index",
             pybind11::arg("block_idx"), pybind11::arg("addr"), pybind11::arg("size"))
        // RS encode for HW recovery: encode k data blocks → 2 parity blocks
        .def("encode_ec_blocks", &ECNaiveNative::encode_ec_blocks_public,
             "Encode k data blocks into 2 parity blocks via ISA-L RS encoder",
             pybind11::arg("data_addrs"),
             pybind11::arg("parity0_addr"),
             pybind11::arg("parity1_addr"),
             pybind11::arg("size"))
        // RS decode recovery (software recovery, synchronous)
        .def("submit_ecnaive_decode_recovery", &ECNaiveNative::submit_ecnaive_decode_recovery,
             "Recover m lost data blocks from k surviving blocks via RS decode (ISA-L)",
             pybind11::arg("k"),
             pybind11::arg("m"),
             pybind11::arg("lost_positions"),
             pybind11::arg("surviving_addrs"),
             pybind11::arg("recovered_addrs"),
             pybind11::arg("size"))
        .def("stop", &ECNaiveNative::stop);
}


