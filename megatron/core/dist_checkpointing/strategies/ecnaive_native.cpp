#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/asio.hpp>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

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

#include <atomic>
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
    
    static const size_t TEMP_BUFFER_SIZE = 1ULL * 1024 * 1024 * 1024;  // 1 GB
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
        
        // Transition QP to RTR
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
        attr.ah_attr.grh.sgid_index = 0;
        attr.ah_attr.grh.hop_limit = 64;
        
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
    
    RdmaConnInfo get_local_conn_info() {
        RdmaConnInfo info{};
        info.qp_num = qp_->qp_num;
        
        ibv_port_attr port_attr;
        if (ibv_query_port(context_, 1, &port_attr)) {
            throw std::runtime_error("Failed to query port");
        }
        info.lid = port_attr.lid;
        
        ibv_gid gid;
        if (ibv_query_gid(context_, 1, 0, &gid)) {
            throw std::runtime_error("Failed to query GID");
        }
        memcpy(info.gid, &gid, 16);
        
        return info;
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
            
            std::vector<ibv_sge> sges;
            std::vector<ibv_send_wr> wrs;
            
            for (size_t i = 0; i < chunk_count; ++i) {
                size_t current_size = std::min(CHUNK_SIZE, remaining);
                
                ibv_sge sge{};
                sge.addr = reinterpret_cast<uint64_t>(data + offset);
                sge.length = current_size;
                sge.lkey = mr->lkey;
                sges.push_back(sge);
                
                ibv_send_wr wr{};
                wr.wr_id = i;
                wr.sg_list = &sges[i];
                wr.num_sge = 1;
                wr.opcode = IBV_WR_SEND;
                wr.send_flags = IBV_SEND_SIGNALED;
                if (i < chunk_count - 1) {
                    wr.next = &wrs[i + 1];
                }
                wrs.push_back(wr);
                
                offset += current_size;
                remaining -= current_size;
            }
            
            // Post send work requests
            ibv_send_wr* bad_wr = nullptr;
            if (ibv_post_send(qp_, &wrs[0], &bad_wr)) {
                throw std::runtime_error("Failed to post send work request");
            }
            
            // Poll for completions
            poll_completion(send_cq_, chunk_count);
        }
    }
    
    void receive_data_chunked(uint8_t* buffer, size_t total_size, ibv_mr* mr) {
        size_t remaining = total_size;
        size_t offset = 0;
        
        while (remaining > 0) {
            size_t chunk_count = std::min(remaining, CHUNK_SIZE * MAX_BATCH_WR) / CHUNK_SIZE;
            if (chunk_count == 0) chunk_count = 1;
            
            std::vector<ibv_sge> sges;
            std::vector<ibv_recv_wr> wrs;
            
            for (size_t i = 0; i < chunk_count; ++i) {
                size_t current_size = std::min(CHUNK_SIZE, remaining);
                
                ibv_sge sge{};
                sge.addr = reinterpret_cast<uint64_t>(buffer + offset);
                sge.length = current_size;
                sge.lkey = mr->lkey;
                sges.push_back(sge);
                
                ibv_recv_wr wr{};
                wr.wr_id = i;
                wr.sg_list = &sges[i];
                wr.num_sge = 1;
                if (i < chunk_count - 1) {
                    wr.next = &wrs[i + 1];
                }
                wrs.push_back(wr);
                
                offset += current_size;
                remaining -= current_size;
            }
            
            // Post receive work requests
            ibv_recv_wr* bad_wr = nullptr;
            if (ibv_post_recv(qp_, &wrs[0], &bad_wr)) {
                throw std::runtime_error("Failed to post receive work request");
            }
            
            // Poll for completions
            poll_completion(recv_cq_, chunk_count);
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
    
    // Save mode sockets: 3 sends + 3 receives per rank
    boost::asio::ip::tcp::socket send_data1_socket_;      // Send d_{i1} to rank i+1
    boost::asio::ip::tcp::socket send_parity0_socket_;    // Send p_{i0} to rank i+2
    boost::asio::ip::tcp::socket send_parity1_socket_;    // Send p_{i1} to rank i+3
    
    boost::asio::ip::tcp::socket recv_parity1_socket_;    // Recv p_{i+1,1} from rank i+1
    boost::asio::ip::tcp::socket recv_parity0_socket_;    // Recv p_{i+2,0} from rank i+2
    boost::asio::ip::tcp::socket recv_data1_socket_;      // Recv d_{i+3,1} from rank i+3
    
    boost::asio::ip::tcp::acceptor recv_parity1_acceptor_;
    boost::asio::ip::tcp::acceptor recv_parity0_acceptor_;
    boost::asio::ip::tcp::acceptor recv_data1_acceptor_;

    // Load mode sockets (rank2 as receiver) - ECLATIN style (6 sockets)
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
    
    // Load mode sockets (other ranks as senders) - ECLATIN style
    boost::asio::ip::tcp::socket load_send_rank0_data2_socket_;
    boost::asio::ip::tcp::socket load_send_rank0_parity2_socket_;
    boost::asio::ip::tcp::socket load_send_rank1_data1_socket_;
    boost::asio::ip::tcp::socket load_send_rank1_parity1_socket_;
    boost::asio::ip::tcp::socket load_send_rank3_data1_socket_;
    boost::asio::ip::tcp::socket load_send_rank3_data2_socket_;

    // EC-NAIVE load mode sockets (rank2 recovery: 8 recv + multiple send)
    // rank2 receiver sockets (8 sockets for full recovery)
    boost::asio::ip::tcp::socket ecnaive_load_recv_rank3_data1_socket_;  // rank2接收d_{2,1} from rank3
    boost::asio::ip::tcp::socket ecnaive_load_recv_rank0_parity0_socket_; // rank2接收p_{2,0} from rank0
    boost::asio::ip::tcp::socket ecnaive_load_recv_rank0_data0_socket_; // rank2接收d_{0,0} from rank0
    boost::asio::ip::tcp::socket ecnaive_load_recv_rank1_data1_socket_; // rank2接收d_{0,1} from rank1
    boost::asio::ip::tcp::socket ecnaive_load_recv_rank1_data0_socket_; // rank2接收d_{1,0} from rank1
    boost::asio::ip::tcp::socket ecnaive_load_recv_rank1_parity1_socket_; // rank2接收p_{1,1} from rank1
    boost::asio::ip::tcp::socket ecnaive_load_recv_rank3_data0_socket_; // rank2接收d_{3,0} from rank3
    boost::asio::ip::tcp::socket ecnaive_load_recv_rank0_data1_socket_; // rank2接收d_{3,1} from rank0
    
    boost::asio::ip::tcp::acceptor ecnaive_load_recv_rank3_data1_acceptor_;
    boost::asio::ip::tcp::acceptor ecnaive_load_recv_rank0_parity0_acceptor_;
    boost::asio::ip::tcp::acceptor ecnaive_load_recv_rank0_data0_acceptor_;
    boost::asio::ip::tcp::acceptor ecnaive_load_recv_rank1_data1_acceptor_;
    boost::asio::ip::tcp::acceptor ecnaive_load_recv_rank1_data0_acceptor_;
    boost::asio::ip::tcp::acceptor ecnaive_load_recv_rank1_parity1_acceptor_;
    boost::asio::ip::tcp::acceptor ecnaive_load_recv_rank3_data0_acceptor_;
    boost::asio::ip::tcp::acceptor ecnaive_load_recv_rank0_data1_acceptor_;
    
    // rank0 sender sockets (3 sockets: p_{2,0}, d_{0,0}, d_{3,1})
    boost::asio::ip::tcp::socket ecnaive_load_send_rank0_parity0_socket_; // rank0发送p_{2,0}
    boost::asio::ip::tcp::socket ecnaive_load_send_rank0_data0_socket_; // rank0发送d_{0,0}
    boost::asio::ip::tcp::socket ecnaive_load_send_rank0_data1_socket_; // rank0发送d_{3,1}
    
    // rank1 sender sockets (3 sockets: d_{0,1}, d_{1,0}, p_{1,1})
    boost::asio::ip::tcp::socket ecnaive_load_send_rank1_data1_socket_; // rank1发送d_{0,1}
    boost::asio::ip::tcp::socket ecnaive_load_send_rank1_data0_socket_; // rank1发送d_{1,0}
    boost::asio::ip::tcp::socket ecnaive_load_send_rank1_parity1_socket_; // rank1发送p_{1,1}
    
    // rank3 sender sockets (2 sockets: d_{2,1}, d_{3,0})
    boost::asio::ip::tcp::socket ecnaive_load_send_rank3_data1_socket_;  // rank3发送d_{2,1}
    boost::asio::ip::tcp::socket ecnaive_load_send_rank3_data0_socket_;  // rank3发送d_{3,0}

    // Save mode connection flags
    std::atomic<bool> send_data1_connected_{false};
    std::atomic<bool> send_parity0_connected_{false};
    std::atomic<bool> send_parity1_connected_{false};
    std::atomic<bool> recv_parity1_connected_{false};
    std::atomic<bool> recv_parity0_connected_{false};
    std::atomic<bool> recv_data1_connected_{false};
    
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
    
    // EC-NAIVE load mode connection flags (rank2 receiver: 8 flags)
    std::atomic<bool> ecnaive_load_recv_rank3_data1_connected_{false};  // d_{2,1} from rank3
    std::atomic<bool> ecnaive_load_recv_rank0_parity0_connected_{false}; // p_{2,0} from rank0
    std::atomic<bool> ecnaive_load_recv_rank0_data0_connected_{false};  // d_{0,0} from rank0
    std::atomic<bool> ecnaive_load_recv_rank1_data1_connected_{false};  // d_{0,1} from rank1
    std::atomic<bool> ecnaive_load_recv_rank1_data0_connected_{false};  // d_{1,0} from rank1
    std::atomic<bool> ecnaive_load_recv_rank1_parity1_connected_{false}; // p_{1,1} from rank1
    std::atomic<bool> ecnaive_load_recv_rank3_data0_connected_{false};  // d_{3,0} from rank3
    std::atomic<bool> ecnaive_load_recv_rank0_data1_connected_{false};  // d_{3,1} from rank0
    
    // EC-NAIVE load mode connection flags (rank0 sender: 3 flags)
    std::atomic<bool> ecnaive_load_send_rank0_parity0_connected_{false}; // p_{2,0}
    std::atomic<bool> ecnaive_load_send_rank0_data0_connected_{false};  // d_{0,0}
    std::atomic<bool> ecnaive_load_send_rank0_data1_connected_{false};  // d_{3,1}
    
    // EC-NAIVE load mode connection flags (rank1 sender: 3 flags)
    std::atomic<bool> ecnaive_load_send_rank1_data1_connected_{false};  // d_{0,1}
    std::atomic<bool> ecnaive_load_send_rank1_data0_connected_{false};  // d_{1,0}
    std::atomic<bool> ecnaive_load_send_rank1_parity1_connected_{false}; // p_{1,1}
    
    // EC-NAIVE load mode connection flags (rank3 sender: 2 flags)
    std::atomic<bool> ecnaive_load_send_rank3_data1_connected_{false};  // d_{2,1}
    std::atomic<bool> ecnaive_load_send_rank3_data0_connected_{false};  // d_{3,0}

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
          ecnaive_load_recv_rank3_data1_socket_(io_context_),
          ecnaive_load_recv_rank0_parity0_socket_(io_context_),
          ecnaive_load_recv_rank0_data0_socket_(io_context_),
          ecnaive_load_recv_rank1_data1_socket_(io_context_),
          ecnaive_load_recv_rank1_data0_socket_(io_context_),
          ecnaive_load_recv_rank1_parity1_socket_(io_context_),
          ecnaive_load_recv_rank3_data0_socket_(io_context_),
          ecnaive_load_recv_rank0_data1_socket_(io_context_),
          ecnaive_load_recv_rank3_data1_acceptor_(io_context_),
          ecnaive_load_recv_rank0_parity0_acceptor_(io_context_),
          ecnaive_load_recv_rank0_data0_acceptor_(io_context_),
          ecnaive_load_recv_rank1_data1_acceptor_(io_context_),
          ecnaive_load_recv_rank1_data0_acceptor_(io_context_),
          ecnaive_load_recv_rank1_parity1_acceptor_(io_context_),
          ecnaive_load_recv_rank3_data0_acceptor_(io_context_),
          ecnaive_load_recv_rank0_data1_acceptor_(io_context_),
          ecnaive_load_send_rank0_parity0_socket_(io_context_),
          ecnaive_load_send_rank0_data0_socket_(io_context_),
          ecnaive_load_send_rank0_data1_socket_(io_context_),
          ecnaive_load_send_rank1_data1_socket_(io_context_),
          ecnaive_load_send_rank1_data0_socket_(io_context_),
          ecnaive_load_send_rank1_parity1_socket_(io_context_),
          ecnaive_load_send_rank3_data1_socket_(io_context_),
          ecnaive_load_send_rank3_data0_socket_(io_context_) {}

    // Save mode getters
    boost::asio::ip::tcp::socket& get_send_data1_socket() { return send_data1_socket_; }
    boost::asio::ip::tcp::socket& get_send_parity0_socket() { return send_parity0_socket_; }
    boost::asio::ip::tcp::socket& get_send_parity1_socket() { return send_parity1_socket_; }
    boost::asio::ip::tcp::socket& get_recv_parity1_socket() { return recv_parity1_socket_; }
    boost::asio::ip::tcp::socket& get_recv_parity0_socket() { return recv_parity0_socket_; }
    boost::asio::ip::tcp::socket& get_recv_data1_socket() { return recv_data1_socket_; }
    
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

    // EC-NAIVE load mode getters (rank2 receiver)
    // EC-NAIVE load mode getters (rank2 receiver: 8 sockets)
    boost::asio::ip::tcp::socket& get_ecnaive_load_recv_rank3_data1_socket() { 
        return ecnaive_load_recv_rank3_data1_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_recv_rank0_parity0_socket() { 
        return ecnaive_load_recv_rank0_parity0_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_recv_rank0_data0_socket() { 
        return ecnaive_load_recv_rank0_data0_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_recv_rank1_data1_socket() { 
        return ecnaive_load_recv_rank1_data1_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_recv_rank1_data0_socket() { 
        return ecnaive_load_recv_rank1_data0_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_recv_rank1_parity1_socket() { 
        return ecnaive_load_recv_rank1_parity1_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_recv_rank3_data0_socket() { 
        return ecnaive_load_recv_rank3_data0_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_recv_rank0_data1_socket() { 
        return ecnaive_load_recv_rank0_data1_socket_; 
    }
    
    // EC-NAIVE load mode getters (rank0 sender: 3 sockets)
    boost::asio::ip::tcp::socket& get_ecnaive_load_send_rank0_parity0_socket() { 
        return ecnaive_load_send_rank0_parity0_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_send_rank0_data0_socket() { 
        return ecnaive_load_send_rank0_data0_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_send_rank0_data1_socket() { 
        return ecnaive_load_send_rank0_data1_socket_; 
    }
    
    // EC-NAIVE load mode getters (rank1 sender: 3 sockets)
    boost::asio::ip::tcp::socket& get_ecnaive_load_send_rank1_data1_socket() { 
        return ecnaive_load_send_rank1_data1_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_send_rank1_data0_socket() { 
        return ecnaive_load_send_rank1_data0_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_send_rank1_parity1_socket() { 
        return ecnaive_load_send_rank1_parity1_socket_; 
    }
    
    // EC-NAIVE load mode getters (rank3 sender: 2 sockets)
    boost::asio::ip::tcp::socket& get_ecnaive_load_send_rank3_data1_socket() { 
        return ecnaive_load_send_rank3_data1_socket_; 
    }
    boost::asio::ip::tcp::socket& get_ecnaive_load_send_rank3_data0_socket() { 
        return ecnaive_load_send_rank3_data0_socket_; 
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
    
    // EC-NAIVE load mode init functions (rank2 receiver)
    void bind_listen_ecnaive_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port);
    void bind_listen_ecnaive_load_recv_rank0_parity0(const std::string& listen_ip, uint16_t port);
    void bind_listen_ecnaive_load_recv_rank0_data0(const std::string& listen_ip, uint16_t port);
    void bind_listen_ecnaive_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port);
    void bind_listen_ecnaive_load_recv_rank1_data0(const std::string& listen_ip, uint16_t port);
    void bind_listen_ecnaive_load_recv_rank1_parity1(const std::string& listen_ip, uint16_t port);
    void bind_listen_ecnaive_load_recv_rank3_data0(const std::string& listen_ip, uint16_t port);
    void bind_listen_ecnaive_load_recv_rank0_data1(const std::string& listen_ip, uint16_t port);
    void accept_ecnaive_load_recv_rank3_data1();
    void accept_ecnaive_load_recv_rank0_parity0();
    void accept_ecnaive_load_recv_rank0_data0();
    void accept_ecnaive_load_recv_rank1_data1();
    void accept_ecnaive_load_recv_rank1_data0();
    void accept_ecnaive_load_recv_rank1_parity1();
    void accept_ecnaive_load_recv_rank3_data0();
    void accept_ecnaive_load_recv_rank0_data1();
    
    // EC-NAIVE load mode init functions (rank0 sender: 3 connections)
    void init_ecnaive_load_send_rank0_parity0(const std::string& rank2_ip, uint16_t port);
    void init_ecnaive_load_send_rank0_data0(const std::string& rank2_ip, uint16_t port);
    void init_ecnaive_load_send_rank0_data1(const std::string& rank2_ip, uint16_t port);
    
    // EC-NAIVE load mode init functions (rank1 sender: 3 connections)
    void init_ecnaive_load_send_rank1_data1(const std::string& rank2_ip, uint16_t port);
    void init_ecnaive_load_send_rank1_data0(const std::string& rank2_ip, uint16_t port);
    void init_ecnaive_load_send_rank1_parity1(const std::string& rank2_ip, uint16_t port);
    
    // EC-NAIVE load mode init functions (rank3 sender: 2 connections)
    void init_ecnaive_load_send_rank3_data1(const std::string& rank2_ip, uint16_t port);
    void init_ecnaive_load_send_rank3_data0(const std::string& rank2_ip, uint16_t port);
    
    // EC-NAIVE load mode connection checks (rank2 receiver: 8 flags)
    bool is_ecnaive_load_recv_rank3_data1_connected() const { 
        return ecnaive_load_recv_rank3_data1_connected_; 
    }
    bool is_ecnaive_load_recv_rank0_parity0_connected() const { 
        return ecnaive_load_recv_rank0_parity0_connected_; 
    }
    bool is_ecnaive_load_recv_rank0_data0_connected() const { 
        return ecnaive_load_recv_rank0_data0_connected_; 
    }
    bool is_ecnaive_load_recv_rank1_data1_connected() const { 
        return ecnaive_load_recv_rank1_data1_connected_; 
    }
    bool is_ecnaive_load_recv_rank1_data0_connected() const { 
        return ecnaive_load_recv_rank1_data0_connected_; 
    }
    bool is_ecnaive_load_recv_rank1_parity1_connected() const { 
        return ecnaive_load_recv_rank1_parity1_connected_; 
    }
    bool is_ecnaive_load_recv_rank3_data0_connected() const { 
        return ecnaive_load_recv_rank3_data0_connected_; 
    }
    bool is_ecnaive_load_recv_rank0_data1_connected() const { 
        return ecnaive_load_recv_rank0_data1_connected_; 
    }
    
    // EC-NAIVE load mode connection checks (rank0 sender: 3 flags)
    bool is_ecnaive_load_send_rank0_parity0_connected() const { 
        return ecnaive_load_send_rank0_parity0_connected_; 
    }
    bool is_ecnaive_load_send_rank0_data0_connected() const { 
        return ecnaive_load_send_rank0_data0_connected_; 
    }
    bool is_ecnaive_load_send_rank0_data1_connected() const { 
        return ecnaive_load_send_rank0_data1_connected_; 
    }
    
    // EC-NAIVE load mode connection checks (rank1 sender: 3 flags)
    bool is_ecnaive_load_send_rank1_data1_connected() const { 
        return ecnaive_load_send_rank1_data1_connected_; 
    }
    bool is_ecnaive_load_send_rank1_data0_connected() const { 
        return ecnaive_load_send_rank1_data0_connected_; 
    }
    bool is_ecnaive_load_send_rank1_parity1_connected() const { 
        return ecnaive_load_send_rank1_parity1_connected_; 
    }
    
    // EC-NAIVE load mode connection checks (rank3 sender: 2 flags)
    bool is_ecnaive_load_send_rank3_data1_connected() const { 
        return ecnaive_load_send_rank3_data1_connected_; 
    }
    bool is_ecnaive_load_send_rank3_data0_connected() const { 
        return ecnaive_load_send_rank3_data0_connected_; 
    }
    
    void wait_for_connections(int timeout_seconds = 30);
    void wait_for_load_connections(int timeout_seconds = 30);
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

void AsioConnectionManager::wait_for_connections(int timeout_seconds) {
    std::unique_lock<std::mutex> lock(connection_mutex_);
    connection_cv_.wait_for(
        lock,
        std::chrono::seconds(timeout_seconds),
        [this]() {
            return send_data1_connected_ && send_parity0_connected_ && send_parity1_connected_ &&
                   recv_parity1_connected_ && recv_parity0_connected_ && recv_data1_connected_;
        }
    );
}

void AsioConnectionManager::wait_for_load_connections(int timeout_seconds) {
    // For rank2: wait for all 6 recv connections
    // For rank0/1/3: wait for 2 send connections each
    if (load_recv_rank0_data2_connected_ || load_recv_rank0_parity2_connected_ ||
        load_recv_rank1_data1_connected_ || load_recv_rank1_parity1_connected_ ||
        load_recv_rank3_data1_connected_ || load_recv_rank3_data2_connected_) {
        // rank2: wait for all 6 recv connections
        int wait_count = 0;
        while (!(load_recv_rank0_data2_connected_ && load_recv_rank0_parity2_connected_ &&
                 load_recv_rank1_data1_connected_ && load_recv_rank1_parity1_connected_ &&
                 load_recv_rank3_data1_connected_ && load_recv_rank3_data2_connected_)) {
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

// ========== EC-NAIVE Load Mode Connection Methods ==========

// EC-NAIVE load mode bind+listen helpers (for rank2, before accept)
void AsioConnectionManager::bind_listen_ecnaive_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        ecnaive_load_recv_rank3_data1_acceptor_.open(endpoint.protocol());
        ecnaive_load_recv_rank3_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        ecnaive_load_recv_rank3_data1_acceptor_.bind(endpoint);
        ecnaive_load_recv_rank3_data1_acceptor_.listen();
        std::cout << "EC-NAIVE: [Rank 2] Bound and listening on ecnaive_load_recv_rank3_data1 port " 
                  << port << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: bind_listen_ecnaive_load_recv_rank3_data1 error: " << e.what() << std::endl;
        throw;
    }
}

void AsioConnectionManager::bind_listen_ecnaive_load_recv_rank0_parity0(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        ecnaive_load_recv_rank0_parity0_acceptor_.open(endpoint.protocol());
        ecnaive_load_recv_rank0_parity0_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        ecnaive_load_recv_rank0_parity0_acceptor_.bind(endpoint);
        ecnaive_load_recv_rank0_parity0_acceptor_.listen();
        std::cout << "EC-NAIVE: [Rank 2] Bound and listening on ecnaive_load_recv_rank0_parity0 port " 
                  << port << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: bind_listen_ecnaive_load_recv_rank0_parity0 error: " << e.what() << std::endl;
        throw;
    }
}

void AsioConnectionManager::bind_listen_ecnaive_load_recv_rank0_data0(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        ecnaive_load_recv_rank0_data0_acceptor_.open(endpoint.protocol());
        ecnaive_load_recv_rank0_data0_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        ecnaive_load_recv_rank0_data0_acceptor_.bind(endpoint);
        ecnaive_load_recv_rank0_data0_acceptor_.listen();
        std::cout << "EC-NAIVE: [Rank 2] Bound and listening on ecnaive_load_recv_rank0_data0 port " 
                  << port << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: bind_listen_ecnaive_load_recv_rank0_data0 error: " << e.what() << std::endl;
        throw;
    }
}

void AsioConnectionManager::bind_listen_ecnaive_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        ecnaive_load_recv_rank1_data1_acceptor_.open(endpoint.protocol());
        ecnaive_load_recv_rank1_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        ecnaive_load_recv_rank1_data1_acceptor_.bind(endpoint);
        ecnaive_load_recv_rank1_data1_acceptor_.listen();
        std::cout << "EC-NAIVE: [Rank 2] Bound and listening on ecnaive_load_recv_rank1_data1 port " 
                  << port << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: bind_listen_ecnaive_load_recv_rank1_data1 error: " << e.what() << std::endl;
        throw;
    }
}

void AsioConnectionManager::bind_listen_ecnaive_load_recv_rank1_data0(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        ecnaive_load_recv_rank1_data0_acceptor_.open(endpoint.protocol());
        ecnaive_load_recv_rank1_data0_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        ecnaive_load_recv_rank1_data0_acceptor_.bind(endpoint);
        ecnaive_load_recv_rank1_data0_acceptor_.listen();
        std::cout << "EC-NAIVE: [Rank 2] Bound and listening on ecnaive_load_recv_rank1_data0 port " 
                  << port << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: bind_listen_ecnaive_load_recv_rank1_data0 error: " << e.what() << std::endl;
        throw;
    }
}

void AsioConnectionManager::bind_listen_ecnaive_load_recv_rank1_parity1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        ecnaive_load_recv_rank1_parity1_acceptor_.open(endpoint.protocol());
        ecnaive_load_recv_rank1_parity1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        ecnaive_load_recv_rank1_parity1_acceptor_.bind(endpoint);
        ecnaive_load_recv_rank1_parity1_acceptor_.listen();
        std::cout << "EC-NAIVE: [Rank 2] Bound and listening on ecnaive_load_recv_rank1_parity1 port " 
                  << port << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: bind_listen_ecnaive_load_recv_rank1_parity1 error: " << e.what() << std::endl;
        throw;
    }
}

void AsioConnectionManager::bind_listen_ecnaive_load_recv_rank3_data0(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        ecnaive_load_recv_rank3_data0_acceptor_.open(endpoint.protocol());
        ecnaive_load_recv_rank3_data0_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        ecnaive_load_recv_rank3_data0_acceptor_.bind(endpoint);
        ecnaive_load_recv_rank3_data0_acceptor_.listen();
        std::cout << "EC-NAIVE: [Rank 2] Bound and listening on ecnaive_load_recv_rank3_data0 port " 
                  << port << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: bind_listen_ecnaive_load_recv_rank3_data0 error: " << e.what() << std::endl;
        throw;
    }
}

void AsioConnectionManager::bind_listen_ecnaive_load_recv_rank0_data1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        ecnaive_load_recv_rank0_data1_acceptor_.open(endpoint.protocol());
        ecnaive_load_recv_rank0_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        ecnaive_load_recv_rank0_data1_acceptor_.bind(endpoint);
        ecnaive_load_recv_rank0_data1_acceptor_.listen();
        std::cout << "EC-NAIVE: [Rank 2] Bound and listening on ecnaive_load_recv_rank0_data1 port " 
                  << port << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: bind_listen_ecnaive_load_recv_rank0_data1 error: " << e.what() << std::endl;
        throw;
    }
}

// EC-NAIVE load mode accept helpers (for rank2, after bind+listen)
void AsioConnectionManager::accept_ecnaive_load_recv_rank3_data1() {
    try {
        ecnaive_load_recv_rank3_data1_acceptor_.accept(ecnaive_load_recv_rank3_data1_socket_);
        ecnaive_load_recv_rank3_data1_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 2] Accepted ecnaive_load_recv_rank3_data1 connection" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: accept_ecnaive_load_recv_rank3_data1 error: " << e.what() << std::endl;
        ecnaive_load_recv_rank3_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_ecnaive_load_recv_rank0_parity0() {
    try {
        ecnaive_load_recv_rank0_parity0_acceptor_.accept(ecnaive_load_recv_rank0_parity0_socket_);
        ecnaive_load_recv_rank0_parity0_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 2] Accepted ecnaive_load_recv_rank0_parity0 connection" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: accept_ecnaive_load_recv_rank0_parity0 error: " << e.what() << std::endl;
        ecnaive_load_recv_rank0_parity0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_ecnaive_load_recv_rank0_data0() {
    try {
        ecnaive_load_recv_rank0_data0_acceptor_.accept(ecnaive_load_recv_rank0_data0_socket_);
        ecnaive_load_recv_rank0_data0_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 2] Accepted ecnaive_load_recv_rank0_data0 connection" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: accept_ecnaive_load_recv_rank0_data0 error: " << e.what() << std::endl;
        ecnaive_load_recv_rank0_data0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_ecnaive_load_recv_rank1_data1() {
    try {
        ecnaive_load_recv_rank1_data1_acceptor_.accept(ecnaive_load_recv_rank1_data1_socket_);
        ecnaive_load_recv_rank1_data1_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 2] Accepted ecnaive_load_recv_rank1_data1 connection" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: accept_ecnaive_load_recv_rank1_data1 error: " << e.what() << std::endl;
        ecnaive_load_recv_rank1_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_ecnaive_load_recv_rank1_data0() {
    try {
        ecnaive_load_recv_rank1_data0_acceptor_.accept(ecnaive_load_recv_rank1_data0_socket_);
        ecnaive_load_recv_rank1_data0_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 2] Accepted ecnaive_load_recv_rank1_data0 connection" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: accept_ecnaive_load_recv_rank1_data0 error: " << e.what() << std::endl;
        ecnaive_load_recv_rank1_data0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_ecnaive_load_recv_rank1_parity1() {
    try {
        ecnaive_load_recv_rank1_parity1_acceptor_.accept(ecnaive_load_recv_rank1_parity1_socket_);
        ecnaive_load_recv_rank1_parity1_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 2] Accepted ecnaive_load_recv_rank1_parity1 connection" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: accept_ecnaive_load_recv_rank1_parity1 error: " << e.what() << std::endl;
        ecnaive_load_recv_rank1_parity1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_ecnaive_load_recv_rank3_data0() {
    try {
        ecnaive_load_recv_rank3_data0_acceptor_.accept(ecnaive_load_recv_rank3_data0_socket_);
        ecnaive_load_recv_rank3_data0_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 2] Accepted ecnaive_load_recv_rank3_data0 connection" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: accept_ecnaive_load_recv_rank3_data0 error: " << e.what() << std::endl;
        ecnaive_load_recv_rank3_data0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_ecnaive_load_recv_rank0_data1() {
    try {
        ecnaive_load_recv_rank0_data1_acceptor_.accept(ecnaive_load_recv_rank0_data1_socket_);
        ecnaive_load_recv_rank0_data1_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 2] Accepted ecnaive_load_recv_rank0_data1 connection" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: accept_ecnaive_load_recv_rank0_data1 error: " << e.what() << std::endl;
        ecnaive_load_recv_rank0_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

// EC-NAIVE load mode init functions (rank0/1/3 sender)
void AsioConnectionManager::init_ecnaive_load_send_rank0_parity0(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(ecnaive_load_send_rank0_parity0_socket_, endpoints);
        ecnaive_load_send_rank0_parity0_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 0] Connected ecnaive_load_send_rank0_parity0 to rank2" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: init_ecnaive_load_send_rank0_parity0 error: " << e.what() << std::endl;
        ecnaive_load_send_rank0_parity0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_ecnaive_load_send_rank3_data1(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(ecnaive_load_send_rank3_data1_socket_, endpoints);
        ecnaive_load_send_rank3_data1_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 3] Connected ecnaive_load_send_rank3_data1 to rank2" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: init_ecnaive_load_send_rank3_data1 error: " << e.what() << std::endl;
        ecnaive_load_send_rank3_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

// EC-NAIVE load mode init functions (rank0 sender: 3 connections)
void AsioConnectionManager::init_ecnaive_load_send_rank0_data0(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(ecnaive_load_send_rank0_data0_socket_, endpoints);
        ecnaive_load_send_rank0_data0_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 0] Connected ecnaive_load_send_rank0_data0 to rank2" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: init_ecnaive_load_send_rank0_data0 error: " << e.what() << std::endl;
        ecnaive_load_send_rank0_data0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_ecnaive_load_send_rank0_data1(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(ecnaive_load_send_rank0_data1_socket_, endpoints);
        ecnaive_load_send_rank0_data1_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 0] Connected ecnaive_load_send_rank0_data1 to rank2" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: init_ecnaive_load_send_rank0_data1 error: " << e.what() << std::endl;
        ecnaive_load_send_rank0_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

// EC-NAIVE load mode init functions (rank1 sender: 3 connections)
void AsioConnectionManager::init_ecnaive_load_send_rank1_data1(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(ecnaive_load_send_rank1_data1_socket_, endpoints);
        ecnaive_load_send_rank1_data1_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 1] Connected ecnaive_load_send_rank1_data1 to rank2" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: init_ecnaive_load_send_rank1_data1 error: " << e.what() << std::endl;
        ecnaive_load_send_rank1_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_ecnaive_load_send_rank1_data0(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(ecnaive_load_send_rank1_data0_socket_, endpoints);
        ecnaive_load_send_rank1_data0_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 1] Connected ecnaive_load_send_rank1_data0 to rank2" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: init_ecnaive_load_send_rank1_data0 error: " << e.what() << std::endl;
        ecnaive_load_send_rank1_data0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_ecnaive_load_send_rank1_parity1(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(ecnaive_load_send_rank1_parity1_socket_, endpoints);
        ecnaive_load_send_rank1_parity1_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 1] Connected ecnaive_load_send_rank1_parity1 to rank2" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: init_ecnaive_load_send_rank1_parity1 error: " << e.what() << std::endl;
        ecnaive_load_send_rank1_parity1_connected_ = false;
        connection_cv_.notify_all();
    }
}

// EC-NAIVE load mode init functions (rank3 sender: 2 connections)
void AsioConnectionManager::init_ecnaive_load_send_rank3_data0(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(ecnaive_load_send_rank3_data0_socket_, endpoints);
        ecnaive_load_send_rank3_data0_connected_ = true;
        std::cout << "EC-NAIVE: [Rank 3] Connected ecnaive_load_send_rank3_data0 to rank2" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "EC-NAIVE: init_ecnaive_load_send_rank3_data0 error: " << e.what() << std::endl;
        ecnaive_load_send_rank3_data0_connected_ = false;
        connection_cv_.notify_all();
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
    
    // EC-NAIVE load mode sockets
    if (ecnaive_load_recv_rank3_data1_socket_.is_open()) ecnaive_load_recv_rank3_data1_socket_.close();
    if (ecnaive_load_recv_rank0_parity0_socket_.is_open()) ecnaive_load_recv_rank0_parity0_socket_.close();
    if (ecnaive_load_recv_rank3_data1_acceptor_.is_open()) ecnaive_load_recv_rank3_data1_acceptor_.close();
    if (ecnaive_load_recv_rank0_parity0_acceptor_.is_open()) ecnaive_load_recv_rank0_parity0_acceptor_.close();
    if (ecnaive_load_send_rank0_parity0_socket_.is_open()) ecnaive_load_send_rank0_parity0_socket_.close();
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

// EC-NAIVE load mode task structures
struct LoadRecvTask {
    // 8个接收缓冲区地址（rank2接收）
    uintptr_t recv_p20_addr;   // p_{2,0} from rank0
    uintptr_t recv_d21_addr;   // d_{2,1} from rank3
    uintptr_t recv_d00_addr;   // d_{0,0} from rank0
    uintptr_t recv_d01_addr;   // d_{0,1} from rank1
    uintptr_t recv_d10_addr;   // d_{1,0} from rank1
    uintptr_t recv_p11_addr;   // p_{1,1} from rank1
    uintptr_t recv_d30_addr;   // d_{3,0} from rank3
    uintptr_t recv_d31_addr;   // d_{3,1} from rank0
    
    // 4个输出地址（直接写入ecnaive_blocks，零拷贝）
    uintptr_t output_data0_addr;           // d_{2,0} = p_{2,0} ⊕ d_{2,1}
    uintptr_t output_recv_parity0_addr;    // p_{0,0} = d_{0,0} ⊕ d_{0,1}
    uintptr_t output_recv_data1_addr;      // d_{1,1} = d_{1,0} ⊕ p_{1,1}
    uintptr_t output_recv_parity1_addr;    // p_{3,1} = d_{3,0} ⊕ d_{3,1}
    
    size_t size;
};

struct LoadXORTask {
    // 8个输入地址（接收缓冲区）
    uintptr_t recv_p20_addr;   // p_{2,0} from rank0
    uintptr_t recv_d21_addr;   // d_{2,1} from rank3
    uintptr_t recv_d00_addr;   // d_{0,0} from rank0
    uintptr_t recv_d01_addr;   // d_{0,1} from rank1
    uintptr_t recv_d10_addr;   // d_{1,0} from rank1
    uintptr_t recv_p11_addr;   // p_{1,1} from rank1
    uintptr_t recv_d30_addr;   // d_{3,0} from rank3
    uintptr_t recv_d31_addr;   // d_{3,1} from rank0
    
    // 4个输出地址（直接写入ecnaive_blocks，零拷贝）
    uintptr_t output_data0_addr;           // d_{2,0} = p_{2,0} ⊕ d_{2,1}
    uintptr_t output_recv_parity0_addr;    // p_{0,0} = d_{0,0} ⊕ d_{0,1}
    uintptr_t output_recv_data1_addr;      // d_{1,1} = d_{1,0} ⊕ p_{1,1}
    uintptr_t output_recv_parity1_addr;    // p_{3,1} = d_{3,0} ⊕ d_{3,1}
    
    size_t size;
};

struct LoadSendTask {
    uintptr_t send_addr;  // mmap地址或buffer地址
    size_t size;
    bool is_mmap;         // 标记是否为mmap（不需要释放）
    int socket_type;      // Socket类型标识: 0=parity0, 1=data0, 2=data1, 3=parity1 (用于区分不同socket)
};

class ECNaiveNative {
public:
    ECNaiveNative(const std::string& send_data1_ip, uint16_t send_data1_port,
                  const std::string& send_parity0_ip, uint16_t send_parity0_port,
                  const std::string& send_parity1_ip, uint16_t send_parity1_port,
                  const std::string& recv_parity1_ip, uint16_t recv_parity1_port,
                  const std::string& recv_parity0_ip, uint16_t recv_parity0_port,
                  const std::string& recv_data1_ip, uint16_t recv_data1_port,
                  bool use_rdma = false)
        : stop_(false),
          send_data1_ip_(send_data1_ip),
          send_data1_port_(send_data1_port),
          send_parity0_ip_(send_parity0_ip),
          send_parity0_port_(send_parity0_port),
          send_parity1_ip_(send_parity1_ip),
          send_parity1_port_(send_parity1_port),
          recv_parity1_ip_(recv_parity1_ip),
          recv_parity1_port_(recv_parity1_port),
          recv_parity0_ip_(recv_parity0_ip),
          recv_parity0_port_(recv_parity0_port),
          recv_data1_ip_(recv_data1_ip),
          recv_data1_port_(recv_data1_port),
          use_rdma_(use_rdma),
          rdma_context_(nullptr),
          rdma_pd_(nullptr),
          rdma_send_cq_(nullptr),
          rdma_recv_cq_(nullptr),
          k_(2),
          rows_(2),
          a_mat_(nullptr),
          g_tbls_(nullptr),
          rank_(-1) {  // Will be set in init_load_connections or set_load_mode
        // Initialize EC encoding tables
        init_ec_encoding();
        
        std::cout << "ECNAIVE: Initializing connections (RDMA: " << (use_rdma_ ? "enabled" : "disabled") << ")..." << std::endl;
        
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
        start_threads();
        std::cout << "ECNAIVE: Pipeline started successfully" << std::endl;
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
            std::cout << "[ECNAIVE RDMA] Buffer already registered at 0x" << std::hex << addr << std::dec << std::endl;
            return;
        }
        
        std::cout << "[ECNAIVE RDMA] Registering buffer at 0x" << std::hex << addr << std::dec 
                  << ", size: " << (size / (1024.0 * 1024.0)) << " MB" << std::endl;
        
        ibv_mr* mr = ibv_reg_mr(rdma_pd_, reinterpret_cast<void*>(addr), size,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        
        if (!mr) {
            throw std::runtime_error("Failed to register memory region for RDMA");
        }
        
        rdma_registered_buffers_[addr] = {mr, addr, size};
        std::cout << "[ECNAIVE RDMA] Buffer registered successfully (total: " << rdma_registered_buffers_.size() << ")" << std::endl;
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
            std::cout << "[ECNAIVE RDMA] Buffer unregistered at 0x" << std::hex << addr << std::dec << std::endl;
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
        
        // Step 1: Encode data blocks to get parity blocks
        encode_ec_blocks(data0_addr, data1_addr, parity0_addr, parity1_addr, size);
        
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
        std::cout << "ECNAIVE: Submitting sentinel to send_data1 pipeline" << std::endl;
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

    // ========== EC-NAIVE Load Mode Submit Interfaces ==========

    // Unified load recovery interface (rank2 only)
    void submit_ecnaive_load_recovery(
        uintptr_t recv_data1_addr,      // d_{3,1} 直接放到最终位置（data0连续内存）
        uintptr_t recv_parity0_addr,    // p_{0,0} 临时recv buffer地址
        uintptr_t recovered_data0_addr, // d_{2,0} 输出地址（与recv_data1_addr相同）
        size_t size
    ) {
        if (rank_ != 2 || !is_load_mode_) {
            std::cerr << "EC-NAIVE: submit_ecnaive_load_recovery called but rank != 2 or not in load mode" << std::endl;
            return;
        }
        
        std::cout << "EC-NAIVE: [Rank 2] Submitting load recovery task (legacy): recv_data1=" << recv_data1_addr
                  << ", recv_parity0=" << recv_parity0_addr
                  << ", recovered_data0=" << recovered_data0_addr
                  << ", size=" << size << std::endl;
        
        // Map old interface to new structure
        // Old logic: d_{2,0} = d_{3,1} XOR p_{0,0}
        // Note: This doesn't match ECNAIVE encoding (d_{2,0} = p_{2,0} XOR d_{2,1})
        // For backward compatibility, map old parameters to new structure
        // Since old interface only provides 2 addresses, we map them and set others to 0
        LoadRecvTask task;
        // Map old parameters - assuming old code meant p_{2,0} instead of p_{0,0} for recovery
        task.recv_p20_addr = recv_parity0_addr;       // p_{2,0} from rank0 (mapped from old p_{0,0})
        task.recv_d31_addr = recv_data1_addr;         // d_{3,1} from rank0
        // Set other addresses to 0 (not used in legacy mode)
        task.recv_d21_addr = 0;
        task.recv_d00_addr = 0;
        task.recv_d01_addr = 0;
        task.recv_d10_addr = 0;
        task.recv_p11_addr = 0;
        task.recv_d30_addr = 0;
        // Output addresses - only data0 is recovered in legacy mode
        task.output_data0_addr = recovered_data0_addr;
        task.output_recv_parity0_addr = 0;
        task.output_recv_data1_addr = 0;
        task.output_recv_parity1_addr = 0;
        task.size = size;
        
        {
            std::lock_guard<std::mutex> lock(load_recv_queue_mutex_);
            load_recv_queue_.push(task);
        }
        load_recv_queue_cv_.notify_one();
    }

    // Full recovery interface (8 recv + 4 XOR)
    void submit_ecnaive_load_recovery_full(
        uintptr_t recv_p20_addr, uintptr_t recv_d21_addr,
        uintptr_t recv_d00_addr, uintptr_t recv_d01_addr,
        uintptr_t recv_d10_addr, uintptr_t recv_p11_addr,
        uintptr_t recv_d30_addr, uintptr_t recv_d31_addr,
        uintptr_t output_data0_addr,
        uintptr_t output_recv_parity0_addr,
        uintptr_t output_recv_data1_addr,
        uintptr_t output_recv_parity1_addr,
        size_t size
    ) {
        if (rank_ != 2 || !is_load_mode_) {
            std::cerr << "EC-NAIVE: submit_ecnaive_load_recovery_full called but rank != 2 or not in load mode" << std::endl;
            return;
        }
        
        std::cout << "EC-NAIVE: [Rank 2] Submitting full recovery task (8 recv + 4 XOR): size=" << size << std::endl;
        
        LoadRecvTask task;
        task.recv_p20_addr = recv_p20_addr;
        task.recv_d21_addr = recv_d21_addr;
        task.recv_d00_addr = recv_d00_addr;
        task.recv_d01_addr = recv_d01_addr;
        task.recv_d10_addr = recv_d10_addr;
        task.recv_p11_addr = recv_p11_addr;
        task.recv_d30_addr = recv_d30_addr;
        task.recv_d31_addr = recv_d31_addr;
        task.output_data0_addr = output_data0_addr;
        task.output_recv_parity0_addr = output_recv_parity0_addr;
        task.output_recv_data1_addr = output_recv_data1_addr;
        task.output_recv_parity1_addr = output_recv_parity1_addr;
        task.size = size;
        
        {
            std::lock_guard<std::mutex> lock(load_recv_queue_mutex_);
            load_recv_queue_.push(task);
        }
        load_recv_queue_cv_.notify_one();
    }

    // Rank0 send interfaces (3 blocks: p_{2,0}, d_{0,0}, d_{3,1})
    void submit_load_send_rank0_parity0(uintptr_t send_addr, size_t size) {
        if (rank_ != 0 || !is_load_mode_) {
            std::cerr << "EC-NAIVE: submit_load_send_rank0_parity0 called but rank != 0 or not in load mode" << std::endl;
            return;
        }
        
        std::cout << "EC-NAIVE: [Rank 0] Submitting load send task (p_{2,0}): send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        
        LoadSendTask task;
        task.send_addr = send_addr;
        task.size = size;
        task.is_mmap = false;
        task.socket_type = 0;  // parity0 socket
        
        {
            std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
            load_send_queue_.push(task);
        }
        load_send_queue_cv_.notify_one();
    }

    void submit_load_send_rank0_data0(uintptr_t send_addr, size_t size) {
        if (rank_ != 0 || !is_load_mode_) {
            std::cerr << "EC-NAIVE: submit_load_send_rank0_data0 called but rank != 0 or not in load mode" << std::endl;
            return;
        }
        
        std::cout << "EC-NAIVE: [Rank 0] Submitting load send task (d_{0,0}): send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        
        LoadSendTask task;
        task.send_addr = send_addr;
        task.size = size;
        task.is_mmap = false;
        task.socket_type = 1;  // data0 socket
        
        {
            std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
            load_send_queue_.push(task);
        }
        load_send_queue_cv_.notify_one();
    }

    void submit_load_send_rank0_data1(uintptr_t send_addr, size_t size) {
        if (rank_ != 0 || !is_load_mode_) {
            std::cerr << "EC-NAIVE: submit_load_send_rank0_data1 called but rank != 0 or not in load mode" << std::endl;
            return;
        }
        
        std::cout << "EC-NAIVE: [Rank 0] Submitting load send task (d_{3,1}): send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        
        LoadSendTask task;
        task.send_addr = send_addr;
        task.size = size;
        task.is_mmap = false;
        task.socket_type = 2;  // data1 socket
        
        {
            std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
            load_send_queue_.push(task);
        }
        load_send_queue_cv_.notify_one();
    }

    // Rank1 send interfaces (3 blocks: d_{0,1}, d_{1,0}, p_{1,1})
    void submit_load_send_rank1_data1(uintptr_t send_addr, size_t size) {
        if (rank_ != 1 || !is_load_mode_) {
            std::cerr << "EC-NAIVE: submit_load_send_rank1_data1 called but rank != 1 or not in load mode" << std::endl;
            return;
        }
        
        std::cout << "EC-NAIVE: [Rank 1] Submitting load send task (d_{0,1}): send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        
        LoadSendTask task;
        task.send_addr = send_addr;
        task.size = size;
        task.is_mmap = false;
        task.socket_type = 2;  // data1 socket
        
        {
            std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
            load_send_queue_.push(task);
        }
        load_send_queue_cv_.notify_one();
    }

    void submit_load_send_rank1_data0(uintptr_t send_addr, size_t size) {
        if (rank_ != 1 || !is_load_mode_) {
            std::cerr << "EC-NAIVE: submit_load_send_rank1_data0 called but rank != 1 or not in load mode" << std::endl;
            return;
        }
        
        std::cout << "EC-NAIVE: [Rank 1] Submitting load send task (d_{1,0}): send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        
        LoadSendTask task;
        task.send_addr = send_addr;
        task.size = size;
        task.is_mmap = false;
        task.socket_type = 1;  // data0 socket
        
        {
            std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
            load_send_queue_.push(task);
        }
        load_send_queue_cv_.notify_one();
    }

    void submit_load_send_rank1_parity1(uintptr_t send_addr, size_t size) {
        if (rank_ != 1 || !is_load_mode_) {
            std::cerr << "EC-NAIVE: submit_load_send_rank1_parity1 called but rank != 1 or not in load mode" << std::endl;
            return;
        }
        
        std::cout << "EC-NAIVE: [Rank 1] Submitting load send task (p_{1,1}): send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        
        LoadSendTask task;
        task.send_addr = send_addr;
        task.size = size;
        task.is_mmap = false;
        task.socket_type = 3;  // parity1 socket
        
        {
            std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
            load_send_queue_.push(task);
        }
        load_send_queue_cv_.notify_one();
    }

    // Rank3 send interfaces (2 blocks: d_{2,1}, d_{3,0})
    void submit_load_send_rank3_data1(uintptr_t send_addr, size_t size) {
        if (rank_ != 3 || !is_load_mode_) {
            std::cerr << "EC-NAIVE: submit_load_send_rank3_data1 called but rank != 3 or not in load mode" << std::endl;
            return;
        }
        
        std::cout << "EC-NAIVE: [Rank 3] Submitting load send task (d_{2,1}): send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        
        LoadSendTask task;
        task.send_addr = send_addr;
        task.size = size;
        task.is_mmap = false;
        task.socket_type = 2;  // data1 socket
        
        {
            std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
            load_send_queue_.push(task);
        }
        load_send_queue_cv_.notify_one();
    }

    void submit_load_send_rank3_data0(uintptr_t send_addr, size_t size) {
        if (rank_ != 3 || !is_load_mode_) {
            std::cerr << "EC-NAIVE: submit_load_send_rank3_data0 called but rank != 3 or not in load mode" << std::endl;
            return;
        }
        
        std::cout << "EC-NAIVE: [Rank 3] Submitting load send task (d_{3,0}): send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        
        LoadSendTask task;
        task.send_addr = send_addr;
        task.size = size;
        task.is_mmap = false;
        task.socket_type = 1;  // data0 socket
        
        {
            std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
            load_send_queue_.push(task);
        }
        load_send_queue_cv_.notify_one();
    }

    // Load mode sentinel interfaces
    void submit_load_recv_sentinel() {
        if (rank_ != 2 || !is_load_mode_) return;
        
        std::cout << "EC-NAIVE: [Rank 2] Submitting sentinel to load recv worker" << std::endl;
        
        LoadRecvTask sentinel;
        sentinel.recv_p20_addr = 0;
        sentinel.recv_d21_addr = 0;
        sentinel.recv_d00_addr = 0;
        sentinel.recv_d01_addr = 0;
        sentinel.recv_d10_addr = 0;
        sentinel.recv_p11_addr = 0;
        sentinel.recv_d30_addr = 0;
        sentinel.recv_d31_addr = 0;
        sentinel.size = 0;
        
        {
            std::lock_guard<std::mutex> lock(load_recv_queue_mutex_);
            load_recv_queue_.push(sentinel);
        }
        load_recv_queue_cv_.notify_one();
    }

    void submit_load_xor_sentinel() {
        if (rank_ != 2 || !is_load_mode_) return;
        
        std::cout << "EC-NAIVE: [Rank 2] Submitting sentinel to load xor worker" << std::endl;
        
        LoadXORTask sentinel;
        sentinel.recv_p20_addr = 0;
        sentinel.recv_d21_addr = 0;
        sentinel.recv_d00_addr = 0;
        sentinel.recv_d01_addr = 0;
        sentinel.recv_d10_addr = 0;
        sentinel.recv_p11_addr = 0;
        sentinel.recv_d30_addr = 0;
        sentinel.recv_d31_addr = 0;
        sentinel.size = 0;
        
        {
            std::lock_guard<std::mutex> lock(load_xor_queue_mutex_);
            load_xor_queue_.push(sentinel);
        }
        load_xor_queue_cv_.notify_one();
    }

    void submit_load_send_sentinel() {
        if ((rank_ != 0 && rank_ != 1 && rank_ != 3) || !is_load_mode_) return;
        
        std::cout << "EC-NAIVE: [Rank " << rank_ << "] Submitting sentinel to load send worker" << std::endl;
        
        LoadSendTask sentinel;
        sentinel.send_addr = 0;
        sentinel.size = 0;
        sentinel.socket_type = -1;  // Sentinel marker
        
        {
            std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
            load_send_queue_.push(sentinel);
        }
        load_send_queue_cv_.notify_one();
    }

    // Software failure mode interfaces (direct send/recv without pipeline)

    // rank3 software failure mode send d21 (once complete transmission)
    void software_send_rank3_data1(uintptr_t send_addr, size_t size) {
        if (rank_ != 3) return;

        // check connection status
        if (!conn_.is_ecnaive_load_send_rank3_data1_connected()) {
            std::cerr << "EC-NAIVE: [Rank 3] Software send socket not connected" << std::endl;
            return;
        }

        auto& socket = conn_.get_ecnaive_load_send_rank3_data1_socket();
        if (!send_with_size(socket, send_addr, size)) {
            std::cerr << "EC-NAIVE: [Rank 3] Software send failed" << std::endl;
        } else {
            std::cout << "EC-NAIVE: [Rank 3] Software sent d21 (size=" << size << ")" << std::endl;
        }
    }

    // rank2 software failure mode receive d21 (once complete transmission)
    void software_recv_data1(uintptr_t recv_addr, size_t size) {
        if (rank_ != 2) return;

        // check connection status
        if (!conn_.is_ecnaive_load_recv_rank3_data1_connected()) {
            std::cerr << "EC-NAIVE: [Rank 2] Software recv socket not connected" << std::endl;
            return;
        }

        auto& socket = conn_.get_ecnaive_load_recv_rank3_data1_socket();
        if (!recv_with_size_bool(socket, reinterpret_cast<void*>(recv_addr), size)) {
            std::cerr << "EC-NAIVE: [Rank 2] Software recv failed" << std::endl;
        } else {
            std::cout << "EC-NAIVE: [Rank 2] Software received d21 (size=" << size << ")" << std::endl;
        }
    }

    // Wait for load completion (rank2 only)
    void wait_for_load_completion() {
        if (rank_ != 2 || !is_load_mode_) return;
        
        std::cout << "EC-NAIVE: [Rank 2] Waiting for load workers to complete..." << std::endl;
        
        int wait_count = 0;
        while (!load_recv_worker_completed_ || !load_xor_worker_completed_) {
            if (wait_count % 100 == 0) {
                std::cout << "EC-NAIVE: [Rank 2] Waiting for load workers: "
                          << "recv=" << (load_recv_worker_completed_ ? "true" : "false")
                          << ", xor=" << (load_xor_worker_completed_ ? "true" : "false") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
        }
        std::cout << "EC-NAIVE: [Rank 2] All load workers completed" << std::endl;
    }

    // Release helpers: Python can poll these to free buffers.
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
        // Save mode flags
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

        // Clear queues
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

        std::cout << "ECNAIVE: Reset encoding completion flags and cleared all queues" << std::endl;
    }

    void wait_for_encoding_completion() {
        // Wait for all 6 workers
        int wait_count = 0;
        while (!send_data1_completed_ || !send_parity0_completed_ || !send_parity1_completed_ ||
               !recv_parity1_completed_ || !recv_parity0_completed_ || !recv_data1_completed_) {
            if (wait_count % 100 == 0) {
                std::cout << "ECNAIVE: Waiting for workers: "
                          << "s_d1=" << (send_data1_completed_ ? "true" : "false")
                          << ", s_p0=" << (send_parity0_completed_ ? "true" : "false")
                          << ", s_p1=" << (send_parity1_completed_ ? "true" : "false")
                          << ", r_p1=" << (recv_parity1_completed_ ? "true" : "false")
                          << ", r_p0=" << (recv_parity0_completed_ ? "true" : "false")
                          << ", r_d1=" << (recv_data1_completed_ ? "true" : "false") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
        }
        std::cout << "ECNAIVE: All workers completed" << std::endl;
    }

    void stop() {
        bool expected = false;
        if (!stop_.compare_exchange_strong(expected, true)) {
            return;  // already stopped
        }
        send_data1_cv_.notify_all();
        send_parity0_cv_.notify_all();
        send_parity1_cv_.notify_all();
        recv_parity1_cv_.notify_all();
        recv_parity0_cv_.notify_all();
        recv_data1_cv_.notify_all();
        
        // Notify load mode workers to stop
        if (is_load_mode_) {
            load_recv_queue_cv_.notify_all();
            load_xor_queue_cv_.notify_all();
            load_send_queue_cv_.notify_all();
        }
        
        if (send_data1_thread_.joinable()) send_data1_thread_.join();
        if (send_parity0_thread_.joinable()) send_parity0_thread_.join();
        if (send_parity1_thread_.joinable()) send_parity1_thread_.join();
        if (recv_parity1_thread_.joinable()) recv_parity1_thread_.join();
        if (recv_parity0_thread_.joinable()) recv_parity0_thread_.join();
        if (recv_data1_thread_.joinable()) recv_data1_thread_.join();
        
        // Load mode worker cleanup
        if (is_load_mode_ && rank_ == 2) {
            if (load_recv_worker_.joinable()) {
                load_recv_worker_.join();
            }
            if (load_xor_worker_.joinable()) {
                load_xor_worker_.join();
            }
        } else if (is_load_mode_ && (rank_ == 0 || rank_ == 1 || rank_ == 3)) {
            if (load_send_worker_.joinable()) {
                load_send_worker_.join();
            }
        }
        
        conn_.cleanup();
    }

    // Load mode functions
    void set_load_mode(bool is_load, int failed_rank, int rank = -1) {
        is_load_mode_ = is_load;
        failed_rank_ = failed_rank;
        failed_rank_in_group_ = (failed_rank >= 0)
            ? (failed_rank % ECNAIVE_RANKS_PER_GROUP)
            : -1;
        if (rank >= 0) {
            rank_ = rank;
        }
        std::cout << "EC-NAIVE: Set load mode: "
                  << (is_load ? "true" : "false") << ", failed_rank=" << failed_rank
                  << ", failed_rank_in_group=" << failed_rank_in_group_ << ", rank=" << rank_ << std::endl;
        
        if (is_load && failed_rank_in_group_ == 2) {
            // Reset flags
            load_recv_worker_completed_ = false;
            load_xor_worker_completed_ = false;
            load_send_worker_completed_ = false;
            load_recv_sentinel_received_ = false;
            load_xor_sentinel_received_ = false;
            load_send_sentinel_received_ = false;
            
            // Start workers based on rank_in_group (rank_ is set in init_ecnaive_load_connections)
            if (rank_ == 2) {
                // rank2: start recv and xor workers
                if (!load_recv_worker_.joinable()) {
                    load_recv_worker_ = std::thread(&ECNaiveNative::load_recv_worker, this);
                }
                if (!load_xor_worker_.joinable()) {
                    load_xor_worker_ = std::thread(&ECNaiveNative::load_xor_worker, this);
                }
            } else if (rank_ == 0 || rank_ == 1 || rank_ == 3) {
                // rank0/1/3: start send worker
                if (!load_send_worker_.joinable()) {
                    load_send_worker_ = std::thread(&ECNaiveNative::load_send_worker, this);
                }
            }
        }
    }

    void init_load_connections(
        int rank,
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
        
        std::cout << "ECLATIN: [Rank " << rank << "] Initializing load connections..." << std::endl;
        
        if (rank == 2) {
            // rank2: Initialize 6 recv sockets (accept connections from rank0/1/3)
            // Step 1: First, bind and listen all acceptors synchronously (before accept)
            std::cout << "ECLATIN: [Rank 2] Binding and listening all acceptors..." << std::endl;
            try {
                conn_.bind_listen_load_recv_rank0_data2(rank2_ip, load_recv_rank0_data2_port);
                conn_.bind_listen_load_recv_rank0_parity2(rank2_ip, load_recv_rank0_parity2_port);
                conn_.bind_listen_load_recv_rank1_data1(rank2_ip, load_recv_rank1_data1_port);
                conn_.bind_listen_load_recv_rank1_parity1(rank2_ip, load_recv_rank1_parity1_port);
                conn_.bind_listen_load_recv_rank3_data1(rank2_ip, load_recv_rank3_data1_port);
                conn_.bind_listen_load_recv_rank3_data2(rank2_ip, load_recv_rank3_data2_port);
                
                std::cout << "ECLATIN: [Rank 2] All acceptors bound and listening" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "ECLATIN: [Rank 2] Failed to bind/listen acceptors: " << e.what() << std::endl;
                throw;
            }
            
            // Step 2: Start accept operations in separate threads (similar to EC-CHECK)
            // These threads will block on accept() until connections arrive
            std::thread recv_init_thread([this]() {
                std::thread r0_d2([this]() {
                    conn_.accept_load_recv_rank0_data2();
                });
                std::thread r0_p2([this]() {
                    conn_.accept_load_recv_rank0_parity2();
                });
                std::thread r1_d1([this]() {
                    conn_.accept_load_recv_rank1_data1();
                });
                std::thread r1_p1([this]() {
                    conn_.accept_load_recv_rank1_parity1();
                });
                std::thread r3_d1([this]() {
                    conn_.accept_load_recv_rank3_data1();
                });
                std::thread r3_d2([this]() {
                    conn_.accept_load_recv_rank3_data2();
                });
                r0_d2.join();
                r0_p2.join();
                r1_d1.join();
                r1_p1.join();
                r3_d1.join();
                r3_d2.join();
            });
            
            // Step 3: Small delay to ensure accept sockets are bound and listening (similar to EC-CHECK)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Step 4: Detach the recv_init_thread so it runs in background
            // The accept operations will block until connections arrive from rank0/1/3
            recv_init_thread.detach();
            
            std::cout << "ECLATIN: [Rank 2] Accept threads started, waiting for connections..." << std::endl;
        } else {
            // rank0/1/3: Initialize 2 send sockets each (connect to rank2)
            if (rank == 0) {
                std::cout << "ECLATIN: [Rank 0] Connecting load send sockets to rank2..." << std::endl;
                conn_.init_load_send_rank0_data2(rank2_ip, load_recv_rank0_data2_port);
                conn_.init_load_send_rank0_parity2(rank2_ip, load_recv_rank0_parity2_port);
                std::cout << "ECLATIN: [Rank 0] Load send sockets connected" << std::endl;
            } else if (rank == 1) {
                std::cout << "ECLATIN: [Rank 1] Connecting load send sockets to rank2..." << std::endl;
                conn_.init_load_send_rank1_data1(rank2_ip, load_recv_rank1_data1_port);
                conn_.init_load_send_rank1_parity1(rank2_ip, load_recv_rank1_parity1_port);
                std::cout << "ECLATIN: [Rank 1] Load send sockets connected" << std::endl;
            } else if (rank == 3) {
                std::cout << "ECLATIN: [Rank 3] Connecting load send sockets to rank2..." << std::endl;
                conn_.init_load_send_rank3_data1(rank2_ip, load_recv_rank3_data1_port);
                conn_.init_load_send_rank3_data2(rank2_ip, load_recv_rank3_data2_port);
                std::cout << "ECLATIN: [Rank 3] Load send sockets connected" << std::endl;
            }
        }
        
            std::cout << "ECLATIN: [Rank " << rank << "] Load connections initialized" << std::endl;
    }

    // EC-NAIVE load mode connection initialization (rank_in_group 2 is receiver per group)
    void init_ecnaive_load_connections(
        int rank_in_group,
        const std::string& rank2_ip,
        uint16_t load_recv_rank3_data1_port,      // port 0: d_{2,1} from rank3
        uint16_t load_recv_rank0_parity0_port,    // port 1: p_{2,0} from rank0
        uint16_t load_recv_rank0_data0_port,      // port 2: d_{0,0} from rank0
        uint16_t load_recv_rank1_data1_port,      // port 3: d_{0,1} from rank1
        uint16_t load_recv_rank1_data0_port,      // port 4: d_{1,0} from rank1
        uint16_t load_recv_rank1_parity1_port,    // port 5: p_{1,1} from rank1
        uint16_t load_recv_rank3_data0_port,      // port 6: d_{3,0} from rank3
        uint16_t load_recv_rank0_data1_port       // port 7: d_{3,1} from rank0
    ) {
        if (!is_load_mode_) {
            std::cerr << "EC-NAIVE: init_ecnaive_load_connections called but not in load mode" << std::endl;
            return;
        }
        
        rank_ = rank_in_group;  // For worker threads (receiver/sender role)
        
        std::cout << "EC-NAIVE: [Rank_in_group " << rank_in_group << "] Initializing load connections (full recovery: 8 ports)..." << std::endl;
        
        if (rank_in_group == 2) {
            // rank2: bind, listen, and accept on 8 ports
            try {
                // Bind and listen all 8 acceptors
                conn_.bind_listen_ecnaive_load_recv_rank3_data1(rank2_ip, load_recv_rank3_data1_port);
                conn_.bind_listen_ecnaive_load_recv_rank0_parity0(rank2_ip, load_recv_rank0_parity0_port);
                conn_.bind_listen_ecnaive_load_recv_rank0_data0(rank2_ip, load_recv_rank0_data0_port);
                conn_.bind_listen_ecnaive_load_recv_rank1_data1(rank2_ip, load_recv_rank1_data1_port);
                conn_.bind_listen_ecnaive_load_recv_rank1_data0(rank2_ip, load_recv_rank1_data0_port);
                conn_.bind_listen_ecnaive_load_recv_rank1_parity1(rank2_ip, load_recv_rank1_parity1_port);
                conn_.bind_listen_ecnaive_load_recv_rank3_data0(rank2_ip, load_recv_rank3_data0_port);
                conn_.bind_listen_ecnaive_load_recv_rank0_data1(rank2_ip, load_recv_rank0_data1_port);
                
                std::cout << "EC-NAIVE: [Rank_in_group 2] All 8 acceptors bound and listening, starting accept threads..." << std::endl;
                
                // Start accept operations in separate threads
                // These threads will block on accept() until connections arrive
                std::thread recv_init_thread([this]() {
                    std::thread accept_threads[8];
                    
                    accept_threads[0] = std::thread([this]() {
                        conn_.accept_ecnaive_load_recv_rank3_data1();
                    });
                    accept_threads[1] = std::thread([this]() {
                        conn_.accept_ecnaive_load_recv_rank0_parity0();
                    });
                    accept_threads[2] = std::thread([this]() {
                        conn_.accept_ecnaive_load_recv_rank0_data0();
                    });
                    accept_threads[3] = std::thread([this]() {
                        conn_.accept_ecnaive_load_recv_rank1_data1();
                    });
                    accept_threads[4] = std::thread([this]() {
                        conn_.accept_ecnaive_load_recv_rank1_data0();
                    });
                    accept_threads[5] = std::thread([this]() {
                        conn_.accept_ecnaive_load_recv_rank1_parity1();
                    });
                    accept_threads[6] = std::thread([this]() {
                        conn_.accept_ecnaive_load_recv_rank3_data0();
                    });
                    accept_threads[7] = std::thread([this]() {
                        conn_.accept_ecnaive_load_recv_rank0_data1();
                    });
                    
                    // Join all accept threads
                    for (auto& t : accept_threads) {
                        t.join();
                    }
                });
                
                // Small delay to ensure accept sockets are bound and listening
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                // Detach the recv_init_thread so it runs in background
                // The accept operations will block until connections arrive from rank0/1/3
                recv_init_thread.detach();
                
                std::cout << "EC-NAIVE: [Rank_in_group 2] All 8 accept threads started, waiting for connections..." << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "EC-NAIVE: [Rank_in_group 2] Failed to initialize load connections: " << e.what() << std::endl;
                throw;
            }
        } else if (rank_in_group == 0) {
            // rank_in_group 0: connect to receiver on 3 ports (p_{2,0}, d_{0,0}, d_{3,1})
            std::cout << "EC-NAIVE: [Rank_in_group 0] Connecting to receiver on 3 ports..." << std::endl;
            conn_.init_ecnaive_load_send_rank0_parity0(rank2_ip, load_recv_rank0_parity0_port);  // p_{2,0}
            conn_.init_ecnaive_load_send_rank0_data0(rank2_ip, load_recv_rank0_data0_port);      // d_{0,0}
            conn_.init_ecnaive_load_send_rank0_data1(rank2_ip, load_recv_rank0_data1_port);      // d_{3,1}
            std::cout << "EC-NAIVE: [Rank_in_group 0] All 3 load connections established" << std::endl;
        } else if (rank_in_group == 1) {
            // rank_in_group 1: connect to receiver on 3 ports (d_{0,1}, d_{1,0}, p_{1,1})
            std::cout << "EC-NAIVE: [Rank_in_group 1] Connecting to receiver on 3 ports..." << std::endl;
            conn_.init_ecnaive_load_send_rank1_data1(rank2_ip, load_recv_rank1_data1_port);      // d_{0,1}
            conn_.init_ecnaive_load_send_rank1_data0(rank2_ip, load_recv_rank1_data0_port);      // d_{1,0}
            conn_.init_ecnaive_load_send_rank1_parity1(rank2_ip, load_recv_rank1_parity1_port);  // p_{1,1}
            std::cout << "EC-NAIVE: [Rank_in_group 1] All 3 load connections established" << std::endl;
        } else if (rank_in_group == 3) {
            // rank_in_group 3: connect to receiver on 2 ports (d_{2,1}, d_{3,0})
            std::cout << "EC-NAIVE: [Rank_in_group 3] Connecting to receiver on 2 ports..." << std::endl;
            conn_.init_ecnaive_load_send_rank3_data1(rank2_ip, load_recv_rank3_data1_port);     // d_{2,1}
            conn_.init_ecnaive_load_send_rank3_data0(rank2_ip, load_recv_rank3_data0_port);     // d_{3,0}
            std::cout << "EC-NAIVE: [Rank_in_group 3] All 2 load connections established" << std::endl;
        }
    }
    
    void wait_for_load_connections(int timeout_seconds = 30) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: wait_for_load_connections called but not in load mode" << std::endl;
            return;
        }
        conn_.wait_for_load_connections(timeout_seconds);
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
        
        // Step 1: Parallel receive all 6 blocks using threads
        std::vector<std::exception_ptr> recv_exceptions(6);
        std::vector<std::thread> recv_threads;
        
        recv_threads.emplace_back([&]() {
            try {
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
                if (!recv_with_size_bool(conn_.get_load_recv_rank3_data2_socket(), 
                                        reinterpret_cast<void*>(rank3_data2_addr), size)) {
                    throw std::runtime_error("Failed to receive rank3_data2");
                }
            } catch (...) {
                recv_exceptions[5] = std::current_exception();
            }
        });
        
        // Join all receive threads
        for (auto& t : recv_threads) {
            t.join();
        }
        
        // Check for exceptions
        for (size_t i = 0; i < recv_exceptions.size(); ++i) {
            if (recv_exceptions[i]) {
                std::rethrow_exception(recv_exceptions[i]);
            }
        }
        
        std::cout << "ECLATIN: [Rank 2] All 6 blocks received" << std::endl;
        
        // Step 2: Parallel XOR recoveries using threads
        std::vector<std::exception_ptr> xor_exceptions(4);
        std::vector<std::thread> xor_threads;
        
        // data1 = rank0.data2 XOR rank1.parity1
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_data1_addr), 
                           reinterpret_cast<void*>(rank0_data2_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_data1_addr), 
                                     reinterpret_cast<void*>(rank1_parity1_addr)};
                xor_gen(2, static_cast<int>(size), xor_array);
            } catch (...) {
                xor_exceptions[0] = std::current_exception();
            }
        });
        
        // data2 = rank0.parity2 XOR rank1.data1
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_data2_addr), 
                           reinterpret_cast<void*>(rank0_parity2_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_data2_addr), 
                                     reinterpret_cast<void*>(rank1_data1_addr)};
                xor_gen(2, static_cast<int>(size), xor_array);
            } catch (...) {
                xor_exceptions[1] = std::current_exception();
            }
        });
        
        // parity1 = rank1.data1 XOR rank3.data2
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_parity1_addr), 
                           reinterpret_cast<void*>(rank1_data1_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_parity1_addr), 
                                     reinterpret_cast<void*>(rank3_data2_addr)};
                xor_gen(2, static_cast<int>(size), xor_array);
            } catch (...) {
                xor_exceptions[2] = std::current_exception();
            }
        });
        
        // parity2 = rank0.data2 XOR rank3.data1
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_parity2_addr), 
                           reinterpret_cast<void*>(rank0_data2_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_parity2_addr), 
                                     reinterpret_cast<void*>(rank3_data1_addr)};
                xor_gen(2, static_cast<int>(size), xor_array);
            } catch (...) {
                xor_exceptions[3] = std::current_exception();
            }
        });
        
        // Join all XOR threads
        for (auto& t : xor_threads) {
            t.join();
        }
        
        // Check for exceptions
        for (size_t i = 0; i < xor_exceptions.size(); ++i) {
            if (xor_exceptions[i]) {
                std::rethrow_exception(xor_exceptions[i]);
            }
        }
        
        std::cout << "ECLATIN: [Rank 2] Recovery completed successfully" << std::endl;
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
        
        // Helper function to get socket by block name
        auto get_socket = [this](const std::string& block_name) -> boost::asio::ip::tcp::socket* {
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
        
        // Parallel send using two threads
        std::exception_ptr thread1_exception = nullptr;
        std::exception_ptr thread2_exception = nullptr;
        
        std::thread thread1([&]() {
            try {
                boost::asio::ip::tcp::socket* sock = get_socket(block1_name);
                if (sock == nullptr || !sock->is_open()) {
                    throw std::runtime_error("ECLATIN: load_send_blocks socket not available for " + block1_name);
                }
                
                std::cout << "ECLATIN: Sending " << block1_name << " to rank2 (size=" << size << ")" << std::endl;
                
                if (!send_with_size(*sock, block1_addr, size)) {
                    throw std::runtime_error("ECLATIN: load_send_blocks send failed for " + block1_name);
                }
                
                std::cout << "ECLATIN: Successfully sent " << block1_name << " to rank2" << std::endl;
            } catch (...) {
                thread1_exception = std::current_exception();
            }
        });
        
        std::thread thread2([&]() {
            try {
                boost::asio::ip::tcp::socket* sock = get_socket(block2_name);
                if (sock == nullptr || !sock->is_open()) {
                    throw std::runtime_error("ECLATIN: load_send_blocks socket not available for " + block2_name);
                }
                
                std::cout << "ECLATIN: Sending " << block2_name << " to rank2 (size=" << size << ")" << std::endl;
                
                if (!send_with_size(*sock, block2_addr, size)) {
                    throw std::runtime_error("ECLATIN: load_send_blocks send failed for " + block2_name);
                }
                
                std::cout << "ECLATIN: Successfully sent " << block2_name << " to rank2" << std::endl;
            } catch (...) {
                thread2_exception = std::current_exception();
            }
        });
        
        // Join both threads
        thread1.join();
        thread2.join();
        
        // Check for exceptions
        if (thread1_exception) {
            std::rethrow_exception(thread1_exception);
        }
        if (thread2_exception) {
            std::rethrow_exception(thread2_exception);
        }
        
        std::cout << "ECLATIN: Both blocks sent successfully" << std::endl;
    }

private:
    std::atomic<bool> stop_;

    // ASIO connections for pipelines
    AsioConnectionManager conn_;
    
    // RDMA configuration and resources
    bool use_rdma_;
    ibv_context* rdma_context_;
    ibv_pd* rdma_pd_;
    ibv_cq* rdma_send_cq_;
    ibv_cq* rdma_recv_cq_;
    std::map<uintptr_t, RdmaBuffer> rdma_registered_buffers_;
    std::mutex rdma_buffer_mutex_;
    
    // Connection channels (ASIO or RDMA)
    std::unique_ptr<IConnectionChannel> send_data1_channel_;
    std::unique_ptr<IConnectionChannel> send_parity0_channel_;
    std::unique_ptr<IConnectionChannel> send_parity1_channel_;
    std::unique_ptr<IConnectionChannel> recv_parity1_channel_;
    std::unique_ptr<IConnectionChannel> recv_parity0_channel_;
    std::unique_ptr<IConnectionChannel> recv_data1_channel_;
    
    // Load mode connection channels (for rank2 receiver and rank0/3 sender)
    std::unique_ptr<IConnectionChannel> load_recv_rank3_data1_channel_;
    std::unique_ptr<IConnectionChannel> load_recv_rank0_parity0_channel_;
    std::unique_ptr<IConnectionChannel> load_send_rank0_parity0_channel_;
    std::unique_ptr<IConnectionChannel> load_send_rank3_data1_channel_;
    
    // Save mode network config
    std::string send_data1_ip_;
    uint16_t send_data1_port_;
    std::string send_parity0_ip_;
    uint16_t send_parity0_port_;
    std::string send_parity1_ip_;
    uint16_t send_parity1_port_;
    std::string recv_parity1_ip_;
    uint16_t recv_parity1_port_;
    std::string recv_parity0_ip_;
    uint16_t recv_parity0_port_;
    std::string recv_data1_ip_;
    uint16_t recv_data1_port_;

    // EC encoding parameters (k=2, rows=2 for ecnaive)
    int k_;
    int rows_;
    unsigned char* a_mat_;    // RS matrix (k * m, where m = k + rows = 4)
    unsigned char* g_tbls_;   // EC encoding tables (32 * k * rows)

    // Save mode pipelines: 3 sends + 3 receives
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

    // Separate release queues for data and parity buffers
    std::queue<uintptr_t> data_buffers_to_release_;
    std::queue<uintptr_t> parity_buffers_to_release_;
    std::mutex release_queue_mutex_;

    // Completion flags
    std::atomic<bool> send_data1_completed_{false};
    std::atomic<bool> send_parity0_completed_{false};
    std::atomic<bool> send_parity1_completed_{false};
    std::atomic<bool> recv_parity1_completed_{false};
    std::atomic<bool> recv_parity0_completed_{false};
    std::atomic<bool> recv_data1_completed_{false};

    // Sentinel received flags
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
    
    // Load mode flags
    std::atomic<bool> is_load_mode_{false};
    int failed_rank_{-1};
    int failed_rank_in_group_{-1};  // failed rank within 4-rank group (for multi-group support)
    int rank_;  // Current rank_in_group (0..3) for load mode, set in init_ecnaive_load_connections

    // EC-NAIVE load mode queues (rank2 only)
    std::queue<LoadRecvTask> load_recv_queue_;
    std::mutex load_recv_queue_mutex_;
    std::condition_variable load_recv_queue_cv_;

    std::queue<LoadXORTask> load_xor_queue_;
    std::mutex load_xor_queue_mutex_;
    std::condition_variable load_xor_queue_cv_;

    // EC-NAIVE load mode queues (rank0/3 only)
    std::queue<LoadSendTask> load_send_queue_;
    std::mutex load_send_queue_mutex_;
    std::condition_variable load_send_queue_cv_;

    // EC-NAIVE load mode worker threads
    std::thread load_recv_worker_;      // rank2 only
    std::thread load_xor_worker_;       // rank2 only
    std::thread load_send_worker_;      // rank0/3 only

    // EC-NAIVE load mode completion flags
    std::atomic<bool> load_recv_worker_completed_{false};
    std::atomic<bool> load_xor_worker_completed_{false};
    std::atomic<bool> load_send_worker_completed_{false};

    // EC-NAIVE load mode sentinel flags
    std::atomic<bool> load_recv_sentinel_received_{false};
    std::atomic<bool> load_xor_sentinel_received_{false};
    std::atomic<bool> load_send_sentinel_received_{false};

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
        
        // Initialize tables using isa-l
        ec_init_tables(k_, rows_, a_mat_, g_tbls_);
        std::cout << "ECNAIVE: EC encoding tables initialized (k=" << k_ << ", rows=" << rows_ << ")" << std::endl;
    }
    
    // EC encoding function: encode 2 data blocks to 2 parity blocks
    void encode_ec_blocks(uintptr_t data0_addr, uintptr_t data1_addr,
                          uintptr_t parity0_addr, uintptr_t parity1_addr,
                          size_t size) {
        if (g_tbls_ == nullptr || a_mat_ == nullptr) {
            std::cerr << "ECNAIVE: ERROR: EC encoding tables not initialized!" << std::endl;
            throw std::runtime_error("ECNAIVE: EC encoding tables not initialized");
        }
        
        unsigned char* srcs[2];
        unsigned char* dests[2];
        srcs[0] = reinterpret_cast<unsigned char*>(data0_addr);
        srcs[1] = reinterpret_cast<unsigned char*>(data1_addr);
        dests[0] = reinterpret_cast<unsigned char*>(parity0_addr);
        dests[1] = reinterpret_cast<unsigned char*>(parity1_addr);
        
        // Use isa-l ec_encode_data: encode 2 data blocks to 2 parity blocks
        ec_encode_data((int)size, k_, rows_, g_tbls_, srcs, dests);
    }
    
    // RDMA initialization
    void init_rdma_resources() {
        std::cout << "[ECNAIVE RDMA] Initializing RDMA resources..." << std::endl;
        
        // Get device list
        int num_devices;
        ibv_device** device_list = ibv_get_device_list(&num_devices);
        if (!device_list || num_devices == 0) {
            throw std::runtime_error("No RDMA devices found");
        }
        
        // Use first device
        rdma_context_ = ibv_open_device(device_list[0]);
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
        
        // Create completion queues
        rdma_send_cq_ = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
        rdma_recv_cq_ = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
        
        if (!rdma_send_cq_ || !rdma_recv_cq_) {
            throw std::runtime_error("Failed to create completion queues");
        }
        
        std::cout << "[ECNAIVE RDMA] RDMA resources initialized successfully" << std::endl;
    }
    
    void cleanup_rdma_resources() {
        if (!use_rdma_) {
            return;
        }
        
        std::cout << "[ECNAIVE RDMA] Cleaning up RDMA resources..." << std::endl;
        
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
        
        // Destroy CQs
        if (rdma_send_cq_) {
            ibv_destroy_cq(rdma_send_cq_);
            rdma_send_cq_ = nullptr;
        }
        if (rdma_recv_cq_) {
            ibv_destroy_cq(rdma_recv_cq_);
            rdma_recv_cq_ = nullptr;
        }
        
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
        
        std::cout << "[ECNAIVE RDMA] RDMA resources cleaned up" << std::endl;
    }

    void start_threads() {
        std::cout << "ECNAIVE: Starting worker threads..." << std::endl;
        send_data1_thread_ = std::thread(&ECNaiveNative::send_data1_worker, this);
        send_parity0_thread_ = std::thread(&ECNaiveNative::send_parity0_worker, this);
        send_parity1_thread_ = std::thread(&ECNaiveNative::send_parity1_worker, this);
        recv_parity1_thread_ = std::thread(&ECNaiveNative::recv_parity1_worker, this);
        recv_parity0_thread_ = std::thread(&ECNaiveNative::recv_parity0_worker, this);
        recv_data1_thread_ = std::thread(&ECNaiveNative::recv_data1_worker, this);
        std::cout << "ECNAIVE: All worker threads started" << std::endl;
    }

    void init_connections() {
        std::cout << "ECNAIVE: Initializing connections..." << std::endl;
        // Start acceptors in separate threads to avoid deadlock (mirror eccheck pattern)
        std::thread recv_init_thread([this]() {
            std::thread r1([this]() { conn_.init_recv_parity1(recv_parity1_ip_, recv_parity1_port_); });
            std::thread r2([this]() { conn_.init_recv_parity0(recv_parity0_ip_, recv_parity0_port_); });
            std::thread r3([this]() { conn_.init_recv_data1(recv_data1_ip_, recv_data1_port_); });
            r1.join();
            r2.join();
            r3.join();
        });

        // Small delay to ensure acceptors are listening
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Connect send sockets (blocking)
        std::cout << "ECNAIVE: Connecting send_data1 socket..." << std::endl;
        conn_.init_send_data1(send_data1_ip_, send_data1_port_);
        std::cout << "ECNAIVE: Connecting send_parity0 socket..." << std::endl;
        conn_.init_send_parity0(send_parity0_ip_, send_parity0_port_);
        std::cout << "ECNAIVE: Connecting send_parity1 socket..." << std::endl;
        conn_.init_send_parity1(send_parity1_ip_, send_parity1_port_);

        recv_init_thread.join();
        std::cout << "ECNAIVE: Waiting for all connections..." << std::endl;
        conn_.wait_for_connections();
        std::cout << "ECNAIVE: All connections established" << std::endl;
    }


    // Save mode workers: 3 sends + 3 receives
    void recv_parity1_worker() {
        std::cout << "ECNAIVE: RecvParity1 worker started" << std::endl;
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
                        std::cout << "ECNAIVE: RecvParity1 worker completed" << std::endl;
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

            // Receive data directly into persistent buffer (no release needed)
                    if (!recv_with_size_bool(
                    conn_.get_recv_parity1_socket(),
                    reinterpret_cast<void*>(task.addr),
                            task.size)) {
                std::cerr << "ECNAIVE: recv_parity1_with_size_bool returned false" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_parity1_with_size_bool returned false");
            }

            if (recv_parity1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_parity1_mutex_);
                if (recv_parity1_q_.empty()) {
                    recv_parity1_completed_ = true;
                    std::cout << "ECNAIVE: RecvParity1 worker completed" << std::endl;
                    recv_parity1_sentinel_received_ = false;
                }
            }
        }
    }

    void recv_parity0_worker() {
        std::cout << "ECNAIVE: RecvParity0 worker started" << std::endl;
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
                        std::cout << "ECNAIVE: RecvParity0 worker completed" << std::endl;
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

            // Receive data directly into persistent buffer (no release needed)
                    if (!recv_with_size_bool(
                    conn_.get_recv_parity0_socket(),
                    reinterpret_cast<void*>(task.addr),
                            task.size)) {
                std::cerr << "ECNAIVE: recv_parity0_with_size_bool returned false" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_parity0_with_size_bool returned false");
            }

            if (recv_parity0_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_parity0_mutex_);
                if (recv_parity0_q_.empty()) {
                    recv_parity0_completed_ = true;
                    std::cout << "ECNAIVE: RecvParity0 worker completed" << std::endl;
                    recv_parity0_sentinel_received_ = false;
                }
            }
        }
    }

    void recv_data1_worker() {
        std::cout << "ECNAIVE: RecvData1 worker started" << std::endl;
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
                        std::cout << "ECNAIVE: RecvData1 worker completed" << std::endl;
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

            // Receive data directly into persistent buffer (no release needed)
            if (!recv_with_size_bool(
                    conn_.get_recv_data1_socket(),
                    reinterpret_cast<void*>(task.addr),
                    task.size)) {
                std::cerr << "ECNAIVE: recv_data1_with_size_bool returned false" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_data1_with_size_bool returned false");
            }

            if (recv_data1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_data1_mutex_);
                if (recv_data1_q_.empty()) {
                    recv_data1_completed_ = true;
                    std::cout << "ECNAIVE: RecvData1 worker completed" << std::endl;
                    recv_data1_sentinel_received_ = false;
                }
            }
        }
    }

    void send_data1_worker() {
        std::cout << "ECNAIVE: SendData1 worker started" << std::endl;
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
                        std::cout << "ECNAIVE: SendData1 worker completed" << std::endl;
                        send_data1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_send_data1_connected()) {
                send_with_size(conn_.get_send_data1_socket(), task.addr, task.size);
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
                    std::cout << "ECNAIVE: SendData1 worker completed" << std::endl;
                    send_data1_sentinel_received_ = false;
                }
            }
        }
    }

    void send_parity0_worker() {
        std::cout << "ECNAIVE: SendParity0 worker started" << std::endl;
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
                        std::cout << "ECNAIVE: SendParity0 worker completed" << std::endl;
                        send_parity0_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_send_parity0_connected()) {
                send_with_size(conn_.get_send_parity0_socket(), task.addr, task.size);
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
                    std::cout << "ECNAIVE: SendParity0 worker completed" << std::endl;
                    send_parity0_sentinel_received_ = false;
                }
            }
        }
    }

    void send_parity1_worker() {
        std::cout << "ECNAIVE: SendParity1 worker started" << std::endl;
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
                        std::cout << "ECNAIVE: SendParity1 worker completed" << std::endl;
                        send_parity1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_send_parity1_connected()) {
                send_with_size(conn_.get_send_parity1_socket(), task.addr, task.size);
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
                    std::cout << "ECNAIVE: SendParity1 worker completed" << std::endl;
                    send_parity1_sentinel_received_ = false;
                }
            }
        }
    }

    // ========== EC-NAIVE Load Mode Workers ==========

    // Load recv worker (rank2 only): parallel receive 8 blocks for full recovery
    void load_recv_worker() {
        std::cout << "EC-NAIVE: [Rank 2] Load recv worker started (full recovery: 8 blocks)" << std::endl;
        
        // Wait for all 8 connections
        while (!stop_ && 
               (!conn_.is_ecnaive_load_recv_rank3_data1_connected() || 
                !conn_.is_ecnaive_load_recv_rank0_parity0_connected() ||
                !conn_.is_ecnaive_load_recv_rank0_data0_connected() ||
                !conn_.is_ecnaive_load_recv_rank1_data1_connected() ||
                !conn_.is_ecnaive_load_recv_rank1_data0_connected() ||
                !conn_.is_ecnaive_load_recv_rank1_parity1_connected() ||
                !conn_.is_ecnaive_load_recv_rank3_data0_connected() ||
                !conn_.is_ecnaive_load_recv_rank0_data1_connected())) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        std::cout << "EC-NAIVE: [Rank 2] All 8 connections established" << std::endl;
        
        while (!stop_) {
            LoadRecvTask task;
            
            {
                std::unique_lock<std::mutex> lock(load_recv_queue_mutex_);
                load_recv_queue_cv_.wait(lock, [this] {
                    return stop_ || !load_recv_queue_.empty();
                });
                
                if (stop_ && load_recv_queue_.empty()) {
                    break;
                }
                
                task = load_recv_queue_.front();
                load_recv_queue_.pop();
            }
            
            // Check sentinel (all addresses are 0)
            if (task.recv_p20_addr == 0 && task.recv_d21_addr == 0 && 
                task.recv_d00_addr == 0 && task.recv_d01_addr == 0 &&
                task.recv_d10_addr == 0 && task.recv_p11_addr == 0 &&
                task.recv_d30_addr == 0 && task.recv_d31_addr == 0 && 
                task.size == 0) {
                load_recv_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(load_recv_queue_mutex_);
                    if (load_recv_queue_.empty()) {
                        load_recv_worker_completed_ = true;
                        load_recv_sentinel_received_ = false;
                        // Submit XOR sentinel before exiting
                        LoadXORTask xor_sentinel;
                        xor_sentinel.recv_p20_addr = 0;
                        xor_sentinel.recv_d21_addr = 0;
                        xor_sentinel.recv_d00_addr = 0;
                        xor_sentinel.recv_d01_addr = 0;
                        xor_sentinel.recv_d10_addr = 0;
                        xor_sentinel.recv_p11_addr = 0;
                        xor_sentinel.recv_d30_addr = 0;
                        xor_sentinel.recv_d31_addr = 0;
                        xor_sentinel.size = 0;
                        {
                            std::lock_guard<std::mutex> xor_lock(load_xor_queue_mutex_);
                            load_xor_queue_.push(xor_sentinel);
                        }
                        load_xor_queue_cv_.notify_one();
                        std::cout << "EC-NAIVE: [Rank 2] Load recv worker: Submitted XOR sentinel before exiting" << std::endl;
                        break;
                    }
                }
                continue;
            }
            
            // Parallel receive 8 blocks using 8 threads
            std::vector<std::thread> recv_threads(8);
            std::vector<std::exception_ptr> recv_exceptions(8);
            
            // Thread 1: Receive p_{2,0} from rank0
            recv_threads[0] = std::thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_ecnaive_load_recv_rank0_parity0_socket(),
                            reinterpret_cast<void*>(task.recv_p20_addr),
                            task.size)) {
                        throw std::runtime_error("Failed to receive p_{2,0} from rank0");
                    }
                } catch (...) {
                    recv_exceptions[0] = std::current_exception();
                }
            });

            // Thread 2: Receive d_{2,1} from rank3
            recv_threads[1] = std::thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_ecnaive_load_recv_rank3_data1_socket(),
                            reinterpret_cast<void*>(task.recv_d21_addr),
                            task.size)) {
                        throw std::runtime_error("Failed to receive d_{2,1} from rank3");
                    }
                } catch (...) {
                    recv_exceptions[1] = std::current_exception();
                }
            });

            // Thread 3: Receive d_{0,0} from rank0
            recv_threads[2] = std::thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_ecnaive_load_recv_rank0_data0_socket(),
                            reinterpret_cast<void*>(task.recv_d00_addr),
                            task.size)) {
                        throw std::runtime_error("Failed to receive d_{0,0} from rank0");
                    }
                } catch (...) {
                    recv_exceptions[2] = std::current_exception();
                }
            });

            // Thread 4: Receive d_{0,1} from rank1
            recv_threads[3] = std::thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_ecnaive_load_recv_rank1_data1_socket(),
                            reinterpret_cast<void*>(task.recv_d01_addr),
                            task.size)) {
                        throw std::runtime_error("Failed to receive d_{0,1} from rank1");
                    }
                } catch (...) {
                    recv_exceptions[3] = std::current_exception();
                }
            });

            // Thread 5: Receive d_{1,0} from rank1
            recv_threads[4] = std::thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_ecnaive_load_recv_rank1_data0_socket(),
                            reinterpret_cast<void*>(task.recv_d10_addr),
                            task.size)) {
                        throw std::runtime_error("Failed to receive d_{1,0} from rank1");
                    }
                } catch (...) {
                    recv_exceptions[4] = std::current_exception();
                }
            });

            // Thread 6: Receive p_{1,1} from rank1
            recv_threads[5] = std::thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_ecnaive_load_recv_rank1_parity1_socket(),
                            reinterpret_cast<void*>(task.recv_p11_addr),
                            task.size)) {
                        throw std::runtime_error("Failed to receive p_{1,1} from rank1");
                    }
                } catch (...) {
                    recv_exceptions[5] = std::current_exception();
                }
            });

            // Thread 7: Receive d_{3,0} from rank3
            recv_threads[6] = std::thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_ecnaive_load_recv_rank3_data0_socket(),
                            reinterpret_cast<void*>(task.recv_d30_addr),
                            task.size)) {
                        throw std::runtime_error("Failed to receive d_{3,0} from rank3");
                    }
                } catch (...) {
                    recv_exceptions[6] = std::current_exception();
                }
            });

            // Thread 8: Receive d_{3,1} from rank0
            recv_threads[7] = std::thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_ecnaive_load_recv_rank0_data1_socket(),
                            reinterpret_cast<void*>(task.recv_d31_addr),
                            task.size)) {
                        throw std::runtime_error("Failed to receive d_{3,1} from rank0");
                    }
                } catch (...) {
                    recv_exceptions[7] = std::current_exception();
                }
            });
            
            // Join all 8 threads
            for (auto& t : recv_threads) {
                t.join();
            }
            
            // Check for exceptions
            for (size_t i = 0; i < 8; i++) {
                if (recv_exceptions[i]) {
                    std::rethrow_exception(recv_exceptions[i]);
                }
            }
            
            // After all 8 blocks are received, submit XOR task
            LoadXORTask xor_task;
            xor_task.recv_p20_addr = task.recv_p20_addr;
            xor_task.recv_d21_addr = task.recv_d21_addr;
            xor_task.recv_d00_addr = task.recv_d00_addr;
            xor_task.recv_d01_addr = task.recv_d01_addr;
            xor_task.recv_d10_addr = task.recv_d10_addr;
            xor_task.recv_p11_addr = task.recv_p11_addr;
            xor_task.recv_d30_addr = task.recv_d30_addr;
            xor_task.recv_d31_addr = task.recv_d31_addr;
            xor_task.output_data0_addr = task.output_data0_addr;
            xor_task.output_recv_parity0_addr = task.output_recv_parity0_addr;
            xor_task.output_recv_data1_addr = task.output_recv_data1_addr;
            xor_task.output_recv_parity1_addr = task.output_recv_parity1_addr;
            xor_task.size = task.size;
            
            {
                std::lock_guard<std::mutex> lock(load_xor_queue_mutex_);
                load_xor_queue_.push(xor_task);
            }
            load_xor_queue_cv_.notify_one();
            
            std::cout << "EC-NAIVE: [Rank 2] Received all 8 blocks, submitted XOR task (size=" 
                      << task.size << ")" << std::endl;
        }
        
        std::cout << "EC-NAIVE: [Rank 2] Load recv worker completed" << std::endl;
    }

    // Load XOR worker (rank2 only): Perform 4 XOR operations for full recovery
    // XOR 1: d_{2,0} = p_{2,0} ⊕ d_{2,1} -> output_data0_addr
    // XOR 2: p_{0,0} = d_{0,0} ⊕ d_{0,1} -> output_recv_parity0_addr
    // XOR 3: d_{1,1} = d_{1,0} ⊕ p_{1,1} -> output_recv_data1_addr
    // XOR 4: p_{3,1} = d_{3,0} ⊕ d_{3,1} -> output_recv_parity1_addr
    void load_xor_worker() {
        std::cout << "EC-NAIVE: [Rank 2] Load XOR worker started (full recovery: 4 XOR operations)" << std::endl;
        
        while (!stop_) {
            LoadXORTask task;
            
            {
                std::unique_lock<std::mutex> lock(load_xor_queue_mutex_);
                load_xor_queue_cv_.wait(lock, [this] {
                    return stop_ || !load_xor_queue_.empty() || load_xor_sentinel_received_.load();
                });
                
                if (stop_ && load_xor_queue_.empty()) {
                    break;
                }
                
                // Check if sentinel received and queue is empty (before popping)
                if (load_xor_sentinel_received_.load() && load_xor_queue_.empty()) {
                    load_xor_worker_completed_ = true;
                    load_xor_sentinel_received_ = false;
                    break;
                }
                
                task = load_xor_queue_.front();
                load_xor_queue_.pop();
            }
            
            // Check sentinel (all addresses are 0)
            if (task.recv_p20_addr == 0 && task.recv_d21_addr == 0 && 
                task.recv_d00_addr == 0 && task.recv_d01_addr == 0 &&
                task.recv_d10_addr == 0 && task.recv_p11_addr == 0 &&
                task.recv_d30_addr == 0 && task.recv_d31_addr == 0 && 
                task.size == 0) {
                load_xor_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(load_xor_queue_mutex_);
                    if (load_xor_queue_.empty()) {
                        load_xor_worker_completed_ = true;
                        load_xor_sentinel_received_ = false;
                        break;
                    }
                }
                continue;
            }
            
            // Perform 4 XOR operations (zero-copy: results written directly to output addresses)
            // Note: xor_gen modifies the first parameter in-place, so we copy first operand to output first
            
            // XOR 1: d_{2,0} = p_{2,0} ⊕ d_{2,1} -> output_data0_addr
            // Copy p_{2,0} to output first, then XOR with d_{2,1}
            memcpy(reinterpret_cast<void*>(task.output_data0_addr), 
                   reinterpret_cast<void*>(task.recv_p20_addr), task.size);
            void* xor1_srcs[2] = {
                reinterpret_cast<void*>(task.output_data0_addr),  // destination (contains p_{2,0})
                reinterpret_cast<void*>(task.recv_d21_addr)       // d_{2,1}
            };
            xor_gen(2, static_cast<int>(task.size), xor1_srcs);
            // Result is now in output_data0_addr (zero-copy)
            
            // XOR 2: p_{0,0} = d_{0,0} ⊕ d_{0,1} -> output_recv_parity0_addr
            memcpy(reinterpret_cast<void*>(task.output_recv_parity0_addr), 
                   reinterpret_cast<void*>(task.recv_d00_addr), task.size);
            void* xor2_srcs[2] = {
                reinterpret_cast<void*>(task.output_recv_parity0_addr),  // destination (contains d_{0,0})
                reinterpret_cast<void*>(task.recv_d01_addr)              // d_{0,1}
            };
            xor_gen(2, static_cast<int>(task.size), xor2_srcs);
            // Result is now in output_recv_parity0_addr (zero-copy)
            
            // XOR 3: d_{1,1} = d_{1,0} ⊕ p_{1,1} -> output_recv_data1_addr
            memcpy(reinterpret_cast<void*>(task.output_recv_data1_addr), 
                   reinterpret_cast<void*>(task.recv_d10_addr), task.size);
            void* xor3_srcs[2] = {
                reinterpret_cast<void*>(task.output_recv_data1_addr),  // destination (contains d_{1,0})
                reinterpret_cast<void*>(task.recv_p11_addr)            // p_{1,1}
            };
            xor_gen(2, static_cast<int>(task.size), xor3_srcs);
            // Result is now in output_recv_data1_addr (zero-copy)
            
            // XOR 4: p_{3,1} = d_{3,0} ⊕ d_{3,1} -> output_recv_parity1_addr
            memcpy(reinterpret_cast<void*>(task.output_recv_parity1_addr), 
                   reinterpret_cast<void*>(task.recv_d30_addr), task.size);
            void* xor4_srcs[2] = {
                reinterpret_cast<void*>(task.output_recv_parity1_addr),  // destination (contains d_{3,0})
                reinterpret_cast<void*>(task.recv_d31_addr)               // d_{3,1}
            };
            xor_gen(2, static_cast<int>(task.size), xor4_srcs);
            // Result is now in output_recv_parity1_addr (zero-copy)
            
            std::cout << "EC-NAIVE: [Rank 2] All 4 XOR operations completed for chunk (size=" 
                      << task.size << ")" << std::endl;
            
            // After processing task, check if sentinel was received and queue is empty
            if (load_xor_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(load_xor_queue_mutex_);
                if (load_xor_queue_.empty()) {
                    load_xor_worker_completed_ = true;
                    load_xor_sentinel_received_ = false;
                    break;
                }
            }
        }
        
        std::cout << "EC-NAIVE: [Rank 2] Load XOR worker completed" << std::endl;
    }

    // Load send worker (rank0/1/3): send blocks to rank2 (multiple sockets)
    void load_send_worker() {
        std::cout << "EC-NAIVE: [Rank " << rank_ << "] Load send worker started" << std::endl;
        
        // Wait for all connections based on rank
        if (rank_ == 0) {
            while (!stop_ && 
                   (!conn_.is_ecnaive_load_send_rank0_parity0_connected() ||
                    !conn_.is_ecnaive_load_send_rank0_data0_connected() ||
                    !conn_.is_ecnaive_load_send_rank0_data1_connected())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::cout << "EC-NAIVE: [Rank 0] All 3 connections established" << std::endl;
        } else if (rank_ == 1) {
            while (!stop_ && 
                   (!conn_.is_ecnaive_load_send_rank1_data1_connected() ||
                    !conn_.is_ecnaive_load_send_rank1_data0_connected() ||
                    !conn_.is_ecnaive_load_send_rank1_parity1_connected())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::cout << "EC-NAIVE: [Rank 1] All 3 connections established" << std::endl;
        } else if (rank_ == 3) {
            while (!stop_ && 
                   (!conn_.is_ecnaive_load_send_rank3_data1_connected() ||
                    !conn_.is_ecnaive_load_send_rank3_data0_connected())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::cout << "EC-NAIVE: [Rank 3] All 2 connections established" << std::endl;
        } else {
            return;  // rank2 doesn't use this worker
        }
        
        while (!stop_) {
            LoadSendTask task;
            
            {
                std::unique_lock<std::mutex> lock(load_send_queue_mutex_);
                load_send_queue_cv_.wait(lock, [this] {
                    return stop_ || !load_send_queue_.empty();
                });
                
                if (stop_ && load_send_queue_.empty()) {
                    break;
                }
                
                task = load_send_queue_.front();
                load_send_queue_.pop();
            }
            
            // Check sentinel
            if (task.send_addr == 0 && task.size == 0) {
                load_send_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
                    if (load_send_queue_.empty()) {
                        load_send_worker_completed_ = true;
                        load_send_sentinel_received_ = false;
                        break;
                    }
                }
                continue;
            }
            
            // Select socket based on rank and socket_type
            boost::asio::ip::tcp::socket* send_socket = nullptr;
            if (rank_ == 0) {
                if (task.socket_type == 0) {
                    send_socket = &conn_.get_ecnaive_load_send_rank0_parity0_socket();  // p_{2,0}
                } else if (task.socket_type == 1) {
                    send_socket = &conn_.get_ecnaive_load_send_rank0_data0_socket();   // d_{0,0}
                } else if (task.socket_type == 2) {
                    send_socket = &conn_.get_ecnaive_load_send_rank0_data1_socket();   // d_{3,1}
                } else {
                    std::cerr << "EC-NAIVE: [Rank 0] Invalid socket_type: " << task.socket_type << std::endl;
                    continue;
                }
            } else if (rank_ == 1) {
                if (task.socket_type == 1) {
                    send_socket = &conn_.get_ecnaive_load_send_rank1_data0_socket();   // d_{1,0}
                } else if (task.socket_type == 2) {
                    send_socket = &conn_.get_ecnaive_load_send_rank1_data1_socket();   // d_{0,1}
                } else if (task.socket_type == 3) {
                    send_socket = &conn_.get_ecnaive_load_send_rank1_parity1_socket(); // p_{1,1}
                } else {
                    std::cerr << "EC-NAIVE: [Rank 1] Invalid socket_type: " << task.socket_type << std::endl;
                    continue;
                }
            } else if (rank_ == 3) {
                if (task.socket_type == 1) {
                    send_socket = &conn_.get_ecnaive_load_send_rank3_data0_socket();   // d_{3,0}
                } else if (task.socket_type == 2) {
                    send_socket = &conn_.get_ecnaive_load_send_rank3_data1_socket();   // d_{2,1}
                } else {
                    std::cerr << "EC-NAIVE: [Rank 3] Invalid socket_type: " << task.socket_type << std::endl;
                    continue;
                }
            }
            
            // Send data (zero-copy from mmap or buffer)
            try {
                send_with_size(*send_socket, task.send_addr, task.size);
                
                std::cout << "EC-NAIVE: [Rank " << rank_ << "] Sent chunk (socket_type=" 
                          << task.socket_type << ", size=" << task.size << ")" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "EC-NAIVE: [Rank " << rank_ << "] Send error: " << e.what() << std::endl;
            }
        }
        
        std::cout << "EC-NAIVE: [Rank " << rank_ << "] Load send worker completed" << std::endl;
    }
};

}  // namespace

PYBIND11_MODULE(ecnaive_native, m) {
    pybind11::class_<ECNaiveNative>(m, "ECNaiveNative")
        .def(pybind11::init<const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            bool>(),
             pybind11::arg("send_data1_ip"),
             pybind11::arg("send_data1_port"),
             pybind11::arg("send_parity0_ip"),
             pybind11::arg("send_parity0_port"),
             pybind11::arg("send_parity1_ip"),
             pybind11::arg("send_parity1_port"),
             pybind11::arg("recv_parity1_ip"),
             pybind11::arg("recv_parity1_port"),
             pybind11::arg("recv_parity0_ip"),
             pybind11::arg("recv_parity0_port"),
             pybind11::arg("recv_data1_ip"),
             pybind11::arg("recv_data1_port"),
             pybind11::arg("use_rdma") = false)
        // RDMA buffer management
        .def("register_buffer", &ECNaiveNative::register_buffer,
             "Register buffer for RDMA operations",
             pybind11::arg("addr"),
             pybind11::arg("size"))
        .def("unregister_buffer", &ECNaiveNative::unregister_buffer,
             "Unregister buffer from RDMA",
             pybind11::arg("addr"))
        // Unified save mode submit function
        .def("submit_ecnaive_save", &ECNaiveNative::submit_ecnaive_save,
             "Unified save mode function: encode and submit send/recv tasks",
             pybind11::arg("data0_addr"),      // d_{i0} - keep, not sent
             pybind11::arg("data1_addr"),      // d_{i1} - send to rank i+1
             pybind11::arg("parity0_addr"),   // p_{i0} - send to rank i+2
             pybind11::arg("parity1_addr"),    // p_{i1} - send to rank i+3
             pybind11::arg("recv_parity1_addr"), // recv p_{i+1,1} from rank i+1
             pybind11::arg("recv_parity0_addr"), // recv p_{i+2,0} from rank i+2
             pybind11::arg("recv_data1_addr"),   // recv d_{i+3,1} from rank i+3
             pybind11::arg("size"))
        // Save mode submit functions: 3 sends + 3 receives (for fine-grained control if needed)
        .def("submit_send_data1", &ECNaiveNative::submit_send_data1,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_send_parity0", &ECNaiveNative::submit_send_parity0,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_send_parity1", &ECNaiveNative::submit_send_parity1,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_recv_parity1", &ECNaiveNative::submit_recv_parity1,
             pybind11::arg("recv_addr"),
             pybind11::arg("size"))
        .def("submit_recv_parity0", &ECNaiveNative::submit_recv_parity0,
             pybind11::arg("recv_addr"),
             pybind11::arg("size"))
        .def("submit_recv_data1", &ECNaiveNative::submit_recv_data1,
             pybind11::arg("recv_addr"),
             pybind11::arg("size"))
        // Common functions
        .def("get_data_buffers_to_release", &ECNaiveNative::get_data_buffers_to_release)
        .def("get_parity_buffers_to_release", &ECNaiveNative::get_parity_buffers_to_release)
        .def("reset_encoding_completion_flags", &ECNaiveNative::reset_encoding_completion_flags)
        .def("wait_for_encoding_completion", &ECNaiveNative::wait_for_encoding_completion)
        // Save mode sentinels
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
             pybind11::arg("rank") = -1)
        // EC-NAIVE load mode functions (rank2 recovery)
        .def("init_ecnaive_load_connections", &ECNaiveNative::init_ecnaive_load_connections,
             "Initialize EC-NAIVE load mode connections (rank_in_group 2 acceptor: 8 ports, 0/1/3 connectors)",
             pybind11::arg("rank_in_group"),
             pybind11::arg("rank2_ip"),
             pybind11::arg("load_recv_rank3_data1_port"),
             pybind11::arg("load_recv_rank0_parity0_port"),
             pybind11::arg("load_recv_rank0_data0_port"),
             pybind11::arg("load_recv_rank1_data1_port"),
             pybind11::arg("load_recv_rank1_data0_port"),
             pybind11::arg("load_recv_rank1_parity1_port"),
             pybind11::arg("load_recv_rank3_data0_port"),
             pybind11::arg("load_recv_rank0_data1_port"))
        .def("submit_ecnaive_load_recovery", &ECNaiveNative::submit_ecnaive_load_recovery,
             "Submit load recovery task for rank2 (recv + XOR) - legacy interface",
             pybind11::arg("recv_data1_addr"),
             pybind11::arg("recv_parity0_addr"),
             pybind11::arg("recovered_data0_addr"),
             pybind11::arg("size"))
        .def("submit_ecnaive_load_recovery_full", &ECNaiveNative::submit_ecnaive_load_recovery_full,
             "Submit full recovery task for rank2 (8 recv + 4 XOR)",
             pybind11::arg("recv_p20_addr"),
             pybind11::arg("recv_d21_addr"),
             pybind11::arg("recv_d00_addr"),
             pybind11::arg("recv_d01_addr"),
             pybind11::arg("recv_d10_addr"),
             pybind11::arg("recv_p11_addr"),
             pybind11::arg("recv_d30_addr"),
             pybind11::arg("recv_d31_addr"),
             pybind11::arg("output_data0_addr"),
             pybind11::arg("output_recv_parity0_addr"),
             pybind11::arg("output_recv_data1_addr"),
             pybind11::arg("output_recv_parity1_addr"),
             pybind11::arg("size"))
        .def("submit_load_send_rank0_parity0", &ECNaiveNative::submit_load_send_rank0_parity0,
             "Submit load send task for rank0 (send p_{2,0} to rank2)",
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_load_send_rank0_data0", &ECNaiveNative::submit_load_send_rank0_data0,
             "Submit load send task for rank0 (send d_{0,0} to rank2)",
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_load_send_rank0_data1", &ECNaiveNative::submit_load_send_rank0_data1,
             "Submit load send task for rank0 (send d_{3,1} to rank2)",
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_load_send_rank1_data1", &ECNaiveNative::submit_load_send_rank1_data1,
             "Submit load send task for rank1 (send d_{0,1} to rank2)",
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_load_send_rank1_data0", &ECNaiveNative::submit_load_send_rank1_data0,
             "Submit load send task for rank1 (send d_{1,0} to rank2)",
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_load_send_rank1_parity1", &ECNaiveNative::submit_load_send_rank1_parity1,
             "Submit load send task for rank1 (send p_{1,1} to rank2)",
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_load_send_rank3_data1", &ECNaiveNative::submit_load_send_rank3_data1,
             "Submit load send task for rank3 (send d_{2,1} to rank2)",
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_load_send_rank3_data0", &ECNaiveNative::submit_load_send_rank3_data0,
             "Submit load send task for rank3 (send d_{3,0} to rank2)",
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_load_recv_sentinel", &ECNaiveNative::submit_load_recv_sentinel,
             "Submit sentinel to load recv worker (rank2 only)")
        .def("submit_load_xor_sentinel", &ECNaiveNative::submit_load_xor_sentinel,
             "Submit sentinel to load xor worker (rank2 only)")
        .def("submit_load_send_sentinel", &ECNaiveNative::submit_load_send_sentinel,
             "Submit sentinel to load send worker (rank0/1/3 only)")
        .def("wait_for_load_completion", &ECNaiveNative::wait_for_load_completion,
             "Wait for load workers to complete (rank2 only)")
        // Software failure mode functions
        .def("software_send_rank3_data1", &ECNaiveNative::software_send_rank3_data1,
             "Software failure mode: rank3 send d21 to rank2",
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("software_recv_data1", &ECNaiveNative::software_recv_data1,
             "Software failure mode: rank2 receive d21 from rank3",
             pybind11::arg("recv_addr"),
             pybind11::arg("size"))
        // Old load mode functions (kept for compatibility, may be removed later)
        .def("init_load_connections", &ECNaiveNative::init_load_connections,
             "Initialize load mode connections (rank0 recv, rank1/2/3 send)",
             pybind11::arg("rank"),
             pybind11::arg("rank0_ip"),
             pybind11::arg("load_recv_rank1_data1_port"),
             pybind11::arg("load_recv_rank1_data2_port"),
             pybind11::arg("load_recv_rank2_data2_port"),
             pybind11::arg("load_recv_rank2_parity2_port"),
             pybind11::arg("load_recv_rank3_data1_port"),
             pybind11::arg("load_recv_rank3_parity1_port"))
        .def("wait_for_load_connections", &ECNaiveNative::wait_for_load_connections,
             "Wait for load mode connections to be established",
             pybind11::arg("timeout_seconds") = 30)
        .def("load_recover", &ECNaiveNative::load_recover,
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
        .def("load_send_blocks", &ECNaiveNative::load_send_blocks,
             "Send two blocks to rank2 in parallel (for rank0, rank1, rank3)",
             pybind11::arg("block1_name"),
             pybind11::arg("block1_addr"),
             pybind11::arg("block2_name"),
             pybind11::arg("block2_addr"),
             pybind11::arg("size"))
        .def("stop", &ECNaiveNative::stop);
}


