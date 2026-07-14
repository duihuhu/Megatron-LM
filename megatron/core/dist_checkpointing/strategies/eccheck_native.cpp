#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <array>
#include <cctype>
#include <cerrno>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <unordered_map>
#include <chrono>
#include <isa-l/erasure_code.h>
#include <isa-l/raid.h>
#include <boost/asio.hpp>
#include <cstdlib>
#include <arpa/inet.h>  // For htonl/ntohl

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

#ifndef ECCHECK_RANKS_PER_GROUP
#define ECCHECK_RANKS_PER_GROUP 4  // Failed rank maps to EC group slot 0..3 (same convention as EC-NAIVE).
#endif

// RDMA includes (ibverbs)
#ifdef __linux__
#include <infiniband/verbs.h>
#include "rdma_device_utils.h"
#include <map>
#include <sys/socket.h>
#endif

// NCCL includes
#ifdef NCCL_AVAILABLE
#include <nccl.h>
#include <cuda_runtime.h>

// ========== Global function: Generate NCCL ID ==========
// This function can be called without creating an instance
std::vector<uint8_t> generate_nccl_id() {
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    const uint8_t* id_bytes = reinterpret_cast<const uint8_t*>(&nccl_id);
    return std::vector<uint8_t>(id_bytes, id_bytes + sizeof(ncclUniqueId));
}
#endif

// ========== RDMA Helper Structures and Functions ==========
#ifdef __linux__

// RDMA connection info for QP setup (exchanged over control socket)
struct RdmaConnInfo {
    uint32_t qp_num;
    uint16_t lid;
    uint8_t gid[16];
} __attribute__((packed));

// RDMA buffer registration info
struct RdmaBufferInfo {
    ibv_mr* mr;
    uintptr_t addr;
    size_t size;
};

// Check if RDMA is available on this system
bool is_rdma_available() {
    int num_devices = 0;
    ibv_device** device_list = ibv_get_device_list(&num_devices);
    
    if (device_list == nullptr || num_devices == 0) {
        return false;
    }
    
    ibv_free_device_list(device_list);
    return true;
}

#else  // Not Linux

// Stub for non-Linux systems
bool is_rdma_available() {
    return false;
}

#endif  // __linux__

// ========== ASIO Connection Manager ==========
class AsioConnectionManager {
private:
    boost::asio::io_context io_context_;
    boost::asio::ip::tcp::socket xor_send_socket_;
    boost::asio::ip::tcp::socket xor_recv_socket_;
    boost::asio::ip::tcp::socket p2p_send_socket_;
    boost::asio::ip::tcp::socket p2p_recv_socket_;
    boost::asio::ip::tcp::socket step6_p2p_send_socket_;
    boost::asio::ip::tcp::socket step6_p2p_recv_socket_;
    boost::asio::ip::tcp::acceptor xor_recv_acceptor_;
    boost::asio::ip::tcp::acceptor p2p_recv_acceptor_;
    boost::asio::ip::tcp::acceptor step6_p2p_recv_acceptor_;
    
    std::atomic<bool> xor_send_connected_;
    std::atomic<bool> xor_recv_connected_;
    std::atomic<bool> p2p_send_connected_;
    std::atomic<bool> p2p_recv_connected_;
    std::atomic<bool> step6_p2p_send_connected_;
    std::atomic<bool> step6_p2p_recv_connected_;
    
    std::mutex connection_mutex_;
    std::condition_variable connection_cv_;
    
public:
    AsioConnectionManager() 
        : io_context_(),
          xor_send_socket_(io_context_),
          xor_recv_socket_(io_context_),
          p2p_send_socket_(io_context_),
          p2p_recv_socket_(io_context_),
          step6_p2p_send_socket_(io_context_),
          step6_p2p_recv_socket_(io_context_),
          xor_recv_acceptor_(io_context_),
          p2p_recv_acceptor_(io_context_),
          step6_p2p_recv_acceptor_(io_context_),
          xor_send_connected_(false),
          xor_recv_connected_(false),
          p2p_send_connected_(false),
          p2p_recv_connected_(false),
          step6_p2p_send_connected_(false),
          step6_p2p_recv_connected_(false) {}
    
    boost::asio::io_context& get_io_context() { return io_context_; }
    boost::asio::ip::tcp::socket& get_xor_send_socket() { return xor_send_socket_; }
    boost::asio::ip::tcp::socket& get_xor_recv_socket() { return xor_recv_socket_; }
    boost::asio::ip::tcp::socket& get_p2p_send_socket() { return p2p_send_socket_; }
    boost::asio::ip::tcp::socket& get_p2p_recv_socket() { return p2p_recv_socket_; }
    boost::asio::ip::tcp::socket& get_step6_p2p_send_socket() { return step6_p2p_send_socket_; }
    boost::asio::ip::tcp::socket& get_step6_p2p_recv_socket() { return step6_p2p_recv_socket_; }
    
    bool is_xor_send_connected() const { return xor_send_connected_; }
    bool is_xor_recv_connected() const { return xor_recv_connected_; }
    bool is_p2p_send_connected() const { return p2p_send_connected_; }
    bool is_p2p_recv_connected() const { return p2p_recv_connected_; }
    bool is_step6_p2p_send_connected() const { return step6_p2p_send_connected_; }
    bool is_step6_p2p_recv_connected() const { return step6_p2p_recv_connected_; }
    
    void init_xor_send(const std::string& partner_ip, uint16_t port);
    void init_xor_recv(const std::string& listen_ip, uint16_t port);
    void init_p2p_send(const std::string& partner_ip, uint16_t port);
    void init_p2p_recv(const std::string& listen_ip, uint16_t port);
    void init_step6_p2p_send(const std::string& partner_ip, uint16_t port);
    void init_step6_p2p_recv(const std::string& listen_ip, uint16_t port);
    void wait_for_connections(int timeout_seconds = 30);
    void cleanup();
};

// ========== ASIO Connection Manager Implementation ==========

void AsioConnectionManager::init_xor_send(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        boost::asio::ip::tcp::resolver::results_type endpoints = 
            resolver.resolve(partner_ip, std::to_string(port));
        
        // std::cout << "ASIO: Connecting XOR send to " << partner_ip << ":" << port << "..." << std::endl;
        
        // Use synchronous connect
        boost::asio::connect(xor_send_socket_, endpoints);
        xor_send_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: XOR send init error: " << e.what() << std::endl;
        xor_send_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_xor_recv(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(
            boost::asio::ip::address::from_string(listen_ip), port);
        
        xor_recv_acceptor_.open(endpoint.protocol());
        xor_recv_acceptor_.set_option(
            boost::asio::ip::tcp::acceptor::reuse_address(true));
        xor_recv_acceptor_.bind(endpoint);
        xor_recv_acceptor_.listen();
        
        
        // Use synchronous accept (will block until connection is established)
        xor_recv_acceptor_.accept(xor_recv_socket_);
        xor_recv_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: XOR recv init error: " << e.what() << std::endl;
        xor_recv_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_p2p_send(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        boost::asio::ip::tcp::resolver::results_type endpoints = 
            resolver.resolve(partner_ip, std::to_string(port));
        
        
        // Use synchronous connect
        boost::asio::connect(p2p_send_socket_, endpoints);
        p2p_send_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: P2P send init error: " << e.what() << std::endl;
        p2p_send_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_p2p_recv(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(
            boost::asio::ip::address::from_string(listen_ip), port);
        
        p2p_recv_acceptor_.open(endpoint.protocol());
        p2p_recv_acceptor_.set_option(
            boost::asio::ip::tcp::acceptor::reuse_address(true));
        p2p_recv_acceptor_.bind(endpoint);
        p2p_recv_acceptor_.listen();
        
        
        // Use synchronous accept (will block until connection is established)
        p2p_recv_acceptor_.accept(p2p_recv_socket_);
        p2p_recv_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: P2P recv init error: " << e.what() << std::endl;
        p2p_recv_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_step6_p2p_send(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        boost::asio::ip::tcp::resolver::results_type endpoints = 
            resolver.resolve(partner_ip, std::to_string(port));
        
        
        // Use synchronous connect
        boost::asio::connect(step6_p2p_send_socket_, endpoints);
        step6_p2p_send_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: Step6 P2P send init error: " << e.what() << std::endl;
        step6_p2p_send_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_step6_p2p_recv(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(
            boost::asio::ip::address::from_string(listen_ip), port);
        
        step6_p2p_recv_acceptor_.open(endpoint.protocol());
        step6_p2p_recv_acceptor_.set_option(
            boost::asio::ip::tcp::acceptor::reuse_address(true));
        step6_p2p_recv_acceptor_.bind(endpoint);
        step6_p2p_recv_acceptor_.listen();
        
        
        // Use synchronous accept (will block until connection is established)
        step6_p2p_recv_acceptor_.accept(step6_p2p_recv_socket_);
        step6_p2p_recv_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: Step6 P2P recv init error: " << e.what() << std::endl;
        step6_p2p_recv_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::wait_for_connections(int timeout_seconds) {
    std::unique_lock<std::mutex> lock(connection_mutex_);
    bool all_connected = connection_cv_.wait_for(
        lock,
        std::chrono::seconds(timeout_seconds),
        [this]() {
            return xor_send_connected_ && xor_recv_connected_ &&
                   p2p_send_connected_ && p2p_recv_connected_;
        }
    );
    
    if (!all_connected) {
        std::cerr << "ASIO: WARNING: Not all connections established within timeout" << std::endl;
        std::cerr << "  XOR send: " << xor_send_connected_ << std::endl;
        std::cerr << "  XOR recv: " << xor_recv_connected_ << std::endl;
        std::cerr << "  P2P send: " << p2p_send_connected_ << std::endl;
        std::cerr << "  P2P recv: " << p2p_recv_connected_ << std::endl;
    } else {
    }
}

void AsioConnectionManager::cleanup() {
    if (xor_send_socket_.is_open()) xor_send_socket_.close();
    if (xor_recv_socket_.is_open()) xor_recv_socket_.close();
    if (p2p_send_socket_.is_open()) p2p_send_socket_.close();
    if (p2p_recv_socket_.is_open()) p2p_recv_socket_.close();
    if (step6_p2p_send_socket_.is_open()) step6_p2p_send_socket_.close();
    if (step6_p2p_recv_socket_.is_open()) step6_p2p_recv_socket_.close();
    if (xor_recv_acceptor_.is_open()) xor_recv_acceptor_.close();
    if (p2p_recv_acceptor_.is_open()) p2p_recv_acceptor_.close();
    if (step6_p2p_recv_acceptor_.is_open()) step6_p2p_recv_acceptor_.close();
}

class ECCHECKNative;

struct EccheckXorPoolWorkerCtx {
    ECCHECKNative* self{nullptr};
    int wid{0};
};

struct EccheckEncodePoolWorkerCtx {
    ECCHECKNative* self{nullptr};
    int wid{0};
};

// RDMA recv: attribute CQ poll-phase time to rank2 load recv workers (sum over chunks).
enum class RdmaLoadRecvPollLane : int { None = 0, EncXor = 1, Step2P2p = 2, Step6P2p = 3 };

class ECCHECKNative {
private:
    int rank_;
    int world_size_;
    int paired_rank_;
    int rank_in_group_;        // Rank within EC group (0-3), for multi-rank same logic as 4-rank
    int failed_rank_in_group_; // Failed rank's position in group (0-3) for load mode
    
    // Buffer management
    std::vector<uintptr_t> data_buffer_addrs_;
    std::vector<uintptr_t> encoding_buffer_addrs_;
    std::vector<size_t> buffer_sizes_;
    
    // ========== 重构：每个encoding线程有独立的通信队列和worker ==========
    
    // Thread 1的任务和队列
    struct EncodingTask {
        uintptr_t data_addr;
        size_t size;
        uintptr_t encoding_addr;
        uintptr_t recv_addr;        // 接收地址
        size_t recv_chunk_size;     // 接收大小
        uintptr_t parity_addr;      // XOR结果地址（parity buffer）
        uintptr_t p2p_own_write_addr;    // P2P: 写入own_buffer的地址
        uintptr_t p2p_partner_write_addr; // P2P: 写入partner_buffer的地址（接收到的数据）
        bool local_is_zero_tail;          // Save path: local data chunk is implicit zero
        bool remote_is_zero_tail;         // Save path: XOR peer data chunk is implicit zero
        bool p2p_data_is_zero_tail;       // Save path: P2P data sender chunk is implicit zero
        size_t p2p_data_size;             // Save path: valid P2P data bytes in this chunk
        size_t sequence_id;               // Save path chunk order for P2P transfers
    };
    
    std::queue<EncodingTask> encoding_tasks_1_;  // Thread1的编码任务
    std::queue<EncodingTask> encoding_tasks_2_;  // Thread2的编码任务
    std::atomic<size_t> save_sequence_id_thread1_{0};
    std::atomic<size_t> save_sequence_id_thread2_{0};
    
    std::mutex encoding_tasks_1_mutex_;
    std::mutex encoding_tasks_2_mutex_;
    std::condition_variable encoding_tasks_1_cv_;
    std::condition_variable encoding_tasks_2_cv_;
    
    // Unified send queue (encoding完成后放入)
    struct SendTask {
        uintptr_t encoding_addr;
        size_t size;
    };
    std::queue<SendTask> send_queue_;
    std::mutex send_queue_mutex_;
    std::condition_variable send_queue_cv_;
    
    // Unified recv task queue (Python提交)
    struct RecvTask {
        uintptr_t recv_addr;
        size_t size;
        uintptr_t parity_addr;      // XOR result address from EncodingTask
        bool local_is_zero_tail;    // Save path: local data chunk is implicit zero
        bool remote_is_zero_tail;   // Save path: remote XOR chunk is implicit zero
        bool p2p_data_is_zero_tail; // Save path: P2P data sender chunk is implicit zero
        size_t p2p_data_size;       // Save path: valid P2P data bytes in this chunk
        size_t sequence_id;         // Save path chunk order for downstream P2P tasks
    };
    std::queue<RecvTask> recv_queue_;
    std::mutex recv_queue_mutex_;
    std::condition_variable recv_queue_cv_;
    
    // Completion flags
    std::atomic<bool> encoding_thread_1_completed_;
    std::atomic<bool> encoding_thread_2_completed_;
    std::atomic<bool> send_worker_completed_;
    std::atomic<bool> recv_worker_completed_;
    std::atomic<bool> xor_worker_completed_;
    std::atomic<bool> p2p_send_worker_completed_;
    std::atomic<bool> p2p_recv_worker_completed_;
    
    // Sentinel received flags (to track if sentinel was received, but queue may not be empty yet)
    std::atomic<bool> encoding_thread_1_sentinel_received_;
    std::atomic<bool> encoding_thread_2_sentinel_received_;
    std::atomic<bool> send_worker_sentinel_received_;
    std::atomic<bool> recv_worker_sentinel_received_;
    std::atomic<bool> xor_worker_sentinel_received_;
    std::atomic<bool> p2p_send_worker_sentinel_received_;
    std::atomic<bool> p2p_recv_worker_sentinel_received_;
    
    // Stop flag for graceful shutdown
    std::atomic<bool> should_stop_threads_;
    
    // Data buffer state tracking
    struct DataBufferState {
        bool thread1_copied;
        bool thread2_copied;
    };
    std::unordered_map<uintptr_t, DataBufferState> data_buffer_states_;
    std::mutex data_buffer_state_mutex_;
    
    // Buffers ready for release
    std::queue<uintptr_t> data_buffers_to_release_;
    std::queue<uintptr_t> encoding_buffers_to_release_;
    std::queue<uintptr_t> parity_buffers_to_release_;
    std::mutex release_queue_mutex_;
    
    // XOR configuration
    struct XORConfig {
        int xor_partner_rank;        // XOR配对的rank
        bool thread0_is_receiver;    // thread0是否接收（做XOR）
        bool thread1_is_receiver;    // thread1是否接收（做XOR）
    };
    XORConfig xor_config_;
    
    // XOR task structure
    struct XORTask {
        uintptr_t local_encoding_addr;   // 本地encoded数据地址
        uintptr_t remote_encoding_addr;  // 接收到的远程encoded数据地址
        uintptr_t parity_addr;           // XOR结果地址（parity buffer）
        size_t size;                     // 数据大小
        uintptr_t p2p_own_write_addr;    // P2P: 写入own_buffer的地址
        uintptr_t p2p_partner_write_addr; // P2P: 写入partner_buffer的地址
        uintptr_t data_addr;             // 原始data地址（用于奇数rank发送data）
        bool local_is_zero_tail;
        bool remote_is_zero_tail;
        bool p2p_data_is_zero_tail;
        size_t p2p_data_size;
        size_t sequence_id;
    };
    
    // Unified XOR task queue
    std::queue<XORTask> xor_queue_;
    std::mutex xor_queue_mutex_;
    std::condition_variable xor_queue_cv_;
    
    // Pending encoding buffers等待XOR（用于匹配encoding和recv）
    // Key: recv_addr, Value: encoding_addr
    std::unordered_map<uintptr_t, uintptr_t> pending_xor_encoding_;
    std::mutex pending_xor_mutex_;
    
    // Track recv_addr -> data_addr mapping (for P2P: odd ranks need to send data)
    std::unordered_map<uintptr_t, uintptr_t> recv_to_data_;
    std::mutex recv_to_data_mutex_;
    
    // Track recv_addr -> parity_addr mapping
    std::unordered_map<uintptr_t, uintptr_t> recv_to_parity_;
    std::mutex recv_to_parity_mutex_;
    
    // Track recv_addr -> P2P addresses mapping
    struct P2PAddresses {
        uintptr_t own_write_addr;
        uintptr_t partner_write_addr;
    };
    std::unordered_map<uintptr_t, P2PAddresses> recv_to_p2p_;
    std::mutex recv_to_p2p_mutex_;
    
    // Pending encoding tasks for load mode Step2 (waiting for P2P completion)
    // Key: data_addr (for rank0/3) or data_buffer_addr (for rank1/2), Value: encoding task info
    struct PendingEncodingTask {
        uintptr_t data_addr;
        size_t size;
        uintptr_t encoding_addr1;
        uintptr_t encoding_addr2;
        uintptr_t recv_addr_thread1;
        uintptr_t recv_addr_thread2;
        size_t recv_chunk_size;
        uintptr_t parity_addr1;
        uintptr_t parity_addr2;
        uintptr_t p2p_own_write_addr;
        uintptr_t p2p_partner_write_addr;
    };
    std::unordered_map<uintptr_t, PendingEncodingTask> pending_encoding_tasks_;
    std::mutex pending_encoding_tasks_mutex_;
    
    // P2P task structures - split into send and recv (like send/recv workers)
    struct P2PSendTask {
        uintptr_t send_buffer_addr;      // Data to send (parity for even ranks, data for odd ranks)
        uintptr_t p2p_own_write_addr;    // Write own parity/data to this address (for memcpy)
        size_t size;
        uintptr_t parity_addr;           // Parity buffer address (for release after send)
        uintptr_t data_addr;             // Data buffer address (for release after send, odd ranks only)
        // New fields for load mode
        bool is_load_mode_transfer;      // true=load mode transfer (Step2 or Step6), false=save stage
        bool is_step6_transfer;         // true=Step6 transfer (rank3->rank2), false=Step2 transfer
        uintptr_t load_mode_data_addr;  // load mode Step2: corresponding data_addr (for finding encoding task)
        size_t zero_fill_tail_size;     // save path: zero-fill remaining bytes after valid data
        size_t sequence_id;             // save path chunk order for network transfer
    };
    
    struct P2PRecvTask {
        uintptr_t recv_buffer_addr;     // Receive partner data/parity to this address
        size_t size;
        // New fields for load mode
        bool is_load_mode_transfer;      // true=load mode transfer, false=save stage
        bool is_step6_transfer;         // true=Step6 transfer (rank3->rank2), false=Step2 transfer
        uintptr_t data_buffer_addr;      // load mode: receive后写入的data_buffer地址（如果与recv_buffer_addr不同）
        bool skip_network;              // save path: fill zeros without receiving data
        size_t zero_fill_tail_size;      // save path: zero-fill remaining bytes after valid data
        size_t sequence_id;              // save path chunk order for network transfer
    };
    
    // P2P task queues (two independent workers)
    std::queue<P2PSendTask> p2p_send_queue_;
    std::queue<P2PRecvTask> p2p_recv_queue_;
    std::mutex p2p_send_queue_mutex_;
    std::mutex p2p_recv_queue_mutex_;
    std::condition_variable p2p_send_queue_cv_;
    std::condition_variable p2p_recv_queue_cv_;
    
    // Worker threads - unified workers for all threads
    std::thread encoder_thread_1_;
    std::thread encoder_thread_2_;
    std::thread send_worker_;       // Unified send worker
    std::thread recv_worker_;       // Unified recv worker
    std::thread xor_worker_;        // Unified XOR worker
    std::thread p2p_send_worker_;   // P2P send worker (independent thread)
    std::thread p2p_recv_worker_;   // P2P recv worker (independent thread)
    
    // ========== Load Mode 专用成员变量 ==========
    struct LoadEncodingTask {
        uintptr_t data_addr;
        size_t size;
        uintptr_t encoding_addr;
        uintptr_t recv_addr;
        size_t recv_chunk_size;
        uintptr_t parity_addr;
        bool is_receiver;
        uintptr_t p2p_partner_write_addr;  // For Step6: rank2 needs this to receive d3
    };
    
    std::queue<LoadEncodingTask> load_encoding_tasks_;
    std::mutex load_encoding_tasks_mutex_;
    std::condition_variable load_encoding_tasks_cv_;
    std::thread load_encoder_worker_;
    std::atomic<bool> load_encoding_completed_;
    std::atomic<bool> load_encoding_sentinel_received_;
    
    // Load mode pending tasks (用于 Step2 P2P 完成后提交 encoding)
    std::unordered_map<uintptr_t, LoadEncodingTask> pending_load_encoding_tasks_;
    std::mutex pending_load_encoding_tasks_mutex_;
    
    // ========== Load Mode 独立的通信队列和 worker ==========
    // Load send queue (for rank0/1 sending encoding to rank2/3)
    struct LoadSendTask {
        uintptr_t encoding_addr;
        size_t size;
    };
    std::queue<LoadSendTask> load_send_queue_;
    std::mutex load_send_queue_mutex_;
    std::condition_variable load_send_queue_cv_;
    std::thread load_send_worker_;
    std::atomic<bool> load_send_worker_completed_;
    std::atomic<bool> load_send_worker_sentinel_received_;
    
    // Load recv queue (for rank2/3 receiving encoding from rank0/1)
    struct LoadRecvTask {
        uintptr_t recv_addr;
        size_t size;
        uintptr_t parity_addr;
    };
    std::queue<LoadRecvTask> load_recv_queue_;
    std::mutex load_recv_queue_mutex_;
    std::condition_variable load_recv_queue_cv_;
    std::thread load_recv_worker_;
    std::atomic<bool> load_recv_worker_completed_;
    std::atomic<bool> load_recv_worker_sentinel_received_;
    
    // Load XOR queue
    struct LoadXORTask {
        uintptr_t local_encoding_addr;
        uintptr_t remote_encoding_addr;
        uintptr_t parity_addr;
        size_t size;
        uintptr_t p2p_partner_write_addr;  // For Step6: rank2 needs this to receive d3
        uintptr_t data_addr;
    };
    std::queue<LoadXORTask> load_xor_queue_;
    std::mutex load_xor_queue_mutex_;
    std::condition_variable load_xor_queue_cv_;
    std::thread load_xor_worker_;
    std::atomic<bool> load_xor_worker_completed_;
    std::atomic<bool> load_xor_worker_sentinel_received_;
    std::atomic<uint64_t> load_xor_total_ns_{0};
    std::atomic<size_t> load_xor_task_count_{0};
    std::atomic<uint64_t> load_xor_e2e_wall_ns_{0};
    std::atomic<bool> load_xor_e2e_wall_valid_{false};
    std::atomic<uint64_t> load_encode_total_ns_{0};
    std::atomic<size_t> load_encode_task_count_{0};
    std::atomic<uint64_t> load_encode_e2e_wall_ns_{0};
    std::atomic<bool> load_encode_e2e_wall_valid_{false};

    // Load path network I/O: per-task wall time summed (similar semantics to EC-NAIVE network_recv_ms)
    std::atomic<uint64_t> load_enc_xor_send_total_ns_{0};
    std::atomic<size_t> load_enc_xor_send_task_count_{0};
    std::atomic<uint64_t> load_enc_xor_recv_total_ns_{0};
    std::atomic<size_t> load_enc_xor_recv_task_count_{0};
    std::atomic<uint64_t> load_step2_p2p_send_total_ns_{0};
    std::atomic<size_t> load_step2_p2p_send_task_count_{0};
    std::atomic<uint64_t> load_step2_p2p_recv_total_ns_{0};
    std::atomic<size_t> load_step2_p2p_recv_task_count_{0};
    std::atomic<uint64_t> load_step6_p2p_send_total_ns_{0};
    std::atomic<size_t> load_step6_p2p_send_task_count_{0};
    std::atomic<uint64_t> load_step6_p2p_recv_total_ns_{0};
    std::atomic<size_t> load_step6_p2p_recv_task_count_{0};

    // Rank 2 load RDMA: sum of ibv_poll_cq wait only (after post_recv + ACK), per worker path.
    std::atomic<uint64_t> load_rank2_rdma_poll_enc_xor_recv_ns_{0};
    std::atomic<uint64_t> load_rank2_rdma_poll_step2_p2p_recv_ns_{0};
    std::atomic<uint64_t> load_rank2_rdma_poll_step6_p2p_recv_ns_{0};

    // Load net: wall from earliest load network op start to latest op end.
    std::mutex load_net_wall_mu_;
    bool load_net_wall_have_any_{false};
    std::chrono::steady_clock::time_point load_net_wall_first_{};
    std::chrono::steady_clock::time_point load_net_wall_last_{};
    std::atomic<uint64_t> load_net_wall_span_ns_{0};

    std::atomic<uint64_t> save_encode_total_ns_{0};
    std::atomic<size_t> save_encode_op_count_{0};

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

    void record_load_encode_op_(uint64_t ns) {
        load_encode_total_ns_.fetch_add(ns, std::memory_order_relaxed);
        load_encode_task_count_.fetch_add(1, std::memory_order_relaxed);
    }

    void record_load_xor_op_(uint64_t ns) {
        load_xor_total_ns_.fetch_add(ns, std::memory_order_relaxed);
        load_xor_task_count_.fetch_add(1, std::memory_order_relaxed);
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

    void touch_load_net_wall_(
        std::chrono::steady_clock::time_point t0,
        std::chrono::steady_clock::time_point t1) {
        std::lock_guard<std::mutex> lk(load_net_wall_mu_);
        if (!load_net_wall_have_any_) {
            load_net_wall_first_ = t0;
            load_net_wall_last_ = t1;
            load_net_wall_have_any_ = true;
        } else {
            if (t0 < load_net_wall_first_) {
                load_net_wall_first_ = t0;
            }
            if (t1 > load_net_wall_last_) {
                load_net_wall_last_ = t1;
            }
        }
        const uint64_t span_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                load_net_wall_last_ - load_net_wall_first_).count());
        load_net_wall_span_ns_.store(span_ns, std::memory_order_relaxed);
    }

    struct SaveEncodeScopeTimer {
        ECCHECKNative* owner;
        std::chrono::steady_clock::time_point t0;
        bool record;
        explicit SaveEncodeScopeTimer(ECCHECKNative* o, bool should_record = true)
            : owner(o), t0(std::chrono::steady_clock::now()), record(should_record) {}
        ~SaveEncodeScopeTimer() {
            if (record && owner != nullptr && !owner->is_load_mode_) {
                const auto t1 = std::chrono::steady_clock::now();
                const uint64_t ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                owner->record_save_encode_op_(ns);
            }
        }
    };

    struct SaveNetScopeTimer {
        ECCHECKNative* owner;
        std::chrono::steady_clock::time_point t0;
        explicit SaveNetScopeTimer(ECCHECKNative* o)
            : owner(o), t0(std::chrono::steady_clock::now()) {}
        ~SaveNetScopeTimer() {
            if (owner != nullptr && !owner->is_load_mode_) {
                const auto t1 = std::chrono::steady_clock::now();
                owner->touch_save_net_wall_(t0, t1);
            }
        }
    };

    // Load recv pipeline: wall clock from earliest recv start to latest recv end (overlapping workers/chunks).
    std::mutex load_recv_pipeline_e2e_mu_;
    bool load_recv_pipeline_e2e_have_any_{false};
    std::chrono::steady_clock::time_point load_recv_pipeline_e2e_first_{};
    std::chrono::steady_clock::time_point load_recv_pipeline_e2e_last_{};

    // Load XOR: 16 pthread workers (CPU affinity via ECCHECK_XOR_CPU_LIST), rank 2/3 only
    static constexpr int kLoadXorPoolSize = 16;
    struct LoadXorPoolJob {
        uintptr_t local_encoding_addr{0};
        uintptr_t remote_encoding_addr{0};
        uintptr_t parity_addr{0};
        size_t size{0};
    };
    std::array<pthread_t, kLoadXorPoolSize> load_xor_pool_threads_{};
    std::array<EccheckXorPoolWorkerCtx, kLoadXorPoolSize> load_xor_pool_ctx_{};
    std::array<int, kLoadXorPoolSize> load_xor_pool_cpus_{};
    std::atomic<bool> load_xor_pool_inited_{false};
    std::atomic<bool> load_xor_pool_stop_{false};
    std::mutex load_xor_pool_mutex_;
    std::condition_variable load_xor_pool_worker_cv_;
    std::condition_variable load_xor_pool_coordinator_cv_;
    std::atomic<uint64_t> load_xor_pool_epoch_{0};
    std::array<uint64_t, kLoadXorPoolSize> load_xor_pool_last_epoch_{};
    std::atomic<int> load_xor_pool_remaining_{0};
    LoadXorPoolJob load_xor_pool_shared_job_{};

    // Load encode: 16 pthread workers (CPU affinity via ECCHECK_ENCODE_CPU_LIST), all ranks in load mode
    static constexpr int kLoadEncodePoolSize = 16;
    struct LoadEncodePoolJob {
        uintptr_t data_addr{0};
        uintptr_t encoding_addr{0};
        size_t size{0};
        bool use_isal_ec{false};
        unsigned char* gftbls_ptr{nullptr};
        uint8_t fallback_coeff{0};
    };
    std::array<pthread_t, kLoadEncodePoolSize> load_encode_pool_threads_{};
    std::array<EccheckEncodePoolWorkerCtx, kLoadEncodePoolSize> load_encode_pool_ctx_{};
    std::array<int, kLoadEncodePoolSize> load_encode_pool_cpus_{};
    std::atomic<bool> load_encode_pool_inited_{false};
    std::atomic<bool> load_encode_pool_stop_{false};
    std::mutex load_encode_pool_mutex_;
    std::condition_variable load_encode_pool_worker_cv_;
    std::condition_variable load_encode_pool_coordinator_cv_;
    std::atomic<uint64_t> load_encode_pool_epoch_{0};
    std::array<uint64_t, kLoadEncodePoolSize> load_encode_pool_last_epoch_{};
    std::atomic<int> load_encode_pool_remaining_{0};
    LoadEncodePoolJob load_encode_pool_shared_job_{};

    // ── Save-path encode pool (16 pthread, ec_encode_data) ────────
    static constexpr int kEcRsEncodePoolSize = 16;
    struct EcRsEncodeJob {
        uintptr_t data_addr{0};
        uintptr_t encoding_addr{0};
        size_t size{0};
        unsigned char* gftbls_ptr{nullptr};
    };
    struct EcRsEncodePoolWorkerCtx { class ECCHECKNative* self{nullptr}; int wid{0}; };
    std::array<pthread_t, kEcRsEncodePoolSize> ec_rs_encode_pool_threads_{};
    std::array<EcRsEncodePoolWorkerCtx, kEcRsEncodePoolSize> ec_rs_encode_pool_ctx_{};
    std::array<int, kEcRsEncodePoolSize> ec_rs_encode_pool_cpus_{};
    std::atomic<bool> ec_rs_encode_pool_inited_{false};
    std::atomic<bool> ec_rs_encode_pool_stop_{false};
    std::mutex ec_rs_encode_pool_mutex_;
    std::condition_variable ec_rs_encode_pool_worker_cv_;
    std::condition_variable ec_rs_encode_pool_coordinator_cv_;
    std::atomic<uint64_t> ec_rs_encode_pool_epoch_{0};
    std::array<uint64_t, kEcRsEncodePoolSize> ec_rs_encode_pool_last_epoch_{};
    std::atomic<int> ec_rs_encode_pool_remaining_{0};
    EcRsEncodeJob ec_rs_encode_pool_shared_job_{};
    std::mutex ec_rs_encode_pool_work_mutex_;  // serialize encoder_1/encoder_2 dispatch

    bool release_data_buffer_once(uintptr_t data_addr) {
        if (data_addr == 0) {
            return false;
        }
        {
            std::lock_guard<std::mutex> state_lock(data_buffer_state_mutex_);
            auto it = data_buffer_states_.find(data_addr);
            if (it == data_buffer_states_.end()) {
                return false;
            }
            data_buffer_states_.erase(it);
        }
        {
            std::lock_guard<std::mutex> release_lock(release_queue_mutex_);
            data_buffers_to_release_.push(data_addr);
        }
        return true;
    }

    // ── Save-path XOR pool (16 pthread, xor_gen) ──────────────────
    static constexpr int kEcXorPoolSize = 16;
    struct EcXorJob {
        uintptr_t dst{0};
        uintptr_t src1{0};
        uintptr_t src2{0};
        size_t size{0};
    };
    struct EcXorPoolWorkerCtx { class ECCHECKNative* self{nullptr}; int wid{0}; };
    std::array<pthread_t, kEcXorPoolSize> ec_xor_pool_threads_{};
    std::array<EcXorPoolWorkerCtx, kEcXorPoolSize> ec_xor_pool_ctx_{};
    std::array<int, kEcXorPoolSize> ec_xor_pool_cpus_{};
    std::atomic<bool> ec_xor_pool_inited_{false};
    std::atomic<bool> ec_xor_pool_stop_{false};
    std::mutex ec_xor_pool_mutex_;
    std::condition_variable ec_xor_pool_worker_cv_;
    std::condition_variable ec_xor_pool_coordinator_cv_;
    std::atomic<uint64_t> ec_xor_pool_epoch_{0};
    std::array<uint64_t, kEcXorPoolSize> ec_xor_pool_last_epoch_{};
    std::atomic<int> ec_xor_pool_remaining_{0};
    EcXorJob ec_xor_pool_shared_job_{};
    std::mutex ec_xor_pool_work_mutex_;
    
    // Load mode pending XOR encoding (matching encoding and recv)
    std::unordered_map<uintptr_t, uintptr_t> load_pending_xor_encoding_;
    std::mutex load_pending_xor_mutex_;
    
    // Load mode recv_addr mappings
    std::unordered_map<uintptr_t, uintptr_t> load_recv_to_parity_;
    std::mutex load_recv_to_parity_mutex_;
    std::unordered_map<uintptr_t, uintptr_t> load_recv_to_data_;
    std::mutex load_recv_to_data_mutex_;
    std::unordered_map<uintptr_t, uintptr_t> load_recv_to_p2p_partner_write_;
    std::mutex load_recv_to_p2p_mutex_;
    
    // Load mode P2P queues and workers (completely separate from save mode)
    std::queue<P2PSendTask> load_p2p_send_queue_;
    std::queue<P2PRecvTask> load_p2p_recv_queue_;
    std::mutex load_p2p_send_queue_mutex_;
    std::mutex load_p2p_recv_queue_mutex_;
    std::condition_variable load_p2p_send_queue_cv_;
    std::condition_variable load_p2p_recv_queue_cv_;
    std::thread load_p2p_send_worker_;
    std::thread load_p2p_recv_worker_;
    std::atomic<bool> load_p2p_send_worker_completed_;
    std::atomic<bool> load_p2p_send_worker_sentinel_received_;
    std::atomic<bool> load_p2p_recv_worker_completed_;
    std::atomic<bool> load_p2p_recv_worker_sentinel_received_;
    
    // Load mode Step6 P2P queues and workers (completely separate from Step2)
    std::queue<P2PSendTask> load_step6_p2p_send_queue_;
    std::queue<P2PRecvTask> load_step6_p2p_recv_queue_;
    std::mutex load_step6_p2p_send_queue_mutex_;
    std::mutex load_step6_p2p_recv_queue_mutex_;
    std::condition_variable load_step6_p2p_send_queue_cv_;
    std::condition_variable load_step6_p2p_recv_queue_cv_;
    std::thread load_step6_p2p_send_worker_;
    std::thread load_step6_p2p_recv_worker_;
    std::atomic<bool> load_step6_p2p_send_worker_completed_;
    std::atomic<bool> load_step6_p2p_send_worker_sentinel_received_;
    std::atomic<bool> load_step6_p2p_recv_worker_completed_;
    std::atomic<bool> load_step6_p2p_recv_worker_sentinel_received_;
    
    // NCCL communicators - 4 independent communicators for send/recv separation
#ifdef NCCL_AVAILABLE
    ncclComm_t nccl_comm_xor_send_;   // XOR send communicator
    ncclComm_t nccl_comm_xor_recv_;   // XOR recv communicator
    ncclComm_t nccl_comm_p2p_send_;   // P2P send communicator
    ncclComm_t nccl_comm_p2p_recv_;   // P2P recv communicator
    bool nccl_xor_send_initialized_;
    bool nccl_xor_recv_initialized_;
    bool nccl_p2p_send_initialized_;
    bool nccl_p2p_recv_initialized_;
#ifdef __GNUC__
#endif
#else
    bool nccl_xor_send_initialized_;
    bool nccl_xor_recv_initialized_;
    bool nccl_p2p_send_initialized_;
    bool nccl_p2p_recv_initialized_;
#endif

    // EC parameters (k, rows=2) and tables
    int k_;
    int rows_;
    int data_block_index_;
    unsigned char *a_mat_;    // RS matrix (k * m)
    unsigned char *g_tbls_;   // tables produced by ec_init_tables (32 * k * rows)
    unsigned char decode_coefficient_0_;
    unsigned char decode_coefficient_1_;

    // NCCL IDs stored as member variables (passed from Python via point-to-point exchange)
    std::vector<uint8_t> nccl_id_xor_send_;   // XOR send ID
    std::vector<uint8_t> nccl_id_xor_recv_;   // XOR recv ID
    std::vector<uint8_t> nccl_id_p2p_send_;   // P2P send ID
    std::vector<uint8_t> nccl_id_p2p_recv_;   // P2P recv ID

    // Synchronization for NCCL initialization
    std::atomic<bool> nccl_xor_send_init_completed_;
    std::atomic<bool> nccl_xor_recv_init_completed_;
    std::atomic<bool> nccl_p2p_send_init_completed_;
    std::atomic<bool> nccl_p2p_recv_init_completed_;
    std::mutex nccl_init_mutex_;
    std::condition_variable nccl_init_cv_;

    // P2P configuration
    int p2p_partner_rank_;  // P2P partner rank (adjacent pairing: 0<->1, 2<->3)
    
    // Load mode configuration
    bool is_load_mode_;     // Flag indicating if in load mode (recovery pipeline)
    bool is_two_failures_load_mode_; // Flag for two-failure hardware recovery mode
    int failed_rank_;       // Failed rank number (e.g., 2 for rank2 recovery scenario)
    
    // ASIO connection manager
    AsioConnectionManager asio_conn_mgr_;
    bool asio_initialized_;
    bool use_asio_;  // Flag to indicate if using ASIO instead of NCCL
    bool use_rdma_;  // Flag to indicate if using RDMA (ibverbs) instead of ASIO/NCCL
    
#ifdef __linux__
    // RDMA resources (ibverbs)
    std::string my_ip_;  // Local IP for RDMA device selection
    ibv_context* rdma_context_;
    ibv_pd* rdma_pd_;
    ibv_cq* rdma_xor_send_cq_;
    ibv_cq* rdma_xor_recv_cq_;
    ibv_cq* rdma_p2p_send_cq_;
    ibv_cq* rdma_p2p_recv_cq_;
    ibv_qp* rdma_xor_send_qp_;
    ibv_qp* rdma_xor_recv_qp_;
    ibv_qp* rdma_xor_qp_;  // Compatibility alias for older/simple paths
    ibv_qp* rdma_p2p_send_qp_;
    ibv_qp* rdma_p2p_recv_qp_;
    ibv_qp* rdma_p2p_qp_;  // Compatibility alias for simple/load paths
    
    // RDMA registered buffers
    std::map<uintptr_t, RdmaBufferInfo> rdma_registered_buffers_;
    std::mutex rdma_buffer_mutex_;
    
    // RDMA temp buffers for unregistered memory (send/recv)
    std::vector<uint8_t> rdma_temp_send_buffer_;
    std::vector<uint8_t> rdma_temp_recv_buffer_;
    ibv_mr* rdma_temp_send_mr_;
    ibv_mr* rdma_temp_recv_mr_;
    std::mutex rdma_xor_send_control_mutex_;
    std::mutex rdma_xor_recv_control_mutex_;
    std::mutex rdma_p2p_send_control_mutex_;
    std::mutex rdma_p2p_recv_control_mutex_;
    
    // Step6 P2P (Load: rank2<->rank3 only)
    ibv_cq* rdma_step6_p2p_send_cq_;
    ibv_cq* rdma_step6_p2p_recv_cq_;
    ibv_qp* rdma_step6_p2p_qp_;
    std::mutex rdma_step6_p2p_send_control_mutex_;
    std::mutex rdma_step6_p2p_recv_control_mutex_;
    
    // TCP sockets for RDMA connection setup (control plane)
    int rdma_listen_sock_;
    int rdma_xor_control_sock_;
    int rdma_p2p_control_sock_;
#endif
    
    // Helper function to synchronize NCCL operation
#ifdef NCCL_AVAILABLE
    void sync_nccl_operation(const char* operation_name) {
        // Simply synchronize the default stream (stream 0) where NCCL operations execute
        // This ensures we wait for all operations on the default stream to complete
        cudaError_t err = cudaStreamSynchronize(0);
        if (err != cudaSuccess) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Failed to synchronize stream for " << operation_name << ": " << cudaGetErrorString(err) << std::endl;
        } else {
        }
    }
#endif

    // ========== XOR配置构建函数 ==========
    
    void build_xor_config() {
        xor_config_.xor_partner_rank = paired_rank_;
        // Use rank_in_group_ (0-3) for XOR role; xor_partner_rank is global (set by Python).
        if (is_two_failures_load_mode_) {
            // ── Two-failure hardware recovery (rig1+rig2 failed) ──
            // XOR roles are INVERTED vs save mode:
            //   Encoder thread 1 (parity row 0): rig0→rig2, rig1→rig3
            //     → rig0 sends enc_0 to rig2, rig2 XORs to recover d2
            //     → rig1 sends enc_0 to rig3, rig3 XORs to verify d3
            //   Encoder thread 2 (parity row 1): rig2→rig0, rig3→rig1
            //     → rig2 sends enc_1 to rig0, rig0 XORs → p0
            //     → rig3 sends enc_1 to rig1, rig1 XORs → 2·d3 or d2
            if (rank_in_group_ == 0) {
                xor_config_.thread0_is_receiver = false;  // thread1: send enc_0(d0) to rig2
                xor_config_.thread1_is_receiver = true;   // thread2: recv enc_1(p2) from rig2, XOR
            } else if (rank_in_group_ == 1) {
                xor_config_.thread0_is_receiver = false;  // thread1: send enc_0(d1) to rig3
                xor_config_.thread1_is_receiver = true;   // thread2: recv enc_1(p3) from rig3, XOR
            } else if (rank_in_group_ == 2) {
                xor_config_.thread0_is_receiver = true;   // thread1: recv enc_0(d0) from rig0, XOR → d2
                xor_config_.thread1_is_receiver = false;  // thread2: send enc_1(p2) to rig0
            } else if (rank_in_group_ == 3) {
                xor_config_.thread0_is_receiver = true;   // thread1: recv enc_0(d1) from rig1, XOR → d3
                xor_config_.thread1_is_receiver = false;  // thread2: send enc_1(p3) to rig1
            } else {
                xor_config_.xor_partner_rank = -1;
                xor_config_.thread0_is_receiver = false;
                xor_config_.thread1_is_receiver = false;
            }
        } else {
            // Save mode: role by rank_in_group (0-3)
            if (rank_in_group_ == 0) {
                xor_config_.thread0_is_receiver = true;   // thread0 recv from partner, XOR
                xor_config_.thread1_is_receiver = false;   // thread1 send to partner
            } else if (rank_in_group_ == 1) {
                xor_config_.thread0_is_receiver = true;
                xor_config_.thread1_is_receiver = false;
            } else if (rank_in_group_ == 2) {
                xor_config_.thread0_is_receiver = false;   // thread0 send to partner
                xor_config_.thread1_is_receiver = true;    // thread1 recv from partner, XOR
            } else if (rank_in_group_ == 3) {
                xor_config_.thread0_is_receiver = false;
                xor_config_.thread1_is_receiver = true;
            } else {
                xor_config_.xor_partner_rank = -1;
                xor_config_.thread0_is_receiver = false;
                xor_config_.thread1_is_receiver = false;
            }
        }
        
    }
    
    // ========== P2P配置构建函数 ==========
    
    void build_p2p_config() {
        // Use Python-provided partner when available; fall back to adjacent pairing.
        if (p2p_partner_rank_ < 0) {
            if (rank_ % 2 == 0) {
                p2p_partner_rank_ = rank_ + 1;
            } else {
                p2p_partner_rank_ = rank_ - 1;
            }
        }

        if (p2p_partner_rank_ < 0 || p2p_partner_rank_ >= world_size_) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Invalid P2P partner rank: "
                      << p2p_partner_rank_ << std::endl;
            p2p_partner_rank_ = -1;
                }

    }

    // Helper: even rank_in_group in each P2P pair sends parity; odd sends data.
    bool is_p2p_parity_sender() const {
        return (rank_in_group_ % 2) == 0;
    }

    // ========== NCCL初始化函数 ==========
    
    // Helper function to map global rank to communicator-internal rank (0 or 1)
    // For thread1 comm (rank0↔rank2): rank0->0, rank2->1
    // For thread2 comm (rank1↔rank3): rank1->0, rank3->1
    // For P2P comm (rank0↔rank1): rank0->0, rank1->1
    // For P2P comm (rank2↔rank3): rank2->0, rank3->1
    int get_rank_in_comm(int global_rank, int comm_type) const {
        // comm_type: 0=thread1 (rank0↔rank2), 1=thread2 (rank1↔rank3), 2=p2p_0_1, 3=p2p_2_3
        if (comm_type == 0) {
            // thread1: rank0↔rank2
            return (global_rank == 0) ? 0 : 1;
        } else if (comm_type == 1) {
            // thread2: rank1↔rank3
            return (global_rank == 1) ? 0 : 1;
        } else if (comm_type == 2) {
            // p2p_0_1: rank0↔rank1
            return (global_rank == 0) ? 0 : 1;
        } else if (comm_type == 3) {
            // p2p_2_3: rank2↔rank3
            return (global_rank == 2) ? 0 : 1;
        }
        return -1; // Error
    }
    
    // Helper function to get peer rank in communicator for thread1/thread2
    // Returns the communicator-internal rank (0 or 1) for the given peer_rank
    int get_peer_rank_in_thread_comm(int peer_rank, bool is_thread1) const {
        if (is_thread1) {
            // thread1 communicator: rank0↔rank2
            if (peer_rank == 0) return 0;
            if (peer_rank == 2) return 1;
        } else {
            // thread2 communicator: rank1↔rank3
            if (peer_rank == 1) return 0;
            if (peer_rank == 3) return 1;
        }
        return -1; // Error
    }
    
    // Helper function to get peer rank in communicator for P2P (works for multi-rank: comm has rank_ and p2p_partner_rank_)
    int get_peer_rank_in_p2p_comm(int peer_rank) const {
        if (peer_rank == rank_) return 0;
        if (peer_rank == p2p_partner_rank_) return 1;
        return -1;  // Error
    }
    
    void init_nccl_xor_send() {
#ifdef NCCL_AVAILABLE
        // Only rank0↔rank2 and rank1↔rank3 participate in XOR send
        int partner_rank = xor_config_.xor_partner_rank;
        if (partner_rank < 0) {
            nccl_xor_send_initialized_ = false;
            nccl_xor_send_init_completed_ = true;
            nccl_init_cv_.notify_all();
            return;
        }
        
        // Read NCCL ID from member variable (set by constructor)
        if (nccl_id_xor_send_.size() != sizeof(ncclUniqueId)) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Invalid NCCL ID size for XOR send: " 
                      << nccl_id_xor_send_.size() << " (expected " << sizeof(ncclUniqueId) << ")" << std::endl;
            nccl_xor_send_initialized_ = false;
            nccl_xor_send_init_completed_ = true;
            nccl_init_cv_.notify_all();
            return;
        }
        
        ncclUniqueId nccl_id;
        std::memcpy(&nccl_id, nccl_id_xor_send_.data(), sizeof(ncclUniqueId));
        
        // Map global rank to communicator-internal rank: smaller rank -> 0, larger rank -> 1
        int local_rank_in_pair = (rank_ < partner_rank) ? 0 : 1;
        
        // Initialize NCCL communicator with 2 ranks only
        ncclResult_t init_result = ncclCommInitRank(&nccl_comm_xor_send_, 2, nccl_id, local_rank_in_pair);
        if (init_result != ncclSuccess) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: XOR send ncclCommInitRank failed: " 
                      << ncclGetErrorString(init_result) << " (code: " << init_result << ")" << std::endl;
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] XOR send init params: nranks=2, local_rank=" 
                      << local_rank_in_pair << ", nccl_id size=" << sizeof(ncclUniqueId) << std::endl;
            std::cerr.flush();
            nccl_xor_send_initialized_ = false;
        } else {
            nccl_xor_send_initialized_ = true;
        }
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_xor_send_init_completed_ = true;
        }
        nccl_init_cv_.notify_all();
#else
        nccl_xor_send_initialized_ = false;
        nccl_xor_send_init_completed_ = true;
        nccl_init_cv_.notify_all();
#endif
    }
    
    void init_nccl_xor_recv() {
#ifdef NCCL_AVAILABLE
        // Only rank0↔rank2 and rank1↔rank3 participate in XOR recv
        int partner_rank = xor_config_.xor_partner_rank;
        if (partner_rank < 0) {
            nccl_xor_recv_initialized_ = false;
            nccl_xor_recv_init_completed_ = true;
            nccl_init_cv_.notify_all();
            return;
        }
        
        // Read NCCL ID from member variable (set by constructor)
        if (nccl_id_xor_recv_.size() != sizeof(ncclUniqueId)) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Invalid NCCL ID size for XOR recv: " 
                      << nccl_id_xor_recv_.size() << " (expected " << sizeof(ncclUniqueId) << ")" << std::endl;
            nccl_xor_recv_initialized_ = false;
            nccl_xor_recv_init_completed_ = true;
            nccl_init_cv_.notify_all();
            return;
        }
        
        ncclUniqueId nccl_id;
        std::memcpy(&nccl_id, nccl_id_xor_recv_.data(), sizeof(ncclUniqueId));
        
        // Map global rank to communicator-internal rank: smaller rank -> 0, larger rank -> 1
        int local_rank_in_pair = (rank_ < partner_rank) ? 0 : 1;
        
        // Initialize NCCL communicator with 2 ranks only
        ncclResult_t init_result = ncclCommInitRank(&nccl_comm_xor_recv_, 2, nccl_id, local_rank_in_pair);
        if (init_result != ncclSuccess) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: XOR recv ncclCommInitRank failed: " 
                      << ncclGetErrorString(init_result) << " (code: " << init_result << ")" << std::endl;
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] XOR recv init params: nranks=2, local_rank=" 
                      << local_rank_in_pair << ", nccl_id size=" << sizeof(ncclUniqueId) << std::endl;
            std::cerr.flush();
            nccl_xor_recv_initialized_ = false;
        } else {
            nccl_xor_recv_initialized_ = true;
        }
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_xor_recv_init_completed_ = true;
        }
        nccl_init_cv_.notify_all();
#else
        nccl_xor_recv_initialized_ = false;
        nccl_xor_recv_init_completed_ = true;
        nccl_init_cv_.notify_all();
#endif
    }
    
    void init_nccl_p2p_send() {
#ifdef NCCL_AVAILABLE
        // P2P pairing: rank0↔rank1 and rank2↔rank3
        int partner_rank = p2p_partner_rank_;
        if (partner_rank < 0) {
            nccl_p2p_send_initialized_ = false;
            nccl_p2p_send_init_completed_ = true;
            nccl_init_cv_.notify_all();
            return;
        }
        
        // Read NCCL ID from member variable (set by constructor)
        if (nccl_id_p2p_send_.size() != sizeof(ncclUniqueId)) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Invalid NCCL ID size for P2P send: " 
                      << nccl_id_p2p_send_.size() << " (expected " << sizeof(ncclUniqueId) << ")" << std::endl;
            nccl_p2p_send_initialized_ = false;
            nccl_p2p_send_init_completed_ = true;
            nccl_init_cv_.notify_all();
            return;
        }
        
        ncclUniqueId nccl_id;
        std::memcpy(&nccl_id, nccl_id_p2p_send_.data(), sizeof(ncclUniqueId));
        
        // Map global rank to communicator-internal rank: smaller rank -> 0, larger rank -> 1
        int local_rank_in_pair = (rank_ < partner_rank) ? 0 : 1;
        
        // Initialize NCCL communicator with 2 ranks only
        ncclResult_t init_result = ncclCommInitRank(&nccl_comm_p2p_send_, 2, nccl_id, local_rank_in_pair);
        if (init_result != ncclSuccess) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: P2P send ncclCommInitRank failed: " 
                      << ncclGetErrorString(init_result) << " (code: " << init_result << ")" << std::endl;
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] P2P send init params: nranks=2, local_rank=" 
                      << local_rank_in_pair << ", nccl_id size=" << sizeof(ncclUniqueId) << std::endl;
            std::cerr.flush();
            nccl_p2p_send_initialized_ = false;
        } else {
            nccl_p2p_send_initialized_ = true;
        }
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_p2p_send_init_completed_ = true;
        }
        nccl_init_cv_.notify_all();
#else
        nccl_p2p_send_initialized_ = false;
        nccl_p2p_send_init_completed_ = true;
        nccl_init_cv_.notify_all();
#endif
    }
    
    void init_nccl_p2p_recv() {
#ifdef NCCL_AVAILABLE
        // P2P pairing: rank0↔rank1 and rank2↔rank3
        int partner_rank = p2p_partner_rank_;
        if (partner_rank < 0) {
            nccl_p2p_recv_initialized_ = false;
            nccl_p2p_recv_init_completed_ = true;
            nccl_init_cv_.notify_all();
            return;
        }
        
        // Read NCCL ID from member variable (set by constructor)
        if (nccl_id_p2p_recv_.size() != sizeof(ncclUniqueId)) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Invalid NCCL ID size for P2P recv: " 
                      << nccl_id_p2p_recv_.size() << " (expected " << sizeof(ncclUniqueId) << ")" << std::endl;
            nccl_p2p_recv_initialized_ = false;
            nccl_p2p_recv_init_completed_ = true;
            nccl_init_cv_.notify_all();
            return;
        }
        
        ncclUniqueId nccl_id;
        std::memcpy(&nccl_id, nccl_id_p2p_recv_.data(), sizeof(ncclUniqueId));
        
        // Map global rank to communicator-internal rank: smaller rank -> 0, larger rank -> 1
        int local_rank_in_pair = (rank_ < partner_rank) ? 0 : 1;
        
        // Initialize NCCL communicator with 2 ranks only
        ncclResult_t init_result = ncclCommInitRank(&nccl_comm_p2p_recv_, 2, nccl_id, local_rank_in_pair);
        if (init_result != ncclSuccess) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: P2P recv ncclCommInitRank failed: " 
                      << ncclGetErrorString(init_result) << " (code: " << init_result << ")" << std::endl;
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] P2P recv init params: nranks=2, local_rank=" 
                      << local_rank_in_pair << ", nccl_id size=" << sizeof(ncclUniqueId) << std::endl;
            std::cerr.flush();
            nccl_p2p_recv_initialized_ = false;
        } else {
            nccl_p2p_recv_initialized_ = true;
        }
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_p2p_recv_init_completed_ = true;
        }
        nccl_init_cv_.notify_all();
#else
        nccl_p2p_recv_initialized_ = false;
        nccl_p2p_recv_init_completed_ = true;
        nccl_init_cv_.notify_all();
#endif
    }
    
    void cleanup_nccl() {
#ifdef NCCL_AVAILABLE
        if (nccl_xor_send_initialized_) {
            ncclCommDestroy(nccl_comm_xor_send_);
        }
        if (nccl_xor_recv_initialized_) {
            ncclCommDestroy(nccl_comm_xor_recv_);
        }
        if (nccl_p2p_send_initialized_) {
            ncclCommDestroy(nccl_comm_p2p_send_);
        }
        if (nccl_p2p_recv_initialized_) {
            ncclCommDestroy(nccl_comm_p2p_recv_);
        }
#endif
    }

    // ========== Worker线程函数 ==========

    static std::array<int, kLoadEncodePoolSize> parse_load_encode_pool_cpus_or_throw() {
        std::array<int, kLoadEncodePoolSize> cpus{};
        const char* env = std::getenv("ECCHECK_ENCODE_CPU_LIST");
        if (!env || !*env) {
            for (int i = 0; i < kLoadEncodePoolSize; ++i) {
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
                throw std::runtime_error("ECCHECK_ENCODE_CPU_LIST: invalid CPU id token");
            }
            parsed.push_back(static_cast<int>(v));
            p = end;
        }
        if (parsed.size() != static_cast<size_t>(kLoadEncodePoolSize)) {
            throw std::runtime_error(
                "ECCHECK_ENCODE_CPU_LIST must contain exactly 16 comma-separated CPU ids "
                "(or unset to use 0..15)");
        }
        for (size_t i = 0; i < cpus.size(); ++i) {
            cpus[i] = parsed[i];
        }
        return cpus;
    }

    void load_encode_pool_init() {
        if (load_encode_pool_inited_.load(std::memory_order_acquire)) {
            return;
        }
        load_encode_pool_cpus_ = parse_load_encode_pool_cpus_or_throw();
        load_encode_pool_stop_.store(false, std::memory_order_release);
        load_encode_pool_epoch_.store(0, std::memory_order_release);
        load_encode_pool_remaining_.store(0, std::memory_order_release);
        for (auto& e : load_encode_pool_last_epoch_) {
            e = 0;
        }
        for (int i = 0; i < kLoadEncodePoolSize; ++i) {
            load_encode_pool_ctx_[static_cast<size_t>(i)].self = this;
            load_encode_pool_ctx_[static_cast<size_t>(i)].wid = i;
            int rc = pthread_create(
                &load_encode_pool_threads_[static_cast<size_t>(i)],
                nullptr,
                &ECCHECKNative::load_encode_pool_pthread_entry,
                &load_encode_pool_ctx_[static_cast<size_t>(i)]);
            if (rc != 0) {
                load_encode_pool_stop_.store(true, std::memory_order_release);
                load_encode_pool_worker_cv_.notify_all();
                for (int j = 0; j < i; ++j) {
                    pthread_join(load_encode_pool_threads_[static_cast<size_t>(j)], nullptr);
                }
                throw std::runtime_error(
                    std::string("EC-CHECK: pthread_create for load encode pool failed: ") + std::strerror(rc));
            }
        }
        load_encode_pool_inited_.store(true, std::memory_order_release);
    }

    void load_encode_pool_shutdown() {
        if (!load_encode_pool_inited_.load(std::memory_order_acquire)) {
            return;
        }
        load_encode_pool_stop_.store(true, std::memory_order_release);
        load_encode_pool_worker_cv_.notify_all();
        for (int i = 0; i < kLoadEncodePoolSize; ++i) {
            pthread_join(load_encode_pool_threads_[static_cast<size_t>(i)], nullptr);
        }
        load_encode_pool_stop_.store(false, std::memory_order_release);
        load_encode_pool_inited_.store(false, std::memory_order_release);
    }

    static void* load_encode_pool_pthread_entry(void* arg) {
        auto* ctx = static_cast<EccheckEncodePoolWorkerCtx*>(arg);
        ctx->self->load_encode_pool_worker_loop(ctx->wid);
        return nullptr;
    }

    void load_encode_pool_execute_stripe_from_job(const LoadEncodePoolJob& job, int wid) {
        const size_t total = job.size;
        const size_t base = total / static_cast<size_t>(kLoadEncodePoolSize);
        const size_t rem = total % static_cast<size_t>(kLoadEncodePoolSize);
        size_t off;
        size_t len;
        if (wid < kLoadEncodePoolSize - 1) {
            off = static_cast<size_t>(wid) * base;
            len = base;
        } else {
            off = static_cast<size_t>(kLoadEncodePoolSize - 1) * base;
            len = base + rem;
        }
        if (len == 0) {
            return;
        }
        auto at = [](uintptr_t base_ptr, size_t o) -> unsigned char* {
            return reinterpret_cast<unsigned char*>(base_ptr + o);
        };
        unsigned char* d = at(job.data_addr, off);
        unsigned char* e = at(job.encoding_addr, off);
        if (job.use_isal_ec) {
            unsigned char* srcs[1] = {d};
            unsigned char* dests[1] = {e};
            ec_encode_data(static_cast<int>(len), 1, 1, job.gftbls_ptr, srcs, dests);
        } else {
            const uint32_t fc = static_cast<uint32_t>(job.fallback_coeff);
            for (size_t i = 0; i < len; ++i) {
                e[i] = static_cast<unsigned char>((static_cast<uint32_t>(d[i]) * fc) & 0xFFu);
            }
        }
    }

    void load_encode_pool_worker_loop(int wid) {
        const int cpu = load_encode_pool_cpus_[static_cast<size_t>(wid)];
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (cpu >= 0 && static_cast<unsigned>(cpu) < CPU_SETSIZE) {
            CPU_SET(static_cast<unsigned>(cpu), &cpuset);
            int af = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
            if (af != 0) {
                std::cerr << "EC-CHECK: load_encode_pool worker " << wid
                          << " pthread_setaffinity_np failed: " << af << std::endl;
            }
        } else {
            std::cerr << "EC-CHECK: load_encode_pool worker " << wid << " CPU id " << cpu
                      << " invalid or >= CPU_SETSIZE, skipping affinity" << std::endl;
        }

        while (true) {
            std::unique_lock<std::mutex> lk(load_encode_pool_mutex_);
            load_encode_pool_worker_cv_.wait(lk, [&] {
                return load_encode_pool_stop_.load(std::memory_order_acquire) ||
                       (load_encode_pool_last_epoch_[static_cast<size_t>(wid)] <
                        load_encode_pool_epoch_.load(std::memory_order_acquire));
            });
            if (load_encode_pool_stop_.load(std::memory_order_acquire)) {
                break;
            }
            uint64_t ep = load_encode_pool_epoch_.load(std::memory_order_acquire);
            LoadEncodePoolJob local_copy = load_encode_pool_shared_job_;
            lk.unlock();

            load_encode_pool_execute_stripe_from_job(local_copy, wid);

            {
                std::lock_guard<std::mutex> guard(load_encode_pool_mutex_);
                load_encode_pool_last_epoch_[static_cast<size_t>(wid)] = ep;
            }

            const int left =
                load_encode_pool_remaining_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            if (left == 0) {
                load_encode_pool_coordinator_cv_.notify_one();
            }
        }
    }

    void load_encode_pool_run_parallel(const LoadEncodePoolJob& task) {
        {
            std::lock_guard<std::mutex> publish(load_encode_pool_mutex_);
            if (should_stop_threads_.load(std::memory_order_acquire)) {
                return;
            }
            load_encode_pool_shared_job_ = task;
            load_encode_pool_epoch_.fetch_add(1, std::memory_order_acq_rel);
            load_encode_pool_remaining_.store(kLoadEncodePoolSize, std::memory_order_release);
        }
        load_encode_pool_worker_cv_.notify_all();
        std::unique_lock<std::mutex> lk(load_encode_pool_mutex_);
        load_encode_pool_coordinator_cv_.wait(lk, [&] {
            return load_encode_pool_remaining_.load(std::memory_order_acquire) == 0 ||
                   should_stop_threads_.load(std::memory_order_acquire);
        });
    }

    // Build one encode job for striped load-encode pool (ec_encode_data k=1,rows=1 is byte-independent).
    bool try_build_load_encode_pool_job(
        uintptr_t data_addr,
        size_t size,
        uintptr_t encoding_addr,
        int coefficient,
        LoadEncodePoolJob* out) {
        if (k_ <= 0 || rows_ != 2 || g_tbls_ == nullptr) {
            out->data_addr = data_addr;
            out->encoding_addr = encoding_addr;
            out->size = size;
            out->use_isal_ec = false;
            out->gftbls_ptr = nullptr;
            out->fallback_coeff = static_cast<uint8_t>(coefficient & 0xFF);
            return true;
        }

        int parity_idx = coefficient;
        if (parity_idx < 0 || parity_idx >= rows_) {
            parity_idx = 0;
        }

        if (is_load_mode_ && failed_rank_in_group_ == 2) {
            if (rank_in_group_ < 2) {
                parity_idx = decode_coefficient_0_;
            } else if (rank_in_group_ < 4) {
                parity_idx = decode_coefficient_1_;
            } else {
                std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: Invalid rank_in_group " << rank_in_group_
                          << " for load mode (expected 0-3)" << std::endl;
                parity_idx = 1;
            }
            if (decode_coefficient_0_ == 0 || decode_coefficient_1_ == 0) {
                std::cerr << "EC-CHECK: [Rank " << rank_ << "] WARNING: Decode coefficient is 0, this may cause issues"
                          << std::endl;
            }
        }

        if (data_block_index_ < 0 || data_block_index_ >= k_) {
            std::cerr << "EC-CHECK: invalid data_block_index_=" << data_block_index_ << " for k=" << k_ << std::endl;
            return false;
        }

        const size_t tbl_offset = (static_cast<size_t>(parity_idx * k_ + data_block_index_)) * 32u;
        out->data_addr = data_addr;
        out->encoding_addr = encoding_addr;
        out->size = size;
        out->use_isal_ec = true;
        out->gftbls_ptr = g_tbls_ + tbl_offset;
        out->fallback_coeff = 0;
        return true;
    }
    
    void encode_with_coefficient(uintptr_t data_addr, size_t size, uintptr_t encoding_addr, int coefficient) {
        // 使用 isa-l 的 EC 编码对整块 buffer 进行编码。
        // 我们在初始化时已经生成了 RS 矩阵并通过 ec_init_tables 产生了 g_tbls_。
        // 每个 encoder 线程只保留自己负责的 parity（encoding_addr 指向本地 parity buffer）。
        // Load mode: optional 16-thread striped pool (ECCHECK_ENCODE_CPU_LIST).

        if (ec_rs_encode_pool_inited_.load(std::memory_order_acquire)) {
            // Save path: report one parity lane only, not both overlapped/serialized lanes.
            std::lock_guard<std::mutex> work_lk(ec_rs_encode_pool_work_mutex_);
            int parity_idx = coefficient;
            if (parity_idx < 0 || parity_idx >= rows_) parity_idx = 0;
            SaveEncodeScopeTimer encode_timer(this, parity_idx == 0);
            if (data_block_index_ >= 0 && data_block_index_ < k_) {
                size_t tbl_off = (static_cast<size_t>(parity_idx * k_ + data_block_index_)) * 32u;
                ec_rs_encode_pool_run_parallel(data_addr, encoding_addr, size, g_tbls_ + tbl_off);
            }
            return;
        }
        SaveEncodeScopeTimer encode_timer(this);
        if (load_encode_pool_inited_.load(std::memory_order_acquire)) {
            LoadEncodePoolJob job{};
            if (try_build_load_encode_pool_job(data_addr, size, encoding_addr, coefficient, &job)) {
                load_encode_pool_run_parallel(job);
            }
            return;
        }

        // 如果没有正确初始化 EC 表，回退到简单乘法
        if (k_ <= 0 || rows_ != 2 || g_tbls_ == nullptr) {
            uint8_t* data_ptr = reinterpret_cast<uint8_t*>(data_addr);
            uint8_t* encoding_ptr = reinterpret_cast<uint8_t*>(encoding_addr);
            for (size_t i = 0; i < size; ++i) {
                encoding_ptr[i] = data_ptr[i] * (uint8_t)(coefficient & 0xFF);
            }
            return;
        }

        unsigned char *data_ptr = reinterpret_cast<unsigned char*>(data_addr);
        unsigned char *enc_ptr = reinterpret_cast<unsigned char*>(encoding_addr);

        // 从 g_tbls_ 中取出对应 (parity_index, vec_index) 的 32 字节表，
        // 并构造 k=1, rows=1 的调用参数。
        int parity_idx = coefficient;
        if (parity_idx < 0 || parity_idx >= rows_) parity_idx = 0;

        // In load mode, use decode coefficients for decoding operation
        // TODO: Compute decode coefficients from inverse matrix of submatrix
        // For now, use coefficient 1 for all ranks (simplified version)
        if (is_load_mode_ && failed_rank_in_group_ == 2) {
            // Use decode coefficients based on rank_in_group
            if (rank_in_group_ < 2) {
                parity_idx = decode_coefficient_0_;
            } else if (rank_in_group_ < 4) {
                parity_idx = decode_coefficient_1_;
            } else {
                std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: Invalid rank_in_group " << rank_in_group_ << " for load mode (expected 0-3)" << std::endl;
                parity_idx = 1;
            }
            
            // Validation check
            if (decode_coefficient_0_ == 0 || decode_coefficient_1_ == 0) {
                std::cerr << "EC-CHECK: [Rank " << rank_ << "] WARNING: Decode coefficient is 0, this may cause issues" << std::endl;
            }
        }

        // Ensure data_block_index_ in range
        if (data_block_index_ < 0 || data_block_index_ >= k_) {
            std::cerr << "EC-CHECK: invalid data_block_index_=" << data_block_index_ << " for k=" << k_ << std::endl;
            return;
        }

        // Compute offset into g_tbls_: layout from ec_init_tables is for i in rows, j in k -> offset = (i*k + j) * 32
        size_t offset = (size_t)((parity_idx * k_) + data_block_index_) * 32;
        unsigned char *gftbls_ptr = g_tbls_ + offset;

        // Prepare single-element arrays for ec_encode_data
        unsigned char *srcs[1];
        unsigned char *dests[1];
        srcs[0] = data_ptr;
        dests[0] = enc_ptr;

        // Call ec_encode_data with len=size, k=1, rows=1, gftbls_ptr
        ec_encode_data((int)size, 1, 1, gftbls_ptr, srcs, dests);
    }
    
    // Encoding Thread 1 worker
    void encoder_worker_1() {
        
        while (!should_stop_threads_) {
            EncodingTask task;
            
            {
                std::unique_lock<std::mutex> lock(encoding_tasks_1_mutex_);
                encoding_tasks_1_cv_.wait(lock, [this] {
                    return !encoding_tasks_1_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && encoding_tasks_1_.empty()) {
                    break;
                }
                
                task = encoding_tasks_1_.front();
                encoding_tasks_1_.pop();
            }
            
            // Check for sentinel (end signal): all fields are 0
            if (task.data_addr == 0 && task.size == 0 && 
                task.encoding_addr == 0 && task.recv_addr == 0 && task.recv_chunk_size == 0 &&
                task.p2p_own_write_addr == 0 && task.p2p_partner_write_addr == 0) {
                encoding_thread_1_sentinel_received_ = true;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
                    if (encoding_tasks_1_.empty()) {
                encoding_thread_1_completed_ = true;
                        // Submit sentinel to downstream workers
                        {
                            std::lock_guard<std::mutex> send_lock(send_queue_mutex_);
                            send_queue_.push({0, 0});
                        }
                        send_queue_cv_.notify_one();
                        {
                            std::lock_guard<std::mutex> recv_lock(recv_queue_mutex_);
                            recv_queue_.push({0, 0, 0, false, false, false, 0, 0});
                        }
                        recv_queue_cv_.notify_one();
                        {
                            std::lock_guard<std::mutex> xor_lock(xor_queue_mutex_);
                            xor_queue_.push({0, 0, 0, 0, 0, 0, 0, false, false, false, 0, 0});
                        }
                        xor_queue_cv_.notify_one();
                        // Reset sentinel flag and continue (don't exit)
                        encoding_thread_1_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;  // Continue processing remaining tasks
            }
            
            // Check if we need to encode/send data
            bool need_encode = (task.data_addr != 0 && task.encoding_addr != 0);
            if (need_encode && task.local_is_zero_tail) {
                std::memset(reinterpret_cast<void*>(task.encoding_addr), 0, task.size);
            }
            
            if (need_encode) {
                // Step 1: Perform encoding (parity index 0 for thread1)
                if (!task.local_is_zero_tail) {
                    encode_with_coefficient(task.data_addr, task.size, task.encoding_addr, 0);
                }
                
                // Mark data buffer as copied by thread 1
                bool release_data_after_state_update = false;
                {
                    std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
                    auto& state = data_buffer_states_[task.data_addr];
                    state.thread1_copied = true;
                    
                    // If both threads copied, check if we need to delay release for P2P
                    // For odd ranks, data buffer is needed by P2P worker, so delay release
                    if (state.thread2_copied) {
                        // In load mode, rank 1 and rank 2 don't send data, so data buffer should be released immediately
                        // In save mode, even ranks release immediately, odd ranks release in P2P worker
                        if (is_load_mode_ && failed_rank_in_group_ == 2 && (rank_in_group_ == 1 || rank_in_group_ == 2)) {
                            release_data_after_state_update = true;
                        } else if (is_p2p_parity_sender()) {
                            // Save mode: parity senders release data buffer immediately
                            // (P2P worker uses parity buffer instead)
                            release_data_after_state_update = true;
                        }
                        // For data senders, data buffer will be released by P2P worker
                    }
                }
                if (release_data_after_state_update) {
                    release_data_buffer_once(task.data_addr);
                }
                
                // Step 2: Handle based on XOR configuration
                if (xor_config_.thread0_is_receiver && task.parity_addr != 0 && task.recv_addr != 0) {
                    // This thread is receiver: save encoding buffer to pending, wait for recv
                    // The recv_worker will trigger XOR when recv completes
                    if (!task.remote_is_zero_tail) {
                    {
                        std::lock_guard<std::mutex> lock(pending_xor_mutex_);
                        pending_xor_encoding_[task.recv_addr] = task.encoding_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_parity_mutex_);
                        recv_to_parity_[task.recv_addr] = task.parity_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_data_mutex_);
                        recv_to_data_[task.recv_addr] = task.data_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_p2p_mutex_);
                        recv_to_p2p_[task.recv_addr] = {task.p2p_own_write_addr, task.p2p_partner_write_addr};
                    }
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread1 encoding completed, "
                    //           << "pending XOR for recv_addr=" << task.recv_addr 
                    //           << ", encoding_addr=" << task.encoding_addr 
                    //           << ", p2p_own=" << task.p2p_own_write_addr
                    //           << ", p2p_partner=" << task.p2p_partner_write_addr << std::endl;
                    }
                    // Note: encoding buffer will be released by XOR worker after XOR completes
                } else {
                    // This thread is sender: send encoding result immediately
                {
                    std::lock_guard<std::mutex> lock(send_queue_mutex_);
                    if (!task.local_is_zero_tail) {
                        send_queue_.push({task.encoding_addr, task.size});
                    }
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 1 (sender): Pushed task to send_queue, "
                    //           << "encoding_addr=" << task.encoding_addr << ", size=" << task.size 
                    //           << ", queue_size=" << send_queue_.size() << std::endl;
                }
                send_queue_cv_.notify_one();
                // std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 1 (sender): Notified send_queue_cv" << std::endl;
                    
                    // Sender doesn't need parity buffer, release it immediately
                    {
                        std::lock_guard<std::mutex> lock(release_queue_mutex_);
                        if (task.parity_addr != 0) {
                            parity_buffers_to_release_.push(task.parity_addr);
                        }
                        if (task.local_is_zero_tail && task.encoding_addr != 0) {
                            encoding_buffers_to_release_.push(task.encoding_addr);
                        }
                    }
                }
            }
            
            // Check if we need to receive data
            bool need_recv = (task.recv_addr != 0 && task.recv_chunk_size != 0);
            
            if (need_recv && xor_config_.thread0_is_receiver && task.remote_is_zero_tail) {
                std::lock_guard<std::mutex> lock(xor_queue_mutex_);
                xor_queue_.push({
                    task.encoding_addr, 0, task.parity_addr, task.size,
                    task.p2p_own_write_addr, task.p2p_partner_write_addr, task.data_addr,
                    task.local_is_zero_tail, true, task.p2p_data_is_zero_tail,
                    task.p2p_data_size, task.sequence_id
                });
                xor_queue_cv_.notify_one();
                need_recv = false;
            }

            if (need_recv && xor_config_.thread0_is_receiver) {
                // Submit recv task to recv_worker with parity_addr
                {
                    std::lock_guard<std::mutex> lock(recv_queue_mutex_);
                    recv_queue_.push({task.recv_addr, task.recv_chunk_size, task.parity_addr, task.local_is_zero_tail, task.remote_is_zero_tail, task.p2p_data_is_zero_tail, task.p2p_data_size, task.sequence_id});
                }
                recv_queue_cv_.notify_one();
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (encoding_thread_1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
                if (encoding_tasks_1_.empty()) {
                    encoding_thread_1_completed_ = true;
                    // Submit sentinel to downstream workers
                    {
                        std::lock_guard<std::mutex> send_lock(send_queue_mutex_);
                        send_queue_.push({0, 0});
                    }
                    send_queue_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> recv_lock(recv_queue_mutex_);
                        recv_queue_.push({0, 0, 0, false, false, false, 0, 0});
                    }
                    recv_queue_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> xor_lock(xor_queue_mutex_);
                        xor_queue_.push({0, 0, 0, 0, 0, 0, 0, false, false, false, 0, 0});
                    }
                    xor_queue_cv_.notify_one();
                    // Reset sentinel flag and continue (don't exit)
                    encoding_thread_1_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }
    
    // Encoding Thread 2 worker
    void encoder_worker_2() {
        
        while (!should_stop_threads_) {
            EncodingTask task;
            
            {
                std::unique_lock<std::mutex> lock(encoding_tasks_2_mutex_);
                encoding_tasks_2_cv_.wait(lock, [this] {
                    return !encoding_tasks_2_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && encoding_tasks_2_.empty()) {
                    break;
                }
                
                task = encoding_tasks_2_.front();
                encoding_tasks_2_.pop();
            }
            
            // Check for sentinel (end signal): all fields are 0
            if (task.data_addr == 0 && task.size == 0 && 
                task.encoding_addr == 0 && task.recv_addr == 0 && task.recv_chunk_size == 0 &&
                task.p2p_own_write_addr == 0 && task.p2p_partner_write_addr == 0) {
                encoding_thread_2_sentinel_received_ = true;
                // std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
                    if (encoding_tasks_2_.empty()) {
                encoding_thread_2_completed_ = true;
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 queue is empty, marking completed" << std::endl;
                        // Submit sentinel to downstream workers
                        {
                            std::lock_guard<std::mutex> send_lock(send_queue_mutex_);
                            send_queue_.push({0, 0});
                        }
                        send_queue_cv_.notify_one();
                        {
                            std::lock_guard<std::mutex> recv_lock(recv_queue_mutex_);
                            recv_queue_.push({0, 0, 0, false, false, false, 0, 0});
                        }
                        recv_queue_cv_.notify_one();
                        {
                            std::lock_guard<std::mutex> xor_lock(xor_queue_mutex_);
                            xor_queue_.push({0, 0, 0, 0, 0, 0, 0, false, false, false, 0, 0});
                        }
                        xor_queue_cv_.notify_one();
                        // Reset sentinel flag and continue (don't exit)
                        encoding_thread_2_sentinel_received_ = false;
                continue;
                    }
                }
                continue;  // Continue processing remaining tasks
            }
            
            // Check if we need to encode/send data
            bool need_encode = (task.data_addr != 0 && task.encoding_addr != 0);
            if (need_encode && task.local_is_zero_tail) {
                std::memset(reinterpret_cast<void*>(task.encoding_addr), 0, task.size);
            }
            
            if (need_encode) {
                // Step 1: Perform encoding (parity index 1 for thread2)
                if (!task.local_is_zero_tail) {
                    encode_with_coefficient(task.data_addr, task.size, task.encoding_addr, 1);
                }
                
                // Mark data buffer as copied by thread 2
                bool release_data_after_state_update = false;
                {
                    std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
                    auto& state = data_buffer_states_[task.data_addr];
                    state.thread2_copied = true;
                    
                    // If both threads copied, check if we need to delay release for P2P
                    // For odd ranks, data buffer is needed by P2P worker, so delay release
                    if (state.thread1_copied) {
                        // In load mode, rank 1 and rank 2 don't send data, so data buffer should be released immediately
                        // In save mode, even ranks release immediately, odd ranks release in P2P worker
                        if (is_load_mode_ && failed_rank_in_group_ == 2 && (rank_in_group_ == 1 || rank_in_group_ == 2)) {
                            release_data_after_state_update = true;
                        } else if (is_p2p_parity_sender()) {
                            // Save mode: parity senders release data buffer immediately
                            // (P2P worker uses parity buffer instead)
                            release_data_after_state_update = true;
                        }
                        // For data senders, data buffer will be released by P2P worker
                    }
                }
                if (release_data_after_state_update) {
                    release_data_buffer_once(task.data_addr);
                }
                
                // Step 2: Handle based on XOR configuration
                if (xor_config_.thread1_is_receiver && task.parity_addr != 0 && task.recv_addr != 0) {
                    // This thread is receiver: save encoding buffer to pending, wait for recv
                    if (!task.remote_is_zero_tail) {
                    {
                        std::lock_guard<std::mutex> lock(pending_xor_mutex_);
                        pending_xor_encoding_[task.recv_addr] = task.encoding_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_parity_mutex_);
                        recv_to_parity_[task.recv_addr] = task.parity_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_data_mutex_);
                        recv_to_data_[task.recv_addr] = task.data_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_p2p_mutex_);
                        recv_to_p2p_[task.recv_addr] = {task.p2p_own_write_addr, task.p2p_partner_write_addr};
                    }
                    }
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread2 encoding completed, "
                    //           << "pending XOR for recv_addr=" << task.recv_addr 
                    //           << ", encoding_addr=" << task.encoding_addr
                    //           << ", p2p_own=" << task.p2p_own_write_addr
                    //           << ", p2p_partner=" << task.p2p_partner_write_addr << std::endl;
                } else {
                    // This thread is sender: send encoding result immediately
                {
                    std::lock_guard<std::mutex> lock(send_queue_mutex_);
                    if (!task.local_is_zero_tail) {
                        send_queue_.push({task.encoding_addr, task.size});
                    }
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 (sender): Pushed task to send_queue, "
                    //           << "encoding_addr=" << task.encoding_addr << ", size=" << task.size 
                    //           << ", queue_size=" << send_queue_.size() << std::endl;
                }
                send_queue_cv_.notify_one();
                // std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 (sender): Notified send_queue_cv" << std::endl;
                    
                    // Sender doesn't need parity buffer, release it immediately
                    {
                        std::lock_guard<std::mutex> lock(release_queue_mutex_);
                        if (task.parity_addr != 0) {
                            parity_buffers_to_release_.push(task.parity_addr);
                        }
                        if (task.local_is_zero_tail && task.encoding_addr != 0) {
                            encoding_buffers_to_release_.push(task.encoding_addr);
                        }
                    }
                }
            }
            
            // Check if we need to receive data
            bool need_recv = (task.recv_addr != 0 && task.recv_chunk_size != 0);
            
            if (need_recv && xor_config_.thread1_is_receiver && task.remote_is_zero_tail) {
                std::lock_guard<std::mutex> lock(xor_queue_mutex_);
                xor_queue_.push({
                    task.encoding_addr, 0, task.parity_addr, task.size,
                    task.p2p_own_write_addr, task.p2p_partner_write_addr, task.data_addr,
                    task.local_is_zero_tail, true, task.p2p_data_is_zero_tail,
                    task.p2p_data_size, task.sequence_id
                });
                xor_queue_cv_.notify_one();
                need_recv = false;
            }

            if (need_recv && xor_config_.thread1_is_receiver) {
                // Submit recv task to recv_worker with parity_addr
                {
                    std::lock_guard<std::mutex> lock(recv_queue_mutex_);
                    recv_queue_.push({task.recv_addr, task.recv_chunk_size, task.parity_addr, task.local_is_zero_tail, task.remote_is_zero_tail, task.p2p_data_is_zero_tail, task.p2p_data_size, task.sequence_id});
                }
                recv_queue_cv_.notify_one();
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (encoding_thread_2_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
                if (encoding_tasks_2_.empty()) {
                    encoding_thread_2_completed_ = true;
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 queue is empty after processing, marking completed" << std::endl;
                    // Submit sentinel to downstream workers
                    {
                        std::lock_guard<std::mutex> send_lock(send_queue_mutex_);
                        send_queue_.push({0, 0});
                    }
                    send_queue_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> recv_lock(recv_queue_mutex_);
                        recv_queue_.push({0, 0, 0, false, false, false, 0, 0});
                    }
                    recv_queue_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> xor_lock(xor_queue_mutex_);
                        xor_queue_.push({0, 0, 0, 0, 0, 0, 0, false, false, false, 0, 0});
                    }
                    xor_queue_cv_.notify_one();
                    // Reset sentinel flag and continue (don't exit)
                    encoding_thread_2_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }
    
    // Unified send worker (handles all encoded chunks that need to be sent)
    void send_worker() {
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker started" << std::endl;
        
        while (!should_stop_threads_) {
            SendTask task;
            
            {
                std::unique_lock<std::mutex> lock(send_queue_mutex_);
                send_queue_cv_.wait(lock, [this] {
                    return !send_queue_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && send_queue_.empty()) {
                    break;
                }
                
                task = send_queue_.front();
                send_queue_.pop();
            }
            
            // Check for sentinel
            if (task.encoding_addr == 0 && task.size == 0) {
                send_worker_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(send_queue_mutex_);
                    if (send_queue_.empty()) {
                        send_worker_completed_ = true;
                        send_worker_sentinel_received_ = false;
                    }
                }
                continue;
            }
            
            // Send data using RDMA, ASIO, or NCCL
            {
            SaveNetScopeTimer save_net_timer(this);
#ifdef __linux__
            if (use_rdma_ && rdma_xor_send_qp_) {
                try {
                    rdma_send_data_via_qp(rdma_xor_send_qp_, rdma_xor_send_cq_, get_rdma_xor_send_control_sock(),
                        rdma_xor_send_control_mutex_, reinterpret_cast<const uint8_t*>(task.encoding_addr), task.size);
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.encoding_addr);
                } catch (const std::exception& e) {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] RDMA XOR send failed: " << e.what() << std::endl;
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.encoding_addr);
                }
            } else
#endif
            if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_xor_send_connected()) {
                // ASIO send path (synchronous)
                // std::cout << "[EC-CHECK ASIO] XOR_Send: Sending " << task.size << " bytes via ASIO" << std::endl;
                uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(task.encoding_addr);
                uint32_t size_net = htonl(static_cast<uint32_t>(task.size));  // Network byte order
                
                try {
                    // Send message header (size) first
                    boost::asio::write(
                        asio_conn_mgr_.get_xor_send_socket(),
                        boost::asio::buffer(&size_net, sizeof(uint32_t))
                    );
                    
                    // Send data
                    boost::asio::write(
                        asio_conn_mgr_.get_xor_send_socket(),
                        boost::asio::buffer(buffer_ptr, task.size)
                    );
                    
                    // Send completed successfully, release buffer
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.encoding_addr);
                } catch (const boost::system::system_error& e) {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] ASIO send failed: " << e.what() << std::endl;
                    // Release buffer even on error to avoid memory leak
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.encoding_addr);
                }
            }
#ifdef NCCL_AVAILABLE
            else if (!use_asio_) {
                // NCCL send path (fallback)
                const bool DISABLE_SEND_RECV_NCCL = false;
                
                int target_rank = xor_config_.xor_partner_rank;
                bool comm_initialized = nccl_xor_send_initialized_;
                ncclComm_t comm_to_use = nccl_comm_xor_send_;
                
                if (comm_initialized && world_size_ > 1 && !DISABLE_SEND_RECV_NCCL) {
                    int target_rank_in_comm = (rank_ < target_rank) ? 1 : 0;
                    if (target_rank_in_comm < 0) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] Invalid target_rank " << target_rank 
                                  << " for XOR send communicator" << std::endl;
                    } else {
                        ncclGroupStart(); 
                        ncclSend(reinterpret_cast<void*>(task.encoding_addr), task.size, 
                                 ncclUint8, target_rank_in_comm, comm_to_use, 0);
                        ncclGroupEnd();
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker: NCCL GroupEnd completed, starting sync..." << std::endl;
                        
                        sync_nccl_operation("Send worker: NCCL send");
                    }
                } else if (DISABLE_SEND_RECV_NCCL) {
                }
                
                // Release encoding buffer (only after NCCL operation is guaranteed complete)
                {
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.encoding_addr);
                }
            }
#endif
            else {
                // No communication method available
                std::cerr << "EC-CHECK: [Rank " << rank_ 
                          << "] WARNING: No communication method available, releasing buffer" << std::endl;
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                encoding_buffers_to_release_.push(task.encoding_addr);
            }
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (send_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(send_queue_mutex_);
                if (send_queue_.empty()) {
                    send_worker_completed_ = true;
                    send_worker_sentinel_received_ = false;
                }
            }
        }
    }
    
    // Unified recv worker
    void recv_worker() {
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker started" << std::endl;
        
        // Wait for initialization to complete
        if (use_asio_) {
            // For ASIO, wait for connection to be established
            while (!should_stop_threads_ && (!asio_initialized_ || !asio_conn_mgr_.is_xor_recv_connected())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } else {
            // For NCCL, wait for NCCL initialization
            {
                std::unique_lock<std::mutex> lock(nccl_init_mutex_);
                nccl_init_cv_.wait(lock, [this] {
                    return nccl_xor_recv_init_completed_.load();
                });
            }
        }
        
        while (!should_stop_threads_) {
            RecvTask task;
            
            {
                std::unique_lock<std::mutex> lock(recv_queue_mutex_);
                recv_queue_cv_.wait(lock, [this] {
                    return !recv_queue_.empty() || should_stop_threads_ || recv_worker_sentinel_received_.load();
                });
                
                if (should_stop_threads_ && recv_queue_.empty()) {
                    break;
                }
                
                if (recv_worker_sentinel_received_.load() && recv_queue_.empty()) {
                    recv_worker_completed_ = true;
                    recv_worker_sentinel_received_ = false;
                    continue;
                }
                
                task = recv_queue_.front();
                recv_queue_.pop();
            }
            
            if (task.recv_addr == 0 && task.size == 0) {
                recv_worker_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(recv_queue_mutex_);
                    if (recv_queue_.empty()) {
                        recv_worker_completed_ = true;
                        recv_worker_sentinel_received_ = false;
                    }
                }
                continue;
            }
            
            // Receive data using RDMA, ASIO, or NCCL
            {
            SaveNetScopeTimer save_net_timer(this);
#ifdef __linux__
            if (use_rdma_ && rdma_xor_recv_qp_) {
                try {
                    size_t recv_size = rdma_receive_data_via_qp(rdma_xor_recv_qp_, rdma_xor_recv_cq_,
                        get_rdma_xor_recv_control_sock(), rdma_xor_recv_control_mutex_,
                        reinterpret_cast<uint8_t*>(task.recv_addr), task.size,
                        RdmaLoadRecvPollLane::None);
                    if (recv_size != task.size) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] RDMA XOR recv size mismatch: expected "
                                  << task.size << ", got " << recv_size << std::endl;
                        continue;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] RDMA XOR recv failed: " << e.what() << std::endl;
                    continue;
                }
            } else
#endif
            if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_xor_recv_connected()) {
                // ASIO recv path (synchronous)
                // std::cout << "[EC-CHECK ASIO] XOR_Recv: Receiving " << task.size << " bytes via ASIO" << std::endl;
                uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(task.recv_addr);
                uint32_t size_net;
                
                try {
                    // Receive message header (size) first
                    boost::asio::read(
                        asio_conn_mgr_.get_xor_recv_socket(),
                        boost::asio::buffer(&size_net, sizeof(uint32_t))
                    );
                    
                    uint32_t size = ntohl(size_net);
                    if (size != task.size) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ 
                                  << "] Size mismatch: expected " << task.size 
                                  << ", got " << size << std::endl;
                        continue;  // Skip this task
                    }
                    
                    // Receive data
                    boost::asio::read(
                        asio_conn_mgr_.get_xor_recv_socket(),
                        boost::asio::buffer(buffer_ptr, size)
                    );
                    
                    // Receive completed successfully, continue with XOR processing below
                } catch (const boost::system::system_error& e) {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] ASIO recv failed: " << e.what() << std::endl;
                    continue;  // Skip XOR processing on error
                }
            }
#ifdef NCCL_AVAILABLE
            else if (!use_asio_) {
                // NCCL recv path (fallback)
                const bool DISABLE_SEND_RECV_NCCL = false;
                int source_rank = xor_config_.xor_partner_rank;
                bool comm_initialized = nccl_xor_recv_initialized_;
                ncclComm_t comm_to_use = nccl_comm_xor_recv_;
                
                if (comm_initialized && world_size_ > 1 && !DISABLE_SEND_RECV_NCCL) {
                    int source_rank_in_comm = (rank_ < source_rank) ? 0 : 1;
                    if (source_rank_in_comm >= 0) {
                        ncclGroupStart();
                        ncclRecv(reinterpret_cast<void*>(task.recv_addr), task.size,
                                 ncclUint8, source_rank_in_comm, comm_to_use, 0);
                        ncclGroupEnd();
                        sync_nccl_operation("Recv worker: NCCL recv");
                    } else {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] Invalid source rank for recv " << source_rank << std::endl;
                    }
                } else if (DISABLE_SEND_RECV_NCCL) {
                }
            }
#endif
            else {
                std::cerr << "EC-CHECK: [Rank " << rank_ 
                          << "] WARNING: No communication method available for recv" << std::endl;
                continue;  // Skip processing
            }
            }
            
            uintptr_t local_encoding_addr = 0;
            uintptr_t parity_addr = 0;
            uintptr_t data_addr = 0;
            uintptr_t p2p_own_write_addr = 0;
            uintptr_t p2p_partner_write_addr = 0;
            
            {
                std::lock_guard<std::mutex> lock(pending_xor_mutex_);
                auto it = pending_xor_encoding_.find(task.recv_addr);
                if (it != pending_xor_encoding_.end()) {
                    local_encoding_addr = it->second;
                    pending_xor_encoding_.erase(it);
                }
            }
            {
                std::lock_guard<std::mutex> lock(recv_to_parity_mutex_);
                auto it = recv_to_parity_.find(task.recv_addr);
                if (it != recv_to_parity_.end()) {
                    parity_addr = it->second;
                    recv_to_parity_.erase(it);
                }
            }
            {
                std::lock_guard<std::mutex> lock(recv_to_data_mutex_);
                auto it = recv_to_data_.find(task.recv_addr);
                if (it != recv_to_data_.end()) {
                    data_addr = it->second;
                    recv_to_data_.erase(it);
                }
            }
            {
                std::lock_guard<std::mutex> lock(recv_to_p2p_mutex_);
                auto it = recv_to_p2p_.find(task.recv_addr);
                if (it != recv_to_p2p_.end()) {
                    p2p_own_write_addr = it->second.own_write_addr;
                    p2p_partner_write_addr = it->second.partner_write_addr;
                    recv_to_p2p_.erase(it);
                }
            }
            
            if (local_encoding_addr != 0 && parity_addr != 0) {
                {
                    std::lock_guard<std::mutex> lock(xor_queue_mutex_);
                    xor_queue_.push({
                        local_encoding_addr,
                        task.recv_addr,
                        parity_addr,
                        task.size,
                        p2p_own_write_addr,
                        p2p_partner_write_addr,
                        data_addr,
                        task.local_is_zero_tail,
                        task.remote_is_zero_tail,
                        task.p2p_data_is_zero_tail,
                        task.p2p_data_size,
                        task.sequence_id
                    });
                }
                xor_queue_cv_.notify_one();
            }
            
            if (recv_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_queue_mutex_);
                if (recv_queue_.empty()) {
                    recv_worker_completed_ = true;
                    recv_worker_sentinel_received_ = false;
                }
            }
        }
    }
    // Unified XOR worker
    void xor_worker() {
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker started" << std::endl;
        
        auto submit_p2p_sentinel = [this]() {
            {
                std::lock_guard<std::mutex> send_lock(p2p_send_queue_mutex_);
                p2p_send_queue_.push({0, 0, 0, 0, 0, false, false, 0, 0, 0});
            }
            p2p_send_queue_cv_.notify_one();
            {
                std::lock_guard<std::mutex> recv_lock(p2p_recv_queue_mutex_);
                p2p_recv_queue_.push({0, 0, false, false, 0, false, 0, 0});
            }
            p2p_recv_queue_cv_.notify_one();
            // std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker: Sent sentinel to P2P workers" << std::endl;
        };
        
        while (!should_stop_threads_) {
            XORTask task;
            
            {
                std::unique_lock<std::mutex> lock(xor_queue_mutex_);
                xor_queue_cv_.wait(lock, [this] {
                    return !xor_queue_.empty() || should_stop_threads_ || xor_worker_sentinel_received_.load();
                });
                
                if (should_stop_threads_ && xor_queue_.empty()) {
                    break;
                }
                
                if (xor_worker_sentinel_received_.load() && xor_queue_.empty()) {
                    xor_worker_completed_ = true;
                    submit_p2p_sentinel();
                    xor_worker_sentinel_received_ = false;
                    continue;
                }
                
                task = xor_queue_.front();
                xor_queue_.pop();
            }
            
            bool is_sentinel = (task.local_encoding_addr == 0 && task.remote_encoding_addr == 0 &&
                                task.parity_addr == 0 && task.size == 0 &&
                                task.p2p_own_write_addr == 0 && task.p2p_partner_write_addr == 0 &&
                                task.data_addr == 0);
            if (is_sentinel) {
                xor_worker_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(xor_queue_mutex_);
                    if (xor_queue_.empty()) {
                        xor_worker_completed_ = true;
                        submit_p2p_sentinel();
                        xor_worker_sentinel_received_ = false;
                    }
                }
                continue;
            }
            
            unsigned char* dest = reinterpret_cast<unsigned char*>(task.parity_addr);
            if (task.local_is_zero_tail && task.remote_is_zero_tail) {
                std::memset(dest, 0, task.size);
            } else if (task.remote_is_zero_tail) {
                std::memcpy(dest, reinterpret_cast<void*>(task.local_encoding_addr), task.size);
            } else if (task.local_is_zero_tail) {
                std::memcpy(dest, reinterpret_cast<void*>(task.remote_encoding_addr), task.size);
            } else {
                unsigned char* srcs[2];
                srcs[0] = reinterpret_cast<unsigned char*>(task.local_encoding_addr);
                srcs[1] = reinterpret_cast<unsigned char*>(task.remote_encoding_addr);
                // XOR via 16-pthread save pool when available (serialized dispatch).
                // Save encode_s tracks RS encode only; XOR is part of parity assembly.
                if (ec_xor_pool_inited_.load(std::memory_order_acquire)) {
                    std::lock_guard<std::mutex> work_lk(ec_xor_pool_work_mutex_);
                    ec_xor_pool_run_parallel(task.parity_addr,
                                             task.local_encoding_addr,
                                             task.remote_encoding_addr,
                                             task.size);
                } else {
                    void* xor_array[3];
                    xor_array[0] = srcs[0];
                    xor_array[1] = srcs[1];
                    xor_array[2] = dest;
                    xor_gen(3, static_cast<int>(task.size), xor_array);
                }
            }
            
            if (task.p2p_own_write_addr != 0 && task.p2p_partner_write_addr != 0 && task.parity_addr != 0) {
                // Release parity buffer after XOR (single-node recovery loads via this path too)
                if (task.parity_addr != 0) {
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    parity_buffers_to_release_.push(task.parity_addr);
                }
                uintptr_t send_buffer = is_p2p_parity_sender() ? task.parity_addr : task.data_addr;
                size_t p2p_send_size = is_p2p_parity_sender() ? task.size : task.p2p_data_size;
                size_t p2p_send_zero_tail = is_p2p_parity_sender() ? 0 : task.size - p2p_send_size;
                bool skip_p2p_send = (!is_p2p_parity_sender() && p2p_send_size == 0);

                size_t p2p_recv_size = is_p2p_parity_sender() ? task.p2p_data_size : task.size;
                size_t p2p_recv_zero_tail = is_p2p_parity_sender() ? task.size - p2p_recv_size : 0;
                bool skip_p2p_recv = (is_p2p_parity_sender() && p2p_recv_size == 0);

                {
                    std::lock_guard<std::mutex> lock(p2p_send_queue_mutex_);
                    p2p_send_queue_.push({
                        skip_p2p_send ? 0 : send_buffer,
                        task.p2p_own_write_addr,
                        p2p_send_size,
                        task.parity_addr,
                        task.data_addr,
                        false,  // is_load_mode_transfer
                        false,  // is_step6_transfer
                        0,      // load_mode_data_addr (not needed for save mode)
                        p2p_send_zero_tail,
                        task.sequence_id
                    });
                }
                p2p_send_queue_cv_.notify_one();

                {
                    std::lock_guard<std::mutex> lock(p2p_recv_queue_mutex_);
                    p2p_recv_queue_.push({
                        task.p2p_partner_write_addr,
                        p2p_recv_size,
                        false,  // is_load_mode_transfer
                        false,  // is_step6_transfer
                        0,      // data_buffer_addr
                        skip_p2p_recv,
                        p2p_recv_zero_tail,
                        task.sequence_id
                    });
                }
                p2p_recv_queue_cv_.notify_one();
            }
            
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                // Always release local encoding buffer
                encoding_buffers_to_release_.push(task.local_encoding_addr);
                
                // Release remote encoding buffer (received encoding)
                if (task.remote_encoding_addr != 0) {
                    encoding_buffers_to_release_.push(task.remote_encoding_addr);
                }
                
                // Parity buffer release logic
                // Note: For load mode, parity buffer release is already handled above (line 1641-1689)
                // Only handle save mode here to avoid duplicate release
                if (!(is_load_mode_ && failed_rank_in_group_ == 2)) {
                    // Save mode: original logic
                if (!is_p2p_parity_sender() && task.parity_addr != 0) {
                    parity_buffers_to_release_.push(task.parity_addr);
                    }
                }
            }
            
            if (xor_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(xor_queue_mutex_);
                if (xor_queue_.empty()) {
                    xor_worker_completed_ = true;
                    submit_p2p_sentinel();
                    xor_worker_sentinel_received_ = false;
                }
            }
        }
    }
    
    // P2P Send Worker - 专门发送P2P数据（独立线程，类似 send_worker_1）
    void p2p_send_worker() {
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker started" << std::endl;
        
        // NCCL is already initialized in main thread, no need to initialize here
        size_t next_save_sequence_id = 0;
        std::map<size_t, P2PSendTask> pending_save_tasks;
        
        while (!should_stop_threads_) {
            P2PSendTask task{};
            bool has_task = false;
            
            {
                std::unique_lock<std::mutex> lock(p2p_send_queue_mutex_);
                auto ready_pending = [&]() {
                    return pending_save_tasks.find(next_save_sequence_id) != pending_save_tasks.end();
                };
                p2p_send_queue_cv_.wait(lock, [this, &ready_pending] {
                    return !p2p_send_queue_.empty() || should_stop_threads_ || ready_pending();
                });
                
                if (should_stop_threads_ && p2p_send_queue_.empty() && !ready_pending()) {
                    break;
                }

                while (!ready_pending() && !p2p_send_queue_.empty()) {
                    P2PSendTask candidate = p2p_send_queue_.front();
                    p2p_send_queue_.pop();
                    bool is_sentinel = (candidate.send_buffer_addr == 0 &&
                                        candidate.p2p_own_write_addr == 0 && candidate.size == 0);
                    if (is_sentinel || candidate.is_load_mode_transfer ||
                        candidate.sequence_id == next_save_sequence_id) {
                        task = candidate;
                        has_task = true;
                        break;
                    }
                    pending_save_tasks[candidate.sequence_id] = candidate;
                }

                if (task.send_buffer_addr == 0 && task.p2p_own_write_addr == 0 &&
                    task.size == 0 && !pending_save_tasks.empty() && ready_pending()) {
                    auto it = pending_save_tasks.find(next_save_sequence_id);
                    task = it->second;
                    pending_save_tasks.erase(it);
                    has_task = true;
                }
                if (!has_task) {
                    continue;
                }
                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: Popped task, "
                //           << "send_buffer_addr=" << task.send_buffer_addr
                //           << ", p2p_own_write_addr=" << task.p2p_own_write_addr
                //           << ", size=" << task.size
                //           << ", queue_size_after_pop=" << p2p_send_queue_.size() << std::endl;
            }
            
            // Check for sentinel
            if (task.send_buffer_addr == 0 && task.p2p_own_write_addr == 0 && task.size == 0) {
                p2p_send_worker_sentinel_received_ = true;
                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(p2p_send_queue_mutex_);
                    if (p2p_send_queue_.empty()) {
                        p2p_send_worker_completed_ = true;
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker queue is empty, marking completed" << std::endl;
                        // Reset sentinel flag and continue (don't exit)
                        p2p_send_worker_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;
            }
            
            // Step 1: Copy own data/parity to own_buffer (save path only)
            if (task.p2p_own_write_addr != 0 && task.send_buffer_addr == 0 &&
                (task.size > 0 || task.zero_fill_tail_size > 0)) {
                std::memset(reinterpret_cast<void*>(task.p2p_own_write_addr), 0,
                            task.size + task.zero_fill_tail_size);
                if (!task.is_load_mode_transfer && !is_p2p_parity_sender() && task.data_addr != 0) {
                    if (release_data_buffer_once(task.data_addr)) {
                        task.data_addr = 0;
                    }
                }
            } else if (task.p2p_own_write_addr != 0 && task.send_buffer_addr != 0 && task.size > 0) {
#ifdef __linux__
                if (use_rdma_ && (!rdma_range_registered(task.p2p_own_write_addr, task.size) ||
                                  !rdma_range_registered(task.send_buffer_addr, task.size))) {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: Invalid P2P send memcpy range "
                              << "(dst=0x" << std::hex << task.p2p_own_write_addr
                              << ", src=0x" << task.send_buffer_addr << std::dec
                              << ", size=" << task.size << ")" << std::endl;
                    throw std::runtime_error("EC-CHECK: Invalid P2P send memcpy range");
                }
#endif
                std::memcpy(reinterpret_cast<void*>(task.p2p_own_write_addr),
                           reinterpret_cast<void*>(task.send_buffer_addr),
                           task.size);
                if (task.zero_fill_tail_size > 0) {
                    std::memset(reinterpret_cast<void*>(task.p2p_own_write_addr + task.size),
                                0, task.zero_fill_tail_size);
                }
                if (!task.is_load_mode_transfer && !is_p2p_parity_sender() && task.data_addr != 0) {
                    if (release_data_buffer_once(task.data_addr)) {
                        task.data_addr = 0;
                    }
                }
                // if (rank_ % 2 == 0) {
                //     std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Copied own parity to own_buffer at "
                //               << task.p2p_own_write_addr << std::endl;
                // } else {
                //     std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Copied own data to own_buffer at "
                //               << task.p2p_own_write_addr << std::endl;
                // }
            } else if (task.p2p_own_write_addr == 0 && task.send_buffer_addr != 0 && task.size > 0) {
                // Load path: direct send from provided buffer (typically mmap)
                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Using provided buffer directly (no local copy)"
                //           << std::endl;
                continue;
            }
            
            // Step 2: Send data using RDMA, ASIO, or NCCL.  In save mode we send
            // from own_buffer after the local copy above, so the reusable data/parity
            // pools can be released without racing the network operation.
            uintptr_t network_send_addr = task.send_buffer_addr;
            if (!task.is_load_mode_transfer && task.p2p_own_write_addr != 0) {
                network_send_addr = task.p2p_own_write_addr;
            }
            if (p2p_partner_rank_ >= 0 && task.size > 0 && network_send_addr != 0) {
            SaveNetScopeTimer save_net_timer(this);
#ifdef __linux__
            if (use_rdma_ && rdma_p2p_send_qp_) {
                    try {
                        rdma_send_data_via_qp(rdma_p2p_send_qp_, rdma_p2p_send_cq_, get_rdma_p2p_send_control_sock(),
                            rdma_p2p_send_control_mutex_, reinterpret_cast<const uint8_t*>(network_send_addr), task.size);
                    } catch (const std::exception& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] RDMA P2P send failed: " << e.what() << std::endl;
                    }
                } else
#endif
                if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_p2p_send_connected()) {
                    // ASIO send path (synchronous)
                    // std::cout << "[EC-CHECK ASIO] P2P_Send: Sending " << task.size << " bytes (" << send_label << ") via ASIO" << std::endl;
                    uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(network_send_addr);
                    uint32_t size_net = htonl(static_cast<uint32_t>(task.size));  // Network byte order
                    
                    try {
                        // Send message header (size) first
                        boost::asio::write(
                            asio_conn_mgr_.get_p2p_send_socket(),
                            boost::asio::buffer(&size_net, sizeof(uint32_t))
                        );
                        
                        // Send data
                        boost::asio::write(
                            asio_conn_mgr_.get_p2p_send_socket(),
                            boost::asio::buffer(buffer_ptr, task.size)
                        );
                        
                        // Send completed successfully
                        // if (task.is_load_mode_transfer) {
                        //     std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P send: Sent partner_file chunk to Rank " 
                        //               << p2p_partner_rank_ << " (size=" << task.size << ")" << std::endl;
                        // } else {
                        //     // Save mode or Step6: original log
                        //     if (rank_ % 2 == 0) {
                        //         std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Sent parity to Rank " 
                        //                   << p2p_partner_rank_ << std::endl;
                        //     } else {
                        //         std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Sent data to Rank " 
                        //                   << p2p_partner_rank_ << std::endl;
                        //     }
                        // }
                    } catch (const boost::system::system_error& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ 
                                  << "] P2P ASIO send failed: " << e.what() << std::endl;
                    }
                }
#ifdef NCCL_AVAILABLE
                else if (!use_asio_) {
                    // NCCL send path (fallback)
                    const bool DISABLE_P2P_NCCL = false;  // Set to true to disable P2P NCCL operations
                    
                    if (nccl_p2p_send_initialized_ && world_size_ > 1 && !DISABLE_P2P_NCCL) {
                        // Verify communicator is valid
                        if (nccl_comm_p2p_send_ == nullptr) {
                            std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: nccl_comm_p2p_send_ is NULL but nccl_p2p_send_initialized_ is true!" << std::endl;
                            std::cerr.flush();
                        } else {
                            // Map global p2p_partner_rank to communicator-internal rank (0 or 1)
                            int partner_rank_in_comm = (rank_ < p2p_partner_rank_) ? 1 : 0;
                            // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: partner_rank_in_comm=" 
                            //           << partner_rank_in_comm << " (from global rank " << p2p_partner_rank_ << ")" << std::endl;
                            if (partner_rank_in_comm < 0 || partner_rank_in_comm >= 2) {
                                // std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: Invalid partner_rank_in_comm=" 
                                //           << partner_rank_in_comm << " (must be 0 or 1 for 2-rank communicator)" << std::endl;
                                // std::cerr << "EC-CHECK: [Rank " << rank_ << "] p2p_partner_rank_=" << p2p_partner_rank_ << std::endl;
                                std::cerr.flush();
                            } else {
                                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: Starting NCCL send ("
                                //           << send_label << "), size=" << task.size
                                //           << ", partner_rank_in_comm=" << partner_rank_in_comm << std::endl;
                                
                                ncclGroupStart();
                                ncclSend(reinterpret_cast<void*>(task.send_buffer_addr), task.size,
                                         ncclUint8, partner_rank_in_comm, nccl_comm_p2p_send_, 0);
                                ncclGroupEnd();
                                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: NCCL GroupEnd completed, starting sync..." << std::endl;
                                
                                // Synchronize NCCL operation before releasing buffer
                                sync_nccl_operation("P2P send worker: NCCL send");
                                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: NCCL sync completed" << std::endl;
                                
                                // if (task.is_load_mode_transfer) {
                                //     std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P send: Sent partner_file chunk to Rank " 
                                //               << p2p_partner_rank_ << " (size=" << task.size << ")" << std::endl;
                                // } else {
                                //     // Save mode or Step6: original log
                                //     if (rank_ % 2 == 0) {
                                //         std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Sent parity to Rank " 
                                //                   << p2p_partner_rank_ << std::endl;
                                //     } else {
                                //         std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Sent data to Rank " 
                                //                   << p2p_partner_rank_ << std::endl;
                                //     }
                                // }
                            }
                        }
                    } else if (DISABLE_P2P_NCCL) {
                    } else {
                    }
                }
#endif
                else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] WARNING: No communication method available for P2P send" << std::endl;
                }
            } else {
            }
            
            // Step 3: For load mode Step2, submit encoding task after P2P send completes
            if (task.is_load_mode_transfer && !task.is_step6_transfer && task.load_mode_data_addr != 0) {
                // Load mode Step2: 发送完成后，查找并提交encoding任务到 load worker
                LoadEncodingTask load_task;
                bool found_task = false;
                {
                    std::lock_guard<std::mutex> lock(pending_load_encoding_tasks_mutex_);
                    auto it = pending_load_encoding_tasks_.find(task.load_mode_data_addr);
                    if (it != pending_load_encoding_tasks_.end()) {
                        load_task = it->second;
                        pending_load_encoding_tasks_.erase(it);
                        found_task = true;
                    }
                }
                
                if (found_task && load_task.data_addr != 0) {
                    // 提交到 load encoding worker
                    submit_load_encoding_task(load_task);
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] WARNING: No pending load encoding task found for data_addr=" 
                              << task.load_mode_data_addr << std::endl;
                }
            }
            
            // 新增：处理 Step6 send（rank3 发送 d3 给 rank2）
            if (task.is_load_mode_transfer && task.is_step6_transfer) {
                // Step6: rank3 发送 d3 (parity_buffer) 给 rank2
                // 发送完成后释放 parity buffer
                {
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    if (task.parity_addr != 0) {
                        parity_buffers_to_release_.push(task.parity_addr);
                    }
                }
            }
            
            // Step 4: Release buffers after send is guaranteed complete
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                
                if (task.is_load_mode_transfer && !task.is_step6_transfer) {
                    // Load mode Step2: release send_buffer (this is temporary buffer from mmap read)
                    if (task.send_buffer_addr != 0) {
                        data_buffers_to_release_.push(task.send_buffer_addr);
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P send: Released send_buffer " 
                        //           << task.send_buffer_addr << std::endl;
                    }
                } else {
                    // Save mode or Step6: original logic
                    // For even ranks, release parity buffer after send completes (parity was sent)
                    // For odd ranks, parity buffer is not sent, so it will be released in XOR worker
                    if (is_p2p_parity_sender() && task.parity_addr != 0) {
                        parity_buffers_to_release_.push(task.parity_addr);
                    }
                    // For data senders (higher rank in P2P pair), release data buffer
                    if (!is_p2p_parity_sender() && task.data_addr != 0) {
                        release_data_buffer_once(task.data_addr);
                    }
                }
            }
            
            if (!task.is_load_mode_transfer) {
                ++next_save_sequence_id;
                p2p_send_queue_cv_.notify_one();
            }

            // After processing task, check if sentinel was received and queue is empty
            if (p2p_send_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(p2p_send_queue_mutex_);
                if (p2p_send_queue_.empty()) {
                    p2p_send_worker_completed_ = true;
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker queue is empty after processing, marking completed" << std::endl;
                    // Reset sentinel flag and continue (don't exit)
                    p2p_send_worker_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }
    
    // P2P Recv Worker - 专门接收P2P数据（独立线程，类似 recv_worker_1）
    void p2p_recv_worker() {
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker started" << std::endl;
        
        // NCCL is already initialized in main thread, no need to initialize here
        size_t next_save_sequence_id = 0;
        std::map<size_t, P2PRecvTask> pending_save_tasks;
        
        while (!should_stop_threads_) {
            P2PRecvTask task{};
            bool has_task = false;
            
            {
                std::unique_lock<std::mutex> lock(p2p_recv_queue_mutex_);
                auto ready_pending = [&]() {
                    return pending_save_tasks.find(next_save_sequence_id) != pending_save_tasks.end();
                };
                p2p_recv_queue_cv_.wait(lock, [this, &ready_pending] {
                    return !p2p_recv_queue_.empty() || should_stop_threads_ || ready_pending();
                });
                
                if (should_stop_threads_ && p2p_recv_queue_.empty() && !ready_pending()) {
                    break;
                }

                while (!ready_pending() && !p2p_recv_queue_.empty()) {
                    P2PRecvTask candidate = p2p_recv_queue_.front();
                    p2p_recv_queue_.pop();
                    bool is_sentinel = (candidate.recv_buffer_addr == 0 && candidate.size == 0);
                    if (is_sentinel || candidate.is_load_mode_transfer ||
                        candidate.sequence_id == next_save_sequence_id) {
                        task = candidate;
                        has_task = true;
                        break;
                    }
                    pending_save_tasks[candidate.sequence_id] = candidate;
                }

                if (task.recv_buffer_addr == 0 && task.size == 0 &&
                    !pending_save_tasks.empty() && ready_pending()) {
                    auto it = pending_save_tasks.find(next_save_sequence_id);
                    task = it->second;
                    pending_save_tasks.erase(it);
                    has_task = true;
                }
                if (!has_task) {
                    continue;
                }
                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: Popped task, "
                //           << "recv_buffer_addr=" << task.recv_buffer_addr
                //           << ", size=" << task.size
                //           << ", queue_size_after_pop=" << p2p_recv_queue_.size() << std::endl;
            }
            
            // Check for sentinel
            if (task.recv_buffer_addr == 0 && task.size == 0) {
                p2p_recv_worker_sentinel_received_ = true;
                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(p2p_recv_queue_mutex_);
                    if (p2p_recv_queue_.empty()) {
                        p2p_recv_worker_completed_ = true;
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker queue is empty, marking completed" << std::endl;
                        // Reset sentinel flag and continue (don't exit)
                        p2p_recv_worker_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;
            }
            
            // Receive data using RDMA, ASIO, or NCCL
            bool task_processed = false;  // Track whether task was successfully processed
            if (task.skip_network && task.recv_buffer_addr != 0 &&
                (task.size > 0 || task.zero_fill_tail_size > 0)) {
                std::memset(reinterpret_cast<void*>(task.recv_buffer_addr), 0,
                            task.size + task.zero_fill_tail_size);
                task_processed = true;
            } else if (p2p_partner_rank_ >= 0 && task.size > 0 && task.recv_buffer_addr != 0) {
#ifdef __linux__
                if (use_rdma_ && rdma_p2p_recv_qp_) {
                    // std::cout << "[EC-CHECK RDMA] Save_P2P_Recv: Receiving " << task.size << " bytes via RDMA" << std::endl;
                    try {
                        size_t recv_size = rdma_receive_data_via_qp(rdma_p2p_recv_qp_, rdma_p2p_recv_cq_,
                            get_rdma_p2p_recv_control_sock(), rdma_p2p_recv_control_mutex_,
                            reinterpret_cast<uint8_t*>(task.recv_buffer_addr), task.size,
                            RdmaLoadRecvPollLane::None);
                        if (recv_size == task.size) task_processed = true;
                        else {
                            std::cerr << "EC-CHECK: [Rank " << rank_ << "] P2P RDMA recv size mismatch: expected "
                                      << task.size << ", got " << recv_size
                                      << ", rank_in_group=" << rank_in_group_
                                      << ", partner=" << p2p_partner_rank_
                                      << ", seq=" << task.sequence_id
                                      << ", expect_role=" << (is_p2p_parity_sender() ? "data" : "parity")
                                      << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] RDMA P2P recv failed: " << e.what() << std::endl;
                    }
                    if (!task_processed) continue;
                } else
#endif
                if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_p2p_recv_connected()) {
                    // ASIO recv path (synchronous)
                    // std::cout << "[EC-CHECK ASIO] P2P_Recv: Receiving " << task.size << " bytes (" << recv_label << ") via ASIO" << std::endl;
                    uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(task.recv_buffer_addr);
                    uint32_t size_net;
                    
                    try {
                        // Receive message header (size) first
                        boost::asio::read(
                            asio_conn_mgr_.get_p2p_recv_socket(),
                            boost::asio::buffer(&size_net, sizeof(uint32_t))
                        );
                        
                        uint32_t size = ntohl(size_net);
                        if (size != task.size) {
                            std::cerr << "EC-CHECK: [Rank " << rank_ 
                                      << "] P2P size mismatch: expected " << task.size 
                                      << ", got " << size << std::endl;
                            continue;  // Skip this task
                        }
                        
                        // Receive data
                        boost::asio::read(
                            asio_conn_mgr_.get_p2p_recv_socket(),
                            boost::asio::buffer(buffer_ptr, size)
                        );
                        
                        // Receive completed successfully
                        // if (rank_ % 2 == 0) {
                        //     std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv: Received data from Rank " << p2p_partner_rank_ << std::endl;
                        // } else {
                        //     std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv: Received parity from Rank " << p2p_partner_rank_ << std::endl;
                        // }
                        task_processed = true;
                    } catch (const boost::system::system_error& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ 
                                  << "] P2P ASIO recv failed: " << e.what() << std::endl;
                        continue;  // Skip processing on error
                    }
                }
#ifdef NCCL_AVAILABLE
                else if (!use_asio_) {
                    bool task_processed_nccl = false;  // Local flag for NCCL path
                    // NCCL recv path (fallback)
                    const bool DISABLE_P2P_NCCL = false;  // Set to true to disable P2P NCCL operations
                    
                    if (nccl_p2p_recv_initialized_ && world_size_ > 1 && !DISABLE_P2P_NCCL) {
                        // Verify communicator is valid
                        if (nccl_comm_p2p_recv_ == nullptr) {
                            std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: nccl_comm_p2p_recv_ is NULL but nccl_p2p_recv_initialized_ is true!" << std::endl;
                            std::cerr.flush();
                        } else {
                            // Map global p2p_partner_rank to communicator-internal rank (0 or 1)
                            int partner_rank_in_comm = (rank_ < p2p_partner_rank_) ? 0 : 1;
                            // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: partner_rank_in_comm=" 
                            //           << partner_rank_in_comm << " (from global rank " << p2p_partner_rank_ << ")" << std::endl;
                            if (partner_rank_in_comm < 0 || partner_rank_in_comm >= 2) {
                                std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: Invalid partner_rank_in_comm=" 
                                          << partner_rank_in_comm << " (must be 0 or 1 for 2-rank communicator)" << std::endl;
                                std::cerr << "EC-CHECK: [Rank " << rank_ << "] p2p_partner_rank_=" << p2p_partner_rank_ << std::endl;
                                std::cerr.flush();
                            } else {
                                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: Starting NCCL recv ("
                                //           << recv_label << "), size=" << task.size
                                //           << ", partner_rank_in_comm=" << partner_rank_in_comm << std::endl;
                                
                                ncclGroupStart();
                                ncclRecv(reinterpret_cast<void*>(task.recv_buffer_addr), task.size,
                                         ncclUint8, partner_rank_in_comm, nccl_comm_p2p_recv_, 0);
                                ncclGroupEnd();
                                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: NCCL GroupEnd completed, starting sync..." << std::endl;
                                
                                // Synchronize NCCL operation
                                sync_nccl_operation("P2P recv worker: NCCL recv");
                                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: NCCL sync completed" << std::endl;
                                
                                // Log will be handled in the common section after recv completes
                                task_processed_nccl = true;
                            }
                        }
                    } else if (DISABLE_P2P_NCCL) {
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: NCCL DISABLED for debugging, task processed (no-op)" << std::endl;
                        // Even when NCCL is disabled, the task is considered processed
                        // This ensures sentinel check logic works correctly
                        task_processed_nccl = true;
                    } else {
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: Skipping NCCL (nccl_p2p_recv_initialized_=" 
                        //           << (nccl_p2p_recv_initialized_ ? "true" : "false") << ", world_size_=" << world_size_ << ")" << std::endl;
                        task_processed_nccl = true;
                    }
                    task_processed = task_processed_nccl;
                }
#endif
                else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] WARNING: No communication method available for P2P recv" << std::endl;
                    task_processed = true;  // Mark as processed to avoid blocking
                }
            } else {
                // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: Skipping recv (p2p_partner_rank_=" 
                //           << p2p_partner_rank_ << ", task.size=" << task.size << "), task processed" << std::endl;
                task_processed = true;
            }
            
            // Load mode Step2: handle received data and submit encoding task
            if (task_processed && task.is_load_mode_transfer && task.data_buffer_addr != 0) {
                // If recv_buffer_addr != data_buffer_addr, need to copy data
                if (task.recv_buffer_addr != task.data_buffer_addr) {
#ifdef __linux__
                    if (use_rdma_ && (!rdma_range_registered(task.data_buffer_addr, task.size) ||
                                      !rdma_range_registered(task.recv_buffer_addr, task.size))) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: Invalid load P2P recv memcpy range "
                                  << "(dst=0x" << std::hex << task.data_buffer_addr
                                  << ", src=0x" << task.recv_buffer_addr << std::dec
                                  << ", size=" << task.size << ")" << std::endl;
                        throw std::runtime_error("EC-CHECK: Invalid load P2P recv memcpy range");
                    }
#endif
                    std::memcpy(
                        reinterpret_cast<void*>(task.data_buffer_addr),
                        reinterpret_cast<void*>(task.recv_buffer_addr),
                        task.size
                    );
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P recv: Copied received data to data_buffer at "
                            //   << task.data_buffer_addr << " (size=" << task.size << ")" << std::endl;
                } else {
                    // recv_buffer_addr == data_buffer_addr, data already in correct position, no copy needed
                }
                
                // Step 2接收完成，查找并提交encoding任务到 load worker
                if (!task.is_step6_transfer) {
                    LoadEncodingTask load_task;
                bool found_task = false;
                {
                        std::lock_guard<std::mutex> lock(pending_load_encoding_tasks_mutex_);
                        auto it = pending_load_encoding_tasks_.find(task.data_buffer_addr);
                        if (it != pending_load_encoding_tasks_.end()) {
                            load_task = it->second;
                            pending_load_encoding_tasks_.erase(it);
                        found_task = true;
                    }
                }
                
                    if (found_task && load_task.data_addr != 0) {
                        // 提交到 load encoding worker
                        submit_load_encoding_task(load_task);
                } else {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] WARNING: No pending load encoding task found for data_buffer=" 
                              << task.data_buffer_addr << std::endl;
                }
                }
            }
            
            // 新增：处理 Step6 recv（rank2 接收 d3）
            if (task.is_load_mode_transfer && task.is_step6_transfer) {
                // Step6: rank2 接收 d3 到 partner_buffer
                // partner_buffer 由 Python 管理，这里不需要释放
            } 
            // else if (task_processed && !task.is_load_mode_transfer) {
                // Save mode or Step6: original log
                // if (rank_ % 2 == 0) {
                //     std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv: Received data from Rank " 
                //               << p2p_partner_rank_ << std::endl;
                // } else {
                //     std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv: Received parity from Rank " 
                //               << p2p_partner_rank_ << std::endl;
                // }
                
            // }
            
            // Note: No buffer release needed here - recv buffer (p2p_partner_write_addr) is managed by Python
            
            if (!task.is_load_mode_transfer) {
                ++next_save_sequence_id;
                p2p_recv_queue_cv_.notify_one();
            }

            // After processing task, check if sentinel was received and queue is empty
            if (p2p_recv_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(p2p_recv_queue_mutex_);
                if (p2p_recv_queue_.empty()) {
                    p2p_recv_worker_completed_ = true;
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker queue is empty after processing, marking completed" << std::endl;
                    // Reset sentinel flag and continue (don't exit)
                    p2p_recv_worker_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }

    void start_pipeline() {
        // Init 16-pthread pools for save-phase encode and XOR
        ec_rs_encode_pool_init();
        ec_xor_pool_init();

        // Start encoding threads
        encoder_thread_1_ = std::thread(&ECCHECKNative::encoder_worker_1, this);
        encoder_thread_2_ = std::thread(&ECCHECKNative::encoder_worker_2, this);

        // Start unified send/recv/xor workers
        send_worker_ = std::thread(&ECCHECKNative::send_worker, this);
        recv_worker_ = std::thread(&ECCHECKNative::recv_worker, this);
        xor_worker_ = std::thread(&ECCHECKNative::xor_worker, this);

        // Start P2P workers (split into send and recv)
        p2p_send_worker_ = std::thread(&ECCHECKNative::p2p_send_worker, this);
        p2p_recv_worker_ = std::thread(&ECCHECKNative::p2p_recv_worker, this);

    }

public:
    ECCHECKNative(int rank, int world_size, int paired_rank,
                  const std::vector<uint8_t>& nccl_id_xor_send,
                  const std::vector<uint8_t>& nccl_id_xor_recv,
                  const std::vector<uint8_t>& nccl_id_p2p_send,
                  const std::vector<uint8_t>& nccl_id_p2p_recv,
                  int rank_in_group = -1,
                  int p2p_partner_rank = -1)
        : rank_(rank), world_size_(world_size), paired_rank_(paired_rank),
          rank_in_group_(rank_in_group >= 0 ? rank_in_group : rank),
          p2p_partner_rank_(p2p_partner_rank),
          failed_rank_in_group_(-1),
          is_two_failures_load_mode_(false),
          encoding_thread_1_completed_(false), encoding_thread_2_completed_(false),
          send_worker_completed_(false), recv_worker_completed_(false),
          xor_worker_completed_(false), p2p_send_worker_completed_(false), p2p_recv_worker_completed_(false),
          encoding_thread_1_sentinel_received_(false), encoding_thread_2_sentinel_received_(false),
          send_worker_sentinel_received_(false), recv_worker_sentinel_received_(false),
          xor_worker_sentinel_received_(false), p2p_send_worker_sentinel_received_(false), p2p_recv_worker_sentinel_received_(false),
          load_encoding_completed_(false), load_encoding_sentinel_received_(false),
          load_send_worker_completed_(false), load_send_worker_sentinel_received_(false),
          load_recv_worker_completed_(false), load_recv_worker_sentinel_received_(false),
          load_xor_worker_completed_(false), load_xor_worker_sentinel_received_(false),
          load_p2p_send_worker_completed_(false), load_p2p_send_worker_sentinel_received_(false),
          load_p2p_recv_worker_completed_(false), load_p2p_recv_worker_sentinel_received_(false),
          load_step6_p2p_send_worker_completed_(false), load_step6_p2p_send_worker_sentinel_received_(false),
          load_step6_p2p_recv_worker_completed_(false), load_step6_p2p_recv_worker_sentinel_received_(false),
          should_stop_threads_(false),
          nccl_xor_send_initialized_(false), nccl_xor_recv_initialized_(false),
          nccl_p2p_send_initialized_(false), nccl_p2p_recv_initialized_(false),
          nccl_xor_send_init_completed_(false), nccl_xor_recv_init_completed_(false),
          nccl_p2p_send_init_completed_(false), nccl_p2p_recv_init_completed_(false),
          k_(0), rows_(0), data_block_index_(0), a_mat_(nullptr), g_tbls_(nullptr),
          decode_coefficient_0_(1), decode_coefficient_1_(1),  // Initialize to 1 for simplified version
          is_load_mode_(false), failed_rank_(-1),
          asio_initialized_(false), use_asio_(false) {

        
        // Store four independent NCCL IDs (Python already guarantees send/recv pairing)
        nccl_id_xor_send_ = nccl_id_xor_send;
        nccl_id_xor_recv_ = nccl_id_xor_recv;
        nccl_id_p2p_send_ = nccl_id_p2p_send;
        nccl_id_p2p_recv_ = nccl_id_p2p_recv;
        
        // Build XOR configuration
        build_xor_config();
        
        // Build P2P configuration
        build_p2p_config();

        // Initialize EC params: k=2 per group, rows=2, data_block_index = rank_in_group / 2
        rows_ = 2;
        k_ = 2;
        data_block_index_ = rank_in_group_ / 2;

        if (k_ > 0) {
            int m = k_ + rows_;
            // allocate matrix a (k * m)
            a_mat_ = (unsigned char*)malloc((size_t)k_ * (size_t)m);
            if (a_mat_ == nullptr) {
                std::cerr << "EC-CHECK: failed to allocate a_mat_" << std::endl;
            } else {
                // generate RS matrix
                gf_gen_rs_matrix(a_mat_, m, k_);

                // allocate g_tbls_: 32 * k * rows
                size_t gtbls_size = 32 * (size_t)k_ * (size_t)rows_;
                void *tmp = nullptr;
                if (posix_memalign(&tmp, 32, gtbls_size) != 0) tmp = nullptr;
                if (tmp == nullptr) tmp = malloc(gtbls_size);
                g_tbls_ = reinterpret_cast<unsigned char*>(tmp);
                if (g_tbls_ == nullptr) {
                    std::cerr << "EC-CHECK: failed to allocate g_tbls_" << std::endl;
                } else {
                    // initialize tables using isa-l
                    ec_init_tables(k_, rows_, a_mat_, g_tbls_);
                }
            }
        } else {
        }

        // Initialize NCCL communicators in main thread (before starting worker threads)
        // This ensures all ranks call ncclCommInitRank simultaneously (synchronized by Python barrier)
        init_nccl_xor_send();
        init_nccl_xor_recv();
        init_nccl_p2p_send();
        init_nccl_p2p_recv();

        // Now start worker threads (NCCL is already initialized)
        start_pipeline();

    }
    
    ~ECCHECKNative() {
        stop_pipeline();
        if (use_asio_) {
#ifdef __linux__
            if (use_rdma_) cleanup_rdma_resources();
#endif
            asio_conn_mgr_.cleanup();
            // No need to stop IO context thread since we're using synchronous I/O
        } else {
            cleanup_nccl();
        }
        if (a_mat_) { free(a_mat_); a_mat_ = nullptr; }
        if (g_tbls_) { free(g_tbls_); g_tbls_ = nullptr; }
    }
    
    // ASIO/RDMA constructor (accepts IP/Port parameters and RDMA flag)
    ECCHECKNative(int rank, int world_size, int paired_rank,
                  const std::string& xor_partner_ip, uint16_t xor_send_port,
                  const std::string& xor_listen_ip, uint16_t xor_recv_port,
                  const std::string& p2p_partner_ip, uint16_t p2p_send_port,
                  const std::string& p2p_listen_ip, uint16_t p2p_recv_port,
                  const std::string& step6_p2p_partner_ip, uint16_t step6_p2p_send_port,
                  const std::string& step6_p2p_listen_ip, uint16_t step6_p2p_recv_port,
                  bool use_rdma = false, int rank_in_group = -1,
                  int p2p_partner_rank = -1)
        : rank_(rank), world_size_(world_size), paired_rank_(paired_rank),
          rank_in_group_(rank_in_group >= 0 ? rank_in_group : rank),
          p2p_partner_rank_(p2p_partner_rank),
          failed_rank_in_group_(-1),
          is_two_failures_load_mode_(false),
          encoding_thread_1_completed_(false), encoding_thread_2_completed_(false),
          send_worker_completed_(false), recv_worker_completed_(false),
          xor_worker_completed_(false), p2p_send_worker_completed_(false), p2p_recv_worker_completed_(false),
          encoding_thread_1_sentinel_received_(false), encoding_thread_2_sentinel_received_(false),
          send_worker_sentinel_received_(false), recv_worker_sentinel_received_(false),
          xor_worker_sentinel_received_(false), p2p_send_worker_sentinel_received_(false), p2p_recv_worker_sentinel_received_(false),
          load_encoding_completed_(false), load_encoding_sentinel_received_(false),
          load_send_worker_completed_(false), load_send_worker_sentinel_received_(false),
          load_recv_worker_completed_(false), load_recv_worker_sentinel_received_(false),
          load_xor_worker_completed_(false), load_xor_worker_sentinel_received_(false),
          load_p2p_send_worker_completed_(false), load_p2p_send_worker_sentinel_received_(false),
          load_p2p_recv_worker_completed_(false), load_p2p_recv_worker_sentinel_received_(false),
          load_step6_p2p_send_worker_completed_(false), load_step6_p2p_send_worker_sentinel_received_(false),
          load_step6_p2p_recv_worker_completed_(false), load_step6_p2p_recv_worker_sentinel_received_(false),
          should_stop_threads_(false),
          nccl_xor_send_initialized_(false), nccl_xor_recv_initialized_(false),
          nccl_p2p_send_initialized_(false), nccl_p2p_recv_initialized_(false),
          nccl_xor_send_init_completed_(false), nccl_xor_recv_init_completed_(false),
          nccl_p2p_send_init_completed_(false), nccl_p2p_recv_init_completed_(false),
          k_(0), rows_(0), data_block_index_(0), a_mat_(nullptr), g_tbls_(nullptr),
          decode_coefficient_0_(1), decode_coefficient_1_(1),  // Initialize to 1 for simplified version
          is_load_mode_(false), failed_rank_(-1),
          asio_initialized_(false), use_asio_(true), use_rdma_(use_rdma)
#ifdef __linux__
          , my_ip_(xor_listen_ip),
          rdma_context_(nullptr), rdma_pd_(nullptr),
          rdma_xor_send_cq_(nullptr), rdma_xor_recv_cq_(nullptr),
          rdma_p2p_send_cq_(nullptr), rdma_p2p_recv_cq_(nullptr),
          rdma_xor_send_qp_(nullptr), rdma_xor_recv_qp_(nullptr),
          rdma_xor_qp_(nullptr), rdma_p2p_send_qp_(nullptr),
          rdma_p2p_recv_qp_(nullptr), rdma_p2p_qp_(nullptr),
          rdma_step6_p2p_send_cq_(nullptr), rdma_step6_p2p_recv_cq_(nullptr), rdma_step6_p2p_qp_(nullptr),
          rdma_temp_send_mr_(nullptr), rdma_temp_recv_mr_(nullptr),
          rdma_listen_sock_(-1), rdma_xor_control_sock_(-1), rdma_p2p_control_sock_(-1)
#endif
    {

        
        // Build XOR configuration
        build_xor_config();
        
        // Build P2P configuration
        build_p2p_config();

        // Initialize EC params: k = world_size / 2, rows = 2, data_block_index = rank / 2
        rows_ = 2;
        if (world_size_ <= 0) {
            k_ = 0;
        } else {
            k_ = world_size_ / 2;
        }
        data_block_index_ = rank_ / 2;

        if (k_ > 0) {
            int m = k_ + rows_;
            // allocate matrix a (k * m)
            a_mat_ = (unsigned char*)malloc((size_t)k_ * (size_t)m);
            if (a_mat_ == nullptr) {
                std::cerr << "EC-CHECK: failed to allocate a_mat_" << std::endl;
            } else {
                // generate RS matrix
                gf_gen_rs_matrix(a_mat_, m, k_);

                // allocate g_tbls_: 32 * k * rows
                size_t gtbls_size = 32 * (size_t)k_ * (size_t)rows_;
                void *tmp = nullptr;
                if (posix_memalign(&tmp, 32, gtbls_size) != 0) tmp = nullptr;
                if (tmp == nullptr) tmp = malloc(gtbls_size);
                g_tbls_ = reinterpret_cast<unsigned char*>(tmp);
                if (g_tbls_ == nullptr) {
                    std::cerr << "EC-CHECK: failed to allocate g_tbls_" << std::endl;
                } else {
                    // initialize tables using isa-l
                    ec_init_tables(k_, rows_, a_mat_, g_tbls_);
                }
            }
        } else {
        }

        // Initialize ASIO connections in main thread (before starting worker threads)
        // Note: Using synchronous connect/accept
        // Strategy: Start accept operations in separate thread, then connect
        // This avoids deadlock when both ranks try to connect simultaneously
        
        // Start accept operations in separate threads (parallel execution)
        // This ensures both XOR and P2P accept operations start listening simultaneously
        std::thread recv_init_thread([this, xor_listen_ip, xor_recv_port, p2p_listen_ip, p2p_recv_port, step6_p2p_listen_ip, step6_p2p_recv_port]() {
            // Start parallel threads for XOR, P2P, and Step6 P2P recv
            std::thread xor_recv_thread([this, xor_listen_ip, xor_recv_port]() {
                asio_conn_mgr_.init_xor_recv(xor_listen_ip, xor_recv_port);
            });
            
            std::thread p2p_recv_thread([this, p2p_listen_ip, p2p_recv_port]() {
                asio_conn_mgr_.init_p2p_recv(p2p_listen_ip, p2p_recv_port);
            });
            
            // Step6 P2P recv: only rank_in_group 2 needs to listen
            std::thread step6_p2p_recv_thread([this, step6_p2p_listen_ip, step6_p2p_recv_port]() {
                if (rank_in_group_ == 2) {
                    asio_conn_mgr_.init_step6_p2p_recv(step6_p2p_listen_ip, step6_p2p_recv_port);
                }
            });
            
            // Wait for all accept operations to complete
            xor_recv_thread.join();
            p2p_recv_thread.join();
            step6_p2p_recv_thread.join();
        });
        
        // Small delay to ensure accept sockets are bound and listening
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Connect operations (will block until connected to partner's accept)
        asio_conn_mgr_.init_xor_send(xor_partner_ip, xor_send_port);
        asio_conn_mgr_.init_p2p_send(p2p_partner_ip, p2p_send_port);
        
        // Step6 P2P send: only rank_in_group 3 needs to connect
        if (rank_in_group_ == 3) {
            asio_conn_mgr_.init_step6_p2p_send(step6_p2p_partner_ip, step6_p2p_send_port);
        }
        
        // Wait for accept operations to complete
        recv_init_thread.join();
        
        // Verify all connections are established
        bool step6_ok = true;
        if (rank_in_group_ == 2) {
            step6_ok = asio_conn_mgr_.is_step6_p2p_recv_connected();
        } else if (rank_in_group_ == 3) {
            step6_ok = asio_conn_mgr_.is_step6_p2p_send_connected();
        }
        
        if (asio_conn_mgr_.is_xor_send_connected() && asio_conn_mgr_.is_xor_recv_connected() &&
            asio_conn_mgr_.is_p2p_send_connected() && asio_conn_mgr_.is_p2p_recv_connected() &&
            step6_ok) {
#ifdef __linux__
            if (use_rdma_) {
                init_rdma_resources();
                std::exception_ptr xor_send_exchange_error = nullptr;
                std::exception_ptr xor_recv_exchange_error = nullptr;
                std::thread xor_send_exchange_thread([this, &xor_send_exchange_error]() {
                    try {
                        exchange_and_connect_qp(
                            asio_conn_mgr_.get_xor_send_socket().native_handle(),
                            rdma_xor_send_qp_,
                            true);
                    } catch (...) {
                        xor_send_exchange_error = std::current_exception();
                    }
                });
                std::thread xor_recv_exchange_thread([this, &xor_recv_exchange_error]() {
                    try {
                        exchange_and_connect_qp(
                            asio_conn_mgr_.get_xor_recv_socket().native_handle(),
                            rdma_xor_recv_qp_,
                            false);
                    } catch (...) {
                        xor_recv_exchange_error = std::current_exception();
                    }
                });
                xor_send_exchange_thread.join();
                xor_recv_exchange_thread.join();
                if (xor_send_exchange_error) {
                    std::rethrow_exception(xor_send_exchange_error);
                }
                if (xor_recv_exchange_error) {
                    std::rethrow_exception(xor_recv_exchange_error);
                }
                std::exception_ptr p2p_send_exchange_error = nullptr;
                std::exception_ptr p2p_recv_exchange_error = nullptr;
                std::thread p2p_send_exchange_thread([this, &p2p_send_exchange_error]() {
                    try {
                        exchange_and_connect_qp(
                            asio_conn_mgr_.get_p2p_send_socket().native_handle(),
                            rdma_p2p_send_qp_,
                            true);
                    } catch (...) {
                        p2p_send_exchange_error = std::current_exception();
                    }
                });
                std::thread p2p_recv_exchange_thread([this, &p2p_recv_exchange_error]() {
                    try {
                        exchange_and_connect_qp(
                            asio_conn_mgr_.get_p2p_recv_socket().native_handle(),
                            rdma_p2p_recv_qp_,
                            false);
                    } catch (...) {
                        p2p_recv_exchange_error = std::current_exception();
                    }
                });
                p2p_send_exchange_thread.join();
                p2p_recv_exchange_thread.join();
                if (p2p_send_exchange_error) {
                    std::rethrow_exception(p2p_send_exchange_error);
                }
                if (p2p_recv_exchange_error) {
                    std::rethrow_exception(p2p_recv_exchange_error);
                }
                rdma_p2p_qp_ = rdma_p2p_send_qp_;
                if (rank_in_group_ == 2 || rank_in_group_ == 3) {
                    int sock_step6 = (rank_in_group_ == 3)
                        ? asio_conn_mgr_.get_step6_p2p_send_socket().native_handle()
                        : asio_conn_mgr_.get_step6_p2p_recv_socket().native_handle();
                    bool step6_first = (rank_in_group_ == 3);
                    exchange_and_connect_qp(sock_step6, rdma_step6_p2p_qp_, step6_first);
                } else {
                }
            }
#endif
            asio_initialized_ = true;
        } else {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] WARNING: Not all ASIO connections established" << std::endl;
            std::cerr << "  XOR send: " << asio_conn_mgr_.is_xor_send_connected() << std::endl;
            std::cerr << "  XOR recv: " << asio_conn_mgr_.is_xor_recv_connected() << std::endl;
            std::cerr << "  P2P send: " << asio_conn_mgr_.is_p2p_send_connected() << std::endl;
            std::cerr << "  P2P recv: " << asio_conn_mgr_.is_p2p_recv_connected() << std::endl;
            asio_initialized_ = false;
        }

        // Now start worker threads (ASIO is already initialized)
        start_pipeline();

    }
    
    void set_buffer_addresses(const std::vector<uintptr_t>& data_addrs,
                             const std::vector<uintptr_t>& encoding_addrs,
                             const std::vector<size_t>& sizes) {
        data_buffer_addrs_ = data_addrs;
        encoding_buffer_addrs_ = encoding_addrs;
        buffer_sizes_ = sizes;
    }
    
    void reset_encoding_completion_flags() {
        // Reset completion flags
        encoding_thread_1_completed_ = false;
        encoding_thread_2_completed_ = false;
        send_worker_completed_ = false;
        recv_worker_completed_ = false;
        xor_worker_completed_ = false;
        p2p_send_worker_completed_ = false;
        p2p_recv_worker_completed_ = false;
        encoding_thread_1_sentinel_received_ = false;
        encoding_thread_2_sentinel_received_ = false;
        send_worker_sentinel_received_ = false;
        recv_worker_sentinel_received_ = false;
        xor_worker_sentinel_received_ = false;
        p2p_send_worker_sentinel_received_ = false;
        p2p_recv_worker_sentinel_received_ = false;
        
        // Clear queues to remove any residual tasks from previous pipeline
        {
            std::lock_guard<std::mutex> lock1(encoding_tasks_1_mutex_);
            while (!encoding_tasks_1_.empty()) {
                encoding_tasks_1_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock2(encoding_tasks_2_mutex_);
            while (!encoding_tasks_2_.empty()) {
                encoding_tasks_2_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock3(send_queue_mutex_);
            while (!send_queue_.empty()) {
                send_queue_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock4(recv_queue_mutex_);
            while (!recv_queue_.empty()) {
                recv_queue_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock5(xor_queue_mutex_);
            while (!xor_queue_.empty()) {
                xor_queue_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock6(p2p_send_queue_mutex_);
            while (!p2p_send_queue_.empty()) {
                p2p_send_queue_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock7(p2p_recv_queue_mutex_);
            while (!p2p_recv_queue_.empty()) {
                p2p_recv_queue_.pop();
            }
        }
        
        // Clear buffer states
        {
            std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
            data_buffer_states_.clear();
        }
        
        // Clear pending encoding tasks
        {
            std::lock_guard<std::mutex> lock(pending_encoding_tasks_mutex_);
            pending_encoding_tasks_.clear();
        }
        
        // Clear pending XOR encoding
        {
            std::lock_guard<std::mutex> lock(pending_xor_mutex_);
            pending_xor_encoding_.clear();
        }
        
        // Clear load mode queues
        {
            std::lock_guard<std::mutex> lock(load_encoding_tasks_mutex_);
            while (!load_encoding_tasks_.empty()) {
                load_encoding_tasks_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock(pending_load_encoding_tasks_mutex_);
            pending_load_encoding_tasks_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
            while (!load_send_queue_.empty()) {
                load_send_queue_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock(load_recv_queue_mutex_);
            while (!load_recv_queue_.empty()) {
                load_recv_queue_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock(load_xor_queue_mutex_);
            while (!load_xor_queue_.empty()) {
                load_xor_queue_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock(load_p2p_send_queue_mutex_);
            while (!load_p2p_send_queue_.empty()) {
                load_p2p_send_queue_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock(load_p2p_recv_queue_mutex_);
            while (!load_p2p_recv_queue_.empty()) {
                load_p2p_recv_queue_.pop();
            }
        }
        
        // Reset load mode flags
        load_encoding_completed_ = false;
        load_encoding_sentinel_received_ = false;
        load_send_worker_completed_ = false;
        load_send_worker_sentinel_received_ = false;
        load_recv_worker_completed_ = false;
        load_recv_worker_sentinel_received_ = false;
        load_xor_worker_completed_ = false;
        load_xor_worker_sentinel_received_ = false;
        load_xor_total_ns_.store(0, std::memory_order_relaxed);
        load_xor_task_count_.store(0, std::memory_order_relaxed);
        load_xor_e2e_wall_ns_.store(0, std::memory_order_relaxed);
        load_xor_e2e_wall_valid_.store(false, std::memory_order_relaxed);
        load_encode_total_ns_.store(0, std::memory_order_relaxed);
        load_encode_task_count_.store(0, std::memory_order_relaxed);
        load_encode_e2e_wall_ns_.store(0, std::memory_order_relaxed);
        load_encode_e2e_wall_valid_.store(false, std::memory_order_relaxed);
        load_enc_xor_send_total_ns_.store(0, std::memory_order_relaxed);
        load_enc_xor_send_task_count_.store(0, std::memory_order_relaxed);
        load_enc_xor_recv_total_ns_.store(0, std::memory_order_relaxed);
        load_enc_xor_recv_task_count_.store(0, std::memory_order_relaxed);
        load_step2_p2p_send_total_ns_.store(0, std::memory_order_relaxed);
        load_step2_p2p_send_task_count_.store(0, std::memory_order_relaxed);
        load_step2_p2p_recv_total_ns_.store(0, std::memory_order_relaxed);
        load_step2_p2p_recv_task_count_.store(0, std::memory_order_relaxed);
        load_step6_p2p_send_total_ns_.store(0, std::memory_order_relaxed);
        load_step6_p2p_send_task_count_.store(0, std::memory_order_relaxed);
        load_step6_p2p_recv_total_ns_.store(0, std::memory_order_relaxed);
        load_step6_p2p_recv_task_count_.store(0, std::memory_order_relaxed);
        load_rank2_rdma_poll_enc_xor_recv_ns_.store(0, std::memory_order_relaxed);
        load_rank2_rdma_poll_step2_p2p_recv_ns_.store(0, std::memory_order_relaxed);
        load_rank2_rdma_poll_step6_p2p_recv_ns_.store(0, std::memory_order_relaxed);
        load_net_wall_span_ns_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(load_net_wall_mu_);
            load_net_wall_have_any_ = false;
        }
        save_encode_total_ns_.store(0, std::memory_order_relaxed);
        save_encode_op_count_.store(0, std::memory_order_relaxed);
        save_net_wall_span_ns_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(save_net_wall_mu_);
            save_net_wall_have_any_ = false;
        }
        {
            std::lock_guard<std::mutex> lk(load_recv_pipeline_e2e_mu_);
            load_recv_pipeline_e2e_have_any_ = false;
        }
        load_p2p_send_worker_completed_ = false;
        load_p2p_send_worker_sentinel_received_ = false;
        load_p2p_recv_worker_completed_ = false;
        load_p2p_recv_worker_sentinel_received_ = false;
        load_step6_p2p_send_worker_completed_ = false;
        load_step6_p2p_send_worker_sentinel_received_ = false;
        load_step6_p2p_recv_worker_completed_ = false;
        load_step6_p2p_recv_worker_sentinel_received_ = false;
        
        // If in load mode, mark unused workers as completed immediately
        if (is_load_mode_) {
            // send worker: only rank0/1 use it
            if (rank_ != 0 && rank_ != 1) {
                load_send_worker_completed_ = true;
            }
            
            // recv worker: only rank2/3 use it
            if (rank_ != 2 && rank_ != 3) {
                load_recv_worker_completed_ = true;
            }
            
            // p2p_send worker: only rank0/3 use it
            if (rank_ != 0 && rank_ != 3) {
                load_p2p_send_worker_completed_ = true;
            }
            
            // p2p_recv worker: only rank1/2 use it
            if (rank_ != 1 && rank_ != 2) {
                load_p2p_recv_worker_completed_ = true;
            }
            
            // step6_p2p_send worker: only rank3 uses it
            if (rank_ != 3) {
                load_step6_p2p_send_worker_completed_ = true;
            }
            
            // step6_p2p_recv worker: only rank2 uses it
            if (rank_ != 2) {
                load_step6_p2p_recv_worker_completed_ = true;
            }
        }
        
        // Clear load mode pending XOR mappings
        {
            std::lock_guard<std::mutex> lock(load_pending_xor_mutex_);
            load_pending_xor_encoding_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(load_recv_to_parity_mutex_);
            load_recv_to_parity_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(load_recv_to_data_mutex_);
            load_recv_to_data_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(load_recv_to_p2p_mutex_);
            load_recv_to_p2p_partner_write_.clear();
        }
        
        // Clear load mode Step6 P2P queues
        {
            std::lock_guard<std::mutex> lock(load_step6_p2p_send_queue_mutex_);
            while (!load_step6_p2p_send_queue_.empty()) {
                load_step6_p2p_send_queue_.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock(load_step6_p2p_recv_queue_mutex_);
            while (!load_step6_p2p_recv_queue_.empty()) {
                load_step6_p2p_recv_queue_.pop();
            }
        }
        
    }
    
    void wait_for_encoding_completion() {
        if (is_two_failures_load_mode_) {
            // Two-failure mode: wait for save-path workers (encoder threads,
            // send/recv/xor/p2p) which are driven by encoding_tasks_1_/_2_ queues.
            // Load workers are not used in this path.
            int wait_count = 0;
            while (true) {
                bool all_completed = true;

                if (!encoding_thread_1_completed_.load())  all_completed = false;
                if (!encoding_thread_2_completed_.load())  all_completed = false;
                if (!send_worker_completed_.load())         all_completed = false;
                if (!recv_worker_completed_.load())         all_completed = false;
                if (!xor_worker_completed_.load())          all_completed = false;
                if (!p2p_send_worker_completed_.load())     all_completed = false;
                if (!p2p_recv_worker_completed_.load())     all_completed = false;

                if (all_completed) break;

                if (++wait_count > 3000) {  // ~30s timeout
                    std::cerr << "EC-CHECK: [Rank " << rank_
                              << "] Two-failure wait_for_encoding_completion timeout:"
                              << " enc1=" << encoding_thread_1_completed_.load()
                              << " enc2=" << encoding_thread_2_completed_.load()
                              << " send=" << send_worker_completed_.load()
                              << " recv=" << recv_worker_completed_.load()
                              << " xor=" << xor_worker_completed_.load()
                              << " p2ps=" << p2p_send_worker_completed_.load()
                              << " p2pr=" << p2p_recv_worker_completed_.load()
                              << std::endl;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            // Two-failure encoding is CPU-only (ec_rs_encode_pool); no CUDA sync needed.
            return;
        }
        if (is_load_mode_) {
            // Load mode: 等待该 rank 实际使用的 load worker 完成
            int wait_count = 0;
            while (true) {
                bool all_used_workers_completed = true;
                
                // encoder: 所有 rank 都使用
                if (!load_encoding_completed_.load()) {
                    all_used_workers_completed = false;
                }
                
                // send: only rank_in_group 0/1 use it
                if ((rank_in_group_ == 0 || rank_in_group_ == 1) && !load_send_worker_completed_.load()) {
                    all_used_workers_completed = false;
                }
                
                // recv: only rank_in_group 2/3 use it
                if ((rank_in_group_ == 2 || rank_in_group_ == 3) && !load_recv_worker_completed_.load()) {
                    all_used_workers_completed = false;
                }
                
                // xor: 所有 rank 都使用
                if (!load_xor_worker_completed_.load()) {
                    all_used_workers_completed = false;
                }
                
                // p2p_send: 只有 rank0/3 使用
                if ((rank_in_group_ == 0 || rank_in_group_ == 3) && !load_p2p_send_worker_completed_.load()) {
                    all_used_workers_completed = false;
                }
                
                // p2p_recv: 只有 rank1/2 使用
                if ((rank_in_group_ == 1 || rank_in_group_ == 2) && !load_p2p_recv_worker_completed_.load()) {
                    all_used_workers_completed = false;
                }
                
                // step6_p2p_send: only rank_in_group 3 uses it
                if (rank_in_group_ == 3 && !load_step6_p2p_send_worker_completed_.load()) {
                    all_used_workers_completed = false;
                }
                
                // step6_p2p_recv: only rank_in_group 2 uses it
                if (rank_in_group_ == 2 && !load_step6_p2p_recv_worker_completed_.load()) {
                    all_used_workers_completed = false;
                }
                
                if (all_used_workers_completed) {
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                wait_count++;
            }
            return;  // Load mode 下直接返回，不等待 save mode 的 workers
        } else {
            // Save mode: 现有逻辑保持不变
        // Wait for all encoding threads to complete
        bool need_thread1 = true;
        
        int encoding_wait_count = 0;
        while ((need_thread1 && !encoding_thread_1_completed_) || !encoding_thread_2_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            encoding_wait_count++;
        }
        }
        
        // Wait for send worker to complete (only for save mode)
        while (!send_worker_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Wait for recv worker to complete
        while (!recv_worker_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Wait for XOR worker to complete
        while (!xor_worker_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Wait for both P2P workers to complete
        while (!p2p_send_worker_completed_ || !p2p_recv_worker_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    // Wait only for XOR worker completion (used before sending Step6 sentinel)
    void wait_for_xor_worker_completion() {
        if (!is_load_mode_) {
            std::cerr << "EC-CHECK: [Rank " << rank_ 
                      << "] wait_for_xor_worker_completion called but not in load mode" << std::endl;
            return;
        }
        
        int wait_count = 0;
        while (!load_xor_worker_completed_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
        }
    }
    
    void stop_pipeline() {
        should_stop_threads_ = true;
        
        // Notify all threads
        encoding_tasks_1_cv_.notify_all();
        encoding_tasks_2_cv_.notify_all();
        send_queue_cv_.notify_all();
        recv_queue_cv_.notify_all();
        xor_queue_cv_.notify_all();
        p2p_send_queue_cv_.notify_all();
        p2p_recv_queue_cv_.notify_all();
        
        // Notify load mode threads
        load_encoding_tasks_cv_.notify_all();
        load_send_queue_cv_.notify_all();
        load_recv_queue_cv_.notify_all();
        load_xor_queue_cv_.notify_all();
        load_p2p_send_queue_cv_.notify_all();
        load_p2p_recv_queue_cv_.notify_all();
        load_step6_p2p_send_queue_cv_.notify_all();
        load_step6_p2p_recv_queue_cv_.notify_all();
        load_xor_pool_worker_cv_.notify_all();
        load_xor_pool_coordinator_cv_.notify_all();
        load_encode_pool_worker_cv_.notify_all();
        load_encode_pool_coordinator_cv_.notify_all();
        
        // Join all threads
        if (encoder_thread_1_.joinable()) encoder_thread_1_.join();
        if (encoder_thread_2_.joinable()) encoder_thread_2_.join();
        if (send_worker_.joinable()) send_worker_.join();
        if (recv_worker_.joinable()) recv_worker_.join();
        if (xor_worker_.joinable()) xor_worker_.join();
        if (p2p_send_worker_.joinable()) p2p_send_worker_.join();
        if (p2p_recv_worker_.joinable()) p2p_recv_worker_.join();
        if (load_encoder_worker_.joinable()) load_encoder_worker_.join();
        ec_rs_encode_pool_shutdown();
        ec_xor_pool_shutdown();
        load_encode_pool_shutdown();
        if (rank_in_group_ == 0 || rank_in_group_ == 1) {
            if (load_send_worker_.joinable()) load_send_worker_.join();
        }
        if (rank_in_group_ == 2 || rank_in_group_ == 3) {
            if (load_recv_worker_.joinable()) load_recv_worker_.join();
        }
        if (load_xor_worker_.joinable()) load_xor_worker_.join();
        load_xor_pool_shutdown();
        if (rank_in_group_ == 0 || rank_in_group_ == 3) {
            if (load_p2p_send_worker_.joinable()) load_p2p_send_worker_.join();
        }
        if (rank_in_group_ == 1 || rank_in_group_ == 2) {
            if (load_p2p_recv_worker_.joinable()) load_p2p_recv_worker_.join();
        }
        if (rank_in_group_ == 3) {
            if (load_step6_p2p_send_worker_.joinable()) load_step6_p2p_send_worker_.join();
        }
        if (rank_in_group_ == 2) {
            if (load_step6_p2p_recv_worker_.joinable()) load_step6_p2p_recv_worker_.join();
        }
        
    }
    
    void submit_data_for_encoding_thread1(uintptr_t data_addr, size_t size, 
                                          uintptr_t encoding_addr, uintptr_t recv_addr, 
                                          size_t recv_chunk_size, uintptr_t parity_addr,
                                          uintptr_t p2p_own_write_addr, uintptr_t p2p_partner_write_addr,
                                          bool local_is_zero_tail = false,
                                          bool remote_is_zero_tail = false,
                                          bool p2p_data_is_zero_tail = false,
                                          size_t p2p_data_size = 0) {
        size_t sequence_id = 0;
        if (data_addr != 0 || size != 0) {
            sequence_id = save_sequence_id_thread1_.fetch_add(1, std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
            encoding_tasks_1_.push({data_addr, size, encoding_addr, recv_addr, recv_chunk_size, 
                                    parity_addr, p2p_own_write_addr, p2p_partner_write_addr,
                                    local_is_zero_tail, remote_is_zero_tail, p2p_data_is_zero_tail,
                                    p2p_data_size, sequence_id});
        }
        encoding_tasks_1_cv_.notify_one();
        
        if (data_addr != 0 && size != 0) {
            std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
            if (data_buffer_states_.find(data_addr) == data_buffer_states_.end()) {
                data_buffer_states_[data_addr] = {false, false};
            }
        }
    }
    
    void submit_data_for_encoding_thread2(uintptr_t data_addr, size_t size,
                                          uintptr_t encoding_addr, uintptr_t recv_addr,
                                          size_t recv_chunk_size, uintptr_t parity_addr,
                                          uintptr_t p2p_own_write_addr, uintptr_t p2p_partner_write_addr,
                                          bool local_is_zero_tail = false,
                                          bool remote_is_zero_tail = false,
                                          bool p2p_data_is_zero_tail = false,
                                          size_t p2p_data_size = 0) {
        size_t sequence_id = 0;
        if (data_addr != 0 || size != 0) {
            sequence_id = save_sequence_id_thread2_.fetch_add(1, std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
            encoding_tasks_2_.push({data_addr, size, encoding_addr, recv_addr, recv_chunk_size,
                                    parity_addr, p2p_own_write_addr, p2p_partner_write_addr,
                                    local_is_zero_tail, remote_is_zero_tail, p2p_data_is_zero_tail,
                                    p2p_data_size, sequence_id});
        }
        encoding_tasks_2_cv_.notify_one();

        if (data_addr != 0 && size != 0) {
            std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
            if (data_buffer_states_.find(data_addr) == data_buffer_states_.end()) {
                data_buffer_states_[data_addr] = {false, false};
            }
        }
    }
    
    std::vector<uintptr_t> get_data_buffers_to_release() {
        std::vector<uintptr_t> buffers;
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        while (!data_buffers_to_release_.empty()) {
            buffers.push_back(data_buffers_to_release_.front());
            data_buffers_to_release_.pop();
        }
        return buffers;
    }
    
    std::vector<uintptr_t> get_encoding_buffers_to_release() {
        std::vector<uintptr_t> buffers;
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        while (!encoding_buffers_to_release_.empty()) {
            buffers.push_back(encoding_buffers_to_release_.front());
            encoding_buffers_to_release_.pop();
        }
        return buffers;
    }
    
    std::vector<uintptr_t> get_parity_buffers_to_release() {
        std::vector<uintptr_t> buffers;
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        while (!parity_buffers_to_release_.empty()) {
            buffers.push_back(parity_buffers_to_release_.front());
            parity_buffers_to_release_.pop();
        }
        return buffers;
    }

    pybind11::dict get_ft_timing_stats() const {
        double net_s = 0.0;
        double encode_s = 0.0;
        if (is_load_mode_) {
            net_s = static_cast<double>(
                load_net_wall_span_ns_.load(std::memory_order_relaxed)) / 1e9;
            const uint64_t enc_total_ns =
                load_encode_total_ns_.load(std::memory_order_relaxed) +
                load_xor_total_ns_.load(std::memory_order_relaxed);
            encode_s = static_cast<double>(enc_total_ns) / 1e9;
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
    
    void submit_data_to_p2p_thread(uintptr_t data_addr, size_t size, std::string ops) {
        if (data_addr == 0 || size == 0) {
            // std::cerr << "EC-CHECK: [Rank " << rank_
            //           << "] submit_data_to_p2p_thread received invalid args: addr="
            //           << data_addr << ", size=" << size << std::endl;
            return;
        }
        
        if (ops == "send") {
            {
                std::lock_guard<std::mutex> lock(p2p_send_queue_mutex_);
                // Load-time P2P transfer doesn't need a local copy, so p2p_own_write_addr/parity/data are 0
                p2p_send_queue_.push({data_addr, 0, size, 0, 0, false, false, 0, 0, 0});
            }
            p2p_send_queue_cv_.notify_one();
            // std::cout << "EC-CHECK: [Rank " << rank_
            //           << "] Queued P2P send task (addr=" << data_addr
            //           << ", size=" << size << ")" << std::endl;
        } else if (ops == "recv") {
            {
                std::lock_guard<std::mutex> lock(p2p_recv_queue_mutex_);
                p2p_recv_queue_.push({data_addr, size, false, false, 0, false, 0, 0});
            }
            p2p_recv_queue_cv_.notify_one();
            // std::cout << "EC-CHECK: [Rank " << rank_
            //           << "] Queued P2P recv task (addr=" << data_addr
            //           << ", size=" << size << ")" << std::endl;
        } else {
            std::cerr << "EC-CHECK: [Rank " << rank_
                      << "] submit_data_to_p2p_thread received unknown op: "
                      << ops << std::endl;
        }
    }
    
    void submit_load_p2p_transfer(
        uintptr_t send_buffer_addr,      // rank0/3: 要发送的数据地址（从mmap读取）
        uintptr_t recv_data_buffer_addr,  // rank1/2: 接收后写入的data_buffer地址
        size_t size,
        bool is_sender,                   // true=sender (rank0/3), false=receiver (rank1/2)
        uintptr_t load_mode_data_addr = 0 // load mode: corresponding data_addr (for finding encoding task)
    ) {
        if (size == 0) {
            // std::cerr << "EC-CHECK: [Rank " << rank_
            //           << "] submit_load_p2p_transfer received invalid size: " << size << std::endl;
            return;
        }
        
        if (is_sender) {
            // Sender: 提交到load_p2p_send_queue_ (独立的 load mode 队列)
            if (send_buffer_addr == 0) {
                // std::cerr << "EC-CHECK: [Rank " << rank_
                //           << "] submit_load_p2p_transfer: sender requires send_buffer_addr" << std::endl;
                return;
            }
            
            {
                std::lock_guard<std::mutex> lock(load_p2p_send_queue_mutex_);
                load_p2p_send_queue_.push({
                    send_buffer_addr,    // send_buffer_addr
                    0,                   // p2p_own_write_addr (Step2不需要)
                    size,                // size
                    0,                   // parity_addr (Step2不需要)
                    0,                   // data_addr (Step2不需要)
                    true,                // is_load_mode_transfer
                    false,               // is_step6_transfer (Step2, not Step6)
                    load_mode_data_addr, // load_mode_data_addr (for finding encoding task after Step2 send)
                    0,
                    0
                });
            }
            load_p2p_send_queue_cv_.notify_one();
            // std::cout << "EC-CHECK: [Rank " << rank_ << "] Queued load P2P send task (addr=" 
            //           << send_buffer_addr << ", size=" << size << ")" << std::endl;
        } else {
            // Receiver: 提交到load_p2p_recv_queue_ (独立的 load mode 队列)
            if (recv_data_buffer_addr == 0) {
                // std::cerr << "EC-CHECK: [Rank " << rank_
                //           << "] submit_load_p2p_transfer: receiver requires recv_data_buffer_addr" << std::endl;
                return;
            }
            
            {
                std::lock_guard<std::mutex> lock(load_p2p_recv_queue_mutex_);
                load_p2p_recv_queue_.push({
                    recv_data_buffer_addr, // recv_buffer_addr (直接使用data_buffer作为接收buffer)
                    size,                  // size
                    true,                  // is_load_mode_transfer
                    false,                 // is_step6_transfer (Step2, not Step6)
                    recv_data_buffer_addr, // data_buffer_addr (接收后数据就在这个地址，无需复制)
                    false,
                    0,
                    0
                });
            }
            load_p2p_recv_queue_cv_.notify_one();
            // std::cout << "EC-CHECK: [Rank " << rank_ << "] Queued load P2P recv task (addr=" 
            //           << recv_data_buffer_addr << ", size=" << size << ")" << std::endl;
        }
    }
    
    void set_load_mode(bool is_load, int failed_rank) {
        is_load_mode_ = is_load;
        failed_rank_ = failed_rank;
        // failed_rank=10 → two-failure hardware recovery (following ECLATIN convention)
        is_two_failures_load_mode_ = (is_load && failed_rank == 10);
        failed_rank_in_group_ =
            (failed_rank >= 0 && failed_rank != 10) ? (failed_rank % ECCHECK_RANKS_PER_GROUP) : -1;
        // Rebuild XOR configuration (needs failed_rank_in_group_ for hardware recovery branches)
        build_xor_config();
        
        // 新增：如果是 load mode，启动该 rank 实际使用的 load worker
        if (is_load && !load_encoder_worker_.joinable()) {
            // Reset all load worker flags
            load_encoding_completed_ = false;
            load_encoding_sentinel_received_ = false;
            load_send_worker_completed_ = false;
            load_send_worker_sentinel_received_ = false;
            load_recv_worker_completed_ = false;
            load_recv_worker_sentinel_received_ = false;
            load_xor_worker_completed_ = false;
            load_xor_worker_sentinel_received_ = false;
            load_xor_total_ns_.store(0, std::memory_order_relaxed);
            load_xor_task_count_.store(0, std::memory_order_relaxed);
            load_xor_e2e_wall_ns_.store(0, std::memory_order_relaxed);
            load_xor_e2e_wall_valid_.store(false, std::memory_order_relaxed);
            load_encode_total_ns_.store(0, std::memory_order_relaxed);
            load_encode_task_count_.store(0, std::memory_order_relaxed);
            load_encode_e2e_wall_ns_.store(0, std::memory_order_relaxed);
            load_encode_e2e_wall_valid_.store(false, std::memory_order_relaxed);
            load_enc_xor_send_total_ns_.store(0, std::memory_order_relaxed);
            load_enc_xor_send_task_count_.store(0, std::memory_order_relaxed);
            load_enc_xor_recv_total_ns_.store(0, std::memory_order_relaxed);
            load_enc_xor_recv_task_count_.store(0, std::memory_order_relaxed);
            load_step2_p2p_send_total_ns_.store(0, std::memory_order_relaxed);
            load_step2_p2p_send_task_count_.store(0, std::memory_order_relaxed);
            load_step2_p2p_recv_total_ns_.store(0, std::memory_order_relaxed);
            load_step2_p2p_recv_task_count_.store(0, std::memory_order_relaxed);
            load_step6_p2p_send_total_ns_.store(0, std::memory_order_relaxed);
            load_step6_p2p_send_task_count_.store(0, std::memory_order_relaxed);
            load_step6_p2p_recv_total_ns_.store(0, std::memory_order_relaxed);
            load_step6_p2p_recv_task_count_.store(0, std::memory_order_relaxed);
            load_rank2_rdma_poll_enc_xor_recv_ns_.store(0, std::memory_order_relaxed);
            load_rank2_rdma_poll_step2_p2p_recv_ns_.store(0, std::memory_order_relaxed);
            load_rank2_rdma_poll_step6_p2p_recv_ns_.store(0, std::memory_order_relaxed);
            load_net_wall_span_ns_.store(0, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lk(load_net_wall_mu_);
                load_net_wall_have_any_ = false;
            }
            {
                std::lock_guard<std::mutex> lk(load_recv_pipeline_e2e_mu_);
                load_recv_pipeline_e2e_have_any_ = false;
            }
            load_p2p_send_worker_completed_ = false;
            load_p2p_send_worker_sentinel_received_ = false;
            load_p2p_recv_worker_completed_ = false;
            load_p2p_recv_worker_sentinel_received_ = false;
            load_step6_p2p_send_worker_completed_ = false;
            load_step6_p2p_send_worker_sentinel_received_ = false;
            load_step6_p2p_recv_worker_completed_ = false;
            load_step6_p2p_recv_worker_sentinel_received_ = false;
            
            load_encode_pool_init();
            // Start load encoder worker (all ranks use it)
            load_encoder_worker_ = std::thread(&ECCHECKNative::load_encoder_worker, this);
            
            // Start load send worker (only rank_in_group 0/1 use it)
            if (rank_in_group_ == 0 || rank_in_group_ == 1) {
                load_send_worker_ = std::thread(&ECCHECKNative::load_send_worker, this);
            } else {
                // rank_in_group 2/3 don't use load_send_queue_, mark as completed immediately
                load_send_worker_completed_ = true;
            }
            
            // Start load recv worker (only rank_in_group 2/3 use it)
            if (rank_in_group_ == 2 || rank_in_group_ == 3) {
                load_recv_worker_ = std::thread(&ECCHECKNative::load_recv_worker, this);
            } else {
                load_recv_worker_completed_ = true;
            }
            
            // Start load XOR worker (all ranks use it, though only rank2/3 actually do XOR)
            if (rank_in_group_ == 2 || rank_in_group_ == 3) {
                load_xor_pool_init();
            }
            load_xor_worker_ = std::thread(&ECCHECKNative::load_xor_worker, this);
            
            // Start load P2P send worker (only rank0/3 use it)
            if (rank_in_group_ == 0 || rank_in_group_ == 3) {
                load_p2p_send_worker_ = std::thread(&ECCHECKNative::load_p2p_send_worker, this);
            } else {
                // rank1/2 don't use load_p2p_send_queue_, mark as completed immediately
                load_p2p_send_worker_completed_ = true;
            }
            
            // Start load P2P recv worker (only rank_in_group 1/2 use it)
            if (rank_in_group_ == 1 || rank_in_group_ == 2) {
                load_p2p_recv_worker_ = std::thread(&ECCHECKNative::load_p2p_recv_worker, this);
            } else {
                // rank0/3 don't use load_p2p_recv_queue_, mark as completed immediately
                load_p2p_recv_worker_completed_ = true;
            }
            
            // Start load Step6 P2P send worker (only rank_in_group 3 uses it)
            if (rank_in_group_ == 3) {
                load_step6_p2p_send_worker_ = std::thread(&ECCHECKNative::load_step6_p2p_send_worker, this);
            } else {
                // rank_in_group 0/1/2 don't use load_step6_p2p_send_queue_, mark as completed immediately
                load_step6_p2p_send_worker_completed_ = true;
            }
            
            // Start load Step6 P2P recv worker (only rank_in_group 2 uses it)
            if (rank_in_group_ == 2) {
                load_step6_p2p_recv_worker_ = std::thread(&ECCHECKNative::load_step6_p2p_recv_worker, this);
            } else {
                // rank_in_group 0/1/3 don't use load_step6_p2p_recv_queue_, mark as completed immediately
                load_step6_p2p_recv_worker_completed_ = true;
            }
            
        }
    }
    
    void submit_load_pipeline_chunk(
        uintptr_t step2_send_addr,        // rank0/3: partner_file地址；rank1/2: 0
        uintptr_t step2_recv_data_addr,   // rank1/2: data_buffer地址；rank0/3: 0
        size_t step2_size,                 // Step 2传输大小
        uintptr_t data_addr,               // 数据buffer地址
        size_t size,                       // 数据大小
        uintptr_t encoding_addr,          // 编码buffer（parity index 1）
        uintptr_t recv_addr,               // 接收地址（只有 rank2/3 需要，其他传 0）
        size_t recv_chunk_size,           // 接收chunk大小（只有 rank2/3 需要，其他传 0）
        uintptr_t parity_addr,            // parity buffer（只有 rank2/3 需要，其他传 0）
        uintptr_t p2p_partner_write_addr   // Step6: rank2 接收 d3 的地址（只有 rank2 需要，其他传 0）
    ) {
        if (!is_load_mode_) {
            std::cerr << "EC-CHECK: [Rank " << rank_ 
                      << "] submit_load_pipeline_chunk called but not in load mode" << std::endl;
            return;
        }
        
        // Determine sender vs receiver by rank_in_group
        bool is_receiver = (rank_in_group_ == 2 || rank_in_group_ == 3);
        
        // 准备 load encoding 任务
        LoadEncodingTask load_task = {
            data_addr, size, encoding_addr,
            recv_addr, recv_chunk_size, parity_addr,
            is_receiver,
            p2p_partner_write_addr  // For Step6: rank2 needs this to receive d3
        };
        
        if (rank_in_group_ == 0 || rank_in_group_ == 3) {
            // Sender: 提交 Step2 P2P 发送任务
            if (step2_send_addr != 0 && step2_size > 0) {
                submit_load_p2p_transfer(step2_send_addr, 0, step2_size, true, data_addr);
                
                // 保存 encoding 任务，等 P2P send 完成后提交
                {
                    std::lock_guard<std::mutex> lock(pending_load_encoding_tasks_mutex_);
                    pending_load_encoding_tasks_[data_addr] = load_task;
                }
            }
        } else if (rank_in_group_ == 1 || rank_in_group_ == 2) {
            // Receiver: 提交 Step2 P2P 接收任务
            if (step2_recv_data_addr != 0 && step2_size > 0) {
                submit_load_p2p_transfer(0, step2_recv_data_addr, step2_size, false, 0);
                
                {
                    std::lock_guard<std::mutex> lock(pending_load_encoding_tasks_mutex_);
                    pending_load_encoding_tasks_[step2_recv_data_addr] = load_task;
                }
            }
        }
    }

    // ========== Two-Failure Encoding Submission ==========

    void submit_two_failure_encoding_chunk(
        uintptr_t data_addr, size_t size,
        uintptr_t enc_addr_0, uintptr_t enc_addr_1,
        uintptr_t recv_addr_1, uintptr_t recv_addr_2,
        size_t recv_chunk_size,
        uintptr_t own_write_addr, uintptr_t partner_write_addr
    ) {
        if (!is_two_failures_load_mode_) {
            std::cerr << "EC-CHECK: [Rank " << rank_
                      << "] submit_two_failure_encoding_chunk called but not in two-failure mode"
                      << std::endl;
            return;
        }

        const size_t sequence_id_1 = save_sequence_id_thread1_.fetch_add(1, std::memory_order_relaxed);
        const size_t sequence_id_2 = save_sequence_id_thread2_.fetch_add(1, std::memory_order_relaxed);

        // Submit TWO encoding tasks to save-path encoder threads.
        // Encoder thread 1 (parity row 0): rig0→rig2, rig1→rig3
        //   receivers: rig2, rig3 → parity_addr = own_write_addr for XOR output
        //   senders:   rig0, rig1 → parity_addr = 0 (send via send_queue)
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
            encoding_tasks_1_.push({
                data_addr, size, enc_addr_0,
                recv_addr_1, recv_chunk_size,
                own_write_addr,  // parity_addr: XOR destination (= own_buffer write offset)
                own_write_addr, partner_write_addr,
                false, false, false, size, sequence_id_1
            });
        }
        encoding_tasks_1_cv_.notify_one();

        // Encoder thread 2 (parity row 1): rig2→rig0, rig3→rig1
        //   receivers: rig0, rig1 → parity_addr = own_write_addr for XOR output
        //   senders:   rig2, rig3 → parity_addr = 0 (send via send_queue)
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
            encoding_tasks_2_.push({
                data_addr, size, enc_addr_1,
                recv_addr_2, recv_chunk_size,
                own_write_addr,  // parity_addr: XOR destination (= own_buffer write offset)
                own_write_addr, partner_write_addr,
                false, false, false, size, sequence_id_2
            });
        }
        encoding_tasks_2_cv_.notify_one();

        // Track data buffer state for release coordination
        {
            std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
            if (data_buffer_states_.find(data_addr) == data_buffer_states_.end()) {
                data_buffer_states_[data_addr] = {false, false};
            }
        }
    }

    void submit_two_failure_encoding_sentinels() {
        if (!is_two_failures_load_mode_) {
            std::cerr << "EC-CHECK: [Rank " << rank_
                      << "] submit_two_failure_encoding_sentinels called but not in two-failure mode"
                      << std::endl;
            return;
        }

        // Send sentinels to both encoder threads
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
            encoding_tasks_1_.push({0, 0, 0, 0, 0, 0, 0, 0, false, false, false, 0, 0});
        }
        encoding_tasks_1_cv_.notify_one();

        {
            std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
            encoding_tasks_2_.push({0, 0, 0, 0, 0, 0, 0, 0, false, false, false, 0, 0});
        }
        encoding_tasks_2_cv_.notify_one();
    }

    // ========== Load Mode (新增，写在文件末尾)==========

    // Load Encoder Worker - load mode encoding worker
    void load_encoder_worker() {
        bool encode_have_chunk = false;
        std::chrono::steady_clock::time_point encode_first_start{};
        std::chrono::steady_clock::time_point encode_last_end{};
        
        while (!should_stop_threads_) {
            LoadEncodingTask task;
            
            {
                std::unique_lock<std::mutex> lock(load_encoding_tasks_mutex_);
                load_encoding_tasks_cv_.wait(lock, [this] {
                    return !load_encoding_tasks_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && load_encoding_tasks_.empty()) {
                    if (encode_have_chunk) {
                        const uint64_t e2e_ns = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(encode_last_end - encode_first_start).count());
                        load_encode_e2e_wall_ns_.store(e2e_ns, std::memory_order_relaxed);
                        load_encode_e2e_wall_valid_.store(true, std::memory_order_relaxed);
                        encode_have_chunk = false;
                    }
                    break;
                }
                
                if (load_encoding_tasks_.empty()) {
                    // Queue is empty, check if sentinel was received
                    if (load_encoding_sentinel_received_.load() && !load_encoding_completed_.load()) {
                        lock.unlock();
                        // Check pending tasks
                        bool can_send_sentinel = false;
                        {
                            std::lock_guard<std::mutex> pending_lock(pending_load_encoding_tasks_mutex_);
                            can_send_sentinel = pending_load_encoding_tasks_.empty();
                            if (!can_send_sentinel) {
                                // Sleep a bit before checking again
                                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            }
                        }
                        
                        if (can_send_sentinel) {
                            // All tasks completed, will send sentinel in the check below
                            // Create a dummy sentinel task to trigger the check
                            task = {0, 0, 0, 0, 0, false, 0};
                        } else {
                            // Still have pending tasks, continue waiting
                            continue;
                        }
                    } else {
                        // No sentinel received yet or already completed, continue waiting
                        continue;
                    }
                } else {
                    task = load_encoding_tasks_.front();
                    load_encoding_tasks_.pop();
                }
            }
            
            // Check for sentinel
            if (task.data_addr == 0 && task.size == 0) {
                if (!load_encoding_sentinel_received_.load()) {
                    load_encoding_sentinel_received_ = true;
                }
                // Skip processing sentinel as a normal task, go to the check below
            }
            
            // Perform encoding (only parity index 1)
            if (task.data_addr != 0 && task.encoding_addr != 0) {
                if (!encode_have_chunk) {
                    encode_first_start = std::chrono::steady_clock::now();
                    encode_have_chunk = true;
                }
                const auto encode_chunk_t0 = std::chrono::steady_clock::now();
                encode_with_coefficient(task.data_addr, task.size, task.encoding_addr, 1);
                const auto encode_chunk_t1 = std::chrono::steady_clock::now();
                encode_last_end = encode_chunk_t1;
                const uint64_t encode_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(encode_chunk_t1 - encode_chunk_t0).count());
                record_load_encode_op_(encode_ns);
                
                // Release data buffer immediately (load mode doesn't need to wait for P2P)
                {
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    data_buffers_to_release_.push(task.data_addr);
                }
                
                // Handle based on sender/receiver role
                if (task.is_receiver && task.parity_addr != 0 && task.recv_addr != 0) {
                    // Receiver (rank2/3): save encoding for XOR
                    {
                        std::lock_guard<std::mutex> lock(load_pending_xor_mutex_);
                        load_pending_xor_encoding_[task.recv_addr] = task.encoding_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(load_recv_to_parity_mutex_);
                        load_recv_to_parity_[task.recv_addr] = task.parity_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(load_recv_to_data_mutex_);
                        load_recv_to_data_[task.recv_addr] = task.data_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(load_recv_to_p2p_mutex_);
                        load_recv_to_p2p_partner_write_[task.recv_addr] = task.p2p_partner_write_addr;
                    }
                    // Submit recv task to load recv queue
                    {
                        std::lock_guard<std::mutex> lock(load_recv_queue_mutex_);
                        load_recv_queue_.push({task.recv_addr, task.recv_chunk_size, task.parity_addr});
                    }
                    load_recv_queue_cv_.notify_one();
                } else {
                    // Sender (rank0/1): send encoding immediately to load send queue
                    {
                        std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
                        load_send_queue_.push({task.encoding_addr, task.size});
                    }
                    load_send_queue_cv_.notify_one();
                    
                    // FIX: Do NOT release encoding buffer here!
                    // The buffer will be released by load_send_worker after sending completes
                    // This matches the behavior of save mode's send_worker
                }
            }
            
            // After processing task, check if sentinel was received and both queues are empty
            if (load_encoding_sentinel_received_.load() && !load_encoding_completed_.load()) {
                bool can_send_sentinel = false;
                {
                    std::lock_guard<std::mutex> lock(load_encoding_tasks_mutex_);
                    if (load_encoding_tasks_.empty()) {
                        std::lock_guard<std::mutex> pending_lock(pending_load_encoding_tasks_mutex_);
                        if (pending_load_encoding_tasks_.empty()) {
                            can_send_sentinel = true;
                        } else {
                            // Still have pending tasks waiting for P2P completion
                        }
                    }
                }
                
                if (can_send_sentinel) {
                    if (encode_have_chunk) {
                        const uint64_t e2e_ns = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(encode_last_end - encode_first_start).count());
                        load_encode_e2e_wall_ns_.store(e2e_ns, std::memory_order_relaxed);
                        load_encode_e2e_wall_valid_.store(true, std::memory_order_relaxed);
                        encode_have_chunk = false;
                    }
                    load_encoding_completed_ = true;
                    // Submit sentinel to downstream load workers (only to queues used by this rank)
                    // rank_in_group 0/1: use load_send_queue_ (send encoding to rank_in_group 2/3)
                    if (rank_in_group_ == 0 || rank_in_group_ == 1) {
                        {
                            std::lock_guard<std::mutex> send_lock(load_send_queue_mutex_);
                            load_send_queue_.push({0, 0});
                        }
                        load_send_queue_cv_.notify_one();
                    }
                    
                    // rank_in_group 2/3: use load_recv_queue_ (receive encoding from rank_in_group 0/1)
                    if (rank_in_group_ == 2 || rank_in_group_ == 3) {
                        {
                            std::lock_guard<std::mutex> recv_lock(load_recv_queue_mutex_);
                            load_recv_queue_.push({0, 0, 0});
                        }
                        load_recv_queue_cv_.notify_one();
                    }
                    
                    // All ranks use load_xor_queue_ (rank2/3 do XOR for recovery)
                    {
                        std::lock_guard<std::mutex> xor_lock(load_xor_queue_mutex_);
                        load_xor_queue_.push({0, 0, 0, 0, 0, 0});
                    }
                    load_xor_queue_cv_.notify_one();
                    
                    // rank0/3: use load_p2p_send_queue_ (Step2 and Step6 P2P send)
                    if (rank_in_group_ == 0 || rank_in_group_ == 3) {
                        {
                            std::lock_guard<std::mutex> p2p_send_lock(load_p2p_send_queue_mutex_);
                            load_p2p_send_queue_.push({0, 0, 0, 0, 0, true, false, 0, 0, 0});
                        }
                        load_p2p_send_queue_cv_.notify_one();
                    }
                    
                    // rank1/2: use load_p2p_recv_queue_ (Step2 and Step6 P2P recv)
                    if (rank_in_group_ == 1 || rank_in_group_ == 2) {
                        {
                            std::lock_guard<std::mutex> p2p_recv_lock(load_p2p_recv_queue_mutex_);
                            load_p2p_recv_queue_.push({0, 0, true, false, 0, false, 0, 0});
                        }
                        load_p2p_recv_queue_cv_.notify_one();
                    }
                    
                    load_encoding_sentinel_received_ = false;
                    // std::cout << "EC-CHECK: [Rank " << rank_ 
                    //           << "] Load encoder: All tasks completed, sentinel sent to downstream workers" << std::endl;
                }
            }
        }
        if (encode_have_chunk) {
            const uint64_t e2e_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(encode_last_end - encode_first_start).count());
            load_encode_e2e_wall_ns_.store(e2e_ns, std::memory_order_relaxed);
            load_encode_e2e_wall_valid_.store(true, std::memory_order_relaxed);
            encode_have_chunk = false;
        }
    }
    
    // 提交 load encoding 任务到 load worker（内部辅助函数）
    void submit_load_encoding_task(const LoadEncodingTask& task) {
        {
            std::lock_guard<std::mutex> lock(load_encoding_tasks_mutex_);
            load_encoding_tasks_.push(task);
        }
        load_encoding_tasks_cv_.notify_one();
    }
    
    // Step6 P2P Send (rank3 发送 d3 给 rank2)
    void submit_load_step6_p2p_send(uintptr_t parity_addr, size_t size) {
        if (rank_ != 3) {
            std::cerr << "EC-CHECK: [Rank " << rank_ 
                      << "] submit_load_step6_p2p_send called but rank is not 3" << std::endl;
            return;
        }
        
        {
            std::lock_guard<std::mutex> lock(load_step6_p2p_send_queue_mutex_);
            load_step6_p2p_send_queue_.push({
                parity_addr,              // send_buffer_addr (d3)
                0,                        // p2p_own_write_addr (Step6 不需要)
                size,
                parity_addr,              // parity_addr (用于发送后释放)
                0,                        // data_addr (Step6 不需要)
                true,                     // is_load_mode_transfer
                true,                     // is_step6_transfer
                0,                        // load_mode_data_addr (Step6 不需要)
                0,
                0
            });
        }
        load_step6_p2p_send_queue_cv_.notify_one();
    }
    
    // Step6 P2P Recv (rank2 接收 d3)
    void submit_load_step6_p2p_recv(uintptr_t partner_buffer_addr, size_t size) {
        if (rank_ != 2) {
            std::cerr << "EC-CHECK: [Rank " << rank_ 
                      << "] submit_load_step6_p2p_recv called but rank is not 2" << std::endl;
            return;
        }
        
        {
            std::lock_guard<std::mutex> lock(load_step6_p2p_recv_queue_mutex_);
            load_step6_p2p_recv_queue_.push({
                partner_buffer_addr,      // recv_buffer_addr (partner_buffer)
                size,
                true,                     // is_load_mode_transfer
                true,                     // is_step6_transfer
                0,                        // data_buffer_addr (Step6 不需要)
                false,
                0,
                0
            });
        }
        load_step6_p2p_recv_queue_cv_.notify_one();
    }
    
    // ========== Load Mode 独立的 Worker 实现 ==========
    
    void touch_load_recv_pipeline_e2e_wall_(
        std::chrono::steady_clock::time_point t0,
        std::chrono::steady_clock::time_point t1) {
        std::lock_guard<std::mutex> lk(load_recv_pipeline_e2e_mu_);
        if (!load_recv_pipeline_e2e_have_any_) {
            load_recv_pipeline_e2e_first_ = t0;
            load_recv_pipeline_e2e_last_ = t1;
            load_recv_pipeline_e2e_have_any_ = true;
        } else {
            if (t0 < load_recv_pipeline_e2e_first_) {
                load_recv_pipeline_e2e_first_ = t0;
            }
            if (t1 > load_recv_pipeline_e2e_last_) {
                load_recv_pipeline_e2e_last_ = t1;
            }
        }
    }

    void record_load_net_ns_(
        std::atomic<uint64_t>& total_ns,
        std::atomic<size_t>& task_count,
        std::chrono::steady_clock::time_point t0,
        bool update_recv_pipeline_e2e_wall = false,
        const char* trace_label = nullptr,
        size_t trace_bytes = 0) {
        const auto t1 = std::chrono::steady_clock::now();
        touch_load_net_wall_(t0, t1);
        if (update_recv_pipeline_e2e_wall) {
            touch_load_recv_pipeline_e2e_wall_(t0, t1);
        }
        const uint64_t ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        total_ns.fetch_add(ns, std::memory_order_relaxed);
        task_count.fetch_add(1, std::memory_order_relaxed);
    }

    // Load Send Worker - rank0/1 发送 encoding 给 rank2/3
    void load_send_worker() {
        
        // Wait for ASIO initialization
        if (use_asio_) {
            while (!should_stop_threads_ && (!asio_initialized_ || !asio_conn_mgr_.is_xor_send_connected())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        while (!should_stop_threads_) {
            LoadSendTask task;
            
            {
                std::unique_lock<std::mutex> lock(load_send_queue_mutex_);
                // FIX: Remove sentinel check from wait condition, match save mode behavior
                load_send_queue_cv_.wait(lock, [this] {
                    return !load_send_queue_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && load_send_queue_.empty()) {
                    break;
                }
                
                task = load_send_queue_.front();
                load_send_queue_.pop();
            }
            
            // Check for sentinel
            bool is_sentinel = (task.encoding_addr == 0 && task.size == 0);
            if (is_sentinel) {
                load_send_worker_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
                    if (load_send_queue_.empty()) {
                        load_send_worker_completed_ = true;
                        load_send_worker_sentinel_received_ = false;
                        break;  // Exit loop when sentinel received and queue is empty
                    }
                }
                continue;  // Continue processing remaining tasks if queue is not empty
            }
            
            // Send encoding using RDMA or ASIO (load mode)
#ifdef __linux__
            if (use_rdma_ && rdma_xor_qp_) {
                // std::cout << "[EC-CHECK RDMA] Load_XOR_Send: Sending " << task.size << " bytes via RDMA" << std::endl;
                const auto t_net = std::chrono::steady_clock::now();
                try {
                    rdma_send_data_via_qp(rdma_xor_send_qp_, rdma_xor_send_cq_, get_rdma_xor_send_control_sock(),
                        rdma_xor_send_control_mutex_, reinterpret_cast<const uint8_t*>(task.encoding_addr), task.size);
                    record_load_net_ns_(load_enc_xor_send_total_ns_, load_enc_xor_send_task_count_, t_net, false,
                                        "enc_xor_send", task.size);
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.encoding_addr);
                } catch (const std::exception& e) {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] Load RDMA XOR send failed: " << e.what() << std::endl;
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.encoding_addr);
                }
            } else
#endif
            if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_xor_send_connected()) {
                uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(task.encoding_addr);
                uint32_t size_net = htonl(static_cast<uint32_t>(task.size));
                
                try {
                    const auto t_net = std::chrono::steady_clock::now();
                    // Send message header (size) first
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load send: About to send header (size=" << task.size << ")" << std::endl;
                    boost::asio::write(
                        asio_conn_mgr_.get_xor_send_socket(),
                        boost::asio::buffer(&size_net, sizeof(uint32_t))
                    );
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load send: Header sent successfully" << std::endl;
                    
                    // Send data
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load send: About to send data (size=" << task.size << ")" << std::endl;
                    boost::asio::write(
                        asio_conn_mgr_.get_xor_send_socket(),
                        boost::asio::buffer(buffer_ptr, task.size)
                    );
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load send: Data sent successfully" << std::endl;
                    record_load_net_ns_(load_enc_xor_send_total_ns_, load_enc_xor_send_task_count_, t_net, false,
                                        "enc_xor_send", task.size);
                    // Send completed successfully, release buffer
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.encoding_addr);
                } catch (const boost::system::system_error& e) {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] Load ASIO send failed: " << e.what() << std::endl;
                    // Release buffer even on error to avoid memory leak
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.encoding_addr);
                }
            } else {
                // ASIO not available - log warning but continue
                std::cerr << "EC-CHECK: [Rank " << rank_ 
                          << "] WARNING: ASIO not available for load send, skipping task" << std::endl;
                // Release buffer even if ASIO is not available
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                encoding_buffers_to_release_.push(task.encoding_addr);
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (load_send_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(load_send_queue_mutex_);
                if (load_send_queue_.empty()) {
                    load_send_worker_completed_ = true;
                    load_send_worker_sentinel_received_ = false;
                    break;  // Exit loop when all tasks are processed
                }
            }
        }
        
    }
    
    // Load Recv Worker - rank2/3 接收 encoding 从 rank0/1
    void load_recv_worker() {
        
        // Wait for ASIO initialization
        if (use_asio_) {
            while (!should_stop_threads_ && (!asio_initialized_ || !asio_conn_mgr_.is_xor_recv_connected())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        while (!should_stop_threads_) {
            LoadRecvTask task;
            
            {
                std::unique_lock<std::mutex> lock(load_recv_queue_mutex_);
                load_recv_queue_cv_.wait(lock, [this] {
                    return !load_recv_queue_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && load_recv_queue_.empty()) {
                    break;
                }
                
                if (load_recv_queue_.empty()) {
                    continue;
                }
                
                task = load_recv_queue_.front();
                load_recv_queue_.pop();
            }
            
            bool is_sentinel = (task.recv_addr == 0 && task.size == 0 && task.parity_addr == 0);
            if (is_sentinel) {
                load_recv_worker_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(load_recv_queue_mutex_);
                    if (load_recv_queue_.empty()) {
                        load_recv_worker_completed_ = true;
                        load_recv_worker_sentinel_received_ = false;
                        break;
                    }
                }
                continue;
            }
            
            // Receive encoding using RDMA or ASIO (load mode)
#ifdef __linux__
            if (use_rdma_ && rdma_xor_qp_) {
                // std::cout << "[EC-CHECK RDMA] Load_XOR_Recv: Receiving " << task.size << " bytes via RDMA" << std::endl;
                const auto t_net = std::chrono::steady_clock::now();
                try {
                    size_t recv_size = rdma_receive_data_via_qp(rdma_xor_recv_qp_, rdma_xor_recv_cq_,
                        get_rdma_xor_recv_control_sock(), rdma_xor_recv_control_mutex_,
                        reinterpret_cast<uint8_t*>(task.recv_addr), task.size,
                        RdmaLoadRecvPollLane::EncXor);
                    if (recv_size != task.size) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] Load RDMA XOR recv size mismatch: expected "
                                  << task.size << ", got " << recv_size << std::endl;
                        continue;
                    }
                    record_load_net_ns_(load_enc_xor_recv_total_ns_, load_enc_xor_recv_task_count_, t_net, true,
                                        "enc_xor_recv", task.size);
                } catch (const std::exception& e) {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] Load RDMA XOR recv failed: " << e.what() << std::endl;
                    continue;
                }
            } else
#endif
            if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_xor_recv_connected()) {
                uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(task.recv_addr);
                uint32_t size_net;
                
                try {
                    const auto t_net = std::chrono::steady_clock::now();
                    // Receive message header (size) first
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load recv: About to receive header (expected_size=" << task.size << ")" << std::endl;
                    boost::asio::read(
                        asio_conn_mgr_.get_xor_recv_socket(),
                        boost::asio::buffer(&size_net, sizeof(uint32_t))
                    );
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load recv: Header received successfully" << std::endl;
                    
                    uint32_t size = ntohl(size_net);
                    if (size != task.size) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ 
                                  << "] Load size mismatch: expected " << task.size 
                                  << ", got " << size << std::endl;
                        continue;  // Skip this task
                    }
                    
                    // Receive data
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load recv: About to receive data (size=" << size << ")" << std::endl;
                    boost::asio::read(
                        asio_conn_mgr_.get_xor_recv_socket(),
                        boost::asio::buffer(buffer_ptr, size)
                    );
                    // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load recv: Data received successfully" << std::endl;
                    record_load_net_ns_(load_enc_xor_recv_total_ns_, load_enc_xor_recv_task_count_, t_net, true,
                                        "enc_xor_recv", task.size);
                    // Receive completed successfully, continue with XOR processing below
                } catch (const boost::system::system_error& e) {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] Load ASIO recv failed: " << e.what() << std::endl;
                    continue;  // Skip XOR processing on error
                }
            }
            
            // Match with local encoding and submit XOR task
            uintptr_t local_encoding_addr = 0;
            {
                std::lock_guard<std::mutex> lock(load_pending_xor_mutex_);
                auto it = load_pending_xor_encoding_.find(task.recv_addr);
                if (it != load_pending_xor_encoding_.end()) {
                    local_encoding_addr = it->second;
                    load_pending_xor_encoding_.erase(it);
                }
            }
            
            if (local_encoding_addr != 0 && task.parity_addr != 0) {
                uintptr_t p2p_partner_write_addr = 0;
                {
                    std::lock_guard<std::mutex> lock(load_recv_to_p2p_mutex_);
                    auto it = load_recv_to_p2p_partner_write_.find(task.recv_addr);
                    if (it != load_recv_to_p2p_partner_write_.end()) {
                        p2p_partner_write_addr = it->second;
                    }
                }
                
                uintptr_t data_addr = 0;
                {
                    std::lock_guard<std::mutex> lock(load_recv_to_data_mutex_);
                    auto it = load_recv_to_data_.find(task.recv_addr);
                    if (it != load_recv_to_data_.end()) {
                        data_addr = it->second;
                    }
                }
                
                {
                    std::lock_guard<std::mutex> lock(load_xor_queue_mutex_);
                    load_xor_queue_.push({
                        local_encoding_addr,
                        task.recv_addr,
                        task.parity_addr,
                        task.size,
                        p2p_partner_write_addr,
                        data_addr
                    });
                }
                load_xor_queue_cv_.notify_one();
            }
            
            // Check if sentinel received and queue is empty
            if (load_recv_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(load_recv_queue_mutex_);
                if (load_recv_queue_.empty()) {
                    load_recv_worker_completed_ = true;
                    load_recv_worker_sentinel_received_ = false;
                    break;
                }
            }
        }
    }

    static std::array<int, kLoadXorPoolSize> parse_load_xor_pool_cpus_or_throw() {
        std::array<int, kLoadXorPoolSize> cpus{};
        const char* env = std::getenv("ECCHECK_XOR_CPU_LIST");
        if (!env || !*env) {
            for (int i = 0; i < kLoadXorPoolSize; ++i) {
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
                throw std::runtime_error("ECCHECK_XOR_CPU_LIST: invalid CPU id token");
            }
            parsed.push_back(static_cast<int>(v));
            p = end;
        }
        if (parsed.size() != static_cast<size_t>(kLoadXorPoolSize)) {
            throw std::runtime_error(
                "ECCHECK_XOR_CPU_LIST must contain exactly 16 comma-separated CPU ids "
                "(or unset to use 0..15)");
        }
        for (size_t i = 0; i < cpus.size(); ++i) {
            cpus[i] = parsed[i];
        }
        return cpus;
    }

    void load_xor_pool_init() {
        if (load_xor_pool_inited_.load(std::memory_order_acquire)) {
            return;
        }
        load_xor_pool_cpus_ = parse_load_xor_pool_cpus_or_throw();
        load_xor_pool_stop_.store(false, std::memory_order_release);
        load_xor_pool_epoch_.store(0, std::memory_order_release);
        load_xor_pool_remaining_.store(0, std::memory_order_release);
        for (auto& e : load_xor_pool_last_epoch_) {
            e = 0;
        }
        for (int i = 0; i < kLoadXorPoolSize; ++i) {
            load_xor_pool_ctx_[static_cast<size_t>(i)].self = this;
            load_xor_pool_ctx_[static_cast<size_t>(i)].wid = i;
            int rc = pthread_create(
                &load_xor_pool_threads_[static_cast<size_t>(i)],
                nullptr,
                &ECCHECKNative::load_xor_pool_pthread_entry,
                &load_xor_pool_ctx_[static_cast<size_t>(i)]);
            if (rc != 0) {
                load_xor_pool_stop_.store(true, std::memory_order_release);
                load_xor_pool_worker_cv_.notify_all();
                for (int j = 0; j < i; ++j) {
                    pthread_join(load_xor_pool_threads_[static_cast<size_t>(j)], nullptr);
                }
                throw std::runtime_error(
                    std::string("EC-CHECK: pthread_create for load XOR pool failed: ") + std::strerror(rc));
            }
        }
        load_xor_pool_inited_.store(true, std::memory_order_release);
    }

    void load_xor_pool_shutdown() {
        if (!load_xor_pool_inited_.load(std::memory_order_acquire)) {
            return;
        }
        load_xor_pool_stop_.store(true, std::memory_order_release);
        load_xor_pool_worker_cv_.notify_all();
        for (int i = 0; i < kLoadXorPoolSize; ++i) {
            pthread_join(load_xor_pool_threads_[static_cast<size_t>(i)], nullptr);
        }
        load_xor_pool_stop_.store(false, std::memory_order_release);
        load_xor_pool_inited_.store(false, std::memory_order_release);
    }

    static void* load_xor_pool_pthread_entry(void* arg) {
        auto* ctx = static_cast<EccheckXorPoolWorkerCtx*>(arg);
        ctx->self->load_xor_pool_worker_loop(ctx->wid);
        return nullptr;
    }

    void load_xor_pool_worker_loop(int wid) {
        const int cpu = load_xor_pool_cpus_[static_cast<size_t>(wid)];
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (cpu >= 0 && static_cast<unsigned>(cpu) < CPU_SETSIZE) {
            CPU_SET(static_cast<unsigned>(cpu), &cpuset);
            int af = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
            if (af != 0) {
                std::cerr << "EC-CHECK: load_xor_pool worker " << wid
                          << " pthread_setaffinity_np failed: " << af << std::endl;
            }
        } else {
            std::cerr << "EC-CHECK: load_xor_pool worker " << wid << " CPU id " << cpu
                      << " invalid or >= CPU_SETSIZE, skipping affinity" << std::endl;
        }

        while (true) {
            std::unique_lock<std::mutex> lk(load_xor_pool_mutex_);
            load_xor_pool_worker_cv_.wait(lk, [&] {
                return load_xor_pool_stop_.load(std::memory_order_acquire) ||
                       (load_xor_pool_last_epoch_[static_cast<size_t>(wid)] <
                        load_xor_pool_epoch_.load(std::memory_order_acquire));
            });
            if (load_xor_pool_stop_.load(std::memory_order_acquire)) {
                break;
            }
            uint64_t e = load_xor_pool_epoch_.load(std::memory_order_acquire);
            LoadXorPoolJob local_copy = load_xor_pool_shared_job_;
            lk.unlock();

            load_xor_pool_execute_stripe_from_job(local_copy, wid);

            {
                std::lock_guard<std::mutex> guard(load_xor_pool_mutex_);
                load_xor_pool_last_epoch_[static_cast<size_t>(wid)] = e;
            }

            const int left =
                load_xor_pool_remaining_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            if (left == 0) {
                load_xor_pool_coordinator_cv_.notify_one();
            }
        }
    }

    void load_xor_pool_execute_stripe_from_job(const LoadXorPoolJob& job, int wid) {
        const size_t total = job.size;
        const size_t base = total / static_cast<size_t>(kLoadXorPoolSize);
        const size_t rem = total % static_cast<size_t>(kLoadXorPoolSize);
        size_t off;
        size_t len;
        if (wid < kLoadXorPoolSize - 1) {
            off = static_cast<size_t>(wid) * base;
            len = base;
        } else {
            off = static_cast<size_t>(kLoadXorPoolSize - 1) * base;
            len = base + rem;
        }
        if (len == 0) {
            return;
        }
        auto at = [](uintptr_t base_ptr, size_t o) -> unsigned char* {
            return reinterpret_cast<unsigned char*>(base_ptr + o);
        };
        unsigned char* p0 = at(job.local_encoding_addr, off);
        unsigned char* p1 = at(job.remote_encoding_addr, off);
        unsigned char* pd = at(job.parity_addr, off);
        void* xor_array[3] = {p0, p1, pd};
        xor_gen(3, static_cast<int>(len), xor_array);
    }

    void load_xor_pool_run_parallel_xor(const LoadXORTask& task) {
        {
            std::lock_guard<std::mutex> publish(load_xor_pool_mutex_);
            if (should_stop_threads_.load(std::memory_order_acquire)) {
                return;
            }
            load_xor_pool_shared_job_.local_encoding_addr = task.local_encoding_addr;
            load_xor_pool_shared_job_.remote_encoding_addr = task.remote_encoding_addr;
            load_xor_pool_shared_job_.parity_addr = task.parity_addr;
            load_xor_pool_shared_job_.size = task.size;
            load_xor_pool_epoch_.fetch_add(1, std::memory_order_acq_rel);
            load_xor_pool_remaining_.store(kLoadXorPoolSize, std::memory_order_release);
        }
        load_xor_pool_worker_cv_.notify_all();
        std::unique_lock<std::mutex> lk(load_xor_pool_mutex_);
        load_xor_pool_coordinator_cv_.wait(lk, [&] {
            return load_xor_pool_remaining_.load(std::memory_order_acquire) == 0 ||
                   should_stop_threads_.load(std::memory_order_acquire);
        });
    }
    

    // ========== Save-path Encode Pool (ec_encode_data) ==========

    void ec_rs_encode_pool_init() {
        if (ec_rs_encode_pool_inited_.load(std::memory_order_acquire)) return;
        ec_rs_encode_pool_cpus_ = parse_load_encode_pool_cpus_or_throw();
        ec_rs_encode_pool_stop_.store(false, std::memory_order_release);
        ec_rs_encode_pool_epoch_.store(0, std::memory_order_release);
        ec_rs_encode_pool_remaining_.store(0, std::memory_order_release);
        for (auto& e : ec_rs_encode_pool_last_epoch_) e = 0;
        for (int i = 0; i < kEcRsEncodePoolSize; ++i) {
            ec_rs_encode_pool_ctx_[static_cast<size_t>(i)].self = this;
            ec_rs_encode_pool_ctx_[static_cast<size_t>(i)].wid = i;
            int rc = pthread_create(&ec_rs_encode_pool_threads_[static_cast<size_t>(i)], nullptr,
                                    &ECCHECKNative::ec_rs_encode_pool_pthread_entry,
                                    &ec_rs_encode_pool_ctx_[static_cast<size_t>(i)]);
            if (rc != 0) {
                ec_rs_encode_pool_stop_.store(true, std::memory_order_release);
                ec_rs_encode_pool_worker_cv_.notify_all();
                for (int j = 0; j < i; ++j)
                    pthread_join(ec_rs_encode_pool_threads_[static_cast<size_t>(j)], nullptr);
                throw std::runtime_error("EC-CHECK: pthread_create for encode pool failed: " +
                                         std::string(std::strerror(rc)));
            }
        }
        ec_rs_encode_pool_inited_.store(true, std::memory_order_release);
    }

    void ec_rs_encode_pool_shutdown() {
        if (!ec_rs_encode_pool_inited_.load(std::memory_order_acquire)) return;
        ec_rs_encode_pool_stop_.store(true, std::memory_order_release);
        ec_rs_encode_pool_worker_cv_.notify_all();
        for (int i = 0; i < kEcRsEncodePoolSize; ++i)
            pthread_join(ec_rs_encode_pool_threads_[static_cast<size_t>(i)], nullptr);
        ec_rs_encode_pool_stop_.store(false, std::memory_order_release);
        ec_rs_encode_pool_inited_.store(false, std::memory_order_release);
    }

    static void* ec_rs_encode_pool_pthread_entry(void* arg) {
        auto* ctx = static_cast<EcRsEncodePoolWorkerCtx*>(arg);
        ctx->self->ec_rs_encode_pool_worker_loop(ctx->wid);
        return nullptr;
    }

    void ec_rs_encode_pool_execute_chunk(const EcRsEncodeJob& job, int wid) {
        const size_t total = job.size;
        const size_t base = total / static_cast<size_t>(kEcRsEncodePoolSize);
        const size_t rem = total % static_cast<size_t>(kEcRsEncodePoolSize);
        size_t off, len;
        if (wid < kEcRsEncodePoolSize - 1) {
            off = static_cast<size_t>(wid) * base;
            len = base;
        } else {
            off = static_cast<size_t>(kEcRsEncodePoolSize - 1) * base;
            len = base + rem;
        }
        if (len == 0) return;
        unsigned char* d = reinterpret_cast<unsigned char*>(job.data_addr + off);
        unsigned char* e = reinterpret_cast<unsigned char*>(job.encoding_addr + off);
        unsigned char* srcs[1] = {d};
        unsigned char* dests[1] = {e};
        ec_encode_data(static_cast<int>(len), 1, 1, job.gftbls_ptr, srcs, dests);
    }

    void ec_rs_encode_pool_worker_loop(int wid) {
        const int cpu = ec_rs_encode_pool_cpus_[static_cast<size_t>(wid)];
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (cpu >= 0 && static_cast<unsigned>(cpu) < CPU_SETSIZE) {
            CPU_SET(static_cast<unsigned>(cpu), &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        }
        while (true) {
            std::unique_lock<std::mutex> lk(ec_rs_encode_pool_mutex_);
            ec_rs_encode_pool_worker_cv_.wait(lk, [&] {
                return ec_rs_encode_pool_stop_.load(std::memory_order_acquire) ||
                       (ec_rs_encode_pool_last_epoch_[static_cast<size_t>(wid)] <
                        ec_rs_encode_pool_epoch_.load(std::memory_order_acquire));
            });
            if (ec_rs_encode_pool_stop_.load(std::memory_order_acquire)) break;
            uint64_t e = ec_rs_encode_pool_epoch_.load(std::memory_order_acquire);
            EcRsEncodeJob local_copy = ec_rs_encode_pool_shared_job_;
            lk.unlock();
            ec_rs_encode_pool_execute_chunk(local_copy, wid);
            {
                std::lock_guard<std::mutex> guard(ec_rs_encode_pool_mutex_);
                ec_rs_encode_pool_last_epoch_[static_cast<size_t>(wid)] = e;
            }
            int left = ec_rs_encode_pool_remaining_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            if (left == 0) ec_rs_encode_pool_coordinator_cv_.notify_all();
        }
    }

    void ec_rs_encode_pool_run_parallel(uintptr_t data_addr, uintptr_t encoding_addr,
                                        size_t size, unsigned char* gftbls_ptr) {
        {
            std::lock_guard<std::mutex> publish(ec_rs_encode_pool_mutex_);
            if (should_stop_threads_.load(std::memory_order_acquire)) return;
            ec_rs_encode_pool_shared_job_ = EcRsEncodeJob{data_addr, encoding_addr, size, gftbls_ptr};
            ec_rs_encode_pool_epoch_.fetch_add(1, std::memory_order_acq_rel);
            ec_rs_encode_pool_remaining_.store(kEcRsEncodePoolSize, std::memory_order_release);
        }
        ec_rs_encode_pool_worker_cv_.notify_all();
        std::unique_lock<std::mutex> lk(ec_rs_encode_pool_mutex_);
        ec_rs_encode_pool_coordinator_cv_.wait(lk, [&] {
            return ec_rs_encode_pool_remaining_.load(std::memory_order_acquire) == 0 ||
                   should_stop_threads_.load(std::memory_order_acquire);
        });
    }

    // ========== Save-path XOR Pool (xor_gen) =======================

    void ec_xor_pool_init() {
        if (ec_xor_pool_inited_.load(std::memory_order_acquire)) return;
        ec_xor_pool_cpus_ = parse_load_xor_pool_cpus_or_throw();
        ec_xor_pool_stop_.store(false, std::memory_order_release);
        ec_xor_pool_epoch_.store(0, std::memory_order_release);
        ec_xor_pool_remaining_.store(0, std::memory_order_release);
        for (auto& e : ec_xor_pool_last_epoch_) e = 0;
        for (int i = 0; i < kEcXorPoolSize; ++i) {
            ec_xor_pool_ctx_[static_cast<size_t>(i)].self = this;
            ec_xor_pool_ctx_[static_cast<size_t>(i)].wid = i;
            int rc = pthread_create(&ec_xor_pool_threads_[static_cast<size_t>(i)], nullptr,
                                    &ECCHECKNative::ec_xor_pool_pthread_entry,
                                    &ec_xor_pool_ctx_[static_cast<size_t>(i)]);
            if (rc != 0) {
                ec_xor_pool_stop_.store(true, std::memory_order_release);
                ec_xor_pool_worker_cv_.notify_all();
                for (int j = 0; j < i; ++j)
                    pthread_join(ec_xor_pool_threads_[static_cast<size_t>(j)], nullptr);
                throw std::runtime_error("EC-CHECK: pthread_create for XOR pool failed: " +
                                         std::string(std::strerror(rc)));
            }
        }
        ec_xor_pool_inited_.store(true, std::memory_order_release);
    }

    void ec_xor_pool_shutdown() {
        if (!ec_xor_pool_inited_.load(std::memory_order_acquire)) return;
        ec_xor_pool_stop_.store(true, std::memory_order_release);
        ec_xor_pool_worker_cv_.notify_all();
        for (int i = 0; i < kEcXorPoolSize; ++i)
            pthread_join(ec_xor_pool_threads_[static_cast<size_t>(i)], nullptr);
        ec_xor_pool_stop_.store(false, std::memory_order_release);
        ec_xor_pool_inited_.store(false, std::memory_order_release);
    }

    static void* ec_xor_pool_pthread_entry(void* arg) {
        auto* ctx = static_cast<EcXorPoolWorkerCtx*>(arg);
        ctx->self->ec_xor_pool_worker_loop(ctx->wid);
        return nullptr;
    }

    void ec_xor_pool_execute_chunk(const EcXorJob& job, int wid) {
        const size_t total = job.size;
        const size_t base = total / static_cast<size_t>(kEcXorPoolSize);
        const size_t rem = total % static_cast<size_t>(kEcXorPoolSize);
        size_t off, len;
        if (wid < kEcXorPoolSize - 1) {
            off = static_cast<size_t>(wid) * base;
            len = base;
        } else {
            off = static_cast<size_t>(kEcXorPoolSize - 1) * base;
            len = base + rem;
        }
        if (len == 0) return;
        void* xor_array[3] = {
            reinterpret_cast<void*>(job.dst + off),
            reinterpret_cast<void*>(job.src1 + off),
            reinterpret_cast<void*>(job.src2 + off),
        };
        xor_gen(3, static_cast<int>(len), xor_array);
    }

    void ec_xor_pool_worker_loop(int wid) {
        const int cpu = ec_xor_pool_cpus_[static_cast<size_t>(wid)];
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (cpu >= 0 && static_cast<unsigned>(cpu) < CPU_SETSIZE) {
            CPU_SET(static_cast<unsigned>(cpu), &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        }
        while (true) {
            std::unique_lock<std::mutex> lk(ec_xor_pool_mutex_);
            ec_xor_pool_worker_cv_.wait(lk, [&] {
                return ec_xor_pool_stop_.load(std::memory_order_acquire) ||
                       (ec_xor_pool_last_epoch_[static_cast<size_t>(wid)] <
                        ec_xor_pool_epoch_.load(std::memory_order_acquire));
            });
            if (ec_xor_pool_stop_.load(std::memory_order_acquire)) break;
            uint64_t e = ec_xor_pool_epoch_.load(std::memory_order_acquire);
            EcXorJob local_copy = ec_xor_pool_shared_job_;
            lk.unlock();
            ec_xor_pool_execute_chunk(local_copy, wid);
            {
                std::lock_guard<std::mutex> guard(ec_xor_pool_mutex_);
                ec_xor_pool_last_epoch_[static_cast<size_t>(wid)] = e;
            }
            int left = ec_xor_pool_remaining_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            if (left == 0) ec_xor_pool_coordinator_cv_.notify_all();
        }
    }

    void ec_xor_pool_run_parallel(uintptr_t dst, uintptr_t src1, uintptr_t src2, size_t size) {
        {
            std::lock_guard<std::mutex> publish(ec_xor_pool_mutex_);
            if (should_stop_threads_.load(std::memory_order_acquire)) return;
            ec_xor_pool_shared_job_ = EcXorJob{dst, src1, src2, size};
            ec_xor_pool_epoch_.fetch_add(1, std::memory_order_acq_rel);
            ec_xor_pool_remaining_.store(kEcXorPoolSize, std::memory_order_release);
        }
        ec_xor_pool_worker_cv_.notify_all();
        std::unique_lock<std::mutex> lk(ec_xor_pool_mutex_);
        ec_xor_pool_coordinator_cv_.wait(lk, [&] {
            return ec_xor_pool_remaining_.load(std::memory_order_acquire) == 0 ||
                   should_stop_threads_.load(std::memory_order_acquire);
        });
    }
    // Load XOR Worker - 执行 XOR 操作并处理 Step6 P2P
    void load_xor_worker() {

        bool xor_have_chunk = false;
        std::chrono::steady_clock::time_point xor_first_start{};
        std::chrono::steady_clock::time_point xor_last_end{};

        // Publish per-load-batch e2e wall (first XOR chunk start -> last XOR chunk end).
        // Unlike storing only at thread exit, this matches EC-NAIVE and wait_for_encoding_completion timing.
        auto publish_load_xor_e2e_for_batch = [this](bool& have_chunk,
                                                     const std::chrono::steady_clock::time_point& first_start,
                                                     const std::chrono::steady_clock::time_point& last_end) {
            if (!have_chunk) {
                return;  // No XOR in this batch; counters stay 0; valid stays false from reset_encoding/set_load_mode.
            }
            const uint64_t e2e_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(last_end - first_start).count());
            load_xor_e2e_wall_ns_.store(e2e_ns, std::memory_order_relaxed);
            load_xor_e2e_wall_valid_.store(true, std::memory_order_relaxed);
            have_chunk = false;
        };
        
        while (!should_stop_threads_) {
            LoadXORTask task;
            
            {
                std::unique_lock<std::mutex> lock(load_xor_queue_mutex_);
                load_xor_queue_cv_.wait(lock, [this] {
                    return !load_xor_queue_.empty() || should_stop_threads_ || load_xor_worker_sentinel_received_.load();
                });
                
                if (should_stop_threads_ && load_xor_queue_.empty()) {
                    if (xor_have_chunk) {
                        const uint64_t e2e_ns = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(xor_last_end - xor_first_start)
                                .count());
                        load_xor_e2e_wall_ns_.store(e2e_ns, std::memory_order_relaxed);
                        load_xor_e2e_wall_valid_.store(true, std::memory_order_relaxed);
                        xor_have_chunk = false;
                    }
                    break;
                }
                
                if (load_xor_worker_sentinel_received_.load() && load_xor_queue_.empty()) {
                    load_xor_worker_completed_ = true;
                    load_xor_worker_sentinel_received_ = false;
                    publish_load_xor_e2e_for_batch(xor_have_chunk, xor_first_start, xor_last_end);
                    continue;
                }
                
                task = load_xor_queue_.front();
                load_xor_queue_.pop();
            }
            
            bool is_sentinel = (task.local_encoding_addr == 0 && task.remote_encoding_addr == 0 &&
                                task.parity_addr == 0 && task.size == 0 &&
                                task.p2p_partner_write_addr == 0 && task.data_addr == 0);
            if (is_sentinel) {
                load_xor_worker_sentinel_received_ = true;
                bool batch_xor_done = false;
                {
                    std::lock_guard<std::mutex> lock(load_xor_queue_mutex_);
                    if (load_xor_queue_.empty()) {
                        load_xor_worker_completed_ = true;
                        load_xor_worker_sentinel_received_ = false;
                        batch_xor_done = true;
                    }
                }
                if (batch_xor_done) {
                    publish_load_xor_e2e_for_batch(xor_have_chunk, xor_first_start, xor_last_end);
                }
                continue;
            }
            
            // Perform XOR (rank 2/3: striped parallel via pthread pool when initialized)
            if (!xor_have_chunk) {
                xor_first_start = std::chrono::steady_clock::now();
                xor_have_chunk = true;
            }
            const auto xor_chunk_t0 = std::chrono::steady_clock::now();
            if (load_xor_pool_inited_.load(std::memory_order_acquire)) {
                load_xor_pool_run_parallel_xor(task);
            } else {
                unsigned char* srcs[2];
                srcs[0] = reinterpret_cast<unsigned char*>(task.local_encoding_addr);
                srcs[1] = reinterpret_cast<unsigned char*>(task.remote_encoding_addr);
                unsigned char* dest = reinterpret_cast<unsigned char*>(task.parity_addr);
                void* xor_array[3];
                xor_array[0] = srcs[0];
                xor_array[1] = srcs[1];
                xor_array[2] = dest;
                xor_gen(3, static_cast<int>(task.size), xor_array);
            }
            const auto xor_chunk_t1 = std::chrono::steady_clock::now();
            xor_last_end = xor_chunk_t1;
            const uint64_t xor_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(xor_chunk_t1 - xor_chunk_t0).count());
            record_load_xor_op_(xor_ns);
            
            // FIX: Release encoding buffers after XOR completes (matches save mode behavior)
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                // Always release local encoding buffer
                if (task.local_encoding_addr != 0) {
                    encoding_buffers_to_release_.push(task.local_encoding_addr);
                }
                
                // Release remote encoding buffer (received encoding)
                if (task.remote_encoding_addr != 0) {
                    encoding_buffers_to_release_.push(task.remote_encoding_addr);
                }
            }
            
            // Release parity buffer after XOR (single-node recovery)
            // Must release here to avoid pool exhaustion with large models (>24 chunks)
            if (task.parity_addr != 0) {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                parity_buffers_to_release_.push(task.parity_addr);
            }
            
            if (load_xor_worker_sentinel_received_.load()) {
                bool batch_xor_done = false;
                {
                    std::lock_guard<std::mutex> lock(load_xor_queue_mutex_);
                    if (load_xor_queue_.empty()) {
                        load_xor_worker_completed_ = true;
                        load_xor_worker_sentinel_received_ = false;
                        batch_xor_done = true;
                        // Note: Step6 P2P sentinels are sent from Python after all Step6 tasks are submitted
                    }
                }
                if (batch_xor_done) {
                    publish_load_xor_e2e_for_batch(xor_have_chunk, xor_first_start, xor_last_end);
                }
            }
        }

        // Thread shutdown: only publish if a batch was in progress (avoid clearing last batch e2e).
        if (xor_have_chunk) {
            const uint64_t e2e_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(xor_last_end - xor_first_start).count());
            load_xor_e2e_wall_ns_.store(e2e_ns, std::memory_order_relaxed);
            load_xor_e2e_wall_valid_.store(true, std::memory_order_relaxed);
            xor_have_chunk = false;
        }
    }
    
    // Load P2P Send Worker - dedicated for load mode P2P send operations (uses ASIO only)
    void load_p2p_send_worker() {
        
        while (!should_stop_threads_) {
            P2PSendTask task;
            
            {
                std::unique_lock<std::mutex> lock(load_p2p_send_queue_mutex_);
                load_p2p_send_queue_cv_.wait(lock, [this] {
                    return !load_p2p_send_queue_.empty() || should_stop_threads_ || load_p2p_send_worker_sentinel_received_.load();
                });
                
                if (should_stop_threads_ && load_p2p_send_queue_.empty()) {
                    break;
                }
                
                if (load_p2p_send_worker_sentinel_received_.load() && load_p2p_send_queue_.empty()) {
                    load_p2p_send_worker_completed_ = true;
                    load_p2p_send_worker_sentinel_received_ = false;
                    continue;
                }
                
                task = load_p2p_send_queue_.front();
                load_p2p_send_queue_.pop();
            }
            
            // Check for sentinel
            if (task.send_buffer_addr == 0 && task.p2p_own_write_addr == 0 && task.size == 0) {
                load_p2p_send_worker_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(load_p2p_send_queue_mutex_);
                    if (load_p2p_send_queue_.empty()) {
                        load_p2p_send_worker_completed_ = true;
                        load_p2p_send_worker_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;
            }
            
            // Send data using RDMA or ASIO (load mode)
            if (p2p_partner_rank_ >= 0 && task.size > 0 && task.send_buffer_addr != 0) {
#ifdef __linux__
                if (use_rdma_ && rdma_p2p_send_qp_) {
                    // std::cout << "[EC-CHECK RDMA] Load_P2P_Send: Sending " << task.size << " bytes via RDMA" << std::endl;
                    const auto t_net = std::chrono::steady_clock::now();
                    try {
                        rdma_send_data_via_qp(rdma_p2p_send_qp_, rdma_p2p_send_cq_, get_rdma_p2p_send_control_sock(),
                            rdma_p2p_send_control_mutex_, reinterpret_cast<const uint8_t*>(task.send_buffer_addr), task.size);
                        record_load_net_ns_(load_step2_p2p_send_total_ns_, load_step2_p2p_send_task_count_, t_net, false,
                                            "step2_p2p_send", task.size);
                    } catch (const std::exception& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] Load RDMA P2P send failed: " << e.what() << std::endl;
                    }
                } else
#endif
                if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_p2p_send_connected()) {
                    uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(task.send_buffer_addr);
                    uint32_t size_net = htonl(static_cast<uint32_t>(task.size));
                    
                    try {
                        const auto t_net = std::chrono::steady_clock::now();
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P send: About to send header (size=" << task.size << ")" << std::endl;
                        boost::asio::write(
                            asio_conn_mgr_.get_p2p_send_socket(),
                            boost::asio::buffer(&size_net, sizeof(uint32_t))
                        );
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P send: Header sent successfully" << std::endl;
                        
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P send: About to send data (size=" << task.size << ")" << std::endl;
                        boost::asio::write(
                            asio_conn_mgr_.get_p2p_send_socket(),
                            boost::asio::buffer(buffer_ptr, task.size)
                        );
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P send: Data sent successfully" << std::endl;
                        record_load_net_ns_(load_step2_p2p_send_total_ns_, load_step2_p2p_send_task_count_, t_net, false,
                                            "step2_p2p_send", task.size);
                    } catch (const boost::system::system_error& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ 
                                  << "] Load P2P ASIO send failed: " << e.what() << std::endl;
                    }
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] WARNING: ASIO not available for load P2P send" << std::endl;
                }
            }
            
            // Step2: After P2P send completes, submit encoding task
            if (!task.is_step6_transfer && task.load_mode_data_addr != 0) {
                LoadEncodingTask load_task;
                bool found_task = false;
                {
                    std::lock_guard<std::mutex> lock(pending_load_encoding_tasks_mutex_);
                    auto it = pending_load_encoding_tasks_.find(task.load_mode_data_addr);
                    if (it != pending_load_encoding_tasks_.end()) {
                        load_task = it->second;
                        pending_load_encoding_tasks_.erase(it);
                        found_task = true;
                    }
                }
                
                if (found_task && load_task.data_addr != 0) {
                    submit_load_encoding_task(load_task);
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] WARNING: No pending load encoding task found for data_addr=" 
                              << task.load_mode_data_addr << std::endl;
                }
            }
            
            // Step6 tasks are now handled by load_step6_p2p_send_worker, not here
            
            if (load_p2p_send_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(load_p2p_send_queue_mutex_);
                if (load_p2p_send_queue_.empty()) {
                    load_p2p_send_worker_completed_ = true;
                    load_p2p_send_worker_sentinel_received_ = false;
                }
            }
        }
    }
    
    // Load P2P Recv Worker - dedicated for load mode P2P recv operations (uses ASIO only)
    void load_p2p_recv_worker() {
        
        while (!should_stop_threads_) {
            P2PRecvTask task;
            
            {
                std::unique_lock<std::mutex> lock(load_p2p_recv_queue_mutex_);
                load_p2p_recv_queue_cv_.wait(lock, [this] {
                    return !load_p2p_recv_queue_.empty() || should_stop_threads_ || load_p2p_recv_worker_sentinel_received_.load();
                });
                
                if (should_stop_threads_ && load_p2p_recv_queue_.empty()) {
                    break;
                }
                
                if (load_p2p_recv_worker_sentinel_received_.load() && load_p2p_recv_queue_.empty()) {
                    load_p2p_recv_worker_completed_ = true;
                    load_p2p_recv_worker_sentinel_received_ = false;
                    continue;
                }
                
                task = load_p2p_recv_queue_.front();
                load_p2p_recv_queue_.pop();
            }
            
            // Check for sentinel
            if (task.recv_buffer_addr == 0 && task.size == 0) {
                load_p2p_recv_worker_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(load_p2p_recv_queue_mutex_);
                    if (load_p2p_recv_queue_.empty()) {
                        load_p2p_recv_worker_completed_ = true;
                        load_p2p_recv_worker_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;
            }
            
            // Receive data using RDMA or ASIO (load mode)
            bool task_processed = false;
            if (p2p_partner_rank_ >= 0 && task.size > 0 && task.recv_buffer_addr != 0) {
#ifdef __linux__
                if (use_rdma_ && rdma_p2p_recv_qp_) {
                    // std::cout << "[EC-CHECK RDMA] Load_P2P_Recv: Receiving " << task.size << " bytes via RDMA" << std::endl;
                    const auto t_net = std::chrono::steady_clock::now();
                    try {
                        size_t recv_size = rdma_receive_data_via_qp(rdma_p2p_recv_qp_, rdma_p2p_recv_cq_,
                            get_rdma_p2p_recv_control_sock(), rdma_p2p_recv_control_mutex_,
                            reinterpret_cast<uint8_t*>(task.recv_buffer_addr), task.size,
                            RdmaLoadRecvPollLane::Step2P2p);
                        if (recv_size == task.size) {
                            task_processed = true;
                            record_load_net_ns_(load_step2_p2p_recv_total_ns_, load_step2_p2p_recv_task_count_, t_net, true,
                                                "step2_p2p_recv", task.size);
                        } else {
                            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Load P2P RDMA recv size mismatch: expected "
                                      << task.size << ", got " << recv_size << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] Load RDMA P2P recv failed: " << e.what() << std::endl;
                        continue;
                    }
                } else
#endif
                if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_p2p_recv_connected()) {
                    uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(task.recv_buffer_addr);
                    uint32_t size_net;
                    
                    try {
                        const auto t_net = std::chrono::steady_clock::now();
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P recv: About to receive header (expected_size=" << task.size << ")" << std::endl;
                        boost::asio::read(
                            asio_conn_mgr_.get_p2p_recv_socket(),
                            boost::asio::buffer(&size_net, sizeof(uint32_t))
                        );
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P recv: Header received successfully" << std::endl;
                        
                        uint32_t size = ntohl(size_net);
                        if (size != task.size) {
                            std::cerr << "EC-CHECK: [Rank " << rank_ 
                                      << "] Load P2P size mismatch: expected " << task.size 
                                      << ", got " << size << std::endl;
                            continue;
                        }
                        
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P recv: About to receive data (size=" << size << ")" << std::endl;
                        boost::asio::read(
                            asio_conn_mgr_.get_p2p_recv_socket(),
                            boost::asio::buffer(buffer_ptr, size)
                        );
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P recv: Data received successfully" << std::endl;
                        record_load_net_ns_(load_step2_p2p_recv_total_ns_, load_step2_p2p_recv_task_count_, t_net, true);
                        task_processed = true;
                    } catch (const boost::system::system_error& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ 
                                  << "] Load P2P ASIO recv failed: " << e.what() << std::endl;
                        continue;
                    }
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] WARNING: ASIO not available for load P2P recv" << std::endl;
                    task_processed = true;  // Mark as processed to avoid blocking
                }
            } else {
                task_processed = true;
            }
            
            // Step2: Handle received data and submit encoding task
            if (task_processed && !task.is_step6_transfer && task.data_buffer_addr != 0) {
                // If recv_buffer_addr != data_buffer_addr, need to copy data
                if (task.recv_buffer_addr != task.data_buffer_addr) {
#ifdef __linux__
                    if (use_rdma_ && (!rdma_range_registered(task.data_buffer_addr, task.size) ||
                                      !rdma_range_registered(task.recv_buffer_addr, task.size))) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: Invalid load mode memcpy range "
                                  << "(dst=0x" << std::hex << task.data_buffer_addr
                                  << ", src=0x" << task.recv_buffer_addr << std::dec
                                  << ", size=" << task.size << ")" << std::endl;
                        throw std::runtime_error("EC-CHECK: Invalid load mode memcpy range");
                    }
#endif
                    std::memcpy(
                        reinterpret_cast<void*>(task.data_buffer_addr),
                        reinterpret_cast<void*>(task.recv_buffer_addr),
                        task.size
                    );
                }
                
                // Submit encoding task to load worker
                LoadEncodingTask load_task;
                bool found_task = false;
                {
                    std::lock_guard<std::mutex> lock(pending_load_encoding_tasks_mutex_);
                    auto it = pending_load_encoding_tasks_.find(task.data_buffer_addr);
                    if (it != pending_load_encoding_tasks_.end()) {
                        load_task = it->second;
                        pending_load_encoding_tasks_.erase(it);
                        found_task = true;
                    }
                }
                
                if (found_task && load_task.data_addr != 0) {
                    submit_load_encoding_task(load_task);
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] WARNING: No pending load encoding task found for data_buffer=" 
                              << task.data_buffer_addr << std::endl;
                }
            }
            
            // Step6: rank2 receives d3 to partner_buffer (no additional action needed)
            // partner_buffer is managed by Python
            
            if (load_p2p_recv_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(load_p2p_recv_queue_mutex_);
                if (load_p2p_recv_queue_.empty()) {
                    load_p2p_recv_worker_completed_ = true;
                    load_p2p_recv_worker_sentinel_received_ = false;
                }
            }
        }
    }
    
    // Load Step6 P2P Send Worker - dedicated for Step6 P2P send operations (rank3 only)
    void load_step6_p2p_send_worker() {
        
        while (!should_stop_threads_) {
            P2PSendTask task;
            
            {
                std::unique_lock<std::mutex> lock(load_step6_p2p_send_queue_mutex_);
                load_step6_p2p_send_queue_cv_.wait(lock, [this] {
                    return !load_step6_p2p_send_queue_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && load_step6_p2p_send_queue_.empty()) {
                    break;
                }
                
                if (load_step6_p2p_send_queue_.empty()) {
                    continue;
                }
                
                task = load_step6_p2p_send_queue_.front();
                load_step6_p2p_send_queue_.pop();
            }
            
            // Check for sentinel
            if (task.send_buffer_addr == 0 && task.p2p_own_write_addr == 0 && task.size == 0) {
                load_step6_p2p_send_worker_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(load_step6_p2p_send_queue_mutex_);
                    if (load_step6_p2p_send_queue_.empty()) {
                        load_step6_p2p_send_worker_completed_ = true;
                        load_step6_p2p_send_worker_sentinel_received_ = false;
                        break;
                    }
                }
                continue;
            }
            
            // Send data using RDMA or ASIO (load mode)
            bool send_success = false;
            if (p2p_partner_rank_ >= 0 && task.size > 0 && task.send_buffer_addr != 0) {
#ifdef __linux__
                if (use_rdma_ && rdma_step6_p2p_qp_) {
                    // std::cout << "[EC-CHECK RDMA] Load_Step6_P2P_Send: Sending " << task.size << " bytes via RDMA" << std::endl;
                    const auto t_net = std::chrono::steady_clock::now();
                    try {
                        rdma_send_data_via_qp(rdma_step6_p2p_qp_, rdma_step6_p2p_send_cq_,
                            get_rdma_step6_p2p_send_control_sock(), rdma_step6_p2p_send_control_mutex_,
                            reinterpret_cast<const uint8_t*>(task.send_buffer_addr), task.size);
                        record_load_net_ns_(load_step6_p2p_send_total_ns_, load_step6_p2p_send_task_count_, t_net);
                        send_success = true;
                    } catch (const std::exception& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] Load Step6 P2P RDMA send failed: " << e.what() << std::endl;
                    }
                } else
#endif
                if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_step6_p2p_send_connected()) {
                    uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(task.send_buffer_addr);
                    uint32_t size_net = htonl(static_cast<uint32_t>(task.size));
                    
                    try {
                        const auto t_net = std::chrono::steady_clock::now();
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load Step6 P2P send: About to send header (size=" << task.size << ")" << std::endl;
                        boost::asio::write(
                            asio_conn_mgr_.get_step6_p2p_send_socket(),
                            boost::asio::buffer(&size_net, sizeof(uint32_t))
                        );
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load Step6 P2P send: Header sent successfully" << std::endl;
                        
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load Step6 P2P send: About to send data (size=" << task.size << ")" << std::endl;
                        boost::asio::write(
                            asio_conn_mgr_.get_step6_p2p_send_socket(),
                            boost::asio::buffer(buffer_ptr, task.size)
                        );
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load Step6 P2P send: Data sent successfully" << std::endl;
                        record_load_net_ns_(load_step6_p2p_send_total_ns_, load_step6_p2p_send_task_count_, t_net);
                        send_success = true;
                    } catch (const boost::system::system_error& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ 
                                  << "] Load Step6 P2P ASIO send failed: " << e.what() << std::endl;
                    }
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] WARNING: ASIO not available for load Step6 P2P send" << std::endl;
                }
            }
            
            // Step6: Release parity buffer after send (always release, even if send failed)
            if (task.parity_addr != 0) {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                parity_buffers_to_release_.push(task.parity_addr);
            }
            
            // Check if sentinel received and queue is empty
            if (load_step6_p2p_send_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(load_step6_p2p_send_queue_mutex_);
                if (load_step6_p2p_send_queue_.empty()) {
                    load_step6_p2p_send_worker_completed_ = true;
                    load_step6_p2p_send_worker_sentinel_received_ = false;
                    break;
                }
            }
        }
    }
    
    // Load Step6 P2P Recv Worker - dedicated for Step6 P2P recv operations (rank2 only)
    void load_step6_p2p_recv_worker() {
        
        while (!should_stop_threads_) {
            P2PRecvTask task;
            
            {
                std::unique_lock<std::mutex> lock(load_step6_p2p_recv_queue_mutex_);
                // FIX: Remove sentinel check from wait condition, match save mode behavior
                load_step6_p2p_recv_queue_cv_.wait(lock, [this] {
                    return !load_step6_p2p_recv_queue_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && load_step6_p2p_recv_queue_.empty()) {
                    break;
                }
                
                task = load_step6_p2p_recv_queue_.front();
                load_step6_p2p_recv_queue_.pop();
            }
            
            // Check for sentinel
            if (task.recv_buffer_addr == 0 && task.size == 0) {
                load_step6_p2p_recv_worker_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(load_step6_p2p_recv_queue_mutex_);
                    if (load_step6_p2p_recv_queue_.empty()) {
                        load_step6_p2p_recv_worker_completed_ = true;
                        load_step6_p2p_recv_worker_sentinel_received_ = false;
                        break;  // Exit loop when sentinel received and queue is empty
                    }
                }
                continue;  // Continue processing remaining tasks if queue is not empty
            }
            
            // Receive data using RDMA or ASIO (load mode)
            bool task_processed = false;
            if (p2p_partner_rank_ >= 0 && task.size > 0 && task.recv_buffer_addr != 0) {
#ifdef __linux__
                if (use_rdma_ && rdma_step6_p2p_qp_) {
                    // std::cout << "[EC-CHECK RDMA] Load_Step6_P2P_Recv: Receiving " << task.size << " bytes via RDMA" << std::endl;
                    const auto t_net = std::chrono::steady_clock::now();
                    try {
                        size_t recv_size = rdma_receive_data_via_qp(rdma_step6_p2p_qp_, rdma_step6_p2p_recv_cq_,
                            get_rdma_step6_p2p_recv_control_sock(), rdma_step6_p2p_recv_control_mutex_,
                            reinterpret_cast<uint8_t*>(task.recv_buffer_addr), task.size,
                            RdmaLoadRecvPollLane::Step6P2p);
                        if (recv_size == task.size) {
                            task_processed = true;
                            record_load_net_ns_(load_step6_p2p_recv_total_ns_, load_step6_p2p_recv_task_count_, t_net, true,
                                                "step6_p2p_recv", task.size);
                        } else {
                            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Load Step6 P2P RDMA recv size mismatch: expected "
                                      << task.size << ", got " << recv_size << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ << "] Load Step6 P2P RDMA recv failed: " << e.what() << std::endl;
                        continue;
                    }
                } else
#endif
                if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_step6_p2p_recv_connected()) {
                    uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(task.recv_buffer_addr);
                    uint32_t size_net;
                    
                    try {
                        const auto t_net = std::chrono::steady_clock::now();
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load Step6 P2P recv: About to receive header (expected_size=" << task.size << ")" << std::endl;
                        boost::asio::read(
                            asio_conn_mgr_.get_step6_p2p_recv_socket(),
                            boost::asio::buffer(&size_net, sizeof(uint32_t))
                        );
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load Step6 P2P recv: Header received successfully" << std::endl;
                        
                        uint32_t size = ntohl(size_net);
                        if (size != task.size) {
                            std::cerr << "EC-CHECK: [Rank " << rank_ 
                                      << "] Load Step6 P2P size mismatch: expected " << task.size 
                                      << ", got " << size << std::endl;
                            continue;
                        }
                        
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load Step6 P2P recv: About to receive data (size=" << size << ")" << std::endl;
                        boost::asio::read(
                            asio_conn_mgr_.get_step6_p2p_recv_socket(),
                            boost::asio::buffer(buffer_ptr, size)
                        );
                        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Load Step6 P2P recv: Data received successfully" << std::endl;
                        record_load_net_ns_(load_step6_p2p_recv_total_ns_, load_step6_p2p_recv_task_count_, t_net, true);
                        task_processed = true;
                    } catch (const boost::system::system_error& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ 
                                  << "] Load Step6 P2P ASIO recv failed: " << e.what() << std::endl;
                        continue;
                    }
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] WARNING: ASIO not available for load Step6 P2P recv" << std::endl;
                    task_processed = true;  // Mark as processed to avoid blocking
                }
            } else {
                task_processed = true;
            }
            
            // Step6: rank2 receives d3 to partner_buffer (no additional action needed)
            // partner_buffer is managed by Python
            
            // After processing task, check if sentinel was received and queue is empty
            if (load_step6_p2p_recv_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(load_step6_p2p_recv_queue_mutex_);
                if (load_step6_p2p_recv_queue_.empty()) {
                    load_step6_p2p_recv_worker_completed_ = true;
                    load_step6_p2p_recv_worker_sentinel_received_ = false;
                    break;  // Exit loop when all tasks are processed
                }
            }
        }
        
    }
    
    // Submit sentinel to load encoder worker
    void submit_load_encoding_sentinel() {
        if (!is_load_mode_) {
            std::cerr << "EC-CHECK: [Rank " << rank_ 
                      << "] submit_load_encoding_sentinel called but not in load mode" << std::endl;
            return;
        }
        
        LoadEncodingTask sentinel = {0, 0, 0, 0, 0, 0, false, 0};
        submit_load_encoding_task(sentinel);
    }
    
    // Submit sentinel to Step6 P2P workers (called after all Step6 tasks are submitted)
    void submit_load_step6_p2p_sentinel() {
        if (!is_load_mode_) {
            std::cerr << "EC-CHECK: [Rank " << rank_ 
                      << "] submit_load_step6_p2p_sentinel called but not in load mode" << std::endl;
            return;
        }
        
        // Send sentinel to Step6 P2P workers if needed
        if (rank_in_group_ == 3) {
            {
                std::lock_guard<std::mutex> step6_send_lock(load_step6_p2p_send_queue_mutex_);
                load_step6_p2p_send_queue_.push({0, 0, 0, 0, 0, true, true, 0, 0, 0});
            }
            load_step6_p2p_send_queue_cv_.notify_one();
        }
        if (rank_in_group_ == 2) {
            {
                std::lock_guard<std::mutex> step6_recv_lock(load_step6_p2p_recv_queue_mutex_);
                load_step6_p2p_recv_queue_.push({0, 0, true, true, 0, false, 0, 0});
            }
            load_step6_p2p_recv_queue_cv_.notify_one();
        }
    }
    
    // Simple synchronous P2P send/recv for rank1 software failure recovery (no worker queue)
    void simple_p2p_send(uintptr_t buffer_addr, size_t size) {
        if (!use_asio_ || !asio_initialized_ || !asio_conn_mgr_.is_p2p_send_connected()) {
            throw std::runtime_error("ASIO P2P send not initialized");
        }
#ifdef __linux__
        if (use_rdma_ && rdma_p2p_send_qp_) {
            bool temp_reg = false;
            if (!rdma_find_mr(buffer_addr, size) && size > rdma_temp_send_buffer_.size()) {
                register_buffer(buffer_addr, size);
                temp_reg = true;
            }
            // std::cout << "[EC-CHECK RDMA] Simple_P2P_Send: Sending " << size << " bytes via RDMA" << std::endl;
            try {
                rdma_send_data_via_qp(rdma_p2p_send_qp_, rdma_p2p_send_cq_, get_rdma_p2p_send_control_sock(),
                    rdma_p2p_send_control_mutex_, reinterpret_cast<const uint8_t*>(buffer_addr), size);
            } catch (...) {
                if (temp_reg) unregister_buffer(buffer_addr);
                throw;
            }
            if (temp_reg) unregister_buffer(buffer_addr);
            return;
        }
#endif
        uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(buffer_addr);
        // Use uint64_t to support data transfers > 4GB
        uint64_t size_net = htonll(static_cast<uint64_t>(size));
        
        try {
            // Send size header (8 bytes for uint64_t)
            boost::asio::write(
                asio_conn_mgr_.get_p2p_send_socket(),
                boost::asio::buffer(&size_net, sizeof(uint64_t))
            );
            
            // Send data
            boost::asio::write(
                asio_conn_mgr_.get_p2p_send_socket(),
                boost::asio::buffer(buffer_ptr, size)
            );
            
        } catch (const boost::system::system_error& e) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Simple P2P send failed: " 
                      << e.what() << std::endl;
            throw;
        }
    }
    
    void simple_p2p_recv(uintptr_t buffer_addr, size_t size) {
        if (!use_asio_ || !asio_initialized_ || !asio_conn_mgr_.is_p2p_recv_connected()) {
            throw std::runtime_error("ASIO P2P recv not initialized");
        }
#ifdef __linux__
        if (use_rdma_ && rdma_p2p_recv_qp_) {
            bool temp_reg = false;
            if (!rdma_find_mr(buffer_addr, size) && size > rdma_temp_recv_buffer_.size()) {
                register_buffer(buffer_addr, size);
                temp_reg = true;
            }
            // std::cout << "[EC-CHECK RDMA] Simple_P2P_Recv: Receiving " << size << " bytes via RDMA" << std::endl;
            try {
                rdma_receive_data_via_qp(rdma_p2p_recv_qp_, rdma_p2p_recv_cq_, get_rdma_p2p_recv_control_sock(),
                    rdma_p2p_recv_control_mutex_, reinterpret_cast<uint8_t*>(buffer_addr), size,
                    RdmaLoadRecvPollLane::None);
            } catch (...) {
                if (temp_reg) unregister_buffer(buffer_addr);
                throw;
            }
            if (temp_reg) unregister_buffer(buffer_addr);
            return;
        }
#endif
        uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(buffer_addr);
        // Use uint64_t to support data transfers > 4GB
        uint64_t size_net;
        
        try {
            // Receive size header (8 bytes for uint64_t)
            boost::asio::read(
                asio_conn_mgr_.get_p2p_recv_socket(),
                boost::asio::buffer(&size_net, sizeof(uint64_t))
            );
            
            uint64_t received_size = ntohll(size_net);
            if (static_cast<size_t>(received_size) != size) {
                throw std::runtime_error(
                    "Size mismatch: expected " + std::to_string(size) + 
                    ", got " + std::to_string(received_size)
                );
            }
            
            // Receive data
            boost::asio::read(
                asio_conn_mgr_.get_p2p_recv_socket(),
                boost::asio::buffer(buffer_ptr, size)
            );
            
        } catch (const boost::system::system_error& e) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Simple P2P recv failed: " 
                      << e.what() << std::endl;
            throw;
        }
    }
    
#ifdef __linux__
    void init_rdma_resources() {
        if (!use_rdma_) return;

        if (ibv_fork_init() != 0) {
            std::cerr << "[EC-CHECK RDMA] WARNING: ibv_fork_init() failed. Forked processes may get Bad address."
                      << std::endl;
        }

        int num_devices;
        ibv_device** device_list = ibv_get_device_list(&num_devices);
        if (!device_list || num_devices == 0) {
            throw std::runtime_error("No RDMA devices found");
        }

        rdma_context_ = ibv_open_device(
            find_rdma_device_by_ip(my_ip_, device_list, num_devices));
        if (!rdma_context_) {
            ibv_free_device_list(device_list);
            throw std::runtime_error("Failed to open RDMA device");
        }
        ibv_free_device_list(device_list);

        rdma_pd_ = ibv_alloc_pd(rdma_context_);
        if (!rdma_pd_) {
            throw std::runtime_error("Failed to allocate protection domain");
        }

        rdma_xor_send_cq_ = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
        rdma_xor_recv_cq_ = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
        if (!rdma_xor_send_cq_ || !rdma_xor_recv_cq_) {
            if (rdma_xor_send_cq_) { ibv_destroy_cq(rdma_xor_send_cq_); rdma_xor_send_cq_ = nullptr; }
            if (rdma_xor_recv_cq_) { ibv_destroy_cq(rdma_xor_recv_cq_); rdma_xor_recv_cq_ = nullptr; }
            ibv_dealloc_pd(rdma_pd_); rdma_pd_ = nullptr;
            ibv_close_device(rdma_context_); rdma_context_ = nullptr;
            throw std::runtime_error("Failed to create XOR completion queues");
        }

        rdma_p2p_send_cq_ = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
        rdma_p2p_recv_cq_ = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
        if (!rdma_p2p_send_cq_ || !rdma_p2p_recv_cq_) {
            ibv_destroy_cq(rdma_xor_send_cq_); rdma_xor_send_cq_ = nullptr;
            ibv_destroy_cq(rdma_xor_recv_cq_); rdma_xor_recv_cq_ = nullptr;
            if (rdma_p2p_send_cq_) { ibv_destroy_cq(rdma_p2p_send_cq_); rdma_p2p_send_cq_ = nullptr; }
            if (rdma_p2p_recv_cq_) { ibv_destroy_cq(rdma_p2p_recv_cq_); rdma_p2p_recv_cq_ = nullptr; }
            ibv_dealloc_pd(rdma_pd_); rdma_pd_ = nullptr;
            ibv_close_device(rdma_context_); rdma_context_ = nullptr;
            throw std::runtime_error("Failed to create P2P completion queues");
        }

        const int MAX_WR = 64;
        ibv_qp_init_attr qp_attr{};
        qp_attr.send_cq = rdma_xor_send_cq_;
        qp_attr.recv_cq = rdma_xor_send_cq_;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr = MAX_WR;
        qp_attr.cap.max_recv_wr = MAX_WR;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        rdma_xor_send_qp_ = ibv_create_qp(rdma_pd_, &qp_attr);
        qp_attr.send_cq = rdma_xor_recv_cq_;
        qp_attr.recv_cq = rdma_xor_recv_cq_;
        rdma_xor_recv_qp_ = ibv_create_qp(rdma_pd_, &qp_attr);
        rdma_xor_qp_ = rdma_xor_send_qp_;
        if (!rdma_xor_send_qp_ || !rdma_xor_recv_qp_) {
            if (rdma_xor_send_qp_) { ibv_destroy_qp(rdma_xor_send_qp_); rdma_xor_send_qp_ = nullptr; }
            if (rdma_xor_recv_qp_) { ibv_destroy_qp(rdma_xor_recv_qp_); rdma_xor_recv_qp_ = nullptr; }
            rdma_xor_qp_ = nullptr;
            ibv_destroy_cq(rdma_xor_send_cq_); rdma_xor_send_cq_ = nullptr;
            ibv_destroy_cq(rdma_xor_recv_cq_); rdma_xor_recv_cq_ = nullptr;
            ibv_destroy_cq(rdma_p2p_send_cq_); rdma_p2p_send_cq_ = nullptr;
            ibv_destroy_cq(rdma_p2p_recv_cq_); rdma_p2p_recv_cq_ = nullptr;
            ibv_dealloc_pd(rdma_pd_); rdma_pd_ = nullptr;
            ibv_close_device(rdma_context_); rdma_context_ = nullptr;
            throw std::runtime_error("Failed to create XOR QPs");
        }

        qp_attr.send_cq = rdma_p2p_send_cq_;
        qp_attr.recv_cq = rdma_p2p_recv_cq_;
        rdma_p2p_send_qp_ = ibv_create_qp(rdma_pd_, &qp_attr);
        rdma_p2p_recv_qp_ = ibv_create_qp(rdma_pd_, &qp_attr);
        rdma_p2p_qp_ = rdma_p2p_send_qp_;
        if (!rdma_p2p_send_qp_ || !rdma_p2p_recv_qp_) {
            if (rdma_p2p_send_qp_) { ibv_destroy_qp(rdma_p2p_send_qp_); rdma_p2p_send_qp_ = nullptr; }
            if (rdma_p2p_recv_qp_) { ibv_destroy_qp(rdma_p2p_recv_qp_); rdma_p2p_recv_qp_ = nullptr; }
            rdma_p2p_qp_ = nullptr;
            if (rdma_xor_send_qp_) { ibv_destroy_qp(rdma_xor_send_qp_); rdma_xor_send_qp_ = nullptr; }
                if (rdma_xor_recv_qp_) { ibv_destroy_qp(rdma_xor_recv_qp_); rdma_xor_recv_qp_ = nullptr; }
                rdma_xor_qp_ = nullptr;
            ibv_destroy_cq(rdma_xor_send_cq_); rdma_xor_send_cq_ = nullptr;
            ibv_destroy_cq(rdma_xor_recv_cq_); rdma_xor_recv_cq_ = nullptr;
            ibv_destroy_cq(rdma_p2p_send_cq_); rdma_p2p_send_cq_ = nullptr;
            ibv_destroy_cq(rdma_p2p_recv_cq_); rdma_p2p_recv_cq_ = nullptr;
            ibv_dealloc_pd(rdma_pd_); rdma_pd_ = nullptr;
            ibv_close_device(rdma_context_); rdma_context_ = nullptr;
            throw std::runtime_error("Failed to create P2P QPs");
        }

        if (rank_in_group_ == 2 || rank_in_group_ == 3) {
            rdma_step6_p2p_send_cq_ = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            rdma_step6_p2p_recv_cq_ = ibv_create_cq(rdma_context_, 256, nullptr, nullptr, 0);
            if (!rdma_step6_p2p_send_cq_ || !rdma_step6_p2p_recv_cq_) {
                if (rdma_step6_p2p_send_cq_) { ibv_destroy_cq(rdma_step6_p2p_send_cq_); rdma_step6_p2p_send_cq_ = nullptr; }
                if (rdma_step6_p2p_recv_cq_) { ibv_destroy_cq(rdma_step6_p2p_recv_cq_); rdma_step6_p2p_recv_cq_ = nullptr; }
                if (rdma_xor_send_qp_) { ibv_destroy_qp(rdma_xor_send_qp_); rdma_xor_send_qp_ = nullptr; }
                if (rdma_xor_recv_qp_) { ibv_destroy_qp(rdma_xor_recv_qp_); rdma_xor_recv_qp_ = nullptr; }
                rdma_xor_qp_ = nullptr;
                if (rdma_p2p_send_qp_) { ibv_destroy_qp(rdma_p2p_send_qp_); rdma_p2p_send_qp_ = nullptr; }
                if (rdma_p2p_recv_qp_) { ibv_destroy_qp(rdma_p2p_recv_qp_); rdma_p2p_recv_qp_ = nullptr; }
                rdma_p2p_qp_ = nullptr;
                ibv_destroy_cq(rdma_xor_send_cq_); rdma_xor_send_cq_ = nullptr;
                ibv_destroy_cq(rdma_xor_recv_cq_); rdma_xor_recv_cq_ = nullptr;
                ibv_destroy_cq(rdma_p2p_send_cq_); rdma_p2p_send_cq_ = nullptr;
                ibv_destroy_cq(rdma_p2p_recv_cq_); rdma_p2p_recv_cq_ = nullptr;
                ibv_dealloc_pd(rdma_pd_); rdma_pd_ = nullptr;
                ibv_close_device(rdma_context_); rdma_context_ = nullptr;
                throw std::runtime_error("Failed to create Step6 P2P completion queues");
            }
            qp_attr.send_cq = rdma_step6_p2p_send_cq_;
            qp_attr.recv_cq = rdma_step6_p2p_recv_cq_;
            rdma_step6_p2p_qp_ = ibv_create_qp(rdma_pd_, &qp_attr);
            if (!rdma_step6_p2p_qp_) {
                ibv_destroy_cq(rdma_step6_p2p_send_cq_); rdma_step6_p2p_send_cq_ = nullptr;
                ibv_destroy_cq(rdma_step6_p2p_recv_cq_); rdma_step6_p2p_recv_cq_ = nullptr;
                if (rdma_xor_send_qp_) { ibv_destroy_qp(rdma_xor_send_qp_); rdma_xor_send_qp_ = nullptr; }
                if (rdma_xor_recv_qp_) { ibv_destroy_qp(rdma_xor_recv_qp_); rdma_xor_recv_qp_ = nullptr; }
                rdma_xor_qp_ = nullptr;
                if (rdma_p2p_send_qp_) { ibv_destroy_qp(rdma_p2p_send_qp_); rdma_p2p_send_qp_ = nullptr; }
                if (rdma_p2p_recv_qp_) { ibv_destroy_qp(rdma_p2p_recv_qp_); rdma_p2p_recv_qp_ = nullptr; }
                rdma_p2p_qp_ = nullptr;
                ibv_destroy_cq(rdma_xor_send_cq_); rdma_xor_send_cq_ = nullptr;
                ibv_destroy_cq(rdma_xor_recv_cq_); rdma_xor_recv_cq_ = nullptr;
                ibv_destroy_cq(rdma_p2p_send_cq_); rdma_p2p_send_cq_ = nullptr;
                ibv_destroy_cq(rdma_p2p_recv_cq_); rdma_p2p_recv_cq_ = nullptr;
                ibv_dealloc_pd(rdma_pd_); rdma_pd_ = nullptr;
                ibv_close_device(rdma_context_); rdma_context_ = nullptr;
                throw std::runtime_error("Failed to create Step6 P2P QP");
            }
        }

        static const size_t TEMP_BUFFER_SIZE = 64ULL * 1024 * 1024;  // 64 MB
        rdma_temp_send_buffer_.resize(TEMP_BUFFER_SIZE);
        rdma_temp_recv_buffer_.resize(TEMP_BUFFER_SIZE);
        rdma_temp_send_mr_ = ibv_reg_mr(rdma_pd_, rdma_temp_send_buffer_.data(), TEMP_BUFFER_SIZE,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        rdma_temp_recv_mr_ = ibv_reg_mr(rdma_pd_, rdma_temp_recv_buffer_.data(), TEMP_BUFFER_SIZE,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!rdma_temp_send_mr_ || !rdma_temp_recv_mr_) {
            if (rdma_temp_send_mr_) { ibv_dereg_mr(rdma_temp_send_mr_); rdma_temp_send_mr_ = nullptr; }
            if (rdma_temp_recv_mr_) { ibv_dereg_mr(rdma_temp_recv_mr_); rdma_temp_recv_mr_ = nullptr; }
            if (rdma_xor_send_qp_) { ibv_destroy_qp(rdma_xor_send_qp_); rdma_xor_send_qp_ = nullptr; }
                if (rdma_xor_recv_qp_) { ibv_destroy_qp(rdma_xor_recv_qp_); rdma_xor_recv_qp_ = nullptr; }
                rdma_xor_qp_ = nullptr;
            if (rdma_p2p_send_qp_) { ibv_destroy_qp(rdma_p2p_send_qp_); rdma_p2p_send_qp_ = nullptr; }
            if (rdma_p2p_recv_qp_) { ibv_destroy_qp(rdma_p2p_recv_qp_); rdma_p2p_recv_qp_ = nullptr; }
            rdma_p2p_qp_ = nullptr;
            ibv_destroy_cq(rdma_xor_send_cq_); rdma_xor_send_cq_ = nullptr;
            ibv_destroy_cq(rdma_xor_recv_cq_); rdma_xor_recv_cq_ = nullptr;
            ibv_destroy_cq(rdma_p2p_send_cq_); rdma_p2p_send_cq_ = nullptr;
            ibv_destroy_cq(rdma_p2p_recv_cq_); rdma_p2p_recv_cq_ = nullptr;
            ibv_dealloc_pd(rdma_pd_); rdma_pd_ = nullptr;
            ibv_close_device(rdma_context_); rdma_context_ = nullptr;
            throw std::runtime_error("Failed to register RDMA temp buffers");
        }

    }

    void exchange_and_connect_qp(int control_sock, ibv_qp* qp, bool we_send_first) {
        RdmaConnInfo local_info{};
        local_info.qp_num = qp->qp_num;
        ibv_port_attr port_attr;
        if (ibv_query_port(rdma_context_, 1, &port_attr)) {
            throw std::runtime_error("EC-CHECK RDMA: failed to query port");
        }
        local_info.lid = port_attr.lid;
        ibv_gid gid;
        if (ibv_query_gid(rdma_context_, 1, 1, &gid)) {
            throw std::runtime_error("EC-CHECK RDMA: failed to query GID");
        }
        std::memcpy(local_info.gid, &gid, 16);

        RdmaConnInfo remote_info;
        std::memset(&remote_info, 0, sizeof(remote_info));
        if (we_send_first) {
            if (::send(control_sock, &local_info, sizeof(local_info), 0) != static_cast<ssize_t>(sizeof(local_info))) {
                throw std::runtime_error("EC-CHECK RDMA: failed to send local RdmaConnInfo");
            }
            if (::recv(control_sock, &remote_info, sizeof(remote_info), MSG_WAITALL) != static_cast<ssize_t>(sizeof(remote_info))) {
                throw std::runtime_error("EC-CHECK RDMA: failed to receive remote RdmaConnInfo");
            }
        } else {
            if (::recv(control_sock, &remote_info, sizeof(remote_info), MSG_WAITALL) != static_cast<ssize_t>(sizeof(remote_info))) {
                throw std::runtime_error("EC-CHECK RDMA: failed to receive remote RdmaConnInfo");
            }
            if (::send(control_sock, &local_info, sizeof(local_info), 0) != static_cast<ssize_t>(sizeof(local_info))) {
                throw std::runtime_error("EC-CHECK RDMA: failed to send local RdmaConnInfo");
            }
        }

        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = 1;
        attr.pkey_index = 0;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            throw std::runtime_error("EC-CHECK RDMA: failed to transition QP to INIT");
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
        std::memcpy(&attr.ah_attr.grh.dgid, remote_info.gid, 16);
        attr.ah_attr.grh.sgid_index = 1; // GID index 1 for erdma (RoCE v2)
        attr.ah_attr.grh.hop_limit = 64;
        if (ibv_modify_qp(qp, &attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
            throw std::runtime_error("EC-CHECK RDMA: failed to transition QP to RTR");
        }
        attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        if (ibv_modify_qp(qp, &attr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
            throw std::runtime_error("EC-CHECK RDMA: failed to transition QP to RTS");
        }
    }

    ibv_mr* rdma_find_mr(uintptr_t addr, size_t size) {
        std::lock_guard<std::mutex> lock(rdma_buffer_mutex_);
        for (auto& kv : rdma_registered_buffers_) {
            if (addr >= kv.first && (addr + size) <= (kv.first + kv.second.size)) {
                return kv.second.mr;
            }
        }
        return nullptr;
    }

    bool rdma_range_registered(uintptr_t addr, size_t size) {
        if (size == 0) return true;
        std::lock_guard<std::mutex> lock(rdma_buffer_mutex_);
        for (const auto& kv : rdma_registered_buffers_) {
            uintptr_t base = kv.first;
            size_t buf_size = kv.second.size;
            if (addr < base) continue;
            uintptr_t offset = addr - base;
            if (offset <= buf_size && size <= (buf_size - offset)) {
                return true;
            }
        }
        return false;
    }

    int get_rdma_xor_send_control_sock() {
        return asio_conn_mgr_.get_xor_send_socket().native_handle();
    }
    int get_rdma_xor_recv_control_sock() {
        return asio_conn_mgr_.get_xor_recv_socket().native_handle();
    }
    int get_rdma_p2p_send_control_sock() {
        return asio_conn_mgr_.get_p2p_send_socket().native_handle();
    }
    int get_rdma_p2p_recv_control_sock() {
        return asio_conn_mgr_.get_p2p_recv_socket().native_handle();
    }
    int get_rdma_step6_p2p_send_control_sock() {
        return asio_conn_mgr_.get_step6_p2p_send_socket().native_handle();
    }
    int get_rdma_step6_p2p_recv_control_sock() {
        return asio_conn_mgr_.get_step6_p2p_recv_socket().native_handle();
    }

    void rdma_poll_completion(ibv_cq* cq, int num_completions) {
        int completed = 0;
        while (completed < num_completions) {
            ibv_wc wc;
            int ret = ibv_poll_cq(cq, 1, &wc);
            if (ret < 0) throw std::runtime_error("EC-CHECK RDMA: failed to poll CQ");
            if (ret > 0) {
                if (wc.status != IBV_WC_SUCCESS) throw std::runtime_error("EC-CHECK RDMA: work completion failed");
                completed++;
            }
        }
    }

    void rdma_send_data_via_qp(ibv_qp* qp, ibv_cq* send_cq, int control_sock, std::mutex& control_mutex,
                               const uint8_t* data, size_t size) {
        // std::cout << "[EC-CHECK RDMA] Sending " << size << " bytes via RDMA" << std::endl;
        static const size_t CHUNK_SIZE = 64ULL * 1024 * 1024;
        static const int MAX_BATCH_WR = 32;
        uint64_t size_net = htonll(static_cast<uint64_t>(size));
        ibv_mr* mr = rdma_find_mr(reinterpret_cast<uintptr_t>(data), size);
        const uint8_t* send_ptr = data;
        if (!mr) {
            if (size > rdma_temp_send_buffer_.size())
                throw std::runtime_error("EC-CHECK RDMA: data size exceeds temp buffer");
            std::memcpy(rdma_temp_send_buffer_.data(), data, size);
            mr = rdma_temp_send_mr_;
            send_ptr = rdma_temp_send_buffer_.data();
        }

        std::lock_guard<std::mutex> lock(control_mutex);
        if (::send(control_sock, &size_net, sizeof(size_net), 0) != sizeof(size_net))
            throw std::runtime_error("EC-CHECK RDMA: failed to send size");

        size_t remaining = size;
        size_t offset = 0;
        while (remaining > 0) {
            int batch_count = static_cast<int>(std::min(
                static_cast<size_t>(MAX_BATCH_WR),
                (remaining + CHUNK_SIZE - 1) / CHUNK_SIZE));
            if (batch_count == 0) batch_count = 1;

            uint8_t ack;
            if (::recv(control_sock, &ack, sizeof(ack), MSG_WAITALL) != sizeof(ack))
                throw std::runtime_error("EC-CHECK RDMA: failed to receive ACK");

            std::vector<ibv_sge> sges(batch_count);
            std::vector<ibv_send_wr> wrs(batch_count);
            int num_wrs = 0;
            for (int i = 0; i < batch_count && remaining > 0; ++i) {
                size_t cur = std::min(remaining, CHUNK_SIZE);
                sges[i].addr = reinterpret_cast<uint64_t>(send_ptr + offset);
                sges[i].length = cur;
                sges[i].lkey = mr->lkey;
                wrs[i].wr_id = i;
                wrs[i].sg_list = &sges[i];
                wrs[i].num_sge = 1;
                wrs[i].opcode = IBV_WR_SEND;
                wrs[i].send_flags = IBV_SEND_SIGNALED;
                wrs[i].next = (i < batch_count - 1) ? &wrs[i + 1] : nullptr;
                offset += cur;
                remaining -= cur;
                num_wrs++;
            }
            if (num_wrs > 0) wrs[num_wrs - 1].next = nullptr;

            ibv_send_wr* bad_wr = nullptr;
            if (ibv_post_send(qp, &wrs[0], &bad_wr))
                throw std::runtime_error("EC-CHECK RDMA: failed to post send");
            rdma_poll_completion(send_cq, num_wrs);
        }
    }

    size_t rdma_receive_data_via_qp(ibv_qp* qp, ibv_cq* recv_cq, int control_sock, std::mutex& control_mutex,
                                    uint8_t* buffer, size_t buffer_size,
                                    RdmaLoadRecvPollLane poll_lane = RdmaLoadRecvPollLane::None) {
        static const size_t CHUNK_SIZE = 64ULL * 1024 * 1024;
        static const int MAX_BATCH_WR = 32;
        uint64_t size_net;
        size_t size;
        
        // Step 1: Receive size via control channel
        // std::cout << "[EC-CHECK RDMA] Receiving data via RDMA (buffer_size=" << buffer_size << ")" << std::endl;
        {
            std::lock_guard<std::mutex> lock(control_mutex);
            if (::recv(control_sock, &size_net, sizeof(size_net), MSG_WAITALL) != sizeof(size_net))
                throw std::runtime_error("EC-CHECK RDMA: failed to receive size");
        }
        size = ntohll(size_net);
        if (size > buffer_size) throw std::runtime_error("EC-CHECK RDMA: received size exceeds buffer");
        
        // Step 2: Prepare buffer and memory region
        ibv_mr* mr = rdma_find_mr(reinterpret_cast<uintptr_t>(buffer), size);
        uint8_t* recv_ptr = buffer;
        bool use_temp = false;
        if (!mr) {
            if (size > rdma_temp_recv_buffer_.size())
                throw std::runtime_error("EC-CHECK RDMA: size exceeds temp buffer");
            mr = rdma_temp_recv_mr_;
            recv_ptr = rdma_temp_recv_buffer_.data();
            use_temp = true;
        }
        
        size_t remaining = size;
        size_t offset = 0;

        // Step 3-6: Process in bounded batches.  Posting all WRs for a 20B
        // checkpoint can exceed the QP recv queue depth before the ACK is sent.
        const auto poll_t0 = std::chrono::steady_clock::now();
        while (remaining > 0) {
            int chunk_count = static_cast<int>(std::min(static_cast<size_t>(MAX_BATCH_WR),
                (remaining + CHUNK_SIZE - 1) / CHUNK_SIZE));
            if (chunk_count == 0) chunk_count = 1;
            
            std::vector<ibv_sge> sges(chunk_count);
            std::vector<ibv_recv_wr> wrs(chunk_count);
            int num_wrs = 0;
            for (int i = 0; i < chunk_count && remaining > 0; ++i) {
                size_t cur = std::min(CHUNK_SIZE, remaining);
                sges[i].addr = reinterpret_cast<uint64_t>(recv_ptr + offset);
                sges[i].length = cur;
                sges[i].lkey = mr->lkey;
                wrs[i].wr_id = i;
                wrs[i].sg_list = &sges[i];
                wrs[i].num_sge = 1;
                wrs[i].next = (i < chunk_count - 1) ? &wrs[i + 1] : nullptr;
                offset += cur;
                remaining -= cur;
                num_wrs++;
            }
            if (num_wrs > 0) wrs[num_wrs - 1].next = nullptr;

            // Step 4: Post this bounded recv batch before sending ACK.
            ibv_recv_wr* bad_wr = nullptr;
            if (ibv_post_recv(qp, &wrs[0], &bad_wr))
                throw std::runtime_error("EC-CHECK RDMA: failed to post recv");

            // Step 5: Tell sender this batch is ready.
            {
                std::lock_guard<std::mutex> lock(control_mutex);
                uint8_t ack = 1;
                if (::send(control_sock, &ack, sizeof(ack), 0) != sizeof(ack))
                    throw std::runtime_error("EC-CHECK RDMA: failed to send ACK");
            }

            // Step 6: Poll this batch before posting more WRs.
            rdma_poll_completion(recv_cq, num_wrs);
        }
        const auto poll_t1 = std::chrono::steady_clock::now();
        if (rank_ == 2 && poll_lane != RdmaLoadRecvPollLane::None) {
            const uint64_t poll_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(poll_t1 - poll_t0).count());
            switch (poll_lane) {
                case RdmaLoadRecvPollLane::EncXor:
                    load_rank2_rdma_poll_enc_xor_recv_ns_.fetch_add(poll_ns, std::memory_order_relaxed);
                    break;
                case RdmaLoadRecvPollLane::Step2P2p:
                    load_rank2_rdma_poll_step2_p2p_recv_ns_.fetch_add(poll_ns, std::memory_order_relaxed);
                    break;
                case RdmaLoadRecvPollLane::Step6P2p:
                    load_rank2_rdma_poll_step6_p2p_recv_ns_.fetch_add(poll_ns, std::memory_order_relaxed);
                    break;
                default:
                    break;
            }
        }

        if (use_temp) std::memcpy(buffer, rdma_temp_recv_buffer_.data(), size);
        return size;
    }

    void cleanup_rdma_resources() {
        if (!use_rdma_) return;

        if (rdma_temp_send_mr_) { ibv_dereg_mr(rdma_temp_send_mr_); rdma_temp_send_mr_ = nullptr; }
        if (rdma_temp_recv_mr_) { ibv_dereg_mr(rdma_temp_recv_mr_); rdma_temp_recv_mr_ = nullptr; }
        rdma_temp_send_buffer_.clear();
        rdma_temp_recv_buffer_.clear();

        {
            std::lock_guard<std::mutex> lock(rdma_buffer_mutex_);
            for (auto& kv : rdma_registered_buffers_) {
                if (kv.second.mr) ibv_dereg_mr(kv.second.mr);
            }
            rdma_registered_buffers_.clear();
        }

        if (rdma_xor_qp_) { if (rdma_xor_send_qp_) { ibv_destroy_qp(rdma_xor_send_qp_); rdma_xor_send_qp_ = nullptr; }
                if (rdma_xor_recv_qp_) { ibv_destroy_qp(rdma_xor_recv_qp_); rdma_xor_recv_qp_ = nullptr; }
                rdma_xor_qp_ = nullptr; }
        if (rdma_p2p_send_qp_) { ibv_destroy_qp(rdma_p2p_send_qp_); rdma_p2p_send_qp_ = nullptr; }
        if (rdma_p2p_recv_qp_) { ibv_destroy_qp(rdma_p2p_recv_qp_); rdma_p2p_recv_qp_ = nullptr; }
        rdma_p2p_qp_ = nullptr;
        if (rdma_step6_p2p_qp_) { ibv_destroy_qp(rdma_step6_p2p_qp_); rdma_step6_p2p_qp_ = nullptr; }

        if (rdma_xor_send_cq_) { ibv_destroy_cq(rdma_xor_send_cq_); rdma_xor_send_cq_ = nullptr; }
        if (rdma_xor_recv_cq_) { ibv_destroy_cq(rdma_xor_recv_cq_); rdma_xor_recv_cq_ = nullptr; }
        if (rdma_p2p_send_cq_) { ibv_destroy_cq(rdma_p2p_send_cq_); rdma_p2p_send_cq_ = nullptr; }
        if (rdma_p2p_recv_cq_) { ibv_destroy_cq(rdma_p2p_recv_cq_); rdma_p2p_recv_cq_ = nullptr; }
        if (rdma_step6_p2p_send_cq_) { ibv_destroy_cq(rdma_step6_p2p_send_cq_); rdma_step6_p2p_send_cq_ = nullptr; }
        if (rdma_step6_p2p_recv_cq_) { ibv_destroy_cq(rdma_step6_p2p_recv_cq_); rdma_step6_p2p_recv_cq_ = nullptr; }

        if (rdma_pd_) { ibv_dealloc_pd(rdma_pd_); rdma_pd_ = nullptr; }
        if (rdma_context_) { ibv_close_device(rdma_context_); rdma_context_ = nullptr; }

    }
#endif

    // RDMA buffer registration methods
    void register_buffer(uintptr_t addr, size_t size) {
#ifdef __linux__
        if (!use_rdma_ || rdma_pd_ == nullptr) {
            // Skip registration if not using RDMA or PD not initialized
            return;
        }
        
        std::lock_guard<std::mutex> lock(rdma_buffer_mutex_);
        
        // Check if already registered
        if (rdma_registered_buffers_.find(addr) != rdma_registered_buffers_.end()) {
            return;
        }
        
        
        ibv_mr* mr = ibv_reg_mr(rdma_pd_, (void*)addr, size,
                                IBV_ACCESS_LOCAL_WRITE | 
                                IBV_ACCESS_REMOTE_WRITE | 
                                IBV_ACCESS_REMOTE_READ);
        
        if (!mr) {
            throw std::runtime_error("Failed to register memory region for RDMA");
        }
        
        rdma_registered_buffers_[addr] = {mr, addr, size};
        
#else
        // No-op on non-Linux systems
        (void)addr;
        (void)size;
#endif
    }
    
    void unregister_buffer(uintptr_t addr) {
#ifdef __linux__
        if (!use_rdma_ || rdma_pd_ == nullptr) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(rdma_buffer_mutex_);
        
        auto it = rdma_registered_buffers_.find(addr);
        if (it != rdma_registered_buffers_.end()) {
            ibv_dereg_mr(it->second.mr);
            rdma_registered_buffers_.erase(it);
            // std::cout << "[EC-CHECK RDMA] Rank " << rank_ << " buffer unregistered at " 
                    //   << std::hex << addr << std::dec << std::endl;
        }
#else
        // No-op on non-Linux systems
        (void)addr;
#endif
    }
};

PYBIND11_MODULE(eccheck_native, m) {
    // Module-level function: Check RDMA availability
    m.def("is_rdma_available", &is_rdma_available, 
          "Check if RDMA is available on this system");
    
    // Module-level function: Generate NCCL ID (can be called without creating an instance)
#ifdef NCCL_AVAILABLE
    m.def("generate_nccl_id", &generate_nccl_id, 
          "Generate a new NCCL unique ID. Returns a list of uint8_t bytes (128 bytes).");
#endif
    
    // Class definition
    pybind11::class_<ECCHECKNative>(m, "ECCHECKNative")
        // NCCL constructor (with optional rank_in_group for multi-rank)
        .def(pybind11::init<int, int, int, const std::vector<uint8_t>&, const std::vector<uint8_t>&, const std::vector<uint8_t>&, const std::vector<uint8_t>&, int, int>(),
             pybind11::arg("rank"), pybind11::arg("world_size"), pybind11::arg("paired_rank"),
             pybind11::arg("nccl_id_xor_send"), pybind11::arg("nccl_id_xor_recv"),
             pybind11::arg("nccl_id_p2p_send"), pybind11::arg("nccl_id_p2p_recv"),
             pybind11::arg("rank_in_group") = -1,
             pybind11::arg("p2p_partner_rank") = -1)
        // ASIO/RDMA constructor (with use_rdma flag)
        .def(pybind11::init<int, int, int,
             const std::string&, uint16_t,
             const std::string&, uint16_t,
             const std::string&, uint16_t,
             const std::string&, uint16_t,
             const std::string&, uint16_t,
             const std::string&, uint16_t,
             bool, int, int>(),
             pybind11::arg("rank"), pybind11::arg("world_size"), pybind11::arg("paired_rank"),
             pybind11::arg("xor_partner_ip"), pybind11::arg("xor_send_port"),
             pybind11::arg("xor_listen_ip"), pybind11::arg("xor_recv_port"),
             pybind11::arg("p2p_partner_ip"), pybind11::arg("p2p_send_port"),
             pybind11::arg("p2p_listen_ip"), pybind11::arg("p2p_recv_port"),
             pybind11::arg("step6_p2p_partner_ip"), pybind11::arg("step6_p2p_send_port"),
             pybind11::arg("step6_p2p_listen_ip"), pybind11::arg("step6_p2p_recv_port"),
             pybind11::arg("use_rdma") = false, pybind11::arg("rank_in_group") = -1,
             pybind11::arg("p2p_partner_rank") = -1)
        .def("set_buffer_addresses", &ECCHECKNative::set_buffer_addresses)
        .def("reset_encoding_completion_flags", &ECCHECKNative::reset_encoding_completion_flags)
        .def("wait_for_encoding_completion", &ECCHECKNative::wait_for_encoding_completion)
        .def("stop_pipeline", &ECCHECKNative::stop_pipeline)
        .def("submit_data_for_encoding_thread1", &ECCHECKNative::submit_data_for_encoding_thread1,
             pybind11::arg("data_addr") = 0,
             pybind11::arg("size") = 0,
             pybind11::arg("encoding_addr") = 0,
             pybind11::arg("recv_addr") = 0,
             pybind11::arg("recv_chunk_size") = 0,
             pybind11::arg("parity_addr") = 0,
             pybind11::arg("p2p_own_write_addr") = 0,
             pybind11::arg("p2p_partner_write_addr") = 0,
             pybind11::arg("local_is_zero_tail") = false,
             pybind11::arg("remote_is_zero_tail") = false,
             pybind11::arg("p2p_data_is_zero_tail") = false,
             pybind11::arg("p2p_data_size") = 0)
        .def("submit_data_for_encoding_thread2", &ECCHECKNative::submit_data_for_encoding_thread2,
             pybind11::arg("data_addr") = 0,
             pybind11::arg("size") = 0,
             pybind11::arg("encoding_addr") = 0,
             pybind11::arg("recv_addr") = 0,
             pybind11::arg("recv_chunk_size") = 0,
             pybind11::arg("parity_addr") = 0,
             pybind11::arg("p2p_own_write_addr") = 0,
             pybind11::arg("p2p_partner_write_addr") = 0,
             pybind11::arg("local_is_zero_tail") = false,
             pybind11::arg("remote_is_zero_tail") = false,
             pybind11::arg("p2p_data_is_zero_tail") = false,
             pybind11::arg("p2p_data_size") = 0)
        .def("get_data_buffers_to_release", &ECCHECKNative::get_data_buffers_to_release)
        .def("get_encoding_buffers_to_release", &ECCHECKNative::get_encoding_buffers_to_release)
        .def("get_parity_buffers_to_release", &ECCHECKNative::get_parity_buffers_to_release)
        .def("get_ft_timing_stats", &ECCHECKNative::get_ft_timing_stats,
             "Return per-rank timing: net_s wall-span; encode_s serial-equivalent encode CPU sum")
        .def("submit_data_to_p2p_thread", &ECCHECKNative::submit_data_to_p2p_thread)
        .def("set_load_mode", &ECCHECKNative::set_load_mode,
             "Set load mode for recovery pipeline",
             pybind11::arg("is_load"),
             pybind11::arg("failed_rank") = -1)
        .def("submit_load_p2p_transfer", &ECCHECKNative::submit_load_p2p_transfer,
             "Submit load mode P2P transfer task (Step2: partner_file transmission)",
             pybind11::arg("send_buffer_addr") = 0,
             pybind11::arg("recv_data_buffer_addr") = 0,
             pybind11::arg("size") = 0,
             pybind11::arg("is_sender") = false,
             pybind11::arg("load_mode_data_addr") = 0)
        .def("submit_load_pipeline_chunk", &ECCHECKNative::submit_load_pipeline_chunk,
             "Submit a complete load pipeline chunk (Step2 P2P -> Encoding -> XOR -> Step6 P2P)",
             pybind11::arg("step2_send_addr") = 0,
             pybind11::arg("step2_recv_data_addr") = 0,
             pybind11::arg("step2_size") = 0,
             pybind11::arg("data_addr") = 0,
             pybind11::arg("size") = 0,
             pybind11::arg("encoding_addr") = 0,
             pybind11::arg("recv_addr") = 0,
             pybind11::arg("recv_chunk_size") = 0,
             pybind11::arg("parity_addr") = 0,
             pybind11::arg("p2p_partner_write_addr") = 0)
        .def("submit_two_failure_encoding_chunk", &ECCHECKNative::submit_two_failure_encoding_chunk,
             "Submit encoding chunk for two-failure recovery (bidirectional XOR exchange)",
             pybind11::arg("data_addr") = 0,
             pybind11::arg("size") = 0,
             pybind11::arg("enc_addr_0") = 0,
             pybind11::arg("enc_addr_1") = 0,
             pybind11::arg("recv_addr_1") = 0,
             pybind11::arg("recv_addr_2") = 0,
             pybind11::arg("recv_chunk_size") = 0,
             pybind11::arg("own_write_addr") = 0,
             pybind11::arg("partner_write_addr") = 0)
        .def("submit_two_failure_encoding_sentinels", &ECCHECKNative::submit_two_failure_encoding_sentinels,
             "Submit sentinels to both encoder threads for two-failure recovery")
        .def("submit_load_step6_p2p_send", &ECCHECKNative::submit_load_step6_p2p_send,
             "Submit Step6 P2P send task (rank3 sends d3 to rank2)",
             pybind11::arg("parity_addr") = 0,
             pybind11::arg("size") = 0)
        .def("submit_load_step6_p2p_recv", &ECCHECKNative::submit_load_step6_p2p_recv,
             "Submit Step6 P2P recv task (rank2 receives d3)",
             pybind11::arg("partner_buffer_addr") = 0,
             pybind11::arg("size") = 0)
        .def("submit_load_encoding_sentinel", &ECCHECKNative::submit_load_encoding_sentinel,
             "Submit sentinel to load encoder worker")
        .def("submit_load_step6_p2p_sentinel", &ECCHECKNative::submit_load_step6_p2p_sentinel,
             "Submit sentinel to Step6 P2P workers (rank2/3 only)")
        .def("wait_for_xor_worker_completion", &ECCHECKNative::wait_for_xor_worker_completion,
             "Wait for XOR worker to complete (used before sending Step6 sentinel)")
        .def("simple_p2p_send", &ECCHECKNative::simple_p2p_send,
             "Simple synchronous P2P send for rank1 software failure recovery",
             pybind11::arg("buffer_addr"), pybind11::arg("size"))
        .def("simple_p2p_recv", &ECCHECKNative::simple_p2p_recv,
             "Simple synchronous P2P recv for rank1 software failure recovery",
             pybind11::arg("buffer_addr"), pybind11::arg("size"))
        .def("register_buffer", &ECCHECKNative::register_buffer,
             "Register buffer for RDMA operations",
             pybind11::arg("addr"), pybind11::arg("size"))
        .def("unregister_buffer", &ECCHECKNative::unregister_buffer,
             "Unregister buffer for RDMA operations",
             pybind11::arg("addr"));
}
