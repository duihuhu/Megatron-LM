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
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <isa-l/erasure_code.h>
#include <isa-l/raid.h>
#include <boost/asio.hpp>
#include <cstdlib>
#include <arpa/inet.h>  // For htonl/ntohl

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

// ========== ASIO Connection Manager ==========
class AsioConnectionManager {
private:
    boost::asio::io_context io_context_;
    boost::asio::ip::tcp::socket xor_send_socket_;
    boost::asio::ip::tcp::socket xor_recv_socket_;
    boost::asio::ip::tcp::socket p2p_send_socket_;
    boost::asio::ip::tcp::socket p2p_recv_socket_;
    boost::asio::ip::tcp::acceptor xor_recv_acceptor_;
    boost::asio::ip::tcp::acceptor p2p_recv_acceptor_;
    
    std::atomic<bool> xor_send_connected_;
    std::atomic<bool> xor_recv_connected_;
    std::atomic<bool> p2p_send_connected_;
    std::atomic<bool> p2p_recv_connected_;
    
    std::mutex connection_mutex_;
    std::condition_variable connection_cv_;
    
public:
    AsioConnectionManager() 
        : io_context_(),
          xor_send_socket_(io_context_),
          xor_recv_socket_(io_context_),
          p2p_send_socket_(io_context_),
          p2p_recv_socket_(io_context_),
          xor_recv_acceptor_(io_context_),
          p2p_recv_acceptor_(io_context_),
          xor_send_connected_(false),
          xor_recv_connected_(false),
          p2p_send_connected_(false),
          p2p_recv_connected_(false) {}
    
    boost::asio::io_context& get_io_context() { return io_context_; }
    boost::asio::ip::tcp::socket& get_xor_send_socket() { return xor_send_socket_; }
    boost::asio::ip::tcp::socket& get_xor_recv_socket() { return xor_recv_socket_; }
    boost::asio::ip::tcp::socket& get_p2p_send_socket() { return p2p_send_socket_; }
    boost::asio::ip::tcp::socket& get_p2p_recv_socket() { return p2p_recv_socket_; }
    
    bool is_xor_send_connected() const { return xor_send_connected_; }
    bool is_xor_recv_connected() const { return xor_recv_connected_; }
    bool is_p2p_send_connected() const { return p2p_send_connected_; }
    bool is_p2p_recv_connected() const { return p2p_recv_connected_; }
    
    void init_xor_send(const std::string& partner_ip, uint16_t port);
    void init_xor_recv(const std::string& listen_ip, uint16_t port);
    void init_p2p_send(const std::string& partner_ip, uint16_t port);
    void init_p2p_recv(const std::string& listen_ip, uint16_t port);
    void wait_for_connections(int timeout_seconds = 30);
    void cleanup();
};

// ========== ASIO Connection Manager Implementation ==========

void AsioConnectionManager::init_xor_send(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        boost::asio::ip::tcp::resolver::results_type endpoints = 
            resolver.resolve(partner_ip, std::to_string(port));
        
        std::cout << "ASIO: Connecting XOR send to " << partner_ip << ":" << port << "..." << std::endl;
        
        // Use synchronous connect
        boost::asio::connect(xor_send_socket_, endpoints);
        xor_send_connected_ = true;
        std::cout << "ASIO: XOR send connected successfully" << std::endl;
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
        
        std::cout << "ASIO: Listening XOR recv on " << listen_ip << ":" << port << "..." << std::endl;
        
        // Use synchronous accept (will block until connection is established)
        xor_recv_acceptor_.accept(xor_recv_socket_);
        xor_recv_connected_ = true;
        std::cout << "ASIO: XOR recv accepted connection successfully" << std::endl;
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
        
        std::cout << "ASIO: Connecting P2P send to " << partner_ip << ":" << port << "..." << std::endl;
        
        // Use synchronous connect
        boost::asio::connect(p2p_send_socket_, endpoints);
        p2p_send_connected_ = true;
        std::cout << "ASIO: P2P send connected successfully" << std::endl;
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
        
        std::cout << "ASIO: Listening P2P recv on " << listen_ip << ":" << port << "..." << std::endl;
        
        // Use synchronous accept (will block until connection is established)
        p2p_recv_acceptor_.accept(p2p_recv_socket_);
        p2p_recv_connected_ = true;
        std::cout << "ASIO: P2P recv accepted connection successfully" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: P2P recv init error: " << e.what() << std::endl;
        p2p_recv_connected_ = false;
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
        std::cout << "ASIO: All connections established successfully" << std::endl;
    }
}

void AsioConnectionManager::cleanup() {
    if (xor_send_socket_.is_open()) xor_send_socket_.close();
    if (xor_recv_socket_.is_open()) xor_recv_socket_.close();
    if (p2p_send_socket_.is_open()) p2p_send_socket_.close();
    if (p2p_recv_socket_.is_open()) p2p_recv_socket_.close();
    if (xor_recv_acceptor_.is_open()) xor_recv_acceptor_.close();
    if (p2p_recv_acceptor_.is_open()) p2p_recv_acceptor_.close();
}

class ECCHECKNative {
private:
    int rank_;
    int world_size_;
    int paired_rank_;
    
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
    };
    
    std::queue<EncodingTask> encoding_tasks_1_;  // Thread1的编码任务
    std::queue<EncodingTask> encoding_tasks_2_;  // Thread2的编码任务
    
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
        uintptr_t parity_addr;      // XOR结果地址（从EncodingTask传递）
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
        uintptr_t load_mode_data_addr;  // load mode Step2: corresponding data_addr (for finding encoding task)
    };
    
    struct P2PRecvTask {
        uintptr_t recv_buffer_addr;     // Receive partner data/parity to this address
        size_t size;
        // New fields for load mode
        bool is_load_mode_transfer;      // true=load mode transfer, false=save stage
        uintptr_t data_buffer_addr;      // load mode: receive后写入的data_buffer地址（如果与recv_buffer_addr不同）
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

    // EC parameters (k, rows=2) and tables
    int k_;
    int rows_;
    int data_block_index_;
    unsigned char *a_mat_;    // RS matrix (k * m)
    unsigned char *g_tbls_;   // tables produced by ec_init_tables (32 * k * rows)
#else
    bool nccl_xor_send_initialized_;
    bool nccl_xor_recv_initialized_;
    bool nccl_p2p_send_initialized_;
    bool nccl_p2p_recv_initialized_;
#endif

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
    int failed_rank_;       // Failed rank number (e.g., 2 for rank2 recovery scenario)
    
    // ASIO connection manager
    AsioConnectionManager asio_conn_mgr_;
    bool asio_initialized_;
    bool use_asio_;  // Flag to indicate if using ASIO instead of NCCL
    
    // Helper function to synchronize NCCL operation
#ifdef NCCL_AVAILABLE
    void sync_nccl_operation(const char* operation_name) {
        // Simply synchronize the default stream (stream 0) where NCCL operations execute
        // This ensures we wait for all operations on the default stream to complete
        std::cout << "EC-CHECK: [Rank " << rank_ << "] sync_nccl_operation: Starting sync for " 
                  << operation_name << "..." << std::endl;
        cudaError_t err = cudaStreamSynchronize(0);
        if (err != cudaSuccess) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Failed to synchronize stream for " 
                      << operation_name << ": " << cudaGetErrorString(err) << std::endl;
        } else {
            std::cout << "EC-CHECK: [Rank " << rank_ << "] sync_nccl_operation: Sync completed for " 
                      << operation_name << std::endl;
        }
    }
#endif

    // ========== XOR配置构建函数 ==========
    
    void build_xor_config() {
        if (is_load_mode_ && failed_rank_ == 2) {
            // Load mode: rank2 recovery scenario
            // XOR pairing is same as save: 0<->2, 1<->3
            // But behavior differs: only rank2 and rank3 do XOR for recovery
            // rank0 thread1: send to rank2 (for recovery)
            // rank1 thread1: send to rank3 (for recovery)
            // rank2 thread1: receive from rank0, do XOR to get d2
            // rank3 thread1: receive from rank1, do XOR to get d3
            if (rank_ == 0) {
                xor_config_.xor_partner_rank = 2;
                xor_config_.thread0_is_receiver = true;   // thread0接收rank2的encode（但load模式下可能不需要）
                xor_config_.thread1_is_receiver = false;  // thread1发送encode给rank2（用于恢复）
            } else if (rank_ == 1) {
                xor_config_.xor_partner_rank = 3;
                xor_config_.thread0_is_receiver = true;   // thread0接收rank3的encode（但load模式下可能不需要）
                xor_config_.thread1_is_receiver = false;  // thread1发送encode给rank3（用于恢复）
            } else if (rank_ == 2) {
                xor_config_.xor_partner_rank = 0;
                xor_config_.thread0_is_receiver = false;  // thread0发送encode给rank0（但load模式下不需要）
                xor_config_.thread1_is_receiver = true;   // thread1接收rank0的encode，做XOR得到d2
            } else if (rank_ == 3) {
                xor_config_.xor_partner_rank = 1;
                xor_config_.thread0_is_receiver = false;  // thread0发送encode给rank1（但load模式下不需要）
                xor_config_.thread1_is_receiver = true;    // thread1接收rank1的encode，做XOR得到d3
            } else {
                xor_config_.xor_partner_rank = -1;
                xor_config_.thread0_is_receiver = false;
                xor_config_.thread1_is_receiver = false;
            }
        } else {
            // Save mode: original logic
            // Hardcoded XOR configuration for 2+2 setup
            // Pairing: rank0<->rank2, rank1<->rank3
            // rank0 thread0: receive from rank2 thread0, do XOR
            // rank0 thread1: send to rank2 thread1
            // rank2 thread0: send to rank0 thread0
            // rank2 thread1: receive from rank0 thread1, do XOR
            // rank1 thread0: receive from rank3 thread0, do XOR
            // rank1 thread1: send to rank3 thread1
            // rank3 thread0: send to rank1 thread0
            // rank3 thread1: receive from rank1 thread1, do XOR
            if (rank_ == 0) {
                xor_config_.xor_partner_rank = 2;
                xor_config_.thread0_is_receiver = true;   // thread0接收rank2的encode，做XOR
                xor_config_.thread1_is_receiver = false;  // thread1发送encode给rank2
            } else if (rank_ == 1) {
                xor_config_.xor_partner_rank = 3;
                xor_config_.thread0_is_receiver = true;   // thread0接收rank3的encode，做XOR
                xor_config_.thread1_is_receiver = false;  // thread1发送encode给rank3
            } else if (rank_ == 2) {
                xor_config_.xor_partner_rank = 0;
                xor_config_.thread0_is_receiver = false;  // thread0发送encode给rank0
                xor_config_.thread1_is_receiver = true;   // thread1接收rank0的encode，做XOR
            } else if (rank_ == 3) {
                xor_config_.xor_partner_rank = 1;
                xor_config_.thread0_is_receiver = false;  // thread0发送encode给rank1
                xor_config_.thread1_is_receiver = true;    // thread1接收rank1的encode，做XOR
            } else {
                // For other ranks, no XOR (fallback)
                xor_config_.xor_partner_rank = -1;
                xor_config_.thread0_is_receiver = false;
                xor_config_.thread1_is_receiver = false;
            }
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR config - "
                  << "partner_rank=" << xor_config_.xor_partner_rank
                  << ", thread0_receiver=" << xor_config_.thread0_is_receiver
                  << ", thread1_receiver=" << xor_config_.thread1_is_receiver
                  << ", load_mode=" << (is_load_mode_ ? "true" : "false") << std::endl;
    }
    
    // ========== P2P配置构建函数 ==========
    
    void build_p2p_config() {
        // P2P pairing: adjacent ranks (0<->1, 2<->3)
        if (rank_ % 2 == 0) {
            // Even rank: pair with next rank
            p2p_partner_rank_ = rank_ + 1;
        } else {
            // Odd rank: pair with previous rank
            p2p_partner_rank_ = rank_ - 1;
        }
        
        // Validate partner rank
        if (p2p_partner_rank_ < 0 || p2p_partner_rank_ >= world_size_) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Invalid P2P partner rank: " 
                      << p2p_partner_rank_ << std::endl;
            p2p_partner_rank_ = -1;
                }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P config - "
                  << "partner_rank=" << p2p_partner_rank_ << std::endl;
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
    
    // Helper function to get peer rank in communicator for P2P
    int get_peer_rank_in_p2p_comm(int peer_rank) const {
        if (rank_ == 0 || rank_ == 1) {
            // p2p_0_1 communicator
            if (peer_rank == 0) return 0;
            if (peer_rank == 1) return 1;
        } else if (rank_ == 2 || rank_ == 3) {
            // p2p_2_3 communicator
            if (peer_rank == 2) return 0;
            if (peer_rank == 3) return 1;
        }
        return -1; // Error
    }
    
    void init_nccl_xor_send() {
#ifdef NCCL_AVAILABLE
        // Only rank0↔rank2 and rank1↔rank3 participate in XOR send
        int partner_rank = xor_config_.xor_partner_rank;
        if (partner_rank < 0) {
            std::cout << "EC-CHECK: [Rank " << rank_ << "] No XOR partner, skipping XOR send communicator" << std::endl;
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
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR send calling ncclCommInitRank (partner=" << partner_rank
                  << ", local_rank=" << local_rank_in_pair << ")..." << std::endl;
        std::cout.flush();
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
            std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR send NCCL communicator initialized successfully (partner=" << partner_rank << ")" << std::endl;
            std::cout.flush();
        }
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_xor_send_init_completed_ = true;
        }
        nccl_init_cv_.notify_all();
#else
        std::cout << "EC-CHECK: NCCL not available, skipping XOR send initialization" << std::endl;
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
            std::cout << "EC-CHECK: [Rank " << rank_ << "] No XOR partner, skipping XOR recv communicator" << std::endl;
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
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR recv calling ncclCommInitRank (partner=" << partner_rank
                  << ", local_rank=" << local_rank_in_pair << ")..." << std::endl;
        std::cout.flush();
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
            std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR recv NCCL communicator initialized successfully (partner=" << partner_rank << ")" << std::endl;
            std::cout.flush();
        }
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_xor_recv_init_completed_ = true;
        }
        nccl_init_cv_.notify_all();
#else
        std::cout << "EC-CHECK: NCCL not available, skipping XOR recv initialization" << std::endl;
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
            std::cout << "EC-CHECK: [Rank " << rank_ << "] No P2P partner, skipping P2P send communicator" << std::endl;
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
        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send calling ncclCommInitRank (partner=" << partner_rank
                  << ", local_rank=" << local_rank_in_pair << ")..." << std::endl;
        std::cout.flush();
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
            std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send NCCL communicator initialized successfully (partner=" << partner_rank << ")" << std::endl;
            std::cout.flush();
        }
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_p2p_send_init_completed_ = true;
        }
        nccl_init_cv_.notify_all();
#else
        std::cout << "EC-CHECK: NCCL not available, skipping P2P send initialization" << std::endl;
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
            std::cout << "EC-CHECK: [Rank " << rank_ << "] No P2P partner, skipping P2P recv communicator" << std::endl;
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
        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv calling ncclCommInitRank (partner=" << partner_rank
                  << ", local_rank=" << local_rank_in_pair << ")..." << std::endl;
        std::cout.flush();
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
            std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv NCCL communicator initialized successfully (partner=" << partner_rank << ")" << std::endl;
            std::cout.flush();
        }
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_p2p_recv_init_completed_ = true;
        }
        nccl_init_cv_.notify_all();
#else
        std::cout << "EC-CHECK: NCCL not available, skipping P2P recv initialization" << std::endl;
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
    
    void encode_with_coefficient(uintptr_t data_addr, size_t size, uintptr_t encoding_addr, int coefficient) {
        // 使用 isa-l 的 EC 编码对整块 buffer 进行编码。
        // 我们在初始化时已经生成了 RS 矩阵并通过 ec_init_tables 产生了 g_tbls_。
        // 每个 encoder 线程只保留自己负责的 parity（encoding_addr 指向本地 parity buffer），

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
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 1 started" << std::endl;
        
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
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 1 received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
                    if (encoding_tasks_1_.empty()) {
                encoding_thread_1_completed_ = true;
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 1 queue is empty, marking completed" << std::endl;
                        // Submit sentinel to downstream workers
                        {
                            std::lock_guard<std::mutex> send_lock(send_queue_mutex_);
                            send_queue_.push({0, 0});
                        }
                        send_queue_cv_.notify_one();
                        {
                            std::lock_guard<std::mutex> recv_lock(recv_queue_mutex_);
                            recv_queue_.push({0, 0, 0});
                        }
                        recv_queue_cv_.notify_one();
                        {
                            std::lock_guard<std::mutex> xor_lock(xor_queue_mutex_);
                            xor_queue_.push({0, 0, 0, 0, 0, 0, 0});
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
            
            if (need_encode) {
                // Step 1: Perform encoding (parity index 0 for thread1)
                encode_with_coefficient(task.data_addr, task.size, task.encoding_addr, 0);
                
                // Mark data buffer as copied by thread 1
                {
                    std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
                    auto& state = data_buffer_states_[task.data_addr];
                    state.thread1_copied = true;
                    
                    // If both threads copied, check if we need to delay release for P2P
                    // For odd ranks, data buffer is needed by P2P worker, so delay release
                    if (state.thread2_copied) {
                        // For odd ranks, data buffer will be released by P2P worker after P2P completes
                        // For even ranks, data buffer can be released immediately
                        if (rank_ % 2 == 0) {
                        std::lock_guard<std::mutex> release_lock(release_queue_mutex_);
                        data_buffers_to_release_.push(task.data_addr);
                        data_buffer_states_.erase(task.data_addr);
                    }
                        // For odd ranks, data buffer will be released by P2P worker
                    }
                }
                
                // Step 2: Handle based on XOR configuration
                if (xor_config_.thread0_is_receiver && task.parity_addr != 0 && task.recv_addr != 0) {
                    // This thread is receiver: save encoding buffer to pending, wait for recv
                    // The recv_worker will trigger XOR when recv completes
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
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread1 encoding completed, "
                              << "pending XOR for recv_addr=" << task.recv_addr 
                              << ", encoding_addr=" << task.encoding_addr 
                              << ", p2p_own=" << task.p2p_own_write_addr
                              << ", p2p_partner=" << task.p2p_partner_write_addr << std::endl;
                    // Note: encoding buffer will be released by XOR worker after XOR completes
                } else {
                    // This thread is sender: send encoding result immediately
                {
                    std::lock_guard<std::mutex> lock(send_queue_mutex_);
                    send_queue_.push({task.encoding_addr, task.size});
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 1 (sender): Pushed task to send_queue, "
                              << "encoding_addr=" << task.encoding_addr << ", size=" << task.size 
                              << ", queue_size=" << send_queue_.size() << std::endl;
                }
                send_queue_cv_.notify_one();
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 1 (sender): Notified send_queue_cv" << std::endl;
                    
                    // Sender doesn't need parity buffer, release it immediately
                    if (task.parity_addr != 0) {
                        std::lock_guard<std::mutex> lock(release_queue_mutex_);
                        parity_buffers_to_release_.push(task.parity_addr);
                    }
                }
            }
            
            // Check if we need to receive data
            bool need_recv = (task.recv_addr != 0 && task.recv_chunk_size != 0);
            
            if (need_recv && xor_config_.thread0_is_receiver) {
                // Submit recv task to recv_worker with parity_addr
                {
                    std::lock_guard<std::mutex> lock(recv_queue_mutex_);
                    recv_queue_.push({task.recv_addr, task.recv_chunk_size, task.parity_addr});
                }
                recv_queue_cv_.notify_one();
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (encoding_thread_1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
                if (encoding_tasks_1_.empty()) {
                    encoding_thread_1_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 1 queue is empty after processing, marking completed" << std::endl;
                    // Submit sentinel to downstream workers
                    {
                        std::lock_guard<std::mutex> send_lock(send_queue_mutex_);
                        send_queue_.push({0, 0});
                    }
                    send_queue_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> recv_lock(recv_queue_mutex_);
                        recv_queue_.push({0, 0, 0});
                    }
                    recv_queue_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> xor_lock(xor_queue_mutex_);
                        xor_queue_.push({0, 0, 0, 0, 0, 0, 0});
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
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 started" << std::endl;
        
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
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
                    if (encoding_tasks_2_.empty()) {
                encoding_thread_2_completed_ = true;
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 queue is empty, marking completed" << std::endl;
                        // Submit sentinel to downstream workers
                        {
                            std::lock_guard<std::mutex> send_lock(send_queue_mutex_);
                            send_queue_.push({0, 0});
                        }
                        send_queue_cv_.notify_one();
                        {
                            std::lock_guard<std::mutex> recv_lock(recv_queue_mutex_);
                            recv_queue_.push({0, 0, 0});
                        }
                        recv_queue_cv_.notify_one();
                        {
                            std::lock_guard<std::mutex> xor_lock(xor_queue_mutex_);
                            xor_queue_.push({0, 0, 0, 0, 0, 0, 0});
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
            
            if (need_encode) {
                // Step 1: Perform encoding (parity index 1 for thread2)
                encode_with_coefficient(task.data_addr, task.size, task.encoding_addr, 1);
                
                // Mark data buffer as copied by thread 2
                {
                    std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
                    auto& state = data_buffer_states_[task.data_addr];
                    state.thread2_copied = true;
                    
                    // If both threads copied, check if we need to delay release for P2P
                    // For odd ranks, data buffer is needed by P2P worker, so delay release
                    if (state.thread1_copied) {
                        // For odd ranks, data buffer will be released by P2P worker after P2P completes
                        // For even ranks, data buffer can be released immediately
                        if (rank_ % 2 == 0) {
                        std::lock_guard<std::mutex> release_lock(release_queue_mutex_);
                        data_buffers_to_release_.push(task.data_addr);
                        data_buffer_states_.erase(task.data_addr);
                    }
                        // For odd ranks, data buffer will be released by P2P worker
                    }
                }
                
                // Step 2: Handle based on XOR configuration
                if (xor_config_.thread1_is_receiver && task.parity_addr != 0 && task.recv_addr != 0) {
                    // This thread is receiver: save encoding buffer to pending, wait for recv
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
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread2 encoding completed, "
                              << "pending XOR for recv_addr=" << task.recv_addr 
                              << ", encoding_addr=" << task.encoding_addr
                              << ", p2p_own=" << task.p2p_own_write_addr
                              << ", p2p_partner=" << task.p2p_partner_write_addr << std::endl;
                } else {
                    // This thread is sender: send encoding result immediately
                {
                    std::lock_guard<std::mutex> lock(send_queue_mutex_);
                    send_queue_.push({task.encoding_addr, task.size});
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 (sender): Pushed task to send_queue, "
                              << "encoding_addr=" << task.encoding_addr << ", size=" << task.size 
                              << ", queue_size=" << send_queue_.size() << std::endl;
                }
                send_queue_cv_.notify_one();
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 (sender): Notified send_queue_cv" << std::endl;
                    
                    // Sender doesn't need parity buffer, release it immediately
                    if (task.parity_addr != 0) {
                        std::lock_guard<std::mutex> lock(release_queue_mutex_);
                        parity_buffers_to_release_.push(task.parity_addr);
                    }
                }
            }
            
            // Check if we need to receive data
            bool need_recv = (task.recv_addr != 0 && task.recv_chunk_size != 0);
            
            if (need_recv && xor_config_.thread1_is_receiver) {
                // Submit recv task to recv_worker with parity_addr
                {
                    std::lock_guard<std::mutex> lock(recv_queue_mutex_);
                    recv_queue_.push({task.recv_addr, task.recv_chunk_size, task.parity_addr});
                }
                recv_queue_cv_.notify_one();
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (encoding_thread_2_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
                if (encoding_tasks_2_.empty()) {
                    encoding_thread_2_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 queue is empty after processing, marking completed" << std::endl;
                    // Submit sentinel to downstream workers
                    {
                        std::lock_guard<std::mutex> send_lock(send_queue_mutex_);
                        send_queue_.push({0, 0});
                    }
                    send_queue_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> recv_lock(recv_queue_mutex_);
                        recv_queue_.push({0, 0, 0});
                    }
                    recv_queue_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> xor_lock(xor_queue_mutex_);
                        xor_queue_.push({0, 0, 0, 0, 0, 0, 0});
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
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker started" << std::endl;
        
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
            
            // Send data using ASIO or NCCL
            if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_xor_send_connected()) {
                // ASIO send path (synchronous)
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
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker: NCCL GroupEnd completed, starting sync..." << std::endl;
                        
                        sync_nccl_operation("Send worker: NCCL send");
                    }
                } else if (DISABLE_SEND_RECV_NCCL) {
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker: NCCL DISABLED for debugging" << std::endl;
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
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                encoding_buffers_to_release_.push(task.encoding_addr);
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
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker started" << std::endl;
        
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
            
            // Receive data using ASIO or NCCL
            if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_xor_recv_connected()) {
                // ASIO recv path (synchronous)
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
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker: NCCL DISABLED for debugging" << std::endl;
                }
            }
#endif
            else {
                std::cerr << "EC-CHECK: [Rank " << rank_ 
                          << "] WARNING: No communication method available for recv" << std::endl;
                continue;  // Skip processing
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
                        data_addr
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
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker started" << std::endl;
        
        auto submit_p2p_sentinel = [this]() {
            {
                std::lock_guard<std::mutex> send_lock(p2p_send_queue_mutex_);
                p2p_send_queue_.push({0, 0, 0, 0, 0, false, 0});
            }
            p2p_send_queue_cv_.notify_one();
            {
                std::lock_guard<std::mutex> recv_lock(p2p_recv_queue_mutex_);
                p2p_recv_queue_.push({0, 0, false, 0});
            }
            p2p_recv_queue_cv_.notify_one();
            std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker: Sent sentinel to P2P workers" << std::endl;
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
            
            unsigned char* srcs[2];
            srcs[0] = reinterpret_cast<unsigned char*>(task.local_encoding_addr);
            srcs[1] = reinterpret_cast<unsigned char*>(task.remote_encoding_addr);
            unsigned char* dest = reinterpret_cast<unsigned char*>(task.parity_addr);
            
            void* xor_array[3];
            xor_array[0] = srcs[0];
            xor_array[1] = srcs[1];
            xor_array[2] = dest;
            
            xor_gen(3, static_cast<int>(task.size), xor_array);
            std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker: XOR completed, parity addr=" << task.parity_addr << std::endl;
            
            if (task.p2p_own_write_addr != 0 && task.p2p_partner_write_addr != 0 && task.parity_addr != 0) {
                if (is_load_mode_ && failed_rank_ == 2) {
                    // Load mode: rank2 and rank3's XOR results need special handling
                    if (rank_ == 2) {
                        // rank2: XOR result (d2) write to p2p_own_write_addr
                        // No need to send, just write to own_buffer
                        if (task.p2p_own_write_addr != 0) {
                            std::memcpy(
                                reinterpret_cast<void*>(task.p2p_own_write_addr),
                                reinterpret_cast<void*>(task.parity_addr),
                                task.size
                            );
                            std::cout << "EC-CHECK: [Rank " << rank_ << "] Load XOR: Wrote d2 to own_buffer at "
                                      << task.p2p_own_write_addr << " (size=" << task.size << ")" << std::endl;
                        }
                        // rank2 will receive d3 from rank3 via P2P in Step6 (handled separately)
                    } else if (rank_ == 3) {
                        // rank3: XOR result (d3) needs to be sent to rank2 (Step6 P2P transfer)
                        // Write d3 to p2p_partner_write_addr (rank2's partner_buffer address)
                        uintptr_t send_buffer = task.parity_addr;  // d3
                        
                        {
                            std::lock_guard<std::mutex> lock(p2p_send_queue_mutex_);
                            p2p_send_queue_.push({
                                send_buffer,
                                0,  // p2p_own_write_addr (not needed for Step6)
                                task.size,
                                task.parity_addr,
                                0,  // data_addr (not needed)
                                false,  // is_load_mode_transfer (this is Step6 P2P, not Step2)
                                0   // load_mode_data_addr (not needed for Step6)
                            });
                        }
                        p2p_send_queue_cv_.notify_one();
                        
                        {
                            std::lock_guard<std::mutex> lock(p2p_recv_queue_mutex_);
                            p2p_recv_queue_.push({
                                task.p2p_partner_write_addr,  // rank2's partner_buffer address
                                task.size,
                                false,  // is_load_mode_transfer (Step6, not Step2)
                                0       // data_buffer_addr (not needed)
                            });
                        }
                        p2p_recv_queue_cv_.notify_one();
                        
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] Load XOR: Queued d3 for P2P send to rank2 "
                                  << "(size=" << task.size << ", target_addr=" << task.p2p_partner_write_addr << ")" << std::endl;
                    }
                } else {
                    // Save mode: original logic
                    uintptr_t send_buffer = (rank_ % 2 == 0) ? task.parity_addr : task.data_addr;
                    
                    {
                        std::lock_guard<std::mutex> lock(p2p_send_queue_mutex_);
                        p2p_send_queue_.push({
                            send_buffer,
                            task.p2p_own_write_addr,
                            task.size,
                            task.parity_addr,
                            task.data_addr,
                            false,  // is_load_mode_transfer
                            0      // load_mode_data_addr (not needed for save mode)
                        });
                    }
                    p2p_send_queue_cv_.notify_one();
                    
                    {
                        std::lock_guard<std::mutex> lock(p2p_recv_queue_mutex_);
                        p2p_recv_queue_.push({
                            task.p2p_partner_write_addr,
                            task.size,
                            false,  // is_load_mode_transfer
                            0       // data_buffer_addr
                        });
                    }
                    p2p_recv_queue_cv_.notify_one();
                }
            }
            
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                encoding_buffers_to_release_.push(task.local_encoding_addr);
                if (rank_ % 2 == 1 && task.parity_addr != 0) {
                    parity_buffers_to_release_.push(task.parity_addr);
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
        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker started" << std::endl;
        
        // NCCL is already initialized in main thread, no need to initialize here
        
        while (!should_stop_threads_) {
            P2PSendTask task;
            
            {
                std::unique_lock<std::mutex> lock(p2p_send_queue_mutex_);
                p2p_send_queue_cv_.wait(lock, [this] {
                    return !p2p_send_queue_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && p2p_send_queue_.empty()) {
                    break;
                }
                
                task = p2p_send_queue_.front();
                p2p_send_queue_.pop();
                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: Popped task, "
                          << "send_buffer_addr=" << task.send_buffer_addr
                          << ", p2p_own_write_addr=" << task.p2p_own_write_addr
                          << ", size=" << task.size
                          << ", queue_size_after_pop=" << p2p_send_queue_.size() << std::endl;
            }
            
            // Check for sentinel
            if (task.send_buffer_addr == 0 && task.p2p_own_write_addr == 0 && task.size == 0) {
                p2p_send_worker_sentinel_received_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(p2p_send_queue_mutex_);
                    if (p2p_send_queue_.empty()) {
                        p2p_send_worker_completed_ = true;
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker queue is empty, marking completed" << std::endl;
                        // Reset sentinel flag and continue (don't exit)
                        p2p_send_worker_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;
            }
            
            // Step 1: Copy own data/parity to own_buffer (save path only)
            if (task.p2p_own_write_addr != 0 && task.send_buffer_addr != 0 && task.size > 0) {
                std::memcpy(reinterpret_cast<void*>(task.p2p_own_write_addr),
                           reinterpret_cast<void*>(task.send_buffer_addr),
                           task.size);
                if (rank_ % 2 == 0) {
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Copied own parity to own_buffer at "
                              << task.p2p_own_write_addr << std::endl;
                } else {
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Copied own data to own_buffer at "
                              << task.p2p_own_write_addr << std::endl;
                }
            } else if (task.p2p_own_write_addr == 0 && task.send_buffer_addr != 0 && task.size > 0) {
                // Load path: direct send from provided buffer (typically mmap)
                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Using provided buffer directly (no local copy)"
                          << std::endl;
            }
            
            // Step 2: Send data using ASIO or NCCL
            if (p2p_partner_rank_ >= 0 && task.size > 0 && task.send_buffer_addr != 0) {
                if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_p2p_send_connected()) {
                    // ASIO send path (synchronous)
                    uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(task.send_buffer_addr);
                    uint32_t size_net = htonl(static_cast<uint32_t>(task.size));  // Network byte order
                    
                    const char* send_label = (rank_ % 2 == 0) ? "parity" : "data";
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: Starting ASIO send ("
                              << send_label << "), size=" << task.size << std::endl;
                    
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
                        if (task.is_load_mode_transfer) {
                            std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P send: Sent partner_file chunk to Rank " 
                                      << p2p_partner_rank_ << " (size=" << task.size << ")" << std::endl;
                        } else {
                            // Save mode or Step6: original log
                            if (rank_ % 2 == 0) {
                                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Sent parity to Rank " 
                                          << p2p_partner_rank_ << std::endl;
                            } else {
                                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Sent data to Rank " 
                                          << p2p_partner_rank_ << std::endl;
                            }
                        }
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
                            std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: partner_rank_in_comm=" 
                                      << partner_rank_in_comm << " (from global rank " << p2p_partner_rank_ << ")" << std::endl;
                            if (partner_rank_in_comm < 0 || partner_rank_in_comm >= 2) {
                                std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: Invalid partner_rank_in_comm=" 
                                          << partner_rank_in_comm << " (must be 0 or 1 for 2-rank communicator)" << std::endl;
                                std::cerr << "EC-CHECK: [Rank " << rank_ << "] p2p_partner_rank_=" << p2p_partner_rank_ << std::endl;
                                std::cerr.flush();
                            } else {
                                const char* send_label = (rank_ % 2 == 0) ? "parity" : "data";
                                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: Starting NCCL send ("
                                          << send_label << "), size=" << task.size
                                          << ", partner_rank_in_comm=" << partner_rank_in_comm << std::endl;
                                std::cout.flush();
                                
                                ncclGroupStart();
                                ncclSend(reinterpret_cast<void*>(task.send_buffer_addr), task.size,
                                         ncclUint8, partner_rank_in_comm, nccl_comm_p2p_send_, 0);
                                ncclGroupEnd();
                                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: NCCL GroupEnd completed, starting sync..." << std::endl;
                                
                                // Synchronize NCCL operation before releasing buffer
                                sync_nccl_operation("P2P send worker: NCCL send");
                                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: NCCL sync completed" << std::endl;
                                
                                if (task.is_load_mode_transfer) {
                                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P send: Sent partner_file chunk to Rank " 
                                              << p2p_partner_rank_ << " (size=" << task.size << ")" << std::endl;
                                } else {
                                    // Save mode or Step6: original log
                                    if (rank_ % 2 == 0) {
                                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Sent parity to Rank " 
                                                  << p2p_partner_rank_ << std::endl;
                                    } else {
                                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send: Sent data to Rank " 
                                                  << p2p_partner_rank_ << std::endl;
                                    }
                                }
                            }
                        }
                    } else if (DISABLE_P2P_NCCL) {
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: NCCL DISABLED for debugging" << std::endl;
                    } else {
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: Skipping NCCL (nccl_p2p_send_initialized_=" 
                                  << (nccl_p2p_send_initialized_ ? "true" : "false") << ", world_size_=" << world_size_ << ")" << std::endl;
                    }
                }
#endif
                else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] WARNING: No communication method available for P2P send" << std::endl;
                }
            } else {
                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker: Skipping send (p2p_partner_rank_=" 
                          << p2p_partner_rank_ << ", task.size=" << task.size << ")" << std::endl;
            }
            
            // Step 3: For load mode Step2, submit encoding task after P2P send completes
            if (task.is_load_mode_transfer && task.load_mode_data_addr != 0) {
                // Load mode Step2: 发送完成后，查找并提交encoding任务
                PendingEncodingTask encoding_task;
                bool found_task = false;
                {
                    std::lock_guard<std::mutex> lock(pending_encoding_tasks_mutex_);
                    auto it = pending_encoding_tasks_.find(task.load_mode_data_addr);
                    if (it != pending_encoding_tasks_.end()) {
                        encoding_task = it->second;
                        pending_encoding_tasks_.erase(it);
                        found_task = true;
                    }
                }
                
                if (found_task && encoding_task.data_addr != 0) {
                    // 提交encoding任务到encoding队列
                    submit_data_for_encoding_thread1(
                        encoding_task.data_addr, encoding_task.size,
                        encoding_task.encoding_addr1, encoding_task.recv_addr_thread1,
                        encoding_task.recv_chunk_size, encoding_task.parity_addr1,
                        encoding_task.p2p_own_write_addr, encoding_task.p2p_partner_write_addr
                    );
                    submit_data_for_encoding_thread2(
                        encoding_task.data_addr, encoding_task.size,
                        encoding_task.encoding_addr2, encoding_task.recv_addr_thread2,
                        encoding_task.recv_chunk_size, encoding_task.parity_addr2,
                        encoding_task.p2p_own_write_addr, encoding_task.p2p_partner_write_addr
                    );
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Step2 P2P send completed, submitted encoding task for data_addr=" 
                              << encoding_task.data_addr << std::endl;
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] WARNING: No pending encoding task found for data_addr=" 
                              << task.load_mode_data_addr << std::endl;
                }
            }
            
            // Step 4: Release buffers after send is guaranteed complete
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                
                if (task.is_load_mode_transfer) {
                    // Load mode Step2: release send_buffer (this is temporary buffer from mmap read)
                    if (task.send_buffer_addr != 0) {
                        data_buffers_to_release_.push(task.send_buffer_addr);
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P send: Released send_buffer " 
                                  << task.send_buffer_addr << std::endl;
                    }
                } else {
                    // Save mode or Step6: original logic
                    // For even ranks, release parity buffer after send completes (parity was sent)
                    // For odd ranks, parity buffer is not sent, so it will be released in XOR worker
                    if (rank_ % 2 == 0 && task.parity_addr != 0) {
                        parity_buffers_to_release_.push(task.parity_addr);
                    }
                    // For odd ranks, release data buffer after send completes (it was delayed in encoder worker)
                    if (rank_ % 2 == 1 && task.data_addr != 0) {
                        data_buffers_to_release_.push(task.data_addr);
                        // Also remove from data_buffer_states_ if it exists
                        {
                            std::lock_guard<std::mutex> state_lock(data_buffer_state_mutex_);
                            data_buffer_states_.erase(task.data_addr);
                        }
                    }
                }
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (p2p_send_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(p2p_send_queue_mutex_);
                if (p2p_send_queue_.empty()) {
                    p2p_send_worker_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P send worker queue is empty after processing, marking completed" << std::endl;
                    // Reset sentinel flag and continue (don't exit)
                    p2p_send_worker_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }
    
    // P2P Recv Worker - 专门接收P2P数据（独立线程，类似 recv_worker_1）
    void p2p_recv_worker() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker started" << std::endl;
        
        // NCCL is already initialized in main thread, no need to initialize here
        
        while (!should_stop_threads_) {
            P2PRecvTask task;
            
            {
                std::unique_lock<std::mutex> lock(p2p_recv_queue_mutex_);
                p2p_recv_queue_cv_.wait(lock, [this] {
                    return !p2p_recv_queue_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && p2p_recv_queue_.empty()) {
                    break;
                }
                
                task = p2p_recv_queue_.front();
                p2p_recv_queue_.pop();
                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: Popped task, "
                          << "recv_buffer_addr=" << task.recv_buffer_addr
                          << ", size=" << task.size
                          << ", queue_size_after_pop=" << p2p_recv_queue_.size() << std::endl;
            }
            
            // Check for sentinel
            if (task.recv_buffer_addr == 0 && task.size == 0) {
                p2p_recv_worker_sentinel_received_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(p2p_recv_queue_mutex_);
                    if (p2p_recv_queue_.empty()) {
                        p2p_recv_worker_completed_ = true;
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker queue is empty, marking completed" << std::endl;
                        // Reset sentinel flag and continue (don't exit)
                        p2p_recv_worker_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;
            }
            
            // Receive data using ASIO or NCCL
            // Track if task was processed (for sentinel check logic)
            bool task_processed = false;
            
            if (p2p_partner_rank_ >= 0 && task.size > 0 && task.recv_buffer_addr != 0) {
                if (use_asio_ && asio_initialized_ && asio_conn_mgr_.is_p2p_recv_connected()) {
                    // ASIO recv path (synchronous)
                    uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(task.recv_buffer_addr);
                    uint32_t size_net;
                    
                    const char* recv_label = (rank_ % 2 == 0) ? "data" : "parity";
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: Starting ASIO recv ("
                              << recv_label << "), size=" << task.size << std::endl;
                    
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
                        if (rank_ % 2 == 0) {
                            std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv: Received data from Rank " << p2p_partner_rank_ << std::endl;
                        } else {
                            std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv: Received parity from Rank " << p2p_partner_rank_ << std::endl;
                        }
                        task_processed = true;
                    } catch (const boost::system::system_error& e) {
                        std::cerr << "EC-CHECK: [Rank " << rank_ 
                                  << "] P2P ASIO recv failed: " << e.what() << std::endl;
                        continue;  // Skip processing on error
                    }
                }
#ifdef NCCL_AVAILABLE
                else if (!use_asio_) {
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
                            std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: partner_rank_in_comm=" 
                                      << partner_rank_in_comm << " (from global rank " << p2p_partner_rank_ << ")" << std::endl;
                            if (partner_rank_in_comm < 0 || partner_rank_in_comm >= 2) {
                                std::cerr << "EC-CHECK: [Rank " << rank_ << "] ERROR: Invalid partner_rank_in_comm=" 
                                          << partner_rank_in_comm << " (must be 0 or 1 for 2-rank communicator)" << std::endl;
                                std::cerr << "EC-CHECK: [Rank " << rank_ << "] p2p_partner_rank_=" << p2p_partner_rank_ << std::endl;
                                std::cerr.flush();
                            } else {
                                const char* recv_label = (rank_ % 2 == 0) ? "data" : "parity";
                                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: Starting NCCL recv ("
                                          << recv_label << "), size=" << task.size
                                          << ", partner_rank_in_comm=" << partner_rank_in_comm << std::endl;
                                std::cout.flush();
                                
                                ncclGroupStart();
                                ncclRecv(reinterpret_cast<void*>(task.recv_buffer_addr), task.size,
                                         ncclUint8, partner_rank_in_comm, nccl_comm_p2p_recv_, 0);
                                ncclGroupEnd();
                                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: NCCL GroupEnd completed, starting sync..." << std::endl;
                                
                                // Synchronize NCCL operation
                                sync_nccl_operation("P2P recv worker: NCCL recv");
                                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: NCCL sync completed" << std::endl;
                                
                                // Log will be handled in the common section after recv completes
                                task_processed = true;
                            }
                        }
                    } else if (DISABLE_P2P_NCCL) {
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: NCCL DISABLED for debugging, task processed (no-op)" << std::endl;
                        // Even when NCCL is disabled, the task is considered processed
                        // This ensures sentinel check logic works correctly
                        task_processed = true;
                    } else {
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: Skipping NCCL (nccl_p2p_recv_initialized_=" 
                                  << (nccl_p2p_recv_initialized_ ? "true" : "false") << ", world_size_=" << world_size_ << ")" << std::endl;
                        task_processed = true;
                    }
                }
#endif
                else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ 
                              << "] WARNING: No communication method available for P2P recv" << std::endl;
                    task_processed = true;  // Mark as processed to avoid blocking
                }
            } else {
                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker: Skipping recv (p2p_partner_rank_=" 
                          << p2p_partner_rank_ << ", task.size=" << task.size << "), task processed" << std::endl;
                task_processed = true;
            }
            
            // Load mode Step2: handle received data and submit encoding task
            if (task_processed && task.is_load_mode_transfer && task.data_buffer_addr != 0) {
                // If recv_buffer_addr != data_buffer_addr, need to copy data
                if (task.recv_buffer_addr != task.data_buffer_addr) {
                    std::memcpy(
                        reinterpret_cast<void*>(task.data_buffer_addr),
                        reinterpret_cast<void*>(task.recv_buffer_addr),
                        task.size
                    );
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P recv: Copied received data to data_buffer at "
                              << task.data_buffer_addr << " (size=" << task.size << ")" << std::endl;
                } else {
                    // recv_buffer_addr == data_buffer_addr, data already in correct position, no copy needed
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Load P2P recv: Received partner_file chunk from Rank " 
                              << p2p_partner_rank_ << " (size=" << task.size << ", data already in data_buffer)" << std::endl;
                }
                
                // Step 2接收完成，查找并提交encoding任务
                PendingEncodingTask encoding_task;
                bool found_task = false;
                {
                    std::lock_guard<std::mutex> lock(pending_encoding_tasks_mutex_);
                    auto it = pending_encoding_tasks_.find(task.data_buffer_addr);
                    if (it != pending_encoding_tasks_.end()) {
                        encoding_task = it->second;
                        pending_encoding_tasks_.erase(it);
                        found_task = true;
                    }
                }
                
                if (found_task && encoding_task.data_addr != 0) {
                    // 提交encoding任务到encoding队列
                    submit_data_for_encoding_thread1(
                        encoding_task.data_addr, encoding_task.size,
                        encoding_task.encoding_addr1, encoding_task.recv_addr_thread1,
                        encoding_task.recv_chunk_size, encoding_task.parity_addr1,
                        encoding_task.p2p_own_write_addr, encoding_task.p2p_partner_write_addr
                    );
                    submit_data_for_encoding_thread2(
                        encoding_task.data_addr, encoding_task.size,
                        encoding_task.encoding_addr2, encoding_task.recv_addr_thread2,
                        encoding_task.recv_chunk_size, encoding_task.parity_addr2,
                        encoding_task.p2p_own_write_addr, encoding_task.p2p_partner_write_addr
                    );
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Step2 P2P recv completed, submitted encoding task for data_addr=" 
                              << encoding_task.data_addr << std::endl;
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] WARNING: No pending encoding task found for data_buffer=" 
                              << task.data_buffer_addr << std::endl;
                }
            } else if (task_processed && !task.is_load_mode_transfer) {
                // Save mode or Step6: original log
                if (rank_ % 2 == 0) {
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv: Received data from Rank " 
                              << p2p_partner_rank_ << std::endl;
                } else {
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv: Received parity from Rank " 
                              << p2p_partner_rank_ << std::endl;
                }
            }
            
            // Note: No buffer release needed here - recv buffer (p2p_partner_write_addr) is managed by Python
            
            // After processing task, check if sentinel was received and queue is empty
            if (p2p_recv_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(p2p_recv_queue_mutex_);
                if (p2p_recv_queue_.empty()) {
                    p2p_recv_worker_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P recv worker queue is empty after processing, marking completed" << std::endl;
                    // Reset sentinel flag and continue (don't exit)
                    p2p_recv_worker_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }

    void start_pipeline() {
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
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Started 7 threads (2 encoding + 1 send + 1 recv + 1 XOR + 2 P2P)" << std::endl;
    }

public:
    ECCHECKNative(int rank, int world_size, int paired_rank,
                  const std::vector<uint8_t>& nccl_id_xor_send,
                  const std::vector<uint8_t>& nccl_id_xor_recv,
                  const std::vector<uint8_t>& nccl_id_p2p_send,
                  const std::vector<uint8_t>& nccl_id_p2p_recv) 
        : rank_(rank), world_size_(world_size), paired_rank_(paired_rank),
          encoding_thread_1_completed_(false), encoding_thread_2_completed_(false),
          send_worker_completed_(false), recv_worker_completed_(false),
          xor_worker_completed_(false), p2p_send_worker_completed_(false), p2p_recv_worker_completed_(false),
          encoding_thread_1_sentinel_received_(false), encoding_thread_2_sentinel_received_(false),
          send_worker_sentinel_received_(false), recv_worker_sentinel_received_(false),
          xor_worker_sentinel_received_(false), p2p_send_worker_sentinel_received_(false), p2p_recv_worker_sentinel_received_(false),
          should_stop_threads_(false),
          nccl_xor_send_initialized_(false), nccl_xor_recv_initialized_(false),
          nccl_p2p_send_initialized_(false), nccl_p2p_recv_initialized_(false),
          nccl_xor_send_init_completed_(false), nccl_xor_recv_init_completed_(false),
          nccl_p2p_send_init_completed_(false), nccl_p2p_recv_init_completed_(false),
          k_(0), rows_(0), data_block_index_(0), a_mat_(nullptr), g_tbls_(nullptr),
          p2p_partner_rank_(-1),
          is_load_mode_(false), failed_rank_(-1),
          asio_initialized_(false), use_asio_(false) {

        std::cout << "EC-CHECK: [Rank " << rank_ << "] Constructor called, initializing EC tables and starting pipeline..." << std::endl;
        
        // Store four independent NCCL IDs (Python already guarantees send/recv pairing)
        nccl_id_xor_send_ = nccl_id_xor_send;
        nccl_id_xor_recv_ = nccl_id_xor_recv;
        nccl_id_p2p_send_ = nccl_id_p2p_send;
        nccl_id_p2p_recv_ = nccl_id_p2p_recv;
        
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
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] EC tables initialized (k=" << k_ << ", rows=" << rows_ << ", data_idx=" << data_block_index_ << ")" << std::endl;
                }
            }
        } else {
            std::cout << "EC-CHECK: [Rank " << rank_ << "] skipping EC init because k<=0" << std::endl;
        }

        // Initialize NCCL communicators in main thread (before starting worker threads)
        // This ensures all ranks call ncclCommInitRank simultaneously (synchronized by Python barrier)
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Initializing NCCL communicators in main thread..." << std::endl;
        init_nccl_xor_send();
        init_nccl_xor_recv();
        init_nccl_p2p_send();
        init_nccl_p2p_recv();
        std::cout << "EC-CHECK: [Rank " << rank_ << "] NCCL communicators initialization completed" << std::endl;

        // Now start worker threads (NCCL is already initialized)
        start_pipeline();

        std::cout << "EC-CHECK: [Rank " << rank_ << "] Pipeline and NCCL initialized successfully" << std::endl;
    }
    
    ~ECCHECKNative() {
        stop_pipeline();
        if (use_asio_) {
            asio_conn_mgr_.cleanup();
            // No need to stop IO context thread since we're using synchronous I/O
        } else {
            cleanup_nccl();
        }
        if (a_mat_) { free(a_mat_); a_mat_ = nullptr; }
        if (g_tbls_) { free(g_tbls_); g_tbls_ = nullptr; }
    }
    
    // ASIO constructor (new, accepts IP/Port parameters)
    ECCHECKNative(int rank, int world_size, int paired_rank,
                  const std::string& xor_partner_ip, uint16_t xor_send_port,
                  const std::string& xor_listen_ip, uint16_t xor_recv_port,
                  const std::string& p2p_partner_ip, uint16_t p2p_send_port,
                  const std::string& p2p_listen_ip, uint16_t p2p_recv_port)
        : rank_(rank), world_size_(world_size), paired_rank_(paired_rank),
          encoding_thread_1_completed_(false), encoding_thread_2_completed_(false),
          send_worker_completed_(false), recv_worker_completed_(false),
          xor_worker_completed_(false), p2p_send_worker_completed_(false), p2p_recv_worker_completed_(false),
          encoding_thread_1_sentinel_received_(false), encoding_thread_2_sentinel_received_(false),
          send_worker_sentinel_received_(false), recv_worker_sentinel_received_(false),
          xor_worker_sentinel_received_(false), p2p_send_worker_sentinel_received_(false), p2p_recv_worker_sentinel_received_(false),
          should_stop_threads_(false),
          nccl_xor_send_initialized_(false), nccl_xor_recv_initialized_(false),
          nccl_p2p_send_initialized_(false), nccl_p2p_recv_initialized_(false),
          nccl_xor_send_init_completed_(false), nccl_xor_recv_init_completed_(false),
          nccl_p2p_send_init_completed_(false), nccl_p2p_recv_init_completed_(false),
          k_(0), rows_(0), data_block_index_(0), a_mat_(nullptr), g_tbls_(nullptr),
          p2p_partner_rank_(-1),
          is_load_mode_(false), failed_rank_(-1),
          asio_initialized_(false), use_asio_(true) {

        std::cout << "EC-CHECK: [Rank " << rank_ << "] ASIO Constructor called, initializing EC tables and ASIO connections..." << std::endl;
        
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
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] EC tables initialized (k=" << k_ << ", rows=" << rows_ << ", data_idx=" << data_block_index_ << ")" << std::endl;
                }
            }
        } else {
            std::cout << "EC-CHECK: [Rank " << rank_ << "] skipping EC init because k<=0" << std::endl;
        }

        // Initialize ASIO connections in main thread (before starting worker threads)
        // Note: Using synchronous connect/accept
        // Strategy: Start accept operations in separate thread, then connect
        // This avoids deadlock when both ranks try to connect simultaneously
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Initializing ASIO connections in main thread..." << std::endl;
        
        // Start accept operations in separate threads (parallel execution)
        // This ensures both XOR and P2P accept operations start listening simultaneously
        std::thread recv_init_thread([this, xor_listen_ip, xor_recv_port, p2p_listen_ip, p2p_recv_port]() {
            // Start two parallel threads for XOR and P2P recv
            std::thread xor_recv_thread([this, xor_listen_ip, xor_recv_port]() {
                asio_conn_mgr_.init_xor_recv(xor_listen_ip, xor_recv_port);
            });
            
            std::thread p2p_recv_thread([this, p2p_listen_ip, p2p_recv_port]() {
                asio_conn_mgr_.init_p2p_recv(p2p_listen_ip, p2p_recv_port);
            });
            
            // Wait for both accept operations to complete
            xor_recv_thread.join();
            p2p_recv_thread.join();
        });
        
        // Small delay to ensure accept sockets are bound and listening
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Connect operations (will block until connected to partner's accept)
        asio_conn_mgr_.init_xor_send(xor_partner_ip, xor_send_port);
        asio_conn_mgr_.init_p2p_send(p2p_partner_ip, p2p_send_port);
        
        // Wait for accept operations to complete
        recv_init_thread.join();
        
        // Verify all connections are established
        if (asio_conn_mgr_.is_xor_send_connected() && asio_conn_mgr_.is_xor_recv_connected() &&
            asio_conn_mgr_.is_p2p_send_connected() && asio_conn_mgr_.is_p2p_recv_connected()) {
            asio_initialized_ = true;
            std::cout << "EC-CHECK: [Rank " << rank_ << "] ASIO connections initialization completed" << std::endl;
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

        std::cout << "EC-CHECK: [Rank " << rank_ << "] Pipeline and ASIO initialized successfully" << std::endl;
    }
    
    void set_buffer_addresses(const std::vector<uintptr_t>& data_addrs,
                             const std::vector<uintptr_t>& encoding_addrs,
                             const std::vector<size_t>& sizes) {
        data_buffer_addrs_ = data_addrs;
        encoding_buffer_addrs_ = encoding_addrs;
        buffer_sizes_ = sizes;
    }
    
    void reset_encoding_completion_flags() {
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
    }
    
    void wait_for_encoding_completion() {
        // Wait for all encoding threads to complete
        int encoding_wait_count = 0;
        while (!encoding_thread_1_completed_ || !encoding_thread_2_completed_) {
            if (encoding_wait_count % 100 == 0) {  // Log every 1 second (100 * 10ms)
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Waiting for encoding threads: "
                          << "thread1=" << (encoding_thread_1_completed_ ? "true" : "false")
                          << ", thread2=" << (encoding_thread_2_completed_ ? "true" : "false") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            encoding_wait_count++;
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Both encoding threads completed" << std::endl;
        
        // Wait for send worker to complete
        int send_wait_count = 0;
        while (!send_worker_completed_) {
            if (send_wait_count % 100 == 0) {
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Waiting for send worker..." << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            send_wait_count++;
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker completed" << std::endl;
        
        // Wait for recv worker to complete
        int recv_wait_count = 0;
        while (!recv_worker_completed_) {
            if (recv_wait_count % 100 == 0) {
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Waiting for recv worker..." << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            recv_wait_count++;
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker completed" << std::endl;
        
        // Wait for XOR worker to complete
        int xor_wait_count = 0;
        while (!xor_worker_completed_) {
            if (xor_wait_count % 100 == 0) {
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Waiting for XOR worker..." << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            xor_wait_count++;
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker completed" << std::endl;
        
        // Wait for both P2P workers to complete
        int p2p_wait_count = 0;
        while (!p2p_send_worker_completed_ || !p2p_recv_worker_completed_) {
            if (p2p_wait_count % 100 == 0) {  // Log every 1 second
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Waiting for P2P workers: "
                          << "send=" << (p2p_send_worker_completed_ ? "true" : "false")
                          << ", recv=" << (p2p_recv_worker_completed_ ? "true" : "false") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            p2p_wait_count++;
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Both P2P workers completed" << std::endl;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] All threads completed (encoding + send + recv + XOR + P2P)" << std::endl;
    }
    
    void stop_pipeline() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Stopping pipeline..." << std::endl;
        
        should_stop_threads_ = true;
        
        // Notify all threads
        encoding_tasks_1_cv_.notify_all();
        encoding_tasks_2_cv_.notify_all();
        send_queue_cv_.notify_all();
        recv_queue_cv_.notify_all();
        xor_queue_cv_.notify_all();
        p2p_send_queue_cv_.notify_all();
        p2p_recv_queue_cv_.notify_all();
        
        // Join all threads
        if (encoder_thread_1_.joinable()) encoder_thread_1_.join();
        if (encoder_thread_2_.joinable()) encoder_thread_2_.join();
        if (send_worker_.joinable()) send_worker_.join();
        if (recv_worker_.joinable()) recv_worker_.join();
        if (xor_worker_.joinable()) xor_worker_.join();
        if (p2p_send_worker_.joinable()) p2p_send_worker_.join();
        if (p2p_recv_worker_.joinable()) p2p_recv_worker_.join();
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Pipeline stopped" << std::endl;
    }
    
    void submit_data_for_encoding_thread1(uintptr_t data_addr, size_t size, 
                                          uintptr_t encoding_addr, uintptr_t recv_addr, 
                                          size_t recv_chunk_size, uintptr_t parity_addr,
                                          uintptr_t p2p_own_write_addr, uintptr_t p2p_partner_write_addr) {
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
            encoding_tasks_1_.push({data_addr, size, encoding_addr, recv_addr, recv_chunk_size, 
                                    parity_addr, p2p_own_write_addr, p2p_partner_write_addr});
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
                                          uintptr_t p2p_own_write_addr, uintptr_t p2p_partner_write_addr) {
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
            encoding_tasks_2_.push({data_addr, size, encoding_addr, recv_addr, recv_chunk_size,
                                    parity_addr, p2p_own_write_addr, p2p_partner_write_addr});
        }
        encoding_tasks_2_cv_.notify_one();
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
    
    void submit_data_to_p2p_thread(uintptr_t data_addr, size_t size, std::string ops) {
        if (data_addr == 0 || size == 0) {
            std::cerr << "EC-CHECK: [Rank " << rank_
                      << "] submit_data_to_p2p_thread received invalid args: addr="
                      << data_addr << ", size=" << size << std::endl;
            return;
        }
        
        if (ops == "send") {
            {
                std::lock_guard<std::mutex> lock(p2p_send_queue_mutex_);
                // Load-time P2P transfer doesn't need a local copy, so p2p_own_write_addr/parity/data are 0
                p2p_send_queue_.push({data_addr, 0, size, 0, 0, false, 0});
            }
            p2p_send_queue_cv_.notify_one();
            std::cout << "EC-CHECK: [Rank " << rank_
                      << "] Queued P2P send task (addr=" << data_addr
                      << ", size=" << size << ")" << std::endl;
        } else if (ops == "recv") {
            {
                std::lock_guard<std::mutex> lock(p2p_recv_queue_mutex_);
                p2p_recv_queue_.push({data_addr, size, false, 0});
            }
            p2p_recv_queue_cv_.notify_one();
            std::cout << "EC-CHECK: [Rank " << rank_
                      << "] Queued P2P recv task (addr=" << data_addr
                      << ", size=" << size << ")" << std::endl;
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
            std::cerr << "EC-CHECK: [Rank " << rank_
                      << "] submit_load_p2p_transfer received invalid size: " << size << std::endl;
            return;
        }
        
        if (is_sender) {
            // Sender: 提交到p2p_send_queue_
            if (send_buffer_addr == 0) {
                std::cerr << "EC-CHECK: [Rank " << rank_
                          << "] submit_load_p2p_transfer: sender requires send_buffer_addr" << std::endl;
                return;
            }
            
            {
                std::lock_guard<std::mutex> lock(p2p_send_queue_mutex_);
                p2p_send_queue_.push({
                    send_buffer_addr,    // send_buffer_addr
                    0,                   // p2p_own_write_addr (Step2不需要)
                    size,                // size
                    0,                   // parity_addr (Step2不需要)
                    0,                   // data_addr (Step2不需要)
                    true,                // is_load_mode_transfer
                    load_mode_data_addr  // load_mode_data_addr (for finding encoding task after Step2 send)
                });
            }
            p2p_send_queue_cv_.notify_one();
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Queued load P2P send task (addr=" 
                      << send_buffer_addr << ", size=" << size << ")" << std::endl;
        } else {
            // Receiver: 提交到p2p_recv_queue_
            if (recv_data_buffer_addr == 0) {
                std::cerr << "EC-CHECK: [Rank " << rank_
                          << "] submit_load_p2p_transfer: receiver requires recv_data_buffer_addr" << std::endl;
                return;
            }
            
            {
                std::lock_guard<std::mutex> lock(p2p_recv_queue_mutex_);
                p2p_recv_queue_.push({
                    recv_data_buffer_addr, // recv_buffer_addr (直接使用data_buffer作为接收buffer)
                    size,                  // size
                    true,                  // is_load_mode_transfer
                    recv_data_buffer_addr  // data_buffer_addr (接收后数据就在这个地址，无需复制)
                });
            }
            p2p_recv_queue_cv_.notify_one();
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Queued load P2P recv task (addr=" 
                      << recv_data_buffer_addr << ", size=" << size << ")" << std::endl;
        }
    }
    
    void set_load_mode(bool is_load, int failed_rank) {
        is_load_mode_ = is_load;
        failed_rank_ = failed_rank;
        // Rebuild XOR configuration (pairing is same as save, but need to update flags)
        build_xor_config();
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Set load mode: " 
                  << (is_load ? "true" : "false") << ", failed_rank=" << failed_rank << std::endl;
    }
    
    void submit_load_pipeline_chunk(
        uintptr_t step2_send_addr,        // rank0/3: partner_file地址；rank1/2: 0
        uintptr_t step2_recv_data_addr,   // rank1/2: data_buffer地址；rank0/3: 0
        size_t step2_size,                 // Step 2传输大小
        uintptr_t data_addr,               // 数据buffer地址
        size_t size,                       // 数据大小
        uintptr_t encoding_addr1,         // thread1编码buffer
        uintptr_t encoding_addr2,         // thread2编码buffer
        uintptr_t recv_addr_thread1,      // thread1接收地址
        uintptr_t recv_addr_thread2,      // thread2接收地址
        size_t recv_chunk_size,           // 接收chunk大小
        uintptr_t parity_addr1,           // thread1 parity buffer
        uintptr_t parity_addr2,           // thread2 parity buffer
        uintptr_t p2p_own_write_addr,      // P2P own buffer写入地址
        uintptr_t p2p_partner_write_addr  // P2P partner buffer写入地址
    ) {
        if (!is_load_mode_) {
            std::cerr << "EC-CHECK: [Rank " << rank_ 
                      << "] submit_load_pipeline_chunk called but not in load mode" << std::endl;
            return;
        }
        
        // 准备encoding任务信息
        PendingEncodingTask encoding_task = {
            data_addr, size, encoding_addr1, encoding_addr2,
            recv_addr_thread1, recv_addr_thread2, recv_chunk_size,
            parity_addr1, parity_addr2,
            p2p_own_write_addr, p2p_partner_write_addr
        };
        
        if (rank_ == 0 || rank_ == 3) {
            // Sender: 提交Step 2 P2P发送任务，保存encoding任务信息
            if (step2_send_addr != 0 && step2_size > 0) {
                // 提交P2P发送任务，传递data_addr用于后续查找encoding任务
                submit_load_p2p_transfer(step2_send_addr, 0, step2_size, true, data_addr);
                
                // 保存encoding任务信息，等P2P send完成后使用
                {
                    std::lock_guard<std::mutex> lock(pending_encoding_tasks_mutex_);
                    pending_encoding_tasks_[data_addr] = encoding_task;
                }
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Saved pending encoding task for data_addr=" 
                          << data_addr << " (will submit after Step2 P2P send)" << std::endl;
            }
        } else if (rank_ == 1 || rank_ == 2) {
            // Receiver: 提交Step 2 P2P接收任务，保存encoding任务信息
            if (step2_recv_data_addr != 0 && step2_size > 0) {
                // 提交P2P接收任务
                submit_load_p2p_transfer(0, step2_recv_data_addr, step2_size, false, 0);
                
                // 保存encoding任务信息，等P2P recv完成后使用
                {
                    std::lock_guard<std::mutex> lock(pending_encoding_tasks_mutex_);
                    pending_encoding_tasks_[step2_recv_data_addr] = encoding_task;
                }
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Saved pending encoding task for data_buffer=" 
                          << step2_recv_data_addr << " (will submit after Step2 P2P recv)" << std::endl;
            }
        }
    }
};

PYBIND11_MODULE(eccheck_native, m) {
    // Module-level function: Generate NCCL ID (can be called without creating an instance)
#ifdef NCCL_AVAILABLE
    m.def("generate_nccl_id", &generate_nccl_id, 
          "Generate a new NCCL unique ID. Returns a list of uint8_t bytes (128 bytes).");
#endif
    
    // Class definition
    pybind11::class_<ECCHECKNative>(m, "ECCHECKNative")
        // NCCL constructor (original)
        .def(pybind11::init<int, int, int, const std::vector<uint8_t>&, const std::vector<uint8_t>&, const std::vector<uint8_t>&, const std::vector<uint8_t>&>())
        // ASIO constructor (new)
        .def(pybind11::init<int, int, int,
             const std::string&, uint16_t,
             const std::string&, uint16_t,
             const std::string&, uint16_t,
             const std::string&, uint16_t>())
        .def("set_buffer_addresses", &ECCHECKNative::set_buffer_addresses)
        .def("reset_encoding_completion_flags", &ECCHECKNative::reset_encoding_completion_flags)
        .def("wait_for_encoding_completion", &ECCHECKNative::wait_for_encoding_completion)
        .def("stop_pipeline", &ECCHECKNative::stop_pipeline)
        .def("submit_data_for_encoding_thread1", &ECCHECKNative::submit_data_for_encoding_thread1)
        .def("submit_data_for_encoding_thread2", &ECCHECKNative::submit_data_for_encoding_thread2)
        .def("get_data_buffers_to_release", &ECCHECKNative::get_data_buffers_to_release)
        .def("get_encoding_buffers_to_release", &ECCHECKNative::get_encoding_buffers_to_release)
        .def("get_parity_buffers_to_release", &ECCHECKNative::get_parity_buffers_to_release)
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
             pybind11::arg("encoding_addr1") = 0,
             pybind11::arg("encoding_addr2") = 0,
             pybind11::arg("recv_addr_thread1") = 0,
             pybind11::arg("recv_addr_thread2") = 0,
             pybind11::arg("recv_chunk_size") = 0,
             pybind11::arg("parity_addr1") = 0,
             pybind11::arg("parity_addr2") = 0,
             pybind11::arg("p2p_own_write_addr") = 0,
             pybind11::arg("p2p_partner_write_addr") = 0);
}
