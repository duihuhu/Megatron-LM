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
#include <cstdlib>

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
    
    // Thread 1的发送队列（编码完成后放入）
    struct SendTask {
        uintptr_t encoding_addr;
        size_t size;
    };
    std::queue<SendTask> send_queue_1_;
    std::queue<SendTask> send_queue_2_;
    std::mutex send_queue_1_mutex_;
    std::mutex send_queue_2_mutex_;
    std::condition_variable send_queue_1_cv_;
    std::condition_variable send_queue_2_cv_;
    
    // 接收任务队列（Python提交）
    struct RecvTask {
        uintptr_t recv_addr;
        size_t size;
        uintptr_t parity_addr;      // XOR结果地址（从EncodingTask传递）
    };
    std::queue<RecvTask> recv_queue_1_;
    std::queue<RecvTask> recv_queue_2_;
    std::mutex recv_queue_1_mutex_;
    std::mutex recv_queue_2_mutex_;
    std::condition_variable recv_queue_1_cv_;
    std::condition_variable recv_queue_2_cv_;
    
    // Completion flags
    std::atomic<bool> encoding_thread_1_completed_;
    std::atomic<bool> encoding_thread_2_completed_;
    std::atomic<bool> send_worker_1_completed_;
    std::atomic<bool> send_worker_2_completed_;
    std::atomic<bool> recv_worker_1_completed_;
    std::atomic<bool> recv_worker_2_completed_;
    std::atomic<bool> xor_worker_1_completed_;
    std::atomic<bool> xor_worker_2_completed_;
    std::atomic<bool> p2p_worker_completed_;
    
    // Sentinel received flags (to track if sentinel was received, but queue may not be empty yet)
    std::atomic<bool> encoding_thread_1_sentinel_received_;
    std::atomic<bool> encoding_thread_2_sentinel_received_;
    std::atomic<bool> send_worker_1_sentinel_received_;
    std::atomic<bool> send_worker_2_sentinel_received_;
    std::atomic<bool> recv_worker_1_sentinel_received_;
    std::atomic<bool> recv_worker_2_sentinel_received_;
    std::atomic<bool> xor_worker_1_sentinel_received_;
    std::atomic<bool> xor_worker_2_sentinel_received_;
    std::atomic<bool> p2p_worker_sentinel_received_;
    
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
    
    // XOR任务队列（每个thread一个）
    std::queue<XORTask> xor_queue_1_;
    std::queue<XORTask> xor_queue_2_;
    std::mutex xor_queue_1_mutex_;
    std::mutex xor_queue_2_mutex_;
    std::condition_variable xor_queue_1_cv_;
    std::condition_variable xor_queue_2_cv_;
    
    // Pending encoding buffers等待XOR（用于匹配encoding和recv）
    // Key: recv_addr, Value: encoding_addr
    std::unordered_map<uintptr_t, uintptr_t> pending_xor_encoding_1_;
    std::unordered_map<uintptr_t, uintptr_t> pending_xor_encoding_2_;
    std::mutex pending_xor_1_mutex_;
    std::mutex pending_xor_2_mutex_;
    
    // Track recv_addr -> data_addr mapping (for P2P: odd ranks need to send data)
    std::unordered_map<uintptr_t, uintptr_t> recv_to_data_1_;
    std::unordered_map<uintptr_t, uintptr_t> recv_to_data_2_;
    std::mutex recv_to_data_1_mutex_;
    std::mutex recv_to_data_2_mutex_;
    
    // Track recv_addr -> parity_addr mapping
    std::unordered_map<uintptr_t, uintptr_t> recv_to_parity_1_;
    std::unordered_map<uintptr_t, uintptr_t> recv_to_parity_2_;
    std::mutex recv_to_parity_1_mutex_;
    std::mutex recv_to_parity_2_mutex_;
    
    // Track recv_addr -> P2P addresses mapping
    struct P2PAddresses {
        uintptr_t own_write_addr;
        uintptr_t partner_write_addr;
    };
    std::unordered_map<uintptr_t, P2PAddresses> recv_to_p2p_1_;
    std::unordered_map<uintptr_t, P2PAddresses> recv_to_p2p_2_;
    std::mutex recv_to_p2p_1_mutex_;
    std::mutex recv_to_p2p_2_mutex_;
    
    // P2P task structure
    struct P2PTask {
        uintptr_t parity_addr;           // Own parity (from XOR, for even ranks to send)
        uintptr_t data_addr;             // Own data (for odd ranks to send)
        uintptr_t p2p_own_write_addr;    // Write own parity/data to this address
        uintptr_t p2p_partner_write_addr; // Write received partner data/parity to this address
        size_t size;                     // Data size
    };
    
    // P2P task queue (single thread)
    std::queue<P2PTask> p2p_queue_;
    std::mutex p2p_queue_mutex_;
    std::condition_variable p2p_queue_cv_;
    
    // Mutex to protect sending sentinel to P2P worker (to avoid race condition)
    std::mutex p2p_sentinel_mutex_;
    
    // Worker threads - 每个encoding线程配备独立的send/recv/xor worker
    std::thread encoder_thread_1_;
    std::thread encoder_thread_2_;
    std::thread send_worker_1_;     // 专门发送thread1的编码数据
    std::thread recv_worker_1_;     // 专门接收给thread1的数据
    std::thread send_worker_2_;     // 专门发送thread2的编码数据
    std::thread recv_worker_2_;     // 专门接收给thread2的数据
    std::thread xor_worker_1_;      // 专门执行thread1的XOR操作
    std::thread xor_worker_2_;      // 专门执行thread2的XOR操作
    std::thread p2p_worker_;        // P2P worker (single thread)
    
    // NCCL communicators - 每个线程有独立的通信域
#ifdef NCCL_AVAILABLE
    ncclComm_t nccl_comm_thread1_;  // Thread1专用通信域
    ncclComm_t nccl_comm_thread2_;  // Thread2专用通信域
    ncclComm_t nccl_comm_p2p_;      // P2P专用通信域
    bool nccl_thread1_initialized_;
    bool nccl_thread2_initialized_;
    bool nccl_p2p_initialized_;
#ifdef __GNUC__
#endif

    // EC parameters (k, rows=2) and tables
    int k_;
    int rows_;
    int data_block_index_;
    unsigned char *a_mat_;    // RS matrix (k * m)
    unsigned char *g_tbls_;   // tables produced by ec_init_tables (32 * k * rows)
#else
    bool nccl_thread1_initialized_;
    bool nccl_thread2_initialized_;
#endif

    // NCCL IDs stored as member variables (passed from Python via broadcast)
    std::vector<uint8_t> nccl_id_thread1_;
    std::vector<uint8_t> nccl_id_thread2_;
    std::vector<uint8_t> nccl_id_p2p_;

    // Synchronization for NCCL initialization
    std::atomic<bool> nccl_thread1_init_completed_;
    std::atomic<bool> nccl_thread2_init_completed_;
    std::atomic<bool> nccl_p2p_init_completed_;
    std::mutex nccl_init_mutex_;
    std::condition_variable nccl_init_cv_;

    // P2P configuration
    int p2p_partner_rank_;  // P2P partner rank (adjacent pairing: 0<->1, 2<->3)

    // ========== XOR配置构建函数 ==========
    
    void build_xor_config() {
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
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR config - "
                  << "partner_rank=" << xor_config_.xor_partner_rank
                  << ", thread0_receiver=" << xor_config_.thread0_is_receiver
                  << ", thread1_receiver=" << xor_config_.thread1_is_receiver << std::endl;
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
    
    void init_nccl_thread1() {
#ifdef NCCL_AVAILABLE
        // Read NCCL ID from member variable (set by constructor)
        if (nccl_id_thread1_.size() != sizeof(ncclUniqueId)) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Invalid NCCL ID size for thread1: " 
                      << nccl_id_thread1_.size() << " (expected " << sizeof(ncclUniqueId) << ")" << std::endl;
            nccl_thread1_initialized_ = false;
            nccl_thread1_init_completed_ = true;
            nccl_init_cv_.notify_all();
            return;
        }
        
        ncclUniqueId nccl_id;
        std::memcpy(&nccl_id, nccl_id_thread1_.data(), sizeof(ncclUniqueId));
        
        // Initialize NCCL communicator
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread1 calling ncclCommInitRank..." << std::endl;
        std::cout.flush();
        ncclCommInitRank(&nccl_comm_thread1_, world_size_, nccl_id, rank_);
        nccl_thread1_initialized_ = true;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread1 NCCL communicator initialized" << std::endl;
        std::cout.flush();
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_thread1_init_completed_ = true;
        }
        nccl_init_cv_.notify_all();
#else
        std::cout << "EC-CHECK: NCCL not available, skipping thread1 initialization" << std::endl;
        nccl_thread1_initialized_ = false;
        nccl_thread1_init_completed_ = true;
        nccl_init_cv_.notify_all();
#endif
    }
    
    void init_nccl_thread2() {
#ifdef NCCL_AVAILABLE
        // Read NCCL ID from member variable (set by constructor)
        if (nccl_id_thread2_.size() != sizeof(ncclUniqueId)) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Invalid NCCL ID size for thread2: " 
                      << nccl_id_thread2_.size() << " (expected " << sizeof(ncclUniqueId) << ")" << std::endl;
            nccl_thread2_initialized_ = false;
            nccl_thread2_init_completed_ = true;
            nccl_init_cv_.notify_all();
            return;
        }
        
        ncclUniqueId nccl_id;
        std::memcpy(&nccl_id, nccl_id_thread2_.data(), sizeof(ncclUniqueId));
        
        // Initialize NCCL communicator
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread2 calling ncclCommInitRank..." << std::endl;
        std::cout.flush();
        ncclCommInitRank(&nccl_comm_thread2_, world_size_, nccl_id, rank_);
        nccl_thread2_initialized_ = true;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread2 NCCL communicator initialized" << std::endl;
        std::cout.flush();
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_thread2_init_completed_ = true;
        }
        nccl_init_cv_.notify_all();
#else
        std::cout << "EC-CHECK: NCCL not available, skipping thread2 initialization" << std::endl;
        nccl_thread2_initialized_ = false;
        nccl_thread2_init_completed_ = true;
        nccl_init_cv_.notify_all();
#endif
    }
    
    void init_nccl_p2p() {
#ifdef NCCL_AVAILABLE
        // Read NCCL ID from member variable (set by constructor)
        if (nccl_id_p2p_.size() != sizeof(ncclUniqueId)) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Invalid NCCL ID size for P2P: " 
                      << nccl_id_p2p_.size() << " (expected " << sizeof(ncclUniqueId) << ")" << std::endl;
            nccl_p2p_initialized_ = false;
            nccl_p2p_init_completed_ = true;
            nccl_init_cv_.notify_all();
            return;
        }
        
        ncclUniqueId nccl_id;
        std::memcpy(&nccl_id, nccl_id_p2p_.data(), sizeof(ncclUniqueId));
        
        // Initialize NCCL communicator for P2P
        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P calling ncclCommInitRank..." << std::endl;
        std::cout.flush();
        ncclCommInitRank(&nccl_comm_p2p_, world_size_, nccl_id, rank_);
        nccl_p2p_initialized_ = true;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P NCCL communicator initialized" << std::endl;
        std::cout.flush();
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_p2p_init_completed_ = true;
        }
        nccl_init_cv_.notify_all();
#else
        std::cout << "EC-CHECK: NCCL not available, skipping P2P initialization" << std::endl;
        nccl_p2p_initialized_ = false;
        nccl_p2p_init_completed_ = true;
        nccl_init_cv_.notify_all();
#endif
    }
    
    void cleanup_nccl() {
#ifdef NCCL_AVAILABLE
        if (nccl_thread1_initialized_) {
            ncclCommDestroy(nccl_comm_thread1_);
        }
        if (nccl_thread2_initialized_) {
            ncclCommDestroy(nccl_comm_thread2_);
        }
        if (nccl_p2p_initialized_) {
            ncclCommDestroy(nccl_comm_p2p_);
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
                            std::lock_guard<std::mutex> send_lock(send_queue_1_mutex_);
                    send_queue_1_.push({0, 0});
                }
                send_queue_1_cv_.notify_one();
                {
                            std::lock_guard<std::mutex> recv_lock(recv_queue_1_mutex_);
                            recv_queue_1_.push({0, 0, 0});
                }
                recv_queue_1_cv_.notify_one();
                        {
                            std::lock_guard<std::mutex> xor_lock(xor_queue_1_mutex_);
                            xor_queue_1_.push({0, 0, 0, 0, 0, 0, 0});
                        }
                        xor_queue_1_cv_.notify_one();
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
                        std::lock_guard<std::mutex> lock(pending_xor_1_mutex_);
                        pending_xor_encoding_1_[task.recv_addr] = task.encoding_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_parity_1_mutex_);
                        recv_to_parity_1_[task.recv_addr] = task.parity_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_data_1_mutex_);
                        recv_to_data_1_[task.recv_addr] = task.data_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_p2p_1_mutex_);
                        recv_to_p2p_1_[task.recv_addr] = {task.p2p_own_write_addr, task.p2p_partner_write_addr};
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
                    std::lock_guard<std::mutex> lock(send_queue_1_mutex_);
                    send_queue_1_.push({task.encoding_addr, task.size});
                }
                send_queue_1_cv_.notify_one();
                    
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
                // Submit recv task to recv_worker_1 with parity_addr
                {
                    std::lock_guard<std::mutex> lock(recv_queue_1_mutex_);
                    recv_queue_1_.push({task.recv_addr, task.recv_chunk_size, task.parity_addr});
                }
                recv_queue_1_cv_.notify_one();
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (encoding_thread_1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
                if (encoding_tasks_1_.empty()) {
                    encoding_thread_1_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 1 queue is empty after processing, marking completed" << std::endl;
                    // Submit sentinel to downstream workers
                    {
                        std::lock_guard<std::mutex> send_lock(send_queue_1_mutex_);
                        send_queue_1_.push({0, 0});
                    }
                    send_queue_1_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> recv_lock(recv_queue_1_mutex_);
                        recv_queue_1_.push({0, 0, 0});
                    }
                    recv_queue_1_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> xor_lock(xor_queue_1_mutex_);
                        xor_queue_1_.push({0, 0, 0, 0, 0, 0, 0});
                    }
                    xor_queue_1_cv_.notify_one();
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
                            std::lock_guard<std::mutex> send_lock(send_queue_2_mutex_);
                    send_queue_2_.push({0, 0});
                }
                send_queue_2_cv_.notify_one();
                {
                            std::lock_guard<std::mutex> recv_lock(recv_queue_2_mutex_);
                            recv_queue_2_.push({0, 0, 0});
                }
                recv_queue_2_cv_.notify_one();
                        {
                            std::lock_guard<std::mutex> xor_lock(xor_queue_2_mutex_);
                            xor_queue_2_.push({0, 0, 0, 0, 0, 0, 0});
                        }
                        xor_queue_2_cv_.notify_one();
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
                        std::lock_guard<std::mutex> lock(pending_xor_2_mutex_);
                        pending_xor_encoding_2_[task.recv_addr] = task.encoding_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_parity_2_mutex_);
                        recv_to_parity_2_[task.recv_addr] = task.parity_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_data_2_mutex_);
                        recv_to_data_2_[task.recv_addr] = task.data_addr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(recv_to_p2p_2_mutex_);
                        recv_to_p2p_2_[task.recv_addr] = {task.p2p_own_write_addr, task.p2p_partner_write_addr};
                    }
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread2 encoding completed, "
                              << "pending XOR for recv_addr=" << task.recv_addr 
                              << ", encoding_addr=" << task.encoding_addr
                              << ", p2p_own=" << task.p2p_own_write_addr
                              << ", p2p_partner=" << task.p2p_partner_write_addr << std::endl;
                } else {
                    // This thread is sender: send encoding result immediately
                {
                    std::lock_guard<std::mutex> lock(send_queue_2_mutex_);
                    send_queue_2_.push({task.encoding_addr, task.size});
                }
                send_queue_2_cv_.notify_one();
                    
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
                // Submit recv task to recv_worker_2 with parity_addr
                {
                    std::lock_guard<std::mutex> lock(recv_queue_2_mutex_);
                    recv_queue_2_.push({task.recv_addr, task.recv_chunk_size, task.parity_addr});
                }
                recv_queue_2_cv_.notify_one();
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (encoding_thread_2_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
                if (encoding_tasks_2_.empty()) {
                    encoding_thread_2_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 queue is empty after processing, marking completed" << std::endl;
                    // Submit sentinel to downstream workers
                    {
                        std::lock_guard<std::mutex> send_lock(send_queue_2_mutex_);
                        send_queue_2_.push({0, 0});
                    }
                    send_queue_2_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> recv_lock(recv_queue_2_mutex_);
                        recv_queue_2_.push({0, 0, 0});
                    }
                    recv_queue_2_cv_.notify_one();
                    {
                        std::lock_guard<std::mutex> xor_lock(xor_queue_2_mutex_);
                        xor_queue_2_.push({0, 0, 0, 0, 0, 0, 0});
                    }
                    xor_queue_2_cv_.notify_one();
                    // Reset sentinel flag and continue (don't exit)
                    encoding_thread_2_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }
    
    // Send Worker 1 - 专门发送thread1的编码数据
    void send_worker_1() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 1 started" << std::endl;
        
        // Initialize NCCL for thread1
        init_nccl_thread1();
        
        while (!should_stop_threads_) {
            SendTask task;
            
            {
                std::unique_lock<std::mutex> lock(send_queue_1_mutex_);
                send_queue_1_cv_.wait(lock, [this] {
                    return !send_queue_1_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && send_queue_1_.empty()) {
                    break;
                }
                
                task = send_queue_1_.front();
                send_queue_1_.pop();
            }
            
            // Check for sentinel
            if (task.encoding_addr == 0 && task.size == 0) {
                send_worker_1_sentinel_received_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 1 received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(send_queue_1_mutex_);
                    if (send_queue_1_.empty()) {
                send_worker_1_completed_ = true;
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 1 queue is empty, marking completed" << std::endl;
                        // Reset sentinel flag and continue (don't exit)
                        send_worker_1_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread1_initialized_ && world_size_ > 1) {
                // Determine target rank: if this thread is receiver, send to paired_rank
                // Otherwise, send to xor_partner_rank
                int target_rank = xor_config_.thread0_is_receiver ? paired_rank_ : xor_config_.xor_partner_rank;
                
                ncclGroupStart(); 
                ncclSend(reinterpret_cast<void*>(task.encoding_addr), task.size, 
                         ncclUint8, target_rank, nccl_comm_thread1_, 0);
                ncclGroupEnd();
                // Synchronize to ensure NCCL operation completes
                cudaDeviceSynchronize();
            }
#endif

            // Release encoding buffer
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                encoding_buffers_to_release_.push(task.encoding_addr);
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (send_worker_1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(send_queue_1_mutex_);
                if (send_queue_1_.empty()) {
                    send_worker_1_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 1 queue is empty after processing, marking completed" << std::endl;
                    // Reset sentinel flag and continue (don't exit)
                    send_worker_1_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }
    
    // Recv Worker 1 - 专门接收给thread1的数据
    void recv_worker_1() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 1 started" << std::endl;
        
        // Wait for NCCL initialization (already done by send_worker_1)
        {
            std::unique_lock<std::mutex> lock(nccl_init_mutex_);
            nccl_init_cv_.wait(lock, [this] {
                return nccl_thread1_init_completed_.load();
            });
        }
        
        while (!should_stop_threads_) {
            RecvTask task;
            
            {
                std::unique_lock<std::mutex> lock(recv_queue_1_mutex_);
                recv_queue_1_cv_.wait(lock, [this] {
                    return !recv_queue_1_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && recv_queue_1_.empty()) {
                    break;
                }
                
                task = recv_queue_1_.front();
                recv_queue_1_.pop();
            }
            
            // Check for sentinel
            if (task.recv_addr == 0 && task.size == 0) {
                recv_worker_1_sentinel_received_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 1 received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(recv_queue_1_mutex_);
                    if (recv_queue_1_.empty()) {
                        recv_worker_1_completed_ = true;
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 1 queue is empty, marking completed" << std::endl;
                        // Reset sentinel flag and continue (don't exit)
                        recv_worker_1_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread1_initialized_ && world_size_ > 1) {
                // Determine source rank: if receiver, receive from xor_partner_rank
                // Otherwise, receive from paired_rank
                int source_rank = xor_config_.thread0_is_receiver ? xor_config_.xor_partner_rank : paired_rank_;
                
                ncclGroupStart();
                ncclRecv(reinterpret_cast<void*>(task.recv_addr), task.size,
                         ncclUint8, source_rank, nccl_comm_thread1_, 0);
                ncclGroupEnd();
                // Synchronize to ensure NCCL operation completes
                cudaDeviceSynchronize();
            }
#endif
            
            // If this recv is for XOR, trigger XOR worker
            if (xor_config_.thread0_is_receiver) {
                uintptr_t local_encoding_addr = 0;
                uintptr_t parity_addr = 0;
                uintptr_t data_addr = 0;
                uintptr_t p2p_own_write_addr = 0;
                uintptr_t p2p_partner_write_addr = 0;
                
                // Find corresponding encoding buffer
                {
                    std::lock_guard<std::mutex> lock(pending_xor_1_mutex_);
                    auto it = pending_xor_encoding_1_.find(task.recv_addr);
                    if (it != pending_xor_encoding_1_.end()) {
                        local_encoding_addr = it->second;
                        pending_xor_encoding_1_.erase(it);
                    }
                }
                
                // Get parity buffer
                {
                    std::lock_guard<std::mutex> lock(recv_to_parity_1_mutex_);
                    auto it = recv_to_parity_1_.find(task.recv_addr);
                    if (it != recv_to_parity_1_.end()) {
                        parity_addr = it->second;
                        recv_to_parity_1_.erase(it);
                    }
                }
                
                // Get data buffer (for P2P: odd ranks need to send data)
                {
                    std::lock_guard<std::mutex> lock(recv_to_data_1_mutex_);
                    auto it = recv_to_data_1_.find(task.recv_addr);
                    if (it != recv_to_data_1_.end()) {
                        data_addr = it->second;
                        recv_to_data_1_.erase(it);
                    }
                }
                
                // Get P2P addresses
                {
                    std::lock_guard<std::mutex> lock(recv_to_p2p_1_mutex_);
                    auto it = recv_to_p2p_1_.find(task.recv_addr);
                    if (it != recv_to_p2p_1_.end()) {
                        p2p_own_write_addr = it->second.own_write_addr;
                        p2p_partner_write_addr = it->second.partner_write_addr;
                        recv_to_p2p_1_.erase(it);
                    }
                }
                
                if (local_encoding_addr != 0 && parity_addr != 0) {
                    // Submit XOR task with P2P addresses and data address
                    {
                        std::lock_guard<std::mutex> lock(xor_queue_1_mutex_);
                        xor_queue_1_.push({
                            local_encoding_addr,      // local encoded
                            task.recv_addr,          // remote encoded (received)
                            parity_addr,              // XOR result
                            task.size,
                            p2p_own_write_addr,      // P2P own buffer address
                            p2p_partner_write_addr,  // P2P partner buffer address
                            data_addr                // data address (for odd ranks)
                        });
                    }
                    xor_queue_1_cv_.notify_one();
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread1 recv completed, "
                              << "triggered XOR: local=" << local_encoding_addr 
                              << ", remote=" << task.recv_addr 
                              << ", parity=" << parity_addr
                              << ", p2p_own=" << p2p_own_write_addr
                              << ", p2p_partner=" << p2p_partner_write_addr << std::endl;
                }
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (recv_worker_1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_queue_1_mutex_);
                if (recv_queue_1_.empty()) {
                    recv_worker_1_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 1 queue is empty after processing, marking completed" << std::endl;
                    // Reset sentinel flag and continue (don't exit)
                    recv_worker_1_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }
    
    // Send Worker 2 - 专门发送thread2的编码数据
    void send_worker_2() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 2 started" << std::endl;
        
        // Initialize NCCL for thread2
        init_nccl_thread2();
        
        while (!should_stop_threads_) {
            SendTask task;
            
            {
                std::unique_lock<std::mutex> lock(send_queue_2_mutex_);
                send_queue_2_cv_.wait(lock, [this] {
                    return !send_queue_2_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && send_queue_2_.empty()) {
                    break;
                }
                
                task = send_queue_2_.front();
                send_queue_2_.pop();
            }
            
            // Check for sentinel
            if (task.encoding_addr == 0 && task.size == 0) {
                send_worker_2_sentinel_received_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 2 received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(send_queue_2_mutex_);
                    if (send_queue_2_.empty()) {
                        send_worker_2_completed_ = true;
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 2 queue is empty, marking completed" << std::endl;
                        // Reset sentinel flag and continue (don't exit)
                        send_worker_2_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread2_initialized_ && world_size_ > 1) {
                // Determine target rank: if this thread is receiver, send to paired_rank
                // Otherwise, send to xor_partner_rank
                int target_rank = xor_config_.thread1_is_receiver ? paired_rank_ : xor_config_.xor_partner_rank;
                
                ncclGroupStart();
                ncclSend(reinterpret_cast<void*>(task.encoding_addr), task.size,
                         ncclUint8, target_rank, nccl_comm_thread2_, 0);
                ncclGroupEnd();
                // Synchronize to ensure NCCL operation completes
                cudaDeviceSynchronize();
            }
#endif

            // Release encoding buffer
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                encoding_buffers_to_release_.push(task.encoding_addr);
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (send_worker_2_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(send_queue_2_mutex_);
                if (send_queue_2_.empty()) {
                    send_worker_2_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 2 queue is empty after processing, marking completed" << std::endl;
                    // Reset sentinel flag and continue (don't exit)
                    send_worker_2_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }
    
    // Recv Worker 2 - 专门接收给thread2的数据
    void recv_worker_2() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 2 started" << std::endl;
        
        // Wait for NCCL initialization (already done by send_worker_2)
        {
            std::unique_lock<std::mutex> lock(nccl_init_mutex_);
            nccl_init_cv_.wait(lock, [this] {
                return nccl_thread2_init_completed_.load();
            });
        }
        
        while (!should_stop_threads_) {
            RecvTask task;
            
            {
                std::unique_lock<std::mutex> lock(recv_queue_2_mutex_);
                recv_queue_2_cv_.wait(lock, [this] {
                    return !recv_queue_2_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && recv_queue_2_.empty()) {
                    break;
                }
                
                task = recv_queue_2_.front();
                recv_queue_2_.pop();
            }
            
            // Check for sentinel
            if (task.recv_addr == 0 && task.size == 0) {
                recv_worker_2_sentinel_received_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 2 received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(recv_queue_2_mutex_);
                    if (recv_queue_2_.empty()) {
                        recv_worker_2_completed_ = true;
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 2 queue is empty, marking completed" << std::endl;
                        // Reset sentinel flag and continue (don't exit)
                        recv_worker_2_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread2_initialized_ && world_size_ > 1) {
                // Determine source rank: if receiver, receive from xor_partner_rank
                // Otherwise, receive from paired_rank
                int source_rank = xor_config_.thread1_is_receiver ? xor_config_.xor_partner_rank : paired_rank_;
                
                ncclGroupStart();
                ncclRecv(reinterpret_cast<void*>(task.recv_addr), task.size,
                         ncclUint8, source_rank, nccl_comm_thread2_, 0);
                ncclGroupEnd();
                // Synchronize to ensure NCCL operation completes
                cudaDeviceSynchronize();
            }
#endif
            
            // If this recv is for XOR, trigger XOR worker
            if (xor_config_.thread1_is_receiver) {
                uintptr_t local_encoding_addr = 0;
                uintptr_t parity_addr = 0;
                uintptr_t data_addr = 0;
                uintptr_t p2p_own_write_addr = 0;
                uintptr_t p2p_partner_write_addr = 0;
                
                // Find corresponding encoding buffer
                {
                    std::lock_guard<std::mutex> lock(pending_xor_2_mutex_);
                    auto it = pending_xor_encoding_2_.find(task.recv_addr);
                    if (it != pending_xor_encoding_2_.end()) {
                        local_encoding_addr = it->second;
                        pending_xor_encoding_2_.erase(it);
                    }
                }
                
                // Get parity buffer
                {
                    std::lock_guard<std::mutex> lock(recv_to_parity_2_mutex_);
                    auto it = recv_to_parity_2_.find(task.recv_addr);
                    if (it != recv_to_parity_2_.end()) {
                        parity_addr = it->second;
                        recv_to_parity_2_.erase(it);
                    }
                }
                
                // Get data buffer (for P2P: odd ranks need to send data)
                {
                    std::lock_guard<std::mutex> lock(recv_to_data_2_mutex_);
                    auto it = recv_to_data_2_.find(task.recv_addr);
                    if (it != recv_to_data_2_.end()) {
                        data_addr = it->second;
                        recv_to_data_2_.erase(it);
                    }
                }
                
                // Get P2P addresses
                {
                    std::lock_guard<std::mutex> lock(recv_to_p2p_2_mutex_);
                    auto it = recv_to_p2p_2_.find(task.recv_addr);
                    if (it != recv_to_p2p_2_.end()) {
                        p2p_own_write_addr = it->second.own_write_addr;
                        p2p_partner_write_addr = it->second.partner_write_addr;
                        recv_to_p2p_2_.erase(it);
                    }
                }
                
                if (local_encoding_addr != 0 && parity_addr != 0) {
                    // Submit XOR task with P2P addresses and data address
                    {
                        std::lock_guard<std::mutex> lock(xor_queue_2_mutex_);
                        xor_queue_2_.push({
                            local_encoding_addr,      // local encoded
                            task.recv_addr,           // remote encoded (received)
                            parity_addr,               // XOR result
                            task.size,
                            p2p_own_write_addr,       // P2P own buffer address
                            p2p_partner_write_addr,   // P2P partner buffer address
                            data_addr                 // data address (for odd ranks)
                        });
                    }
                    xor_queue_2_cv_.notify_one();
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread2 recv completed, "
                              << "triggered XOR: local=" << local_encoding_addr 
                              << ", remote=" << task.recv_addr 
                              << ", parity=" << parity_addr
                              << ", p2p_own=" << p2p_own_write_addr
                              << ", p2p_partner=" << p2p_partner_write_addr << std::endl;
                }
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (recv_worker_2_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_queue_2_mutex_);
                if (recv_queue_2_.empty()) {
                    recv_worker_2_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 2 queue is empty after processing, marking completed" << std::endl;
                    // Reset sentinel flag and continue (don't exit)
                    recv_worker_2_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }

    // XOR Worker 1 - 专门执行thread1的XOR操作
    void xor_worker_1() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 1 started" << std::endl;
        
        while (!should_stop_threads_) {
            XORTask task;
            
            {
                std::unique_lock<std::mutex> lock(xor_queue_1_mutex_);
                xor_queue_1_cv_.wait(lock, [this] {
                    return !xor_queue_1_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && xor_queue_1_.empty()) {
                    break;
                }
                
                task = xor_queue_1_.front();
                xor_queue_1_.pop();
            }
            
            // Check for sentinel
            if (task.local_encoding_addr == 0 && task.remote_encoding_addr == 0 && 
                task.parity_addr == 0 && task.size == 0 &&
                task.p2p_own_write_addr == 0 && task.p2p_partner_write_addr == 0 &&
                task.data_addr == 0) {
                xor_worker_1_sentinel_received_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 1 received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(xor_queue_1_mutex_);
                    if (xor_queue_1_.empty()) {
                        xor_worker_1_completed_ = true;
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 1 queue is empty, marking completed" << std::endl;
                        // Submit sentinel to P2P worker only when both XOR workers complete
                        // Use mutex to ensure only one worker sends the sentinel
                        {
                            std::lock_guard<std::mutex> p2p_sentinel_lock(p2p_sentinel_mutex_);
                            if (xor_worker_1_completed_.load() && xor_worker_2_completed_.load()) {
                                {
                                    std::lock_guard<std::mutex> p2p_lock(p2p_queue_mutex_);
                                    p2p_queue_.push({0, 0, 0, 0, 0});
                                }
                                p2p_queue_cv_.notify_one();
                                std::cout << "EC-CHECK: [Rank " << rank_ << "] Both XOR workers completed, sentinel sent to P2P worker" << std::endl;
                            }
                        }
                        // Reset sentinel flag and continue (don't exit)
                        xor_worker_1_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;  // If queue is not empty, continue processing remaining tasks
            }
            
            // Perform XOR using isa-l xor_gen
            unsigned char* srcs[2];
            srcs[0] = reinterpret_cast<unsigned char*>(task.local_encoding_addr);
            srcs[1] = reinterpret_cast<unsigned char*>(task.remote_encoding_addr);
            unsigned char* dest = reinterpret_cast<unsigned char*>(task.parity_addr);
            
            void* xor_array[3];
            xor_array[0] = srcs[0];
            xor_array[1] = srcs[1];
            xor_array[2] = dest;
            
            // Call isa-l xor_gen: vects=3 (2 sources + 1 dest), len=size
            xor_gen(3, (int)task.size, xor_array);
            
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread1 XOR completed, parity at " 
                      << task.parity_addr 
                      << ", P2P addresses: own=" << task.p2p_own_write_addr
                      << ", partner=" << task.p2p_partner_write_addr << std::endl;
            
            // Submit P2P task after XOR completes
            if (task.p2p_own_write_addr != 0 && task.p2p_partner_write_addr != 0 && task.parity_addr != 0) {
                {
                    std::lock_guard<std::mutex> lock(p2p_queue_mutex_);
                    p2p_queue_.push({
                        task.parity_addr,              // Own parity (for even ranks to send)
                        task.data_addr,                // Own data (for odd ranks to send)
                        task.p2p_own_write_addr,       // Write own parity/data here
                        task.p2p_partner_write_addr,   // Write received partner data/parity here
                        task.size
                    });
                }
                p2p_queue_cv_.notify_one();
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread1 XOR completed, submitted P2P task: "
                          << "parity=" << task.parity_addr
                          << ", data=" << task.data_addr
                          << ", own_write=" << task.p2p_own_write_addr
                          << ", partner_write=" << task.p2p_partner_write_addr << std::endl;
            }
            
            // Release encoding buffers (both local and remote) after XOR
            // Note: parity buffer will be released by P2P worker after it's copied
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                encoding_buffers_to_release_.push(task.local_encoding_addr);
                //encoding_buffers_to_release_.push(task.remote_encoding_addr);
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (xor_worker_1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(xor_queue_1_mutex_);
                if (xor_queue_1_.empty()) {
                    xor_worker_1_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 1 queue is empty after processing, marking completed" << std::endl;
                    // Submit sentinel to P2P worker only when both XOR workers complete
                    // Use mutex to ensure only one worker sends the sentinel
                    {
                        std::lock_guard<std::mutex> p2p_sentinel_lock(p2p_sentinel_mutex_);
                        if (xor_worker_1_completed_.load() && xor_worker_2_completed_.load()) {
                            {
                                std::lock_guard<std::mutex> p2p_lock(p2p_queue_mutex_);
                                p2p_queue_.push({0, 0, 0, 0, 0});
                            }
                            p2p_queue_cv_.notify_one();
                            std::cout << "EC-CHECK: [Rank " << rank_ << "] Both XOR workers completed, sentinel sent to P2P worker" << std::endl;
                        }
                    }
                    // Reset sentinel flag and continue (don't exit)
                    xor_worker_1_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }
    
    // XOR Worker 2 - 专门执行thread2的XOR操作
    void xor_worker_2() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 2 started" << std::endl;
        
        while (!should_stop_threads_) {
            XORTask task;
            
            {
                std::unique_lock<std::mutex> lock(xor_queue_2_mutex_);
                xor_queue_2_cv_.wait(lock, [this] {
                    return !xor_queue_2_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && xor_queue_2_.empty()) {
                    break;
                }
                
                task = xor_queue_2_.front();
                xor_queue_2_.pop();
            }
            
            // Check for sentinel
            if (task.local_encoding_addr == 0 && task.remote_encoding_addr == 0 && 
                task.parity_addr == 0 && task.size == 0 &&
                task.p2p_own_write_addr == 0 && task.p2p_partner_write_addr == 0 &&
                task.data_addr == 0) {
                xor_worker_2_sentinel_received_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 2 received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(xor_queue_2_mutex_);
                    if (xor_queue_2_.empty()) {
                        xor_worker_2_completed_ = true;
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 2 queue is empty, marking completed" << std::endl;
                        // Submit sentinel to P2P worker only when both XOR workers complete
                        // Use mutex to ensure only one worker sends the sentinel
                        {
                            std::lock_guard<std::mutex> p2p_sentinel_lock(p2p_sentinel_mutex_);
                            if (xor_worker_1_completed_.load() && xor_worker_2_completed_.load()) {
                                {
                                    std::lock_guard<std::mutex> p2p_lock(p2p_queue_mutex_);
                                    p2p_queue_.push({0, 0, 0, 0, 0});
                                }
                                p2p_queue_cv_.notify_one();
                                std::cout << "EC-CHECK: [Rank " << rank_ << "] Both XOR workers completed, sentinel sent to P2P worker" << std::endl;
                            }
                        }
                        // Reset sentinel flag and continue (don't exit)
                        xor_worker_2_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;  // If queue is not empty, continue processing remaining tasks
            }
            
            // Perform XOR using isa-l xor_gen
            unsigned char* srcs[2];
            srcs[0] = reinterpret_cast<unsigned char*>(task.local_encoding_addr);
            srcs[1] = reinterpret_cast<unsigned char*>(task.remote_encoding_addr);
            unsigned char* dest = reinterpret_cast<unsigned char*>(task.parity_addr);
            
            void* xor_array[3];
            xor_array[0] = srcs[0];
            xor_array[1] = srcs[1];
            xor_array[2] = dest;
            
            // Call isa-l xor_gen: vects=3 (2 sources + 1 dest), len=size
            xor_gen(3, (int)task.size, xor_array);
            
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread2 XOR completed, parity at " 
                      << task.parity_addr
                      << ", P2P addresses: own=" << task.p2p_own_write_addr
                      << ", partner=" << task.p2p_partner_write_addr << std::endl;
            
            // Submit P2P task after XOR completes
            if (task.p2p_own_write_addr != 0 && task.p2p_partner_write_addr != 0 && task.parity_addr != 0) {
                {
                    std::lock_guard<std::mutex> lock(p2p_queue_mutex_);
                    p2p_queue_.push({
                        task.parity_addr,              // Own parity (for even ranks to send)
                        task.data_addr,                // Own data (for odd ranks to send)
                        task.p2p_own_write_addr,       // Write own parity/data here
                        task.p2p_partner_write_addr,   // Write received partner data/parity here
                        task.size
                    });
                }
                p2p_queue_cv_.notify_one();
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread2 XOR completed, submitted P2P task: "
                          << "parity=" << task.parity_addr
                          << ", data=" << task.data_addr
                          << ", own_write=" << task.p2p_own_write_addr
                          << ", partner_write=" << task.p2p_partner_write_addr << std::endl;
            }
            
            // Release encoding buffers (both local and remote) after XOR
            // Note: parity buffer will be released by P2P worker after it's copied
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                encoding_buffers_to_release_.push(task.local_encoding_addr);
                //encoding_buffers_to_release_.push(task.remote_encoding_addr);
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (xor_worker_2_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(xor_queue_2_mutex_);
                if (xor_queue_2_.empty()) {
                    xor_worker_2_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 2 queue is empty after processing, marking completed" << std::endl;
                    // Submit sentinel to P2P worker only when both XOR workers complete
                    // Use mutex to ensure only one worker sends the sentinel
                    {
                        std::lock_guard<std::mutex> p2p_sentinel_lock(p2p_sentinel_mutex_);
                        if (xor_worker_1_completed_.load() && xor_worker_2_completed_.load()) {
                            {
                                std::lock_guard<std::mutex> p2p_lock(p2p_queue_mutex_);
                                p2p_queue_.push({0, 0, 0, 0, 0});
                            }
                            p2p_queue_cv_.notify_one();
                            std::cout << "EC-CHECK: [Rank " << rank_ << "] Both XOR workers completed, sentinel sent to P2P worker" << std::endl;
                        }
                    }
                    // Reset sentinel flag and continue (don't exit)
                    xor_worker_2_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }
    
    // P2P Worker - 处理P2P通信（单线程）
    void p2p_worker() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P worker started" << std::endl;
        
        // Initialize NCCL for P2P
        init_nccl_p2p();
        
        while (!should_stop_threads_) {
            P2PTask task;
            
            {
                std::unique_lock<std::mutex> lock(p2p_queue_mutex_);
                p2p_queue_cv_.wait(lock, [this] {
                    return !p2p_queue_.empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && p2p_queue_.empty()) {
                    break;
                }
                
                task = p2p_queue_.front();
                p2p_queue_.pop();
            }
            
            // Check for sentinel
            if (task.parity_addr == 0 && task.data_addr == 0 && task.p2p_own_write_addr == 0 && 
                task.p2p_partner_write_addr == 0 && task.size == 0) {
                p2p_worker_sentinel_received_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P worker received sentinel, waiting for queue to empty" << std::endl;
                // Check if queue is empty now
                {
                    std::lock_guard<std::mutex> lock(p2p_queue_mutex_);
                    if (p2p_queue_.empty()) {
                        p2p_worker_completed_ = true;
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P worker queue is empty, marking completed" << std::endl;
                        // Reset sentinel flag and continue (don't exit)
                        p2p_worker_sentinel_received_ = false;
                        continue;
                    }
                }
                continue;
            }
            
            // Step 1: Copy own data/parity to own_buffer
            if (task.p2p_own_write_addr != 0 && task.size > 0) {
                if (rank_ % 2 == 0) {
                    // Even rank: copy parity to own_buffer
                    if (task.parity_addr != 0) {
                        std::memcpy(reinterpret_cast<void*>(task.p2p_own_write_addr),
                                   reinterpret_cast<void*>(task.parity_addr),
                                   task.size);
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P: Copied own parity to own_buffer at "
                                  << task.p2p_own_write_addr << std::endl;
                    }
                } else {
                    // Odd rank: copy data to own_buffer
                    if (task.data_addr != 0) {
                        std::memcpy(reinterpret_cast<void*>(task.p2p_own_write_addr),
                                   reinterpret_cast<void*>(task.data_addr),
                                   task.size);
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P: Copied own data to own_buffer at "
                                  << task.p2p_own_write_addr << std::endl;
                    }
                }
            }
            
            // Step 2: P2P communication
            // Behavior depends on rank:
            // - Even ranks (0, 2): receive data from partner, send parity to partner
            // - Odd ranks (1, 3): send data to partner, receive parity from partner
            if (p2p_partner_rank_ >= 0 && task.size > 0) {
#ifdef NCCL_AVAILABLE
                if (nccl_p2p_initialized_ && world_size_ > 1) {
                    if (rank_ % 2 == 0) {
                        // Even rank: receive data from partner, send parity to partner
                        ncclGroupStart();
                        ncclRecv(reinterpret_cast<void*>(task.p2p_partner_write_addr), task.size,
                                ncclUint8, p2p_partner_rank_, nccl_comm_p2p_, 0);
                        ncclSend(reinterpret_cast<void*>(task.parity_addr), task.size,
                                ncclUint8, p2p_partner_rank_, nccl_comm_p2p_, 0);
                        ncclGroupEnd();
                        // Synchronize to ensure NCCL operation completes
                        cudaDeviceSynchronize();
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P: Received data from Rank " 
                                  << p2p_partner_rank_ << ", sent parity to Rank " << p2p_partner_rank_ << std::endl;
                    } else {
                        // Odd rank: send data to partner, receive parity from partner
                        if (task.data_addr != 0) {
                            ncclGroupStart();
                            ncclSend(reinterpret_cast<void*>(task.data_addr), task.size,
                                    ncclUint8, p2p_partner_rank_, nccl_comm_p2p_, 0);
                            ncclRecv(reinterpret_cast<void*>(task.p2p_partner_write_addr), task.size,
                                    ncclUint8, p2p_partner_rank_, nccl_comm_p2p_, 0);
                            ncclGroupEnd();
                            // Synchronize to ensure NCCL operation completes
                            cudaDeviceSynchronize();
                            std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P: Sent data to Rank " 
                                      << p2p_partner_rank_ << ", received parity from Rank " << p2p_partner_rank_ << std::endl;
                        } else {
                            std::cerr << "EC-CHECK: [Rank " << rank_ << "] P2P ERROR: Odd rank but data_addr is 0!" << std::endl;
                        }
                    }
                }
#endif
            }
            
            // Step 3: Release buffers after P2P operations complete
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                if (task.parity_addr != 0) {
                    parity_buffers_to_release_.push(task.parity_addr);
                }
                // For odd ranks, release data buffer after P2P completes (it was delayed in encoder worker)
                if (rank_ % 2 == 1 && task.data_addr != 0) {
                    data_buffers_to_release_.push(task.data_addr);
                    // Also remove from data_buffer_states_ if it exists
                    {
                        std::lock_guard<std::mutex> state_lock(data_buffer_state_mutex_);
                        data_buffer_states_.erase(task.data_addr);
                    }
                }
            }
            
            // After processing task, check if sentinel was received and queue is empty
            if (p2p_worker_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(p2p_queue_mutex_);
                if (p2p_queue_.empty()) {
                    p2p_worker_completed_ = true;
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P worker queue is empty after processing, marking completed" << std::endl;
                    // Reset sentinel flag and continue (don't exit)
                    p2p_worker_sentinel_received_ = false;
                    continue;
                }
            }
        }
    }

    void start_pipeline() {
        // Start encoding threads
        encoder_thread_1_ = std::thread(&ECCHECKNative::encoder_worker_1, this);
        encoder_thread_2_ = std::thread(&ECCHECKNative::encoder_worker_2, this);
        
        // Start send/recv workers for each thread
        send_worker_1_ = std::thread(&ECCHECKNative::send_worker_1, this);
        recv_worker_1_ = std::thread(&ECCHECKNative::recv_worker_1, this);
        send_worker_2_ = std::thread(&ECCHECKNative::send_worker_2, this);
        recv_worker_2_ = std::thread(&ECCHECKNative::recv_worker_2, this);
        
        // Start XOR workers
        xor_worker_1_ = std::thread(&ECCHECKNative::xor_worker_1, this);
        xor_worker_2_ = std::thread(&ECCHECKNative::xor_worker_2, this);
        
        // Start P2P worker
        p2p_worker_ = std::thread(&ECCHECKNative::p2p_worker, this);
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Started 9 threads (2 encoding + 4 send/recv + 2 XOR + 1 P2P)" << std::endl;
    }

public:
    ECCHECKNative(int rank, int world_size, int paired_rank,
                  const std::vector<uint8_t>& nccl_id_thread1,
                  const std::vector<uint8_t>& nccl_id_thread2,
                  const std::vector<uint8_t>& nccl_id_p2p) 
        : rank_(rank), world_size_(world_size), paired_rank_(paired_rank), 
          nccl_id_thread1_(nccl_id_thread1),
          nccl_id_thread2_(nccl_id_thread2),
          nccl_id_p2p_(nccl_id_p2p),
          encoding_thread_1_completed_(false), encoding_thread_2_completed_(false),
          send_worker_1_completed_(false), send_worker_2_completed_(false),
          recv_worker_1_completed_(false), recv_worker_2_completed_(false),
          xor_worker_1_completed_(false), xor_worker_2_completed_(false),
          p2p_worker_completed_(false),
          encoding_thread_1_sentinel_received_(false), encoding_thread_2_sentinel_received_(false),
          send_worker_1_sentinel_received_(false), send_worker_2_sentinel_received_(false),
          recv_worker_1_sentinel_received_(false), recv_worker_2_sentinel_received_(false),
          xor_worker_1_sentinel_received_(false), xor_worker_2_sentinel_received_(false),
          p2p_worker_sentinel_received_(false),
          should_stop_threads_(false),
          nccl_thread1_initialized_(false), nccl_thread2_initialized_(false),
          nccl_p2p_initialized_(false),
          nccl_thread1_init_completed_(false), nccl_thread2_init_completed_(false),
          nccl_p2p_init_completed_(false),
          k_(0), rows_(0), data_block_index_(0), a_mat_(nullptr), g_tbls_(nullptr),
          p2p_partner_rank_(-1) {

        std::cout << "EC-CHECK: [Rank " << rank_ << "] Constructor called, initializing EC tables and starting pipeline..." << std::endl;
        
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

        start_pipeline();

        // Wait for all NCCL communicators to be initialized
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Waiting for NCCL initialization (thread1, thread2, and P2P)..." << std::endl;
        std::unique_lock<std::mutex> lock(nccl_init_mutex_);
        nccl_init_cv_.wait(lock, [this] { 
            return nccl_thread1_init_completed_.load() && nccl_thread2_init_completed_.load() && nccl_p2p_init_completed_.load(); 
        });

        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Pipeline and NCCL initialized successfully" << std::endl;
    }
    
    ~ECCHECKNative() {
        stop_pipeline();
        cleanup_nccl();
        if (a_mat_) { free(a_mat_); a_mat_ = nullptr; }
        if (g_tbls_) { free(g_tbls_); g_tbls_ = nullptr; }
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
        send_worker_1_completed_ = false;
        send_worker_2_completed_ = false;
        recv_worker_1_completed_ = false;
        recv_worker_2_completed_ = false;
        xor_worker_1_completed_ = false;
        xor_worker_2_completed_ = false;
        p2p_worker_completed_ = false;
        encoding_thread_1_sentinel_received_ = false;
        encoding_thread_2_sentinel_received_ = false;
        send_worker_1_sentinel_received_ = false;
        send_worker_2_sentinel_received_ = false;
        recv_worker_1_sentinel_received_ = false;
        recv_worker_2_sentinel_received_ = false;
        xor_worker_1_sentinel_received_ = false;
        xor_worker_2_sentinel_received_ = false;
        p2p_worker_sentinel_received_ = false;
    }
    
    void wait_for_encoding_completion() {
        // Wait for all encoding threads to complete
        while (!encoding_thread_1_completed_ || !encoding_thread_2_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Both encoding threads completed" << std::endl;
        
        // Wait for all send workers to complete
        while (!send_worker_1_completed_ || !send_worker_2_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Both send workers completed" << std::endl;
        
        // Wait for all recv workers to complete
        while (!recv_worker_1_completed_ || !recv_worker_2_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Both recv workers completed" << std::endl;
        
        // Wait for all XOR workers to complete
        while (!xor_worker_1_completed_ || !xor_worker_2_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Both XOR workers completed" << std::endl;
        
        // Wait for P2P worker to complete
        while (!p2p_worker_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] P2P worker completed" << std::endl;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] All threads completed (encoding + send + recv + XOR + P2P)" << std::endl;
    }
    
    void stop_pipeline() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Stopping pipeline..." << std::endl;
        
        should_stop_threads_ = true;
        
        // Notify all threads
        encoding_tasks_1_cv_.notify_all();
        encoding_tasks_2_cv_.notify_all();
        send_queue_1_cv_.notify_all();
        send_queue_2_cv_.notify_all();
        recv_queue_1_cv_.notify_all();
        recv_queue_2_cv_.notify_all();
        xor_queue_1_cv_.notify_all();
        xor_queue_2_cv_.notify_all();
        p2p_queue_cv_.notify_all();
        
        // Join all threads
        if (encoder_thread_1_.joinable()) encoder_thread_1_.join();
        if (encoder_thread_2_.joinable()) encoder_thread_2_.join();
        if (send_worker_1_.joinable()) send_worker_1_.join();
        if (recv_worker_1_.joinable()) recv_worker_1_.join();
        if (send_worker_2_.joinable()) send_worker_2_.join();
        if (recv_worker_2_.joinable()) recv_worker_2_.join();
        if (xor_worker_1_.joinable()) xor_worker_1_.join();
        if (xor_worker_2_.joinable()) xor_worker_2_.join();
        if (p2p_worker_.joinable()) p2p_worker_.join();
        
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
};

PYBIND11_MODULE(eccheck_native, m) {
    // Module-level function: Generate NCCL ID (can be called without creating an instance)
#ifdef NCCL_AVAILABLE
    m.def("generate_nccl_id", &generate_nccl_id, 
          "Generate a new NCCL unique ID. Returns a list of uint8_t bytes (128 bytes).");
#endif
    
    // Class definition
    pybind11::class_<ECCHECKNative>(m, "ECCHECKNative")
        .def(pybind11::init<int, int, int, const std::vector<uint8_t>&, const std::vector<uint8_t>&, const std::vector<uint8_t>&>())
        .def("set_buffer_addresses", &ECCHECKNative::set_buffer_addresses)
        .def("reset_encoding_completion_flags", &ECCHECKNative::reset_encoding_completion_flags)
        .def("wait_for_encoding_completion", &ECCHECKNative::wait_for_encoding_completion)
        .def("stop_pipeline", &ECCHECKNative::stop_pipeline)
        .def("submit_data_for_encoding_thread1", &ECCHECKNative::submit_data_for_encoding_thread1)
        .def("submit_data_for_encoding_thread2", &ECCHECKNative::submit_data_for_encoding_thread2)
        .def("get_data_buffers_to_release", &ECCHECKNative::get_data_buffers_to_release)
        .def("get_encoding_buffers_to_release", &ECCHECKNative::get_encoding_buffers_to_release)
        .def("get_parity_buffers_to_release", &ECCHECKNative::get_parity_buffers_to_release);
}
