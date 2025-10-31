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
#include <fstream>
#include <chrono>
#include <isa-l/erasure_code.h>
#include <isa-l/raid.h>
#include <cstdlib>
#include <cstdint>

// NCCL includes
#ifdef NCCL_AVAILABLE
#include <nccl.h>
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
        uintptr_t parity_addr;      // Parity缓冲区地址（XOR结果）
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
    };
    std::queue<RecvTask> recv_queue_1_;
    std::queue<RecvTask> recv_queue_2_;
    std::mutex recv_queue_1_mutex_;
    std::mutex recv_queue_2_mutex_;
    std::condition_variable recv_queue_1_cv_;
    std::condition_variable recv_queue_2_cv_;
    
    // XOR任务队列（接收完成后提交）
    struct XorTask {
        uintptr_t local_encoding_addr;  // 本地编码数据地址
        uintptr_t recv_encoding_addr;   // 接收到的编码数据地址
        uintptr_t parity_addr;          // Parity缓冲区地址（XOR结果）
        size_t size;                    // 数据大小
    };
    std::queue<XorTask> xor_queue_1_;
    std::queue<XorTask> xor_queue_2_;
    std::mutex xor_queue_1_mutex_;
    std::mutex xor_queue_2_mutex_;
    std::condition_variable xor_queue_1_cv_;
    std::condition_variable xor_queue_2_cv_;
    
    // Completion flags
    std::atomic<bool> encoding_thread_1_completed_;
    std::atomic<bool> encoding_thread_2_completed_;
    std::atomic<bool> send_worker_1_completed_;
    std::atomic<bool> send_worker_2_completed_;
    std::atomic<bool> recv_worker_1_completed_;
    std::atomic<bool> recv_worker_2_completed_;
    std::atomic<bool> xor_worker_1_completed_;
    std::atomic<bool> xor_worker_2_completed_;
    
    // Stop flag for graceful shutdown
    std::atomic<bool> should_stop_threads_;
    
    // Data buffer state tracking
    struct DataBufferState {
        int copies_completed;
        int copies_expected;
    };
    std::unordered_map<uintptr_t, DataBufferState> data_buffer_states_;
    std::mutex data_buffer_state_mutex_;

    // Encoding buffer refcounts: how many downstream steps still need this encoding
    std::unordered_map<uintptr_t, int> encoding_ref_counts_;
    std::mutex encoding_ref_counts_mutex_;
    
    // Recv address to encoding/parity address mapping (for XOR task submission)
    struct RecvMapping {
        uintptr_t encoding_addr;
        uintptr_t parity_addr;
        size_t size;
    };
    std::unordered_map<uintptr_t, RecvMapping> recv_to_xor_mapping_1_;  // For thread 1
    std::unordered_map<uintptr_t, RecvMapping> recv_to_xor_mapping_2_;  // For thread 2
    std::mutex recv_mapping_1_mutex_;
    std::mutex recv_mapping_2_mutex_;
    
    // Buffers ready for release
    std::queue<uintptr_t> data_buffers_to_release_;
    std::queue<uintptr_t> encoding_buffers_to_release_;
    std::queue<uintptr_t> parity_buffers_to_release_;
    std::mutex release_queue_mutex_;
    
    // Persistent stores for final results (in-memory), optional
    uintptr_t persist_recv_base_ = 0;      // Base address for persistent recv store
    uintptr_t persist_parity_base_ = 0;    // Base address for persistent parity store
    size_t persist_recv_capacity_ = 0;     // Total bytes capacity of recv store
    size_t persist_parity_capacity_ = 0;   // Total bytes capacity of parity store
    std::atomic<size_t> persist_recv_offset_{0};
    std::atomic<size_t> persist_parity_offset_{0};
    std::atomic<bool> persist_recv_enabled_{false};
    std::atomic<bool> persist_parity_enabled_{false};
    
    // Worker threads - 每个encoding线程配备独立的send/recv/xor worker
    std::thread encoder_thread_1_;
    std::thread encoder_thread_2_;
    std::thread send_worker_1_;     // 专门发送thread1的编码数据
    std::thread recv_worker_1_;     // 专门接收给thread1的数据
    std::thread send_worker_2_;     // 专门发送thread2的编码数据
    std::thread recv_worker_2_;     // 专门接收给thread2的数据
    std::thread xor_worker_1_;      // 专门执行thread1的XOR操作
    std::thread xor_worker_2_;      // 专门执行thread2的XOR操作
    
    // NCCL communicators - 每个线程有独立的通信域
#ifdef NCCL_AVAILABLE
    ncclComm_t nccl_comm_thread1_;  // Thread1专用通信域
    ncclComm_t nccl_comm_thread2_;  // Thread2专用通信域
    bool nccl_thread1_initialized_;
    bool nccl_thread2_initialized_;
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

    // Synchronization for NCCL initialization
    std::atomic<bool> nccl_thread1_init_completed_;
    std::atomic<bool> nccl_thread2_init_completed_;
    std::mutex nccl_init_mutex_;
    std::condition_variable nccl_init_cv_;

    // ========== NCCL初始化函数 ==========
    
    void init_nccl_thread1() {
#ifdef NCCL_AVAILABLE
        ncclUniqueId nccl_id;
        std::string id_file = "/tmp/eccheck_nccl_thread1_id.txt";
        
        if (rank_ == 0) {
            // Rank 0 creates NCCL ID and writes to file
            ncclGetUniqueId(&nccl_id);
            std::ofstream outfile(id_file, std::ios::binary);
            outfile.write(reinterpret_cast<char*>(&nccl_id), sizeof(ncclUniqueId));
            outfile.close();
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread1 NCCL ID written to " << id_file << std::endl;
        } else {
            // Other ranks read NCCL ID from file (with retry)
            for (int retry = 0; retry < 100; ++retry) {
                std::ifstream infile(id_file, std::ios::binary);
                if (infile.good()) {
                    infile.read(reinterpret_cast<char*>(&nccl_id), sizeof(ncclUniqueId));
                    infile.close();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread1 NCCL ID read from " << id_file << std::endl;
        }
        
        // Initialize NCCL communicator
        ncclCommInitRank(&nccl_comm_thread1_, world_size_, nccl_id, rank_);
        nccl_thread1_initialized_ = true;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread1 NCCL communicator initialized" << std::endl;
        
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
        ncclUniqueId nccl_id;
        std::string id_file = "/tmp/eccheck_nccl_thread2_id.txt";
        
        if (rank_ == 0) {
            // Rank 0 creates NCCL ID and writes to file
            ncclGetUniqueId(&nccl_id);
            std::ofstream outfile(id_file, std::ios::binary);
            outfile.write(reinterpret_cast<char*>(&nccl_id), sizeof(ncclUniqueId));
            outfile.close();
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread2 NCCL ID written to " << id_file << std::endl;
        } else {
            // Other ranks read NCCL ID from file (with retry)
            for (int retry = 0; retry < 100; ++retry) {
                std::ifstream infile(id_file, std::ios::binary);
                if (infile.good()) {
                    infile.read(reinterpret_cast<char*>(&nccl_id), sizeof(ncclUniqueId));
                    infile.close();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread2 NCCL ID read from " << id_file << std::endl;
        }
        
        // Initialize NCCL communicator
        ncclCommInitRank(&nccl_comm_thread2_, world_size_, nccl_id, rank_);
        nccl_thread2_initialized_ = true;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Thread2 NCCL communicator initialized" << std::endl;
        
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
    
    void cleanup_nccl() {
#ifdef NCCL_AVAILABLE
        if (nccl_thread1_initialized_) {
            ncclCommDestroy(nccl_comm_thread1_);
        }
        if (nccl_thread2_initialized_) {
            ncclCommDestroy(nccl_comm_thread2_);
        }
#endif
    }

    // Columns runtime config (for current 2 columns)
    struct ColumnCfg { int coefficient; int send_peer; int recv_peer; };
    ColumnCfg col1_cfg_{0, -1, -1};
    ColumnCfg col2_cfg_{1, -1, -1};
    int num_columns_ = 2;

    // ========== Worker线程函数 ==========
    
    void encode_with_parity_index(uintptr_t data_addr, size_t size, uintptr_t encoding_addr, int parity_idx) {
        // 使用 isa-l 的 EC 编码对整块 buffer 进行编码。
        // 我们在初始化时已经生成了 RS 矩阵并通过 ec_init_tables 产生了 g_tbls_。
        // 每个 encoder 线程只保留自己负责的 parity（encoding_addr 指向本地 parity buffer）。

        // 如果没有正确初始化 EC 表，回退到简单乘法
        if (k_ <= 0 || rows_ != 2 || g_tbls_ == nullptr) {
            uint8_t* data_ptr = reinterpret_cast<uint8_t*>(data_addr);
            uint8_t* encoding_ptr = reinterpret_cast<uint8_t*>(encoding_addr);
            for (size_t i = 0; i < size; ++i) {
                encoding_ptr[i] = data_ptr[i];
            }
            return;
        }

        unsigned char *data_ptr = reinterpret_cast<unsigned char*>(data_addr);
        unsigned char *enc_ptr = reinterpret_cast<unsigned char*>(encoding_addr);

        // 从 g_tbls_ 中取出对应 (parity_index, data_block_index_) 的 32 字节表，
        // 并构造 k=1, rows=1 的调用参数。
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
                task.encoding_addr == 0 && task.recv_addr == 0 && task.recv_chunk_size == 0 && task.parity_addr == 0) {
                encoding_thread_1_completed_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 1 received sentinel, marking completed" << std::endl;
                
                // Submit sentinel to send_worker_1
                {
                    std::lock_guard<std::mutex> lock(send_queue_1_mutex_);
                    send_queue_1_.push({0, 0});
                }
                send_queue_1_cv_.notify_one();
                
                // Submit sentinel to recv_worker_1
                {
                    std::lock_guard<std::mutex> lock(recv_queue_1_mutex_);
                    recv_queue_1_.push({0, 0});
                }
                recv_queue_1_cv_.notify_one();
                
                // Submit sentinel to xor_worker_1
                {
                    std::lock_guard<std::mutex> lock(xor_queue_1_mutex_);
                    xor_queue_1_.push({0, 0, 0, 0});
                }
                xor_queue_1_cv_.notify_one();
                
                continue;  // Continue waiting for next round
            }
            
            // Check if we need to encode/send data
            bool need_encode = (task.data_addr != 0 && task.encoding_addr != 0);
            
            if (need_encode) {
                // Perform encoding selecting GF table by (parity_idx=0, data_block_index_=rank%k)
                encode_with_parity_index(task.data_addr, task.size, task.encoding_addr, 0);
                
                // Mark data buffer copy completion (count-based)
                {
                    std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
                    auto it = data_buffer_states_.find(task.data_addr);
                    if (it != data_buffer_states_.end()) {
                        it->second.copies_completed += 1;
                        if (it->second.copies_completed >= it->second.copies_expected) {
                            std::lock_guard<std::mutex> release_lock(release_queue_mutex_);
                            data_buffers_to_release_.push(task.data_addr);
                            data_buffer_states_.erase(it);
                        }
                    }
                }
                
                // Submit encoding result to send_worker_1
                {
                    std::lock_guard<std::mutex> lock(send_queue_1_mutex_);
                    send_queue_1_.push({task.encoding_addr, task.size});
                }
                send_queue_1_cv_.notify_one();
                // Increment encoding refcount for pending send
                if (task.encoding_addr != 0) {
                    std::lock_guard<std::mutex> rlock(encoding_ref_counts_mutex_);
                    encoding_ref_counts_[task.encoding_addr] += 1;
                }
            }
            
            // Check if we need to receive data
            bool need_recv = (task.recv_addr != 0 && task.recv_chunk_size != 0);
            
            if (need_recv) {
                // Submit recv task to recv_worker_1
                {
                    std::lock_guard<std::mutex> lock(recv_queue_1_mutex_);
                    recv_queue_1_.push({task.recv_addr, task.recv_chunk_size});
                }
                recv_queue_1_cv_.notify_one();
            }
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 1 exiting" << std::endl;
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
                task.encoding_addr == 0 && task.recv_addr == 0 && task.recv_chunk_size == 0 && task.parity_addr == 0) {
                encoding_thread_2_completed_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 received sentinel, marking completed" << std::endl;
                
                // Submit sentinel to send_worker_2
                {
                    std::lock_guard<std::mutex> lock(send_queue_2_mutex_);
                    send_queue_2_.push({0, 0});
                }
                send_queue_2_cv_.notify_one();
                
                // Submit sentinel to recv_worker_2
                {
                    std::lock_guard<std::mutex> lock(recv_queue_2_mutex_);
                    recv_queue_2_.push({0, 0});
                }
                recv_queue_2_cv_.notify_one();
                
                // Submit sentinel to xor_worker_2
                {
                    std::lock_guard<std::mutex> lock(xor_queue_2_mutex_);
                    xor_queue_2_.push({0, 0, 0, 0});
                }
                xor_queue_2_cv_.notify_one();
                
                continue;
            }
            
            // Check if we need to encode/send data
            bool need_encode = (task.data_addr != 0 && task.encoding_addr != 0);
            
            if (need_encode) {
                // Perform encoding selecting GF table by (parity_idx=1, data_block_index_=rank%k)
                encode_with_parity_index(task.data_addr, task.size, task.encoding_addr, 1);
                
                // Mark data buffer copy completion (count-based)
                {
                    std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
                    auto it = data_buffer_states_.find(task.data_addr);
                    if (it != data_buffer_states_.end()) {
                        it->second.copies_completed += 1;
                        if (it->second.copies_completed >= it->second.copies_expected) {
                            std::lock_guard<std::mutex> release_lock(release_queue_mutex_);
                            data_buffers_to_release_.push(task.data_addr);
                            data_buffer_states_.erase(it);
                        }
                    }
                }
                
                // Submit encoding result to send_worker_2
                {
                    std::lock_guard<std::mutex> lock(send_queue_2_mutex_);
                    send_queue_2_.push({task.encoding_addr, task.size});
                }
                send_queue_2_cv_.notify_one();
                // Increment encoding refcount for pending send
                if (task.encoding_addr != 0) {
                    std::lock_guard<std::mutex> rlock(encoding_ref_counts_mutex_);
                    encoding_ref_counts_[task.encoding_addr] += 1;
                }
            }
            
            // Check if we need to receive data
            bool need_recv = (task.recv_addr != 0 && task.recv_chunk_size != 0);
            
            if (need_recv) {
                // Submit recv task to recv_worker_2
                {
                    std::lock_guard<std::mutex> lock(recv_queue_2_mutex_);
                    recv_queue_2_.push({task.recv_addr, task.recv_chunk_size});
                }
                recv_queue_2_cv_.notify_one();
            }
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread 2 exiting" << std::endl;
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
                send_worker_1_completed_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 1 received sentinel, marking completed" << std::endl;
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread1_initialized_ && world_size_ > 1) {
                ncclGroupStart(); 
                int peer = (col1_cfg_.send_peer >= 0 ? col1_cfg_.send_peer : paired_rank_);
                ncclSend(reinterpret_cast<void*>(task.encoding_addr), task.size, 
                         ncclUint8, peer, nccl_comm_thread1_, 0);
                ncclGroupEnd();
            }
#endif

            // Decrement encoding refcount and release if reaches zero
            if (task.encoding_addr != 0) {
                bool should_release = false;
                {
                    std::lock_guard<std::mutex> rlock(encoding_ref_counts_mutex_);
                    auto it = encoding_ref_counts_.find(task.encoding_addr);
                    if (it != encoding_ref_counts_.end()) {
                        it->second -= 1;
                        if (it->second <= 0) {
                            should_release = true;
                            encoding_ref_counts_.erase(it);
                        }
                    } else {
                        should_release = true; // safety fallback
                    }
                }
                if (should_release) {
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.encoding_addr);
                }
            }
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 1 exiting" << std::endl;
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
                recv_worker_1_completed_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 1 received sentinel, marking completed" << std::endl;
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread1_initialized_ && world_size_ > 1) {
                ncclGroupStart();
                int peer = (col1_cfg_.recv_peer >= 0 ? col1_cfg_.recv_peer : paired_rank_);
                ncclRecv(reinterpret_cast<void*>(task.recv_addr), task.size,
                         ncclUint8, peer, nccl_comm_thread1_, 0);
                ncclGroupEnd();
            }
#endif
            
            // Persist recv data into continuous store if enabled (thread1 only)
            if (persist_recv_enabled_.load() && persist_recv_base_ != 0 && task.size > 0) {
                size_t offset = persist_recv_offset_.fetch_add(task.size);
                if (offset + task.size <= persist_recv_capacity_) {
                    void* dst = reinterpret_cast<void*>(persist_recv_base_ + offset);
                    void* src = reinterpret_cast<void*>(task.recv_addr);
                    std::memcpy(dst, src, task.size);
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] recv persist overflow: offset="
                              << offset << ", size=" << task.size << ", cap=" << persist_recv_capacity_ << std::endl;
                }
            }
            
            // After receiving, submit XOR task: local_encoding XOR recv_encoding -> parity
            {
                std::lock_guard<std::mutex> lock(recv_mapping_1_mutex_);
                auto it = recv_to_xor_mapping_1_.find(task.recv_addr);
                if (it != recv_to_xor_mapping_1_.end()) {
                    const RecvMapping& mapping = it->second;
                    // Increment encoding refcount for XOR (XOR will use this encoding)
                    {
                        std::lock_guard<std::mutex> rlock(encoding_ref_counts_mutex_);
                        encoding_ref_counts_[mapping.encoding_addr] += 1;
                    }
                    // Submit XOR task: local encoding XOR received encoding -> parity
                    {
                        std::lock_guard<std::mutex> xor_lock(xor_queue_1_mutex_);
                        xor_queue_1_.push({mapping.encoding_addr, task.recv_addr, mapping.parity_addr, mapping.size});
                    }
                    xor_queue_1_cv_.notify_one();
                    // Remove from mapping after submitting
                    recv_to_xor_mapping_1_.erase(it);
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] Warning: No mapping found for recv_addr=" << task.recv_addr << std::endl;
                }
            }
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 1 exiting" << std::endl;
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
                send_worker_2_completed_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 2 received sentinel, marking completed" << std::endl;
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread2_initialized_ && world_size_ > 1) {
                ncclGroupStart();
                int peer = (col2_cfg_.send_peer >= 0 ? col2_cfg_.send_peer : paired_rank_);
                ncclSend(reinterpret_cast<void*>(task.encoding_addr), task.size,
                         ncclUint8, peer, nccl_comm_thread2_, 0);
                ncclGroupEnd();
            }
#endif

            // Decrement encoding refcount and release if reaches zero
            if (task.encoding_addr != 0) {
                bool should_release = false;
                {
                    std::lock_guard<std::mutex> rlock(encoding_ref_counts_mutex_);
                    auto it = encoding_ref_counts_.find(task.encoding_addr);
                    if (it != encoding_ref_counts_.end()) {
                        it->second -= 1;
                        if (it->second <= 0) {
                            should_release = true;
                            encoding_ref_counts_.erase(it);
                        }
                    } else {
                        should_release = true; // safety fallback
                    }
                }
                if (should_release) {
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.encoding_addr);
                }
            }
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker 2 exiting" << std::endl;
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
                recv_worker_2_completed_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 2 received sentinel, marking completed" << std::endl;
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread2_initialized_ && world_size_ > 1) {
                ncclGroupStart();
                int peer = (col2_cfg_.recv_peer >= 0 ? col2_cfg_.recv_peer : paired_rank_);
                ncclRecv(reinterpret_cast<void*>(task.recv_addr), task.size,
                         ncclUint8, peer, nccl_comm_thread2_, 0);
                ncclGroupEnd();
            }
#endif
            
            // Thread2: skip persisting recv to avoid double-writing into single store
            
            // After receiving, submit XOR task: local_encoding XOR recv_encoding -> parity
            {
                std::lock_guard<std::mutex> lock(recv_mapping_2_mutex_);
                auto it = recv_to_xor_mapping_2_.find(task.recv_addr);
                if (it != recv_to_xor_mapping_2_.end()) {
                    const RecvMapping& mapping = it->second;
                    // Increment encoding refcount for XOR (XOR will use this encoding)
                    {
                        std::lock_guard<std::mutex> rlock(encoding_ref_counts_mutex_);
                        encoding_ref_counts_[mapping.encoding_addr] += 1;
                    }
                    // Submit XOR task: local encoding XOR received encoding -> parity
                    {
                        std::lock_guard<std::mutex> xor_lock(xor_queue_2_mutex_);
                        xor_queue_2_.push({mapping.encoding_addr, task.recv_addr, mapping.parity_addr, mapping.size});
                    }
                    xor_queue_2_cv_.notify_one();
                    // Remove from mapping after submitting
                    recv_to_xor_mapping_2_.erase(it);
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] Warning: No mapping found for recv_addr=" << task.recv_addr << std::endl;
                }
            }
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 2 exiting" << std::endl;
    }
    
    // XOR Worker 1 - 执行本地编码数据与接收编码数据的XOR操作
    void xor_worker_1() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 1 started" << std::endl;
        
        while (!should_stop_threads_) {
            XorTask task;
            
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
            if (task.local_encoding_addr == 0 && task.recv_encoding_addr == 0 && 
                task.parity_addr == 0 && task.size == 0) {
                xor_worker_1_completed_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 1 received sentinel, marking completed" << std::endl;
                continue;
            }
            
            // Perform XOR: local_encoding XOR recv_encoding -> parity
            // ISA-L xor_gen requires: array[0..N-1] = sources, array[N] = dest
            void* xor_array[3];
            xor_array[0] = reinterpret_cast<void*>(task.local_encoding_addr);
            xor_array[1] = reinterpret_cast<void*>(task.recv_encoding_addr);
            xor_array[2] = reinterpret_cast<void*>(task.parity_addr);
            
            // xor_gen(vects=3, len=size, array): XOR of sources into dest
            // For 2 sources, vects=3 (array[0], array[1], array[2])
            int result = xor_gen(3, (int)task.size, xor_array);
            if (result != 0) {
                std::cerr << "EC-CHECK: [Rank " << rank_ << "] XOR worker 1 failed: xor_gen returned " << result << std::endl;
            }
            
            // Persist parity result into continuous store if enabled (thread1 only)
            if (persist_parity_enabled_.load() && persist_parity_base_ != 0 && task.size > 0) {
                size_t offset = persist_parity_offset_.fetch_add(task.size);
                if (offset + task.size <= persist_parity_capacity_) {
                    void* dst = reinterpret_cast<void*>(persist_parity_base_ + offset);
                    void* src = reinterpret_cast<void*>(task.parity_addr);
                    std::memcpy(dst, src, task.size);
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] parity persist overflow: offset="
                              << offset << ", size=" << task.size << ", cap=" << persist_parity_capacity_ << std::endl;
                }
            }
            
            // After XOR completion, decrement encoding refcount for the local encoding
            if (task.local_encoding_addr != 0) {
                bool should_release = false;
                {
                    std::lock_guard<std::mutex> rlock(encoding_ref_counts_mutex_);
                    auto it = encoding_ref_counts_.find(task.local_encoding_addr);
                    if (it != encoding_ref_counts_.end()) {
                        it->second -= 1;
                        if (it->second <= 0) {
                            should_release = true;
                            encoding_ref_counts_.erase(it);
                        }
                    }
                }
                if (should_release) {
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.local_encoding_addr);
                }
            }

            // Release parity buffer after XOR completion
            // Note: Parity result is stored, but buffer can be reused for next chunk if needed
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                parity_buffers_to_release_.push(task.parity_addr);
            }
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 1 exiting" << std::endl;
    }
    
    // XOR Worker 2 - 执行本地编码数据与接收编码数据的XOR操作
    void xor_worker_2() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 2 started" << std::endl;
        
        while (!should_stop_threads_) {
            XorTask task;
            
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
            if (task.local_encoding_addr == 0 && task.recv_encoding_addr == 0 && 
                task.parity_addr == 0 && task.size == 0) {
                xor_worker_2_completed_ = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 2 received sentinel, marking completed" << std::endl;
                continue;
            }
            
            // Perform XOR: local_encoding XOR recv_encoding -> parity
            // ISA-L xor_gen requires: array[0..N-1] = sources, array[N] = dest
            void* xor_array[3];
            xor_array[0] = reinterpret_cast<void*>(task.local_encoding_addr);
            xor_array[1] = reinterpret_cast<void*>(task.recv_encoding_addr);
            xor_array[2] = reinterpret_cast<void*>(task.parity_addr);
            
            // xor_gen(vects=3, len=size, array): XOR of sources into dest
            // For 2 sources, vects=3 (array[0], array[1], array[2])
            int result = xor_gen(3, (int)task.size, xor_array);
            if (result != 0) {
                std::cerr << "EC-CHECK: [Rank " << rank_ << "] XOR worker 2 failed: xor_gen returned " << result << std::endl;
            }
            
            // Thread2: skip persisting parity to avoid double-writing into single store
            
            // After XOR completion, decrement encoding refcount for the local encoding
            if (task.local_encoding_addr != 0) {
                bool should_release = false;
                {
                    std::lock_guard<std::mutex> rlock(encoding_ref_counts_mutex_);
                    auto it = encoding_ref_counts_.find(task.local_encoding_addr);
                    if (it != encoding_ref_counts_.end()) {
                        it->second -= 1;
                        if (it->second <= 0) {
                            should_release = true;
                            encoding_ref_counts_.erase(it);
                        }
                    }
                }
                if (should_release) {
                    std::lock_guard<std::mutex> lock(release_queue_mutex_);
                    encoding_buffers_to_release_.push(task.local_encoding_addr);
                }
            }

            // Release parity buffer after XOR completion
            // Note: Parity result is stored, but buffer can be reused for next chunk if needed
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                parity_buffers_to_release_.push(task.parity_addr);
            }
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker 2 exiting" << std::endl;
    }

    void start_pipeline() {
        // Clean up any existing NCCL ID files before starting
        std::string thread1_id_file = "/tmp/eccheck_nccl_thread1_id.txt";
        std::string thread2_id_file = "/tmp/eccheck_nccl_thread2_id.txt";
        
        // Check and remove thread1 ID file if exists
        std::ifstream file1(thread1_id_file);
        if (file1.good()) {
            file1.close();
            std::remove(thread1_id_file.c_str());
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Removed existing thread1 ID file" << std::endl;
        }
        
        // Check and remove thread2 ID file if exists
        std::ifstream file2(thread2_id_file);
        if (file2.good()) {
            file2.close();
            std::remove(thread2_id_file.c_str());
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Removed existing thread2 ID file" << std::endl;
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] NCCL ID file cleanup completed" << std::endl;
        
        // Start encoding threads
        encoder_thread_1_ = std::thread(&ECCHECKNative::encoder_worker_1, this);
        encoder_thread_2_ = std::thread(&ECCHECKNative::encoder_worker_2, this);
        
        // Start send/recv/xor workers for each thread
        send_worker_1_ = std::thread(&ECCHECKNative::send_worker_1, this);
        recv_worker_1_ = std::thread(&ECCHECKNative::recv_worker_1, this);
        xor_worker_1_ = std::thread(&ECCHECKNative::xor_worker_1, this);
        send_worker_2_ = std::thread(&ECCHECKNative::send_worker_2, this);
        recv_worker_2_ = std::thread(&ECCHECKNative::recv_worker_2, this);
        xor_worker_2_ = std::thread(&ECCHECKNative::xor_worker_2, this);
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Started 8 threads (2 encoding + 2 send + 2 recv + 2 XOR)" << std::endl;
    }

public:
    ECCHECKNative(int rank, int world_size, int paired_rank) 
        : rank_(rank), world_size_(world_size), paired_rank_(paired_rank), 
          encoding_thread_1_completed_(false), encoding_thread_2_completed_(false),
          send_worker_1_completed_(false), send_worker_2_completed_(false),
          recv_worker_1_completed_(false), recv_worker_2_completed_(false),
          xor_worker_1_completed_(false), xor_worker_2_completed_(false),
          should_stop_threads_(false),
          nccl_thread1_initialized_(false), nccl_thread2_initialized_(false),
          nccl_thread1_init_completed_(false), nccl_thread2_init_completed_(false),
          k_(0), rows_(0), data_block_index_(0), a_mat_(nullptr), g_tbls_(nullptr) {

        std::cout << "EC-CHECK: [Rank " << rank_ << "] Constructor called, initializing EC tables and starting pipeline..." << std::endl;

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

        // Wait for both NCCL communicators to be initialized
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Waiting for NCCL initialization (thread1 and thread2)..." << std::endl;
        std::unique_lock<std::mutex> lock(nccl_init_mutex_);
        nccl_init_cv_.wait(lock, [this] { 
            return nccl_thread1_init_completed_.load() && nccl_thread2_init_completed_.load(); 
        });

        std::cout << "EC-CHECK: [Rank " << rank_ << "] Pipeline and NCCL initialized successfully" << std::endl;
    }
    
    void set_columns_config(const std::vector<std::map<std::string, int>>& cols) {
        // Only first two columns used for current 2+2; ignore extras
        if (!cols.empty()) {
            const auto &c0 = cols[0];
            col1_cfg_.coefficient = c0.count("coefficient") ? c0.at("coefficient") : 0;
            col1_cfg_.send_peer   = c0.count("send_peer")   ? c0.at("send_peer")   : -1;
            col1_cfg_.recv_peer   = c0.count("recv_peer")   ? c0.at("recv_peer")   : -1;
        }
        if (cols.size() > 1) {
            const auto &c1 = cols[1];
            col2_cfg_.coefficient = c1.count("coefficient") ? c1.at("coefficient") : 1;
            col2_cfg_.send_peer   = c1.count("send_peer")   ? c1.at("send_peer")   : -1;
            col2_cfg_.recv_peer   = c1.count("recv_peer")   ? c1.at("recv_peer")   : -1;
        }
        num_columns_ = std::max(1, std::min((int)cols.size(), 2));
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Columns config applied: col0(coef="
                  << col1_cfg_.coefficient << ", send_peer=" << (col1_cfg_.send_peer>=0?col1_cfg_.send_peer:paired_rank_)
                  << ", recv_peer=" << (col1_cfg_.recv_peer>=0?col1_cfg_.recv_peer:paired_rank_)
                  << ") col1(coef=" << col2_cfg_.coefficient << ", send_peer="
                  << (col2_cfg_.send_peer>=0?col2_cfg_.send_peer:paired_rank_) << ", recv_peer="
                  << (col2_cfg_.recv_peer>=0?col2_cfg_.recv_peer:paired_rank_) << ")" << std::endl;
    }
    
    void set_persist_stores(uintptr_t recv_base, uintptr_t parity_base,
                            size_t recv_capacity, size_t parity_capacity,
                            int persist_recv, int persist_parity) {
        persist_recv_base_ = recv_base;
        persist_parity_base_ = parity_base;
        persist_recv_capacity_ = recv_capacity;
        persist_parity_capacity_ = parity_capacity;
        persist_recv_enabled_ = (persist_recv != 0);
        persist_parity_enabled_ = (persist_parity != 0);
        persist_recv_offset_ = 0;
        persist_parity_offset_ = 0;
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Registered persistent stores (recv="
                  << (persist_recv_enabled_ ? "on" : "off") << ", parity="
                  << (persist_parity_enabled_ ? "on" : "off") << ")" << std::endl;
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
    }
    
    void wait_for_encoding_completion() {
        // Wait for all encoding threads to complete
        while (!encoding_thread_1_completed_ || !encoding_thread_2_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Both encoding threads completed" << std::endl;
        
        // Wait for all send workers to complete
        while (!send_worker_1_completed_ || !send_worker_2_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Both send workers completed" << std::endl;
        
        // Wait for all recv workers to complete
        while (!recv_worker_1_completed_ || !recv_worker_2_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Both recv workers completed" << std::endl;
        
        // Wait for all XOR workers to complete
        while (!xor_worker_1_completed_ || !xor_worker_2_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Both XOR workers completed" << std::endl;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] All threads completed (encoding + send + recv + XOR)" << std::endl;
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
        
        // Join all threads
        if (encoder_thread_1_.joinable()) encoder_thread_1_.join();
        if (encoder_thread_2_.joinable()) encoder_thread_2_.join();
        if (send_worker_1_.joinable()) send_worker_1_.join();
        if (recv_worker_1_.joinable()) recv_worker_1_.join();
        if (xor_worker_1_.joinable()) xor_worker_1_.join();
        if (send_worker_2_.joinable()) send_worker_2_.join();
        if (recv_worker_2_.joinable()) recv_worker_2_.join();
        if (xor_worker_2_.joinable()) xor_worker_2_.join();
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Pipeline stopped" << std::endl;
    }
    
    void submit_data_for_encoding_thread1(uintptr_t data_addr, size_t size, 
                                          uintptr_t encoding_addr, uintptr_t recv_addr, 
                                          size_t recv_chunk_size, uintptr_t parity_addr) {
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
            encoding_tasks_1_.push({data_addr, size, encoding_addr, recv_addr, recv_chunk_size, parity_addr});
        }
        encoding_tasks_1_cv_.notify_one();
        
        if (data_addr != 0 && size != 0) {
            std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
            auto &state = data_buffer_states_[data_addr];
            if (state.copies_expected == 0) {
                state.copies_expected = std::max(1, num_columns_);
                state.copies_completed = 0;
            }
        }
        
        // Register recv_addr to encoding/parity mapping for XOR task submission
        // Note: encoding_refcount for XOR will be incremented when recv completes and XOR task is created
        if (recv_addr != 0 && recv_chunk_size != 0 && encoding_addr != 0 && parity_addr != 0) {
            std::lock_guard<std::mutex> lock(recv_mapping_1_mutex_);
            recv_to_xor_mapping_1_[recv_addr] = {encoding_addr, parity_addr, recv_chunk_size};
        }
    }
    
    void submit_data_for_encoding_thread2(uintptr_t data_addr, size_t size,
                                          uintptr_t encoding_addr, uintptr_t recv_addr,
                                          size_t recv_chunk_size, uintptr_t parity_addr) {
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
            encoding_tasks_2_.push({data_addr, size, encoding_addr, recv_addr, recv_chunk_size, parity_addr});
        }
        encoding_tasks_2_cv_.notify_one();
        
        // Ensure data buffer state exists and has expected count (mirrors thread1 submission)
        if (data_addr != 0 && size != 0) {
            std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
            auto &state = data_buffer_states_[data_addr];
            if (state.copies_expected == 0) {
                state.copies_expected = std::max(1, num_columns_);
                state.copies_completed = 0;
            }
        }
        
        // Register recv_addr to encoding/parity mapping for XOR task submission
        // Note: encoding_refcount for XOR will be incremented when recv completes and XOR task is created
        if (recv_addr != 0 && recv_chunk_size != 0 && encoding_addr != 0 && parity_addr != 0) {
            std::lock_guard<std::mutex> lock(recv_mapping_2_mutex_);
            recv_to_xor_mapping_2_[recv_addr] = {encoding_addr, parity_addr, recv_chunk_size};
        }
    }
    
    // Generic submit API for future N-column support: currently dispatches to thread1/2
    void submit_data_for_encoding(int column_idx,
                                  uintptr_t data_addr, size_t size,
                                  uintptr_t encoding_addr, uintptr_t recv_addr,
                                  size_t recv_chunk_size, uintptr_t parity_addr) {
        if (column_idx == 0) {
            submit_data_for_encoding_thread1(data_addr, size, encoding_addr, recv_addr, recv_chunk_size, parity_addr);
        } else if (column_idx == 1) {
            submit_data_for_encoding_thread2(data_addr, size, encoding_addr, recv_addr, recv_chunk_size, parity_addr);
        } else {
            // For now, only 2 columns are implemented in the engine
            std::cerr << "EC-CHECK: submit_data_for_encoding: column_idx=" << column_idx
                      << " not implemented (only 0 and 1 supported)" << std::endl;
#ifdef PYBIND11_VERSION
            throw pybind11::value_error("Only 2 columns (indices 0 and 1) are supported currently");
#else
            return;
#endif
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
};

PYBIND11_MODULE(eccheck_native, m) {
    pybind11::class_<ECCHECKNative>(m, "ECCHECKNative")
        .def(pybind11::init<int, int, int>())
        .def("set_columns_config", &ECCHECKNative::set_columns_config)
        .def("set_persist_stores", &ECCHECKNative::set_persist_stores,
             pybind11::arg("recv_base"),
             pybind11::arg("parity_base"),
             pybind11::arg("recv_capacity"),
             pybind11::arg("parity_capacity"),
             pybind11::arg("persist_recv"),
             pybind11::arg("persist_parity"))
        .def("set_buffer_addresses", &ECCHECKNative::set_buffer_addresses)
        .def("reset_encoding_completion_flags", &ECCHECKNative::reset_encoding_completion_flags)
        .def("wait_for_encoding_completion", &ECCHECKNative::wait_for_encoding_completion)
        .def("stop_pipeline", &ECCHECKNative::stop_pipeline)
        .def("submit_data_for_encoding_thread1", &ECCHECKNative::submit_data_for_encoding_thread1,
             pybind11::arg("data_addr"), pybind11::arg("size"), 
             pybind11::arg("encoding_addr"), pybind11::arg("recv_addr"), 
             pybind11::arg("recv_chunk_size"), pybind11::arg("parity_addr"))
        .def("submit_data_for_encoding_thread2", &ECCHECKNative::submit_data_for_encoding_thread2,
             pybind11::arg("data_addr"), pybind11::arg("size"),
             pybind11::arg("encoding_addr"), pybind11::arg("recv_addr"),
             pybind11::arg("recv_chunk_size"), pybind11::arg("parity_addr"))
        .def("submit_data_for_encoding", &ECCHECKNative::submit_data_for_encoding,
             pybind11::arg("column_idx"),
             pybind11::arg("data_addr"), pybind11::arg("size"),
             pybind11::arg("encoding_addr"), pybind11::arg("recv_addr"),
             pybind11::arg("recv_chunk_size"), pybind11::arg("parity_addr"))
        .def("get_data_buffers_to_release", &ECCHECKNative::get_data_buffers_to_release)
        .def("get_encoding_buffers_to_release", &ECCHECKNative::get_encoding_buffers_to_release)
        .def("get_parity_buffers_to_release", &ECCHECKNative::get_parity_buffers_to_release);
}
