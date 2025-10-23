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
    
    // Completion flags
    std::atomic<bool> encoding_thread_1_completed_;
    std::atomic<bool> encoding_thread_2_completed_;
    
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
    std::mutex release_queue_mutex_;
    
    // Worker threads - 每个encoding线程配备独立的send/recv worker
    std::thread encoder_thread_1_;
    std::thread encoder_thread_2_;
    std::thread send_worker_1_;     // 专门发送thread1的编码数据
    std::thread recv_worker_1_;     // 专门接收给thread1的数据
    std::thread send_worker_2_;     // 专门发送thread2的编码数据
    std::thread recv_worker_2_;     // 专门接收给thread2的数据
    
    // NCCL communicators - 每个线程有独立的通信域
#ifdef NCCL_AVAILABLE
    ncclComm_t nccl_comm_thread1_;  // Thread1专用通信域
    ncclComm_t nccl_comm_thread2_;  // Thread2专用通信域
    bool nccl_thread1_initialized_;
    bool nccl_thread2_initialized_;
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

    // ========== Worker线程函数 ==========
    
    void encode_with_coefficient(uintptr_t data_addr, size_t size, uintptr_t encoding_addr, int coefficient) {
        // 简单的编码：乘以系数
        uint8_t* data_ptr = reinterpret_cast<uint8_t*>(data_addr);
        uint8_t* encoding_ptr = reinterpret_cast<uint8_t*>(encoding_addr);
        
        for (size_t i = 0; i < size; ++i) {
            encoding_ptr[i] = data_ptr[i] * coefficient;
        }
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
            
            // Check for sentinel (end signal)
            if (task.data_addr == 0 && task.size == 0) {
                encoding_thread_1_completed_ = true;
                
                // Submit sentinel to send_worker_1
                {
                    std::lock_guard<std::mutex> lock(send_queue_1_mutex_);
                    send_queue_1_.push({0, 0});
                }
                send_queue_1_cv_.notify_one();
                
                continue;  // Continue waiting for next round
            }
            
            // Perform encoding
            encode_with_coefficient(task.data_addr, task.size, task.encoding_addr, 1);
            
            // Mark data buffer as copied by thread 1
            {
                std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
                auto& state = data_buffer_states_[task.data_addr];
                state.thread1_copied = true;
                
                // If both threads copied, release data buffer
                if (state.thread2_copied) {
                    std::lock_guard<std::mutex> release_lock(release_queue_mutex_);
                    data_buffers_to_release_.push(task.data_addr);
                    data_buffer_states_.erase(task.data_addr);
                }
            }
            
            // Submit encoding result to send_worker_1
            {
                std::lock_guard<std::mutex> lock(send_queue_1_mutex_);
                send_queue_1_.push({task.encoding_addr, task.size});
            }
            send_queue_1_cv_.notify_one();
            
            // Submit recv task to recv_worker_1
            {
                std::lock_guard<std::mutex> lock(recv_queue_1_mutex_);
                recv_queue_1_.push({task.recv_addr, task.recv_chunk_size});
            }
            recv_queue_1_cv_.notify_one();
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
            
            // Check for sentinel
            if (task.data_addr == 0 && task.size == 0) {
                encoding_thread_2_completed_ = true;
                
                // Submit sentinel to send_worker_2
                {
                    std::lock_guard<std::mutex> lock(send_queue_2_mutex_);
                    send_queue_2_.push({0, 0});
                }
                send_queue_2_cv_.notify_one();
                
                continue;
            }
            
            // Perform encoding
            encode_with_coefficient(task.data_addr, task.size, task.encoding_addr, 2);
            
            // Mark data buffer as copied by thread 2
            {
                std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
                auto& state = data_buffer_states_[task.data_addr];
                state.thread2_copied = true;
                
                // If both threads copied, release data buffer
                if (state.thread1_copied) {
                    std::lock_guard<std::mutex> release_lock(release_queue_mutex_);
                    data_buffers_to_release_.push(task.data_addr);
                    data_buffer_states_.erase(task.data_addr);
                }
            }
            
            // Submit encoding result to send_worker_2
            {
                std::lock_guard<std::mutex> lock(send_queue_2_mutex_);
                send_queue_2_.push({task.encoding_addr, task.size});
            }
            send_queue_2_cv_.notify_one();
            
            // Submit recv task to recv_worker_2
            {
                std::lock_guard<std::mutex> lock(recv_queue_2_mutex_);
                recv_queue_2_.push({task.recv_addr, task.recv_chunk_size});
            }
            recv_queue_2_cv_.notify_one();
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
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread1_initialized_ && world_size_ > 1) {
                ncclSend(reinterpret_cast<void*>(task.encoding_addr), task.size, 
                         ncclUint8, paired_rank_, nccl_comm_thread1_, 0);
            }
#endif

            // Release encoding buffer
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                encoding_buffers_to_release_.push(task.encoding_addr);
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
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread1_initialized_ && world_size_ > 1) {
                ncclRecv(reinterpret_cast<void*>(task.recv_addr), task.size,
                         ncclUint8, paired_rank_, nccl_comm_thread1_, 0);
            }
#endif
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
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread2_initialized_ && world_size_ > 1) {
                ncclSend(reinterpret_cast<void*>(task.encoding_addr), task.size,
                         ncclUint8, paired_rank_, nccl_comm_thread2_, 0);
            }
#endif

            // Release encoding buffer
            {
                std::lock_guard<std::mutex> lock(release_queue_mutex_);
                encoding_buffers_to_release_.push(task.encoding_addr);
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
                continue;
            }
            
#ifdef NCCL_AVAILABLE
            if (nccl_thread2_initialized_ && world_size_ > 1) {
                ncclRecv(reinterpret_cast<void*>(task.recv_addr), task.size,
                         ncclUint8, paired_rank_, nccl_comm_thread2_, 0);
            }
#endif
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker 2 exiting" << std::endl;
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
        
        // Start send/recv workers for each thread
        send_worker_1_ = std::thread(&ECCHECKNative::send_worker_1, this);
        recv_worker_1_ = std::thread(&ECCHECKNative::recv_worker_1, this);
        send_worker_2_ = std::thread(&ECCHECKNative::send_worker_2, this);
        recv_worker_2_ = std::thread(&ECCHECKNative::recv_worker_2, this);
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Started 6 threads (2 encoding + 4 send/recv)" << std::endl;
    }

public:
    ECCHECKNative(int rank, int world_size, int paired_rank) 
        : rank_(rank), world_size_(world_size), paired_rank_(paired_rank), 
          encoding_thread_1_completed_(false), encoding_thread_2_completed_(false),
          should_stop_threads_(false),
          nccl_thread1_initialized_(false), nccl_thread2_initialized_(false),
          nccl_thread1_init_completed_(false), nccl_thread2_init_completed_(false) {
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Constructor called, starting pipeline..." << std::endl;
        
        start_pipeline();
        
        // Wait for both NCCL communicators to be initialized
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Waiting for NCCL initialization (thread1 and thread2)..." << std::endl;
        std::unique_lock<std::mutex> lock(nccl_init_mutex_);
        nccl_init_cv_.wait(lock, [this] { 
            return nccl_thread1_init_completed_.load() && nccl_thread2_init_completed_.load(); 
        });
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Pipeline and NCCL initialized successfully" << std::endl;
    }
    
    ~ECCHECKNative() {
        stop_pipeline();
        cleanup_nccl();
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
    }
    
    void wait_for_encoding_completion() {
        while (!encoding_thread_1_completed_ || !encoding_thread_2_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Both encoding threads completed" << std::endl;
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
        
        // Join all threads
        if (encoder_thread_1_.joinable()) encoder_thread_1_.join();
        if (encoder_thread_2_.joinable()) encoder_thread_2_.join();
        if (send_worker_1_.joinable()) send_worker_1_.join();
        if (recv_worker_1_.joinable()) recv_worker_1_.join();
        if (send_worker_2_.joinable()) send_worker_2_.join();
        if (recv_worker_2_.joinable()) recv_worker_2_.join();
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Pipeline stopped" << std::endl;
    }
    
    void submit_data_for_encoding_thread1(uintptr_t data_addr, size_t size, 
                                          uintptr_t encoding_addr, uintptr_t recv_addr, 
                                          size_t recv_chunk_size) {
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
            encoding_tasks_1_.push({data_addr, size, encoding_addr, recv_addr, recv_chunk_size});
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
                                          size_t recv_chunk_size) {
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
            encoding_tasks_2_.push({data_addr, size, encoding_addr, recv_addr, recv_chunk_size});
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
};

PYBIND11_MODULE(eccheck_native, m) {
    pybind11::class_<ECCHECKNative>(m, "ECCHECKNative")
        .def(pybind11::init<int, int, int>())
        .def("set_buffer_addresses", &ECCHECKNative::set_buffer_addresses)
        .def("reset_encoding_completion_flags", &ECCHECKNative::reset_encoding_completion_flags)
        .def("wait_for_encoding_completion", &ECCHECKNative::wait_for_encoding_completion)
        .def("stop_pipeline", &ECCHECKNative::stop_pipeline)
        .def("submit_data_for_encoding_thread1", &ECCHECKNative::submit_data_for_encoding_thread1)
        .def("submit_data_for_encoding_thread2", &ECCHECKNative::submit_data_for_encoding_thread2)
        .def("get_data_buffers_to_release", &ECCHECKNative::get_data_buffers_to_release)
        .def("get_encoding_buffers_to_release", &ECCHECKNative::get_encoding_buffers_to_release);
}
