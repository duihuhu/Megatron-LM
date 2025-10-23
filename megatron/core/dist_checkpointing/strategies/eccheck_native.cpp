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
    
    // Buffer management is now handled by Python directly
    // No need for free buffer pools
    
    // Pipeline queues
    std::queue<std::pair<int, size_t> > encode_queue_1_;  // First encoding queue
    std::queue<std::pair<int, size_t> > encode_queue_2_;  // Second encoding queue
    std::queue<std::pair<int, size_t> > exchange_queue_;
    std::mutex encode_queue_1_mutex_;
    std::mutex encode_queue_2_mutex_;
    std::mutex exchange_queue_mutex_;
    std::condition_variable encode_queue_1_cv_;
    std::condition_variable encode_queue_2_cv_;
    std::condition_variable exchange_queue_cv_;
    
    // Encoding tasks with addresses - separate for each thread
    struct EncodingTask {
        uintptr_t data_addr;
        size_t size;
        uintptr_t encoding_addr;
    };
    std::queue<EncodingTask> encoding_tasks_1_;  // For thread 1
    std::queue<EncodingTask> encoding_tasks_2_;  // For thread 2
    std::mutex encoding_tasks_1_mutex_;  // Mutex for thread 1 queue
    std::mutex encoding_tasks_2_mutex_;  // Mutex for thread 2 queue
    std::condition_variable encoding_tasks_1_cv_;  // Condition variable for thread 1
    std::condition_variable encoding_tasks_2_cv_;  // Condition variable for thread 2
    
    // Completion flags
    std::atomic<bool> encoding_thread_1_completed_;
    std::atomic<bool> encoding_thread_2_completed_;
    
    // Stop flag for graceful shutdown
    std::atomic<bool> should_stop_threads_;
    
    // Data buffer state tracking - simplified approach
    // Track which data buffers have been consumed by both threads
    struct DataBufferState {
        bool thread1_copied;  // Thread 1 has copied the data
        bool thread2_copied;  // Thread 2 has copied the data
    };
    std::unordered_map<uintptr_t, DataBufferState> data_buffer_states_;
    std::mutex data_buffer_state_mutex_;
    
    // Buffers ready for release (avoid GIL deadlock by not calling Python callbacks from C++ threads)
    std::queue<uintptr_t> data_buffers_to_release_;
    std::queue<uintptr_t> encoding_buffers_to_release_;
    std::mutex release_queue_mutex_;
    
    // Worker threads
    std::thread encoder_thread_1_;  // First encoding thread (coefficient 1)
    std::thread encoder_thread_2_;  // Second encoding thread (coefficient 2)
    std::thread exchange_thread_;
    
    // NCCL communicator
#ifdef NCCL_AVAILABLE
    ncclComm_t nccl_comm_;
    bool nccl_initialized_;
#else
    bool nccl_initialized_;
#endif

    void start_pipeline() {
        
        // Start two encoder threads
        encoder_thread_1_ = std::thread(&ECCHECKNative::encoder_worker_1, this);
        encoder_thread_2_ = std::thread(&ECCHECKNative::encoder_worker_2, this);
        
        // Start exchange thread
        // std::cout << "EC-CHECK: [Rank " << rank_ << "] Creating exchange thread..." << std::endl;
        // exchange_thread_ = std::thread(&ECCHECKNative::exchange_worker, this);        
    }

public:
    ECCHECKNative(int rank, int world_size, int paired_rank) 
        : rank_(rank), world_size_(world_size), paired_rank_(paired_rank), nccl_initialized_(false),
          encoding_thread_1_completed_(false), encoding_thread_2_completed_(false),
          should_stop_threads_(false) {
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Constructor called, starting pipeline..." << std::endl;
        
        // Start pipeline after NCCL initialization
        start_pipeline();
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Pipeline started successfully" << std::endl;
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
        // Reset completion flags for new encoding round
        encoding_thread_1_completed_ = false;
        encoding_thread_2_completed_ = false;
    }
    
    void wait_for_encoding_completion() {
        // Wait for both encoding threads to signal completion
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Waiting for encoding completion - Thread 1: " 
                  << (encoding_thread_1_completed_ ? "completed" : "pending") 
                  << ", Thread 2: " << (encoding_thread_2_completed_ ? "completed" : "pending") << std::endl;
        
        while (!encoding_thread_1_completed_ || !encoding_thread_2_completed_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Both encoding threads completed" << std::endl;
    }
    
    void stop_pipeline() {
        
        // Signal encoding task queues to wake up threads with mutex protection
        {
            std::lock_guard<std::mutex> lock1(encoding_tasks_1_mutex_);
            encoding_tasks_1_.push({0, 0, 0});  // Sentinel
        }
        {
            std::lock_guard<std::mutex> lock2(encoding_tasks_2_mutex_);
            encoding_tasks_2_.push({0, 0, 0});  // Sentinel
        }
        
        {
            std::lock_guard<std::mutex> lock(exchange_queue_mutex_);
            exchange_queue_.push({-1, 0});  // Sentinel
        }
        exchange_queue_cv_.notify_one();
        
        // Join threads
        if (encoder_thread_1_.joinable()) {
            encoder_thread_1_.join();
        }
        if (encoder_thread_2_.joinable()) {
            encoder_thread_2_.join();
        }
        if (exchange_thread_.joinable()) {
            exchange_thread_.join();
        }
    }
    
    void submit_data_for_encoding_thread1(uintptr_t data_addr, size_t size, uintptr_t encoding_addr) {
        // Store task for thread 1 with mutex protection
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_1_mutex_);
            encoding_tasks_1_.push({data_addr, size, encoding_addr});
        }
        // Notify thread 1 that new task is available
        encoding_tasks_1_cv_.notify_one();
        
        // Initialize data buffer state (only for non-sentinel tasks)
        if (data_addr != 0 && size != 0) {
            std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
            if (data_buffer_states_.find(data_addr) == data_buffer_states_.end()) {
                // First time seeing this buffer - initialize state
                data_buffer_states_[data_addr] = {false, false};
            }
        }
    }
    
    void submit_data_for_encoding_thread2(uintptr_t data_addr, size_t size, uintptr_t encoding_addr) {
        // Store task for thread 2 with mutex protection
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_2_mutex_);
            encoding_tasks_2_.push({data_addr, size, encoding_addr});
        }
        // Notify thread 2 that new task is available
        encoding_tasks_2_cv_.notify_one();
    }
    
    void submit_encoding_for_exchange_with_addr(uintptr_t enc_addr, size_t size) {
        {
            std::lock_guard<std::mutex> lock(exchange_queue_mutex_);
            // Store encoding address for exchange
            exchange_queue_.push({-1, size});  // Use -1 to indicate address-based
        }
        exchange_queue_cv_.notify_one();
    }
    
    void queue_data_buffer_for_release(uintptr_t data_addr) {
        // Add to release queue instead of calling Python callback (avoid GIL deadlock)
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        data_buffers_to_release_.push(data_addr);
    }
    
    void queue_encoding_buffer_for_release(uintptr_t encoding_addr) {
        // Add to release queue instead of calling Python callback (avoid GIL deadlock)
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        encoding_buffers_to_release_.push(encoding_addr);
    }
    
    void mark_data_buffer_copied_by_thread(uintptr_t data_addr, int thread_id) {
        // Mark that this thread has finished reading from the data buffer
        // thread_id: 1 or 2
        bool should_release = false;
        {
            std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
            
            auto it = data_buffer_states_.find(data_addr);
            if (it == data_buffer_states_.end()) {
                std::cout << "EC-CHECK: [Rank " << rank_ << "] WARNING: data_addr=" << data_addr 
                          << " not found in state map (thread " << thread_id << ")" << std::endl;
                return;
            }
            
            // Mark as copied by this thread
            if (thread_id == 1) {
                it->second.thread1_copied = true;
            } else if (thread_id == 2) {
                it->second.thread2_copied = true;
            }
            
            // Check if both threads have copied the data
            if (it->second.thread1_copied && it->second.thread2_copied) {
                // Both threads are done with this buffer - mark for release
                // Remove from tracking
                data_buffer_states_.erase(it);
                should_release = true;
            }
        }
        
        // Queue for release outside of the lock to avoid deadlock
        if (should_release) {
            queue_data_buffer_for_release(data_addr);
        }
    }
    
    // Python-callable methods to get buffers ready for release
    std::vector<uintptr_t> get_data_buffers_to_release() {
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        std::vector<uintptr_t> result;
        while (!data_buffers_to_release_.empty()) {
            result.push_back(data_buffers_to_release_.front());
            data_buffers_to_release_.pop();
        }
        return result;
    }
    
    std::vector<uintptr_t> get_encoding_buffers_to_release() {
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        std::vector<uintptr_t> result;
        while (!encoding_buffers_to_release_.empty()) {
            result.push_back(encoding_buffers_to_release_.front());
            encoding_buffers_to_release_.pop();
        }
        return result;
    }

private:
    void init_nccl() {
#ifdef NCCL_AVAILABLE
        if (world_size_ > 1) {
            try {
                ncclUniqueId nccl_id;
                if (rank_ == 0) {
                    ncclGetUniqueId(&nccl_id);
                    // Broadcast nccl_id to all ranks (simplified)
                }
                
                ncclCommInitRank(&nccl_comm_, world_size_, nccl_id, rank_);
                nccl_initialized_ = true;
                std::cout << "NCCL initialized successfully for rank " << rank_ << std::endl;
            } catch (const std::exception& e) {
                std::cout << "NCCL initialization failed: " << e.what() << std::endl;
                nccl_initialized_ = false;
            }
        }
#else
        std::cout << "NCCL not available, skipping initialization" << std::endl;
#endif
    }
    
    void cleanup_nccl() {
#ifdef NCCL_AVAILABLE
        if (nccl_initialized_) {
            ncclCommDestroy(nccl_comm_);
        }
#endif
    }
    
    void encoder_worker_1() {
        // First encoding thread (coefficient 1)
        while (true) {
            EncodingTask task;
            
            // Wait for tasks using condition variable
            {
                std::unique_lock<std::mutex> lock(encoding_tasks_1_mutex_);
                // Wait until there's a task available
                encoding_tasks_1_cv_.wait(lock, [this] { return !encoding_tasks_1_.empty(); });
                
                task = encoding_tasks_1_.front();
                encoding_tasks_1_.pop();
            }
            
            if (task.data_addr == 0 && task.size == 0) {
                // End of stream - signal completion but continue waiting for next round
                encoding_thread_1_completed_ = true;
                continue;  // Continue waiting for next round, don't exit
            }
        
            // Now perform encoding (can be slow, but data buffer is already marked as copied)
            encode_with_coefficient(task.data_addr, task.size, task.encoding_addr, 1);
            
            // CRITICAL: Mark data buffer as copied by this thread FIRST
            // This allows the buffer to be released as soon as both threads copy the data
            mark_data_buffer_copied_by_thread(task.data_addr, 1);

            // Queue encoding buffer for release (avoid GIL deadlock)
            queue_encoding_buffer_for_release(task.encoding_addr);
            
        }
    }
    
    void encoder_worker_2() {
        // Second encoding thread (coefficient 2)
        while (true) {
            EncodingTask task;
            
            // Wait for tasks using condition variable
            {
                std::unique_lock<std::mutex> lock(encoding_tasks_2_mutex_);
                // Wait until there's a task available
                encoding_tasks_2_cv_.wait(lock, [this] { return !encoding_tasks_2_.empty(); });
                
                task = encoding_tasks_2_.front();
                encoding_tasks_2_.pop();
            }
            
            if (task.data_addr == 0 && task.size == 0) {
                // End of stream - signal completion but continue waiting for next round
                encoding_thread_2_completed_ = true;
                continue;  // Continue waiting for next round, don't exit
            }
            
            // Now perform encoding (can be slow, but data buffer is already marked as copied)
            encode_with_coefficient(task.data_addr, task.size, task.encoding_addr, 2);

            // CRITICAL: Mark data buffer as copied by this thread FIRST
            // This allows the buffer to be released as soon as both threads copy the data
            mark_data_buffer_copied_by_thread(task.data_addr, 2);
            
            // Queue encoding buffer for release (avoid GIL deadlock)
            queue_encoding_buffer_for_release(task.encoding_addr);
            
        }
    }
    
    void encode_with_coefficient(uintptr_t src_addr, size_t size, uintptr_t dst_addr, int coefficient) {
        // Simple coefficient encoding: multiply by coefficient
        uint8_t* src = reinterpret_cast<uint8_t*>(src_addr);
        uint8_t* dst = reinterpret_cast<uint8_t*>(dst_addr);
        
        for (size_t i = 0; i < size; ++i) {
            dst[i] = src[i] * coefficient;
        }
        // std::cout << "EC-CHECK: Encoding with coefficient " << coefficient << " for size " << size << std::endl;
        // todo: implement the encoding with coefficient
        // return;
    }
    
    void nccl_sendrecv(uintptr_t addr, size_t size) {
#ifdef NCCL_AVAILABLE
        if (nccl_initialized_ && world_size_ > 1) {
            // Send to paired rank
            ncclSend(reinterpret_cast<void*>(addr), size, ncclUint8, paired_rank_, nccl_comm_, 0);
            
            // Receive from paired rank (simplified - should use separate recv buffer)
            ncclRecv(reinterpret_cast<void*>(addr), size, ncclUint8, paired_rank_, nccl_comm_, 0);
        }
#else
        // Fallback: no-op if NCCL not available
        std::cout << "NCCL not available, skipping send/recv for addr=" << addr << ", size=" << size << std::endl;
#endif
    }
};

// Python bindings
PYBIND11_MODULE(eccheck_native, m) {
    m.doc() = "EC-CHECK native C++ implementation";
    
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
