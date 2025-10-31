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
#include <optional>
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
    
    // ========== Vectorized: per-column queues, mutexes, CVs, threads ==========
    
    static constexpr int MAX_COLUMNS = 32;  // Maximum supported columns
    
    // XOR mode enumeration
    enum class XorMode {
        WITH_RECV_ENCODING = 0,  // Default: local_encoding XOR recv_encoding -> parity
        WITH_ZERO_PARITY = 1,    // local_encoding XOR zero_parity -> parity
        INCREMENTAL = 2          // local_encoding XOR existing_parity -> parity
    };
    
    struct EncodingTask {
        uintptr_t data_addr;
        size_t size;
        uintptr_t encoding_addr;
        uintptr_t recv_addr;        // 接收地址（0表示不需要recv）
        size_t recv_chunk_size;     // 接收大小（0表示不需要recv）
        uintptr_t parity_addr;      // Parity缓冲区地址（XOR结果）
        uintptr_t zero_parity_addr; // 零初始化parity缓冲区地址（用于with_zero_parity模式）
        XorMode xor_mode;           // XOR模式
        int send_count;             // Number of times to send (default: 1)
        int recv_count;             // Number of times to recv (default: 1)
        EncodingTask() : data_addr(0), size(0), encoding_addr(0), recv_addr(0), recv_chunk_size(0),
                        parity_addr(0), zero_parity_addr(0), xor_mode(XorMode::WITH_RECV_ENCODING),
                        send_count(1), recv_count(1) {}
    };
    
    struct SendTask {
        uintptr_t encoding_addr;
        size_t size;
        int count;  // Number of times to send (default: 1)
        SendTask() : encoding_addr(0), size(0), count(1) {}
        SendTask(uintptr_t addr, size_t sz, int cnt = 1) : encoding_addr(addr), size(sz), count(cnt) {}
    };
    
    struct RecvTask {
        uintptr_t recv_addr;
        size_t size;
        int count;  // Number of times to recv (default: 1)
        RecvTask() : recv_addr(0), size(0), count(1) {}
        RecvTask(uintptr_t addr, size_t sz, int cnt = 1) : recv_addr(addr), size(sz), count(cnt) {}
    };
    
    struct XorTask {
        uintptr_t local_encoding_addr;  // 本地编码数据地址
        uintptr_t recv_encoding_addr;   // 接收到的编码数据地址（或零初始化parity地址）
        uintptr_t parity_addr;          // Parity缓冲区地址（XOR结果）
        size_t size;                    // 数据大小
        XorMode mode;                   // XOR模式
    };
    
    // Per-column queues and synchronization (use std::array for non-movable types)
    std::array<std::queue<EncodingTask>, MAX_COLUMNS> encoding_tasks_;  // One per column
    std::array<std::mutex, MAX_COLUMNS> encoding_tasks_mutex_;
    std::array<std::condition_variable, MAX_COLUMNS> encoding_tasks_cv_;
    
    std::array<std::queue<SendTask>, MAX_COLUMNS> send_queues_;
    std::array<std::mutex, MAX_COLUMNS> send_queue_mutex_;
    std::array<std::condition_variable, MAX_COLUMNS> send_queue_cv_;
    
    std::array<std::queue<RecvTask>, MAX_COLUMNS> recv_queues_;
    std::array<std::mutex, MAX_COLUMNS> recv_queue_mutex_;
    std::array<std::condition_variable, MAX_COLUMNS> recv_queue_cv_;
    
    std::array<std::queue<XorTask>, MAX_COLUMNS> xor_queues_;
    std::array<std::mutex, MAX_COLUMNS> xor_queue_mutex_;
    std::array<std::condition_variable, MAX_COLUMNS> xor_queue_cv_;
    
    // Completion flags (per column)
    std::array<std::atomic<bool>, MAX_COLUMNS> encoding_thread_completed_;
    std::array<std::atomic<bool>, MAX_COLUMNS> send_worker_completed_;
    std::array<std::atomic<bool>, MAX_COLUMNS> recv_worker_completed_;
    std::array<std::atomic<bool>, MAX_COLUMNS> xor_worker_completed_;
    
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
        XorMode xor_mode;
    };
    std::array<std::unordered_map<uintptr_t, RecvMapping>, MAX_COLUMNS> recv_to_xor_mappings_;  // One per column
    std::array<std::mutex, MAX_COLUMNS> recv_mapping_mutex_;
    
    // Buffers ready for release
    std::queue<uintptr_t> data_buffers_to_release_;
    std::queue<uintptr_t> encoding_buffers_to_release_;
    std::queue<uintptr_t> parity_buffers_to_release_;
    std::mutex release_queue_mutex_;
    
    // Mutex for shared parity buffer updates (all columns share the same parity)
    // Key: parity buffer address, Value: mutex for that buffer
    std::unordered_map<uintptr_t, std::mutex*> parity_buffer_mutexes_;
    std::mutex parity_mutex_map_mutex_;  // Protects the map itself
    
    // Persistent stores for final results (in-memory), optional
    uintptr_t persist_recv_base_ = 0;      // Base address for persistent recv store
    uintptr_t persist_parity_base_ = 0;    // Base address for persistent parity store
    size_t persist_recv_capacity_ = 0;     // Total bytes capacity of recv store
    size_t persist_parity_capacity_ = 0;   // Total bytes capacity of parity store
    std::atomic<size_t> persist_recv_offset_{0};
    std::atomic<size_t> persist_parity_offset_{0};
    std::atomic<bool> persist_recv_enabled_{false};
    std::atomic<bool> persist_parity_enabled_{false};
    
    // Worker threads - 4 threads per column (encoder, send, recv, xor)
    // Use optional because std::thread is not default-constructible in C++11
    std::array<std::optional<std::thread>, MAX_COLUMNS> encoder_threads_;
    std::array<std::optional<std::thread>, MAX_COLUMNS> send_workers_;
    std::array<std::optional<std::thread>, MAX_COLUMNS> recv_workers_;
    std::array<std::optional<std::thread>, MAX_COLUMNS> xor_workers_;
    
    // NCCL communicators - one per column
#ifdef NCCL_AVAILABLE
    std::array<ncclComm_t, MAX_COLUMNS> nccl_comms_;  // One per column
    std::array<bool, MAX_COLUMNS> nccl_initialized_;
#endif

    // EC parameters (k, rows=2) and tables
    int k_;
    int rows_;
    int data_block_index_;
    unsigned char *a_mat_;    // RS matrix (k * m)
    unsigned char *g_tbls_;   // tables produced by ec_init_tables (32 * k * rows)

    // Synchronization for NCCL initialization
    std::array<std::atomic<bool>, MAX_COLUMNS> nccl_init_completed_;  // One per column
    std::mutex nccl_init_mutex_;
    std::condition_variable nccl_init_cv_;

    // ========== NCCL初始化函数 ==========
    
    void init_nccl_column(int column_idx) {
#ifdef NCCL_AVAILABLE
        if (column_idx < 0 || column_idx >= num_columns_) return;
        
        ncclUniqueId nccl_id;
        std::string id_file = "/tmp/eccheck_nccl_column" + std::to_string(column_idx) + "_id.txt";
        
        if (rank_ == 0) {
            // Rank 0 creates NCCL ID and writes to file
            ncclGetUniqueId(&nccl_id);
            std::ofstream outfile(id_file, std::ios::binary);
            outfile.write(reinterpret_cast<char*>(&nccl_id), sizeof(ncclUniqueId));
            outfile.close();
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Column" << column_idx << " NCCL ID written to " << id_file << std::endl;
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
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Column" << column_idx << " NCCL ID read from " << id_file << std::endl;
        }
        
        // Initialize NCCL communicator
        ncclCommInitRank(&nccl_comms_[column_idx], world_size_, nccl_id, rank_);
        nccl_initialized_[column_idx] = true;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Column" << column_idx << " NCCL communicator initialized" << std::endl;
        
        // Signal completion
        {
            std::lock_guard<std::mutex> lock(nccl_init_mutex_);
            nccl_init_completed_[column_idx] = true;
        }
        nccl_init_cv_.notify_all();
#else
        std::cout << "EC-CHECK: NCCL not available, skipping column" << column_idx << " initialization" << std::endl;
        nccl_initialized_[column_idx] = false;
        nccl_init_completed_[column_idx] = true;
        nccl_init_cv_.notify_all();
#endif
    }
    
    // Legacy functions for backward compatibility
    void init_nccl_thread1() { init_nccl_column(0); }
    void init_nccl_thread2() { init_nccl_column(1); }
    
    void cleanup_nccl() {
#ifdef NCCL_AVAILABLE
        for (int i = 0; i < num_columns_; ++i) {
            if (nccl_initialized_[i]) {
                ncclCommDestroy(nccl_comms_[i]);
        }
        }
#endif
    }

    // Columns runtime config (vectorized for N columns)
    struct ColumnCfg { int coefficient; int send_peer; int recv_peer; };
    std::vector<ColumnCfg> column_configs_;
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
    
    // Unified encoder worker for column_idx
    void encoder_worker(int column_idx) {
        if (column_idx < 0 || column_idx >= num_columns_) return;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread " << column_idx << " started" << std::endl;
        
        while (!should_stop_threads_) {
            EncodingTask task;
            
            {
                std::unique_lock<std::mutex> lock(encoding_tasks_mutex_[column_idx]);
                encoding_tasks_cv_[column_idx].wait(lock, [this, column_idx] {
                    return !encoding_tasks_[column_idx].empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && encoding_tasks_[column_idx].empty()) {
                    break;
                }
                
                task = encoding_tasks_[column_idx].front();
                encoding_tasks_[column_idx].pop();
            }
            
            // Check for sentinel (end signal): all fields are 0
            if (task.data_addr == 0 && task.size == 0 && 
                task.encoding_addr == 0 && task.recv_addr == 0 && task.recv_chunk_size == 0 && 
                task.parity_addr == 0 && task.zero_parity_addr == 0) {
                encoding_thread_completed_[column_idx] = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread " << column_idx << " received sentinel, marking completed" << std::endl;
                
                // Submit sentinel to send_worker
                {
                    std::lock_guard<std::mutex> lock(send_queue_mutex_[column_idx]);
                    send_queues_[column_idx].push(SendTask(0, 0, 1));
                }
                send_queue_cv_[column_idx].notify_one();
                
                // Submit sentinel to recv_worker
                {
                    std::lock_guard<std::mutex> lock(recv_queue_mutex_[column_idx]);
                    recv_queues_[column_idx].push(RecvTask(0, 0, 1));
                }
                recv_queue_cv_[column_idx].notify_one();
                
                // Submit sentinel to xor_worker
                {
                    std::lock_guard<std::mutex> lock(xor_queue_mutex_[column_idx]);
                    xor_queues_[column_idx].push({0, 0, 0, 0, XorMode::WITH_RECV_ENCODING});
                }
                xor_queue_cv_[column_idx].notify_one();
                
                continue;  // Continue waiting for next round
            }
            
            // Check if we need to encode/send data
            bool need_encode = (task.data_addr != 0 && task.encoding_addr != 0);
            
            if (need_encode) {
                // Perform encoding selecting GF table by (parity_idx=column_idx, data_block_index_=rank%k)
                encode_with_parity_index(task.data_addr, task.size, task.encoding_addr, column_idx);
                
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
                
                // Submit encoding result to send_worker (if send is needed)
                // Always send for now (can be made conditional based on pipeline config)
                // Increment encoding refcount for pending send (count times)
                if (task.encoding_addr != 0) {
                    std::lock_guard<std::mutex> rlock(encoding_ref_counts_mutex_);
                    encoding_ref_counts_[task.encoding_addr] += task.send_count;
                }
                {
                    std::lock_guard<std::mutex> lock(send_queue_mutex_[column_idx]);
                    send_queues_[column_idx].push(SendTask(task.encoding_addr, task.size, task.send_count));
                }
                send_queue_cv_[column_idx].notify_one();
            }
            
            // Check if we need to receive data
            bool need_recv = (task.recv_addr != 0 && task.recv_chunk_size != 0);
            
            if (need_recv) {
                // Submit recv task to recv_worker (with count)
                {
                    std::lock_guard<std::mutex> lock(recv_queue_mutex_[column_idx]);
                    recv_queues_[column_idx].push(RecvTask(task.recv_addr, task.recv_chunk_size, task.recv_count));
                }
                recv_queue_cv_[column_idx].notify_one();
            } else if (task.parity_addr != 0 && task.zero_parity_addr != 0 && 
                       task.xor_mode == XorMode::WITH_ZERO_PARITY) {
                // If recv is not needed but we have zero_parity_addr, submit XOR directly
                // This handles the case where first column XORs with zero-initialized parity
                {
                    std::lock_guard<std::mutex> lock(xor_queue_mutex_[column_idx]);
                    xor_queues_[column_idx].push({
                        task.encoding_addr,           // local_encoding
                        task.zero_parity_addr,        // zero_parity (used as second source)
                        task.parity_addr,              // result parity
                        task.size,                     // size
                        XorMode::WITH_ZERO_PARITY      // mode
                    });
                }
                xor_queue_cv_[column_idx].notify_one();
                
                // Increment encoding refcount for XOR
                if (task.encoding_addr != 0) {
                    std::lock_guard<std::mutex> rlock(encoding_ref_counts_mutex_);
                    encoding_ref_counts_[task.encoding_addr] += 1;
                }
            }
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Encoder thread " << column_idx << " exiting" << std::endl;
    }
    
    // Legacy functions for backward compatibility
    void encoder_worker_1() { encoder_worker(0); }
    void encoder_worker_2() { encoder_worker(1); }
    
    // Unified send worker for column_idx
    void send_worker(int column_idx) {
        if (column_idx < 0 || column_idx >= num_columns_) return;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker " << column_idx << " started" << std::endl;
        
        // Initialize NCCL for this column
        init_nccl_column(column_idx);
        
        while (!should_stop_threads_) {
            SendTask task;
            
            {
                std::unique_lock<std::mutex> lock(send_queue_mutex_[column_idx]);
                send_queue_cv_[column_idx].wait(lock, [this, column_idx] {
                    return !send_queues_[column_idx].empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && send_queues_[column_idx].empty()) {
                    break;
                }
                
                task = send_queues_[column_idx].front();
                send_queues_[column_idx].pop();
            }
            
            // Check for sentinel
            if (task.encoding_addr == 0 && task.size == 0) {
                send_worker_completed_[column_idx] = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker " << column_idx << " received sentinel, marking completed" << std::endl;
                continue;
            }
            
            // Execute send count times
#ifdef NCCL_AVAILABLE
            if (nccl_initialized_[column_idx] && world_size_ > 1) {
                int peer = (column_configs_[column_idx].send_peer >= 0 ? column_configs_[column_idx].send_peer : paired_rank_);
                for (int i = 0; i < task.count; ++i) {
                    ncclGroupStart(); 
                    ncclSend(reinterpret_cast<void*>(task.encoding_addr), task.size, 
                             ncclUint8, peer, nccl_comms_[column_idx], 0);
                    ncclGroupEnd();
                    if (task.count > 1 && i < task.count - 1) {
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker " << column_idx 
                                  << " completed send " << (i + 1) << "/" << task.count << std::endl;
                    }
                }
                if (task.count > 1) {
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker " << column_idx 
                              << " completed all " << task.count << " sends" << std::endl;
                }
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
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Send worker " << column_idx << " exiting" << std::endl;
    }
    
    // Legacy functions for backward compatibility
    void send_worker_1() { send_worker(0); }
    void send_worker_2() { send_worker(1); }
    
    // Unified recv worker for column_idx
    void recv_worker(int column_idx) {
        if (column_idx < 0 || column_idx >= num_columns_) return;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker " << column_idx << " started" << std::endl;
        
        // Wait for NCCL initialization (already done by send_worker)
        {
            std::unique_lock<std::mutex> lock(nccl_init_mutex_);
            nccl_init_cv_.wait(lock, [this, column_idx] {
                return nccl_init_completed_[column_idx].load();
            });
        }
        
        while (!should_stop_threads_) {
            RecvTask task;
            
            {
                std::unique_lock<std::mutex> lock(recv_queue_mutex_[column_idx]);
                recv_queue_cv_[column_idx].wait(lock, [this, column_idx] {
                    return !recv_queues_[column_idx].empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && recv_queues_[column_idx].empty()) {
                    break;
                }
                
                task = recv_queues_[column_idx].front();
                recv_queues_[column_idx].pop();
            }
            
            // Check for sentinel
            if (task.recv_addr == 0 && task.size == 0) {
                recv_worker_completed_[column_idx] = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker " << column_idx << " received sentinel, marking completed" << std::endl;
                continue;
            }
            
            // Execute recv count times
            // Note: For multiple recv, we use sequential offsets in the recv buffer
#ifdef NCCL_AVAILABLE
            if (nccl_initialized_[column_idx] && world_size_ > 1) {
                int peer = (column_configs_[column_idx].recv_peer >= 0 ? column_configs_[column_idx].recv_peer : paired_rank_);
                for (int i = 0; i < task.count; ++i) {
                    uintptr_t recv_offset_addr = task.recv_addr + (i * task.size);
                    ncclGroupStart();
                    ncclRecv(reinterpret_cast<void*>(recv_offset_addr), task.size,
                             ncclUint8, peer, nccl_comms_[column_idx], 0);
                    ncclGroupEnd();
                    if (task.count > 1 && i < task.count - 1) {
                        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker " << column_idx 
                                  << " completed recv " << (i + 1) << "/" << task.count << std::endl;
                    }
                }
                if (task.count > 1) {
                    std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker " << column_idx 
                              << " completed all " << task.count << " recvs" << std::endl;
                }
            }
#endif
            
            // Persist recv data into continuous store if enabled (column0 only to avoid double-writing)
            if (column_idx == 0 && persist_recv_enabled_.load() && persist_recv_base_ != 0 && task.size > 0) {
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
            
            // After receiving, submit XOR task for incremental update: parity XOR recv_encoding -> parity
            {
                std::lock_guard<std::mutex> lock(recv_mapping_mutex_[column_idx]);
                auto it = recv_to_xor_mappings_[column_idx].find(task.recv_addr);
                if (it != recv_to_xor_mappings_[column_idx].end()) {
                    const RecvMapping& mapping = it->second;
                    
                    // For incremental mode (shared parity), we don't need local encoding refcount
                    // We only XOR the received encoding with existing parity
                    if (mapping.xor_mode != XorMode::INCREMENTAL) {
                        // For other modes, increment encoding refcount
                        {
                            std::lock_guard<std::mutex> rlock(encoding_ref_counts_mutex_);
                            encoding_ref_counts_[mapping.encoding_addr] += 1;
                        }
                    }
                    
                    // Submit XOR task
                    {
                        std::lock_guard<std::mutex> xor_lock(xor_queue_mutex_[column_idx]);
                        if (mapping.xor_mode == XorMode::INCREMENTAL) {
                            // For incremental: parity XOR recv_encoding -> parity (in-place)
                            // Pass parity_addr as local_encoding_addr (not used), recv_addr as recv_encoding, parity_addr as both source and dest
                            xor_queues_[column_idx].push({
                                mapping.parity_addr,    // local_encoding_addr (not used in incremental, but required by struct)
                                task.recv_addr,        // recv_encoding_addr (the received encoding to XOR)
                                mapping.parity_addr,    // parity_addr (both source and destination)
                                mapping.size,
                                XorMode::INCREMENTAL
                            });
                        } else {
                            // For other modes: local encoding XOR received encoding -> parity
                            xor_queues_[column_idx].push({
                                mapping.encoding_addr, task.recv_addr, mapping.parity_addr, 
                                mapping.size, mapping.xor_mode
                            });
                        }
                    }
                    xor_queue_cv_[column_idx].notify_one();
                    // Remove from mapping after submitting
                    recv_to_xor_mappings_[column_idx].erase(it);
                } else {
                    std::cerr << "EC-CHECK: [Rank " << rank_ << "] Warning: No mapping found for recv_addr=" << task.recv_addr << std::endl;
                }
            }
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Recv worker " << column_idx << " exiting" << std::endl;
    }
    
    // Legacy functions for backward compatibility
    void recv_worker_1() { recv_worker(0); }
    void recv_worker_2() { recv_worker(1); }
    
    // Unified XOR worker for column_idx
    void xor_worker(int column_idx) {
        if (column_idx < 0 || column_idx >= num_columns_) return;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker " << column_idx << " started" << std::endl;
        
        while (!should_stop_threads_) {
            XorTask task;
            
            {
                std::unique_lock<std::mutex> lock(xor_queue_mutex_[column_idx]);
                xor_queue_cv_[column_idx].wait(lock, [this, column_idx] {
                    return !xor_queues_[column_idx].empty() || should_stop_threads_;
                });
                
                if (should_stop_threads_ && xor_queues_[column_idx].empty()) {
                    break;
                }
                
                task = xor_queues_[column_idx].front();
                xor_queues_[column_idx].pop();
            }
            
            // Check for sentinel
            if (task.local_encoding_addr == 0 && task.recv_encoding_addr == 0 && 
                task.parity_addr == 0 && task.size == 0) {
                xor_worker_completed_[column_idx] = true;
                std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker " << column_idx << " received sentinel, marking completed" << std::endl;
                continue;
            }
            
            // Perform XOR based on mode
            int result = 0;
            
            if (task.mode == XorMode::WITH_ZERO_PARITY) {
                // Mode 1: local_encoding XOR zero_parity -> parity
                // Column 0: Initialize parity buffer with encoding XOR zero
                void* xor_array[3];
                xor_array[0] = reinterpret_cast<void*>(task.local_encoding_addr);
                xor_array[1] = reinterpret_cast<void*>(task.recv_encoding_addr);  // This is zero_parity_addr in this mode
                xor_array[2] = reinterpret_cast<void*>(task.parity_addr);
                
                // For column 0, we don't need locking since it's the first update
                // But still get/create mutex for future incremental updates
                {
                    std::lock_guard<std::mutex> map_lock(parity_mutex_map_mutex_);
                    if (parity_buffer_mutexes_.find(task.parity_addr) == parity_buffer_mutexes_.end()) {
                        parity_buffer_mutexes_[task.parity_addr] = new std::mutex();
                    }
                }
                
                result = xor_gen(3, (int)task.size, xor_array);
            } else if (task.mode == XorMode::WITH_RECV_ENCODING) {
                // Mode 0: local_encoding XOR recv_encoding -> parity
                void* xor_array[3];
                xor_array[0] = reinterpret_cast<void*>(task.local_encoding_addr);
                xor_array[1] = reinterpret_cast<void*>(task.recv_encoding_addr);
                xor_array[2] = reinterpret_cast<void*>(task.parity_addr);
                result = xor_gen(3, (int)task.size, xor_array);
            } else if (task.mode == XorMode::INCREMENTAL) {
                // Mode 2: parity XOR recv_encoding -> parity (in-place update)
                // For incremental update of shared parity: existing_parity XOR recv_encoding -> parity
                // Note: local_encoding is not used in this mode, only recv_encoding
                void* xor_array[3];
                xor_array[0] = reinterpret_cast<void*>(task.parity_addr);     // Source 1: existing parity
                xor_array[1] = reinterpret_cast<void*>(task.recv_encoding_addr);  // Source 2: recv_encoding
                xor_array[2] = reinterpret_cast<void*>(task.parity_addr);  // Destination: update in-place
                
                // Get mutex for this shared parity buffer to ensure thread safety
                std::mutex* parity_mutex = nullptr;
                {
                    std::lock_guard<std::mutex> map_lock(parity_mutex_map_mutex_);
                    auto it = parity_buffer_mutexes_.find(task.parity_addr);
                    if (it == parity_buffer_mutexes_.end()) {
                        // Create new mutex for this parity buffer
                        parity_mutex = new std::mutex();
                        parity_buffer_mutexes_[task.parity_addr] = parity_mutex;
                    } else {
                        parity_mutex = it->second;
                    }
                }
                
                // Lock the parity buffer before XOR update
                std::lock_guard<std::mutex> parity_lock(*parity_mutex);
                result = xor_gen(3, (int)task.size, xor_array);
            } else {
                // Fallback to default mode
                void* xor_array[3];
                xor_array[0] = reinterpret_cast<void*>(task.local_encoding_addr);
                xor_array[1] = reinterpret_cast<void*>(task.recv_encoding_addr);
                xor_array[2] = reinterpret_cast<void*>(task.parity_addr);
                result = xor_gen(3, (int)task.size, xor_array);
            }
            
            if (result != 0) {
                std::cerr << "EC-CHECK: [Rank " << rank_ << "] XOR worker " << column_idx 
                          << " failed: xor_gen returned " << result << " (mode=" << (int)task.mode << ")" << std::endl;
            }
            
            // Persist parity result into continuous store if enabled (column0 only to avoid double-writing)
            if (column_idx == 0 && persist_parity_enabled_.load() && persist_parity_base_ != 0 && task.size > 0) {
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
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] XOR worker " << column_idx << " exiting" << std::endl;
    }
    
    // Legacy functions for backward compatibility
    void xor_worker_1() { xor_worker(0); }
    void xor_worker_2() { xor_worker(1); }

    void start_pipeline() {
        // Clean up any existing NCCL ID files before starting (all columns)
        for (int i = 0; i < num_columns_; ++i) {
            std::string id_file = "/tmp/eccheck_nccl_column" + std::to_string(i) + "_id.txt";
            std::ifstream file(id_file);
            if (file.good()) {
                file.close();
                std::remove(id_file.c_str());
                std::cout << "EC-CHECK: [Rank " << rank_ << "] Removed existing column" << i << " NCCL ID file" << std::endl;
            }
        }
        
        // Also clean up legacy thread1/thread2 files for backward compatibility
        std::string thread1_id_file = "/tmp/eccheck_nccl_thread1_id.txt";
        std::string thread2_id_file = "/tmp/eccheck_nccl_thread2_id.txt";
        std::ifstream file1(thread1_id_file);
        if (file1.good()) {
            file1.close();
            std::remove(thread1_id_file.c_str());
        }
        std::ifstream file2(thread2_id_file);
        if (file2.good()) {
            file2.close();
            std::remove(thread2_id_file.c_str());
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] NCCL ID file cleanup completed" << std::endl;
        
        // Join any existing threads first (should be empty if called correctly)
        for (int i = 0; i < num_columns_; ++i) {
            if (encoder_threads_[i].has_value() && encoder_threads_[i]->joinable()) encoder_threads_[i]->join();
            if (send_workers_[i].has_value() && send_workers_[i]->joinable()) send_workers_[i]->join();
            if (recv_workers_[i].has_value() && recv_workers_[i]->joinable()) recv_workers_[i]->join();
            if (xor_workers_[i].has_value() && xor_workers_[i]->joinable()) xor_workers_[i]->join();
        }
        
        // Start 4 threads per column: encoder, send, recv, xor
        for (int i = 0; i < num_columns_; ++i) {
            encoder_threads_[i].emplace(&ECCHECKNative::encoder_worker, this, i);
            send_workers_[i].emplace(&ECCHECKNative::send_worker, this, i);
            recv_workers_[i].emplace(&ECCHECKNative::recv_worker, this, i);
            xor_workers_[i].emplace(&ECCHECKNative::xor_worker, this, i);
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Started " << (4 * num_columns_) 
                  << " threads (" << num_columns_ << " encoding + " << num_columns_ 
                  << " send + " << num_columns_ << " recv + " << num_columns_ << " XOR)" << std::endl;
    }

public:
    ECCHECKNative(int rank, int world_size, int paired_rank) 
        : rank_(rank), world_size_(world_size), paired_rank_(paired_rank), 
          should_stop_threads_(false),
          k_(0), rows_(0), data_block_index_(0), a_mat_(nullptr), g_tbls_(nullptr) {
        
        // Initialize vectors for default 2 columns (will be resized by set_columns_config if needed)
        num_columns_ = 2;
        column_configs_.resize(num_columns_);
        column_configs_[0] = {0, -1, -1};
        column_configs_[1] = {1, -1, -1};
        
        // std::array automatically initializes all elements
        // Initialize atomic<bool> elements explicitly
        for (int i = 0; i < MAX_COLUMNS; ++i) {
            encoding_thread_completed_[i] = false;
            send_worker_completed_[i] = false;
            recv_worker_completed_[i] = false;
            xor_worker_completed_[i] = false;
            nccl_init_completed_[i] = false;
#ifdef NCCL_AVAILABLE
            nccl_initialized_[i] = false;
#endif
        }

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

        // Wait for all NCCL communicators to be initialized
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Waiting for NCCL initialization (" << num_columns_ << " columns)..." << std::endl;
        std::unique_lock<std::mutex> lock(nccl_init_mutex_);
        nccl_init_cv_.wait(lock, [this] { 
            for (int i = 0; i < num_columns_; ++i) {
                if (!nccl_init_completed_[i].load()) return false;
            }
            return true;
        });

        std::cout << "EC-CHECK: [Rank " << rank_ << "] Pipeline and NCCL initialized successfully" << std::endl;
    }
    
    void set_columns_config(const std::vector<std::map<std::string, int>>& cols) {
        int new_num_columns = std::max(1, (int)cols.size());
        
        // Check limit
        if (new_num_columns > MAX_COLUMNS) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Error: requested " << new_num_columns 
                      << " columns, but MAX_COLUMNS=" << MAX_COLUMNS << std::endl;
            new_num_columns = MAX_COLUMNS;
        }
        
        // All arrays are pre-allocated with MAX_COLUMNS elements in constructor
        // We just need to update num_columns_ and column_configs_
        num_columns_ = new_num_columns;
        column_configs_.resize(num_columns_);
        
        // Parse column configs
        for (size_t i = 0; i < cols.size() && i < (size_t)num_columns_; ++i) {
            const auto &c = cols[i];
            column_configs_[i].coefficient = c.count("coefficient") ? c.at("coefficient") : (int)i;
            column_configs_[i].send_peer   = c.count("send_peer")   ? c.at("send_peer")   : -1;
            column_configs_[i].recv_peer   = c.count("recv_peer")   ? c.at("recv_peer")   : -1;
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Columns config applied:";
        for (int i = 0; i < num_columns_; ++i) {
            std::cout << " col" << i << "(coef=" << column_configs_[i].coefficient
                      << ", send_peer=" << (column_configs_[i].send_peer>=0?column_configs_[i].send_peer:paired_rank_)
                      << ", recv_peer=" << (column_configs_[i].recv_peer>=0?column_configs_[i].recv_peer:paired_rank_)
                      << ")";
        }
        std::cout << std::endl;
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
        
        // Cleanup parity buffer mutexes to prevent memory leaks
        {
            std::lock_guard<std::mutex> map_lock(parity_mutex_map_mutex_);
            for (auto& pair : parity_buffer_mutexes_) {
                delete pair.second;
            }
            parity_buffer_mutexes_.clear();
        }
    }
    
    void set_buffer_addresses(const std::vector<uintptr_t>& data_addrs,
                             const std::vector<uintptr_t>& encoding_addrs,
                             const std::vector<size_t>& sizes) {
        data_buffer_addrs_ = data_addrs;
        encoding_buffer_addrs_ = encoding_addrs;
        buffer_sizes_ = sizes;
    }
    
    void reset_encoding_completion_flags() {
        for (int i = 0; i < num_columns_; ++i) {
            encoding_thread_completed_[i] = false;
            send_worker_completed_[i] = false;
            recv_worker_completed_[i] = false;
            xor_worker_completed_[i] = false;
        }
    }
    
    void wait_for_encoding_completion() {
        // Wait for all encoding threads to complete
        while (true) {
            bool all_done = true;
            for (int i = 0; i < num_columns_; ++i) {
                if (!encoding_thread_completed_[i].load()) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] All " << num_columns_ << " encoding threads completed" << std::endl;
        
        // Wait for all send workers to complete
        while (true) {
            bool all_done = true;
            for (int i = 0; i < num_columns_; ++i) {
                if (!send_worker_completed_[i].load()) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] All " << num_columns_ << " send workers completed" << std::endl;
        
        // Wait for all recv workers to complete
        while (true) {
            bool all_done = true;
            for (int i = 0; i < num_columns_; ++i) {
                if (!recv_worker_completed_[i].load()) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] All " << num_columns_ << " recv workers completed" << std::endl;
        
        // Wait for all XOR workers to complete
        while (true) {
            bool all_done = true;
            for (int i = 0; i < num_columns_; ++i) {
                if (!xor_worker_completed_[i].load()) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "EC-CHECK: [Rank " << rank_ << "] All " << num_columns_ << " XOR workers completed" << std::endl;
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] All threads completed (" 
                  << num_columns_ << " encoding + " << num_columns_ << " send + " 
                  << num_columns_ << " recv + " << num_columns_ << " XOR)" << std::endl;
    }
    
    void stop_pipeline() {
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Stopping pipeline..." << std::endl;
        
        should_stop_threads_ = true;
        
        // Notify all threads (all columns)
        for (int i = 0; i < num_columns_; ++i) {
            encoding_tasks_cv_[i].notify_all();
            send_queue_cv_[i].notify_all();
            recv_queue_cv_[i].notify_all();
            xor_queue_cv_[i].notify_all();
        }
        
        // Join all threads (all columns)
        for (int i = 0; i < num_columns_; ++i) {
            if (encoder_threads_[i].has_value() && encoder_threads_[i]->joinable()) encoder_threads_[i]->join();
            if (send_workers_[i].has_value() && send_workers_[i]->joinable()) send_workers_[i]->join();
            if (recv_workers_[i].has_value() && recv_workers_[i]->joinable()) recv_workers_[i]->join();
            if (xor_workers_[i].has_value() && xor_workers_[i]->joinable()) xor_workers_[i]->join();
        }
        
        std::cout << "EC-CHECK: [Rank " << rank_ << "] Pipeline stopped" << std::endl;
    }
    
    // Generic submit API for N-column support: routes to column-specific queue
    // Extended to support zero_parity_addr and xor_mode
    void submit_data_for_encoding(int column_idx,
                                  uintptr_t data_addr, size_t size,
                                          uintptr_t encoding_addr, uintptr_t recv_addr, 
                                  size_t recv_chunk_size, uintptr_t parity_addr,
                                  uintptr_t zero_parity_addr = 0,
                                  int xor_mode_int = 0,
                                  int send_count = 1,
                                  int recv_count = 1) {
        if (column_idx < 0 || column_idx >= num_columns_) {
            std::cerr << "EC-CHECK: submit_data_for_encoding: column_idx=" << column_idx
                      << " out of range (num_columns_=" << num_columns_ << ")" << std::endl;
#ifdef PYBIND11_VERSION
            throw pybind11::value_error("Column index out of range");
#else
            return;
#endif
        }
        
        XorMode xor_mode = static_cast<XorMode>(xor_mode_int);
        if (xor_mode_int < 0 || xor_mode_int > 2) {
            xor_mode = XorMode::WITH_RECV_ENCODING;  // Default
        }
        
        // Push to column-specific encoding queue
        {
            std::lock_guard<std::mutex> lock(encoding_tasks_mutex_[column_idx]);
        EncodingTask task;
        task.data_addr = data_addr;
        task.size = size;
        task.encoding_addr = encoding_addr;
        task.recv_addr = recv_addr;
        task.recv_chunk_size = recv_chunk_size;
        task.parity_addr = parity_addr;
        task.zero_parity_addr = zero_parity_addr;
        task.xor_mode = xor_mode;
        task.send_count = send_count;
        task.recv_count = recv_count;
        encoding_tasks_[column_idx].push(task);
        }
        encoding_tasks_cv_[column_idx].notify_one();
        
        // Update data buffer state (shared across columns)
        if (data_addr != 0 && size != 0) {
            std::lock_guard<std::mutex> lock(data_buffer_state_mutex_);
            auto &state = data_buffer_states_[data_addr];
            if (state.copies_expected == 0) {
                state.copies_expected = std::max(1, num_columns_);
                state.copies_completed = 0;
            }
        }
        
        // Register recv_addr to encoding/parity mapping for XOR task submission
        // Only register if recv is needed (recv_addr != 0)
        if (recv_addr != 0 && recv_chunk_size != 0 && encoding_addr != 0 && parity_addr != 0) {
            std::lock_guard<std::mutex> lock(recv_mapping_mutex_[column_idx]);
            RecvMapping mapping;
            mapping.encoding_addr = encoding_addr;
            mapping.parity_addr = parity_addr;
            mapping.size = recv_chunk_size;
            mapping.xor_mode = xor_mode;
            recv_to_xor_mappings_[column_idx][recv_addr] = mapping;
        }
    }
    
    // Legacy thread1/2 interfaces for backward compatibility (delegate to generic API)
    void submit_data_for_encoding_thread1(uintptr_t data_addr, size_t size,
                                          uintptr_t encoding_addr, uintptr_t recv_addr,
                                          size_t recv_chunk_size, uintptr_t parity_addr) {
        submit_data_for_encoding(0, data_addr, size, encoding_addr, recv_addr, recv_chunk_size, parity_addr);
    }
    
    void submit_data_for_encoding_thread2(uintptr_t data_addr, size_t size,
                                          uintptr_t encoding_addr, uintptr_t recv_addr,
                                          size_t recv_chunk_size, uintptr_t parity_addr) {
        submit_data_for_encoding(1, data_addr, size, encoding_addr, recv_addr, recv_chunk_size, parity_addr);
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
    
    // Post-XOR operations: send/recv data or parity after XOR completes
    void post_xor_send(int target_rank, uintptr_t data_addr, size_t size) {
        if (size == 0 || data_addr == 0) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] post_xor_send: Invalid parameters" << std::endl;
            return;
        }
        
        // Use column 0's NCCL communicator for post-xor operations
        int use_column = 0;
        
#ifdef NCCL_AVAILABLE
        if (nccl_initialized_[use_column] && world_size_ > 1) {
            // Wait for NCCL to be ready
            {
                std::unique_lock<std::mutex> lock(nccl_init_mutex_);
                nccl_init_cv_.wait(lock, [this, use_column] {
                    return nccl_init_completed_[use_column].load();
                });
            }
            
            ncclGroupStart();
            ncclSend(reinterpret_cast<void*>(data_addr), size, 
                     ncclUint8, target_rank, nccl_comms_[use_column], 0);
            ncclGroupEnd();
            
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Post-XOR send to rank " 
                      << target_rank << ", size: " << size << " bytes" << std::endl;
        } else {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Post-XOR send failed: NCCL not initialized" << std::endl;
        }
#else
        std::cerr << "EC-CHECK: [Rank " << rank_ << "] Post-XOR send: NCCL not available" << std::endl;
#endif
    }
    
    void post_xor_recv(int source_rank, uintptr_t recv_addr, size_t size) {
        if (size == 0 || recv_addr == 0) {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] post_xor_recv: Invalid parameters" << std::endl;
            return;
        }
        
        // Use column 0's NCCL communicator for post-xor operations
        int use_column = 0;
        
#ifdef NCCL_AVAILABLE
        if (nccl_initialized_[use_column] && world_size_ > 1) {
            // Wait for NCCL to be ready
            {
                std::unique_lock<std::mutex> lock(nccl_init_mutex_);
                nccl_init_cv_.wait(lock, [this, use_column] {
                    return nccl_init_completed_[use_column].load();
                });
            }
            
            ncclGroupStart();
            ncclRecv(reinterpret_cast<void*>(recv_addr), size, 
                     ncclUint8, source_rank, nccl_comms_[use_column], 0);
            ncclGroupEnd();
            
            std::cout << "EC-CHECK: [Rank " << rank_ << "] Post-XOR recv from rank " 
                      << source_rank << ", size: " << size << " bytes" << std::endl;
        } else {
            std::cerr << "EC-CHECK: [Rank " << rank_ << "] Post-XOR recv failed: NCCL not initialized" << std::endl;
        }
#else
        std::cerr << "EC-CHECK: [Rank " << rank_ << "] Post-XOR recv: NCCL not available" << std::endl;
#endif
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
             pybind11::arg("recv_chunk_size"), pybind11::arg("parity_addr"),
             pybind11::arg("zero_parity_addr") = 0,
             pybind11::arg("xor_mode_int") = 0,
             pybind11::arg("send_count") = 1,
             pybind11::arg("recv_count") = 1)
        .def("get_data_buffers_to_release", &ECCHECKNative::get_data_buffers_to_release)
        .def("get_encoding_buffers_to_release", &ECCHECKNative::get_encoding_buffers_to_release)
        .def("get_parity_buffers_to_release", &ECCHECKNative::get_parity_buffers_to_release)
        .def("post_xor_send", &ECCHECKNative::post_xor_send,
             pybind11::arg("target_rank"),
             pybind11::arg("data_addr"),
             pybind11::arg("size"))
        .def("post_xor_recv", &ECCHECKNative::post_xor_recv,
             pybind11::arg("source_rank"),
             pybind11::arg("recv_addr"),
             pybind11::arg("size"));
}
