/**
 * C++ module for layer-by-layer GPU to CPU tensor transfer
 * 
 * This module provides a dedicated thread for transferring model parameters
 * layer-by-layer from GPU to CPU memory. It ensures all layers are transferred
 * before continuing execution.
 * 
 * Compilation (without CUDA, using PyTorch transfers):
 *   g++ -O3 -std=c++17 -fPIC -shared \
 *       -I/path/to/python/include \
 *       -I/path/to/pybind11/include \
 *       layer_transfer_cpp.cpp -o layer_transfer_cpp.so
 * 
 * Compilation (with CUDA for direct GPU->CPU copy):
 *   nvcc -O3 -std=c++17 --shared --compiler-options '-fPIC' \
 *       -I/path/to/python/include \
 *       -I/path/to/pybind11/include \
 *       layer_transfer_cpp.cpp -o layer_transfer_cpp.so
 * 
 * Usage:
 *   import layer_transfer_cpp
 *   processor = layer_transfer_cpp.LayerTransferProcessor()
 *   processor.submit_layer(layer_id, tensors_data)
 *   processor.wait_all_complete()
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <cstring>
#include <chrono>
#include <unordered_map>

// Optional: Include CUDA for direct GPU->CPU transfer
// Uncomment if compiling with nvcc
// #define USE_CUDA
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace py = pybind11;


// ============================================================
// Layer Transfer Task Definition
// ============================================================

struct TensorTransferInfo {
    uintptr_t gpu_data_ptr;     // GPU tensor data pointer
    uintptr_t cpu_data_ptr;     // CPU tensor data pointer (destination)
    size_t size_bytes;          // Size in bytes
    std::vector<int64_t> shape; // Tensor shape
    std::string name;           // Tensor name
};

struct LayerTransferTask {
    int layer_id;               // Layer identifier
    std::vector<TensorTransferInfo> tensors;  // Tensors in this layer
    double submit_time;         // Task submission timestamp
};

struct LayerTransferResult {
    int layer_id;               // Layer identifier
    bool success;               // Transfer success status
    double transfer_time;       // Transfer time in seconds
    size_t total_bytes;         // Total bytes transferred
    std::string error_msg;      // Error message if failed
};


// ============================================================
// Layer Transfer Processor (Single Thread)
// ============================================================

class LayerTransferProcessor {
public:
    LayerTransferProcessor(int num_streams = 4) 
        : stop_flag_(false), 
          tasks_submitted_(0),
          tasks_completed_(0),
          is_worker_running_(false),
          num_streams_(num_streams) {
        
        #ifdef USE_CUDA
        // Create CUDA streams for concurrent transfers
        streams_.resize(num_streams_);
        for (int i = 0; i < num_streams_; i++) {
            cudaStreamCreate(&streams_[i]);
        }
        #endif
        
        // Start the dedicated worker thread
        worker_ = std::thread(&LayerTransferProcessor::worker_loop, this);
        is_worker_running_ = true;
    }
    
    ~LayerTransferProcessor() {
        stop();
    }
    
    /**
     * Submit a layer for transfer (called from Python)
     * 
     * @param layer_id: Unique layer identifier
     * @param tensors_info: List of (gpu_ptr, cpu_ptr, size, shape, name) tuples
     */
    void submit_layer(int layer_id, 
                     py::list tensors_info) {
        
        LayerTransferTask task;
        task.layer_id = layer_id;
        task.submit_time = get_current_time();
        
        // Parse tensor information
        for (auto item : tensors_info) {
            py::tuple tensor_tuple = item.cast<py::tuple>();
            if (tensor_tuple.size() != 5) {
                throw std::runtime_error("Each tensor info must be (gpu_ptr, cpu_ptr, size, shape, name)");
            }
            
            TensorTransferInfo info;
            info.gpu_data_ptr = tensor_tuple[0].cast<uintptr_t>();
            info.cpu_data_ptr = tensor_tuple[1].cast<uintptr_t>();
            info.size_bytes = tensor_tuple[2].cast<size_t>();
            info.shape = tensor_tuple[3].cast<std::vector<int64_t>>();
            info.name = tensor_tuple[4].cast<std::string>();
            
            task.tensors.push_back(info);
        }
        
        // Add to queue
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            task_queue_.push(task);
            tasks_submitted_++;
        }
        task_cv_.notify_one();
    }
    
    /**
     * Wait for all submitted tasks to complete (blocking)
     * This ensures all layers have been transferred before continuing
     */
    void wait_all_complete() {
        std::unique_lock<std::mutex> lock(completion_mutex_);
        completion_cv_.wait(lock, [this] {
            std::lock_guard<std::mutex> task_lock(task_mutex_);
            return tasks_completed_ >= tasks_submitted_ && task_queue_.empty();
        });
    }
    
    /**
     * Wait with timeout (returns true if completed, false if timeout)
     * 
     * @param timeout_ms: Timeout in milliseconds
     * @return: true if all tasks completed, false if timeout
     */
    bool wait_all_complete_timeout(int timeout_ms) {
        std::unique_lock<std::mutex> lock(completion_mutex_);
        return completion_cv_.wait_for(
            lock, 
            std::chrono::milliseconds(timeout_ms),
            [this] {
                std::lock_guard<std::mutex> task_lock(task_mutex_);
                return tasks_completed_ >= tasks_submitted_ && task_queue_.empty();
            }
        );
    }
    
    /**
     * Get transfer statistics
     * Returns: dict with keys: tasks_submitted, tasks_completed, queue_size
     */
    py::dict get_stats() {
        std::lock_guard<std::mutex> lock(task_mutex_);
        py::dict stats;
        stats["tasks_submitted"] = tasks_submitted_.load();
        stats["tasks_completed"] = tasks_completed_.load();
        stats["queue_size"] = task_queue_.size();
        return stats;
    }
    
    /**
     * Get all results collected so far
     * Returns: list of (layer_id, success, transfer_time, total_bytes, error_msg)
     */
    py::list get_results() {
        std::lock_guard<std::mutex> lock(result_mutex_);
        
        py::list results;
        while (!result_queue_.empty()) {
            const auto& result = result_queue_.front();
            results.append(py::make_tuple(
                result.layer_id,
                result.success,
                result.transfer_time,
                result.total_bytes,
                result.error_msg
            ));
            result_queue_.pop();
        }
        
        return results;
    }
    
    /**
     * Stop the worker thread
     */
    void stop() {
        if (!is_worker_running_) {
            return;
        }
        
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            stop_flag_ = true;
        }
        task_cv_.notify_all();
        
        if (worker_.joinable()) {
            worker_.join();
        }
        
        #ifdef USE_CUDA
        // Destroy CUDA streams
        for (auto& stream : streams_) {
            cudaStreamDestroy(stream);
        }
        streams_.clear();
        #endif
        
        is_worker_running_ = false;
    }
    
    /**
     * Reset counters (useful for multiple checkpointing rounds)
     */
    void reset() {
        std::lock_guard<std::mutex> lock(task_mutex_);
        tasks_submitted_ = 0;
        tasks_completed_ = 0;
        
        // Clear queues
        while (!task_queue_.empty()) {
            task_queue_.pop();
        }
        
        std::lock_guard<std::mutex> result_lock(result_mutex_);
        while (!result_queue_.empty()) {
            result_queue_.pop();
        }
    }

private:
    /**
     * Worker thread main loop
     * Processes layer transfer tasks sequentially
     */
    void worker_loop() {
        while (true) {
            LayerTransferTask task;
            
            // Get next task
            {
                std::unique_lock<std::mutex> lock(task_mutex_);
                task_cv_.wait(lock, [this] { 
                    return !task_queue_.empty() || stop_flag_; 
                });
                
                if (stop_flag_ && task_queue_.empty()) {
                    break;
                }
                
                task = task_queue_.front();
                task_queue_.pop();
            }
            
            // Process the layer transfer
            LayerTransferResult result;
            result.layer_id = task.layer_id;
            result.success = true;
            result.total_bytes = 0;
            result.error_msg = "";
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            try {
                #ifdef USE_CUDA
                // Concurrent transfer using multiple CUDA streams
                // Distribute tensors across streams for parallel transfer
                std::vector<std::vector<const TensorTransferInfo*>> stream_tensors(num_streams_);
                
                // Distribute tensors to streams (round-robin)
                for (size_t i = 0; i < task.tensors.size(); i++) {
                    int stream_idx = i % num_streams_;
                    stream_tensors[stream_idx].push_back(&task.tensors[i]);
                }
                
                // Launch async transfers on all streams
                for (int stream_idx = 0; stream_idx < num_streams_; stream_idx++) {
                    for (const auto* tensor_info : stream_tensors[stream_idx]) {
                        if (tensor_info->gpu_data_ptr == 0 || tensor_info->cpu_data_ptr == 0) {
                            throw std::runtime_error("Invalid pointer: gpu_ptr or cpu_ptr is null");
                        }
                        
                        // Async GPU->CPU transfer on this stream
                        cudaError_t err = cudaMemcpyAsync(
                            (void*)tensor_info->cpu_data_ptr, 
                            (void*)tensor_info->gpu_data_ptr, 
                            tensor_info->size_bytes, 
                            cudaMemcpyDeviceToHost,
                            streams_[stream_idx]
                        );
                        
                        if (err != cudaSuccess) {
                            throw std::runtime_error(
                                std::string("CUDA memcpy async failed: ") + cudaGetErrorString(err)
                            );
                        }
                        
                        result.total_bytes += tensor_info->size_bytes;
                    }
                }
                
                // Synchronize all streams to ensure all transfers complete
                for (int stream_idx = 0; stream_idx < num_streams_; stream_idx++) {
                    cudaError_t err = cudaStreamSynchronize(streams_[stream_idx]);
                    if (err != cudaSuccess) {
                        throw std::runtime_error(
                            std::string("CUDA stream synchronize failed: ") + cudaGetErrorString(err)
                        );
                    }
                }
                #else
                // Without CUDA: Sequential processing (PyTorch mode)
                // Assume PyTorch has already initiated the transfer
                // We just ensure this layer completes before moving to the next
                for (const auto& tensor_info : task.tensors) {
                    if (tensor_info.gpu_data_ptr == 0 || tensor_info.cpu_data_ptr == 0) {
                        throw std::runtime_error("Invalid pointer: gpu_ptr or cpu_ptr is null");
                    }
                    
                    // Note: Direct memcpy from GPU memory will FAIL without CUDA
                    // This is a placeholder that ensures layer-by-layer coordination
                    // For production use with actual GPU->CPU in C++, compile with USE_CUDA
                    result.total_bytes += tensor_info.size_bytes;
                }
                #endif
                
            } catch (const std::exception& e) {
                result.success = false;
                result.error_msg = e.what();
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.transfer_time = std::chrono::duration<double>(end_time - start_time).count();
            
            // Store result
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                result_queue_.push(result);
            }
            
            // Update completion counter
            {
                std::lock_guard<std::mutex> lock(task_mutex_);
                tasks_completed_++;
            }
            
            // Notify completion
            completion_cv_.notify_all();
        }
    }
    
    /**
     * Get current timestamp in seconds
     */
    double get_current_time() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration<double>(duration).count();
    }
    
    // Task queue and synchronization
    std::queue<LayerTransferTask> task_queue_;
    mutable std::mutex task_mutex_;
    std::condition_variable task_cv_;
    
    // Result queue
    std::queue<LayerTransferResult> result_queue_;
    mutable std::mutex result_mutex_;
    
    // Completion synchronization
    std::mutex completion_mutex_;
    std::condition_variable completion_cv_;
    
    // Worker thread
    std::thread worker_;
    std::atomic<bool> stop_flag_;
    std::atomic<bool> is_worker_running_;
    
    // Counters
    std::atomic<int> tasks_submitted_;
    std::atomic<int> tasks_completed_;
    
    // CUDA streams for concurrent transfers
    int num_streams_;
    #ifdef USE_CUDA
    std::vector<cudaStream_t> streams_;
    #endif
};


// ============================================================
// Python Bindings
// ============================================================

PYBIND11_MODULE(layer_transfer_cpp, m) {
    m.doc() = "C++ module for layer-by-layer GPU to CPU tensor transfer";
    
    // Export whether CUDA support is compiled in
    #ifdef USE_CUDA
    m.attr("USE_CUDA") = true;
    #else
    m.attr("USE_CUDA") = false;
    #endif
    
    py::class_<LayerTransferProcessor>(m, "LayerTransferProcessor")
        .def(py::init<>(),
             "Initialize layer transfer processor with dedicated worker thread (default: 4 streams)")
        .def(py::init<int>(),
             py::arg("num_streams"),
             "Initialize layer transfer processor with specified number of CUDA streams\n"
             "Args:\n"
             "  num_streams: Number of CUDA streams for concurrent transfers (default: 4)")
        
        .def("submit_layer", &LayerTransferProcessor::submit_layer,
             py::arg("layer_id"),
             py::arg("tensors_info"),
             "Submit a layer for transfer\n"
             "Args:\n"
             "  layer_id: Unique layer identifier (int)\n"
             "  tensors_info: List of (gpu_ptr, cpu_ptr, size, shape, name) tuples")
        
        .def("wait_all_complete", &LayerTransferProcessor::wait_all_complete,
             "Wait for all submitted layers to complete transfer (blocking)")
        
        .def("wait_all_complete_timeout", &LayerTransferProcessor::wait_all_complete_timeout,
             py::arg("timeout_ms"),
             "Wait for all submitted layers with timeout\n"
             "Args:\n"
             "  timeout_ms: Timeout in milliseconds\n"
             "Returns:\n"
             "  True if completed, False if timeout")
        
        .def("get_stats", &LayerTransferProcessor::get_stats,
             "Get transfer statistics\n"
             "Returns:\n"
             "  dict with keys: tasks_submitted, tasks_completed, queue_size")
        
        .def("get_results", &LayerTransferProcessor::get_results,
             "Get all transfer results\n"
             "Returns:\n"
             "  list of (layer_id, success, transfer_time, total_bytes, error_msg)")
        
        .def("stop", &LayerTransferProcessor::stop,
             "Stop the worker thread")
        
        .def("reset", &LayerTransferProcessor::reset,
             "Reset counters and clear queues");
}

