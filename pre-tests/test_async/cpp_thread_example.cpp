/**
 * C++ 线程 + CPUMemoryPool 示例
 * 
 * 展示如何使用 C++ 线程直接访问内存地址，无需序列化
 * 
 * 编译：
 *   g++ -O3 -std=c++17 -fPIC -shared \
 *       -I/path/to/python/include \
 *       -I/path/to/pybind11/include \
 *       cpp_thread_example.cpp -o cpp_thread_example.so
 * 
 * 使用：
 *   import cpp_thread_example
 *   processor = cpp_thread_example.AsyncTensorProcessor()
 *   processor.submit(address, size, shape)
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

namespace py = pybind11;


// ============================================================
// Tensor 任务定义
// ============================================================

struct TensorTask {
    uintptr_t address;      // 内存地址（从 CPUMemoryPool 分配）
    size_t size;            // 字节数
    std::vector<int64_t> shape;  // shape
    int tensor_id;          // tensor ID
};

struct TensorResult {
    int tensor_id;
    float sum;              // 计算结果
    double compute_time;    // 计算耗时（秒）
};


// ============================================================
// 异步 Tensor 处理器（C++ 线程版本）
// ============================================================

class AsyncTensorProcessor {
public:
    AsyncTensorProcessor(int num_threads = 2) 
        : stop_flag_(false) {
        
        // 启动工作线程
        for (int i = 0; i < num_threads; i++) {
            workers_.emplace_back(&AsyncTensorProcessor::worker_loop, this);
        }
    }
    
    ~AsyncTensorProcessor() {
        stop();
    }
    
    /**
     * 提交任务（Python 调用）
     * 
     * @param address: 内存地址（从 CPUMemoryPool.allocate() 返回）
     * @param size: 字节数
     * @param shape: Tensor shape
     * @param tensor_id: Tensor ID
     */
    void submit(uintptr_t address, size_t size, 
                std::vector<int64_t> shape, int tensor_id) {
        
        TensorTask task{address, size, shape, tensor_id};
        
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            task_queue_.push(task);
        }
        task_cv_.notify_one();
    }
    
    /**
     * 获取结果（Python 调用）
     * 
     * @param timeout_ms: 超时时间（毫秒）
     * @return: (tensor_id, sum, compute_time)
     */
    py::tuple get_result(int timeout_ms = 5000) {
        std::unique_lock<std::mutex> lock(result_mutex_);
        
        // 等待结果
        if (!result_cv_.wait_for(
                lock, 
                std::chrono::milliseconds(timeout_ms),
                [this]{ return !result_queue_.empty(); })) {
            throw std::runtime_error("Timeout waiting for result");
        }
        
        TensorResult result = result_queue_.front();
        result_queue_.pop();
        
        return py::make_tuple(result.tensor_id, result.sum, result.compute_time);
    }
    
    /**
     * 停止所有工作线程
     */
    void stop() {
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            stop_flag_ = true;
        }
        task_cv_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    /**
     * 获取队列大小
     */
    int get_task_queue_size() const {
        std::lock_guard<std::mutex> lock(task_mutex_);
        return task_queue_.size();
    }
    
    int get_result_queue_size() const {
        std::lock_guard<std::mutex> lock(result_mutex_);
        return result_queue_.size();
    }

private:
    /**
     * 工作线程主循环
     */
    void worker_loop() {
        while (true) {
            TensorTask task;
            
            // 获取任务
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
            
            // 处理任务（核心部分）
            auto start = std::chrono::high_resolution_clock::now();
            
            // ✅ 关键：直接访问内存地址（无需反序列化）
            float* data = reinterpret_cast<float*>(task.address);
            size_t num_elements = task.size / sizeof(float);
            
            // 计算 sum（模拟实际计算）
            float sum = 0.0f;
            // for (size_t i = 0; i < num_elements; i++) {
                // sum += data[i];
            // }
            
            auto end = std::chrono::high_resolution_clock::now();
            double compute_time = std::chrono::duration<double>(end - start).count();
            
            // 返回结果
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                result_queue_.push({task.tensor_id, sum, compute_time});
            }
            result_cv_.notify_one();
        }
    }
    
    // 任务队列
    std::queue<TensorTask> task_queue_;
    mutable std::mutex task_mutex_;
    std::condition_variable task_cv_;
    
    // 结果队列
    std::queue<TensorResult> result_queue_;
    mutable std::mutex result_mutex_;
    std::condition_variable result_cv_;
    
    // 工作线程
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_flag_;
};


// ============================================================
// 高级功能：XOR 计算示例
// ============================================================

class XORProcessor {
public:
    /**
     * 对两个 tensor 执行 XOR（用于 checkpoint）
     * 
     * @param addr1: 第一个 tensor 地址
     * @param addr2: 第二个 tensor 地址
     * @param result_addr: 结果地址
     * @param size: 字节数
     */
    static void xor_tensors(uintptr_t addr1, uintptr_t addr2, 
                           uintptr_t result_addr, size_t size) {
        
        // ✅ 直接访问内存（无需序列化）
        const uint8_t* data1 = reinterpret_cast<const uint8_t*>(addr1);
        const uint8_t* data2 = reinterpret_cast<const uint8_t*>(addr2);
        uint8_t* result = reinterpret_cast<uint8_t*>(result_addr);
        
        // 按字节 XOR
        for (size_t i = 0; i < size; i++) {
            result[i] = data1[i] ^ data2[i];
        }
    }
    
    /**
     * 批量 XOR（多线程）
     */
    static void xor_tensors_parallel(uintptr_t addr1, uintptr_t addr2,
                                     uintptr_t result_addr, size_t size,
                                     int num_threads = 4) {
        
        std::vector<std::thread> threads;
        size_t chunk_size = size / num_threads;
        
        for (int i = 0; i < num_threads; i++) {
            size_t start = i * chunk_size;
            size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
            
            threads.emplace_back([=]() {
                const uint8_t* data1 = reinterpret_cast<const uint8_t*>(addr1);
                const uint8_t* data2 = reinterpret_cast<const uint8_t*>(addr2);
                uint8_t* result = reinterpret_cast<uint8_t*>(result_addr);
                
                for (size_t j = start; j < end; j++) {
                    result[j] = data1[j] ^ data2[j];
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
};


// ============================================================
// Python 绑定
// ============================================================

PYBIND11_MODULE(cpp_thread_example, m) {
    m.doc() = "C++ 线程 + CPUMemoryPool 示例（无需序列化）";
    
    // AsyncTensorProcessor 类
    py::class_<AsyncTensorProcessor>(m, "AsyncTensorProcessor")
        .def(py::init<int>(), py::arg("num_threads") = 2,
             "初始化异步处理器\n"
             "Args:\n"
             "  num_threads: 工作线程数量")
        
        .def("submit", &AsyncTensorProcessor::submit,
             py::arg("address"),
             py::arg("size"),
             py::arg("shape"),
             py::arg("tensor_id"),
             "提交任务\n"
             "Args:\n"
             "  address: 内存地址（从 CPUMemoryPool.allocate() 返回）\n"
             "  size: 字节数\n"
             "  shape: Tensor shape (list of ints)\n"
             "  tensor_id: Tensor ID")
        
        .def("get_result", &AsyncTensorProcessor::get_result,
             py::arg("timeout_ms") = 5000,
             "获取结果\n"
             "Args:\n"
             "  timeout_ms: 超时时间（毫秒）\n"
             "Returns:\n"
             "  (tensor_id, sum, compute_time)")
        
        .def("stop", &AsyncTensorProcessor::stop,
             "停止所有工作线程")
        
        .def("get_task_queue_size", &AsyncTensorProcessor::get_task_queue_size,
             "获取任务队列大小")
        
        .def("get_result_queue_size", &AsyncTensorProcessor::get_result_queue_size,
             "获取结果队列大小");
    
    // XORProcessor 类
    py::class_<XORProcessor>(m, "XORProcessor")
        .def_static("xor_tensors", &XORProcessor::xor_tensors,
                   py::arg("addr1"),
                   py::arg("addr2"),
                   py::arg("result_addr"),
                   py::arg("size"),
                   "执行 XOR（单线程）\n"
                   "Args:\n"
                   "  addr1: 第一个 tensor 地址\n"
                   "  addr2: 第二个 tensor 地址\n"
                   "  result_addr: 结果地址\n"
                   "  size: 字节数")
        
        .def_static("xor_tensors_parallel", &XORProcessor::xor_tensors_parallel,
                   py::arg("addr1"),
                   py::arg("addr2"),
                   py::arg("result_addr"),
                   py::arg("size"),
                   py::arg("num_threads") = 4,
                   "执行 XOR（多线程）");
}

