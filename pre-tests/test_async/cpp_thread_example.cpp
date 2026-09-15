/**
 * C++ Thread + CPUMemoryPool Example
 *
 * Shows how C++ threads access memory addresses directly without serialization
 *
 * build:
 *   g++ -O3 -std=c++17 -fPIC -shared \
 *       -I/path/to/python/include \
 *       -I/path/to/pybind11/include \
 *       cpp_thread_example.cpp -o cpp_thread_example.so
 *
 * use:
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
// Tensor Task Definition
// ============================================================

struct TensorTask {
    uintptr_t address;      // memory address (from CPUMemoryPool Allocate)
    size_t size;            // number of bytes
    std::vector<int64_t> shape;  // shape
    int tensor_id;          // tensor ID
};

struct TensorResult {
    int tensor_id;
    float sum;              // computation result
    double compute_time;    // computation time (seconds)
};


// ============================================================
// asynchronous Tensor processor (C++ threadversion)
// ============================================================

class AsyncTensorProcessor {
public:
    AsyncTensorProcessor(int num_threads = 2)
        : stop_flag_(false) {

        // Start worker thread
        for (int i = 0; i < num_threads; i++) {
            workers_.emplace_back(&AsyncTensorProcessor::worker_loop, this);
        }
    }

    ~AsyncTensorProcessor() {
        stop();
    }

    /**
     * Submit the task (Python call)
     *
     * @param address: memory address (from CPUMemoryPool.allocate() Return)
     * @param size: number of bytes
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
     * Get result (Python call)
     *
     * @param timeout_ms: timeout (milliseconds)
     * @return: (tensor_id, sum, compute_time)
     */
    py::tuple get_result(int timeout_ms = 5000) {
        std::unique_lock<std::mutex> lock(result_mutex_);

        // Wait for results
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
     * Stop all worker threads
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
     * Get queue size
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
     * Worker thread main loop
     */
    void worker_loop() {
        while (true) {
            TensorTask task;

            // Get a task
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

            // Process the task (core section)
            auto start = std::chrono::high_resolution_clock::now();

            // ✅ key: access memory addresses directly (no need to deserialize)
            float* data = reinterpret_cast<float*>(task.address);
            size_t num_elements = task.size / sizeof(float);

            // compute sum (simulate actual computation)
            float sum = 0.0f;
            // for (size_t i = 0; i < num_elements; i++) {
                // sum += data[i];
            // }

            auto end = std::chrono::high_resolution_clock::now();
            double compute_time = std::chrono::duration<double>(end - start).count();

            // Return results
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                result_queue_.push({task.tensor_id, sum, compute_time});
            }
            result_cv_.notify_one();
        }
    }

    // Task queue
    std::queue<TensorTask> task_queue_;
    mutable std::mutex task_mutex_;
    std::condition_variable task_cv_;

    // Result queue
    std::queue<TensorResult> result_queue_;
    mutable std::mutex result_mutex_;
    std::condition_variable result_cv_;

    // worker thread
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_flag_;
};


// ============================================================
// Advanced Features: XOR Computation Example
// ============================================================

class XORProcessor {
public:
    /**
     * Perform XOR on two tensors (for checkpointing)
     *
     * @param addr1: first tensor address
     * @param addr2: second tensor address
     * @param result_addr: result address
     * @param size: number of bytes
     */
    static void xor_tensors(uintptr_t addr1, uintptr_t addr2,
                           uintptr_t result_addr, size_t size) {

        // ✅ directly access memory (no serialization required)
        const uint8_t* data1 = reinterpret_cast<const uint8_t*>(addr1);
        const uint8_t* data2 = reinterpret_cast<const uint8_t*>(addr2);
        uint8_t* result = reinterpret_cast<uint8_t*>(result_addr);

        // Byte-wise XOR
        for (size_t i = 0; i < size; i++) {
            result[i] = data1[i] ^ data2[i];
        }
    }

    /**
     * batch XOR (multithreaded)
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
// Python Bindings
// ============================================================

PYBIND11_MODULE(cpp_thread_example, m) {
    m.doc() = "C++ Thread + CPUMemoryPool Example (no serialization required)";

    // AsyncTensorProcessor class
    py::class_<AsyncTensorProcessor>(m, "AsyncTensorProcessor")
        .def(py::init<int>(), py::arg("num_threads") = 2,
             "Initialize asynchronous processor\n"
             "Args:\n"
             "  num_threads: number of worker threads")

        .def("submit", &AsyncTensorProcessor::submit,
             py::arg("address"),
             py::arg("size"),
             py::arg("shape"),
             py::arg("tensor_id"),
             "Submit the task\n"
             "Args:\n"
             "  address: memory address (from CPUMemoryPool.allocate() Return)\n"
             "  size: number of bytes\n"
             "  shape: Tensor shape (list of ints)\n"
             "  tensor_id: Tensor ID")

        .def("get_result", &AsyncTensorProcessor::get_result,
             py::arg("timeout_ms") = 5000,
             "Get result\n"
             "Args:\n"
             "  timeout_ms: timeout (milliseconds)\n"
             "Returns:\n"
             "  (tensor_id, sum, compute_time)")

        .def("stop", &AsyncTensorProcessor::stop,
             "Stop all worker threads")

        .def("get_task_queue_size", &AsyncTensorProcessor::get_task_queue_size,
             "Get task queue size")

        .def("get_result_queue_size", &AsyncTensorProcessor::get_result_queue_size,
             "Get result queue size");

    // XORProcessor class
    py::class_<XORProcessor>(m, "XORProcessor")
        .def_static("xor_tensors", &XORProcessor::xor_tensors,
                   py::arg("addr1"),
                   py::arg("addr2"),
                   py::arg("result_addr"),
                   py::arg("size"),
                   "execute XOR (single-threaded)\n"
                   "Args:\n"
                   "  addr1: first tensor address\n"
                   "  addr2: second tensor address\n"
                   "  result_addr: result address\n"
                   "  size: number of bytes")

        .def_static("xor_tensors_parallel", &XORProcessor::xor_tensors_parallel,
                   py::arg("addr1"),
                   py::arg("addr2"),
                   py::arg("result_addr"),
                   py::arg("size"),
                   py::arg("num_threads") = 4,
                   "execute XOR (multithreaded)");
}

