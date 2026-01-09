#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>

// Simple barrier for thread synchronization
class Barrier {
public:
    explicit Barrier(size_t count) : threshold_(count), count_(count), generation_(0) {}
    
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        auto gen = generation_;
        if (--count_ == 0) {
            generation_++;
            count_ = threshold_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this, gen] { return gen != generation_; });
        }
    }
    
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t threshold_;
    size_t count_;
    size_t generation_;
};

// Statistics structure for tracking throughput
struct Statistics {
    std::atomic<uint64_t> bytes_transferred{0};
    std::atomic<uint64_t> operations_completed{0};
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    
    void reset() {
        bytes_transferred = 0;
        operations_completed = 0;
        start_time = std::chrono::steady_clock::now();
    }
    
    void finish() {
        end_time = std::chrono::steady_clock::now();
    }
    
    double get_throughput_gbps() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        if (duration == 0) return 0.0;
        
        double gbits = (bytes_transferred.load() * 8.0) / (1024.0 * 1024.0 * 1024.0);
        double seconds = duration / 1e6;
        return gbits / seconds;
    }
    
    double get_duration_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
    }
};

// RDMA connection info exchanged via TCP
struct RdmaConnInfo {
    uint32_t qp_num;     // Queue Pair number
    uint16_t lid;        // Local ID
    uint8_t gid[16];     // GID for RoCE
} __attribute__((packed));

// RDMA resources for one connection
struct RdmaResources {
    ibv_context* context = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* send_cq = nullptr;
    ibv_cq* recv_cq = nullptr;
    ibv_qp* qp = nullptr;
    ibv_mr* mr = nullptr;
    char* buffer = nullptr;
    size_t buffer_size = 0;
    int max_wr = 16;  // Max work requests
    
    ~RdmaResources() {
        if (mr) ibv_dereg_mr(mr);
        if (qp) ibv_destroy_qp(qp);
        if (send_cq) ibv_destroy_cq(send_cq);
        if (recv_cq) ibv_destroy_cq(recv_cq);
        if (pd) ibv_dealloc_pd(pd);
        if (context) ibv_close_device(context);
        if (buffer) free(buffer);  // Use free() for posix_memalign
    }
};

// Initialize RDMA device and resources
bool init_rdma_resources(RdmaResources& res, size_t buffer_size, int max_wr = 16) {
    // Get device list
    int num_devices;
    ibv_device** device_list = ibv_get_device_list(&num_devices);
    if (!device_list || num_devices == 0) {
        std::cerr << "Failed to get RDMA device list" << std::endl;
        return false;
    }
    
    // Open first device
    res.context = ibv_open_device(device_list[0]);
    ibv_free_device_list(device_list);
    
    if (!res.context) {
        std::cerr << "Failed to open RDMA device" << std::endl;
        return false;
    }
    
    // Allocate protection domain
    res.pd = ibv_alloc_pd(res.context);
    if (!res.pd) {
        std::cerr << "Failed to allocate protection domain" << std::endl;
        return false;
    }
    
    // Create completion queues
    res.send_cq = ibv_create_cq(res.context, max_wr, nullptr, nullptr, 0);
    if (!res.send_cq) {
        std::cerr << "Failed to create send completion queue" << std::endl;
        return false;
    }
    
    res.recv_cq = ibv_create_cq(res.context, max_wr, nullptr, nullptr, 0);
    if (!res.recv_cq) {
        std::cerr << "Failed to create recv completion queue" << std::endl;
        return false;
    }
    
    // Create Queue Pair
    res.max_wr = max_wr;
    ibv_qp_init_attr qp_init_attr = {};
    qp_init_attr.send_cq = res.send_cq;
    qp_init_attr.recv_cq = res.recv_cq;
    qp_init_attr.qp_type = IBV_QPT_RC;  // Reliable Connection
    qp_init_attr.cap.max_send_wr = max_wr;
    qp_init_attr.cap.max_recv_wr = max_wr;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    
    res.qp = ibv_create_qp(res.pd, &qp_init_attr);
    if (!res.qp) {
        std::cerr << "Failed to create queue pair" << std::endl;
        return false;
    }
    
    // Allocate and register buffer - use aligned allocation for better performance
    res.buffer_size = buffer_size;
    
    // Use posix_memalign for page-aligned memory (4KB alignment)
    int ret = posix_memalign(reinterpret_cast<void**>(&res.buffer), 4096, buffer_size);
    if (ret != 0) {
        std::cerr << "Failed to allocate aligned memory" << std::endl;
        return false;
    }
    std::memset(res.buffer, 0, buffer_size);
    
    res.mr = ibv_reg_mr(res.pd, res.buffer, buffer_size,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!res.mr) {
        std::cerr << "Failed to register memory region" << std::endl;
        return false;
    }
    
    // Transition QP to INIT state
    ibv_qp_attr qp_attr = {};
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = 1;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    
    if (ibv_modify_qp(res.qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        std::cerr << "Failed to modify QP to INIT" << std::endl;
        return false;
    }
    
    return true;
}

// Get local connection info
RdmaConnInfo get_local_conn_info(const RdmaResources& res) {
    RdmaConnInfo info;
    std::memset(&info, 0, sizeof(info));
    
    ibv_port_attr port_attr;
    ibv_query_port(res.context, 1, &port_attr);
    
    info.qp_num = res.qp->qp_num;
    info.lid = port_attr.lid;
    
    // Try to get GID for RoCE
    ibv_gid gid;
    if (ibv_query_gid(res.context, 1, 0, &gid) == 0) {
        std::memcpy(info.gid, &gid, 16);
    }
    
    return info;
}

// Connect QP to remote peer
bool connect_qp(RdmaResources& res, const RdmaConnInfo& remote_info) {
    // Transition to RTR (Ready to Receive)
    ibv_qp_attr qp_attr = {};
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;  // Use 4KB MTU for better performance
    qp_attr.dest_qp_num = remote_info.qp_num;
    qp_attr.rq_psn = 0;
    qp_attr.max_dest_rd_atomic = 1;
    qp_attr.min_rnr_timer = 12;
    
    // Check if we should use GID (RoCE) or LID (InfiniBand)
    bool use_gid = (remote_info.lid == 0);
    
    qp_attr.ah_attr.is_global = use_gid ? 1 : 0;
    qp_attr.ah_attr.dlid = remote_info.lid;
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = 1;
    
    if (use_gid) {
        // Use GID for RoCE
        std::memcpy(&qp_attr.ah_attr.grh.dgid, remote_info.gid, 16);
        qp_attr.ah_attr.grh.flow_label = 0;
        qp_attr.ah_attr.grh.sgid_index = 0;
        qp_attr.ah_attr.grh.hop_limit = 255;
        qp_attr.ah_attr.grh.traffic_class = 0;
    }
    
    int ret = ibv_modify_qp(res.qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                      IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    
    if (ret) {
        std::cerr << "Failed to modify QP to RTR, error code: " << ret 
                  << ", errno: " << errno << " (" << strerror(errno) << ")" << std::endl;
        std::cerr << "Remote QP: " << remote_info.qp_num 
                  << ", Remote LID: " << remote_info.lid 
                  << ", Using GID: " << (use_gid ? "yes" : "no") << std::endl;
        return false;
    }
    
    // Transition to RTS (Ready to Send)
    std::memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.max_rd_atomic = 1;
    
    if (ibv_modify_qp(res.qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                      IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
        std::cerr << "Failed to modify QP to RTS" << std::endl;
        return false;
    }
    
    return true;
}

// Exchange connection info via TCP
bool exchange_conn_info(int sock_fd, const RdmaConnInfo& local_info, RdmaConnInfo& remote_info) {
    // Send local info
    if (send(sock_fd, &local_info, sizeof(local_info), 0) != sizeof(local_info)) {
        std::cerr << "Failed to send connection info" << std::endl;
        return false;
    }
    
    // Receive remote info
    if (recv(sock_fd, &remote_info, sizeof(remote_info), MSG_WAITALL) != sizeof(remote_info)) {
        std::cerr << "Failed to receive connection info" << std::endl;
        return false;
    }
    
    return true;
}

// Post receive work request
bool post_recv(RdmaResources& res, uint64_t wr_id = 0) {
    ibv_sge sge = {};
    sge.addr = reinterpret_cast<uint64_t>(res.buffer);
    sge.length = res.buffer_size;
    sge.lkey = res.mr->lkey;
    
    ibv_recv_wr wr = {};
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    
    ibv_recv_wr* bad_wr;
    if (ibv_post_recv(res.qp, &wr, &bad_wr)) {
        std::cerr << "Failed to post recv" << std::endl;
        return false;
    }
    
    return true;
}

// Post send work request
bool post_send(RdmaResources& res, size_t length, uint64_t wr_id = 0) {
    ibv_sge sge = {};
    sge.addr = reinterpret_cast<uint64_t>(res.buffer);
    sge.length = length;
    sge.lkey = res.mr->lkey;
    
    ibv_send_wr wr = {};
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    
    ibv_send_wr* bad_wr;
    if (ibv_post_send(res.qp, &wr, &bad_wr)) {
        std::cerr << "Failed to post send" << std::endl;
        return false;
    }
    
    return true;
}

// Wait for work completion - optimized polling
bool poll_completion(ibv_cq* cq, int num_completions = 1) {
    int total_completed = 0;
    ibv_wc wc[16];  // Batch poll up to 16 completions
    
    while (total_completed < num_completions) {
        int n = ibv_poll_cq(cq, std::min(16, num_completions - total_completed), wc);
        if (n < 0) {
            std::cerr << "Failed to poll CQ" << std::endl;
            return false;
        }
        
        for (int i = 0; i < n; ++i) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                std::cerr << "Work completion with error: " << ibv_wc_status_str(wc[i].status) 
                         << " (opcode: " << wc[i].opcode << ")" << std::endl;
                return false;
            }
            total_completed++;
        }
        
        // If no completions, yield CPU briefly to reduce busy-wait overhead
        if (n == 0) {
            // Optional: std::this_thread::yield();
            // For lowest latency, keep spinning without yield
        }
    }
    return true;
}

// Server worker thread - two-sided RDMA
void server_worker(int client_sock, size_t buffer_size, size_t iterations) {
    try {
        std::cout << "Server thread started for client connection" << std::endl;
        
        // Initialize RDMA resources
        RdmaResources res;
        if (!init_rdma_resources(res, buffer_size)) {
            std::cerr << "Failed to initialize RDMA resources" << std::endl;
            return;
        }
        
        // Exchange connection info
        RdmaConnInfo local_info = get_local_conn_info(res);
        RdmaConnInfo remote_info;
        if (!exchange_conn_info(client_sock, local_info, remote_info)) {
            return;
        }
        
        // Connect QP
        if (!connect_qp(res, remote_info)) {
            return;
        }
        
        std::cout << "Server RDMA connection established" << std::endl;
        
        // Signal ready to client
        char ready = 1;
        send(client_sock, &ready, 1, 0);
        
        // Server loop: just receive data (like ib_send_lat server)
        for (size_t i = 0; i < iterations; ++i) {
            // Post receive before client sends
            if (!post_recv(res, i)) {
                std::cerr << "Failed to post recv" << std::endl;
                break;
            }
            
            // Wait for receive completion
            if (!poll_completion(res.recv_cq, 1)) {
                std::cerr << "Recv completion failed" << std::endl;
                break;
            }
        }
        
        std::cout << "Server thread finished, processed " << iterations << " messages" << std::endl;
        close(client_sock);
        
    } catch (std::exception& e) {
        std::cerr << "Server worker error: " << e.what() << std::endl;
    }
}

// Server function
void run_server(unsigned short port, size_t num_threads, size_t buffer_size, size_t iterations) {
    try {
        int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_sock < 0) {
            std::cerr << "Failed to create socket" << std::endl;
            return;
        }
        
        int opt = 1;
        setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        sockaddr_in server_addr = {};
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(port);
        
        if (bind(listen_sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Failed to bind socket" << std::endl;
            close(listen_sock);
            return;
        }
        
        if (listen(listen_sock, 10) < 0) {
            std::cerr << "Failed to listen" << std::endl;
            close(listen_sock);
            return;
        }
        
        std::cout << "RDMA Server (Two-Sided) listening on port " << port << std::endl;
        std::cout << "Max concurrent threads: " << num_threads << std::endl;
        std::cout << "Buffer size: " << buffer_size << " bytes ("
                 << (buffer_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
        std::cout << "Waiting for connections..." << std::endl;
        
        std::vector<std::thread> threads;
        
        while (true) {
            sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            int client_sock = accept(listen_sock, (sockaddr*)&client_addr, &addr_len);
            
            if (client_sock < 0) {
                std::cerr << "Failed to accept connection" << std::endl;
                continue;
            }
            
            char client_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
            std::cout << "Accepted connection from " << client_ip << std::endl;
            
            threads.emplace_back(server_worker, client_sock, buffer_size, iterations);
            
            // Clean up finished threads
            threads.erase(
                std::remove_if(threads.begin(), threads.end(),
                    [](std::thread& t) {
                        return !t.joinable();
                    }),
                threads.end()
            );
        }
        
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        
        close(listen_sock);
        
    } catch (std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
}

// Client worker thread - two-sided RDMA
void client_worker(const std::string& host, unsigned short port,
                   size_t data_size, size_t iterations,
                   Statistics& stats, bool is_warmup, Barrier* barrier = nullptr) {
    try {
        // Connect to server via TCP for control
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::cerr << "Failed to create socket" << std::endl;
            return;
        }
        
        hostent* server = gethostbyname(host.c_str());
        if (!server) {
            std::cerr << "Failed to resolve hostname" << std::endl;
            close(sock);
            return;
        }
        
        sockaddr_in server_addr = {};
        server_addr.sin_family = AF_INET;
        std::memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);
        server_addr.sin_port = htons(port);
        
        if (connect(sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Failed to connect to server" << std::endl;
            close(sock);
            return;
        }
        
        if (!is_warmup) {
            std::cout << "Client thread connected to server" << std::endl;
        }
        
        // Initialize RDMA resources
        RdmaResources res;
        if (!init_rdma_resources(res, data_size)) {
            std::cerr << "Failed to initialize RDMA resources" << std::endl;
            close(sock);
            return;
        }
        
        // Fill buffer with test data
        for (size_t i = 0; i < data_size; ++i) {
            res.buffer[i] = static_cast<char>(i % 256);
        }
        
        // Exchange connection info
        RdmaConnInfo local_info = get_local_conn_info(res);
        RdmaConnInfo remote_info;
        if (!exchange_conn_info(sock, local_info, remote_info)) {
            close(sock);
            return;
        }
        
        // Connect QP
        if (!connect_qp(res, remote_info)) {
            close(sock);
            return;
        }
        
        if (!is_warmup) {
            std::cout << "Client RDMA connection established" << std::endl;
        }
        
        // Wait for server ready signal
        char ready = 0;
        recv(sock, &ready, 1, MSG_WAITALL);
        
        // Synchronize all threads before starting data transfer
        if (barrier) {
            barrier->wait();
        }
        
        // Start timing just before data transfer
        auto start_time = std::chrono::steady_clock::now();
        
        // Client loop: only send data (like ib_send_lat)
        for (size_t i = 0; i < iterations; ++i) {
            // Post send
            if (!post_send(res, data_size, i)) {
                std::cerr << "Failed to post send" << std::endl;
                break;
            }
            
            // Wait for send completion
            if (!poll_completion(res.send_cq, 1)) {
                std::cerr << "Send completion failed" << std::endl;
                break;
            }
            
            stats.bytes_transferred += data_size;  // only send
            stats.operations_completed++;
        }
        
        // Stop timing immediately after data transfer
        auto end_time = std::chrono::steady_clock::now();
        
        // Print per-thread statistics
        if (!is_warmup) {
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count();
            double duration_ms = duration_us / 1000.0;
            double duration_sec = duration_us / 1e6;
            double avg_latency_us = duration_us / static_cast<double>(iterations);
            
            // Throughput calculation: (bytes * 8 bits) / (time_in_seconds * 1024^3)
            double total_bits = static_cast<double>(data_size) * iterations * 8.0;
            double throughput_gbps = total_bits / (duration_sec * 1024.0 * 1024.0 * 1024.0);
            
            std::cout << "Thread completed " << iterations 
                     << " iterations in " << std::fixed << std::setprecision(2)
                     << duration_ms << " ms ("
                     << duration_sec << " sec)" << std::endl;
            std::cout << "  Average latency: " << avg_latency_us << " us ("
                     << (avg_latency_us / 1000.0) << " ms)" << std::endl;
            std::cout << "  Throughput: " << throughput_gbps << " Gbps" << std::endl;
        }
        
        close(sock);
        
    } catch (std::exception& e) {
        std::cerr << "Client worker error: " << e.what() << std::endl;
    }
}

// Run client
void run_client(const std::string& host, unsigned short port,
                size_t num_threads, size_t data_size, size_t iterations,
                size_t warmup_iterations) {
    try {
        // Warmup phase
        if (warmup_iterations > 0) {
            std::cout << "\n=== Warmup Phase ===" << std::endl;
            std::cout << "Running " << warmup_iterations << " warmup iterations per thread..." << std::endl;
            
            Statistics warmup_stats;
            warmup_stats.reset();
            
            std::vector<std::thread> warmup_threads;
            for (size_t i = 0; i < num_threads; ++i) {
                warmup_threads.emplace_back(client_worker, host, port, data_size,
                                          warmup_iterations, std::ref(warmup_stats), true, nullptr);
            }
            
            for (auto& t : warmup_threads) {
                t.join();
            }
            
            warmup_stats.finish();
            std::cout << "Warmup completed in " << warmup_stats.get_duration_ms()
                     << " ms" << std::endl;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        // Actual test phase
        std::cout << "\n=== Test Phase ===" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Host: " << host << ":" << port << std::endl;
        std::cout << "  Threads: " << num_threads << std::endl;
        std::cout << "  Data size: " << data_size << " bytes ("
                 << (data_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
        std::cout << "  Iterations per thread: " << iterations << std::endl;
        std::cout << "  Total iterations: " << (iterations * num_threads) << std::endl;
        std::cout << "  Total data: " 
                 << (data_size * iterations * num_threads / (1024.0 * 1024.0 * 1024.0))
                 << " GB (unidirectional)" << std::endl;
        std::cout << "\nRunning RDMA test (Single-Direction SEND, like ib_send_lat)..." << std::endl;
        std::cout << "Establishing connections and preparing..." << std::endl;
        
        Statistics stats;
        Barrier barrier(num_threads);
        
        std::vector<std::thread> threads;
        
        // Reset timing - will be set by first thread to start
        stats.reset();
        
        // Start all threads (they will connect, then wait at barrier)
        for (size_t i = 0; i < num_threads; ++i) {
            threads.emplace_back(client_worker, host, port, data_size,
                              iterations, std::ref(stats), false, &barrier);
        }
        
        std::cout << "All threads synchronized, data transfer in progress..." << std::endl;
        
        // Wait for all threads to complete
        for (auto& t : threads) {
            t.join();
        }
        
        // Stop timing
        stats.finish();
        
        // Print results
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Duration: " << stats.get_duration_ms() << " ms ("
                 << (stats.get_duration_ms() / 1000.0) << " seconds)" << std::endl;
        std::cout << "Total bytes transferred: "
                 << stats.bytes_transferred.load() << " bytes ("
                 << (stats.bytes_transferred.load() / (1024.0 * 1024.0)) << " MB, "
                 << (stats.bytes_transferred.load() / (1024.0 * 1024.0 * 1024.0)) << " GB)"
                 << std::endl;
        std::cout << "Operations completed: " << stats.operations_completed.load()
                 << std::endl;
        std::cout << "Throughput: " << stats.get_throughput_gbps() << " Gbps" << std::endl;
        
        double ops_per_sec = (stats.operations_completed.load() * 1000.0) /
                            stats.get_duration_ms();
        std::cout << "Operations per second: " << std::setprecision(0) << ops_per_sec << std::endl;
        
        double avg_latency_ms = stats.get_duration_ms() / stats.operations_completed.load();
        std::cout << "Average latency per operation: " << std::setprecision(3) 
                 << avg_latency_ms << " ms" << std::endl;
        
    } catch (std::exception& e) {
        std::cerr << "Client error: " << e.what() << std::endl;
    }
}

void print_usage(const char* prog_name) {
    std::cout << "RDMA Latency Test (like ib_send_lat)\n" << std::endl;
    std::cout << "Usage: " << prog_name << " <mode> [options]\n" << std::endl;
    std::cout << "Modes:" << std::endl;
    std::cout << "  server  - Run as server (receives data)" << std::endl;
    std::cout << "  client  - Run as client (sends data)\n" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --host <hostname>      Server hostname or IP (client mode, default: 127.0.0.1)" << std::endl;
    std::cout << "  --port <port>          Port number (default: 12345)" << std::endl;
    std::cout << "  --threads <n>          Number of threads (default: 1)" << std::endl;
    std::cout << "  --size <bytes>         Data size in bytes (default: 65536)" << std::endl;
    std::cout << "  --size-mb <MB>         Data size in MB (overrides --size)" << std::endl;
    std::cout << "  --iterations <n>       Iterations per thread (client mode, default: 1000)" << std::endl;
    std::cout << "  --warmup <n>           Warmup iterations per thread (client mode, default: 100)" << std::endl;
    std::cout << "\nOperation Mode:" << std::endl;
    std::cout << "  - Uses RDMA SEND/RECV operations" << std::endl;
    std::cout << "  - Client sends data, server receives" << std::endl;
    std::cout << "  - Unidirectional data transfer (like ib_send_lat)" << std::endl;
    std::cout << "  - Measures send latency and throughput" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  Server: " << prog_name << " server --port 12345 --threads 4 --size-mb 10" << std::endl;
    std::cout << "  Client: " << prog_name << " client --host 192.168.1.100 --port 12345 --threads 4 --size-mb 10 --iterations 1000" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string mode = argv[1];
    std::string host = "127.0.0.1";
    unsigned short port = 12345;
    size_t num_threads = 1;
    size_t data_size = 65536;
    size_t iterations = 1000;
    size_t warmup_iterations = 100;
    
    // Parse arguments
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        
        std::string arg = argv[i];
        std::string value = argv[i + 1];
        
        if (arg == "--host") {
            host = value;
        } else if (arg == "--port") {
            port = static_cast<unsigned short>(std::stoi(value));
        } else if (arg == "--threads") {
            num_threads = std::stoull(value);
        } else if (arg == "--size") {
            data_size = std::stoull(value);
        } else if (arg == "--size-mb") {
            data_size = std::stoull(value) * 1024 * 1024;
            std::cout << "Using data size: " << value << " MB (" << data_size << " bytes)" << std::endl;
        } else if (arg == "--iterations") {
            iterations = std::stoull(value);
        } else if (arg == "--warmup") {
            warmup_iterations = std::stoull(value);
        }
    }
    
    if (mode == "server") {
        run_server(port, num_threads, data_size, iterations);
    } else if (mode == "client") {
        run_client(host, port, num_threads, data_size, iterations, warmup_iterations);
    } else {
        std::cerr << "Invalid mode. Use 'server' or 'client'" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    return 0;
}
