// Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

/**
 * Gemini Replicas Native C++ Module with ASIO
 * 
 * Provides multi-replica data transfer using Boost.ASIO for network communication.
 * Supports configurable number of replicas with round-robin placement strategy.
 * 
 * Features:
 * - Multi-target broadcast (send to multiple ranks simultaneously)
 * - Asynchronous I/O for efficient network communication
 * - Zero-copy data transfer using raw memory pointers
 * 
 * Build:
 *   bash build_gemini_replicas.sh
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <boost/asio.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstring>
#include <cerrno>
#include <algorithm>
#include <deque>
#include <map>

// RDMA headers
#include <infiniband/verbs.h>

#include "rdma_device_utils.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace py = pybind11;

// Forward declaration for interface
class IGeminiReplicasConnectionManager {
public:
    virtual ~IGeminiReplicasConnectionManager() = default;
    
    virtual void initialize_connections() = 0;
    virtual void connect_and_wait() = 0;
    virtual void broadcast_to_targets(const uint8_t* data, size_t size) = 0;
    virtual std::tuple<int, size_t, std::unique_ptr<boost::asio::ip::tcp::socket>> peek_incoming_data() = 0;
    virtual void receive_data_into_buffer(std::unique_ptr<boost::asio::ip::tcp::socket> socket,
                                          uint8_t* buffer, size_t buffer_size, size_t expected_size) = 0;
    virtual std::pair<int, size_t> receive_data(uint8_t* buffer, size_t buffer_size) = 0;
    virtual bool is_connected() const = 0;
    
    // RDMA-specific methods (no-op for ASIO)
    virtual void register_buffer(uintptr_t addr, size_t size) {}
    virtual void unregister_buffer(uintptr_t addr) {}
    
    // Receive from a specific source rank (for RDMA to avoid unnecessary memcpy)
    virtual std::pair<int, size_t> receive_data_from_source(int source_rank, uint8_t* buffer, size_t buffer_size) {
        // Default implementation: just call receive_data and check source
        auto [actual_source, size] = receive_data(buffer, buffer_size);
        if (actual_source != source_rank) {
            throw std::runtime_error("Received data from wrong source");
        }
        return {actual_source, size};
    }
};

// RDMA-specific structures
struct RdmaConnInfo {
    uint32_t qp_num;
    uint16_t lid;
    uint8_t gid[16];
} __attribute__((packed));

struct RdmaBuffer {
    ibv_mr* mr;
    uintptr_t addr;
    size_t size;
};

/**
 * ASIO Connection Manager for Multi-Replica Communication
 * 
 * Manages TCP connections for broadcasting data to multiple target ranks
 * and receiving data from multiple source ranks.
 */
class GeminiReplicasAsioConnectionManager : public IGeminiReplicasConnectionManager {
private:
    boost::asio::io_context io_context_;
    
    // Send sockets for each target rank
    std::vector<std::unique_ptr<boost::asio::ip::tcp::socket>> send_sockets_;
    
    // Receive sockets (one for each incoming connection) and acceptor
    std::vector<std::unique_ptr<boost::asio::ip::tcp::socket>> recv_sockets_;
    std::unique_ptr<boost::asio::ip::tcp::acceptor> recv_acceptor_;
    
    // Connection status flags (use unique_ptr to avoid copy/move issues with atomic)
    std::vector<std::unique_ptr<std::atomic<bool>>> send_connected_;
    std::atomic<int> recv_connected_count_{0};
    int expected_recv_connections_;
    
    // Mutex for recv_sockets_ access
    std::mutex recv_sockets_mutex_;
    
    // Synchronization
    std::mutex connection_mutex_;
    std::condition_variable connection_cv_;
    
    // Rank information
    int rank_;
    int world_size_;
    std::vector<int> target_ranks_;
    
    // Network configuration
    std::vector<std::string> target_ips_;
    std::vector<int> target_ports_;
    std::string my_ip_;
    int my_port_;

public:
    GeminiReplicasAsioConnectionManager(
        int rank, int world_size,
        const std::vector<int>& target_ranks,
        const std::vector<std::string>& target_ips,
        const std::vector<int>& target_ports,
        const std::string& my_ip, int my_port,
        int expected_recv_connections
    )
        : rank_(rank),
          world_size_(world_size),
          target_ranks_(target_ranks),
          target_ips_(target_ips),
          target_ports_(target_ports),
          expected_recv_connections_(expected_recv_connections),
          my_ip_(my_ip),
          my_port_(my_port)
    {
        // Initialize send sockets and connection flags
        for (size_t i = 0; i < target_ranks_.size(); ++i) {
            send_sockets_.emplace_back(std::make_unique<boost::asio::ip::tcp::socket>(io_context_));
            send_connected_.emplace_back(std::make_unique<std::atomic<bool>>(false));
        }
        
        recv_acceptor_ = std::make_unique<boost::asio::ip::tcp::acceptor>(io_context_);
        
        std::cout << "[Rank " << rank_ << "] GeminiReplicasAsioConnectionManager created with " 
                  << target_ranks_.size() << " targets and expecting " 
                  << expected_recv_connections << " incoming connections" << std::endl;
    }
    
    ~GeminiReplicasAsioConnectionManager() {
        cleanup();
    }
    
    void initialize_connections() {
        // Phase 1: Start acceptor (non-blocking)
        start_acceptor();
    }
    
    void connect_and_wait() {
        // Phase 2: Connect to all targets and wait for all connections
        for (size_t i = 0; i < target_ranks_.size(); ++i) {
            connect_to_target(i);
        }
        wait_for_connections();
    }
    
    void broadcast_to_targets(const uint8_t* data, size_t size) {
        /**
         * Broadcast data to all target ranks simultaneously.
         * Uses asynchronous writes for better performance.
         * 
         * Protocol: [source_rank(4 bytes)][size(8 bytes)][data]
         */
        if (!is_connected()) {
            throw std::runtime_error("Not all connections are established");
        }
        
        std::vector<std::thread> send_threads;
        std::vector<std::exception_ptr> exceptions(target_ranks_.size());
        
        // Send to all targets in parallel
        for (size_t i = 0; i < target_ranks_.size(); ++i) {
            send_threads.emplace_back([this, i, data, size, &exceptions]() {
                try {
                    // Send source rank first (4 bytes)
                    int32_t source_rank = rank_;
                    boost::asio::write(*send_sockets_[i], 
                        boost::asio::buffer(&source_rank, sizeof(source_rank)));
                    
                    // Send size (8 bytes)
                    uint64_t size_network = size;
                    boost::asio::write(*send_sockets_[i], 
                        boost::asio::buffer(&size_network, sizeof(size_network)));
                    
                    // Send data
                    boost::asio::write(*send_sockets_[i], 
                        boost::asio::buffer(data, size));
                    
                    std::cout << "[Rank " << rank_ << "] Sent " << size 
                              << " bytes to target rank " << target_ranks_[i] << std::endl;
                } catch (...) {
                    exceptions[i] = std::current_exception();
                }
            });
        }
        
        // Wait for all sends to complete
        for (auto& t : send_threads) {
            t.join();
        }
        
        // Check for exceptions
        for (size_t i = 0; i < exceptions.size(); ++i) {
            if (exceptions[i]) {
                try {
                    std::rethrow_exception(exceptions[i]);
                } catch (const std::exception& e) {
                    throw std::runtime_error(
                        "Failed to send to target rank " + std::to_string(target_ranks_[i]) + 
                        ": " + e.what()
                    );
                }
            }
        }
    }
    
    std::tuple<int, size_t, std::unique_ptr<boost::asio::ip::tcp::socket>> peek_incoming_data() {
        /**
         * Peek the next incoming connection to get source_rank and data size.
         * Returns the socket so caller can read the actual data into appropriate buffer.
         * 
         * Protocol: [source_rank(4 bytes)][size(8 bytes)][data]
         * 
         * Returns: (source_rank, data_size, socket)
         */
        // Wait and get socket in one atomic operation
        std::unique_ptr<boost::asio::ip::tcp::socket> socket;
        
        // Busy-wait loop with small sleep to avoid deadlock
        bool got_socket = false;
        while (!got_socket) {
            {
                std::lock_guard<std::mutex> lock(recv_sockets_mutex_);
                if (!recv_sockets_.empty()) {
                    socket = std::move(recv_sockets_.front());
                    recv_sockets_.erase(recv_sockets_.begin());
                    got_socket = true;
                }
            }
            
            if (!got_socket) {
                // Wait a bit before retry
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        // Receive source rank first (4 bytes)
        int32_t source_rank;
        boost::asio::read(*socket, 
            boost::asio::buffer(&source_rank, sizeof(source_rank)));
        
        // Receive size (8 bytes)
        uint64_t size_network;
        boost::asio::read(*socket, 
            boost::asio::buffer(&size_network, sizeof(size_network)));
        
        size_t size = size_network;
        
        std::cout << "[Rank " << rank_ << "] Incoming data: " << size 
                  << " bytes from source rank " << source_rank << std::endl;
        
        // Return socket so caller can read the data
        return {source_rank, size, std::move(socket)};
    }
    
    void receive_data_into_buffer(std::unique_ptr<boost::asio::ip::tcp::socket> socket,
                                   uint8_t* buffer, size_t buffer_size, size_t expected_size) {
        /**
         * Receive data from an already-opened socket directly into target buffer.
         * Socket is returned to the pool for reuse (not closed).
         * 
         * Args:
         *   socket: The socket to read from (source_rank and size already read)
         *   buffer: Target buffer to receive data into
         *   buffer_size: Size of the target buffer
         *   expected_size: Expected data size (already read from socket header)
         */
        if (expected_size > buffer_size) {
            throw std::runtime_error(
                "Expected size (" + std::to_string(expected_size) + 
                ") exceeds buffer size (" + std::to_string(buffer_size) + ")"
            );
        }
        
        // Receive data directly into target buffer
        boost::asio::read(*socket, boost::asio::buffer(buffer, expected_size));
        
        // Return socket to the pool for reuse (DO NOT CLOSE)
        {
            std::lock_guard<std::mutex> lock(recv_sockets_mutex_);
            recv_sockets_.push_back(std::move(socket));
        }
    }
    
    std::pair<int, size_t> receive_data(uint8_t* buffer, size_t buffer_size) {
        /**
         * Receive data from one source rank (legacy method for compatibility).
         * This method gets the next available recv socket and reads from it.
         * For multiple sources, this should be called multiple times.
         * Socket is returned to the pool for reuse (not closed).
         * 
         * Protocol: [source_rank(4 bytes)][size(8 bytes)][data]
         * 
         * Returns: (source_rank, actual_size)
         */
        auto [source_rank, size, socket] = peek_incoming_data();
        
        if (size > buffer_size) {
            throw std::runtime_error(
                "Received size (" + std::to_string(size) + 
                ") exceeds buffer size (" + std::to_string(buffer_size) + ")"
            );
        }
        
        // Receive data
        boost::asio::read(*socket, 
            boost::asio::buffer(buffer, size));
        
        std::cout << "[Rank " << rank_ << "] Received " << size 
                  << " bytes from source rank " << source_rank << std::endl;
        
        // Return socket to the pool for reuse (DO NOT CLOSE)
        {
            std::lock_guard<std::mutex> lock(recv_sockets_mutex_);
            recv_sockets_.push_back(std::move(socket));
        }
        
        return {source_rank, size};
    }
    
    bool is_connected() const {
        if (recv_connected_count_ < expected_recv_connections_) {
            return false;
        }
        for (const auto& flag : send_connected_) {
            if (!(*flag)) {
                return false;
            }
        }
        return true;
    }

private:
    void start_acceptor() {
        try {
            boost::asio::ip::tcp::endpoint endpoint(
                boost::asio::ip::address::from_string(my_ip_),
                my_port_
            );
            
            recv_acceptor_->open(endpoint.protocol());
            recv_acceptor_->set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
            recv_acceptor_->bind(endpoint);
            recv_acceptor_->listen();
            
            std::cout << "[Rank " << rank_ << "] Acceptor started on " 
                      << my_ip_ << ":" << my_port_ 
                      << " (expecting " << expected_recv_connections_ << " connections)" << std::endl;
            
            // Start accepting multiple connections
            accept_next_connection();
            
            // Run io_context in a separate thread
            std::thread([this]() {
                io_context_.run();
            }).detach();
            
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed to start acceptor: " + std::string(e.what())
            );
        }
    }
    
    void accept_next_connection() {
        // Create a new socket for the next connection
        auto new_socket = std::make_unique<boost::asio::ip::tcp::socket>(io_context_);
        auto socket_ptr = new_socket.get();
        
        recv_acceptor_->async_accept(*socket_ptr,
            [this, socket_ptr, new_socket = std::move(new_socket)](const boost::system::error_code& ec) mutable {
                if (!ec) {
                    // Store the connected socket
                    {
                        std::lock_guard<std::mutex> lock(recv_sockets_mutex_);
                        recv_sockets_.push_back(std::move(new_socket));
                    }
                    
                    int count = ++recv_connected_count_;
                    std::cout << "[Rank " << rank_ << "] Accepted connection " << count 
                              << "/" << expected_recv_connections_ << std::endl;
                    
                    {
                        std::lock_guard<std::mutex> lock(connection_mutex_);
                        connection_cv_.notify_all();
                    }
                    
                    // Accept next connection if we haven't reached the expected count
                    if (count < expected_recv_connections_) {
                        accept_next_connection();
                    }
                } else {
                    std::cerr << "[Rank " << rank_ << "] Accept failed: " 
                              << ec.message() << std::endl;
                }
            });
    }
    
    void connect_to_target(size_t target_idx) {
        if (target_idx >= target_ranks_.size()) {
            throw std::runtime_error("Invalid target index");
        }
        
        try {
            boost::asio::ip::tcp::endpoint endpoint(
                boost::asio::ip::address::from_string(target_ips_[target_idx]),
                target_ports_[target_idx]
            );
            
            std::cout << "[Rank " << rank_ << "] Connecting to target rank " 
                      << target_ranks_[target_idx] << " at " 
                      << target_ips_[target_idx] << ":" << target_ports_[target_idx] << std::endl;
            
            // Retry logic for connection
            int max_retries = 10;
            for (int retry = 0; retry < max_retries; ++retry) {
                try {
                    send_sockets_[target_idx]->connect(endpoint);
                    
                    std::lock_guard<std::mutex> lock(connection_mutex_);
                    *send_connected_[target_idx] = true;
                    connection_cv_.notify_all();
                    
                    std::cout << "[Rank " << rank_ << "] Connected to target rank " 
                              << target_ranks_[target_idx] << std::endl;
                    return;
                } catch (const std::exception& e) {
                    if (retry < max_retries - 1) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    } else {
                        throw;
                    }
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed to connect to target rank " + std::to_string(target_ranks_[target_idx]) + 
                ": " + std::string(e.what())
            );
        }
    }
    
    void wait_for_connections() {
        std::unique_lock<std::mutex> lock(connection_mutex_);
        connection_cv_.wait(lock, [this]() {
            return is_connected();
        });
        
        std::cout << "[Rank " << rank_ << "] All connections established" << std::endl;
    }
    
    void cleanup() {
        try {
            // Close all send sockets
            for (auto& socket : send_sockets_) {
                if (socket && socket->is_open()) {
                    socket->close();
                }
            }
            
            // Close all receive sockets
            std::lock_guard<std::mutex> lock(recv_sockets_mutex_);
            for (auto& socket : recv_sockets_) {
                if (socket && socket->is_open()) {
                    socket->close();
                }
            }
            
            // Close acceptor
            if (recv_acceptor_ && recv_acceptor_->is_open()) {
                recv_acceptor_->close();
            }
            
            io_context_.stop();
        } catch (...) {
            // Ignore errors during cleanup
        }
    }
};

/**
 * RDMA Connection Manager for Multi-Replica Communication
 * 
 * Manages RDMA connections for broadcasting data to multiple target ranks
 * and receiving data from multiple source ranks using InfiniBand verbs.
 */
class GeminiReplicasRdmaConnectionManager : public IGeminiReplicasConnectionManager {
private:
    // RDMA resources
    ibv_context* context_;
    ibv_pd* pd_;
    ibv_cq* send_cq_;
    ibv_cq* recv_cq_;
    
    // Queue pairs for each target rank
    std::vector<ibv_qp*> send_qps_;
    
    // Queue pairs for receiving (one per expected source)
    std::vector<ibv_qp*> recv_qps_;
    std::vector<int> recv_source_ranks_;  // Track which rank each recv QP is for
    
    // TCP sockets for control messages (size exchange, ACKs)
    std::vector<int> control_socks_send_;  // One per target rank
    int listen_sock_;
    std::vector<int> control_socks_recv_;  // One per source rank
    
    // Connection status
    std::atomic<bool> connected_{false};
    
    // Synchronization
    std::mutex connection_mutex_;
    std::condition_variable connection_cv_;
    std::mutex buffer_mutex_;
    std::mutex recv_mutex_;
    
    // Rank information
    int rank_;
    int world_size_;
    std::vector<int> target_ranks_;
    int expected_recv_connections_;
    
    // Network configuration
    std::vector<std::string> target_ips_;
    std::vector<int> target_ports_;
    std::string my_ip_;
    int my_port_;
    
    // Registered buffers
    std::map<uintptr_t, RdmaBuffer> registered_buffers_;
    
    // Temporary work request buffers (for unregistered data)
    std::vector<uint8_t> temp_send_buffer_;
    std::vector<uint8_t> temp_recv_buffer_;
    ibv_mr* temp_send_mr_;
    ibv_mr* temp_recv_mr_;
    
    static const size_t TEMP_BUFFER_SIZE = 2ULL * 1024 * 1024 * 1024; // 2 GB
    static const int MAX_WR = 64;
    static const int MAX_SGE = 1;
    static const size_t CHUNK_SIZE = 64 * 1024 * 1024;  // 64 MB per RDMA operation
    static const int MAX_BATCH_WR = 32;

public:
    GeminiReplicasRdmaConnectionManager(
        int rank, int world_size,
        const std::vector<int>& target_ranks,
        const std::vector<std::string>& target_ips,
        const std::vector<int>& target_ports,
        const std::string& my_ip, int my_port,
        int expected_recv_connections
    )
        : context_(nullptr),
          pd_(nullptr),
          send_cq_(nullptr),
          recv_cq_(nullptr),
          listen_sock_(-1),
          rank_(rank),
          world_size_(world_size),
          target_ranks_(target_ranks),
          expected_recv_connections_(expected_recv_connections),
          target_ips_(target_ips),
          target_ports_(target_ports),
          my_ip_(my_ip),
          my_port_(my_port),
          temp_send_mr_(nullptr),
          temp_recv_mr_(nullptr)
    {
        std::cout << "[Rank " << rank_ << "] Creating GeminiReplicasRdmaConnectionManager with " 
                  << target_ranks_.size() << " targets and expecting " 
                  << expected_recv_connections_ << " sources (RDMA)" << std::endl;
        
        // Initialize send control sockets
        control_socks_send_.resize(target_ranks_.size(), -1);
        
        std::cout << "[Rank " << rank_ << "] GeminiReplicasRdmaConnectionManager created" << std::endl;
    }
    
    ~GeminiReplicasRdmaConnectionManager() {
        cleanup();
    }
    
    void initialize_connections() override {
        std::cout << "[Rank " << rank_ << "] Initializing RDMA connections..." << std::endl;
        
        // Initialize RDMA resources
        init_rdma_resources();
        
        // Start TCP listener for control messages
        start_tcp_listener();
        
        std::cout << "[Rank " << rank_ << "] RDMA initialization complete (Phase 1)" << std::endl;
    }
    
    void connect_and_wait() override {
        std::cout << "[Rank " << rank_ << "] Connecting to targets and waiting for connections..." << std::endl;
        
        try {
            // Accept control connections from source ranks
            std::exception_ptr accept_exception = nullptr;
            std::thread accept_thread([this, &accept_exception]() {
                try {
                    for (int i = 0; i < expected_recv_connections_; ++i) {
                        std::cout << "[Rank " << rank_ << "] Accepting connection " << (i+1) 
                                  << "/" << expected_recv_connections_ << "..." << std::endl;
                        accept_tcp_connection();
                    }
                } catch (...) {
                    accept_exception = std::current_exception();
                }
            });
            
            // Small delay to let receivers start accepting
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            // Connect to target ranks (control channel + RDMA QP)
            for (size_t i = 0; i < target_ranks_.size(); ++i) {
                std::cout << "[Rank " << rank_ << "] Connecting to target " << (i+1) 
                          << "/" << target_ranks_.size() << " (rank " << target_ranks_[i] << ")..." << std::endl;
                connect_to_target(i);
            }
            
            accept_thread.join();
            
            // Check if accept thread had an exception
            if (accept_exception) {
                std::rethrow_exception(accept_exception);
            }
            
            // Wait for all connections
            std::cout << "[Rank " << rank_ << "] Waiting for all connections to be ready..." << std::endl;
            wait_for_connections();
            
            // Warmup RDMA connections
            std::cout << "[Rank " << rank_ << "] Warming up RDMA connections..." << std::endl;
            warmup_rdma_connections();
            
            std::cout << "[Rank " << rank_ << "] All RDMA connections established and warmed up" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Rank " << rank_ << "] ERROR in connect_and_wait: " << e.what() << std::endl;
            throw;
        } catch (...) {
            std::cerr << "[Rank " << rank_ << "] ERROR in connect_and_wait: Unknown exception" << std::endl;
            throw;
        }
    }
    
    void register_buffer(uintptr_t addr, size_t size) override {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        if (registered_buffers_.find(addr) != registered_buffers_.end()) {
            std::cout << "[Rank " << rank_ << "] Buffer already registered at 0x" << std::hex << addr << std::dec << std::endl;
            return;
        }
        
        ibv_mr* mr = ibv_reg_mr(pd_, reinterpret_cast<void*>(addr), size,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr) {
            throw std::runtime_error("Failed to register buffer for RDMA");
        }
        
        registered_buffers_[addr] = {mr, addr, size};
        std::cout << "[Rank " << rank_ << "] Registered buffer at 0x" << std::hex << addr << std::dec 
                  << ", size: " << (size / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    }
    
    void unregister_buffer(uintptr_t addr) override {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        auto it = registered_buffers_.find(addr);
        if (it == registered_buffers_.end()) {
            return;
        }
        
        ibv_dereg_mr(it->second.mr);
        registered_buffers_.erase(it);
        std::cout << "[Rank " << rank_ << "] Unregistered buffer at 0x" << std::hex << addr << std::dec << std::endl;
    }

private:
    void send_data_chunked(const uint8_t* data, size_t total_size, ibv_mr* mr, ibv_qp* qp) {
        size_t chunk_count = (total_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        std::vector<ibv_sge> sges(chunk_count);
        std::vector<ibv_send_wr> wrs(chunk_count);
        
        for (size_t i = 0; i < chunk_count; ++i) {
            size_t offset = i * CHUNK_SIZE;
            size_t chunk_size = std::min(CHUNK_SIZE, total_size - offset);
            
            sges[i].addr = reinterpret_cast<uint64_t>(data + offset);
            sges[i].length = chunk_size;
            sges[i].lkey = mr->lkey;
            
            wrs[i].wr_id = i;
            wrs[i].sg_list = &sges[i];
            wrs[i].num_sge = 1;
            wrs[i].opcode = IBV_WR_SEND;
            wrs[i].send_flags = (i == chunk_count - 1) ? IBV_SEND_SIGNALED : 0;
            wrs[i].next = (i < chunk_count - 1) ? &wrs[i + 1] : nullptr;
        }
        
        // Post send work requests in batches
        for (size_t batch_start = 0; batch_start < chunk_count; batch_start += MAX_BATCH_WR) {
            size_t batch_end = std::min(batch_start + MAX_BATCH_WR, chunk_count);
            
            ibv_send_wr* bad_wr = nullptr;
            if (ibv_post_send(qp, &wrs[batch_start], &bad_wr) != 0) {
                throw std::runtime_error("Failed to post send work request");
            }
            
            // Poll completions for signaled requests
            for (size_t i = batch_start; i < batch_end; ++i) {
                if (wrs[i].send_flags & IBV_SEND_SIGNALED) {
                    poll_completion(send_cq_, 1);
                }
            }
        }
    }
    
    void receive_data_chunked(uint8_t* buffer, size_t total_size, ibv_mr* mr, ibv_qp* qp) {
        size_t chunk_count = (total_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        std::vector<ibv_sge> sges(chunk_count);
        std::vector<ibv_recv_wr> wrs(chunk_count);
        
        for (size_t i = 0; i < chunk_count; ++i) {
            size_t offset = i * CHUNK_SIZE;
            size_t chunk_size = std::min(CHUNK_SIZE, total_size - offset);
            
            sges[i].addr = reinterpret_cast<uint64_t>(buffer + offset);
            sges[i].length = chunk_size;
            sges[i].lkey = mr->lkey;
            
            wrs[i].wr_id = i;
            wrs[i].sg_list = &sges[i];
            wrs[i].num_sge = 1;
            wrs[i].next = (i < chunk_count - 1) ? &wrs[i + 1] : nullptr;
        }
        
        // Post receive work requests in batches
        for (size_t batch_start = 0; batch_start < chunk_count; batch_start += MAX_BATCH_WR) {
            ibv_recv_wr* bad_wr = nullptr;
            if (ibv_post_recv(qp, &wrs[batch_start], &bad_wr) != 0) {
                throw std::runtime_error("Failed to post receive work request");
            }
        }
        
        // Poll completions
        poll_completion(recv_cq_, chunk_count);
    }

public:
    void broadcast_to_targets(const uint8_t* data, size_t size) override {
        if (!connected_) {
            throw std::runtime_error("Not connected");
        }
        
        // Send size to all targets via control channel
        uint64_t size_net = htobe64(size);
        for (size_t i = 0; i < target_ranks_.size(); ++i) {
            if (send(control_socks_send_[i], &size_net, sizeof(size_net), 0) != sizeof(size_net)) {
                throw std::runtime_error("Failed to send size to target " + std::to_string(target_ranks_[i]));
            }
        }
        
        // Find or use temp buffer for MR
        ibv_mr* mr = find_registered_mr(reinterpret_cast<uintptr_t>(data), size);
        bool use_temp = (mr == nullptr);
        
        if (use_temp) {
            if (size > temp_send_buffer_.size()) {
                throw std::runtime_error("Data size exceeds temporary buffer size");
            }
            std::memcpy(temp_send_buffer_.data(), data, size);
            mr = temp_send_mr_;
        }
        
        const uint8_t* send_data = use_temp ? temp_send_buffer_.data() : data;
        
        // Send data to all targets via RDMA
        for (size_t i = 0; i < target_ranks_.size(); ++i) {
            send_data_chunked(send_data, size, mr, send_qps_[i]);
        }
        
        // Wait for ACKs from all targets
        for (size_t i = 0; i < target_ranks_.size(); ++i) {
            char ack;
            if (recv(control_socks_send_[i], &ack, 1, MSG_WAITALL) != 1 || ack != 'A') {
                throw std::runtime_error("Failed to receive ACK from target " + std::to_string(target_ranks_[i]));
            }
        }
    }
    
    std::tuple<int, size_t, std::unique_ptr<boost::asio::ip::tcp::socket>> peek_incoming_data() override {
        // Not used for RDMA (handled in receive_data)
        throw std::runtime_error("peek_incoming_data not supported for RDMA");
    }
    
    void receive_data_into_buffer(std::unique_ptr<boost::asio::ip::tcp::socket> socket,
                                   uint8_t* buffer, size_t buffer_size, size_t expected_size) override {
        // Not used for RDMA
        throw std::runtime_error("receive_data_into_buffer not supported for RDMA");
    }
    
    std::pair<int, size_t> receive_data(uint8_t* buffer, size_t buffer_size) override {
        std::lock_guard<std::mutex> lock(recv_mutex_);
        
        if (!connected_) {
            throw std::runtime_error("Not connected");
        }
        
        // Receive from any available source
        for (size_t i = 0; i < control_socks_recv_.size(); ++i) {
            fd_set read_fds;
            FD_ZERO(&read_fds);
            FD_SET(control_socks_recv_[i], &read_fds);
            
            struct timeval tv = {0, 1000}; // 1ms timeout
            int ret = select(control_socks_recv_[i] + 1, &read_fds, nullptr, nullptr, &tv);
            
            if (ret > 0) {
                return receive_data_from_qp(i, buffer, buffer_size);
            }
        }
        
        throw std::runtime_error("No data available from any source");
    }
    
    std::pair<int, size_t> receive_data_from_source(int source_rank, uint8_t* buffer, size_t buffer_size) override {
        std::lock_guard<std::mutex> lock(recv_mutex_);
        
        if (!connected_) {
            throw std::runtime_error("Not connected");
        }
        
        // Find the QP index for this source rank
        auto it = std::find(recv_source_ranks_.begin(), recv_source_ranks_.end(), source_rank);
        if (it == recv_source_ranks_.end()) {
            throw std::runtime_error("Source rank " + std::to_string(source_rank) + " not found");
        }
        
        size_t qp_idx = std::distance(recv_source_ranks_.begin(), it);
        
        // Wait for data from this specific source (with timeout)
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(control_socks_recv_[qp_idx], &read_fds);
        
        struct timeval tv = {30, 0}; // 30 second timeout
        int ret = select(control_socks_recv_[qp_idx] + 1, &read_fds, nullptr, nullptr, &tv);
        
        if (ret <= 0) {
            throw std::runtime_error("Timeout waiting for data from source rank " + std::to_string(source_rank));
        }
        
        return receive_data_from_qp(qp_idx, buffer, buffer_size);
    }

private:
    std::pair<int, size_t> receive_data_from_qp(size_t qp_idx, uint8_t* buffer, size_t buffer_size) {
        // Receive size
        uint64_t size_net;
        if (recv(control_socks_recv_[qp_idx], &size_net, sizeof(size_net), MSG_WAITALL) != sizeof(size_net)) {
            throw std::runtime_error("Failed to receive size from source");
        }
        size_t recv_size = be64toh(size_net);
        
        if (recv_size > buffer_size) {
            throw std::runtime_error("Received size exceeds buffer size");
        }
        
        // Find or use temp buffer for MR
        ibv_mr* mr = find_registered_mr(reinterpret_cast<uintptr_t>(buffer), recv_size);
        bool use_temp = (mr == nullptr);
        
        if (use_temp) {
            if (recv_size > temp_recv_buffer_.size()) {
                throw std::runtime_error("Receive size exceeds temporary buffer size");
            }
            mr = temp_recv_mr_;
        }
        
        uint8_t* recv_buffer = use_temp ? temp_recv_buffer_.data() : buffer;
        
        // Receive data via RDMA
        receive_data_chunked(recv_buffer, recv_size, mr, recv_qps_[qp_idx]);
        
        if (use_temp) {
            std::memcpy(buffer, temp_recv_buffer_.data(), recv_size);
        }
        
        // Send ACK
        char ack = 'A';
        if (send(control_socks_recv_[qp_idx], &ack, 1, 0) != 1) {
            throw std::runtime_error("Failed to send ACK");
        }
        
        return {recv_source_ranks_[qp_idx], recv_size};
    }

public:
    
    bool is_connected() const override {
        return connected_;
    }

private:
    void warmup_rdma_connections() {
        std::cout << "[Rank " << rank_ << "] Warming up RDMA connections..." << std::endl;
        
        const size_t warmup_size = 1024;
        std::vector<uint8_t> warmup_data(warmup_size, 0xAB);
        
        // Step 1: Post all receives first (to avoid deadlock)
        std::cout << "[Rank " << rank_ << "] Posting " << recv_qps_.size() << " warmup receives..." << std::endl;
        for (size_t i = 0; i < recv_qps_.size(); ++i) {
            try {
                // Post receive for warmup data
                ibv_sge sge;
                sge.addr = reinterpret_cast<uint64_t>(temp_recv_buffer_.data() + i * warmup_size);
                sge.length = warmup_size;
                sge.lkey = temp_recv_mr_->lkey;
                
                ibv_recv_wr wr{};
                wr.wr_id = i;
                wr.sg_list = &sge;
                wr.num_sge = 1;
                
                ibv_recv_wr* bad_wr = nullptr;
                if (ibv_post_recv(recv_qps_[i], &wr, &bad_wr) != 0) {
                    throw std::runtime_error("Failed to post warmup receive for source " + std::to_string(recv_source_ranks_[i]));
                }
                std::cout << "[Rank " << rank_ << "] Posted warmup receive for source " << recv_source_ranks_[i] << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[Rank " << rank_ << "] Failed to post warmup receive: " << e.what() << std::endl;
                throw;
            }
        }
        
        // Step 2: Send warmup data to all targets
        std::cout << "[Rank " << rank_ << "] Sending " << send_qps_.size() << " warmup messages..." << std::endl;
        for (size_t i = 0; i < send_qps_.size(); ++i) {
            try {
                std::memcpy(temp_send_buffer_.data() + i * warmup_size, warmup_data.data(), warmup_size);
                
                // Send warmup data
                ibv_sge sge;
                sge.addr = reinterpret_cast<uint64_t>(temp_send_buffer_.data() + i * warmup_size);
                sge.length = warmup_size;
                sge.lkey = temp_send_mr_->lkey;
                
                ibv_send_wr wr{};
                wr.wr_id = i;
                wr.sg_list = &sge;
                wr.num_sge = 1;
                wr.opcode = IBV_WR_SEND;
                wr.send_flags = IBV_SEND_SIGNALED;
                
                ibv_send_wr* bad_wr = nullptr;
                if (ibv_post_send(send_qps_[i], &wr, &bad_wr) != 0) {
                    throw std::runtime_error("Failed to post warmup send to target " + std::to_string(target_ranks_[i]));
                }
                std::cout << "[Rank " << rank_ << "] Posted warmup send to target " << target_ranks_[i] << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[Rank " << rank_ << "] Failed to post warmup send: " << e.what() << std::endl;
                throw;
            }
        }
        
        // Step 3: Wait for all sends to complete
        std::cout << "[Rank " << rank_ << "] Waiting for " << send_qps_.size() << " send completions..." << std::endl;
        for (size_t i = 0; i < send_qps_.size(); ++i) {
            poll_completion(send_cq_, 1);
            std::cout << "[Rank " << rank_ << "] Warmup send " << (i+1) << "/" << send_qps_.size() << " completed" << std::endl;
        }
        
        // Step 4: Wait for all receives to complete
        std::cout << "[Rank " << rank_ << "] Waiting for " << recv_qps_.size() << " receive completions..." << std::endl;
        for (size_t i = 0; i < recv_qps_.size(); ++i) {
            poll_completion(recv_cq_, 1);
            std::cout << "[Rank " << rank_ << "] Warmup receive " << (i+1) << "/" << recv_qps_.size() << " completed" << std::endl;
        }
        
        std::cout << "[Rank " << rank_ << "] RDMA warmup complete" << std::endl;
    }
    
    void init_rdma_resources() {
        // Get RDMA device
        int num_devices;
        ibv_device** device_list = ibv_get_device_list(&num_devices);
        if (!device_list || num_devices == 0) {
            throw std::runtime_error("No RDMA devices found");
        }
        
        context_ = ibv_open_device(find_rdma_device_by_ip(my_ip_, device_list, num_devices));
        if (!context_) {
            ibv_free_device_list(device_list);
            throw std::runtime_error("Failed to open RDMA device");
        }
        ibv_free_device_list(device_list);
        
        // Allocate protection domain
        pd_ = ibv_alloc_pd(context_);
        if (!pd_) {
            throw std::runtime_error("Failed to allocate protection domain");
        }
        
        // Create completion queues
        send_cq_ = ibv_create_cq(context_, MAX_WR * target_ranks_.size(), nullptr, nullptr, 0);
        recv_cq_ = ibv_create_cq(context_, MAX_WR * expected_recv_connections_, nullptr, nullptr, 0);
        if (!send_cq_ || !recv_cq_) {
            throw std::runtime_error("Failed to create completion queues");
        }
        
        // Create queue pairs for sending
        for (size_t i = 0; i < target_ranks_.size(); ++i) {
            ibv_qp_init_attr qp_init_attr{};
            qp_init_attr.send_cq = send_cq_;
            qp_init_attr.recv_cq = recv_cq_;
            qp_init_attr.qp_type = IBV_QPT_RC;
            qp_init_attr.cap.max_send_wr = MAX_WR;
            qp_init_attr.cap.max_recv_wr = MAX_WR;
            qp_init_attr.cap.max_send_sge = MAX_SGE;
            qp_init_attr.cap.max_recv_sge = MAX_SGE;
            
            ibv_qp* qp = ibv_create_qp(pd_, &qp_init_attr);
            if (!qp) {
                throw std::runtime_error("Failed to create send QP for target " + std::to_string(target_ranks_[i]));
            }
            send_qps_.push_back(qp);
        }
        
        // Create queue pairs for receiving (will be populated during connection)
        // recv_qps_ will be created on-demand during accept_tcp_connection
        
        // Allocate temporary buffers
        temp_send_buffer_.resize(TEMP_BUFFER_SIZE);
        temp_recv_buffer_.resize(TEMP_BUFFER_SIZE);
        
        temp_send_mr_ = ibv_reg_mr(pd_, temp_send_buffer_.data(), TEMP_BUFFER_SIZE,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        temp_recv_mr_ = ibv_reg_mr(pd_, temp_recv_buffer_.data(), TEMP_BUFFER_SIZE,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        
        if (!temp_send_mr_ || !temp_recv_mr_) {
            throw std::runtime_error("Failed to register temporary buffers");
        }
        
        std::cout << "[Rank " << rank_ << "] RDMA resources initialized" << std::endl;
    }
    
    void start_tcp_listener() {
        listen_sock_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_sock_ < 0) {
            throw std::runtime_error("Failed to create listen socket");
        }
        
        int opt = 1;
        setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(my_port_);
        inet_pton(AF_INET, my_ip_.c_str(), &addr.sin_addr);
        
        if (bind(listen_sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            throw std::runtime_error("Failed to bind listen socket");
        }
        
        if (listen(listen_sock_, expected_recv_connections_) < 0) {
            throw std::runtime_error("Failed to listen on socket");
        }
        
        std::cout << "[Rank " << rank_ << "] TCP listener started on " << my_ip_ << ":" << my_port_ << std::endl;
    }
    
    void accept_tcp_connection() {
        try {
            std::cout << "[Rank " << rank_ << "] Waiting to accept incoming connection..." << std::endl;
            int client_sock = accept(listen_sock_, nullptr, nullptr);
            if (client_sock < 0) {
                throw std::runtime_error("Failed to accept connection: " + std::string(strerror(errno)));
            }
            std::cout << "[Rank " << rank_ << "] Accepted TCP connection" << std::endl;
            
            // Exchange QP info
            // First, create a new QP for this incoming connection
            std::cout << "[Rank " << rank_ << "] Creating recv QP..." << std::endl;
            ibv_qp_init_attr qp_init_attr{};
            qp_init_attr.send_cq = send_cq_;
            qp_init_attr.recv_cq = recv_cq_;
            qp_init_attr.qp_type = IBV_QPT_RC;
            qp_init_attr.cap.max_send_wr = MAX_WR;
            qp_init_attr.cap.max_recv_wr = MAX_WR;
            qp_init_attr.cap.max_send_sge = MAX_SGE;
            qp_init_attr.cap.max_recv_sge = MAX_SGE;
            
            ibv_qp* qp = ibv_create_qp(pd_, &qp_init_attr);
            if (!qp) {
                close(client_sock);
                throw std::runtime_error("Failed to create recv QP: " + std::string(strerror(errno)));
            }
            
            std::cout << "[Rank " << rank_ << "] Exchanging QP info..." << std::endl;
            RdmaConnInfo local_info = get_local_conn_info(qp);
            RdmaConnInfo remote_info;
            
            // Exchange connection info with remote
            if (!exchange_conn_info(client_sock, local_info, remote_info)) {
                ibv_destroy_qp(qp);
                close(client_sock);
                throw std::runtime_error("Failed to exchange connection info");
            }
            
            // Connect QP
            std::cout << "[Rank " << rank_ << "] Connecting recv QP..." << std::endl;
            if (!connect_qp(qp, remote_info)) {
                ibv_destroy_qp(qp);
                close(client_sock);
                throw std::runtime_error("Failed to connect recv QP");
            }
            
            // Receive source rank from sender
            std::cout << "[Rank " << rank_ << "] Receiving source rank ID..." << std::endl;
            int32_t source_rank_net;
            if (recv(client_sock, &source_rank_net, sizeof(source_rank_net), MSG_WAITALL) != sizeof(source_rank_net)) {
                ibv_destroy_qp(qp);
                close(client_sock);
                throw std::runtime_error("Failed to receive source rank: " + std::string(strerror(errno)));
            }
            int source_rank = ntohl(source_rank_net);
            
            // Store the QP and source rank
            recv_qps_.push_back(qp);
            recv_source_ranks_.push_back(source_rank);
            control_socks_recv_.push_back(client_sock);
            
            std::cout << "[Rank " << rank_ << "] Successfully accepted RDMA connection from rank " << source_rank << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Rank " << rank_ << "] ERROR in accept_tcp_connection: " << e.what() << std::endl;
            throw;
        }
        
        // Check if all connections established
        std::cout << "[Rank " << rank_ << "] Connection status: recv_qps=" << recv_qps_.size() 
                  << "/" << expected_recv_connections_ << ", send_qps=" << send_qps_.size() 
                  << "/" << target_ranks_.size() << std::endl;
        
        if (recv_qps_.size() == static_cast<size_t>(expected_recv_connections_) &&
            send_qps_.size() == target_ranks_.size() &&
            std::all_of(control_socks_send_.begin(), control_socks_send_.end(), [](int s) { return s >= 0; })) {
            std::lock_guard<std::mutex> lock(connection_mutex_);
            connected_ = true;
            connection_cv_.notify_all();
            std::cout << "[Rank " << rank_ << "] All connections complete, notifying waiters" << std::endl;
        }
    }
    
    void connect_to_target(size_t target_idx) {
        try {
            std::cout << "[Rank " << rank_ << "] Creating socket for target " << target_ranks_[target_idx] << std::endl;
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) {
                throw std::runtime_error("Failed to create socket: " + std::string(strerror(errno)));
            }
            
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(target_ports_[target_idx]);
            if (inet_pton(AF_INET, target_ips_[target_idx].c_str(), &addr.sin_addr) <= 0) {
                close(sock);
                throw std::runtime_error("Invalid IP address: " + target_ips_[target_idx]);
            }
            
            // Retry connection
            std::cout << "[Rank " << rank_ << "] Connecting to " << target_ips_[target_idx] 
                      << ":" << target_ports_[target_idx] << std::endl;
            int max_retries = 20;
            for (int retry = 0; retry < max_retries; ++retry) {
                if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
                    std::cout << "[Rank " << rank_ << "] TCP connected to target " << target_ranks_[target_idx] << std::endl;
                    break;
                }
                if (retry == max_retries - 1) {
                    close(sock);
                    throw std::runtime_error("Failed to connect to target " + std::to_string(target_ranks_[target_idx]) + 
                                           " at " + target_ips_[target_idx] + ":" + std::to_string(target_ports_[target_idx]) +
                                           " after " + std::to_string(max_retries) + " retries: " + std::string(strerror(errno)));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            
            // Exchange QP info
            std::cout << "[Rank " << rank_ << "] Exchanging QP info with target " << target_ranks_[target_idx] << std::endl;
            RdmaConnInfo local_info = get_local_conn_info(send_qps_[target_idx]);
            RdmaConnInfo remote_info;
            
            if (!exchange_conn_info(sock, local_info, remote_info)) {
                close(sock);
                throw std::runtime_error("Failed to exchange connection info with target " + std::to_string(target_ranks_[target_idx]));
            }
            
            // Connect QP
            std::cout << "[Rank " << rank_ << "] Connecting QP to target " << target_ranks_[target_idx] << std::endl;
            if (!connect_qp(send_qps_[target_idx], remote_info)) {
                close(sock);
                throw std::runtime_error("Failed to connect QP to target " + std::to_string(target_ranks_[target_idx]));
            }
            
            // Send my rank to receiver
            std::cout << "[Rank " << rank_ << "] Sending rank ID to target " << target_ranks_[target_idx] << std::endl;
            int32_t my_rank_net = htonl(rank_);
            if (send(sock, &my_rank_net, sizeof(my_rank_net), 0) != sizeof(my_rank_net)) {
                close(sock);
                throw std::runtime_error("Failed to send rank to target " + std::to_string(target_ranks_[target_idx]));
            }
            
            control_socks_send_[target_idx] = sock;
            
            std::cout << "[Rank " << rank_ << "] Successfully connected to target " << target_ranks_[target_idx] << std::endl;
            
            // Check if all connections established
            std::cout << "[Rank " << rank_ << "] Connection status: recv_qps=" << recv_qps_.size() 
                      << "/" << expected_recv_connections_ << ", send_qps=" << send_qps_.size() 
                      << "/" << target_ranks_.size() << ", control_socks_send=" 
                      << std::count_if(control_socks_send_.begin(), control_socks_send_.end(), [](int s) { return s >= 0; })
                      << "/" << target_ranks_.size() << std::endl;
            
            if (recv_qps_.size() == static_cast<size_t>(expected_recv_connections_) &&
                send_qps_.size() == target_ranks_.size() &&
                std::all_of(control_socks_send_.begin(), control_socks_send_.end(), [](int s) { return s >= 0; })) {
                std::lock_guard<std::mutex> lock(connection_mutex_);
                connected_ = true;
                connection_cv_.notify_all();
                std::cout << "[Rank " << rank_ << "] All connections complete (from connect_to_target), notifying waiters" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Rank " << rank_ << "] ERROR connecting to target " << target_ranks_[target_idx] << ": " << e.what() << std::endl;
            throw;
        }
    }
    
    RdmaConnInfo get_local_conn_info(ibv_qp* qp) {
        ibv_port_attr port_attr;
        if (ibv_query_port(context_, 1, &port_attr) != 0) {
            throw std::runtime_error("Failed to query port");
        }
        
        RdmaConnInfo info;
        info.qp_num = qp->qp_num;
        info.lid = port_attr.lid;
        
        ibv_gid gid;
        if (ibv_query_gid(context_, 1, 1, &gid) == 0) {
            std::memcpy(info.gid, &gid, 16);
        } else {
            std::memset(info.gid, 0, 16);
        }
        
        return info;
    }
    
    bool exchange_conn_info(int sock_fd, const RdmaConnInfo& local_info, RdmaConnInfo& remote_info) {
        if (send(sock_fd, &local_info, sizeof(local_info), 0) != sizeof(local_info)) {
            return false;
        }
        if (recv(sock_fd, &remote_info, sizeof(remote_info), MSG_WAITALL) != sizeof(remote_info)) {
            return false;
        }
        return true;
    }
    
    bool connect_qp(ibv_qp* qp, const RdmaConnInfo& remote_info) {
        // Transition to INIT
        std::cout << "[Rank " << rank_ << "] QP transition: RESET -> INIT" << std::endl;
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = 1;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        
        int ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (ret != 0) {
            std::cerr << "[Rank " << rank_ << "] Failed to transition QP to INIT: " << strerror(errno) 
                      << " (ret=" << ret << ")" << std::endl;
            return false;
        }
        
        // Transition to RTR
        std::cout << "[Rank " << rank_ << "] QP transition: INIT -> RTR (remote QP=" << remote_info.qp_num 
                  << ", LID=" << remote_info.lid << ")" << std::endl;
        std::memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_4096;
        attr.dest_qp_num = remote_info.qp_num;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        
        // Check if we should use GID (RoCE) or LID (InfiniBand)
        bool use_gid = false;
        for (int i = 0; i < 16; ++i) {
            if (remote_info.gid[i] != 0) {
                use_gid = true;
                break;
            }
        }
        
        if (use_gid) {
            std::cout << "[Rank " << rank_ << "] Using GID (RoCE mode)" << std::endl;
            attr.ah_attr.is_global = 1;
            attr.ah_attr.grh.dgid = *reinterpret_cast<const ibv_gid*>(remote_info.gid);
            attr.ah_attr.grh.flow_label = 0;
            attr.ah_attr.grh.sgid_index = 1; // GID index 1 for erdma (RoCE v2)
            attr.ah_attr.grh.hop_limit = 255;
            attr.ah_attr.grh.traffic_class = 0;
        } else {
            std::cout << "[Rank " << rank_ << "] Using LID (InfiniBand mode)" << std::endl;
            attr.ah_attr.is_global = 0;
            attr.ah_attr.dlid = remote_info.lid;
        }
        
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = 1;
        
        ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (ret != 0) {
            std::cerr << "[Rank " << rank_ << "] Failed to transition QP to RTR: " << strerror(errno) 
                      << " (ret=" << ret << ")" << std::endl;
            return false;
        }
        
        // Transition to RTS
        std::cout << "[Rank " << rank_ << "] QP transition: RTR -> RTS" << std::endl;
        std::memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        attr.sq_psn = 0;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.max_rd_atomic = 1;
        
        ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                          IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
        if (ret != 0) {
            std::cerr << "[Rank " << rank_ << "] Failed to transition QP to RTS: " << strerror(errno) 
                      << " (ret=" << ret << ")" << std::endl;
            return false;
        }
        
        std::cout << "[Rank " << rank_ << "] QP successfully connected (RESET -> INIT -> RTR -> RTS)" << std::endl;
        return true;
    }
    
    ibv_mr* find_registered_mr(uintptr_t addr, size_t size) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        for (auto& [buf_addr, buf] : registered_buffers_) {
            if (addr >= buf_addr && (addr + size) <= (buf_addr + buf.size)) {
                return buf.mr;
            }
        }
        return nullptr;
    }
    
    void poll_completion(ibv_cq* cq, int num_completions) {
        int polled = 0;
        while (polled < num_completions) {
            ibv_wc wc;
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) {
                throw std::runtime_error("Failed to poll completion queue");
            }
            if (n > 0) {
                if (wc.status != IBV_WC_SUCCESS) {
                    throw std::runtime_error("Work completion failed with status " + std::to_string(wc.status));
                }
                polled++;
            }
        }
    }
    
    void wait_for_connections() {
        std::unique_lock<std::mutex> lock(connection_mutex_);
        connection_cv_.wait(lock, [this]() {
            return connected_.load();
        });
        
        std::cout << "[Rank " << rank_ << "] All RDMA connections established" << std::endl;
    }
    
    void cleanup() {
        try {
            // Unregister all buffers
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                for (auto& [addr, buf] : registered_buffers_) {
                    ibv_dereg_mr(buf.mr);
                }
                registered_buffers_.clear();
            }
            
            // Cleanup temporary buffers
            if (temp_send_mr_) {
                ibv_dereg_mr(temp_send_mr_);
                temp_send_mr_ = nullptr;
            }
            if (temp_recv_mr_) {
                ibv_dereg_mr(temp_recv_mr_);
                temp_recv_mr_ = nullptr;
            }
            
            // Destroy queue pairs
            for (auto qp : send_qps_) {
                if (qp) ibv_destroy_qp(qp);
            }
            for (auto qp : recv_qps_) {
                if (qp) ibv_destroy_qp(qp);
            }
            send_qps_.clear();
            recv_qps_.clear();
            
            // Destroy completion queues
            if (send_cq_) {
                ibv_destroy_cq(send_cq_);
                send_cq_ = nullptr;
            }
            if (recv_cq_) {
                ibv_destroy_cq(recv_cq_);
                recv_cq_ = nullptr;
            }
            
            // Deallocate protection domain
            if (pd_) {
                ibv_dealloc_pd(pd_);
                pd_ = nullptr;
            }
            
            // Close device
            if (context_) {
                ibv_close_device(context_);
                context_ = nullptr;
            }
            
            // Close control sockets
            for (int sock : control_socks_send_) {
                if (sock >= 0) close(sock);
            }
            for (int sock : control_socks_recv_) {
                if (sock >= 0) close(sock);
            }
            if (listen_sock_ >= 0) {
                close(listen_sock_);
                listen_sock_ = -1;
            }
            
            control_socks_send_.clear();
            control_socks_recv_.clear();
        } catch (...) {
            // Ignore errors during cleanup
        }
    }
};

/**
 * Main Gemini Replicas Native Class
 * 
 * Provides Python interface for multi-replica data transfer.
 */
class GeminiReplicasNative {
private:
    int rank_;
    int world_size_;
    std::vector<int> target_ranks_;
    bool use_rdma_;
    
    std::unique_ptr<IGeminiReplicasConnectionManager> connection_manager_;
    
    std::atomic<bool> initialized_{false};

    // ---- Worker-thread model (aligned with ecnaive) ----
    // Persistent worker threads: one send_worker + one recv_worker per source.
    // Main thread only submits tasks and polls atomic flags — never blocks on I/O.

    bool workers_started_{false};
    std::atomic<bool> stop_workers_{false};

    // Send worker
    std::thread send_worker_thread_;
    std::mutex send_mutex_;
    std::condition_variable send_cv_;
    uintptr_t send_task_addr_{0};
    size_t send_task_size_{0};
    std::atomic<bool> send_ready_{false};
    std::atomic<bool> send_done_{false};
    std::atomic<bool> send_error_{false};
    std::string send_error_msg_;
    std::mutex send_error_mutex_;

    // Recv workers (one per source rank)
    std::vector<int> recv_source_ranks_;
    std::vector<std::thread> recv_worker_threads_;
    std::vector<std::unique_ptr<std::mutex>> recv_mutexes_;
    std::vector<std::unique_ptr<std::condition_variable>> recv_cvs_;
    std::vector<uintptr_t> recv_task_addrs_;
    std::vector<size_t> recv_task_sizes_;
    std::deque<std::atomic<bool>> recv_ready_;
    std::deque<std::atomic<bool>> recv_done_;
    std::deque<std::atomic<bool>> recv_error_;
    std::vector<std::string> recv_error_msgs_;
    std::mutex recv_error_mutex_;

public:
    GeminiReplicasNative(
        int rank, int world_size,
        const std::vector<int>& target_ranks,
        const std::vector<std::string>& target_ips,
        const std::vector<int>& target_ports,
        const std::string& my_ip, int my_port,
        int num_source_ranks,
        bool use_rdma = false
    )
        : rank_(rank),
          world_size_(world_size),
          target_ranks_(target_ranks),
          use_rdma_(use_rdma)
    {
        std::cout << "[Rank " << rank_ << "] Creating GeminiReplicasNative with " 
                  << target_ranks_.size() << " targets and " 
                  << num_source_ranks << " sources (mode: " 
                  << (use_rdma_ ? "RDMA" : "ASIO") << ")" << std::endl;
        
        // Phase 1: Create connection manager and start acceptor
        if (use_rdma_) {
            connection_manager_ = std::make_unique<GeminiReplicasRdmaConnectionManager>(
                rank_, world_size_,
                target_ranks, target_ips, target_ports,
                my_ip, my_port,
                num_source_ranks  // Pass number of expected incoming connections
            );
        } else {
            connection_manager_ = std::make_unique<GeminiReplicasAsioConnectionManager>(
                rank_, world_size_,
                target_ranks, target_ips, target_ports,
                my_ip, my_port,
                num_source_ranks  // Pass number of expected incoming connections
            );
        }
        
        connection_manager_->initialize_connections();
        
        std::cout << "[Rank " << rank_ << "] GeminiReplicasNative created (Phase 1 complete)" << std::endl;
    }
    
    ~GeminiReplicasNative() {
        std::cout << "[Rank " << rank_ << "] Destroying GeminiReplicasNative" << std::endl;
        stop_workers();
    }
    
    void finalize_connections() {
        /**
         * Phase 2: Connect to all targets and wait for all connections.
         * Should be called after all ranks have started their acceptors.
         */
        std::cout << "[Rank " << rank_ << "] Finalizing connections (Phase 2)..." << std::endl;
        
        connection_manager_->connect_and_wait();
        
        initialized_ = true;
        
        std::cout << "[Rank " << rank_ << "] All connections finalized" << std::endl;
    }
    
    void broadcast_to_targets(uintptr_t buffer_addr, size_t buffer_size) {
        /**
         * Broadcast data to all target ranks.
         * 
         * Args:
         *   buffer_addr: Memory address of the buffer to send
         *   buffer_size: Size of the buffer in bytes
         */
        if (!initialized_) {
            throw std::runtime_error("GeminiReplicasNative not initialized");
        }
        
        const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer_addr);
        
        std::cout << "[Rank " << rank_ << "] Broadcasting " << buffer_size 
                  << " bytes to " << target_ranks_.size() << " targets" << std::endl;
        
        connection_manager_->broadcast_to_targets(data, buffer_size);
        
        std::cout << "[Rank " << rank_ << "] Broadcast completed" << std::endl;
    }
    
    std::pair<int, size_t> receive_from_source(uintptr_t buffer_addr, size_t buffer_size) {
        /**
         * Receive data from one source rank.
         * 
         * Args:
         *   buffer_addr: Memory address of the buffer to receive into
         *   buffer_size: Size of the buffer in bytes
         * 
         * Returns:
         *   (source_rank, actual_size)
         */
        if (!initialized_) {
            throw std::runtime_error("GeminiReplicasNative not initialized");
        }
        
        uint8_t* buffer = reinterpret_cast<uint8_t*>(buffer_addr);
        
        std::cout << "[Rank " << rank_ << "] Receiving data (buffer size: " 
                  << buffer_size << " bytes)" << std::endl;
        
        auto [source_rank, received_size] = connection_manager_->receive_data(buffer, buffer_size);
        
        std::cout << "[Rank " << rank_ << "] Received " << received_size 
                  << " bytes from source rank " << source_rank << std::endl;
        
        return {source_rank, received_size};
    }
    
    // ==================== Worker-thread model ====================
    // Persistent workers handle I/O; main thread only submits + polls.
    // send_failures are safely isolated from recv_failures (unlike the
    // old per-exchange std::thread that could std::terminate on recv throw).

    void start_workers(const std::vector<int>& source_ranks) {
        if (workers_started_) return;
        workers_started_ = true;
        recv_source_ranks_ = source_ranks;

        int n_recv = static_cast<int>(source_ranks.size());
        recv_mutexes_.reserve(n_recv);
        recv_cvs_.reserve(n_recv);
        recv_task_addrs_.resize(n_recv, 0);
        recv_task_sizes_.resize(n_recv, 0);
        recv_ready_.resize(n_recv);
        recv_done_.resize(n_recv);
        recv_error_.resize(n_recv);
        recv_error_msgs_.resize(n_recv);
        for (int i = 0; i < n_recv; ++i) {
            recv_ready_[i] = false;
            recv_done_[i] = false;
            recv_error_[i] = false;
            recv_mutexes_.emplace_back(std::make_unique<std::mutex>());
            recv_cvs_.emplace_back(std::make_unique<std::condition_variable>());
        }

        send_worker_thread_ = std::thread(&GeminiReplicasNative::send_worker_func, this);
        for (int i = 0; i < n_recv; ++i) {
            recv_worker_threads_.emplace_back(
                &GeminiReplicasNative::recv_worker_func, this, i);
        }
        std::cout << "[Rank " << rank_ << "] Workers started: 1 send + "
                  << n_recv << " recv threads" << std::endl;
    }

    void stop_workers() {
        bool expected = false;
        if (!stop_workers_.compare_exchange_strong(expected, true))
            return;

        send_cv_.notify_all();
        for (auto& cv : recv_cvs_) cv->notify_all();

        if (send_worker_thread_.joinable()) send_worker_thread_.join();
        for (auto& t : recv_worker_threads_)
            if (t.joinable()) t.join();

        workers_started_ = false;
        std::cout << "[Rank " << rank_ << "] Workers stopped" << std::endl;
    }

private:
    void send_worker_func() {
        while (true) {
            {
                std::unique_lock<std::mutex> lk(send_mutex_);
                send_cv_.wait(lk, [this] {
                    return stop_workers_ || send_ready_;
                });
            }
            if (stop_workers_) break;
            if (!send_ready_) continue;

            try {
                const uint8_t* data = reinterpret_cast<const uint8_t*>(send_task_addr_);
                connection_manager_->broadcast_to_targets(data, send_task_size_);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(send_error_mutex_);
                send_error_msg_ = e.what();
                send_error_ = true;
            } catch (...) {
                std::lock_guard<std::mutex> lk(send_error_mutex_);
                send_error_msg_ = "unknown send error";
                send_error_ = true;
            }

            send_ready_ = false;
            send_done_ = true;
        }
    }

    void recv_worker_func(int idx) {
        int source_rank = recv_source_ranks_[idx];
        while (true) {
            {
                std::unique_lock<std::mutex> lk(*recv_mutexes_[idx]);
                recv_cvs_[idx]->wait(lk, [this, idx] {
                    return stop_workers_ || recv_ready_[idx];
                });
            }
            if (stop_workers_) break;
            if (!recv_ready_[idx]) continue;

            try {
                uint8_t* buf = reinterpret_cast<uint8_t*>(recv_task_addrs_[idx]);
                connection_manager_->receive_data_from_source(
                    source_rank, buf, recv_task_sizes_[idx]);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(recv_error_mutex_);
                recv_error_msgs_[idx] = e.what();
                recv_error_[idx] = true;
            } catch (...) {
                std::lock_guard<std::mutex> lk(recv_error_mutex_);
                recv_error_msgs_[idx] = "unknown recv error";
                recv_error_[idx] = true;
            }

            recv_ready_[idx] = false;
            recv_done_[idx] = true;
        }
    }

public:
    void reset_exchange_state() {
        // Reset all per-exchange flags.  Workers are idle (ready=false, done=true).
        send_ready_ = false;
        send_done_ = false;
        send_error_ = false;
        send_task_addr_ = 0;
        send_task_size_ = 0;
        for (size_t i = 0; i < recv_source_ranks_.size(); ++i) {
            recv_ready_[i] = false;
            recv_done_[i] = false;
            recv_error_[i] = false;
            recv_task_addrs_[i] = 0;
            recv_task_sizes_[i] = 0;
        }
    }

public:
    void submit_send_buffer(uintptr_t buffer_addr, size_t buffer_size) {
        if (!initialized_)
            throw std::runtime_error("GeminiReplicasNative not initialized");
        if (!workers_started_)
            throw std::runtime_error("Workers not started — call start_workers first");

        send_task_addr_ = buffer_addr;
        send_task_size_ = buffer_size;
        send_ready_ = true;
        send_done_ = false;
        send_error_ = false;
        send_cv_.notify_one();

        std::cout << "[Rank " << rank_ << "] Submitted send buffer: "
                  << buffer_size << " bytes" << std::endl;
    }

    void submit_recv_buffer(int source_rank, uintptr_t buffer_addr, size_t buffer_size) {
        if (!initialized_)
            throw std::runtime_error("GeminiReplicasNative not initialized");
        if (!workers_started_)
            throw std::runtime_error("Workers not started — call start_workers first");

        // Find the recv slot for this source_rank
        int idx = -1;
        for (size_t i = 0; i < recv_source_ranks_.size(); ++i) {
            if (recv_source_ranks_[i] == source_rank) { idx = static_cast<int>(i); break; }
        }
        if (idx < 0)
            throw std::runtime_error("Source rank " + std::to_string(source_rank) +
                                     " not in registered recv source_ranks");

        recv_task_addrs_[idx] = buffer_addr;
        recv_task_sizes_[idx] = buffer_size;
        recv_ready_[idx] = true;
        recv_done_[idx] = false;
        recv_error_[idx] = false;
        recv_cvs_[idx]->notify_one();

        std::cout << "[Rank " << rank_ << "] Submitted recv buffer for source rank "
                  << source_rank << ": " << buffer_size << " bytes" << std::endl;
    }

    void wait_for_exchange_completion() {
        // Busy-poll until all workers are done (same style as ecnaive).
        int wait_count = 0;
        while (true) {
            bool all_done = send_done_;
            if (all_done) {
                for (size_t i = 0; i < recv_source_ranks_.size(); ++i) {
                    if (!recv_done_[i]) { all_done = false; break; }
                }
            }
            // Also stop if any worker hit an error
            bool any_error = send_error_;
            if (!any_error) {
                for (size_t i = 0; i < recv_source_ranks_.size(); ++i) {
                    if (recv_error_[i]) { any_error = true; break; }
                }
            }
            if (all_done || any_error) break;

            if (wait_count % 200 == 0 && wait_count > 0) {
                std::cout << "[Rank " << rank_ << "] Waiting for exchange (send="
                          << send_done_ << " recvs_done=";
                for (size_t i = 0; i < recv_source_ranks_.size(); ++i)
                    std::cout << recv_done_[i];
                std::cout << ")..." << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            wait_count++;
        }

        // Re-throw any worker errors
        if (send_error_) {
            std::string msg;
            { std::lock_guard<std::mutex> lk(send_error_mutex_); msg = send_error_msg_; }
            throw std::runtime_error("[Rank " + std::to_string(rank_) + "] Send worker error: " + msg);
        }
        for (size_t i = 0; i < recv_source_ranks_.size(); ++i) {
            if (recv_error_[i]) {
                std::string msg;
                { std::lock_guard<std::mutex> lk(recv_error_mutex_); msg = recv_error_msgs_[i]; }
                throw std::runtime_error("[Rank " + std::to_string(rank_) + "] Recv worker error (src="
                    + std::to_string(recv_source_ranks_[i]) + "): " + msg);
            }
        }

        std::cout << "[Rank " << rank_ << "] Exchange completed successfully" << std::endl;
    }
    
    bool is_initialized() const {
        return initialized_;
    }
    
    int get_rank() const {
        return rank_;
    }
    
    std::vector<int> get_target_ranks() const {
        return target_ranks_;
    }
    
    void register_buffer(uintptr_t buffer_addr, size_t buffer_size) {
        /**
         * Register buffer for RDMA operations.
         * Only effective when use_rdma is enabled.
         * 
         * Args:
         *   buffer_addr: Memory address of the buffer
         *   buffer_size: Size of the buffer in bytes
         */
        if (!use_rdma_) {
            // No-op for ASIO mode
            return;
        }
        
        std::cout << "[Rank " << rank_ << "] Registering buffer at 0x" << std::hex << buffer_addr 
                  << std::dec << ", size: " << (buffer_size / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        
        connection_manager_->register_buffer(buffer_addr, buffer_size);
        
        std::cout << "[Rank " << rank_ << "] Buffer registered successfully" << std::endl;
    }
    
    void unregister_buffer(uintptr_t buffer_addr) {
        /**
         * Unregister buffer for RDMA operations.
         * Only effective when use_rdma is enabled.
         * 
         * Args:
         *   buffer_addr: Memory address of the buffer
         */
        if (!use_rdma_) {
            // No-op for ASIO mode
            return;
        }
        
        std::cout << "[Rank " << rank_ << "] Unregistering buffer at 0x" << std::hex << buffer_addr << std::dec << std::endl;
        
        connection_manager_->unregister_buffer(buffer_addr);
        
        std::cout << "[Rank " << rank_ << "] Buffer unregistered successfully" << std::endl;
    }
};

// Python bindings
PYBIND11_MODULE(gemini_replicas_native, m) {
    m.doc() = "Gemini Replicas Native C++ Module with ASIO/RDMA for multi-replica data transfer";
    
    py::class_<GeminiReplicasNative>(m, "GeminiReplicasNative")
        .def(py::init<int, int, const std::vector<int>&, const std::vector<std::string>&, 
                      const std::vector<int>&, const std::string&, int, int, bool>(),
             py::arg("rank"),
             py::arg("world_size"),
             py::arg("target_ranks"),
             py::arg("target_ips"),
             py::arg("target_ports"),
             py::arg("my_ip"),
             py::arg("my_port"),
             py::arg("num_source_ranks"),
             py::arg("use_rdma") = false,
             "Create GeminiReplicasNative instance (Phase 1: start acceptor)")
        .def("finalize_connections", &GeminiReplicasNative::finalize_connections,
             "Finalize connections (Phase 2: connect to all targets)")
        .def("broadcast_to_targets", &GeminiReplicasNative::broadcast_to_targets,
             py::arg("buffer_addr"),
             py::arg("buffer_size"),
             "Broadcast data to all target ranks")
        .def("receive_from_source", &GeminiReplicasNative::receive_from_source,
             py::arg("buffer_addr"),
             py::arg("buffer_size"),
             "Receive data from one source rank")
        .def("submit_send_buffer", &GeminiReplicasNative::submit_send_buffer,
             py::arg("buffer_addr"),
             py::arg("buffer_size"),
             "Submit send buffer (wakes send worker, non-blocking)")
        .def("submit_recv_buffer", &GeminiReplicasNative::submit_recv_buffer,
             py::arg("source_rank"),
             py::arg("buffer_addr"),
             py::arg("buffer_size"),
             "Submit receive buffer for a specific source rank (wakes recv worker, non-blocking)")
        .def("start_workers", &GeminiReplicasNative::start_workers,
             py::arg("source_ranks"),
             "Start persistent send+recv worker threads (call after finalize_connections)")
        .def("wait_for_exchange_completion", &GeminiReplicasNative::wait_for_exchange_completion,
             "Block until all send/recv workers finish the current exchange (polls atomics, 5ms sleep)")
        .def("reset_exchange_state", &GeminiReplicasNative::reset_exchange_state,
             "Reset per-exchange flags for the next exchange")
        .def("is_initialized", &GeminiReplicasNative::is_initialized,
             "Check if fully initialized")
        .def("get_rank", &GeminiReplicasNative::get_rank,
             "Get current rank")
        .def("get_target_ranks", &GeminiReplicasNative::get_target_ranks,
             "Get list of target ranks")
        .def("register_buffer", &GeminiReplicasNative::register_buffer,
             py::arg("buffer_addr"),
             py::arg("buffer_size"),
             "Register buffer for RDMA operations (RDMA mode only)")
        .def("unregister_buffer", &GeminiReplicasNative::unregister_buffer,
             py::arg("buffer_addr"),
             "Unregister buffer for RDMA operations (RDMA mode only)");
}

