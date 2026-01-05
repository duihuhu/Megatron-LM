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

namespace py = pybind11;

/**
 * ASIO Connection Manager for Multi-Replica Communication
 * 
 * Manages TCP connections for broadcasting data to multiple target ranks
 * and receiving data from multiple source ranks.
 */
class GeminiReplicasAsioConnectionManager {
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
          my_ip_(my_ip),
          my_port_(my_port),
          expected_recv_connections_(expected_recv_connections)
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
 * Main Gemini Replicas Native Class
 * 
 * Provides Python interface for multi-replica data transfer.
 */
class GeminiReplicasNative {
private:
    int rank_;
    int world_size_;
    std::vector<int> target_ranks_;
    
    std::unique_ptr<GeminiReplicasAsioConnectionManager> connection_manager_;
    
    std::atomic<bool> initialized_{false};
    
    // Buffer management for exchange
    uintptr_t send_buffer_addr_{0};
    size_t send_buffer_size_{0};
    
    struct RecvBufferInfo {
        int source_rank;
        uintptr_t buffer_addr;
        size_t buffer_size;
    };
    std::vector<RecvBufferInfo> recv_buffers_;

public:
    GeminiReplicasNative(
        int rank, int world_size,
        const std::vector<int>& target_ranks,
        const std::vector<std::string>& target_ips,
        const std::vector<int>& target_ports,
        const std::string& my_ip, int my_port,
        int num_source_ranks
    )
        : rank_(rank),
          world_size_(world_size),
          target_ranks_(target_ranks)
    {
        std::cout << "[Rank " << rank_ << "] Creating GeminiReplicasNative with " 
                  << target_ranks_.size() << " targets and " 
                  << num_source_ranks << " sources" << std::endl;
        
        // Phase 1: Create connection manager and start acceptor
        connection_manager_ = std::make_unique<GeminiReplicasAsioConnectionManager>(
            rank_, world_size_,
            target_ranks, target_ips, target_ports,
            my_ip, my_port,
            num_source_ranks  // Pass number of expected incoming connections
        );
        
        connection_manager_->initialize_connections();
        
        std::cout << "[Rank " << rank_ << "] GeminiReplicasNative created (Phase 1 complete)" << std::endl;
    }
    
    ~GeminiReplicasNative() {
        std::cout << "[Rank " << rank_ << "] Destroying GeminiReplicasNative" << std::endl;
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
    
    void submit_send_buffer(uintptr_t buffer_addr, size_t buffer_size) {
        /**
         * Submit send buffer for later execution.
         * 
         * Args:
         *   buffer_addr: Memory address of the buffer to send
         *   buffer_size: Size of the buffer in bytes
         */
        if (!initialized_) {
            throw std::runtime_error("GeminiReplicasNative not initialized");
        }
        
        send_buffer_addr_ = buffer_addr;
        send_buffer_size_ = buffer_size;
        
        std::cout << "[Rank " << rank_ << "] Submitted send buffer: " 
                  << buffer_size << " bytes" << std::endl;
    }
    
    void submit_recv_buffer(int source_rank, uintptr_t buffer_addr, size_t buffer_size) {
        /**
         * Submit receive buffer for a specific source rank.
         * 
         * Args:
         *   source_rank: Rank to receive data from
         *   buffer_addr: Memory address of the buffer to receive into
         *   buffer_size: Size of the buffer in bytes
         */
        if (!initialized_) {
            throw std::runtime_error("GeminiReplicasNative not initialized");
        }
        
        recv_buffers_.push_back({source_rank, buffer_addr, buffer_size});
        
        std::cout << "[Rank " << rank_ << "] Submitted recv buffer for source rank " 
                  << source_rank << ": " << buffer_size << " bytes" << std::endl;
    }
    
    void execute_exchange() {
        /**
         * Execute concurrent send/receive operations using submitted buffers.
         * 
         * This method:
         * 1. Starts a background thread to send data to all target ranks
         * 2. Receives data from all source ranks (in any order, matched by source_rank)
         * 3. Data is received directly into the pre-allocated buffers (no extra copy!)
         * 
         * Blocks until all operations are complete.
         */
        if (!initialized_) {
            throw std::runtime_error("GeminiReplicasNative not initialized");
        }
        
        if (send_buffer_addr_ == 0) {
            throw std::runtime_error("Send buffer not submitted");
        }
        
        std::cout << "[Rank " << rank_ << "] Starting exchange: send to " 
                  << target_ranks_.size() << " targets, receive from " 
                  << recv_buffers_.size() << " sources" << std::endl;
        
        // Create a map: source_rank -> buffer_info for fast lookup
        std::map<int, RecvBufferInfo> recv_buffer_map;
        for (const auto& recv_info : recv_buffers_) {
            recv_buffer_map[recv_info.source_rank] = recv_info;
        }
        
        // Start send in background thread
        std::exception_ptr send_exception = nullptr;
        std::thread send_thread([this, &send_exception]() {
            try {
                const uint8_t* data = reinterpret_cast<const uint8_t*>(send_buffer_addr_);
                connection_manager_->broadcast_to_targets(data, send_buffer_size_);
            } catch (...) {
                send_exception = std::current_exception();
            }
        });
        
        // Receive from all sources in main thread (in any order)
        for (size_t i = 0; i < recv_buffers_.size(); ++i) {
            std::cout << "[Rank " << rank_ << "] Waiting for data from any source (" 
                      << (i + 1) << "/" << recv_buffers_.size() << ")..." << std::endl;
            
            // Peek incoming connection to get source_rank and size
            auto [actual_source_rank, data_size, socket] = connection_manager_->peek_incoming_data();
            
            // Find the corresponding buffer
            auto it = recv_buffer_map.find(actual_source_rank);
            if (it == recv_buffer_map.end()) {
                throw std::runtime_error(
                    "Received data from unexpected source rank " + std::to_string(actual_source_rank)
                );
            }
            
            const auto& recv_info = it->second;
            
            if (data_size != recv_info.buffer_size) {
                std::cerr << "[Rank " << rank_ << "] Warning: Incoming data size " << data_size 
                          << " bytes from rank " << actual_source_rank 
                          << ", expected " << recv_info.buffer_size << " bytes" << std::endl;
            }
            
            // Receive data directly into the target buffer (no extra copy!)
            uint8_t* target_buffer = reinterpret_cast<uint8_t*>(recv_info.buffer_addr);
            connection_manager_->receive_data_into_buffer(
                std::move(socket), target_buffer, recv_info.buffer_size, data_size
            );
            
            std::cout << "[Rank " << rank_ << "] Received " << data_size 
                      << " bytes from source rank " << actual_source_rank 
                      << " directly into target buffer (zero-copy)" << std::endl;
            
            // Remove from map so we don't receive from the same source twice
            recv_buffer_map.erase(it);
        }
        
        // Wait for send thread to complete
        send_thread.join();
        
        // Check for send exception
        if (send_exception) {
            std::rethrow_exception(send_exception);
        }
        
        std::cout << "[Rank " << rank_ << "] Exchange completed successfully" << std::endl;
        
        // Clear buffers for next exchange
        send_buffer_addr_ = 0;
        send_buffer_size_ = 0;
        recv_buffers_.clear();
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
};

// Python bindings
PYBIND11_MODULE(gemini_replicas_native, m) {
    m.doc() = "Gemini Replicas Native C++ Module with ASIO for multi-replica data transfer";
    
    py::class_<GeminiReplicasNative>(m, "GeminiReplicasNative")
        .def(py::init<int, int, const std::vector<int>&, const std::vector<std::string>&, 
                      const std::vector<int>&, const std::string&, int, int>(),
             py::arg("rank"),
             py::arg("world_size"),
             py::arg("target_ranks"),
             py::arg("target_ips"),
             py::arg("target_ports"),
             py::arg("my_ip"),
             py::arg("my_port"),
             py::arg("num_source_ranks"),
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
             "Submit send buffer for later execution")
        .def("submit_recv_buffer", &GeminiReplicasNative::submit_recv_buffer,
             py::arg("source_rank"),
             py::arg("buffer_addr"),
             py::arg("buffer_size"),
             "Submit receive buffer for a specific source rank")
        .def("execute_exchange", &GeminiReplicasNative::execute_exchange,
             "Execute concurrent send/receive operations")
        .def("is_initialized", &GeminiReplicasNative::is_initialized,
             "Check if fully initialized")
        .def("get_rank", &GeminiReplicasNative::get_rank,
             "Get current rank")
        .def("get_target_ranks", &GeminiReplicasNative::get_target_ranks,
             "Get list of target ranks");
}

