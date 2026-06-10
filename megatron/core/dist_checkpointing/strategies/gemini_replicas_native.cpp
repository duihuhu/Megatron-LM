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
#include <unordered_map>

// RDMA headers
#include <infiniband/verbs.h>

#include "rdma_device_utils.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace py = pybind11;

namespace {

// Match gemini_native.cpp TCP connect retry (100 x 100ms).
constexpr int kGeminiReplicasTcpConnectMaxRetries = 100;
constexpr int kGeminiReplicasTcpConnectRetryDelayMs = 100;

// Connect TCP socket with retry (aligned with gemini_native patterns).
void asio_tcp_connect_with_retry(
    boost::asio::io_context& io_ctx,
    boost::asio::ip::tcp::socket& socket,
    const std::string& host,
    int port,
    int rank,
    int target_rank)
{
    boost::asio::ip::tcp::resolver resolver(io_ctx);
    auto endpoints = resolver.resolve(host, std::to_string(port));

    for (int attempt = 0; attempt < kGeminiReplicasTcpConnectMaxRetries; ++attempt) {
        boost::system::error_code ec;
        boost::asio::connect(socket, endpoints, ec);
        if (!ec) {
            if (attempt > 0) {
                std::cout << "[Rank " << rank << "] TCP connected to target rank "
                          << target_rank << " after " << (attempt + 1) << " attempts"
                          << std::endl;
            }
            return;
        }
        if (attempt < kGeminiReplicasTcpConnectMaxRetries - 1) {
            boost::system::error_code close_ec;
            socket.close(close_ec);
            socket = boost::asio::ip::tcp::socket(io_ctx);
            std::this_thread::sleep_for(
                std::chrono::milliseconds(kGeminiReplicasTcpConnectRetryDelayMs));
        } else {
            throw std::runtime_error(
                "Failed to connect to target rank " + std::to_string(target_rank) +
                " at " + host + ":" + std::to_string(port) + " after " +
                std::to_string(kGeminiReplicasTcpConnectMaxRetries) +
                " attempts: " + ec.message());
        }
    }
}

}  // namespace

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

    // Send data to a single target (not broadcast).  Reuses the existing
    // connection to the given target index.  Protocol matches broadcast_to_targets:
    // [source_rank(4B)][size(8B)][data] for ASIO, or size+ACK+RDMA for RDMA.
    virtual void send_to_one_target(size_t target_idx, const uint8_t* data,
                                    size_t size, int source_rank) = 0;

    // Return the list of source ranks in recv connection order.
    // Used by GeminiReplicasNative to build source_rank→index mapping for
    // directed recv during hardware recovery.
    virtual std::vector<int> get_recv_source_ranks() const = 0;
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

    // ---- Directed P2P methods (for hardware recovery) ----

    void send_to_one_target(size_t target_idx, const uint8_t* data,
                            size_t size, int source_rank) override {
        /**
         * Send data to a single target rank (not broadcast).
         * Protocol matches broadcast_to_targets: [source_rank(4B)][size(8B)][data]
         * Uses synchronous blocking ASIO write on the specific socket.
         */
        if (!(*send_connected_[target_idx])) {
            throw std::runtime_error("Send socket to target index "
                + std::to_string(target_idx) + " not connected");
        }
        int32_t src = static_cast<int32_t>(source_rank);
        boost::asio::write(*send_sockets_[target_idx],
            boost::asio::buffer(&src, sizeof(src)));
        uint64_t sz = static_cast<uint64_t>(size);
        boost::asio::write(*send_sockets_[target_idx],
            boost::asio::buffer(&sz, sizeof(sz)));
        boost::asio::write(*send_sockets_[target_idx],
            boost::asio::buffer(data, size));
        std::cout << "[Rank " << rank_ << "] Sent " << size
                  << " bytes to target rank " << target_ranks_[target_idx]
                  << " (directed P2P)" << std::endl;
    }

    std::vector<int> get_recv_source_ranks() const override {
        // ASIO uses a socket pool — no fixed source-rank to socket mapping.
        // Directed recv will use the pool-based receive_data() with a
        // source_rank verification (safe because recovery has only one sender).
        return {};
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
            std::cout << "[Rank " << rank_ << "] Connecting to target rank " 
                      << target_ranks_[target_idx] << " at " 
                      << target_ips_[target_idx] << ":" << target_ports_[target_idx] << std::endl;

            send_sockets_[target_idx] =
                std::make_unique<boost::asio::ip::tcp::socket>(io_context_);
            asio_tcp_connect_with_retry(
                io_context_, *send_sockets_[target_idx],
                target_ips_[target_idx], target_ports_[target_idx],
                rank_, target_ranks_[target_idx]);

            std::lock_guard<std::mutex> lock(connection_mutex_);
            *send_connected_[target_idx] = true;
            connection_cv_.notify_all();

            std::cout << "[Rank " << rank_ << "] Connected to target rank "
                      << target_ranks_[target_idx] << std::endl;
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

    // ASIO resources (connection setup, matching eccheck pattern)
    boost::asio::io_context io_context_;
    std::unique_ptr<boost::asio::ip::tcp::acceptor> recv_acceptor_;
    std::unique_ptr<std::thread> io_thread_;  // runs io_context

    // TCP control sockets (raw fds from ASIO sockets' native_handle)
    std::vector<int> control_socks_send_;  // One per target rank
    std::vector<int> control_socks_recv_;  // One per source rank

    // ASIO socket objects kept alive so native_handle fds stay valid
    std::vector<std::unique_ptr<boost::asio::ip::tcp::socket>> send_socks_;
    std::vector<std::unique_ptr<boost::asio::ip::tcp::socket>> recv_socks_;
    
    // Connection status
    std::atomic<bool> connected_{false};
    
    // Synchronization
    std::mutex connection_mutex_;
    std::condition_variable connection_cv_;
    std::mutex buffer_mutex_;
    std::mutex recv_mutex_;
    std::mutex send_cq_poll_mutex_;
    
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
    static const int MAX_WR = 256;
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

        // Start ASIO io_context in background thread.
        // The open acceptor keeps io_context busy; when all connections are
        // accepted and acceptor is closed, io_context::run() exits naturally.
        io_thread_ = std::make_unique<std::thread>([this]() {
            io_context_.run();
        });

        // Initialize RDMA resources
        init_rdma_resources();

        // Start TCP listener via ASIO (matching eccheck pattern)
        start_tcp_listener();

        std::cout << "[Rank " << rank_ << "] RDMA initialization complete (Phase 1)" << std::endl;
    }
    
    void connect_and_wait() override {
        std::cout << "[Rank " << rank_ << "] Connecting to targets and waiting for connections..." << std::endl;

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

        // Connect to target ranks (control channel + RDMA QP).
        // IMPORTANT: wrap in try-catch so that accept_thread is joined
        // even on failure — otherwise std::thread destructor calls
        // std::terminate (same pattern as the old execute_exchange bug).
        std::exception_ptr connect_exception = nullptr;
        try {
            for (size_t i = 0; i < target_ranks_.size(); ++i) {
                std::cout << "[Rank " << rank_ << "] Connecting to target " << (i+1)
                          << "/" << target_ranks_.size() << " (rank " << target_ranks_[i] << ")..." << std::endl;
                connect_to_target(i);
            }
        } catch (...) {
            connect_exception = std::current_exception();
        }

        accept_thread.join();

        if (accept_exception) {
            try { std::rethrow_exception(accept_exception); }
            catch (const std::exception& e) {
                std::cerr << "[Rank " << rank_ << "] ERROR in accept thread: " << e.what() << std::endl;
                throw;
            }
        }
        if (connect_exception) {
            try { std::rethrow_exception(connect_exception); }
            catch (const std::exception& e) {
                std::cerr << "[Rank " << rank_ << "] ERROR in connect_and_wait: " << e.what() << std::endl;
                throw;
            }
        }

        // Wait for all connections
        std::cout << "[Rank " << rank_ << "] Waiting for all connections to be ready..." << std::endl;
        wait_for_connections();

        // No warmup — matching eccheck. QPs are connected during
        // exchange_and_connect and the first real data transfer validates them.
        std::cout << "[Rank " << rank_ << "] All RDMA connections established (no warmup)" << std::endl;
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
            // Signal the last WR of each batch so we can poll per-batch.
            // (The chain is broken at batch boundaries, so each batch must
            // generate its own completion.)
            bool is_batch_last = (i + 1 == chunk_count) || ((i + 1) % MAX_BATCH_WR == 0);
            wrs[i].send_flags = is_batch_last ? IBV_SEND_SIGNALED : 0;
            wrs[i].next = (i < chunk_count - 1) ? &wrs[i + 1] : nullptr;
        }
        
        // Post send work requests in batches
        for (size_t batch_start = 0; batch_start < chunk_count; batch_start += MAX_BATCH_WR) {
            size_t batch_end = std::min(batch_start + MAX_BATCH_WR, chunk_count);
            
            // Detach this batch's tail from the next batch
            if (batch_end < chunk_count)
                wrs[batch_end - 1].next = nullptr;

            ibv_send_wr* bad_wr = nullptr;
            if (ibv_post_send(qp, &wrs[batch_start], &bad_wr) != 0) {
                throw std::runtime_error("Failed to post send work request");
            }

            // Poll completions for signaled requests in this batch.
            // Serialize send_cq polls so parallel target sends do not steal WRs.
            for (size_t i = batch_start; i < batch_end; ++i) {
                if (wrs[i].send_flags & IBV_SEND_SIGNALED) {
                    std::lock_guard<std::mutex> cq_lock(send_cq_poll_mutex_);
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
        
        for (size_t batch_start = 0; batch_start < chunk_count; batch_start += MAX_BATCH_WR) {
            size_t batch_end = std::min(batch_start + MAX_BATCH_WR, chunk_count);

            // Detach this batch's tail from the next batch
            if (batch_end < chunk_count)
                wrs[batch_end - 1].next = nullptr;

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
        
        // Send size to all targets via control channel, then wait for ACK
        // (receiver posts recv WRs and ACKs — matches ecnaive ordering).
        uint64_t size_net = htobe64(size);
        for (size_t i = 0; i < target_ranks_.size(); ++i) {
            if (send(control_socks_send_[i], &size_net, sizeof(size_net), 0) != sizeof(size_net)) {
                throw std::runtime_error("Failed to send size to target " + std::to_string(target_ranks_[i]));
            }
        }
        // Wait for ACKs from all targets (accept in any order).
        // Using sequential recv() on each socket can cause distributed deadlock
        // when combined with recv_mutex_ on the receiver side: if the first
        // target's ACK is delayed (its recv worker is blocked behind another
        // recv worker holding recv_mutex_ during RDMA), the sender hangs even
        // though other targets already sent their ACKs.
        std::vector<bool> acked(target_ranks_.size(), false);
        size_t acked_count = 0;
        while (acked_count < target_ranks_.size()) {
            fd_set read_fds;
            FD_ZERO(&read_fds);
            int max_fd = 0;
            for (size_t i = 0; i < target_ranks_.size(); ++i) {
                if (acked[i]) continue;
                FD_SET(control_socks_send_[i], &read_fds);
                if (control_socks_send_[i] > max_fd) {
                    max_fd = control_socks_send_[i];
                }
            }
            struct timeval tv = {0, 500};
            int ret = select(max_fd + 1, &read_fds, nullptr, nullptr, &tv);
            if (ret > 0) {
                for (size_t i = 0; i < target_ranks_.size(); ++i) {
                    if (acked[i]) continue;
                    if (!FD_ISSET(control_socks_send_[i], &read_fds)) continue;
                    char ack;
                    if (recv(control_socks_send_[i], &ack, 1, MSG_WAITALL) != 1 || ack != 'A') {
                        throw std::runtime_error("Failed to receive ACK from target "
                            + std::to_string(target_ranks_[i]));
                    }
                    acked[i] = true;
                    acked_count++;
                }
            }
        }

        ibv_mr* mr = find_registered_mr(reinterpret_cast<uintptr_t>(data), size);
        if (mr == nullptr) {
            throw std::runtime_error(
                "RDMA broadcast: send buffer is not registered (addr=0x"
                + std::to_string(reinterpret_cast<uintptr_t>(data))
                + ", size=" + std::to_string(size) + ")");
        }

        const uint8_t* send_data = data;

        if (target_ranks_.size() <= 1) {
            if (!target_ranks_.empty()) {
                send_data_chunked(send_data, size, mr, send_qps_[0]);
            }
            return;
        }

        // Parallel RDMA send to all targets (wall time ~ max, not sum).
        std::vector<std::exception_ptr> send_exceptions(target_ranks_.size());
        std::vector<std::thread> send_threads;
        send_threads.reserve(target_ranks_.size());
        for (size_t i = 0; i < target_ranks_.size(); ++i) {
            send_threads.emplace_back([this, i, send_data, size, mr, &send_exceptions]() {
                try {
                    send_data_chunked(send_data, size, mr, send_qps_[i]);
                } catch (...) {
                    send_exceptions[i] = std::current_exception();
                }
            });
        }
        for (auto& t : send_threads) {
            t.join();
        }
        for (size_t i = 0; i < send_exceptions.size(); ++i) {
            if (send_exceptions[i]) {
                try {
                    std::rethrow_exception(send_exceptions[i]);
                } catch (const std::exception& e) {
                    throw std::runtime_error(
                        "Failed RDMA send to target rank " + std::to_string(target_ranks_[i])
                        + ": " + e.what());
                }
            }
        }
    }

    // ---- Directed P2P methods (for hardware recovery) ----

    void send_to_one_target(size_t target_idx, const uint8_t* data,
                            size_t size, int source_rank) override {
        /**
         * Send data to a single target via RDMA.
         * Control channel handshake (size + ACK) on the specific control socket,
         * then RDMA data transfer on the specific QP.
         */
        if (!connected_) {
            throw std::runtime_error("Not connected");
        }
        // Step 1: TCP control — send size, wait for ACK
        uint64_t sz = htobe64(size);
        if (send(control_socks_send_[target_idx], &sz, sizeof(sz), MSG_NOSIGNAL)
            != static_cast<ssize_t>(sizeof(sz))) {
            throw std::runtime_error("Failed to send size to target rank "
                + std::to_string(target_ranks_[target_idx]));
        }
        char ack;
        if (recv(control_socks_send_[target_idx], &ack, 1, MSG_WAITALL) != 1 || ack != 'A') {
            throw std::runtime_error("Failed to receive ACK from target rank "
                + std::to_string(target_ranks_[target_idx]));
        }

        // Step 2: RDMA data transfer on the single QP
        ibv_mr* mr = find_registered_mr(reinterpret_cast<uintptr_t>(data), size);
        bool use_temp = (mr == nullptr);
        if (use_temp) {
            if (size > temp_send_buffer_.size()) {
                throw std::runtime_error("Data size exceeds temporary send buffer size");
            }
            std::memcpy(temp_send_buffer_.data(), data, size);
            mr = temp_send_mr_;
        }
        const uint8_t* send_data = use_temp ? temp_send_buffer_.data() : data;
        send_data_chunked(send_data, size, mr, send_qps_[target_idx]);

        std::cout << "[Rank " << rank_ << "] Sent " << size
                  << " bytes to target rank " << target_ranks_[target_idx]
                  << " via RDMA (directed P2P)" << std::endl;
    }

    std::vector<int> get_recv_source_ranks() const override {
        return recv_source_ranks_;
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
        if (!connected_) {
            throw std::runtime_error("Not connected");
        }

        // TCP handshake: find an available source, receive size, send ACK.
        int found_idx = -1;
        uint64_t size_net;
        {
            std::lock_guard<std::mutex> lock(recv_mutex_);

            // Receive from any available source
            for (size_t i = 0; i < control_socks_recv_.size(); ++i) {
                fd_set read_fds;
                FD_ZERO(&read_fds);
                FD_SET(control_socks_recv_[i], &read_fds);

                struct timeval tv = {0, 1000}; // 1ms timeout
                int ret = select(control_socks_recv_[i] + 1, &read_fds, nullptr, nullptr, &tv);

                if (ret > 0) {
                    found_idx = static_cast<int>(i);
                    break;
                }
            }

            if (found_idx < 0) {
                throw std::runtime_error("No data available from any source");
            }

            // Receive size
            if (recv(control_socks_recv_[found_idx], &size_net, sizeof(size_net), MSG_WAITALL) != sizeof(size_net)) {
                throw std::runtime_error("Failed to receive size from source");
            }

            // Send ACK
            char ack = 'A';
            if (send(control_socks_recv_[found_idx], &ack, 1, 0) != 1) {
                throw std::runtime_error("Failed to send ACK");
            }
        }
        // Mutex released — RDMA transfer without blocking other recv workers.

        size_t recv_size = be64toh(size_net);

        if (recv_size > buffer_size) {
            throw std::runtime_error("Received size exceeds buffer size");
        }

        // Find or use temp buffer for MR
        ibv_mr* mr = find_registered_mr(reinterpret_cast<uintptr_t>(buffer), recv_size);
        bool use_temp = (mr == nullptr);

        if (use_temp) {
            std::lock_guard<std::mutex> lock(recv_mutex_);
            if (recv_size > temp_recv_buffer_.size()) {
                throw std::runtime_error("Receive size exceeds temporary buffer size");
            }
            mr = temp_recv_mr_;
            receive_data_chunked(temp_recv_buffer_.data(), recv_size, mr, recv_qps_[found_idx]);
            std::memcpy(buffer, temp_recv_buffer_.data(), recv_size);
        } else {
            receive_data_chunked(buffer, recv_size, mr, recv_qps_[found_idx]);
        }

        return {recv_source_ranks_[found_idx], recv_size};
    }
    
    std::pair<int, size_t> receive_data_from_source(int source_rank, uint8_t* buffer, size_t buffer_size) override {
        if (!connected_) {
            throw std::runtime_error("Not connected");
        }

        // Find the QP index for this source rank
        auto it = std::find(recv_source_ranks_.begin(), recv_source_ranks_.end(), source_rank);
        if (it == recv_source_ranks_.end()) {
            throw std::runtime_error("Source rank " + std::to_string(source_rank) + " not found");
        }

        size_t qp_idx = std::distance(recv_source_ranks_.begin(), it);

        // TCP handshake: receive size, send ACK.
        // Only hold recv_mutex_ for the short TCP portion — not for the
        // RDMA transfer which can take seconds.  This prevents recv workers
        // from serializing on the mutex and creating a distributed deadlock
        // with the sender's ACK-wait loop.
        uint64_t size_net;
        {
            std::lock_guard<std::mutex> lock(recv_mutex_);

            // Wait for data from this specific source (with timeout)
            fd_set read_fds;
            FD_ZERO(&read_fds);
            FD_SET(control_socks_recv_[qp_idx], &read_fds);

            struct timeval tv = {30, 0}; // 30 second timeout
            int ret = select(control_socks_recv_[qp_idx] + 1, &read_fds, nullptr, nullptr, &tv);

            if (ret <= 0) {
                throw std::runtime_error("Timeout waiting for data from source rank " + std::to_string(source_rank));
            }

            // Receive size
            if (recv(control_socks_recv_[qp_idx], &size_net, sizeof(size_net), MSG_WAITALL) != sizeof(size_net)) {
                throw std::runtime_error("Failed to receive size from source");
            }

            // Send ACK BEFORE releasing mutex — sender is blocked on recv() for this ACK.
            // We must send it while holding the mutex so the sender can proceed with RDMA write.
            char ack = 'A';
            if (send(control_socks_recv_[qp_idx], &ack, 1, 0) != 1) {
                throw std::runtime_error("Failed to send ACK");
            }
        }
        // Mutex released — RDMA transfer runs without blocking other recv workers.

        size_t recv_size = be64toh(size_net);

        if (recv_size > buffer_size) {
            throw std::runtime_error("Received size exceeds buffer size");
        }

        // Find or use temp buffer for MR
        ibv_mr* mr = find_registered_mr(reinterpret_cast<uintptr_t>(buffer), recv_size);
        bool use_temp = (mr == nullptr);

        if (use_temp) {
            // Temp buffer is shared — must hold mutex.
            // In practice buffers are always pre-registered so this path is rare.
            std::lock_guard<std::mutex> lock(recv_mutex_);
            if (recv_size > temp_recv_buffer_.size()) {
                throw std::runtime_error("Receive size exceeds temporary buffer size");
            }
            mr = temp_recv_mr_;
            receive_data_chunked(temp_recv_buffer_.data(), recv_size, mr, recv_qps_[qp_idx]);
            std::memcpy(buffer, temp_recv_buffer_.data(), recv_size);
        } else {
            receive_data_chunked(buffer, recv_size, mr, recv_qps_[qp_idx]);
        }

        return {recv_source_ranks_[qp_idx], recv_size};
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

        // Send ACK BEFORE posting recv WRs (matches ecnaive ordering).
        // Sender waits for this ACK before starting RDMA send, so
        // the recv QP is guaranteed ready.
        char ack = 'A';
        if (send(control_socks_recv_[qp_idx], &ack, 1, 0) != 1) {
            throw std::runtime_error("Failed to send ACK");
        }

        // Receive data via RDMA
        receive_data_chunked(recv_buffer, recv_size, mr, recv_qps_[qp_idx]);

        if (use_temp) {
            std::memcpy(buffer, temp_recv_buffer_.data(), recv_size);
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
        
        // Step 3: Wait for all sends to complete (with timeout, matching
        // eccheck which has no warmup — failures here are non-fatal)
        std::cout << "[Rank " << rank_ << "] Waiting for " << send_qps_.size() << " send completions..." << std::endl;
        try {
            for (size_t i = 0; i < send_qps_.size(); ++i) {
                poll_completion_timeout(send_cq_, 1, 5);  // 5 second timeout per completion
                std::cout << "[Rank " << rank_ << "] Warmup send " << (i+1) << "/" << send_qps_.size() << " completed" << std::endl;
            }

            // Step 4: Wait for all receives to complete
            std::cout << "[Rank " << rank_ << "] Waiting for " << recv_qps_.size() << " receive completions..." << std::endl;
            for (size_t i = 0; i < recv_qps_.size(); ++i) {
                poll_completion_timeout(recv_cq_, 1, 5);
                std::cout << "[Rank " << rank_ << "] Warmup receive " << (i+1) << "/" << recv_qps_.size() << " completed" << std::endl;
            }
            std::cout << "[Rank " << rank_ << "] RDMA warmup complete" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Rank " << rank_ << "] Warmup timed out: " << e.what()
                      << " — skipping (non-fatal, matches eccheck)" << std::endl;
        }
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
        // Use ASIO acceptor instead of raw socket/bind/listen (matching eccheck pattern).
        // ASIO's reuse_address handles TIME_WAIT automatically — no retry loop needed.
        boost::asio::ip::tcp::endpoint endpoint(
            boost::asio::ip::address::from_string(my_ip_), my_port_);
        recv_acceptor_ = std::make_unique<boost::asio::ip::tcp::acceptor>(io_context_);
        recv_acceptor_->open(endpoint.protocol());
        recv_acceptor_->set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        recv_acceptor_->bind(endpoint);
        recv_acceptor_->listen(expected_recv_connections_);
        std::cout << "[Rank " << rank_ << "] ASIO acceptor listening on " << my_ip_ << ":" << my_port_ << std::endl;
    }
    
    void accept_tcp_connection() {
        try {
            std::cout << "[Rank " << rank_ << "] Waiting to accept incoming connection..." << std::endl;
            // ASIO synchronous accept (matching eccheck pattern)
            auto sock = std::make_unique<boost::asio::ip::tcp::socket>(io_context_);
            recv_acceptor_->accept(*sock);
            int client_sock = sock->native_handle();
            std::cout << "[Rank " << rank_ << "] Accepted TCP connection (ASIO)" << std::endl;

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
                throw std::runtime_error("Failed to create recv QP: " + std::string(strerror(errno)));
            }

            // Acceptor receives QP info first (same as eccheck's exchange(false))
            std::cout << "[Rank " << rank_ << "] Exchanging QP info (recv-first)..." << std::endl;
            RdmaConnInfo local_info = get_local_conn_info(qp);
            RdmaConnInfo remote_info;
            if (!exchange_conn_info(client_sock, local_info, remote_info, false)) {
                ibv_destroy_qp(qp);
                throw std::runtime_error("Failed to exchange connection info");
            }

            // Connect QP
            std::cout << "[Rank " << rank_ << "] Connecting recv QP..." << std::endl;
            if (!connect_qp(qp, remote_info)) {
                ibv_destroy_qp(qp);
                throw std::runtime_error("Failed to connect recv QP");
            }

            // Receive source rank from sender
            std::cout << "[Rank " << rank_ << "] Receiving source rank ID..." << std::endl;
            int32_t source_rank_net;
            if (recv(client_sock, &source_rank_net, sizeof(source_rank_net), MSG_WAITALL) != sizeof(source_rank_net)) {
                ibv_destroy_qp(qp);
                throw std::runtime_error("Failed to receive source rank: " + std::string(strerror(errno)));
            }
            int source_rank = ntohl(source_rank_net);

            // Store the QP, source rank, raw fd, and keep ASIO socket alive
            recv_qps_.push_back(qp);
            recv_source_ranks_.push_back(source_rank);
            control_socks_recv_.push_back(client_sock);
            recv_socks_.push_back(std::move(sock));

            std::cout << "[Rank " << rank_ << "] Successfully accepted RDMA connection from rank " << source_rank << std::endl;

            // If we're the last to finish (accept side), signal connected_
            if (recv_qps_.size() == static_cast<size_t>(expected_recv_connections_) &&
                send_qps_.size() == target_ranks_.size() &&
                std::all_of(control_socks_send_.begin(), control_socks_send_.end(), [](int s) { return s >= 0; })) {
                std::lock_guard<std::mutex> lock(connection_mutex_);
                connected_ = true;
                connection_cv_.notify_all();
                std::cout << "[Rank " << rank_ << "] All connections complete (from accept side), notifying waiters" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Rank " << rank_ << "] ERROR in accept_tcp_connection: " << e.what() << std::endl;
            throw;
        }
    }

    void connect_to_target(size_t target_idx) {
        try {
            std::cout << "[Rank " << rank_ << "] Connecting to target "
                      << target_ranks_[target_idx] << " at "
                      << target_ips_[target_idx] << ":" << target_ports_[target_idx] << std::endl;

            auto sock = std::make_unique<boost::asio::ip::tcp::socket>(io_context_);
            asio_tcp_connect_with_retry(
                io_context_, *sock,
                target_ips_[target_idx], target_ports_[target_idx],
                rank_, target_ranks_[target_idx]);
            int fd = sock->native_handle();
            std::cout << "[Rank " << rank_ << "] TCP connected to target "
                      << target_ranks_[target_idx] << " (ASIO)" << std::endl;

            // Connector sends QP info first (same as eccheck's exchange(true))
            std::cout << "[Rank " << rank_ << "] Exchanging QP info (send-first) with target "
                      << target_ranks_[target_idx] << std::endl;
            RdmaConnInfo local_info = get_local_conn_info(send_qps_[target_idx]);
            RdmaConnInfo remote_info;
            if (!exchange_conn_info(fd, local_info, remote_info, true)) {
                throw std::runtime_error("Failed to exchange connection info with target "
                    + std::to_string(target_ranks_[target_idx]));
            }

            // Connect QP
            std::cout << "[Rank " << rank_ << "] Connecting QP to target " << target_ranks_[target_idx] << std::endl;
            if (!connect_qp(send_qps_[target_idx], remote_info)) {
                throw std::runtime_error("Failed to connect QP to target " + std::to_string(target_ranks_[target_idx]));
            }

            // Send my rank to receiver
            std::cout << "[Rank " << rank_ << "] Sending rank ID to target " << target_ranks_[target_idx] << std::endl;
            int32_t my_rank_net = htonl(rank_);
            if (send(fd, &my_rank_net, sizeof(my_rank_net), 0) != sizeof(my_rank_net)) {
                throw std::runtime_error("Failed to send rank to target " + std::to_string(target_ranks_[target_idx]));
            }

            control_socks_send_[target_idx] = fd;
            send_socks_.push_back(std::move(sock));

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
    
    // Exchange QP connection info over a TCP control socket.
    // we_send_first=true: connector sends its info first, then receives remote's
    // we_send_first=false: acceptor receives remote's info first, then sends its own
    // (same deadlock-avoidance pattern as eccheck's exchange_and_connect_qp)
    bool exchange_conn_info(int sock_fd, const RdmaConnInfo& local_info,
                            RdmaConnInfo& remote_info, bool we_send_first) {
        if (we_send_first) {
            if (send(sock_fd, &local_info, sizeof(local_info), 0) != sizeof(local_info))
                return false;
            if (recv(sock_fd, &remote_info, sizeof(remote_info), MSG_WAITALL) != sizeof(remote_info))
                return false;
        } else {
            if (recv(sock_fd, &remote_info, sizeof(remote_info), MSG_WAITALL) != sizeof(remote_info))
                return false;
            if (send(sock_fd, &local_info, sizeof(local_info), 0) != sizeof(local_info))
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
        // Tight spin on the hot path (matches ecnaive_native). No sleep between polls.
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

    // Spin with a deadline (seconds). Used by warmup to avoid hanging forever
    // if a peer hasn't posted recv WRs yet.
    void poll_completion_timeout(ibv_cq* cq, int num_completions, int timeout_secs) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_secs);
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
                continue;
            }
            if (std::chrono::steady_clock::now() > deadline) {
                throw std::runtime_error("poll_completion_timeout expired after "
                    + std::to_string(timeout_secs) + "s (got " + std::to_string(polled)
                    + "/" + std::to_string(num_completions) + " completions)");
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
            
            // Stop ASIO io_context and background thread
            io_context_.stop();
            if (io_thread_ && io_thread_->joinable()) {
                io_thread_->join();
            }
            // Destroy ASIO socket objects (closes fds automatically)
            send_socks_.clear();
            recv_socks_.clear();
            recv_acceptor_.reset();

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

    // Rank→connection-index maps for directed P2P (hardware recovery).
    // Built in finalize_connections() after connect_and_wait().
    std::unordered_map<int, size_t> target_rank_to_idx_;
    std::unordered_map<int, size_t> source_rank_to_idx_;

    std::unique_ptr<IGeminiReplicasConnectionManager> connection_manager_;

    std::atomic<bool> initialized_{false};

    // ---- Worker-thread model (aligned with ecnaive) ----
    // Persistent worker threads: one send_worker + one recv_worker per source.
    // Main thread only submits tasks and polls atomic flags — never blocks on I/O.
    //
    // NOTE: Gemini uses single-slot atomics (not queues+sentinels) because
    // there is only 1 send + N recv tasks per exchange.  ecnaive uses
    // per-channel queues because it has many tasks per channel.  Our simpler
    // model avoids the sentinel ordering issues seen with queues.

    bool workers_started_{false};
    std::atomic<bool> stop_workers_{false};

    // Send worker (single-slot atomic)
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

    // Recv workers (one per source rank, single-slot atomics)
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

    std::mutex exchange_wait_mutex_;
    std::condition_variable exchange_wait_cv_;

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

        // Build rank→connection-index maps for directed P2P (hardware recovery)
        for (size_t i = 0; i < target_ranks_.size(); ++i) {
            target_rank_to_idx_[target_ranks_[i]] = i;
        }
        auto recv_src = connection_manager_->get_recv_source_ranks();
        for (size_t i = 0; i < recv_src.size(); ++i) {
            source_rank_to_idx_[recv_src[i]] = i;
        }
        std::cout << "[Rank " << rank_ << "] Built rank→idx maps: "
                  << target_rank_to_idx_.size() << " targets, "
                  << source_rank_to_idx_.size() << " sources" << std::endl;

        initialized_ = true;

        std::cout << "[Rank " << rank_ << "] All connections finalized" << std::endl;
    }

    // ---- Directed P2P methods (for hardware recovery) ----

    void send_to_rank(int target_rank, uintptr_t buffer_addr, size_t size) {
        /**
         * Send data to a single target rank over the existing connection.
         * Synchronous, blocking — does not use worker threads.
         */
        if (!initialized_) {
            throw std::runtime_error("GeminiReplicasNative not initialized");
        }
        auto it = target_rank_to_idx_.find(target_rank);
        if (it == target_rank_to_idx_.end()) {
            throw std::runtime_error("Target rank " + std::to_string(target_rank)
                + " not in target_ranks_");
        }
        const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer_addr);
        connection_manager_->send_to_one_target(it->second, data, size, rank_);
    }

    void recv_from_rank(int source_rank, uintptr_t buffer_addr, size_t expected_size) {
        /**
         * Receive data from a specific source rank over the existing connection.
         * Synchronous, blocking — does not use worker threads.
         *
         * ASIO: uses the pool-based receive_data() with source_rank verification
         *       (safe because hardware recovery has only one sender per receiver).
         * RDMA: uses the indexed receive_data_from_source().
         */
        if (!initialized_) {
            throw std::runtime_error("GeminiReplicasNative not initialized");
        }
        uint8_t* buffer = reinterpret_cast<uint8_t*>(buffer_addr);
        if (use_rdma_) {
            // RDMA: source_rank→QP index lookup + directed recv
            auto result = connection_manager_->receive_data_from_source(
                source_rank, buffer, expected_size);
            if (result.second != expected_size) {
                throw std::runtime_error("Size mismatch: expected "
                    + std::to_string(expected_size) + " got "
                    + std::to_string(result.second));
            }
        } else {
            // ASIO: pool-based recv with source_rank verification
            auto result = connection_manager_->receive_data(buffer, expected_size);
            if (result.first != source_rank) {
                throw std::runtime_error("Expected data from source "
                    + std::to_string(source_rank) + " but got from "
                    + std::to_string(result.first));
            }
            if (result.second != expected_size) {
                throw std::runtime_error("Size mismatch: expected "
                    + std::to_string(expected_size) + " got "
                    + std::to_string(result.second));
            }
        }
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
    bool exchange_all_done_or_error() const {
        if (send_error_.load()) return true;
        for (size_t i = 0; i < recv_source_ranks_.size(); ++i) {
            if (recv_error_[i].load()) return true;
        }
        if (!send_done_.load()) return false;
        for (size_t i = 0; i < recv_source_ranks_.size(); ++i) {
            if (!recv_done_[i].load()) return false;
        }
        return true;
    }

    void notify_exchange_waiters() {
        exchange_wait_cv_.notify_all();
    }

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
            notify_exchange_waiters();
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
            notify_exchange_waiters();
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

        {
            std::lock_guard<std::mutex> lk(send_mutex_);
            send_task_addr_ = buffer_addr;
            send_task_size_ = buffer_size;
            send_ready_ = true;
            send_done_ = false;
            send_error_ = false;
            send_cv_.notify_one();
        }

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

        {
            std::lock_guard<std::mutex> lk(*recv_mutexes_[idx]);
            recv_task_addrs_[idx] = buffer_addr;
            recv_task_sizes_[idx] = buffer_size;
            recv_ready_[idx] = true;
            recv_done_[idx] = false;
            recv_error_[idx] = false;
            recv_cvs_[idx]->notify_one();
        }

        std::cout << "[Rank " << rank_ << "] Submitted recv buffer for source rank "
                  << source_rank << ": " << buffer_size << " bytes" << std::endl;
    }

    void wait_for_exchange_completion() {
        std::unique_lock<std::mutex> lk(exchange_wait_mutex_);
        exchange_wait_cv_.wait(lk, [this] {
            return exchange_all_done_or_error();
        });

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
        .def("stop_workers", &GeminiReplicasNative::stop_workers,
             "Stop all worker threads gracefully (for reinit during recovery)")
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
             "Unregister buffer for RDMA operations (RDMA mode only)")
        .def("send_to_rank", &GeminiReplicasNative::send_to_rank,
             py::arg("target_rank"), py::arg("buffer_addr"), py::arg("size"),
             "Directed P2P send to a single target rank for hardware recovery (synchronous, blocking)")
        .def("recv_from_rank", &GeminiReplicasNative::recv_from_rank,
             py::arg("source_rank"), py::arg("buffer_addr"), py::arg("expected_size"),
             "Directed P2P recv from a specific source rank for hardware recovery (synchronous, blocking)");
}

