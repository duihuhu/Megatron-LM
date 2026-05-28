// Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

/**
 * Gemini Native C++ Module
 * 
 * This module provides ASIO-based network communication for Gemini replica-level
 * data transfer between paired ranks (0<->2, 1<->3).
 * 
 * Key features:
 * - Single bidirectional connection per rank pair
 * - Asynchronous data transfer using Boost.ASIO
 * - Direct buffer transfer without serialization
 * - Thread-safe operation
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <boost/asio.hpp>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// RDMA headers (ibverbs)
#include <infiniband/verbs.h>

#include "rdma_device_utils.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include <map>
#include <sstream>

namespace py = pybind11;

namespace {

// ============================================================================
// Connection Manager Interface
// ============================================================================

class IConnectionManager {
public:
    virtual ~IConnectionManager() = default;
    
    virtual void initialize_connections() = 0;
    virtual void connect_and_wait() = 0;
    virtual void send_data(const uint8_t* data, size_t size) = 0;
    virtual size_t receive_data(uint8_t* buffer, size_t buffer_size) = 0;
    virtual bool is_connected() const = 0;
    
    // RDMA-specific methods (no-op for ASIO)
    virtual void register_buffer(uintptr_t addr, size_t size) {}
    virtual void unregister_buffer(uintptr_t addr) {}
};

// ============================================================================
// RDMA Connection Info (for ibverbs)
// ============================================================================

// RDMA connection info exchanged via TCP (same as rdma_throughput_test.cpp)
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

// ============================================================================
// Gemini ASIO Connection Manager
// ============================================================================

class GeminiAsioConnectionManager : public IConnectionManager {
private:
    boost::asio::io_context io_context_;
    
    // Send and receive sockets for paired rank communication
    boost::asio::ip::tcp::socket send_socket_;
    boost::asio::ip::tcp::socket recv_socket_;
    boost::asio::ip::tcp::acceptor recv_acceptor_;
    
    // Connection status flags
    std::atomic<bool> send_connected_{false};
    std::atomic<bool> recv_connected_{false};
    
    // Synchronization
    std::mutex connection_mutex_;
    std::condition_variable connection_cv_;
    
    // Rank information
    int rank_;
    int world_size_;
    int partner_rank_;
    
    // Network configuration
    std::string partner_ip_;
    int partner_port_;
    std::string my_ip_;
    int my_port_;

public:
    GeminiAsioConnectionManager(
        int rank, int world_size, int partner_rank,
        const std::string& partner_ip, int partner_port,
        const std::string& my_ip, int my_port
    )
        : send_socket_(io_context_),
          recv_socket_(io_context_),
          recv_acceptor_(io_context_),
          rank_(rank),
          world_size_(world_size),
          partner_rank_(partner_rank),
          partner_ip_(partner_ip),
          partner_port_(partner_port),
          my_ip_(my_ip),
          my_port_(my_port)
    {
        std::cout << "[Gemini] Rank " << rank_ << " initializing ASIO connection manager" << std::endl;
        std::cout << "[Gemini] Partner: Rank " << partner_rank_ 
                  << " at " << partner_ip_ << ":" << partner_port_ << std::endl;
        std::cout << "[Gemini] My listen address: " << my_ip_ << ":" << my_port_ << std::endl;
    }
    
    ~GeminiAsioConnectionManager() {
        cleanup();
    }
    
    void initialize_connections() override {
        std::cout << "[Gemini ASIO] Rank " << rank_ << " starting connection initialization..." << std::endl;
        
        // Start acceptor for receiving connections
        start_acceptor();
        
        std::cout << "[Gemini ASIO] Rank " << rank_ << " acceptor started, ready for phase 2" << std::endl;
    }
    
    void connect_and_wait() override {
        std::cout << "[Gemini ASIO] Rank " << rank_ << " starting phase 2: connecting to partner..." << std::endl;
        
        // Connect to partner (with retry logic)
        connect_to_partner();
        
        // Wait for all connections to be established
        wait_for_connections();
        
        std::cout << "[Gemini ASIO] Rank " << rank_ << " all connections established successfully" << std::endl;
    }
    
    // Send data to partner rank
    void send_data(const uint8_t* data, size_t size) override {
        if (!send_connected_) {
            throw std::runtime_error("Send socket not connected");
        }
        
        try {
            std::cout << "[Gemini ASIO] Rank " << rank_ << " sending " << size << " bytes via ASIO to rank " 
                      << partner_rank_ << std::endl;
            // Send size first (8 bytes)
            uint64_t size_network = htobe64(size);
            boost::asio::write(send_socket_, boost::asio::buffer(&size_network, sizeof(size_network)));
            
            // Send data
            boost::asio::write(send_socket_, boost::asio::buffer(data, size));
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to send data: ") + e.what());
        }
    }
    
    // Receive data from partner rank
    size_t receive_data(uint8_t* buffer, size_t buffer_size) override {
        if (!recv_connected_) {
            throw std::runtime_error("Receive socket not connected");
        }
        
        try {
            std::cout << "[Gemini ASIO] Rank " << rank_ << " receiving via ASIO from rank " 
                      << partner_rank_ << " (buffer_size=" << buffer_size << ")" << std::endl;
            // Receive size first (8 bytes)
            uint64_t size_network;
            boost::asio::read(recv_socket_, boost::asio::buffer(&size_network, sizeof(size_network)));
            size_t size = be64toh(size_network);
            
            if (size > buffer_size) {
                throw std::runtime_error("Received data size exceeds buffer size");
            }
            
            // Receive data
            boost::asio::read(recv_socket_, boost::asio::buffer(buffer, size));
            
            return size;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to receive data: ") + e.what());
        }
    }
    
    bool is_connected() const override {
        return send_connected_ && recv_connected_;
    }

private:
    void start_acceptor() {
        try {
            boost::asio::ip::tcp::endpoint endpoint(
                boost::asio::ip::address::from_string(my_ip_),
                my_port_
            );
            
            recv_acceptor_.open(endpoint.protocol());
            recv_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
            recv_acceptor_.bind(endpoint);
            recv_acceptor_.listen();
            
            std::cout << "[Gemini] Rank " << rank_ << " acceptor listening on " 
                      << my_ip_ << ":" << my_port_ << std::endl;
            
            // Start async accept
            recv_acceptor_.async_accept(recv_socket_,
                [this](const boost::system::error_code& error) {
                    if (!error) {
                        std::cout << "[Gemini] Rank " << rank_ << " accepted connection from partner" << std::endl;
                        recv_connected_ = true;
                        connection_cv_.notify_all();
                    } else {
                        std::cerr << "[Gemini] Rank " << rank_ << " accept failed: " << error.message() << std::endl;
                    }
                });
            
            // Run io_context in a separate thread
            std::thread([this]() {
                io_context_.run();
            }).detach();
            
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to start acceptor: ") + e.what());
        }
    }
    
    void connect_to_partner() {
        const int max_retries = 100;
        const int retry_delay_ms = 100;
        
        for (int attempt = 0; attempt < max_retries; ++attempt) {
            try {
                boost::asio::ip::tcp::endpoint endpoint(
                    boost::asio::ip::address::from_string(partner_ip_),
                    partner_port_
                );
                
                send_socket_.connect(endpoint);
                send_connected_ = true;
                
                std::cout << "[Gemini] Rank " << rank_ << " connected to partner rank " 
                          << partner_rank_ << " at " << partner_ip_ << ":" << partner_port_ << std::endl;
                
                connection_cv_.notify_all();
                return;
                
            } catch (const std::exception& e) {
                if (attempt < max_retries - 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
                } else {
                    throw std::runtime_error(
                        std::string("Failed to connect to partner after ") + 
                        std::to_string(max_retries) + " attempts: " + e.what()
                    );
                }
            }
        }
    }
    
    void wait_for_connections() {
        std::unique_lock<std::mutex> lock(connection_mutex_);
        connection_cv_.wait(lock, [this]() {
            return send_connected_ && recv_connected_;
        });
        
        std::cout << "[Gemini] Rank " << rank_ << " all connections ready" << std::endl;
    }
    
    void cleanup() {
        try {
            if (send_socket_.is_open()) {
                send_socket_.close();
            }
            if (recv_socket_.is_open()) {
                recv_socket_.close();
            }
            if (recv_acceptor_.is_open()) {
                recv_acceptor_.close();
            }
            io_context_.stop();
        } catch (...) {
            // Ignore cleanup errors
        }
    }
};

// ============================================================================
// Gemini RDMA Connection Manager (ibverbs, no RDMA CM)
// ============================================================================

class GeminiRdmaConnectionManager : public IConnectionManager {
private:
    // ibverbs resources
    ibv_context* context_;
    ibv_pd* pd_;
    ibv_cq* send_cq_;
    ibv_cq* recv_cq_;
    ibv_qp* qp_;
    
    // TCP sockets for connection setup and control
    int listen_sock_;
    int control_sock_send_;  // Socket for sending control messages (size, recv ACK)
    int control_sock_recv_;  // Socket for receiving control messages (recv size, send ACK)
    
    // Connection status
    std::atomic<bool> connected_{false};
    
    // Synchronization
    std::mutex connection_mutex_;
    std::condition_variable connection_cv_;
    std::mutex buffer_mutex_;
    
    // Rank information
    int rank_;
    int world_size_;
    int partner_rank_;
    
    // Network configuration
    std::string partner_ip_;
    int partner_port_;
    std::string my_ip_;
    int my_port_;
    
    // Registered buffers
    std::map<uintptr_t, RdmaBuffer> registered_buffers_;
    
    // Temporary work request buffers (for unregistered data)
    std::vector<uint8_t> temp_send_buffer_;
    std::vector<uint8_t> temp_recv_buffer_;
    ibv_mr* temp_send_mr_;
    ibv_mr* temp_recv_mr_;
    
    static const size_t TEMP_BUFFER_SIZE = 2ULL * 1024 * 1024 * 1024; // 2 GB (fallback for unregistered buffers)
    static const int MAX_WR = 64;       // QP capacity (increased for better performance)
    static const int MAX_SGE = 1;
    static const size_t CHUNK_SIZE = 64 * 1024 * 1024;  // 64 MB per RDMA operation (safe for most hardware)
    static const int MAX_BATCH_WR = 32; // Max work requests per batch (increased for fewer batches)

public:
    GeminiRdmaConnectionManager(
        int rank, int world_size, int partner_rank,
        const std::string& partner_ip, int partner_port,
        const std::string& my_ip, int my_port
    )
        : context_(nullptr),
          pd_(nullptr),
          send_cq_(nullptr),
          recv_cq_(nullptr),
          qp_(nullptr),
          listen_sock_(-1),
          control_sock_send_(-1),
          control_sock_recv_(-1),
          temp_send_mr_(nullptr),
          temp_recv_mr_(nullptr),
          rank_(rank),
          world_size_(world_size),
          partner_rank_(partner_rank),
          partner_ip_(partner_ip),
          partner_port_(partner_port),
          my_ip_(my_ip),
          my_port_(my_port)
    {
        std::cout << "[Gemini RDMA] Rank " << rank_ << " initializing RDMA connection manager (ibverbs)" << std::endl;
        std::cout << "[Gemini RDMA] Partner: Rank " << partner_rank_ 
                  << " at " << partner_ip_ << ":" << partner_port_ << std::endl;
        std::cout << "[Gemini RDMA] My listen address: " << my_ip_ << ":" << my_port_ << std::endl;
        
        // Initialize temporary buffers
        temp_send_buffer_.resize(TEMP_BUFFER_SIZE);
        temp_recv_buffer_.resize(TEMP_BUFFER_SIZE);
    }
    
    ~GeminiRdmaConnectionManager() {
        cleanup();
    }
    
    void initialize_connections() override {
        std::cout << "[Gemini RDMA] Rank " << rank_ << " starting RDMA initialization (ibverbs)..." << std::endl;
        
        // Step 1: Initialize RDMA device and resources
        init_rdma_resources();
        
        // Step 2: Start TCP listener for connection info exchange
        start_tcp_listener();
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " RDMA initialization complete" << std::endl;
    }
    
    void connect_and_wait() override {
        std::cout << "[Gemini RDMA] Rank " << rank_ << " connecting to partner..." << std::endl;
        
        // Only the rank with lower rank number initiates connection
        // This avoids both ranks trying to connect simultaneously
        if (rank_ < partner_rank_) {
            std::cout << "[Gemini RDMA] Rank " << rank_ << " will connect to partner (rank " 
                      << partner_rank_ << ")" << std::endl;
            connect_to_partner_tcp();
        } else {
            std::cout << "[Gemini RDMA] Rank " << rank_ << " waiting for connection from partner (rank " 
                      << partner_rank_ << ")" << std::endl;
        }
        
        // Wait for connection to be ready (either from connect or accept thread)
        std::unique_lock<std::mutex> lock(connection_mutex_);
        connection_cv_.wait(lock, [this] { return connected_.load(); });
        lock.unlock();
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " RDMA connection established" << std::endl;
        
        // Warmup: Send/receive small test messages to initialize RDMA path
        warmup_rdma_connection();
    }
    
    void register_buffer(uintptr_t addr, size_t size) override {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        // Check if already registered
        if (registered_buffers_.find(addr) != registered_buffers_.end()) {
            std::cout << "[Gemini RDMA] Rank " << rank_ << " buffer already registered at " 
                      << std::hex << "0x" << addr << std::dec 
                      << " (size: " << (size / (1024.0 * 1024.0)) << " MB)" << std::endl;
            return;
        }
        
        if (!pd_) {
            throw std::runtime_error("Protection domain not initialized");
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " registering buffer at " 
                  << std::hex << "0x" << addr << std::dec 
                  << ", size: " << (size / (1024.0 * 1024.0)) << " MB (" << size << " bytes)" << std::endl;
        
        ibv_mr* mr = ibv_reg_mr(pd_, (void*)addr, size,
                                IBV_ACCESS_LOCAL_WRITE | 
                                IBV_ACCESS_REMOTE_WRITE | 
                                IBV_ACCESS_REMOTE_READ);
        
        if (!mr) {
            throw std::runtime_error("Failed to register memory region");
        }
        
        registered_buffers_[addr] = {mr, addr, size};
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " buffer registered successfully "
                  << "(total registered: " << registered_buffers_.size() << ")" << std::endl;
    }
    
    void unregister_buffer(uintptr_t addr) override {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        auto it = registered_buffers_.find(addr);
        if (it != registered_buffers_.end()) {
            ibv_dereg_mr(it->second.mr);
            registered_buffers_.erase(it);
            std::cout << "[Gemini RDMA] Rank " << rank_ << " buffer unregistered at " 
                      << std::hex << addr << std::dec << std::endl;
        }
    }
    
private:
    // Helper: Send data in chunks to avoid exceeding RDMA message size limits
    // Optimized version: batch post send work requests in groups, then poll completions
    void send_data_chunked(const uint8_t* data, size_t total_size, ibv_mr* mr) {
        auto start_time = std::chrono::steady_clock::now();
        
        size_t offset = 0;
        size_t chunk_count = (total_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        size_t batch_count = (chunk_count + MAX_BATCH_WR - 1) / MAX_BATCH_WR;
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " sending " << total_size 
                  << " bytes (" << (total_size / (1024.0 * 1024.0)) << " MB) in " 
                  << chunk_count << " chunks, " << batch_count << " batches" << std::endl;
        std::cout << "[Gemini RDMA] Config: MAX_WR=" << MAX_WR << ", MAX_BATCH_WR=" << MAX_BATCH_WR 
                  << ", CHUNK_SIZE=" << (CHUNK_SIZE / (1024.0 * 1024.0)) << " MB" << std::endl;
        
        // Prepare all SGEs and work requests
        std::vector<ibv_sge> sges(chunk_count);
        std::vector<ibv_send_wr> wrs(chunk_count);
        
        size_t chunk_idx = 0;
        while (offset < total_size) {
            size_t chunk_size = std::min(CHUNK_SIZE, total_size - offset);
            
            // Prepare SGE for this chunk
            sges[chunk_idx].addr = reinterpret_cast<uint64_t>(data + offset);
            sges[chunk_idx].length = chunk_size;
            sges[chunk_idx].lkey = mr->lkey;
            
            // Prepare send work request
            memset(&wrs[chunk_idx], 0, sizeof(ibv_send_wr));
            wrs[chunk_idx].wr_id = reinterpret_cast<uint64_t>(this) + offset;
            wrs[chunk_idx].sg_list = &sges[chunk_idx];
            wrs[chunk_idx].num_sge = 1;
            wrs[chunk_idx].opcode = IBV_WR_SEND;
            wrs[chunk_idx].send_flags = IBV_SEND_SIGNALED;
            wrs[chunk_idx].next = nullptr;  // Will be set when chaining
            
            offset += chunk_size;
            chunk_idx++;
        }
        
        // Post work requests in batches with pipelined completion polling
        // Strategy: post batch -> wait for completion -> post next batch
        // This ensures QP capacity is not exceeded
        size_t total_posted = 0;
        
        while (total_posted < chunk_count) {
            size_t batch_size = std::min(static_cast<size_t>(MAX_BATCH_WR), chunk_count - total_posted);
            
            // Chain work requests in this batch
            for (size_t i = total_posted; i < total_posted + batch_size - 1; ++i) {
                wrs[i].next = &wrs[i + 1];
            }
            wrs[total_posted + batch_size - 1].next = nullptr;
            
            std::cout << "[Gemini RDMA] Rank " << rank_ << " posting batch " << (total_posted / MAX_BATCH_WR + 1)
                      << ": " << batch_size << " send work requests (total: " << (total_posted + batch_size) 
                      << "/" << chunk_count << ")" << std::endl;
            
            // Post this batch
            ibv_send_wr* bad_wr;
            int post_ret = ibv_post_send(qp_, &wrs[total_posted], &bad_wr);
            if (post_ret != 0) {
                std::cerr << "[Gemini RDMA] ibv_post_send batch failed at offset " << total_posted 
                          << " with error: " << post_ret << " (" << strerror(post_ret) << ")" << std::endl;
                throw std::runtime_error("Failed to post RDMA send batch");
            }
            
            // Wait for this batch to complete before posting next batch
            // This ensures QP capacity is freed up
            std::cout << "[Gemini RDMA] Rank " << rank_ << " waiting for batch " << (total_posted / MAX_BATCH_WR + 1)
                      << " completions (" << batch_size << " WRs)..." << std::endl;
            
            try {
                poll_completion(send_cq_, batch_size);
            } catch (const std::exception& e) {
                std::cerr << "[Gemini RDMA] Batch " << (total_posted / MAX_BATCH_WR + 1) 
                          << " send completion failed: " << e.what() << std::endl;
                throw;
            }
            
            total_posted += batch_size;
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double throughput_gbps = (total_size * 8.0) / (duration_ms * 1e6);  // Gbps
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " all " << chunk_count << " chunks sent successfully in " 
                  << duration_ms << " ms (" << throughput_gbps << " Gbps)" << std::endl;
    }
    
    // Helper: Receive data in chunks
    // Optimized version: pre-post recv work requests in groups, then poll completions
    void receive_data_chunked(uint8_t* buffer, size_t total_size, ibv_mr* mr, bool use_temp, size_t already_posted = 0) {
        auto start_time = std::chrono::steady_clock::now();
        
        size_t offset = 0;
        size_t chunk_count = (total_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        size_t batch_count = (chunk_count + MAX_BATCH_WR - 1) / MAX_BATCH_WR;
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " receiving " << total_size 
                  << " bytes (" << (total_size / (1024.0 * 1024.0)) << " MB) in " 
                  << chunk_count << " chunks, " << batch_count << " batches" << std::endl;
        
        // Post remaining work requests if any, then poll all completions
        size_t remaining_chunks = chunk_count - already_posted;
        size_t total_posted = already_posted;
        
        if (remaining_chunks > 0) {
            std::vector<ibv_sge> sges(remaining_chunks);
            std::vector<ibv_recv_wr> wrs(remaining_chunks);
            
            // Calculate offset for remaining chunks
            offset = already_posted * CHUNK_SIZE;
            size_t chunk_idx = 0;
            while (offset < total_size && chunk_idx < remaining_chunks) {
                size_t chunk_size = std::min(CHUNK_SIZE, total_size - offset);
                
                // Prepare SGE for this chunk
                sges[chunk_idx].addr = use_temp ? reinterpret_cast<uint64_t>(temp_recv_buffer_.data() + offset)
                                                : reinterpret_cast<uint64_t>(buffer + offset);
                sges[chunk_idx].length = chunk_size;
                sges[chunk_idx].lkey = mr->lkey;
                
                // Prepare receive work request
                memset(&wrs[chunk_idx], 0, sizeof(ibv_recv_wr));
                wrs[chunk_idx].wr_id = reinterpret_cast<uint64_t>(this) + offset;
                wrs[chunk_idx].sg_list = &sges[chunk_idx];
                wrs[chunk_idx].num_sge = 1;
                wrs[chunk_idx].next = nullptr;  // Will be set when chaining
                
                offset += chunk_size;
                chunk_idx++;
            }
            
            // Post remaining work requests in batches
            while (total_posted < chunk_count) {
                size_t batch_size = std::min(static_cast<size_t>(MAX_BATCH_WR), chunk_count - total_posted);
                size_t batch_start_idx = total_posted - already_posted;
                
                // Chain work requests in this batch
                for (size_t i = batch_start_idx; i < batch_start_idx + batch_size - 1; ++i) {
                    wrs[i].next = &wrs[i + 1];
                }
                wrs[batch_start_idx + batch_size - 1].next = nullptr;
                
                std::cout << "[Gemini RDMA] Rank " << rank_ << " posting batch " << (total_posted / MAX_BATCH_WR + 1)
                          << ": " << batch_size << " recv work requests (total: " << (total_posted + batch_size) 
                          << "/" << chunk_count << ")" << std::endl;
                
                // Post this batch
                ibv_recv_wr* bad_wr;
                int post_ret = ibv_post_recv(qp_, &wrs[batch_start_idx], &bad_wr);
                if (post_ret != 0) {
                    std::cerr << "[Gemini RDMA] ibv_post_recv batch failed at offset " << total_posted 
                              << " with error: " << post_ret << " (" << strerror(post_ret) << ")" << std::endl;
                    throw std::runtime_error("Failed to post RDMA receive batch");
                }
                
                total_posted += batch_size;
            }
        } else {
            std::cout << "[Gemini RDMA] Rank " << rank_ << " all recv WRs already posted, polling completions..." << std::endl;
        }
        
        // Poll completions for all chunks (including already posted first batch)
        total_posted = 0;
        while (total_posted < chunk_count) {
            size_t batch_size = std::min(static_cast<size_t>(MAX_BATCH_WR), chunk_count - total_posted);
            
            std::cout << "[Gemini RDMA] Rank " << rank_ << " waiting for batch " << (total_posted / MAX_BATCH_WR + 1)
                      << " completions (" << batch_size << " WRs)..." << std::endl;
            
            try {
                poll_completion(recv_cq_, batch_size);
            } catch (const std::exception& e) {
                std::cerr << "[Gemini RDMA] Batch " << (total_posted / MAX_BATCH_WR + 1) 
                          << " receive completion failed: " << e.what() << std::endl;
                throw;
            }
            
            total_posted += batch_size;
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double throughput_gbps = (total_size * 8.0) / (duration_ms * 1e6);  // Gbps
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " all " << chunk_count << " chunks received successfully in " 
                  << duration_ms << " ms (" << throughput_gbps << " Gbps)" << std::endl;
        
        // Copy from temporary buffer if needed
        if (use_temp) {
            std::memcpy(buffer, temp_recv_buffer_.data(), total_size);
        }
    }

public:
    void send_data(const uint8_t* data, size_t size) override {
        if (!connected_) {
            throw std::runtime_error("RDMA connection not established");
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " sending " << size 
                  << " bytes via RDMA SEND" << std::endl;
        
        // Step 1: Send size via dedicated send control socket
        uint64_t size_network = htobe64(size);
        if (send(control_sock_send_, &size_network, sizeof(size_network), 0) != sizeof(size_network)) {
            throw std::runtime_error("Failed to send size via control_sock_send");
        }
        
        // Step 2: Wait for receiver's immediate ACK (receiver sends ACK after receiving size, before posting recv WRs)
        // This avoids deadlock in exchange_buffers where both ranks try to send and receive simultaneously
        uint8_t ack;
        if (recv(control_sock_send_, &ack, sizeof(ack), MSG_WAITALL) != sizeof(ack)) {
            throw std::runtime_error("Failed to receive ACK from receiver");
        }
        std::cout << "[Gemini RDMA] Rank " << rank_ << " received immediate ACK, receiver will post recv WRs" << std::endl;
        
        // Step 3: Check if buffer is registered
        uintptr_t addr = reinterpret_cast<uintptr_t>(data);
        ibv_mr* mr = find_registered_mr(addr, size);
        
        // If not registered, use temporary buffer (normal for small data like metadata)
        if (!mr) {
            std::cout << "[Gemini RDMA] Rank " << rank_ << " send buffer NOT registered - using temp buffer (" 
                      << size << " bytes)" << std::endl;
            
            if (size > TEMP_BUFFER_SIZE) {
                throw std::runtime_error("Data size exceeds temporary buffer size");
            }
            
            // Copy to temporary buffer (already registered during initialization)
            std::memcpy(temp_send_buffer_.data(), data, size);
            mr = temp_send_mr_;
            data = temp_send_buffer_.data();
        } else {
            std::cout << "[Gemini RDMA] Rank " << rank_ << " send buffer IS registered" << std::endl;
        }
        
        // Send data in chunks to avoid RDMA message size limits
        send_data_chunked(data, size, mr);
    }
    
    size_t receive_data(uint8_t* buffer, size_t buffer_size) override {
        if (!connected_) {
            throw std::runtime_error("RDMA connection not established");
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " receiving via RDMA RECV" << std::endl;
        
        // Receive size via dedicated recv control socket
        uint64_t size_network;
        if (recv(control_sock_recv_, &size_network, sizeof(size_network), MSG_WAITALL) != sizeof(size_network)) {
            throw std::runtime_error("Failed to receive size via control_sock_recv");
        }
        size_t size = be64toh(size_network);
        
        if (size > buffer_size) {
            throw std::runtime_error("Received size exceeds buffer size");
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " expecting " << size << " bytes" << std::endl;
        
        // Check if buffer is registered
        uintptr_t addr = reinterpret_cast<uintptr_t>(buffer);
        ibv_mr* mr = find_registered_mr(addr, size);
        
        // If not registered, use temporary buffer (normal for small data like metadata)
        bool use_temp = false;
        if (!mr) {
            std::cout << "[Gemini RDMA] Rank " << rank_ << " recv buffer NOT registered - using temp buffer (" 
                      << size << " bytes)" << std::endl;
            
            if (size > TEMP_BUFFER_SIZE) {
                throw std::runtime_error("Data size exceeds temporary buffer size");
            }
            
            // Use temporary buffer (already registered during initialization)
            mr = temp_recv_mr_;
            use_temp = true;
        } else {
            std::cout << "[Gemini RDMA] Rank " << rank_ << " recv buffer IS registered" << std::endl;
        }
        
        // Step 1: Prepare and post all recv WRs BEFORE sending ACK
        // This ensures receiver is ready when sender receives ACK and posts send WRs
        size_t chunk_count = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        size_t first_batch_size = std::min(static_cast<size_t>(MAX_BATCH_WR), chunk_count);
        
        // Prepare SGEs and work requests for first batch
        std::vector<ibv_sge> first_batch_sges(first_batch_size);
        std::vector<ibv_recv_wr> first_batch_wrs(first_batch_size);
        
        size_t offset = 0;
        for (size_t i = 0; i < first_batch_size; ++i) {
            size_t chunk_size = std::min(CHUNK_SIZE, size - offset);
            
            first_batch_sges[i].addr = use_temp ? reinterpret_cast<uint64_t>(temp_recv_buffer_.data() + offset)
                                                : reinterpret_cast<uint64_t>(buffer + offset);
            first_batch_sges[i].length = chunk_size;
            first_batch_sges[i].lkey = mr->lkey;
            
            memset(&first_batch_wrs[i], 0, sizeof(ibv_recv_wr));
            first_batch_wrs[i].wr_id = reinterpret_cast<uint64_t>(this) + offset;
            first_batch_wrs[i].sg_list = &first_batch_sges[i];
            first_batch_wrs[i].num_sge = 1;
            first_batch_wrs[i].next = (i < first_batch_size - 1) ? &first_batch_wrs[i + 1] : nullptr;
            
            offset += chunk_size;
        }
        first_batch_wrs[first_batch_size - 1].next = nullptr;
        
        // Post first batch of recv WRs
        std::cout << "[Gemini RDMA] Rank " << rank_ << " posting first batch of " << first_batch_size 
                  << " recv WRs before sending ACK..." << std::endl;
        ibv_recv_wr* bad_wr;
        int post_ret = ibv_post_recv(qp_, &first_batch_wrs[0], &bad_wr);
        if (post_ret != 0) {
            std::cerr << "[Gemini RDMA] ibv_post_recv first batch failed with error: " << post_ret 
                      << " (" << strerror(post_ret) << ")" << std::endl;
            throw std::runtime_error("Failed to post first batch of RDMA receive WRs");
        }
        std::cout << "[Gemini RDMA] Rank " << rank_ << " posted first batch of recv WRs" << std::endl;
        
        // Step 2: Send ACK AFTER posting recv WRs
        // This tells sender that receiver is ready to receive data
        uint8_t ack = 1;
        if (send(control_sock_recv_, &ack, sizeof(ack), 0) != sizeof(ack)) {
            throw std::runtime_error("Failed to send ACK to sender");
        }
        std::cout << "[Gemini RDMA] Rank " << rank_ << " sent ACK to sender (after posting recv WRs)" << std::endl;
        
        // Step 3: Receive data in chunks (will handle remaining batches and polling)
        receive_data_chunked(buffer, size, mr, use_temp, first_batch_size);
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " received total " << size << " bytes" << std::endl;
        
        return size;
    }
    
    bool is_connected() const override {
        return connected_;
    }

private:
    // Warmup RDMA connection by sending/receiving test messages
    void warmup_rdma_connection() {
        std::cout << "[Gemini RDMA] Rank " << rank_ << " starting RDMA warmup..." << std::endl;
        
        const size_t warmup_size = 1024;  // 1KB test message
        const int warmup_rounds = 3;      // Number of warmup rounds
        
        try {
            for (int round = 0; round < warmup_rounds; ++round) {
                if (rank_ < partner_rank_) {
                    // Lower rank sends first, then receives
                    send_data(temp_send_buffer_.data(), warmup_size);
                    receive_data(temp_recv_buffer_.data(), warmup_size);
                } else {
                    // Higher rank receives first, then sends
                    receive_data(temp_recv_buffer_.data(), warmup_size);
                    send_data(temp_send_buffer_.data(), warmup_size);
                }
            }
            
            std::cout << "[Gemini RDMA] Rank " << rank_ << " RDMA warmup completed (" 
                      << warmup_rounds << " rounds)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Gemini RDMA] Rank " << rank_ << " warmup failed: " << e.what() << std::endl;
            std::cerr << "[Gemini RDMA] Continuing without warmup..." << std::endl;
        }
    }
    
    // Initialize RDMA resources using ibverbs (same as rdma_throughput_test.cpp)
    void init_rdma_resources() {
        // Get device list
        int num_devices;
        ibv_device** device_list = ibv_get_device_list(&num_devices);
        if (!device_list || num_devices == 0) {
            throw std::runtime_error("Failed to get RDMA device list");
        }
        
        // Open first device
        context_ = ibv_open_device(find_rdma_device_by_ip(my_ip_, device_list, num_devices));
        ibv_free_device_list(device_list);
        
        if (!context_) {
            throw std::runtime_error("Failed to open RDMA device");
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " opened RDMA device: " 
                  << ibv_get_device_name(context_->device) << std::endl;
        
        // Allocate protection domain
        pd_ = ibv_alloc_pd(context_);
        if (!pd_) {
            throw std::runtime_error("Failed to allocate protection domain");
        }
        
        // Create completion queues
        send_cq_ = ibv_create_cq(context_, MAX_WR, nullptr, nullptr, 0);
        if (!send_cq_) {
            throw std::runtime_error("Failed to create send completion queue");
        }
        
        recv_cq_ = ibv_create_cq(context_, MAX_WR, nullptr, nullptr, 0);
        if (!recv_cq_) {
            throw std::runtime_error("Failed to create recv completion queue");
        }
        
        // Create Queue Pair
        ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = send_cq_;
        qp_init_attr.recv_cq = recv_cq_;
        qp_init_attr.qp_type = IBV_QPT_RC;  // Reliable Connection
        qp_init_attr.cap.max_send_wr = MAX_WR;
        qp_init_attr.cap.max_recv_wr = MAX_WR;
        qp_init_attr.cap.max_send_sge = MAX_SGE;
        qp_init_attr.cap.max_recv_sge = MAX_SGE;
        
        qp_ = ibv_create_qp(pd_, &qp_init_attr);
        if (!qp_) {
            throw std::runtime_error("Failed to create queue pair");
        }
        
        // Print actual QP capabilities (may be different from requested)
        std::cout << "[Gemini RDMA] Rank " << rank_ << " QP capabilities:" << std::endl;
        std::cout << "[Gemini RDMA]   max_send_wr: " << qp_init_attr.cap.max_send_wr << std::endl;
        std::cout << "[Gemini RDMA]   max_recv_wr: " << qp_init_attr.cap.max_recv_wr << std::endl;
        std::cout << "[Gemini RDMA]   max_send_sge: " << qp_init_attr.cap.max_send_sge << std::endl;
        std::cout << "[Gemini RDMA]   max_recv_sge: " << qp_init_attr.cap.max_recv_sge << std::endl;
        std::cout << "[Gemini RDMA]   max_inline_data: " << qp_init_attr.cap.max_inline_data << std::endl;
        
        // Transition QP to INIT state
        ibv_qp_attr qp_attr = {};
        qp_attr.qp_state = IBV_QPS_INIT;
        qp_attr.pkey_index = 0;
        qp_attr.port_num = 1;
        qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        
        if (ibv_modify_qp(qp_, &qp_attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            throw std::runtime_error("Failed to modify QP to INIT");
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " RDMA resources initialized (QP number: " 
                  << qp_->qp_num << ")" << std::endl;
        
        // Register temporary buffers for unregistered data (e.g., metadata)
        std::cout << "[Gemini RDMA] Rank " << rank_ << " registering temporary buffers (" 
                  << (TEMP_BUFFER_SIZE / (1024.0 * 1024.0)) << " MB each)..." << std::endl;
        
        temp_send_mr_ = ibv_reg_mr(pd_, temp_send_buffer_.data(), TEMP_BUFFER_SIZE,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!temp_send_mr_) {
            throw std::runtime_error("Failed to register temporary send buffer");
        }
        
        temp_recv_mr_ = ibv_reg_mr(pd_, temp_recv_buffer_.data(), TEMP_BUFFER_SIZE,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!temp_recv_mr_) {
            throw std::runtime_error("Failed to register temporary receive buffer");
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " temporary buffers registered successfully" << std::endl;
    }
    
    // Start TCP listener for connection info exchange
    void start_tcp_listener() {
        listen_sock_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_sock_ < 0) {
            throw std::runtime_error("Failed to create listen socket");
        }
        
        int opt = 1;
        setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        sockaddr_in server_addr = {};
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(my_port_);
        inet_pton(AF_INET, my_ip_.c_str(), &server_addr.sin_addr);
        
        if (bind(listen_sock_, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            throw std::runtime_error("Failed to bind listen socket");
        }
        
        if (listen(listen_sock_, 2) < 0) {  // Allow 2 connections
            throw std::runtime_error("Failed to listen on socket");
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " TCP listener started on " 
                  << my_ip_ << ":" << my_port_ << std::endl;
        
        // Start accept thread
        std::thread([this]() {
            accept_tcp_connection();
        }).detach();
    }
    
    // Accept TWO TCP connections: one for send control, one for recv control
    void accept_tcp_connection() {
        // Accept first connection (for RDMA info exchange and send control)
        sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int sock1 = accept(listen_sock_, (sockaddr*)&client_addr, &addr_len);
        
        if (sock1 < 0) {
            std::cerr << "[Gemini RDMA] Failed to accept first TCP connection" << std::endl;
            return;
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " accepted first TCP connection from partner" << std::endl;
        
        // Exchange RDMA connection info on first socket
        RdmaConnInfo local_info = get_local_conn_info();
        RdmaConnInfo remote_info;
        
        if (!exchange_conn_info(sock1, local_info, remote_info)) {
            std::cerr << "[Gemini RDMA] Failed to exchange connection info" << std::endl;
            close(sock1);
            return;
        }
        
        // Connect QP to remote peer
        if (!connect_qp(remote_info)) {
            std::cerr << "[Gemini RDMA] Failed to connect QP" << std::endl;
            close(sock1);
            return;
        }
        
        // Accept second connection (for recv control)
        addr_len = sizeof(client_addr);
        int sock2 = accept(listen_sock_, (sockaddr*)&client_addr, &addr_len);
        
        if (sock2 < 0) {
            std::cerr << "[Gemini RDMA] Failed to accept second TCP connection" << std::endl;
            close(sock1);
            return;
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " accepted second TCP connection from partner" << std::endl;
        
        // Assign sockets: first connection for recv control, second for send control
        control_sock_recv_ = sock1;
        control_sock_send_ = sock2;
        
        connected_ = true;
        connection_cv_.notify_all();
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " RDMA connection established (accepted, 2 sockets)" << std::endl;
    }
    
    // Connect to partner via TCP TWICE: one for send control, one for recv control
    void connect_to_partner_tcp() {
        sockaddr_in server_addr = {};
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(partner_port_);
        inet_pton(AF_INET, partner_ip_.c_str(), &server_addr.sin_addr);
        
        const int max_retries = 100;
        
        // Create and connect first socket (for RDMA info exchange and recv control)
        int sock1 = socket(AF_INET, SOCK_STREAM, 0);
        if (sock1 < 0) {
            throw std::runtime_error("Failed to create first socket");
        }
        
        for (int attempt = 0; attempt < max_retries; ++attempt) {
            if (connect(sock1, (sockaddr*)&server_addr, sizeof(server_addr)) == 0) {
                break;
            }
            if (attempt == max_retries - 1) {
                close(sock1);
                throw std::runtime_error("Failed to connect first socket to partner via TCP");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " connected first socket to partner via TCP" << std::endl;
        
        // Exchange RDMA connection info on first socket
        RdmaConnInfo local_info = get_local_conn_info();
        RdmaConnInfo remote_info;
        
        if (!exchange_conn_info(sock1, local_info, remote_info)) {
            close(sock1);
            throw std::runtime_error("Failed to exchange connection info");
        }
        
        // Connect QP to remote peer
        if (!connect_qp(remote_info)) {
            close(sock1);
            throw std::runtime_error("Failed to connect QP");
        }
        
        // Create and connect second socket (for send control)
        int sock2 = socket(AF_INET, SOCK_STREAM, 0);
        if (sock2 < 0) {
            close(sock1);
            throw std::runtime_error("Failed to create second socket");
        }
        
        for (int attempt = 0; attempt < max_retries; ++attempt) {
            if (connect(sock2, (sockaddr*)&server_addr, sizeof(server_addr)) == 0) {
                break;
            }
            if (attempt == max_retries - 1) {
                close(sock1);
                close(sock2);
                throw std::runtime_error("Failed to connect second socket to partner via TCP");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " connected second socket to partner via TCP" << std::endl;
        
        // Assign sockets: first connection for send control, second for recv control
        control_sock_send_ = sock1;
        control_sock_recv_ = sock2;
        
        connected_ = true;
        connection_cv_.notify_all();
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " RDMA connection established (connected, 2 sockets)" << std::endl;
    }
    
    // Get local connection info (same as rdma_throughput_test.cpp)
    RdmaConnInfo get_local_conn_info() {
        RdmaConnInfo info;
        std::memset(&info, 0, sizeof(info));
        
        ibv_port_attr port_attr;
        ibv_query_port(context_, 1, &port_attr);
        
        info.qp_num = qp_->qp_num;
        info.lid = port_attr.lid;
        
        // Try to get GID for RoCE
        ibv_gid gid;
        if (ibv_query_gid(context_, 1, 1, &gid) == 0) {
            std::memcpy(info.gid, &gid, 16);
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " local QP info: QP=" << info.qp_num 
                  << ", LID=" << info.lid << std::endl;
        
        return info;
    }
    
    // Exchange connection info via TCP (same as rdma_throughput_test.cpp)
    bool exchange_conn_info(int sock_fd, const RdmaConnInfo& local_info, RdmaConnInfo& remote_info) {
        // Send local info
        if (send(sock_fd, &local_info, sizeof(local_info), 0) != sizeof(local_info)) {
            std::cerr << "[Gemini RDMA] Failed to send connection info" << std::endl;
            return false;
        }
        
        // Receive remote info
        if (recv(sock_fd, &remote_info, sizeof(remote_info), MSG_WAITALL) != sizeof(remote_info)) {
            std::cerr << "[Gemini RDMA] Failed to receive connection info" << std::endl;
            return false;
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " exchanged connection info - Remote QP=" 
                  << remote_info.qp_num << ", LID=" << remote_info.lid << std::endl;
        return true;
    }
    
    // Connect QP to remote peer (same as rdma_throughput_test.cpp)
    bool connect_qp(const RdmaConnInfo& remote_info) {
        // Query port attributes to get active MTU
        ibv_port_attr port_attr;
        if (ibv_query_port(context_, 1, &port_attr) != 0) {
            std::cerr << "[Gemini RDMA] Failed to query port attributes" << std::endl;
            return false;
        }
        
        // Use the active MTU (what the port is currently using)
        ibv_mtu mtu = port_attr.active_mtu;
        std::cout << "[Gemini RDMA] Rank " << rank_ << " using MTU: " << mtu 
                  << " (1=" << IBV_MTU_256 << ", 2=" << IBV_MTU_512 << ", 3=" << IBV_MTU_1024 
                  << ", 4=" << IBV_MTU_2048 << ", 5=" << IBV_MTU_4096 << ")" << std::endl;
        
        // Transition to RTR (Ready to Receive)
        ibv_qp_attr qp_attr = {};
        qp_attr.qp_state = IBV_QPS_RTR;
        qp_attr.path_mtu = mtu;  // Use detected MTU instead of hardcoded 4096
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
            std::memcpy(&qp_attr.ah_attr.grh.dgid, remote_info.gid, 16);
            qp_attr.ah_attr.grh.flow_label = 0;
            qp_attr.ah_attr.grh.sgid_index = 1; // GID index 1 for erdma (RoCE v2)
            qp_attr.ah_attr.grh.hop_limit = 255;
            qp_attr.ah_attr.grh.traffic_class = 0;
        }
        
        int ret = ibv_modify_qp(qp_, &qp_attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        
        if (ret) {
            std::cerr << "[Gemini RDMA] Failed to modify QP to RTR: " << ret << std::endl;
            return false;
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " QP transitioned to RTR" << std::endl;
        
        // Transition to RTS (Ready to Send)
        std::memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.qp_state = IBV_QPS_RTS;
        qp_attr.sq_psn = 0;
        qp_attr.timeout = 14;
        qp_attr.retry_cnt = 7;
        qp_attr.rnr_retry = 7;
        qp_attr.max_rd_atomic = 1;
        
        if (ibv_modify_qp(qp_, &qp_attr,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
            std::cerr << "[Gemini RDMA] Failed to modify QP to RTS" << std::endl;
            return false;
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " QP transitioned to RTS - connection ready" << std::endl;
        return true;
    }
    
    // Find registered memory region containing the given address range
    ibv_mr* find_registered_mr(uintptr_t addr, size_t size) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        for (auto& pair : registered_buffers_) {
            const RdmaBuffer& buf = pair.second;
            if (addr >= buf.addr && (addr + size) <= (buf.addr + buf.size)) {
                std::cout << "[Gemini RDMA] Rank " << rank_ << " found registered MR: "
                          << "requested [" << std::hex << "0x" << addr << std::dec << ", size=" << size << "], "
                          << "registered [" << std::hex << "0x" << buf.addr << std::dec << ", size=" << buf.size << "]"
                          << std::endl;
                return buf.mr;
            }
        }
        
        // Not found - print all registered buffers for debugging
        std::cout << "[Gemini RDMA] Rank " << rank_ << " buffer NOT found. Registered buffers:" << std::endl;
        for (auto& pair : registered_buffers_) {
            const RdmaBuffer& buf = pair.second;
            std::cout << "[Gemini RDMA]   - " << std::hex << "0x" << buf.addr << std::dec 
                      << ", size=" << buf.size << " (" << (buf.size / (1024.0 * 1024.0)) << " MB)" << std::endl;
        }
        
        return nullptr;
    }
    
    // Send immediate value (size header)
    // Poll completion queue (same as rdma_throughput_test.cpp)
    void poll_completion(ibv_cq* cq, int num_completions) {
        int total_completed = 0;
        ibv_wc wc[16];  // Batch poll up to 16 completions
        
        while (total_completed < num_completions) {
            int n = ibv_poll_cq(cq, std::min(16, num_completions - total_completed), wc);
            if (n < 0) {
                throw std::runtime_error("Failed to poll CQ");
            }
            
            for (int i = 0; i < n; ++i) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    std::stringstream ss;
                    ss << "Work completion with error: " << ibv_wc_status_str(wc[i].status);
                    throw std::runtime_error(ss.str());
                }
                total_completed++;
            }
        }
    }
    
    // Cleanup resources
    void cleanup() {
        // Unregister all buffers
        for (auto& pair : registered_buffers_) {
            ibv_dereg_mr(pair.second.mr);
        }
        registered_buffers_.clear();
        
        // Destroy temporary MRs
        if (temp_send_mr_) {
            ibv_dereg_mr(temp_send_mr_);
            temp_send_mr_ = nullptr;
        }
        if (temp_recv_mr_) {
            ibv_dereg_mr(temp_recv_mr_);
            temp_recv_mr_ = nullptr;
        }
        
        // Destroy QP
        if (qp_) {
            ibv_destroy_qp(qp_);
            qp_ = nullptr;
        }
        
        // Destroy CQs
        if (send_cq_) {
            ibv_destroy_cq(send_cq_);
            send_cq_ = nullptr;
        }
        if (recv_cq_) {
            ibv_destroy_cq(recv_cq_);
            recv_cq_ = nullptr;
        }
        
        // Deallocate PD
        if (pd_) {
            ibv_dealloc_pd(pd_);
            pd_ = nullptr;
        }
        
        // Close device context
        if (context_) {
            ibv_close_device(context_);
            context_ = nullptr;
        }
        
        // Close sockets
        if (control_sock_send_ >= 0) {
            close(control_sock_send_);
            control_sock_send_ = -1;
        }
        if (control_sock_recv_ >= 0) {
            close(control_sock_recv_);
            control_sock_recv_ = -1;
        }
        if (listen_sock_ >= 0) {
            close(listen_sock_);
            listen_sock_ = -1;
        }
        
        std::cout << "[Gemini RDMA] Rank " << rank_ << " cleanup complete" << std::endl;
    }
};

// ============================================================================
// Gemini Native Class (Python Interface)
// ============================================================================

class GeminiNative {
private:
    int rank_;
    int world_size_;
    int partner_rank_;
    bool use_rdma_;
    
    std::unique_ptr<IConnectionManager> connection_manager_;
    
    std::atomic<bool> initialized_{false};

public:
    GeminiNative(
        int rank, int world_size, int partner_rank,
        const std::string& partner_ip, int partner_port,
        const std::string& my_ip, int my_port,
        bool use_rdma = false
    )
        : rank_(rank),
          world_size_(world_size),
          partner_rank_(partner_rank),
          use_rdma_(use_rdma)
    {
        const char* mode_str = use_rdma_ ? "RDMA" : "ASIO";
        std::cout << "[Gemini] Creating GeminiNative instance for rank " << rank_ 
                  << " (mode: " << mode_str << ")" << std::endl;
        std::cout << "[Gemini] Partner: rank " << partner_rank_ << std::endl;
        
        try {
            // Create connection manager based on mode
            if (use_rdma_) {
                connection_manager_ = std::make_unique<GeminiRdmaConnectionManager>(
                    rank_, world_size_, partner_rank_,
                    partner_ip, partner_port,
                    my_ip, my_port
                );
            } else {
                connection_manager_ = std::make_unique<GeminiAsioConnectionManager>(
                    rank_, world_size_, partner_rank_,
                    partner_ip, partner_port,
                    my_ip, my_port
                );
            }
            
            // Phase 1: Start acceptor only (non-blocking)
             std::cout << "[Gemini] Phase 1: Starting acceptor for rank " << rank_ << "..." << std::endl;
            connection_manager_->initialize_connections();
            
             std::cout << "[Gemini] GeminiNative instance created (acceptor ready) for rank " << rank_ << std::endl;
            std::cout << "[Gemini] Note: Call finalize_connections() after all ranks are ready" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "[Gemini] Failed to create GeminiNative: " << e.what() << std::endl;
            throw;
        }
    }
    
    ~GeminiNative() {
        std::cout << "[Gemini] Destroying GeminiNative instance for rank " << rank_ << std::endl;
    }
    
    // Finalize connections (Phase 2: connect to partner and wait)
    void finalize_connections() {
        if (initialized_) {
            std::cout << "[Gemini] Rank " << rank_ << " already initialized, skipping finalize" << std::endl;
            return;
        }
        
        try {
            std::cout << "[Gemini] Phase 2: Rank " << rank_ << " connecting to partner..." << std::endl;
            connection_manager_->connect_and_wait();
            initialized_ = true;
            std::cout << "[Gemini] Rank " << rank_ << " connections finalized successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Gemini] Failed to finalize connections for rank " << rank_ << ": " << e.what() << std::endl;
            throw;
        }
    }
    
    // Register buffer for RDMA operations (no-op for ASIO)
    void register_buffer(uintptr_t buffer_addr, size_t buffer_size) {
        if (buffer_addr == 0) {
            throw std::runtime_error("Invalid buffer address (null pointer)");
        }
        
        if (buffer_size == 0) {
            throw std::runtime_error("Invalid buffer size (zero)");
        }
        
            connection_manager_->register_buffer(buffer_addr, buffer_size);
    }
    
    // Unregister buffer (no-op for ASIO)
    void unregister_buffer(uintptr_t buffer_addr) {
        if (buffer_addr == 0) {
            throw std::runtime_error("Invalid buffer address (null pointer)");
        }
        
            connection_manager_->unregister_buffer(buffer_addr);
    }
    
    // Send buffer to partner rank (using raw memory address and size)
    void send_buffer(uintptr_t buffer_addr, size_t buffer_size) {
        if (!initialized_) {
            throw std::runtime_error("GeminiNative not initialized");
        }
        
        if (buffer_addr == 0) {
            throw std::runtime_error("Invalid buffer address (null pointer)");
        }
        
        if (buffer_size == 0) {
            throw std::runtime_error("Invalid buffer size (zero)");
        }
        
        const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer_addr);
        
        std::cout << "[Gemini] Rank " << rank_ << " sending buffer of size " 
                  << buffer_size << " bytes (" << (buffer_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
        
        connection_manager_->send_data(data, buffer_size);
    }
    
    // Receive buffer from partner rank (using raw memory address and size)
    size_t receive_buffer(uintptr_t buffer_addr, size_t buffer_size) {
        if (!initialized_) {
            throw std::runtime_error("GeminiNative not initialized");
        }
        
        if (buffer_addr == 0) {
            throw std::runtime_error("Invalid buffer address (null pointer)");
        }
        
        if (buffer_size == 0) {
            throw std::runtime_error("Invalid buffer size (zero)");
        }
        
        uint8_t* data = reinterpret_cast<uint8_t*>(buffer_addr);
        
        std::cout << "[Gemini] Rank " << rank_ << " receiving buffer (max size: " 
                  << buffer_size << " bytes)" << std::endl;
        
        size_t received_size = connection_manager_->receive_data(data, buffer_size);
        
        std::cout << "[Gemini] Rank " << rank_ << " received " << received_size 
                  << " bytes (" << (received_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
        
        return received_size;
    }
    
    // Exchange buffers with partner rank (send and receive simultaneously)
    size_t exchange_buffers(uintptr_t send_buffer_addr, size_t send_buffer_size,
                           uintptr_t recv_buffer_addr, size_t recv_buffer_size) {
        if (!initialized_) {
            throw std::runtime_error("GeminiNative not initialized");
        }
        
        std::cout << "[Gemini] Rank " << rank_ << " starting buffer exchange with rank " 
                  << partner_rank_ << std::endl;
        
        size_t received_size = 0;
        std::exception_ptr send_exception = nullptr;
        std::exception_ptr recv_exception = nullptr;
        
        // Launch send and receive in separate threads for simultaneous operation
        std::thread send_thread([this, send_buffer_addr, send_buffer_size, &send_exception]() {
            try {
                this->send_buffer(send_buffer_addr, send_buffer_size);
            } catch (...) {
                send_exception = std::current_exception();
            }
        });
        
        std::thread recv_thread([this, recv_buffer_addr, recv_buffer_size, &received_size, &recv_exception]() {
            try {
                received_size = this->receive_buffer(recv_buffer_addr, recv_buffer_size);
            } catch (...) {
                recv_exception = std::current_exception();
            }
        });
        
        // Wait for both operations to complete
        send_thread.join();
        recv_thread.join();
        
        // Re-throw any exceptions that occurred
        if (send_exception) {
            std::rethrow_exception(send_exception);
        }
        if (recv_exception) {
            std::rethrow_exception(recv_exception);
        }
        
        std::cout << "[Gemini] Rank " << rank_ << " buffer exchange completed" << std::endl;
        
        return received_size;
    }
    
    bool is_initialized() const {
        return initialized_;
    }
    
    int get_rank() const {
        return rank_;
    }
    
    int get_partner_rank() const {
        return partner_rank_;
    }
};

} // anonymous namespace

// ============================================================================
// Python Bindings
// ============================================================================

PYBIND11_MODULE(gemini_native, m) {
    m.doc() = "Gemini Native C++ Module for replica-level data transfer using ASIO or RDMA";
    
    // Static utility functions
    m.def("is_rdma_available", []() {
        // Check for RDMA devices using ibverbs
        int num_devices;
        ibv_device** device_list = ibv_get_device_list(&num_devices);
        if (!device_list || num_devices == 0) {
            if (device_list) {
                ibv_free_device_list(device_list);
            }
            return false;
        }
        ibv_free_device_list(device_list);
        return true;
    }, "Check if RDMA is available on the system");
    
    py::class_<GeminiNative>(m, "GeminiNative")
        .def(py::init<int, int, int, const std::string&, int, const std::string&, int, bool>(),
             py::arg("rank"),
             py::arg("world_size"),
             py::arg("partner_rank"),
             py::arg("partner_ip"),
             py::arg("partner_port"),
             py::arg("my_ip"),
             py::arg("my_port"),
             py::arg("use_rdma") = false,
             "Initialize Gemini native module with ASIO or RDMA transport.\n\n"
             "Args:\n"
             "    rank: Current rank\n"
             "    world_size: Total number of ranks\n"
             "    partner_rank: Paired rank for replica exchange\n"
             "    partner_ip: Partner's IP address\n"
             "    partner_port: Partner's receive port\n"
             "    my_ip: This rank's IP address\n"
             "    my_port: This rank's receive port\n"
             "    use_rdma: Use RDMA transport (default: False, uses ASIO)\n"
             "Note: Only starts listener. Call finalize_connections() after all ranks are ready.\n")
        .def("finalize_connections", &GeminiNative::finalize_connections,
             "Finalize connections to partner rank (Phase 2).\n"
             "Call this after all ranks have been initialized and are ready to connect.\n")
        .def("register_buffer", &GeminiNative::register_buffer,
             py::arg("buffer_addr"),
             py::arg("buffer_size"),
             "Register buffer for RDMA operations (no-op for ASIO).\n\n"
             "For RDMA: Register buffer during first allocation in save phase.\n"
             "Args:\n"
             "    buffer_addr: Memory address of the buffer (uintptr_t)\n"
             "    buffer_size: Size of the buffer in bytes\n")
        .def("unregister_buffer", &GeminiNative::unregister_buffer,
             py::arg("buffer_addr"),
             "Unregister buffer (no-op for ASIO).\n\n"
             "Args:\n"
             "    buffer_addr: Memory address of the buffer (uintptr_t)\n")
        .def("send_buffer", &GeminiNative::send_buffer,
             py::arg("buffer_addr"),
             py::arg("buffer_size"),
             "Send buffer to partner rank using raw memory address.\n\n"
             "Args:\n"
             "    buffer_addr: Memory address of the buffer (uintptr_t)\n"
             "    buffer_size: Size of the buffer in bytes\n")
        .def("receive_buffer", &GeminiNative::receive_buffer,
             py::arg("buffer_addr"),
             py::arg("buffer_size"),
             "Receive buffer from partner rank using raw memory address.\n\n"
             "Args:\n"
             "    buffer_addr: Memory address of the buffer (uintptr_t)\n"
             "    buffer_size: Size of the buffer in bytes\n"
             "Returns:\n"
             "    Number of bytes received\n")
        .def("exchange_buffers", &GeminiNative::exchange_buffers,
             py::arg("send_buffer_addr"),
             py::arg("send_buffer_size"),
             py::arg("recv_buffer_addr"),
             py::arg("recv_buffer_size"),
             "Exchange buffers with partner rank simultaneously using raw memory addresses.\n\n"
             "Args:\n"
             "    send_buffer_addr: Memory address of the send buffer (uintptr_t)\n"
             "    send_buffer_size: Size of the send buffer in bytes\n"
             "    recv_buffer_addr: Memory address of the receive buffer (uintptr_t)\n"
             "    recv_buffer_size: Size of the receive buffer in bytes\n"
             "Returns:\n"
             "    Number of bytes received\n")
        .def("is_initialized", &GeminiNative::is_initialized,
             "Check if the module is initialized")
        .def("get_rank", &GeminiNative::get_rank,
             "Get current rank")
        .def("get_partner_rank", &GeminiNative::get_partner_rank,
             "Get partner rank");
}