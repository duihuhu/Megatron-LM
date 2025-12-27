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

#include <atomic>
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

namespace py = pybind11;

namespace {

// ============================================================================
// Gemini ASIO Connection Manager
// ============================================================================

class GeminiAsioConnectionManager {
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
    
    void initialize_connections() {
        std::cout << "[Gemini] Rank " << rank_ << " starting connection initialization..." << std::endl;
        
        // Start acceptor for receiving connections
        start_acceptor();
        
        std::cout << "[Gemini] Rank " << rank_ << " acceptor started, ready for phase 2" << std::endl;
    }
    
    void connect_and_wait() {
        std::cout << "[Gemini] Rank " << rank_ << " starting phase 2: connecting to partner..." << std::endl;
        
        // Connect to partner (with retry logic)
        connect_to_partner();
        
        // Wait for all connections to be established
        wait_for_connections();
        
        std::cout << "[Gemini] Rank " << rank_ << " all connections established successfully" << std::endl;
    }
    
    // Send data to partner rank
    void send_data(const uint8_t* data, size_t size) {
        if (!send_connected_) {
            throw std::runtime_error("Send socket not connected");
        }
        
        try {
            // Send size first (8 bytes)
            uint64_t size_network = htobe64(size);
            boost::asio::write(send_socket_, boost::asio::buffer(&size_network, sizeof(size_network)));
            
            // Send data
            boost::asio::write(send_socket_, boost::asio::buffer(data, size));
            
            std::cout << "[Gemini] Rank " << rank_ << " sent " << size << " bytes to rank " 
                      << partner_rank_ << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to send data: ") + e.what());
        }
    }
    
    // Receive data from partner rank
    size_t receive_data(uint8_t* buffer, size_t buffer_size) {
        if (!recv_connected_) {
            throw std::runtime_error("Receive socket not connected");
        }
        
        try {
            // Receive size first (8 bytes)
            uint64_t size_network;
            boost::asio::read(recv_socket_, boost::asio::buffer(&size_network, sizeof(size_network)));
            size_t size = be64toh(size_network);
            
            if (size > buffer_size) {
                throw std::runtime_error("Received data size exceeds buffer size");
            }
            
            // Receive data
            boost::asio::read(recv_socket_, boost::asio::buffer(buffer, size));
            
            std::cout << "[Gemini] Rank " << rank_ << " received " << size << " bytes from rank " 
                      << partner_rank_ << std::endl;
            
            return size;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to receive data: ") + e.what());
        }
    }
    
    bool is_connected() const {
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
// Gemini Native Class (Python Interface)
// ============================================================================

class GeminiNative {
private:
    int rank_;
    int world_size_;
    int partner_rank_;
    
    std::unique_ptr<GeminiAsioConnectionManager> connection_manager_;
    
    std::atomic<bool> initialized_{false};

public:
    GeminiNative(
        int rank, int world_size, int partner_rank,
        const std::string& partner_ip, int partner_port,
        const std::string& my_ip, int my_port
    )
        : rank_(rank),
          world_size_(world_size),
          partner_rank_(partner_rank)
    {
        std::cout << "[Gemini] Creating GeminiNative instance for rank " << rank_ << std::endl;
        std::cout << "[Gemini] Partner: rank " << partner_rank_ << std::endl;
        
        try {
            // Create connection manager
            connection_manager_ = std::make_unique<GeminiAsioConnectionManager>(
                rank_, world_size_, partner_rank_,
                partner_ip, partner_port,
                my_ip, my_port
            );
            
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
    m.doc() = "Gemini Native C++ Module for replica-level data transfer using ASIO";
    
    py::class_<GeminiNative>(m, "GeminiNative")
        .def(py::init<int, int, int, const std::string&, int, const std::string&, int>(),
             py::arg("rank"),
             py::arg("world_size"),
             py::arg("partner_rank"),
             py::arg("partner_ip"),
             py::arg("partner_port"),
             py::arg("my_ip"),
             py::arg("my_port"),
             "Initialize Gemini native module with ASIO connections.\n\n"
             "Args:\n"
             "    rank: Current rank\n"
             "    world_size: Total number of ranks\n"
             "    partner_rank: Paired rank for replica exchange\n"
             "    partner_ip: Partner's IP address\n"
             "    partner_port: Partner's receive port\n"
             "    my_ip: This rank's IP address\n"
             "    my_port: This rank's receive port\n"
             "Note: Only starts acceptor. Call finalize_connections() after all ranks are ready.\n")
        .def("finalize_connections", &GeminiNative::finalize_connections,
             "Finalize connections to partner rank (Phase 2).\n"
             "Call this after all ranks have been initialized and are ready to connect.\n")
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

