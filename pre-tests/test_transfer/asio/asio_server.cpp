/**
 * ASIO Server - Latency Test Receiver
 * 
 * This server receives data packets from clients and measures latency.
 * It echoes back a timestamp to allow round-trip latency measurement.
 * 
 * Compile:
 *   g++ -std=c++17 -O3 asio_server.cpp -o asio_server -lboost_system -pthread
 * 
 * Usage:
 *   ./asio_server <port>
 */

#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cstring>
#include <thread>
#include <atomic>
#include <memory>

using boost::asio::ip::tcp;
namespace asio = boost::asio;

struct PacketHeader {
    uint64_t packet_id;
    uint64_t timestamp_ns;  // Client send timestamp in nanoseconds
    uint32_t data_size;     // Size of payload data in bytes
};

class ServerSession : public boost::enable_shared_from_this<ServerSession> {
public:
    ServerSession(tcp::socket socket, int session_id, size_t max_packet_size = 128 * 1024 * 1024)
        : socket_(std::move(socket)), session_id_(session_id),
          max_packet_size_(max_packet_size),
          packets_received_(0), total_latency_ns_(0) {
        // Set TCP_NODELAY to reduce latency
        socket_.set_option(tcp::no_delay(true));
        
        // Pre-allocate buffers to avoid memory allocation during transmission
        // Allocate for maximum expected packet size
        size_t max_total_size = sizeof(PacketHeader) + max_packet_size;
        packet_buffer_.resize(max_total_size);
        echo_buffer_.resize(max_total_size);
        
        std::cout << "[Session " << session_id_ << "] Pre-allocated buffers: " 
                  << (max_total_size / (1024.0 * 1024.0)) << " MB" << std::endl;
    }

    void start() {
        do_read();
    }

private:
    void do_read() {
        auto self = shared_from_this();
        
        // First read header directly into packet_buffer_ to know the data size
        // This avoids copying header later
        // Note: header.data_size equals packet_size (payload size, excluding header)
        asio::async_read(socket_,
            asio::buffer(packet_buffer_.data(), sizeof(PacketHeader)),
            [this, self](boost::system::error_code ec, std::size_t length) {
                if (!ec && length == sizeof(PacketHeader)) {
                    // Extract header for accessing data_size
                    std::memcpy(&header_, packet_buffer_.data(), sizeof(PacketHeader));
                    read_complete_packet();
                } else if (ec != asio::error::eof) {
                    std::cerr << "Error reading header: " << ec.message() << std::endl;
                }
            });
    }

    void read_complete_packet() {
        auto recv_start = std::chrono::high_resolution_clock::now();
        
        // Read data portion directly into pre-allocated buffer
        // Buffer is already allocated, no memory allocation needed
        // Header is already in packet_buffer_ from do_read()
        // We read data directly after the header position
        size_t total_packet_size = sizeof(PacketHeader) + header_.data_size;
        
        // Read data portion - one complete packet, no splitting, no copying
        auto self = shared_from_this();
        asio::async_read(socket_,
            asio::buffer(packet_buffer_.data() + sizeof(PacketHeader), header_.data_size),
            [this, self, recv_start, total_packet_size](boost::system::error_code ec, std::size_t length) {
                if (!ec && length == header_.data_size) {
                    auto recv_complete_time = std::chrono::high_resolution_clock::now();
                    auto recv_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        recv_complete_time - recv_start).count();
                    
                    // No data copying needed - packet_buffer_ already contains header + data
                    // We can directly use packet_buffer_ for echo
                    handle_packet(recv_duration_us, total_packet_size);
                } else {
                    std::cerr << "Error reading data: " << ec.message() << std::endl;
                }
            });
    }

    void handle_packet(int64_t recv_duration_us, size_t total_packet_size) {
        auto handle_start = std::chrono::high_resolution_clock::now();
        auto receive_time = std::chrono::high_resolution_clock::now();
        auto receive_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            receive_time.time_since_epoch()).count();

        // Calculate one-way latency (client send -> server receive)
        int64_t one_way_latency_ns = 0;
        if (header_.timestamp_ns > 0) {
            one_way_latency_ns = receive_time_ns - static_cast<int64_t>(header_.timestamp_ns);
        }

        auto prep_start = std::chrono::high_resolution_clock::now();

        // Update header timestamp directly in packet_buffer_ (minimal copy)
        // packet_buffer_ already contains header + data, we just update the timestamp
        PacketHeader* echo_header = reinterpret_cast<PacketHeader*>(packet_buffer_.data());
        echo_header->timestamp_ns = receive_time_ns;
        
        // No data copying needed - packet_buffer_ already contains the complete packet
        // We can directly use packet_buffer_ for echo, avoiding data copy
        // This minimizes overhead during echo transmission

        auto prep_end = std::chrono::high_resolution_clock::now();
        auto prep_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            prep_end - prep_start).count();

        packets_received_++;
        
        // Show detailed breakdown for first few packets
        if (packets_received_ <= 3) {
            std::cout << "\n[Session " << session_id_ << "] Packet " << header_.packet_id 
                      << " Time Breakdown:" << std::endl;
            std::cout << "  ========================================" << std::endl;
            std::cout << "  Total latency (client send -> server recv): " 
                      << std::fixed << std::setprecision(2) << (one_way_latency_ns / 1000.0) << " us" << std::endl;
            std::cout << "  Data size: " << std::fixed << std::setprecision(2) 
                      << (header_.data_size / (1024.0 * 1024.0)) << " MB" << std::endl;
            std::cout << "  ========================================" << std::endl;
            std::cout << "  Server-side processing breakdown:" << std::endl;
            std::cout << "    Receive time: " << recv_duration_us << " us" << std::endl;
            std::cout << "    Prepare echo: " << prep_duration_us << " us" << std::endl;
            std::cout << "    Total handle_packet: " << std::chrono::duration_cast<std::chrono::microseconds>(
                prep_end - handle_start).count() << " us" << std::endl;
        }
        
        // Send complete echo packet directly from packet_buffer_ - no data copying
        // packet_buffer_ already contains header (with updated timestamp) + data
        auto send_start = std::chrono::high_resolution_clock::now();
        auto self = shared_from_this();
        asio::async_write(socket_,
            asio::buffer(packet_buffer_.data(), total_packet_size),
            [this, self, send_start, prep_duration_us, recv_duration_us](boost::system::error_code ec, std::size_t /*bytes_sent*/) {
                if (!ec) {
                    auto send_end = std::chrono::high_resolution_clock::now();
                    auto send_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        send_end - send_start).count();
                    
                    // Show send time for first few packets
                    if (packets_received_ <= 3) {
                        std::cout << "    Post send: " << send_duration_us << " us" << std::endl;
                        std::cout << "  ========================================" << std::endl;
                        std::cout << "  Note: Total latency includes:" << std::endl;
                        std::cout << "    1. Network transmission time" << std::endl;
                        std::cout << "    2. TCP/IP stack processing" << std::endl;
                        std::cout << "    3. Server software processing (" 
                                  << (recv_duration_us + prep_duration_us + send_duration_us) << " us)" << std::endl;
                    }
                    do_read();
                }
            });
    }

    tcp::socket socket_;
    int session_id_;
    PacketHeader header_;
    size_t max_packet_size_;
    std::vector<uint8_t> packet_buffer_;    // Pre-allocated complete packet buffer (header + data)
    std::vector<uint8_t> echo_buffer_;      // Pre-allocated echo buffer (not used, kept for compatibility)
    std::atomic<uint64_t> packets_received_;
    std::atomic<uint64_t> total_latency_ns_;
};

class Server {
public:
    Server(asio::io_context& io_context, short port)
        : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)),
          session_counter_(0) {
        do_accept();
    }

private:
    void do_accept() {
        acceptor_.async_accept(
            [this](boost::system::error_code ec, tcp::socket socket) {
                if (!ec) {
                    int session_id = ++session_counter_;
                    std::cout << "New client connected, session ID: " << session_id << std::endl;
                    auto session = boost::make_shared<ServerSession>(std::move(socket), session_id);
                    session->start();
                }
                do_accept();
            });
    }

    tcp::acceptor acceptor_;
    std::atomic<int> session_counter_;
};

int main(int argc, char* argv[]) {
    try {
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <port>" << std::endl;
            return 1;
        }

        short port = static_cast<short>(std::atoi(argv[1]));
        
        std::cout << "=========================================" << std::endl;
        std::cout << "ASIO Server - Latency Test Receiver" << std::endl;
        std::cout << "Listening on port: " << port << std::endl;
        std::cout << "=========================================" << std::endl;

        asio::io_context io_context;
        Server server(io_context, port);
        
        // Run in multiple threads for better performance
        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        
        std::vector<std::thread> threads;
        for (size_t i = 1; i < num_threads; ++i) {
            threads.emplace_back([&io_context]() {
                io_context.run();
            });
        }
        
        io_context.run();
        
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

