/**
 * ASIO Client - Latency Test Sender
 * 
 * This client sends data packets to server and measures latency.
 * It measures both one-way and round-trip latencies.
 * 
 * Compile:
 *   g++ -std=c++17 -O3 asio_client.cpp -o asio_client -lboost_system -pthread
 * 
 * Usage:
 *   ./asio_client <host> <port> <packet_size> <num_packets> [interval_us] [warmup_packets]
 */

#include <boost/asio.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cstring>
#include <thread>
#include <cmath>

using boost::asio::ip::tcp;
namespace asio = boost::asio;

struct PacketHeader {
    uint64_t packet_id;
    uint64_t timestamp_ns;  // Send timestamp in nanoseconds
    uint32_t data_size;     // Size of payload data in bytes
};

class LatencyClient {
public:
    LatencyClient(asio::io_context& io_context,
                  const std::string& host, const std::string& port,
                  size_t packet_size, size_t num_packets, int interval_us, size_t warmup_packets = 0)
        : socket_(io_context),
          packet_size_(packet_size),
          num_packets_(num_packets),
          interval_us_(interval_us),
          warmup_packets_(warmup_packets),
          packets_sent_(0),
          packets_received_(0),
          packets_counted_(0),
          total_one_way_ns_(0),
          total_round_trip_ns_(0),
          total_round_trip_ns_squared_(0),
          min_one_way_ns_(INT64_MAX),
          max_one_way_ns_(0),
          min_round_trip_ns_(INT64_MAX),
          max_round_trip_ns_(0) {
        
        tcp::resolver resolver(io_context);
        endpoints_ = resolver.resolve(host, port);
    }

    void start() {
        asio::async_connect(socket_, endpoints_,
            [this](boost::system::error_code ec, tcp::endpoint) {
                if (!ec) {
                    // Set TCP_NODELAY to reduce latency
                    socket_.set_option(tcp::no_delay(true));
                    
                    // Set large send buffer to avoid blocking on large packets
                    // Default TCP send buffer is usually 64KB-256KB, which is too small for 64MB packets
                    // This causes async_write to block waiting for buffer space, increasing overhead
                    size_t requested_buffer_size = std::max(packet_size_ * 2, size_t(128 * 1024 * 1024));  // At least 2x packet size or 128MB
                    
                    // Get initial buffer size before setting
                    asio::socket_base::send_buffer_size initial_size;
                    socket_.get_option(initial_size);
                    
                    boost::system::error_code ec_buf;
                    socket_.set_option(asio::socket_base::send_buffer_size(requested_buffer_size), ec_buf);
                    
                    asio::socket_base::send_buffer_size actual_size;
                    socket_.get_option(actual_size);
                    
                    std::cout << "TCP send buffer configuration:" << std::endl;
                    std::cout << "  Initial size: " << (initial_size.value() / (1024.0 * 1024.0)) << " MB" << std::endl;
                    std::cout << "  Requested: " << (requested_buffer_size / (1024.0 * 1024.0)) << " MB" << std::endl;
                    std::cout << "  Actual: " << (actual_size.value() / (1024.0 * 1024.0)) << " MB" << std::endl;
                    
                    if (ec_buf) {
                        std::cout << "  ERROR: Failed to set buffer size: " << ec_buf.message() << std::endl;
                    }
                    
                    if (actual_size.value() < requested_buffer_size) {
                        std::cout << "  WARNING: Actual buffer size is smaller than requested!" << std::endl;
                        std::cout << "           This is likely limited by system settings." << std::endl;
                        std::cout << "           To fix this, run as root:" << std::endl;
                        std::cout << "             sudo sysctl -w net.core.wmem_max=134217728" << std::endl;
                        std::cout << "             sudo sysctl -w net.ipv4.tcp_wmem=\"4096 16384 134217728\"" << std::endl;
                        std::cout << "           Or check current limits:" << std::endl;
                        std::cout << "             sysctl net.core.wmem_max net.ipv4.tcp_wmem" << std::endl;
                    }
                    
                    if (actual_size.value() < packet_size_) {
                        int estimated_chunks = static_cast<int>(std::ceil(packet_size_ / static_cast<double>(actual_size.value())));
                        double overhead_per_chunk_us = 50.0;  // Typical system call overhead
                        double estimated_total_overhead_us = estimated_chunks * overhead_per_chunk_us;
                        
                        std::cout << "  CRITICAL: Buffer size (" << (actual_size.value() / (1024.0 * 1024.0)) 
                                  << " MB) is much smaller than packet size (" 
                                  << (packet_size_ / (1024.0 * 1024.0)) << " MB)!" << std::endl;
                        std::cout << "           Data will be sent in ~" << estimated_chunks << " chunks." << std::endl;
                        std::cout << "           Estimated overhead: ~" << std::fixed << std::setprecision(0)
                                  << estimated_total_overhead_us << " us (" << (estimated_total_overhead_us / 1000.0) << " ms)" << std::endl;
                        std::cout << "           This explains the high async_write overhead!" << std::endl;
                        std::cout << "           RECOMMENDATION: Use RDMA for zero-copy transfer (no kernel buffer needed)" << std::endl;
                    } else {
                        std::cout << "  OK: Buffer size is sufficient for packet size." << std::endl;
                    }
                    
                    std::cout << "Connected to server" << std::endl;
                    
                    // Pre-allocate all buffers before starting transmission
                    // This ensures no memory allocation during sending/receiving
                    size_t total_packet_size = sizeof(PacketHeader) + packet_size_;
                    send_buffer_.resize(total_packet_size);
                    receive_buffer_.resize(total_packet_size);
                    
                    std::cout << "Pre-allocated buffers:" << std::endl;
                    std::cout << "  Send buffer: " << (send_buffer_.size() / (1024.0 * 1024.0)) << " MB" << std::endl;
                    std::cout << "  Receive buffer: " << (receive_buffer_.size() / (1024.0 * 1024.0)) << " MB" << std::endl;
                    std::cout << "Note: Assuming data already exists in send buffer, no data copying during transmission" << std::endl;
                    
                    // Benchmark pure memcpy performance for comparison
                    std::cout << "\nBenchmarking pure memcpy performance..." << std::endl;
                    std::vector<uint8_t> src(packet_size_, 0xAA);
                    std::vector<uint8_t> dst(packet_size_, 0);
                    auto memcpy_start = std::chrono::high_resolution_clock::now();
                    std::memcpy(dst.data(), src.data(), packet_size_);
                    auto memcpy_end = std::chrono::high_resolution_clock::now();
                    auto memcpy_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        memcpy_end - memcpy_start).count();
                    double memcpy_bandwidth_gbps = (packet_size_ * 8.0) / (memcpy_duration_us / 1e6) / 1e9;
                    std::cout << "  Pure memcpy (" << (packet_size_ / (1024.0 * 1024.0)) << " MB): " 
                              << memcpy_duration_us << " us (" << std::fixed << std::setprecision(2) 
                              << memcpy_bandwidth_gbps << " Gbps)" << std::endl;
                    std::cout << "  Note: async_write overhead = send_duration_us - memcpy_time" << std::endl;
                    std::cout << "        Typical async_write overhead includes: system call, kernel buffer management, etc." << std::endl;
                    
                    start_sending();
                } else {
                    std::cerr << "Connect failed: " << ec.message() << std::endl;
                }
            });
    }

private:
    void start_sending() {
        send_next_packet();
    }

    void send_next_packet() {
        if (packets_sent_ >= num_packets_) {
            socket_.close();
            print_statistics();
            return;
        }

        auto prep_start = std::chrono::high_resolution_clock::now();

        // Update packet header directly in pre-allocated send buffer
        // Note: packet_size_ is the data size (payload only, excluding header)
        // Buffer is already allocated, data is assumed to already exist
        PacketHeader* header = reinterpret_cast<PacketHeader*>(send_buffer_.data());
        header->packet_id = packets_sent_;
        auto header_prep_time = std::chrono::high_resolution_clock::now();
        header->timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            header_prep_time.time_since_epoch()).count();
        header->data_size = packet_size_;  // data_size equals packet_size_ (payload size)

        // No data copying needed - data is assumed to already exist in send_buffer_
        // This minimizes overhead during transmission

        auto prep_end = std::chrono::high_resolution_clock::now();
        auto prep_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            prep_end - prep_start).count();

        // Record send start time (after header update)
        auto send_time = std::chrono::high_resolution_clock::now();

        // Get packet_id before lambda (atomic cannot be copied directly)
        size_t packet_id = packets_sent_;

        // Send complete packet as a single buffer - one packet per send, no splitting
        size_t total_packet_size = sizeof(PacketHeader) + packet_size_;
        asio::async_write(socket_,
            asio::buffer(send_buffer_.data(), total_packet_size),
            [this, send_time, packet_id, prep_duration_us](boost::system::error_code ec, std::size_t /*bytes_sent*/) {
                if (!ec) {
                    auto send_complete_time = std::chrono::high_resolution_clock::now();
                    auto send_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        send_complete_time - send_time).count();
                    packets_sent_++;
                    receive_response(send_time, packet_id, prep_duration_us, send_duration_us);
                } else {
                    std::cerr << "Send failed: " << ec.message() << std::endl;
                }
            });
    }

    void receive_response(std::chrono::high_resolution_clock::time_point send_time,
                          uint64_t expected_packet_id, int64_t prep_duration_us, int64_t send_duration_us) {
        auto recv_start = std::chrono::high_resolution_clock::now();
        
        // Read complete response packet (header + data) directly into pre-allocated buffer
        // Total size = header size + data size (packet_size_)
        // Buffer is already allocated, no memory allocation needed
        size_t total_packet_size = sizeof(PacketHeader) + packet_size_;

        // Read entire packet in one operation - one packet per read, no splitting
        asio::async_read(socket_,
            asio::buffer(receive_buffer_.data(), total_packet_size),
            [this, send_time, expected_packet_id, prep_duration_us, send_duration_us, recv_start]
            (boost::system::error_code ec, std::size_t length) {
                if (!ec && length == sizeof(PacketHeader) + packet_size_) {
                    auto recv_complete_time = std::chrono::high_resolution_clock::now();
                    auto recv_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        recv_complete_time - recv_start).count();
                    // Extract header from buffer (minimal copy, only header)
                    std::memcpy(&response_header_, receive_buffer_.data(), sizeof(PacketHeader));
                    handle_response(send_time, expected_packet_id, prep_duration_us, send_duration_us, recv_duration_us);
                } else if (ec != asio::error::eof) {
                    std::cerr << "Receive failed: " << ec.message() << std::endl;
                }
            });
    }

    void handle_response(std::chrono::high_resolution_clock::time_point send_time,
                         uint64_t packet_id, int64_t prep_duration_us, int64_t send_duration_us, int64_t recv_duration_us) {
        auto receive_time = std::chrono::high_resolution_clock::now();
        packets_received_++;
        
        // Calculate round-trip latency
        auto round_trip_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            receive_time - send_time).count();
        double round_trip_us = round_trip_ns / 1000.0;

        // Calculate one-way latency (using server timestamp)
        int64_t one_way_ns = 0;
        if (response_header_.timestamp_ns > 0) {
            auto client_send_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                send_time.time_since_epoch()).count();
            one_way_ns = static_cast<int64_t>(response_header_.timestamp_ns) - client_send_ns;
        }

        // Calculate time breakdown
        // Note: In ASIO async operations:
        //   - async_write: Completes when data is copied to kernel send buffer
        //     For large packets (64MB), this can take significant time (copying 64MB to kernel)
        //   - async_read: Completes when data is read from kernel receive buffer
        //     This includes waiting for network data to arrive + copying from kernel
        //
        // Time breakdown:
        //   - prep_duration_us: Header update time (minimal, < 1us)
        //   - send_duration_us: Time to copy data to kernel send buffer (can be large for big packets)
        //   - recv_duration_us: Time from async_read start to data available
        //     This includes: network transmission + server processing + kernel receive overhead
        //   - round_trip_us: Total time from send_time to receive_time
        //
        // Important: send_duration_us and recv_duration_us are measured from different start points
        //   - send_duration_us: from send_time to send_complete_time
        //   - recv_duration_us: from recv_start to recv_complete_time
        //   - There may be a gap between send_complete and recv_start (network transmission happening)
        //
        // More accurate estimation:
        //   - Client send overhead: prep_duration_us + send_duration_us
        //   - Network + server time: recv_duration_us (most of it is network/server, not kernel overhead)
        //   - The gap between send_complete and recv_start is included in recv_duration_us
        double client_overhead_us = prep_duration_us + send_duration_us;
        // recv_duration_us includes network transmission + server processing + small kernel overhead
        // For large packets, kernel overhead is negligible compared to network time
        double estimated_network_server_time_us = recv_duration_us;
        double estimated_gap_us = round_trip_us - client_overhead_us - estimated_network_server_time_us;
        if (estimated_gap_us < 0) estimated_gap_us = 0;
        
        // Skip warmup packets in statistics
        if (packets_received_ <= warmup_packets_) {
            // Warmup phase - don't count in statistics, but show breakdown for first packet
            if (packets_received_ == 1) {
            std::cout << "\n[Warmup Packet 0] Time Breakdown:" << std::endl;
            std::cout << "  Header update: " << prep_duration_us << " us (no data copy, data assumed pre-existing)" << std::endl;
            std::cout << "  Client send overhead: " << send_duration_us << " us (copy " 
                      << (packet_size_ / (1024.0 * 1024.0)) << " MB to kernel buffer)" << std::endl;
            std::cout << "  Network + server time: " << std::fixed << std::setprecision(2) 
                      << estimated_network_server_time_us << " us (network transmission + server processing)" << std::endl;
            if (estimated_gap_us > 100) {
                std::cout << "  Time gap (unaccounted): " << std::fixed << std::setprecision(2) 
                          << estimated_gap_us << " us" << std::endl;
            }
            std::cout << "  Total round-trip: " << std::fixed << std::setprecision(2) 
                      << round_trip_us << " us" << std::endl;
            }
            continue_after_response();
            return;
        }

        // Accumulate statistics
        if (one_way_ns > 0) {
            total_one_way_ns_ += one_way_ns;
            if (one_way_ns < min_one_way_ns_) min_one_way_ns_ = one_way_ns;
            if (one_way_ns > max_one_way_ns_) max_one_way_ns_ = one_way_ns;
        }

        total_round_trip_ns_ += round_trip_ns;
        total_round_trip_ns_squared_ += (round_trip_ns * round_trip_ns);
        if (round_trip_ns < min_round_trip_ns_) min_round_trip_ns_ = round_trip_ns;
        if (round_trip_ns > max_round_trip_ns_) max_round_trip_ns_ = round_trip_ns;

        packets_counted_++;
        
        // Show detailed breakdown for first counted packet
        if (packets_counted_ == 1) {
            // Get actual TCP send buffer size for analysis
            asio::socket_base::send_buffer_size actual_buffer_size;
            socket_.get_option(actual_buffer_size);
            size_t buffer_size_bytes = actual_buffer_size.value();
            
            // Estimate pure memcpy time for comparison
            // Typical memcpy speed: 10-20 GB/s on modern systems
            // For 64MB: 64MB / 15GB/s ≈ 4.3ms
            double estimated_memcpy_time_us = (packet_size_ / (1024.0 * 1024.0 * 1024.0)) / 15.0 * 1e6;  // Assume 15 GB/s
            double async_write_overhead_us = send_duration_us - estimated_memcpy_time_us;
            if (async_write_overhead_us < 0) async_write_overhead_us = 0;
            
            // Calculate estimated number of chunks if buffer is smaller than packet
            int estimated_chunks = 1;
            if (buffer_size_bytes > 0 && buffer_size_bytes < packet_size_) {
                estimated_chunks = static_cast<int>(std::ceil(packet_size_ / static_cast<double>(buffer_size_bytes)));
            }
            
            std::cout << "\n[Packet " << packet_id << "] Time Breakdown:" << std::endl;
            std::cout << "  Header update: " << prep_duration_us << " us (no data copy, data assumed pre-existing)" << std::endl;
            std::cout << "  Client send overhead: " << send_duration_us << " us (copy " 
                      << (packet_size_ / (1024.0 * 1024.0)) << " MB to kernel buffer)" << std::endl;
            std::cout << "    - Estimated memcpy time: ~" << std::fixed << std::setprecision(2) 
                      << estimated_memcpy_time_us << " us (assuming 15 GB/s memory bandwidth)" << std::endl;
            std::cout << "    - ASIO async_write overhead: ~" << std::fixed << std::setprecision(2) 
                      << async_write_overhead_us << " us (system calls, kernel buffer management)" << std::endl;
            std::cout << "    - TCP send buffer size: " << (buffer_size_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
            if (buffer_size_bytes > 0 && buffer_size_bytes < packet_size_) {
                std::cout << "    - Estimated chunks: ~" << estimated_chunks 
                          << " (buffer < packet size, data will be sent in chunks)" << std::endl;
                double overhead_per_chunk_us = async_write_overhead_us / estimated_chunks;
                std::cout << "    - Overhead per chunk: ~" << std::fixed << std::setprecision(2) 
                          << overhead_per_chunk_us << " us" << std::endl;
            }
            
            if (send_duration_us > estimated_memcpy_time_us * 2) {
                std::cout << "    WARNING: send_duration_us is significantly higher than expected memcpy time!" << std::endl;
                std::cout << "             This indicates significant overhead in async_write." << std::endl;
                std::cout << "             Analysis:" << std::endl;
                
                if (buffer_size_bytes > 0 && buffer_size_bytes < packet_size_) {
                    // Buffer too small - chunking is the main issue
                    std::cout << "             1. TCP send buffer (" << (buffer_size_bytes / (1024.0 * 1024.0)) 
                              << " MB) is smaller than packet size -> data copied in " << estimated_chunks << " chunks" << std::endl;
                    std::cout << "                Each chunk requires system call + buffer management overhead" << std::endl;
                    std::cout << "                Estimated overhead: " << std::fixed << std::setprecision(2)
                              << (overhead_per_chunk_us * estimated_chunks / 1000.0) << " ms" << std::endl;
                    std::cout << "             Solutions:" << std::endl;
                    std::cout << "             - Increase system limits: sudo sysctl -w net.core.wmem_max=134217728" << std::endl;
                    std::cout << "             - Or use RDMA for zero-copy transfer (no kernel buffer needed)" << std::endl;
                } else {
                    // Buffer is large enough, but still high overhead
                    std::cout << "             1. Buffer size (" << (buffer_size_bytes / (1024.0 * 1024.0)) 
                              << " MB) is sufficient, but overhead is still high (" 
                              << (async_write_overhead_us / 1000.0) << " ms)" << std::endl;
                    std::cout << "                Possible reasons:" << std::endl;
                    std::cout << "                a) ASIO may still split large writes into multiple system calls" << std::endl;
                    std::cout << "                   (Some implementations limit single write() to avoid blocking)" << std::endl;
                    std::cout << "                b) System call overhead: write() for 64MB may take time even with large buffer" << std::endl;
                    std::cout << "                   (Kernel needs to copy 64MB from user space to kernel space)" << std::endl;
                    std::cout << "                c) ASIO event loop and internal buffering overhead" << std::endl;
                    std::cout << "                d) Kernel buffer management and TCP protocol processing" << std::endl;
                    
                    // Estimate how many system calls might be needed
                    // Linux typically limits single write() to avoid blocking, often 1MB-4MB chunks
                    double typical_chunk_size_mb = 2.0;  // Assume 2MB chunks
                    int estimated_syscalls = static_cast<int>(std::ceil((packet_size_ / (1024.0 * 1024.0)) / typical_chunk_size_mb));
                    double overhead_per_syscall_us = async_write_overhead_us / estimated_syscalls;
                    
                    std::cout << "                Estimated system calls: ~" << estimated_syscalls 
                              << " (assuming " << typical_chunk_size_mb << " MB chunks)" << std::endl;
                    std::cout << "                Overhead per system call: ~" << std::fixed << std::setprecision(2)
                              << overhead_per_syscall_us << " us" << std::endl;
                    
                    std::cout << "             Solutions:" << std::endl;
                    std::cout << "             - Use RDMA for zero-copy transfer (recommended for large packets)" << std::endl;
                    std::cout << "               RDMA avoids kernel buffer entirely, reducing overhead to ~1-2ms" << std::endl;
                    std::cout << "             - Consider using sendfile() for file-based transfers (zero-copy in kernel)" << std::endl;
                    std::cout << "             - Accept the overhead: " << (async_write_overhead_us / 1000.0) 
                              << " ms is reasonable for 64MB TCP transfer" << std::endl;
                }
            } else {
                std::cout << "    OK: send_duration_us is reasonable compared to memcpy time." << std::endl;
            }
            std::cout << "  Network + server time: " << std::fixed << std::setprecision(2) 
                      << estimated_network_server_time_us << " us (network transmission + server processing)" << std::endl;
            if (estimated_gap_us > 100) {
                std::cout << "  Time gap (unaccounted): " << std::fixed << std::setprecision(2) 
                          << estimated_gap_us << " us" << std::endl;
            }
            std::cout << "  Total round-trip: " << std::fixed << std::setprecision(2) 
                      << round_trip_us << " us" << std::endl;
            if (one_way_ns > 0) {
                std::cout << "  One-way latency (server timestamp): " << std::fixed << std::setprecision(2) 
                          << (one_way_ns / 1000.0) << " us" << std::endl;
            }
        }
        
        continue_after_response();
    }
    
    void continue_after_response() {
        // Data already read in receive_response, just continue
        continue_sending();
    }

    void continue_sending() {
        if (interval_us_ > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(interval_us_));
        }
        send_next_packet();
    }

    void print_statistics() {
        std::cout << "\n=========================================" << std::endl;
        std::cout << "Latency Test Statistics" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Packet size (data): " << packet_size_ << " bytes" << std::endl;
        std::cout << "  Total packet size: " << (sizeof(PacketHeader) + packet_size_) << " bytes" << std::endl;
        std::cout << "  Total packets sent: " << packets_sent_ << std::endl;
        std::cout << "  Total packets received: " << packets_received_ << std::endl;
        if (warmup_packets_ > 0) {
            std::cout << "  Warmup packets (excluded): " << warmup_packets_ << std::endl;
        }
        std::cout << "  Packets counted for statistics: " << packets_counted_ << std::endl;
        std::cout << std::endl;

        if (packets_counted_ > 0) {
            // Round-trip latency statistics
            double avg_rt_ns = static_cast<double>(total_round_trip_ns_) / packets_counted_;
            double avg_rt_us = avg_rt_ns / 1000.0;
            double min_rt_us = min_round_trip_ns_ / 1000.0;
            double max_rt_us = max_round_trip_ns_ / 1000.0;
            
            // Calculate standard deviation
            double variance_ns = (static_cast<double>(total_round_trip_ns_squared_) / packets_counted_) - (avg_rt_ns * avg_rt_ns);
            double stddev_ns = (variance_ns > 0) ? std::sqrt(variance_ns) : 0.0;
            double stddev_us = stddev_ns / 1000.0;
            
            std::cout << "Round-Trip Latency Statistics:" << std::endl;
            std::cout << "  Packets measured: " << packets_counted_ << std::endl;
            std::cout << "  Average: " << std::fixed << std::setprecision(2) << avg_rt_us << " us" << std::endl;
            std::cout << "  Minimum: " << std::fixed << std::setprecision(2) << min_rt_us << " us" << std::endl;
            std::cout << "  Maximum: " << std::fixed << std::setprecision(2) << max_rt_us << " us" << std::endl;
            std::cout << "  Std Dev: " << std::fixed << std::setprecision(2) << stddev_us << " us" << std::endl;
            std::cout << std::endl;

            // One-way latency statistics (if available)
            if (total_one_way_ns_ > 0) {
                double avg_ow_us = (total_one_way_ns_ / packets_counted_) / 1000.0;
                double min_ow_us = min_one_way_ns_ / 1000.0;
                double max_ow_us = max_one_way_ns_ / 1000.0;
                
                std::cout << "One-Way Latency (estimated):" << std::endl;
                std::cout << "  Average: " << std::fixed << std::setprecision(2) << avg_ow_us << " us" << std::endl;
                std::cout << "  Minimum: " << std::fixed << std::setprecision(2) << min_ow_us << " us" << std::endl;
                std::cout << "  Maximum: " << std::fixed << std::setprecision(2) << max_ow_us << " us" << std::endl;
                std::cout << std::endl;
            }

            // Calculate throughput (only for counted packets)
            double total_bytes = (packets_counted_ * (sizeof(PacketHeader) + packet_size_)) * 2.0; // Round-trip
            double total_time_sec = total_round_trip_ns_ / 1e9;
            double throughput_mbps = (total_bytes * 8.0) / (total_time_sec * 1e6);
            
            std::cout << "Throughput: " << std::fixed << std::setprecision(2) 
                      << throughput_mbps << " Mbps" << std::endl;
        } else {
            std::cout << "No packets were counted for statistics." << std::endl;
        }
        std::cout << "=========================================" << std::endl;
    }

    tcp::socket socket_;
    tcp::resolver::results_type endpoints_;
    size_t packet_size_;
    size_t num_packets_;
    int interval_us_;
    size_t warmup_packets_;
    
    std::vector<uint8_t> send_buffer_;      // Complete packet buffer (header + data)
    PacketHeader response_header_;
    std::vector<uint8_t> receive_buffer_;   // Complete response buffer (header + data)
    
    std::atomic<size_t> packets_sent_;
    std::atomic<size_t> packets_received_;
    std::atomic<size_t> packets_counted_;
    
    std::atomic<int64_t> total_one_way_ns_;
    std::atomic<int64_t> total_round_trip_ns_;
    std::atomic<int64_t> total_round_trip_ns_squared_;  // For standard deviation calculation
    std::atomic<int64_t> min_one_way_ns_;
    std::atomic<int64_t> max_one_way_ns_;
    std::atomic<int64_t> min_round_trip_ns_;
    std::atomic<int64_t> max_round_trip_ns_;
};

int main(int argc, char* argv[]) {
    try {
        if (argc < 5 || argc > 7) {
            std::cerr << "Usage: " << argv[0] 
                      << " <host> <port> <packet_size> <num_packets> [interval_us] [warmup_packets]" << std::endl;
            std::cerr << "  host: Server hostname or IP address" << std::endl;
            std::cerr << "  port: Server port number" << std::endl;
            std::cerr << "  packet_size: Size of payload data in bytes (excluding header)" << std::endl;
            std::cerr << "  num_packets: Number of packets to send (for statistics calculation)" << std::endl;
            std::cerr << "  interval_us: Optional interval between packets in microseconds (default: 0)" << std::endl;
            std::cerr << "  warmup_packets: Optional number of warmup packets to exclude from statistics (default: 0)" << std::endl;
            std::cerr << std::endl;
            std::cerr << "Note: Each packet is sent as a complete unit (header + data), no splitting." << std::endl;
            std::cerr << "      Round-trip latency is measured for each packet and statistics are calculated." << std::endl;
            return 1;
        }

        std::string host = argv[1];
        std::string port = argv[2];
        size_t packet_size = std::stoull(argv[3]);
        size_t num_packets = std::stoull(argv[4]);
        int interval_us = (argc >= 6) ? std::atoi(argv[5]) : 0;
        size_t warmup_packets = (argc == 7) ? std::stoull(argv[6]) : 0;

        std::cout << "=========================================" << std::endl;
        std::cout << "ASIO Client - Latency Test Sender" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "Target: " << host << ":" << port << std::endl;
        std::cout << "Packet size (data only): " << packet_size << " bytes" << std::endl;
        std::cout << "Total packet size (header + data): " << (sizeof(PacketHeader) + packet_size) << " bytes" << std::endl;
        std::cout << "Number of packets to send: " << num_packets << std::endl;
        std::cout << "Warmup packets (excluded from stats): " << warmup_packets << std::endl;
        std::cout << "Packets for statistics: " << (num_packets > warmup_packets ? num_packets - warmup_packets : num_packets) << std::endl;
        std::cout << "Interval between packets: " << interval_us << " us" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "Starting test..." << std::endl;

        asio::io_context io_context;
        LatencyClient client(io_context, host, port, packet_size, num_packets, interval_us, warmup_packets);
        
        client.start();
        
        io_context.run();
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

