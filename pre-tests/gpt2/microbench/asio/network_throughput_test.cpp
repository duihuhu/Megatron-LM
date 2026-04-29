#include <boost/asio.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <iomanip>
#include <cstring>

namespace asio = boost::asio;
namespace po = boost::program_options;
using boost::asio::ip::tcp;

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
        
        // bytes to gigabits: bytes * 8 / (1024^3)
        // microseconds to seconds: / 1e6
        double gbits = (bytes_transferred.load() * 8.0) / (1024.0 * 1024.0 * 1024.0);
        double seconds = duration / 1e6;
        return gbits / seconds;
    }
    
    double get_duration_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
    }
};

// Server worker thread - handles one connection synchronously
void server_worker(tcp::socket socket, size_t buffer_size) {
    try {
        std::vector<char> buffer(buffer_size);
        std::cout << "Server thread started for connection from "
                  << socket.remote_endpoint() << std::endl;
        
        while (true) {
            // Read complete message
            boost::system::error_code ec;
            size_t bytes_read = asio::read(socket, asio::buffer(buffer), ec);
            
            if (ec == asio::error::eof) {
                // Connection closed by client
                break;
            } else if (ec) {
                std::cerr << "Read error: " << ec.message() << std::endl;
                break;
            }
            
            // Echo back the data
            asio::write(socket, asio::buffer(buffer, bytes_read));
        }
        
        std::cout << "Server thread finished for connection from "
                  << socket.remote_endpoint() << std::endl;
                  
    } catch (std::exception& e) {
        std::cerr << "Server worker error: " << e.what() << std::endl;
    }
}

// Server function
void run_server(unsigned short port, size_t num_threads, size_t buffer_size) {
    try {
        asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), port));
        
        std::cout << "Server listening on port " << port << std::endl;
        std::cout << "Max concurrent threads: " << num_threads << std::endl;
        std::cout << "Buffer size: " << buffer_size << " bytes" << std::endl;
        std::cout << "Waiting for connections..." << std::endl;
        
        std::vector<std::thread> threads;
        
        // Accept connections and spawn threads
        while (true) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
            
            std::cout << "Accepted connection from " << socket.remote_endpoint() << std::endl;
            
            // Spawn a thread to handle this connection
            threads.emplace_back(server_worker, std::move(socket), buffer_size);
            
            // Clean up finished threads
            threads.erase(
                std::remove_if(threads.begin(), threads.end(),
                    [](std::thread& t) {
                        if (t.joinable()) {
                            return false;
                        }
                        return true;
                    }),
                threads.end()
            );
            
            // Limit concurrent connections
            if (threads.size() >= num_threads) {
                std::cout << "Reached max threads (" << num_threads 
                         << "), waiting for a thread to finish..." << std::endl;
                if (!threads.empty() && threads[0].joinable()) {
                    threads[0].join();
                    threads.erase(threads.begin());
                }
            }
        }
        
        // Join all remaining threads
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        
    } catch (std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
}

// Client worker thread - synchronous send/receive
void client_worker(const std::string& host, const std::string& port,
                   size_t data_size, size_t iterations,
                   Statistics& stats, bool is_warmup) {
    try {
        asio::io_context io_context;
        tcp::resolver resolver(io_context);
        auto endpoints = resolver.resolve(host, port);
        
        tcp::socket socket(io_context);
        asio::connect(socket, endpoints);
        
        if (!is_warmup) {
            std::cout << "Client thread connected to " << socket.remote_endpoint() << std::endl;
        }
        
        // Prepare buffers
        std::vector<char> send_buffer(data_size);
        std::vector<char> recv_buffer(data_size);
        
        // Fill send buffer with test data
        for (size_t i = 0; i < data_size; ++i) {
            send_buffer[i] = static_cast<char>(i % 256);
        }
        
        // Send and receive iterations
        for (size_t i = 0; i < iterations; ++i) {
            // Synchronous write - guarantees all data is sent
            asio::write(socket, asio::buffer(send_buffer));
            
            // Synchronous read - guarantees all data is received
            asio::read(socket, asio::buffer(recv_buffer));
            
            // Update statistics
            stats.bytes_transferred += data_size * 2;  // send + receive
            stats.operations_completed++;
        }
        
        if (!is_warmup) {
            std::cout << "Client thread completed " << iterations << " iterations" << std::endl;
        }
        
    } catch (std::exception& e) {
        std::cerr << "Client worker error: " << e.what() << std::endl;
    }
}

// Run client
void run_client(const std::string& host, const std::string& port,
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
                                          warmup_iterations, std::ref(warmup_stats), true);
            }
            
            for (auto& t : warmup_threads) {
                t.join();
            }
            
            warmup_stats.finish();
            std::cout << "Warmup completed in " << warmup_stats.get_duration_ms()
                     << " ms" << std::endl;
            
            // Small delay after warmup
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
                 << (data_size * iterations * num_threads * 2 / (1024.0 * 1024.0 * 1024.0))
                 << " GB (bidirectional)" << std::endl;
        std::cout << "\nRunning test..." << std::endl;
        
        Statistics stats;
        stats.reset();
        
        std::vector<std::thread> threads;
        for (size_t i = 0; i < num_threads; ++i) {
            threads.emplace_back(client_worker, host, port, data_size,
                              iterations, std::ref(stats), false);
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
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

int main(int argc, char* argv[]) {
    try {
        po::options_description desc("Network Throughput Test - Options");
        desc.add_options()
            ("help,h", "Show help message")
            ("mode,m", po::value<std::string>()->required(),
             "Mode: 'server' or 'client'")
            ("host", po::value<std::string>()->default_value("127.0.0.1"),
             "Server hostname or IP (client mode)")
            ("port,p", po::value<unsigned short>()->default_value(12345),
             "Port number")
            ("threads,t", po::value<size_t>()->default_value(1),
             "Number of threads")
            ("size,s", po::value<size_t>()->default_value(65536),
             "Data size in bytes (use --size-mb for MB)")
            ("size-mb", po::value<size_t>(),
             "Data size in MB (overrides --size if specified)")
            ("iterations,n", po::value<size_t>()->default_value(1000),
             "Number of iterations per thread (client mode)")
            ("warmup,w", po::value<size_t>()->default_value(100),
             "Number of warmup iterations per thread (client mode)");
        
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            std::cout << "\nExample usage:" << std::endl;
            std::cout << "  Server: " << argv[0] << " --mode server --port 12345 --threads 4 --size-mb 10" << std::endl;
            std::cout << "  Client: " << argv[0] << " --mode client --host 192.168.1.100 --port 12345 "
                     << "--threads 4 --size-mb 10 --iterations 1000 --warmup 100" << std::endl;
            std::cout << "\nNotes:" << std::endl;
            std::cout << "  - Each client thread establishes its own connection to the server" << std::endl;
            std::cout << "  - Server spawns one thread per connection for handling" << std::endl;
            std::cout << "  - All operations are synchronous (blocking I/O)" << std::endl;
            std::cout << "  - Data size should be the same on both server and client" << std::endl;
            return 0;
        }
        
        po::notify(vm);
        
        std::string mode = vm["mode"].as<std::string>();
        unsigned short port = vm["port"].as<unsigned short>();
        size_t num_threads = vm["threads"].as<size_t>();
        
        // Handle data size - MB takes precedence over bytes
        size_t data_size;
        if (vm.count("size-mb")) {
            data_size = vm["size-mb"].as<size_t>() * 1024 * 1024;
            std::cout << "Using data size: " << vm["size-mb"].as<size_t>() 
                     << " MB (" << data_size << " bytes)" << std::endl;
        } else {
            data_size = vm["size"].as<size_t>();
        }
        
        if (mode == "server") {
            run_server(port, num_threads, data_size);
        } else if (mode == "client") {
            std::string host = vm["host"].as<std::string>();
            size_t iterations = vm["iterations"].as<size_t>();
            size_t warmup_iterations = vm["warmup"].as<size_t>();
            
            run_client(host, std::to_string(port), num_threads,
                      data_size, iterations, warmup_iterations);
        } else {
            std::cerr << "Invalid mode. Use 'server' or 'client'." << std::endl;
            return 1;
        }
        
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
