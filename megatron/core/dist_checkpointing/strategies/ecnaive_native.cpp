#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <isa-l/erasure_code.h>
#include <isa-l/raid.h>


namespace {

// ASIO connection manager (pattern from eccheck_native)
class AsioConnectionManager {
private:
    boost::asio::io_context io_context_;
    
    // Save mode sockets: 3 sends + 3 receives per rank
    boost::asio::ip::tcp::socket send_data1_socket_;      // Send d_{i1} to rank i+1
    boost::asio::ip::tcp::socket send_parity0_socket_;    // Send p_{i0} to rank i+2
    boost::asio::ip::tcp::socket send_parity1_socket_;    // Send p_{i1} to rank i+3
    
    boost::asio::ip::tcp::socket recv_parity1_socket_;    // Recv p_{i+1,1} from rank i+1
    boost::asio::ip::tcp::socket recv_parity0_socket_;    // Recv p_{i+2,0} from rank i+2
    boost::asio::ip::tcp::socket recv_data1_socket_;      // Recv d_{i+3,1} from rank i+3
    
    boost::asio::ip::tcp::acceptor recv_parity1_acceptor_;
    boost::asio::ip::tcp::acceptor recv_parity0_acceptor_;
    boost::asio::ip::tcp::acceptor recv_data1_acceptor_;

    // Load mode sockets (rank2 as receiver)
    boost::asio::ip::tcp::socket load_recv_rank0_data2_socket_;
    boost::asio::ip::tcp::socket load_recv_rank0_parity2_socket_;
    boost::asio::ip::tcp::socket load_recv_rank1_data1_socket_;
    boost::asio::ip::tcp::socket load_recv_rank1_parity1_socket_;
    boost::asio::ip::tcp::socket load_recv_rank3_data1_socket_;
    boost::asio::ip::tcp::socket load_recv_rank3_data2_socket_;
    boost::asio::ip::tcp::acceptor load_recv_rank0_data2_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank0_parity2_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank1_data1_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank1_parity1_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank3_data1_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank3_data2_acceptor_;
    
    // Load mode sockets (other ranks as senders)
    boost::asio::ip::tcp::socket load_send_rank0_data2_socket_;
    boost::asio::ip::tcp::socket load_send_rank0_parity2_socket_;
    boost::asio::ip::tcp::socket load_send_rank1_data1_socket_;
    boost::asio::ip::tcp::socket load_send_rank1_parity1_socket_;
    boost::asio::ip::tcp::socket load_send_rank3_data1_socket_;
    boost::asio::ip::tcp::socket load_send_rank3_data2_socket_;

    // Save mode connection flags
    std::atomic<bool> send_data1_connected_{false};
    std::atomic<bool> send_parity0_connected_{false};
    std::atomic<bool> send_parity1_connected_{false};
    std::atomic<bool> recv_parity1_connected_{false};
    std::atomic<bool> recv_parity0_connected_{false};
    std::atomic<bool> recv_data1_connected_{false};
    
    // Load mode connection flags (rank2 receiver)
    std::atomic<bool> load_recv_rank0_data2_connected_{false};
    std::atomic<bool> load_recv_rank0_parity2_connected_{false};
    std::atomic<bool> load_recv_rank1_data1_connected_{false};
    std::atomic<bool> load_recv_rank1_parity1_connected_{false};
    std::atomic<bool> load_recv_rank3_data1_connected_{false};
    std::atomic<bool> load_recv_rank3_data2_connected_{false};
    
    // Load mode connection flags (other ranks sender)
    std::atomic<bool> load_send_rank0_data2_connected_{false};
    std::atomic<bool> load_send_rank0_parity2_connected_{false};
    std::atomic<bool> load_send_rank1_data1_connected_{false};
    std::atomic<bool> load_send_rank1_parity1_connected_{false};
    std::atomic<bool> load_send_rank3_data1_connected_{false};
    std::atomic<bool> load_send_rank3_data2_connected_{false};

    std::mutex connection_mutex_;
    std::condition_variable connection_cv_;

public:
    AsioConnectionManager()
        : io_context_(),
          send_data1_socket_(io_context_),
          send_parity0_socket_(io_context_),
          send_parity1_socket_(io_context_),
          recv_parity1_socket_(io_context_),
          recv_parity0_socket_(io_context_),
          recv_data1_socket_(io_context_),
          recv_parity1_acceptor_(io_context_),
          recv_parity0_acceptor_(io_context_),
          recv_data1_acceptor_(io_context_),
          load_recv_rank0_data2_socket_(io_context_),
          load_recv_rank0_parity2_socket_(io_context_),
          load_recv_rank1_data1_socket_(io_context_),
          load_recv_rank1_parity1_socket_(io_context_),
          load_recv_rank3_data1_socket_(io_context_),
          load_recv_rank3_data2_socket_(io_context_),
          load_recv_rank0_data2_acceptor_(io_context_),
          load_recv_rank0_parity2_acceptor_(io_context_),
          load_recv_rank1_data1_acceptor_(io_context_),
          load_recv_rank1_parity1_acceptor_(io_context_),
          load_recv_rank3_data1_acceptor_(io_context_),
          load_recv_rank3_data2_acceptor_(io_context_),
          load_send_rank0_data2_socket_(io_context_),
          load_send_rank0_parity2_socket_(io_context_),
          load_send_rank1_data1_socket_(io_context_),
          load_send_rank1_parity1_socket_(io_context_),
          load_send_rank3_data1_socket_(io_context_),
          load_send_rank3_data2_socket_(io_context_) {}

    // Save mode getters
    boost::asio::ip::tcp::socket& get_send_data1_socket() { return send_data1_socket_; }
    boost::asio::ip::tcp::socket& get_send_parity0_socket() { return send_parity0_socket_; }
    boost::asio::ip::tcp::socket& get_send_parity1_socket() { return send_parity1_socket_; }
    boost::asio::ip::tcp::socket& get_recv_parity1_socket() { return recv_parity1_socket_; }
    boost::asio::ip::tcp::socket& get_recv_parity0_socket() { return recv_parity0_socket_; }
    boost::asio::ip::tcp::socket& get_recv_data1_socket() { return recv_data1_socket_; }
    
    // Load mode getters (rank2 receiver)
    boost::asio::ip::tcp::socket& get_load_recv_rank0_data2_socket() { return load_recv_rank0_data2_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank0_parity2_socket() { return load_recv_rank0_parity2_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank1_data1_socket() { return load_recv_rank1_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank1_parity1_socket() { return load_recv_rank1_parity1_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank3_data1_socket() { return load_recv_rank3_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank3_data2_socket() { return load_recv_rank3_data2_socket_; }
    
    // Load mode getters (other ranks sender)
    boost::asio::ip::tcp::socket& get_load_send_rank0_data2_socket() { return load_send_rank0_data2_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank0_parity2_socket() { return load_send_rank0_parity2_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank1_data1_socket() { return load_send_rank1_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank1_parity1_socket() { return load_send_rank1_parity1_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank3_data1_socket() { return load_send_rank3_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank3_data2_socket() { return load_send_rank3_data2_socket_; }

    // Save mode connection checks
    bool is_send_data1_connected() const { return send_data1_connected_; }
    bool is_send_parity0_connected() const { return send_parity0_connected_; }
    bool is_send_parity1_connected() const { return send_parity1_connected_; }
    bool is_recv_parity1_connected() const { return recv_parity1_connected_; }
    bool is_recv_parity0_connected() const { return recv_parity0_connected_; }
    bool is_recv_data1_connected() const { return recv_data1_connected_; }

    // Save mode init functions
    void init_send_data1(const std::string& partner_ip, uint16_t port);
    void init_send_parity0(const std::string& partner_ip, uint16_t port);
    void init_send_parity1(const std::string& partner_ip, uint16_t port);
    void init_recv_parity1(const std::string& listen_ip, uint16_t port);
    void init_recv_parity0(const std::string& listen_ip, uint16_t port);
    void init_recv_data1(const std::string& listen_ip, uint16_t port);
    
    // Load mode init functions (rank2 as receiver)
    void init_load_recv_rank0_data2(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank0_parity2(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank1_parity1(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank3_data2(const std::string& listen_ip, uint16_t port);
    
    // Load mode init functions (other ranks as senders)
    void init_load_send_rank0_data2(const std::string& rank2_ip, uint16_t port);
    void init_load_send_rank0_parity2(const std::string& rank2_ip, uint16_t port);
    void init_load_send_rank1_data1(const std::string& rank2_ip, uint16_t port);
    void init_load_send_rank1_parity1(const std::string& rank2_ip, uint16_t port);
    void init_load_send_rank3_data1(const std::string& rank2_ip, uint16_t port);
    void init_load_send_rank3_data2(const std::string& rank2_ip, uint16_t port);
    
    // Load mode bind+listen helpers (for rank2, before accept)
    void bind_listen_load_recv_rank0_data2(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank0_parity2(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank1_parity1(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank3_data2(const std::string& listen_ip, uint16_t port);
    
    // Load mode accept helpers (for rank2, after bind+listen)
    void accept_load_recv_rank0_data2();
    void accept_load_recv_rank0_parity2();
    void accept_load_recv_rank1_data1();
    void accept_load_recv_rank1_parity1();
    void accept_load_recv_rank3_data1();
    void accept_load_recv_rank3_data2();
    
    void wait_for_connections(int timeout_seconds = 30);
    void wait_for_load_connections(int timeout_seconds = 30);
    void cleanup();
};

// Save mode init functions
void AsioConnectionManager::init_send_data1(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(send_data1_socket_, endpoints);
        send_data1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: send_data1 init error: " << e.what() << std::endl;
        send_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_send_parity0(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(send_parity0_socket_, endpoints);
        send_parity0_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: send_parity0 init error: " << e.what() << std::endl;
        send_parity0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_send_parity1(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(send_parity1_socket_, endpoints);
        send_parity1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: send_parity1 init error: " << e.what() << std::endl;
        send_parity1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_recv_parity1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        recv_parity1_acceptor_.open(endpoint.protocol());
        recv_parity1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        recv_parity1_acceptor_.bind(endpoint);
        recv_parity1_acceptor_.listen();
        recv_parity1_acceptor_.accept(recv_parity1_socket_);
        recv_parity1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: recv_parity1 init error: " << e.what() << std::endl;
        recv_parity1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_recv_parity0(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        recv_parity0_acceptor_.open(endpoint.protocol());
        recv_parity0_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        recv_parity0_acceptor_.bind(endpoint);
        recv_parity0_acceptor_.listen();
        recv_parity0_acceptor_.accept(recv_parity0_socket_);
        recv_parity0_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: recv_parity0 init error: " << e.what() << std::endl;
        recv_parity0_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_recv_data1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        recv_data1_acceptor_.open(endpoint.protocol());
        recv_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        recv_data1_acceptor_.bind(endpoint);
        recv_data1_acceptor_.listen();
        recv_data1_acceptor_.accept(recv_data1_socket_);
        recv_data1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: recv_data1 init error: " << e.what() << std::endl;
        recv_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::wait_for_connections(int timeout_seconds) {
    std::unique_lock<std::mutex> lock(connection_mutex_);
    connection_cv_.wait_for(
        lock,
        std::chrono::seconds(timeout_seconds),
        [this]() {
            return send_data1_connected_ && send_parity0_connected_ && send_parity1_connected_ &&
                   recv_parity1_connected_ && recv_parity0_connected_ && recv_data1_connected_;
        }
    );
}

void AsioConnectionManager::wait_for_load_connections(int timeout_seconds) {
    // For rank2: wait for all 6 recv connections
    // For rank0/1/3: wait for 2 send connections each
    if (load_recv_rank0_data2_connected_ || load_recv_rank0_parity2_connected_ ||
        load_recv_rank1_data1_connected_ || load_recv_rank1_parity1_connected_ ||
        load_recv_rank3_data1_connected_ || load_recv_rank3_data2_connected_) {
        // rank2: wait for all 6 recv connections
        int wait_count = 0;
        while (!(load_recv_rank0_data2_connected_ && load_recv_rank0_parity2_connected_ &&
                 load_recv_rank1_data1_connected_ && load_recv_rank1_parity1_connected_ &&
                 load_recv_rank3_data1_connected_ && load_recv_rank3_data2_connected_)) {
            if (wait_count % 100 == 0) {
                std::cout << "ECLATIN: [Rank 2] Waiting for load connections: "
                          << "r0_d2=" << (load_recv_rank0_data2_connected_ ? "true" : "false")
                          << ", r0_p2=" << (load_recv_rank0_parity2_connected_ ? "true" : "false")
                          << ", r1_d1=" << (load_recv_rank1_data1_connected_ ? "true" : "false")
                          << ", r1_p1=" << (load_recv_rank1_parity1_connected_ ? "true" : "false")
                          << ", r3_d1=" << (load_recv_rank3_data1_connected_ ? "true" : "false")
                          << ", r3_d2=" << (load_recv_rank3_data2_connected_ ? "true" : "false") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Rank 2] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    } else if (load_send_rank0_data2_connected_ || load_send_rank0_parity2_connected_) {
        // rank0: wait for 2 send connections
        int wait_count = 0;
        while (!(load_send_rank0_data2_connected_ && load_send_rank0_parity2_connected_)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Rank 0] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    } else if (load_send_rank1_data1_connected_ || load_send_rank1_parity1_connected_) {
        // rank1: wait for 2 send connections
        int wait_count = 0;
        while (!(load_send_rank1_data1_connected_ && load_send_rank1_parity1_connected_)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Rank 1] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    } else if (load_send_rank3_data1_connected_ || load_send_rank3_data2_connected_) {
        // rank3: wait for 2 send connections
        int wait_count = 0;
        while (!(load_send_rank3_data1_connected_ && load_send_rank3_data2_connected_)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Rank 3] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    }
}

void AsioConnectionManager::init_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank1_data1_acceptor_.open(endpoint.protocol());
        load_recv_rank1_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank1_data1_acceptor_.bind(endpoint);
        load_recv_rank1_data1_acceptor_.listen();
        load_recv_rank1_data1_acceptor_.accept(load_recv_rank1_data1_socket_);
        load_recv_rank1_data1_connected_ = true;
        std::cout << "ASIO: load_recv_rank1_data1 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank1_data1 init error: " << e.what() << std::endl;
        load_recv_rank1_data1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank1_parity1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank1_parity1_acceptor_.open(endpoint.protocol());
        load_recv_rank1_parity1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank1_parity1_acceptor_.bind(endpoint);
        load_recv_rank1_parity1_acceptor_.listen();
        load_recv_rank1_parity1_acceptor_.accept(load_recv_rank1_parity1_socket_);
        load_recv_rank1_parity1_connected_ = true;
        std::cout << "ASIO: load_recv_rank1_parity1 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank1_parity1 init error: " << e.what() << std::endl;
        load_recv_rank1_parity1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank0_data2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank0_data2_acceptor_.open(endpoint.protocol());
        load_recv_rank0_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank0_data2_acceptor_.bind(endpoint);
        load_recv_rank0_data2_acceptor_.listen();
        load_recv_rank0_data2_acceptor_.accept(load_recv_rank0_data2_socket_);
        load_recv_rank0_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank0_data2 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank0_data2 init error: " << e.what() << std::endl;
        load_recv_rank0_data2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank0_parity2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank0_parity2_acceptor_.open(endpoint.protocol());
        load_recv_rank0_parity2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank0_parity2_acceptor_.bind(endpoint);
        load_recv_rank0_parity2_acceptor_.listen();
        load_recv_rank0_parity2_acceptor_.accept(load_recv_rank0_parity2_socket_);
        load_recv_rank0_parity2_connected_ = true;
        std::cout << "ASIO: load_recv_rank0_parity2 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank0_parity2 init error: " << e.what() << std::endl;
        load_recv_rank0_parity2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank3_data1_acceptor_.open(endpoint.protocol());
        load_recv_rank3_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank3_data1_acceptor_.bind(endpoint);
        load_recv_rank3_data1_acceptor_.listen();
        load_recv_rank3_data1_acceptor_.accept(load_recv_rank3_data1_socket_);
        load_recv_rank3_data1_connected_ = true;
        std::cout << "ASIO: load_recv_rank3_data1 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank3_data1 init error: " << e.what() << std::endl;
        load_recv_rank3_data1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank3_data2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank3_data2_acceptor_.open(endpoint.protocol());
        load_recv_rank3_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank3_data2_acceptor_.bind(endpoint);
        load_recv_rank3_data2_acceptor_.listen();
        load_recv_rank3_data2_acceptor_.accept(load_recv_rank3_data2_socket_);
        load_recv_rank3_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank3_data2 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank3_data2 init error: " << e.what() << std::endl;
        load_recv_rank3_data2_connected_ = false;
        throw;
    }
}

// Load mode init functions (other ranks as senders)
void AsioConnectionManager::init_load_send_rank0_data2(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank0_data2_socket_, endpoints);
        load_send_rank0_data2_connected_ = true;
        std::cout << "ASIO: load_send_rank0_data2 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank0_data2 init error: " << e.what() << std::endl;
        load_send_rank0_data2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank0_parity2(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank0_parity2_socket_, endpoints);
        load_send_rank0_parity2_connected_ = true;
        std::cout << "ASIO: load_send_rank0_parity2 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank0_parity2 init error: " << e.what() << std::endl;
        load_send_rank0_parity2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank1_data1(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank1_data1_socket_, endpoints);
        load_send_rank1_data1_connected_ = true;
        std::cout << "ASIO: load_send_rank1_data1 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank1_data1 init error: " << e.what() << std::endl;
        load_send_rank1_data1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank1_parity1(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank1_parity1_socket_, endpoints);
        load_send_rank1_parity1_connected_ = true;
        std::cout << "ASIO: load_send_rank1_parity1 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank1_parity1 init error: " << e.what() << std::endl;
        load_send_rank1_parity1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank3_data1(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank3_data1_socket_, endpoints);
        load_send_rank3_data1_connected_ = true;
        std::cout << "ASIO: load_send_rank3_data1 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank3_data1 init error: " << e.what() << std::endl;
        load_send_rank3_data1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank3_data2(const std::string& rank2_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank2_ip, std::to_string(port));
        boost::asio::connect(load_send_rank3_data2_socket_, endpoints);
        load_send_rank3_data2_connected_ = true;
        std::cout << "ASIO: load_send_rank3_data2 connected to rank2" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank3_data2 init error: " << e.what() << std::endl;
        load_send_rank3_data2_connected_ = false;
        throw;
    }
}

// Load mode bind+listen helpers (for rank2, before accept)
void AsioConnectionManager::bind_listen_load_recv_rank0_data2(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank0_data2_acceptor_.open(endpoint.protocol());
    load_recv_rank0_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank0_data2_acceptor_.bind(endpoint);
    load_recv_rank0_data2_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank0_parity2(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank0_parity2_acceptor_.open(endpoint.protocol());
    load_recv_rank0_parity2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank0_parity2_acceptor_.bind(endpoint);
    load_recv_rank0_parity2_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank1_data1_acceptor_.open(endpoint.protocol());
    load_recv_rank1_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank1_data1_acceptor_.bind(endpoint);
    load_recv_rank1_data1_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank1_parity1(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank1_parity1_acceptor_.open(endpoint.protocol());
    load_recv_rank1_parity1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank1_parity1_acceptor_.bind(endpoint);
    load_recv_rank1_parity1_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank3_data1_acceptor_.open(endpoint.protocol());
    load_recv_rank3_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank3_data1_acceptor_.bind(endpoint);
    load_recv_rank3_data1_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank3_data2(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank3_data2_acceptor_.open(endpoint.protocol());
    load_recv_rank3_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank3_data2_acceptor_.bind(endpoint);
    load_recv_rank3_data2_acceptor_.listen();
}

// Load mode accept helpers (for rank2, after bind+listen)
void AsioConnectionManager::accept_load_recv_rank0_data2() {
    try {
        load_recv_rank0_data2_acceptor_.accept(load_recv_rank0_data2_socket_);
        load_recv_rank0_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank0_data2 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank0_data2 accept error: " << e.what() << std::endl;
        load_recv_rank0_data2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank0_parity2() {
    try {
        load_recv_rank0_parity2_acceptor_.accept(load_recv_rank0_parity2_socket_);
        load_recv_rank0_parity2_connected_ = true;
        std::cout << "ASIO: load_recv_rank0_parity2 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank0_parity2 accept error: " << e.what() << std::endl;
        load_recv_rank0_parity2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank1_data1() {
    try {
        load_recv_rank1_data1_acceptor_.accept(load_recv_rank1_data1_socket_);
        load_recv_rank1_data1_connected_ = true;
        std::cout << "ASIO: load_recv_rank1_data1 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank1_data1 accept error: " << e.what() << std::endl;
        load_recv_rank1_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank1_parity1() {
    try {
        load_recv_rank1_parity1_acceptor_.accept(load_recv_rank1_parity1_socket_);
        load_recv_rank1_parity1_connected_ = true;
        std::cout << "ASIO: load_recv_rank1_parity1 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank1_parity1 accept error: " << e.what() << std::endl;
        load_recv_rank1_parity1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank3_data1() {
    try {
        load_recv_rank3_data1_acceptor_.accept(load_recv_rank3_data1_socket_);
        load_recv_rank3_data1_connected_ = true;
        std::cout << "ASIO: load_recv_rank3_data1 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank3_data1 accept error: " << e.what() << std::endl;
        load_recv_rank3_data1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank3_data2() {
    try {
        load_recv_rank3_data2_acceptor_.accept(load_recv_rank3_data2_socket_);
        load_recv_rank3_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank3_data2 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank3_data2 accept error: " << e.what() << std::endl;
        load_recv_rank3_data2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::cleanup() {
    // Save mode sockets
    if (send_data1_socket_.is_open()) send_data1_socket_.close();
    if (send_parity0_socket_.is_open()) send_parity0_socket_.close();
    if (send_parity1_socket_.is_open()) send_parity1_socket_.close();
    if (recv_parity1_socket_.is_open()) recv_parity1_socket_.close();
    if (recv_parity0_socket_.is_open()) recv_parity0_socket_.close();
    if (recv_data1_socket_.is_open()) recv_data1_socket_.close();
    if (recv_parity1_acceptor_.is_open()) recv_parity1_acceptor_.close();
    if (recv_parity0_acceptor_.is_open()) recv_parity0_acceptor_.close();
    if (recv_data1_acceptor_.is_open()) recv_data1_acceptor_.close();
    
    // Load mode sockets (rank2 receiver)
    if (load_recv_rank0_data2_socket_.is_open()) load_recv_rank0_data2_socket_.close();
    if (load_recv_rank0_parity2_socket_.is_open()) load_recv_rank0_parity2_socket_.close();
    if (load_recv_rank1_data1_socket_.is_open()) load_recv_rank1_data1_socket_.close();
    if (load_recv_rank1_parity1_socket_.is_open()) load_recv_rank1_parity1_socket_.close();
    if (load_recv_rank3_data1_socket_.is_open()) load_recv_rank3_data1_socket_.close();
    if (load_recv_rank3_data2_socket_.is_open()) load_recv_rank3_data2_socket_.close();
    if (load_recv_rank0_data2_acceptor_.is_open()) load_recv_rank0_data2_acceptor_.close();
    if (load_recv_rank0_parity2_acceptor_.is_open()) load_recv_rank0_parity2_acceptor_.close();
    if (load_recv_rank1_data1_acceptor_.is_open()) load_recv_rank1_data1_acceptor_.close();
    if (load_recv_rank1_parity1_acceptor_.is_open()) load_recv_rank1_parity1_acceptor_.close();
    if (load_recv_rank3_data1_acceptor_.is_open()) load_recv_rank3_data1_acceptor_.close();
    if (load_recv_rank3_data2_acceptor_.is_open()) load_recv_rank3_data2_acceptor_.close();
    
    // Load mode sockets (other ranks sender)
    if (load_send_rank0_data2_socket_.is_open()) load_send_rank0_data2_socket_.close();
    if (load_send_rank0_parity2_socket_.is_open()) load_send_rank0_parity2_socket_.close();
    if (load_send_rank1_data1_socket_.is_open()) load_send_rank1_data1_socket_.close();
    if (load_send_rank1_parity1_socket_.is_open()) load_send_rank1_parity1_socket_.close();
    if (load_send_rank3_data1_socket_.is_open()) load_send_rank3_data1_socket_.close();
    if (load_send_rank3_data2_socket_.is_open()) load_send_rank3_data2_socket_.close();
}


bool send_with_size(boost::asio::ip::tcp::socket& sock, uintptr_t addr, size_t size) {
    try {
        uint32_t sz_net = htonl(static_cast<uint32_t>(size));
        boost::asio::write(sock, boost::asio::buffer(&sz_net, sizeof(uint32_t)));
        boost::asio::write(sock, boost::asio::buffer(reinterpret_cast<void*>(addr), size));
        return true;
    } catch (...) {
        return false;
    }
}

bool recv_with_size_bool(boost::asio::ip::tcp::socket& sock, void* buf, size_t size) {
    try {
        uint32_t sz_net = 0;
        boost::asio::read(sock, boost::asio::buffer(&sz_net, sizeof(uint32_t)));
        if (ntohl(sz_net) != size) {
            return false;
        }
        boost::asio::read(sock, boost::asio::buffer(buf, size));
        return true;
    } catch (...) {
        return false;
    }
}


struct SendTask {
    uintptr_t addr{0};
    size_t size{0};
};

struct RecvTask {
    uintptr_t addr{0};
    size_t size{0};
};

class ECNaiveNative {
public:
    ECNaiveNative(const std::string& send_data1_ip, uint16_t send_data1_port,
                  const std::string& send_parity0_ip, uint16_t send_parity0_port,
                  const std::string& send_parity1_ip, uint16_t send_parity1_port,
                  const std::string& recv_parity1_ip, uint16_t recv_parity1_port,
                  const std::string& recv_parity0_ip, uint16_t recv_parity0_port,
                  const std::string& recv_data1_ip, uint16_t recv_data1_port)
        : stop_(false),
          send_data1_ip_(send_data1_ip),
          send_data1_port_(send_data1_port),
          send_parity0_ip_(send_parity0_ip),
          send_parity0_port_(send_parity0_port),
          send_parity1_ip_(send_parity1_ip),
          send_parity1_port_(send_parity1_port),
          recv_parity1_ip_(recv_parity1_ip),
          recv_parity1_port_(recv_parity1_port),
          recv_parity0_ip_(recv_parity0_ip),
          recv_parity0_port_(recv_parity0_port),
          recv_data1_ip_(recv_data1_ip),
          recv_data1_port_(recv_data1_port),
          k_(2),
          rows_(2),
          a_mat_(nullptr),
          g_tbls_(nullptr) {
        // Initialize EC encoding tables
        init_ec_encoding();
        
        std::cout << "ECNAIVE: Initializing connections..." << std::endl;
        init_connections();
        start_threads();
        std::cout << "ECNAIVE: Pipeline started successfully" << std::endl;
    }

    ~ECNaiveNative() {
        stop();
        
        // Free EC encoding tables
        if (g_tbls_ != nullptr) {
            free(g_tbls_);
            g_tbls_ = nullptr;
        }
        if (a_mat_ != nullptr) {
            free(a_mat_);
            a_mat_ = nullptr;
        }
    }

    // Save mode pipelines: 3 sends + 3 receives
    void submit_send_data1(uintptr_t send_addr, size_t size) {
        std::cout << "ECNAIVE: Submitting send_data1 task: send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(send_data1_mutex_);
            send_data1_q_.push({send_addr, size});
        }
        send_data1_cv_.notify_one();
    }

    void submit_send_parity0(uintptr_t send_addr, size_t size) {
        std::cout << "ECNAIVE: Submitting send_parity0 task: send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(send_parity0_mutex_);
            send_parity0_q_.push({send_addr, size});
        }
        send_parity0_cv_.notify_one();
    }

    void submit_send_parity1(uintptr_t send_addr, size_t size) {
        std::cout << "ECNAIVE: Submitting send_parity1 task: send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(send_parity1_mutex_);
            send_parity1_q_.push({send_addr, size});
        }
        send_parity1_cv_.notify_one();
    }

    void submit_recv_parity1(uintptr_t recv_addr, size_t size) {
        std::cout << "ECNAIVE: Submitting recv_parity1 task: recv_addr=" << recv_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_parity1_mutex_);
            recv_parity1_q_.push({recv_addr, size});
        }
        recv_parity1_cv_.notify_one();
    }

    void submit_recv_parity0(uintptr_t recv_addr, size_t size) {
        std::cout << "ECNAIVE: Submitting recv_parity0 task: recv_addr=" << recv_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_parity0_mutex_);
            recv_parity0_q_.push({recv_addr, size});
        }
        recv_parity0_cv_.notify_one();
    }

    void submit_recv_data1(uintptr_t recv_addr, size_t size) {
        std::cout << "ECNAIVE: Submitting recv_data1 task: recv_addr=" << recv_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_data1_mutex_);
            recv_data1_q_.push({recv_addr, size});
        }
        recv_data1_cv_.notify_one();
    }

    // Unified save mode submit function: encode and submit send/recv tasks
    // rank i: keep d_{i0}, send d_{i1} to rank i+1, p_{i0} to rank i+2, p_{i1} to rank i+3
    // rank i: recv p_{i+1,1} from rank i+1, p_{i+2,0} from rank i+2, d_{i+3,1} from rank i+3
    void submit_ecnaive_save(uintptr_t data0_addr,      // d_{i0} - keep, not sent
                             uintptr_t data1_addr,      // d_{i1} - send to rank i+1
                             uintptr_t parity0_addr,    // p_{i0} - send to rank i+2
                             uintptr_t parity1_addr,    // p_{i1} - send to rank i+3
                             uintptr_t recv_parity1_addr, // recv p_{i+1,1} from rank i+1
                             uintptr_t recv_parity0_addr, // recv p_{i+2,0} from rank i+2
                             uintptr_t recv_data1_addr,   // recv d_{i+3,1} from rank i+3
                             size_t size) {
        std::cout << "ECNAIVE: Submitting save task: data0=" << data0_addr
                  << ", data1=" << data1_addr
                  << ", parity0=" << parity0_addr
                  << ", parity1=" << parity1_addr
                  << ", recv_p1=" << recv_parity1_addr
                  << ", recv_p0=" << recv_parity0_addr
                  << ", recv_d1=" << recv_data1_addr
                  << ", size=" << size << std::endl;
        
        // Step 1: Encode data blocks to get parity blocks
        encode_ec_blocks(data0_addr, data1_addr, parity0_addr, parity1_addr, size);
        
        // Step 2: Submit send tasks (data1, parity0, parity1)
        submit_send_data1(data1_addr, size);
        submit_send_parity0(parity0_addr, size);
        submit_send_parity1(parity1_addr, size);
        
        // Step 3: Submit receive tasks (recv_parity1, recv_parity0, recv_data1)
        submit_recv_parity1(recv_parity1_addr, size);
        submit_recv_parity0(recv_parity0_addr, size);
        submit_recv_data1(recv_data1_addr, size);
    }

    // Submit sentinels to signal pipeline completion
    void submit_send_data1_sentinel() {
        std::cout << "ECNAIVE: Submitting sentinel to send_data1 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(send_data1_mutex_);
            send_data1_q_.push({0, 0});
        }
        send_data1_cv_.notify_one();
    }

    void submit_send_parity0_sentinel() {
        std::cout << "ECNAIVE: Submitting sentinel to send_parity0 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(send_parity0_mutex_);
            send_parity0_q_.push({0, 0});
        }
        send_parity0_cv_.notify_one();
    }

    void submit_send_parity1_sentinel() {
        std::cout << "ECNAIVE: Submitting sentinel to send_parity1 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(send_parity1_mutex_);
            send_parity1_q_.push({0, 0});
        }
        send_parity1_cv_.notify_one();
    }

    void submit_recv_parity1_sentinel() {
        std::cout << "ECNAIVE: Submitting sentinel to recv_parity1 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_parity1_mutex_);
            recv_parity1_q_.push({0, 0});
        }
        recv_parity1_cv_.notify_one();
    }

    void submit_recv_parity0_sentinel() {
        std::cout << "ECNAIVE: Submitting sentinel to recv_parity0 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_parity0_mutex_);
            recv_parity0_q_.push({0, 0});
        }
        recv_parity0_cv_.notify_one();
    }

    void submit_recv_data1_sentinel() {
        std::cout << "ECNAIVE: Submitting sentinel to recv_data1 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(recv_data1_mutex_);
            recv_data1_q_.push({0, 0});
        }
        recv_data1_cv_.notify_one();
    }

    // Release helpers: Python can poll these to free buffers.
    // Data buffers (data1_addr) are released after send operations complete
    std::vector<uintptr_t> get_data_buffers_to_release() {
        std::vector<uintptr_t> buffers;
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        while (!data_buffers_to_release_.empty()) {
            buffers.push_back(data_buffers_to_release_.front());
            data_buffers_to_release_.pop();
        }
        return buffers;
    }
    
    // Parity buffers (parity0_addr, parity1_addr) are released after send operations complete
    std::vector<uintptr_t> get_parity_buffers_to_release() {
        std::vector<uintptr_t> buffers;
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        while (!parity_buffers_to_release_.empty()) {
            buffers.push_back(parity_buffers_to_release_.front());
            parity_buffers_to_release_.pop();
        }
        return buffers;
    }

    void reset_encoding_completion_flags() {
        // Save mode flags
        send_data1_completed_ = false;
        send_parity0_completed_ = false;
        send_parity1_completed_ = false;
        recv_parity1_completed_ = false;
        recv_parity0_completed_ = false;
        recv_data1_completed_ = false;
        send_data1_sentinel_received_ = false;
        send_parity0_sentinel_received_ = false;
        send_parity1_sentinel_received_ = false;
        recv_parity1_sentinel_received_ = false;
        recv_parity0_sentinel_received_ = false;
        recv_data1_sentinel_received_ = false;

        // Clear queues
        {
            std::lock_guard<std::mutex> lock(send_data1_mutex_);
            while (!send_data1_q_.empty()) send_data1_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(send_parity0_mutex_);
            while (!send_parity0_q_.empty()) send_parity0_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(send_parity1_mutex_);
            while (!send_parity1_q_.empty()) send_parity1_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(recv_parity1_mutex_);
            while (!recv_parity1_q_.empty()) recv_parity1_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(recv_parity0_mutex_);
            while (!recv_parity0_q_.empty()) recv_parity0_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(recv_data1_mutex_);
            while (!recv_data1_q_.empty()) recv_data1_q_.pop();
        }

        std::cout << "ECNAIVE: Reset encoding completion flags and cleared all queues" << std::endl;
    }

    void wait_for_encoding_completion() {
        // Wait for all 6 workers
        int wait_count = 0;
        while (!send_data1_completed_ || !send_parity0_completed_ || !send_parity1_completed_ ||
               !recv_parity1_completed_ || !recv_parity0_completed_ || !recv_data1_completed_) {
            if (wait_count % 100 == 0) {
                std::cout << "ECNAIVE: Waiting for workers: "
                          << "s_d1=" << (send_data1_completed_ ? "true" : "false")
                          << ", s_p0=" << (send_parity0_completed_ ? "true" : "false")
                          << ", s_p1=" << (send_parity1_completed_ ? "true" : "false")
                          << ", r_p1=" << (recv_parity1_completed_ ? "true" : "false")
                          << ", r_p0=" << (recv_parity0_completed_ ? "true" : "false")
                          << ", r_d1=" << (recv_data1_completed_ ? "true" : "false") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
        }
        std::cout << "ECNAIVE: All workers completed" << std::endl;
    }

    void stop() {
        bool expected = false;
        if (!stop_.compare_exchange_strong(expected, true)) {
            return;  // already stopped
        }
        send_data1_cv_.notify_all();
        send_parity0_cv_.notify_all();
        send_parity1_cv_.notify_all();
        recv_parity1_cv_.notify_all();
        recv_parity0_cv_.notify_all();
        recv_data1_cv_.notify_all();
        if (send_data1_thread_.joinable()) send_data1_thread_.join();
        if (send_parity0_thread_.joinable()) send_parity0_thread_.join();
        if (send_parity1_thread_.joinable()) send_parity1_thread_.join();
        if (recv_parity1_thread_.joinable()) recv_parity1_thread_.join();
        if (recv_parity0_thread_.joinable()) recv_parity0_thread_.join();
        if (recv_data1_thread_.joinable()) recv_data1_thread_.join();
        conn_.cleanup();
    }

    // Load mode functions
    void set_load_mode(bool is_load, int failed_rank) {
        is_load_mode_ = is_load;
        failed_rank_ = failed_rank;
        std::cout << "ECLATIN: Set load mode: " 
                  << (is_load ? "true" : "false") << ", failed_rank=" << failed_rank << std::endl;
    }

    void init_load_connections(
        int rank,
        const std::string& rank2_ip,
        uint16_t load_recv_rank0_data2_port,
        uint16_t load_recv_rank0_parity2_port,
        uint16_t load_recv_rank1_data1_port,
        uint16_t load_recv_rank1_parity1_port,
        uint16_t load_recv_rank3_data1_port,
        uint16_t load_recv_rank3_data2_port
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: init_load_connections called but not in load mode" << std::endl;
            return;
        }
        
        std::cout << "ECLATIN: [Rank " << rank << "] Initializing load connections..." << std::endl;
        
        if (rank == 2) {
            // rank2: Initialize 6 recv sockets (accept connections from rank0/1/3)
            // Step 1: First, bind and listen all acceptors synchronously (before accept)
            std::cout << "ECLATIN: [Rank 2] Binding and listening all acceptors..." << std::endl;
            try {
                conn_.bind_listen_load_recv_rank0_data2(rank2_ip, load_recv_rank0_data2_port);
                conn_.bind_listen_load_recv_rank0_parity2(rank2_ip, load_recv_rank0_parity2_port);
                conn_.bind_listen_load_recv_rank1_data1(rank2_ip, load_recv_rank1_data1_port);
                conn_.bind_listen_load_recv_rank1_parity1(rank2_ip, load_recv_rank1_parity1_port);
                conn_.bind_listen_load_recv_rank3_data1(rank2_ip, load_recv_rank3_data1_port);
                conn_.bind_listen_load_recv_rank3_data2(rank2_ip, load_recv_rank3_data2_port);
                
                std::cout << "ECLATIN: [Rank 2] All acceptors bound and listening" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "ECLATIN: [Rank 2] Failed to bind/listen acceptors: " << e.what() << std::endl;
                throw;
            }
            
            // Step 2: Start accept operations in separate threads (similar to EC-CHECK)
            // These threads will block on accept() until connections arrive
            std::thread recv_init_thread([this]() {
                std::thread r0_d2([this]() {
                    conn_.accept_load_recv_rank0_data2();
                });
                std::thread r0_p2([this]() {
                    conn_.accept_load_recv_rank0_parity2();
                });
                std::thread r1_d1([this]() {
                    conn_.accept_load_recv_rank1_data1();
                });
                std::thread r1_p1([this]() {
                    conn_.accept_load_recv_rank1_parity1();
                });
                std::thread r3_d1([this]() {
                    conn_.accept_load_recv_rank3_data1();
                });
                std::thread r3_d2([this]() {
                    conn_.accept_load_recv_rank3_data2();
                });
                r0_d2.join();
                r0_p2.join();
                r1_d1.join();
                r1_p1.join();
                r3_d1.join();
                r3_d2.join();
            });
            
            // Step 3: Small delay to ensure accept sockets are bound and listening (similar to EC-CHECK)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Step 4: Detach the recv_init_thread so it runs in background
            // The accept operations will block until connections arrive from rank0/1/3
            recv_init_thread.detach();
            
            std::cout << "ECLATIN: [Rank 2] Accept threads started, waiting for connections..." << std::endl;
        } else {
            // rank0/1/3: Initialize 2 send sockets each (connect to rank2)
            if (rank == 0) {
                std::cout << "ECLATIN: [Rank 0] Connecting load send sockets to rank2..." << std::endl;
                conn_.init_load_send_rank0_data2(rank2_ip, load_recv_rank0_data2_port);
                conn_.init_load_send_rank0_parity2(rank2_ip, load_recv_rank0_parity2_port);
                std::cout << "ECLATIN: [Rank 0] Load send sockets connected" << std::endl;
            } else if (rank == 1) {
                std::cout << "ECLATIN: [Rank 1] Connecting load send sockets to rank2..." << std::endl;
                conn_.init_load_send_rank1_data1(rank2_ip, load_recv_rank1_data1_port);
                conn_.init_load_send_rank1_parity1(rank2_ip, load_recv_rank1_parity1_port);
                std::cout << "ECLATIN: [Rank 1] Load send sockets connected" << std::endl;
            } else if (rank == 3) {
                std::cout << "ECLATIN: [Rank 3] Connecting load send sockets to rank2..." << std::endl;
                conn_.init_load_send_rank3_data1(rank2_ip, load_recv_rank3_data1_port);
                conn_.init_load_send_rank3_data2(rank2_ip, load_recv_rank3_data2_port);
                std::cout << "ECLATIN: [Rank 3] Load send sockets connected" << std::endl;
            }
        }
        
            std::cout << "ECLATIN: [Rank " << rank << "] Load connections initialized" << std::endl;
    }
    
    void wait_for_load_connections(int timeout_seconds = 30) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: wait_for_load_connections called but not in load mode" << std::endl;
            return;
        }
        conn_.wait_for_load_connections(timeout_seconds);
    }

    // Unified recovery interface for rank2 (parallel recv + parallel XOR)
    void load_recover(
        // Receive buffers (6 blocks from other ranks)
        uintptr_t rank0_data2_addr,
        uintptr_t rank0_parity2_addr,
        uintptr_t rank1_data1_addr,
        uintptr_t rank1_parity1_addr,
        uintptr_t rank3_data1_addr,
        uintptr_t rank3_data2_addr,
        // Recovered buffers (4 blocks to write results)
        uintptr_t recovered_data1_addr,
        uintptr_t recovered_data2_addr,
        uintptr_t recovered_parity1_addr,
        uintptr_t recovered_parity2_addr,
        size_t size
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: [Rank 2] load_recover called but not in load mode" << std::endl;
            return;
        }
        
        std::cout << "ECLATIN: [Rank 2] Starting recovery (size=" << size << ")" << std::endl;
        
        // Step 1: Parallel receive all 6 blocks using threads
        std::vector<std::exception_ptr> recv_exceptions(6);
        std::vector<std::thread> recv_threads;
        
        recv_threads.emplace_back([&]() {
            try {
                if (!recv_with_size_bool(conn_.get_load_recv_rank0_data2_socket(), 
                                        reinterpret_cast<void*>(rank0_data2_addr), size)) {
                    throw std::runtime_error("Failed to receive rank0_data2");
                }
            } catch (...) {
                recv_exceptions[0] = std::current_exception();
            }
        });
        
        recv_threads.emplace_back([&]() {
            try {
                if (!recv_with_size_bool(conn_.get_load_recv_rank0_parity2_socket(), 
                                        reinterpret_cast<void*>(rank0_parity2_addr), size)) {
                    throw std::runtime_error("Failed to receive rank0_parity2");
                }
            } catch (...) {
                recv_exceptions[1] = std::current_exception();
            }
        });
        
        recv_threads.emplace_back([&]() {
            try {
                if (!recv_with_size_bool(conn_.get_load_recv_rank1_data1_socket(), 
                                        reinterpret_cast<void*>(rank1_data1_addr), size)) {
                    throw std::runtime_error("Failed to receive rank1_data1");
                }
            } catch (...) {
                recv_exceptions[2] = std::current_exception();
            }
        });
        
        recv_threads.emplace_back([&]() {
            try {
                if (!recv_with_size_bool(conn_.get_load_recv_rank1_parity1_socket(), 
                                        reinterpret_cast<void*>(rank1_parity1_addr), size)) {
                    throw std::runtime_error("Failed to receive rank1_parity1");
                }
            } catch (...) {
                recv_exceptions[3] = std::current_exception();
            }
        });
        
        recv_threads.emplace_back([&]() {
            try {
                if (!recv_with_size_bool(conn_.get_load_recv_rank3_data1_socket(), 
                                        reinterpret_cast<void*>(rank3_data1_addr), size)) {
                    throw std::runtime_error("Failed to receive rank3_data1");
                }
            } catch (...) {
                recv_exceptions[4] = std::current_exception();
            }
        });
        
        recv_threads.emplace_back([&]() {
            try {
                if (!recv_with_size_bool(conn_.get_load_recv_rank3_data2_socket(), 
                                        reinterpret_cast<void*>(rank3_data2_addr), size)) {
                    throw std::runtime_error("Failed to receive rank3_data2");
                }
            } catch (...) {
                recv_exceptions[5] = std::current_exception();
            }
        });
        
        // Join all receive threads
        for (auto& t : recv_threads) {
            t.join();
        }
        
        // Check for exceptions
        for (size_t i = 0; i < recv_exceptions.size(); ++i) {
            if (recv_exceptions[i]) {
                std::rethrow_exception(recv_exceptions[i]);
            }
        }
        
        std::cout << "ECLATIN: [Rank 2] All 6 blocks received" << std::endl;
        
        // Step 2: Parallel XOR recoveries using threads
        std::vector<std::exception_ptr> xor_exceptions(4);
        std::vector<std::thread> xor_threads;
        
        // data1 = rank0.data2 XOR rank1.parity1
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_data1_addr), 
                           reinterpret_cast<void*>(rank0_data2_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_data1_addr), 
                                     reinterpret_cast<void*>(rank1_parity1_addr)};
                xor_gen(2, static_cast<int>(size), xor_array);
            } catch (...) {
                xor_exceptions[0] = std::current_exception();
            }
        });
        
        // data2 = rank0.parity2 XOR rank1.data1
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_data2_addr), 
                           reinterpret_cast<void*>(rank0_parity2_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_data2_addr), 
                                     reinterpret_cast<void*>(rank1_data1_addr)};
                xor_gen(2, static_cast<int>(size), xor_array);
            } catch (...) {
                xor_exceptions[1] = std::current_exception();
            }
        });
        
        // parity1 = rank1.data1 XOR rank3.data2
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_parity1_addr), 
                           reinterpret_cast<void*>(rank1_data1_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_parity1_addr), 
                                     reinterpret_cast<void*>(rank3_data2_addr)};
                xor_gen(2, static_cast<int>(size), xor_array);
            } catch (...) {
                xor_exceptions[2] = std::current_exception();
            }
        });
        
        // parity2 = rank0.data2 XOR rank3.data1
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_parity2_addr), 
                           reinterpret_cast<void*>(rank0_data2_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_parity2_addr), 
                                     reinterpret_cast<void*>(rank3_data1_addr)};
                xor_gen(2, static_cast<int>(size), xor_array);
            } catch (...) {
                xor_exceptions[3] = std::current_exception();
            }
        });
        
        // Join all XOR threads
        for (auto& t : xor_threads) {
            t.join();
        }
        
        // Check for exceptions
        for (size_t i = 0; i < xor_exceptions.size(); ++i) {
            if (xor_exceptions[i]) {
                std::rethrow_exception(xor_exceptions[i]);
            }
        }
        
        std::cout << "ECLATIN: [Rank 2] Recovery completed successfully" << std::endl;
    }

    // Unified send interface for other ranks (rank1, rank2, rank3) - parallel send two blocks
    void load_send_blocks(
        const std::string& block1_name,
        uintptr_t block1_addr,
        const std::string& block2_name,
        uintptr_t block2_addr,
        size_t size
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: load_send_blocks called but not in load mode" << std::endl;
            return;
        }
        
        std::cout << "ECLATIN: Starting parallel send of " << block1_name 
                  << " and " << block2_name << " to rank2 (size=" << size << ")" << std::endl;
        
        // Helper function to get socket by block name
        auto get_socket = [this](const std::string& block_name) -> boost::asio::ip::tcp::socket* {
            if (block_name == "rank0_data2") {
                return &conn_.get_load_send_rank0_data2_socket();
            } else if (block_name == "rank0_parity2") {
                return &conn_.get_load_send_rank0_parity2_socket();
            } else if (block_name == "rank1_data1") {
                return &conn_.get_load_send_rank1_data1_socket();
            } else if (block_name == "rank1_parity1") {
                return &conn_.get_load_send_rank1_parity1_socket();
            } else if (block_name == "rank3_data1") {
                return &conn_.get_load_send_rank3_data1_socket();
            } else if (block_name == "rank3_data2") {
                return &conn_.get_load_send_rank3_data2_socket();
            }
            return nullptr;
        };
        
        // Parallel send using two threads
        std::exception_ptr thread1_exception = nullptr;
        std::exception_ptr thread2_exception = nullptr;
        
        std::thread thread1([&]() {
            try {
                boost::asio::ip::tcp::socket* sock = get_socket(block1_name);
                if (sock == nullptr || !sock->is_open()) {
                    throw std::runtime_error("ECLATIN: load_send_blocks socket not available for " + block1_name);
                }
                
                std::cout << "ECLATIN: Sending " << block1_name << " to rank2 (size=" << size << ")" << std::endl;
                
                if (!send_with_size(*sock, block1_addr, size)) {
                    throw std::runtime_error("ECLATIN: load_send_blocks send failed for " + block1_name);
                }
                
                std::cout << "ECLATIN: Successfully sent " << block1_name << " to rank2" << std::endl;
            } catch (...) {
                thread1_exception = std::current_exception();
            }
        });
        
        std::thread thread2([&]() {
            try {
                boost::asio::ip::tcp::socket* sock = get_socket(block2_name);
                if (sock == nullptr || !sock->is_open()) {
                    throw std::runtime_error("ECLATIN: load_send_blocks socket not available for " + block2_name);
                }
                
                std::cout << "ECLATIN: Sending " << block2_name << " to rank2 (size=" << size << ")" << std::endl;
                
                if (!send_with_size(*sock, block2_addr, size)) {
                    throw std::runtime_error("ECLATIN: load_send_blocks send failed for " + block2_name);
                }
                
                std::cout << "ECLATIN: Successfully sent " << block2_name << " to rank2" << std::endl;
            } catch (...) {
                thread2_exception = std::current_exception();
            }
        });
        
        // Join both threads
        thread1.join();
        thread2.join();
        
        // Check for exceptions
        if (thread1_exception) {
            std::rethrow_exception(thread1_exception);
        }
        if (thread2_exception) {
            std::rethrow_exception(thread2_exception);
        }
        
        std::cout << "ECLATIN: Both blocks sent successfully" << std::endl;
    }

private:
    std::atomic<bool> stop_;

    // ASIO connections for pipelines
    AsioConnectionManager conn_;
    
    // Save mode network config
    std::string send_data1_ip_;
    uint16_t send_data1_port_;
    std::string send_parity0_ip_;
    uint16_t send_parity0_port_;
    std::string send_parity1_ip_;
    uint16_t send_parity1_port_;
    std::string recv_parity1_ip_;
    uint16_t recv_parity1_port_;
    std::string recv_parity0_ip_;
    uint16_t recv_parity0_port_;
    std::string recv_data1_ip_;
    uint16_t recv_data1_port_;

    // EC encoding parameters (k=2, rows=2 for ecnaive)
    int k_;
    int rows_;
    unsigned char* a_mat_;    // RS matrix (k * m, where m = k + rows = 4)
    unsigned char* g_tbls_;   // EC encoding tables (32 * k * rows)

    // Save mode pipelines: 3 sends + 3 receives
    std::queue<SendTask> send_data1_q_;
    std::mutex send_data1_mutex_;
    std::condition_variable send_data1_cv_;
    std::queue<SendTask> send_parity0_q_;
    std::mutex send_parity0_mutex_;
    std::condition_variable send_parity0_cv_;
    std::queue<SendTask> send_parity1_q_;
    std::mutex send_parity1_mutex_;
    std::condition_variable send_parity1_cv_;

    std::queue<RecvTask> recv_parity1_q_;
    std::mutex recv_parity1_mutex_;
    std::condition_variable recv_parity1_cv_;
    std::queue<RecvTask> recv_parity0_q_;
    std::mutex recv_parity0_mutex_;
    std::condition_variable recv_parity0_cv_;
    std::queue<RecvTask> recv_data1_q_;
    std::mutex recv_data1_mutex_;
    std::condition_variable recv_data1_cv_;

    // Separate release queues for data and parity buffers
    std::queue<uintptr_t> data_buffers_to_release_;
    std::queue<uintptr_t> parity_buffers_to_release_;
    std::mutex release_queue_mutex_;

    // Completion flags
    std::atomic<bool> send_data1_completed_{false};
    std::atomic<bool> send_parity0_completed_{false};
    std::atomic<bool> send_parity1_completed_{false};
    std::atomic<bool> recv_parity1_completed_{false};
    std::atomic<bool> recv_parity0_completed_{false};
    std::atomic<bool> recv_data1_completed_{false};

    // Sentinel received flags
    std::atomic<bool> send_data1_sentinel_received_{false};
    std::atomic<bool> send_parity0_sentinel_received_{false};
    std::atomic<bool> send_parity1_sentinel_received_{false};
    std::atomic<bool> recv_parity1_sentinel_received_{false};
    std::atomic<bool> recv_parity0_sentinel_received_{false};
    std::atomic<bool> recv_data1_sentinel_received_{false};

    std::thread send_data1_thread_;
    std::thread send_parity0_thread_;
    std::thread send_parity1_thread_;
    std::thread recv_parity1_thread_;
    std::thread recv_parity0_thread_;
    std::thread recv_data1_thread_;
    
    // Load mode flags
    std::atomic<bool> is_load_mode_{false};
    int failed_rank_{-1};

    // EC encoding initialization
    void init_ec_encoding() {
        int m = k_ + rows_;  // m = 2 + 2 = 4
        
        // Allocate RS matrix (k * m = 2 * 4)
        a_mat_ = (unsigned char*)malloc((size_t)k_ * (size_t)m);
        if (a_mat_ == nullptr) {
            std::cerr << "ECNAIVE: Failed to allocate EC encoding matrix" << std::endl;
            throw std::runtime_error("ECNAIVE: Failed to allocate EC encoding matrix");
        }
        
        // Generate RS matrix
        gf_gen_rs_matrix(a_mat_, m, k_);
        
        // Allocate encoding tables: 32 * k * rows = 32 * 2 * 2
        size_t gtbls_size = 32 * (size_t)k_ * (size_t)rows_;
        void* tmp = nullptr;
        if (posix_memalign(&tmp, 32, gtbls_size) != 0) tmp = nullptr;
        if (tmp == nullptr) tmp = malloc(gtbls_size);
        g_tbls_ = reinterpret_cast<unsigned char*>(tmp);
        if (g_tbls_ == nullptr) {
            std::cerr << "ECNAIVE: Failed to allocate EC encoding tables" << std::endl;
            free(a_mat_);
            a_mat_ = nullptr;
            throw std::runtime_error("ECNAIVE: Failed to allocate EC encoding tables");
        }
        
        // Initialize tables using isa-l
        ec_init_tables(k_, rows_, a_mat_, g_tbls_);
        std::cout << "ECNAIVE: EC encoding tables initialized (k=" << k_ << ", rows=" << rows_ << ")" << std::endl;
    }
    
    // EC encoding function: encode 2 data blocks to 2 parity blocks
    void encode_ec_blocks(uintptr_t data0_addr, uintptr_t data1_addr,
                          uintptr_t parity0_addr, uintptr_t parity1_addr,
                          size_t size) {
        if (g_tbls_ == nullptr || a_mat_ == nullptr) {
            std::cerr << "ECNAIVE: ERROR: EC encoding tables not initialized!" << std::endl;
            throw std::runtime_error("ECNAIVE: EC encoding tables not initialized");
        }
        
        unsigned char* srcs[2];
        unsigned char* dests[2];
        srcs[0] = reinterpret_cast<unsigned char*>(data0_addr);
        srcs[1] = reinterpret_cast<unsigned char*>(data1_addr);
        dests[0] = reinterpret_cast<unsigned char*>(parity0_addr);
        dests[1] = reinterpret_cast<unsigned char*>(parity1_addr);
        
        // Use isa-l ec_encode_data: encode 2 data blocks to 2 parity blocks
        ec_encode_data((int)size, k_, rows_, g_tbls_, srcs, dests);
    }

    void start_threads() {
        std::cout << "ECNAIVE: Starting worker threads..." << std::endl;
        send_data1_thread_ = std::thread(&ECNaiveNative::send_data1_worker, this);
        send_parity0_thread_ = std::thread(&ECNaiveNative::send_parity0_worker, this);
        send_parity1_thread_ = std::thread(&ECNaiveNative::send_parity1_worker, this);
        recv_parity1_thread_ = std::thread(&ECNaiveNative::recv_parity1_worker, this);
        recv_parity0_thread_ = std::thread(&ECNaiveNative::recv_parity0_worker, this);
        recv_data1_thread_ = std::thread(&ECNaiveNative::recv_data1_worker, this);
        std::cout << "ECNAIVE: All worker threads started" << std::endl;
    }

    void init_connections() {
        std::cout << "ECNAIVE: Initializing connections..." << std::endl;
        // Start acceptors in separate threads to avoid deadlock (mirror eccheck pattern)
        std::thread recv_init_thread([this]() {
            std::thread r1([this]() { conn_.init_recv_parity1(recv_parity1_ip_, recv_parity1_port_); });
            std::thread r2([this]() { conn_.init_recv_parity0(recv_parity0_ip_, recv_parity0_port_); });
            std::thread r3([this]() { conn_.init_recv_data1(recv_data1_ip_, recv_data1_port_); });
            r1.join();
            r2.join();
            r3.join();
        });

        // Small delay to ensure acceptors are listening
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Connect send sockets (blocking)
        std::cout << "ECNAIVE: Connecting send_data1 socket..." << std::endl;
        conn_.init_send_data1(send_data1_ip_, send_data1_port_);
        std::cout << "ECNAIVE: Connecting send_parity0 socket..." << std::endl;
        conn_.init_send_parity0(send_parity0_ip_, send_parity0_port_);
        std::cout << "ECNAIVE: Connecting send_parity1 socket..." << std::endl;
        conn_.init_send_parity1(send_parity1_ip_, send_parity1_port_);

        recv_init_thread.join();
        std::cout << "ECNAIVE: Waiting for all connections..." << std::endl;
        conn_.wait_for_connections();
        std::cout << "ECNAIVE: All connections established" << std::endl;
    }


    // Save mode workers: 3 sends + 3 receives
    void recv_parity1_worker() {
        std::cout << "ECNAIVE: RecvParity1 worker started" << std::endl;
        while (!stop_) {
            RecvTask task;
            {
                std::unique_lock<std::mutex> lk(recv_parity1_mutex_);
                recv_parity1_cv_.wait(lk, [this] { return stop_ || !recv_parity1_q_.empty(); });
                if (stop_) break;
                task = recv_parity1_q_.front();
                recv_parity1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                recv_parity1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(recv_parity1_mutex_);
                    if (recv_parity1_q_.empty()) {
                        recv_parity1_completed_ = true;
                        std::cout << "ECNAIVE: RecvParity1 worker completed" << std::endl;
                        recv_parity1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }

            // Ensure connection is ready
            if (!conn_.is_recv_parity1_connected()) {
                std::cerr << "ECNAIVE: ERROR: recv_parity1 socket not connected!" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_parity1 socket not connected");
            }

            // Receive data directly into persistent buffer (no release needed)
                    if (!recv_with_size_bool(
                    conn_.get_recv_parity1_socket(),
                    reinterpret_cast<void*>(task.addr),
                            task.size)) {
                std::cerr << "ECNAIVE: recv_parity1_with_size_bool returned false" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_parity1_with_size_bool returned false");
            }

            if (recv_parity1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_parity1_mutex_);
                if (recv_parity1_q_.empty()) {
                    recv_parity1_completed_ = true;
                    std::cout << "ECNAIVE: RecvParity1 worker completed" << std::endl;
                    recv_parity1_sentinel_received_ = false;
                }
            }
        }
    }

    void recv_parity0_worker() {
        std::cout << "ECNAIVE: RecvParity0 worker started" << std::endl;
        while (!stop_) {
            RecvTask task;
            {
                std::unique_lock<std::mutex> lk(recv_parity0_mutex_);
                recv_parity0_cv_.wait(lk, [this] { return stop_ || !recv_parity0_q_.empty(); });
                if (stop_) break;
                task = recv_parity0_q_.front();
                recv_parity0_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                recv_parity0_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(recv_parity0_mutex_);
                    if (recv_parity0_q_.empty()) {
                        recv_parity0_completed_ = true;
                        std::cout << "ECNAIVE: RecvParity0 worker completed" << std::endl;
                        recv_parity0_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }

            // Ensure connection is ready
            if (!conn_.is_recv_parity0_connected()) {
                std::cerr << "ECNAIVE: ERROR: recv_parity0 socket not connected!" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_parity0 socket not connected");
            }

            // Receive data directly into persistent buffer (no release needed)
            if (!recv_with_size_bool(
                    conn_.get_recv_parity0_socket(),
                    reinterpret_cast<void*>(task.addr),
                    task.size)) {
                std::cerr << "ECNAIVE: recv_parity0_with_size_bool returned false" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_parity0_with_size_bool returned false");
            }

            if (recv_parity0_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_parity0_mutex_);
                if (recv_parity0_q_.empty()) {
                    recv_parity0_completed_ = true;
                    std::cout << "ECNAIVE: RecvParity0 worker completed" << std::endl;
                    recv_parity0_sentinel_received_ = false;
                }
            }
        }
    }

    void recv_data1_worker() {
        std::cout << "ECNAIVE: RecvData1 worker started" << std::endl;
        while (!stop_) {
            RecvTask task;
            {
                std::unique_lock<std::mutex> lk(recv_data1_mutex_);
                recv_data1_cv_.wait(lk, [this] { return stop_ || !recv_data1_q_.empty(); });
                if (stop_) break;
                task = recv_data1_q_.front();
                recv_data1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                recv_data1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(recv_data1_mutex_);
                    if (recv_data1_q_.empty()) {
                        recv_data1_completed_ = true;
                        std::cout << "ECNAIVE: RecvData1 worker completed" << std::endl;
                        recv_data1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }

            // Ensure connection is ready
            if (!conn_.is_recv_data1_connected()) {
                std::cerr << "ECNAIVE: ERROR: recv_data1 socket not connected!" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_data1 socket not connected");
            }

            // Receive data directly into persistent buffer (no release needed)
            if (!recv_with_size_bool(
                    conn_.get_recv_data1_socket(),
                    reinterpret_cast<void*>(task.addr),
                    task.size)) {
                std::cerr << "ECNAIVE: recv_data1_with_size_bool returned false" << std::endl;
                throw std::runtime_error("ECNAIVE: recv_data1_with_size_bool returned false");
            }

            if (recv_data1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(recv_data1_mutex_);
                if (recv_data1_q_.empty()) {
                    recv_data1_completed_ = true;
                    std::cout << "ECNAIVE: RecvData1 worker completed" << std::endl;
                    recv_data1_sentinel_received_ = false;
                }
            }
        }
    }

    void send_data1_worker() {
        std::cout << "ECNAIVE: SendData1 worker started" << std::endl;
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(send_data1_mutex_);
                send_data1_cv_.wait(lk, [this] { return stop_ || !send_data1_q_.empty(); });
                if (stop_) break;
                task = send_data1_q_.front();
                send_data1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                send_data1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(send_data1_mutex_);
                    if (send_data1_q_.empty()) {
                        send_data1_completed_ = true;
                        std::cout << "ECNAIVE: SendData1 worker completed" << std::endl;
                        send_data1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_send_data1_connected()) {
                send_with_size(conn_.get_send_data1_socket(), task.addr, task.size);
            }
            // Release data buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                data_buffers_to_release_.push(task.addr);
            }

            if (send_data1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(send_data1_mutex_);
                if (send_data1_q_.empty()) {
                    send_data1_completed_ = true;
                    std::cout << "ECNAIVE: SendData1 worker completed" << std::endl;
                    send_data1_sentinel_received_ = false;
                }
            }
        }
    }

    void send_parity0_worker() {
        std::cout << "ECNAIVE: SendParity0 worker started" << std::endl;
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(send_parity0_mutex_);
                send_parity0_cv_.wait(lk, [this] { return stop_ || !send_parity0_q_.empty(); });
                if (stop_) break;
                task = send_parity0_q_.front();
                send_parity0_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                send_parity0_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(send_parity0_mutex_);
                    if (send_parity0_q_.empty()) {
                        send_parity0_completed_ = true;
                        std::cout << "ECNAIVE: SendParity0 worker completed" << std::endl;
                        send_parity0_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_send_parity0_connected()) {
                send_with_size(conn_.get_send_parity0_socket(), task.addr, task.size);
            }
            // Release parity buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                parity_buffers_to_release_.push(task.addr);
            }

            if (send_parity0_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(send_parity0_mutex_);
                if (send_parity0_q_.empty()) {
                    send_parity0_completed_ = true;
                    std::cout << "ECNAIVE: SendParity0 worker completed" << std::endl;
                    send_parity0_sentinel_received_ = false;
                }
            }
        }
    }

    void send_parity1_worker() {
        std::cout << "ECNAIVE: SendParity1 worker started" << std::endl;
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(send_parity1_mutex_);
                send_parity1_cv_.wait(lk, [this] { return stop_ || !send_parity1_q_.empty(); });
                if (stop_) break;
                task = send_parity1_q_.front();
                send_parity1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                send_parity1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(send_parity1_mutex_);
                    if (send_parity1_q_.empty()) {
                        send_parity1_completed_ = true;
                        std::cout << "ECNAIVE: SendParity1 worker completed" << std::endl;
                        send_parity1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_send_parity1_connected()) {
                send_with_size(conn_.get_send_parity1_socket(), task.addr, task.size);
            }
            // Release parity buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                parity_buffers_to_release_.push(task.addr);
            }

            if (send_parity1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(send_parity1_mutex_);
                if (send_parity1_q_.empty()) {
                    send_parity1_completed_ = true;
                    std::cout << "ECNAIVE: SendParity1 worker completed" << std::endl;
                    send_parity1_sentinel_received_ = false;
                }
            }
        }
    }
};

}  // namespace

PYBIND11_MODULE(ecnaive_native, m) {
    pybind11::class_<ECNaiveNative>(m, "ECNaiveNative")
        .def(pybind11::init<const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t>())
        // Unified save mode submit function
        .def("submit_ecnaive_save", &ECNaiveNative::submit_ecnaive_save,
             "Unified save mode function: encode and submit send/recv tasks",
             pybind11::arg("data0_addr"),      // d_{i0} - keep, not sent
             pybind11::arg("data1_addr"),      // d_{i1} - send to rank i+1
             pybind11::arg("parity0_addr"),   // p_{i0} - send to rank i+2
             pybind11::arg("parity1_addr"),    // p_{i1} - send to rank i+3
             pybind11::arg("recv_parity1_addr"), // recv p_{i+1,1} from rank i+1
             pybind11::arg("recv_parity0_addr"), // recv p_{i+2,0} from rank i+2
             pybind11::arg("recv_data1_addr"),   // recv d_{i+3,1} from rank i+3
             pybind11::arg("size"))
        // Save mode submit functions: 3 sends + 3 receives (for fine-grained control if needed)
        .def("submit_send_data1", &ECNaiveNative::submit_send_data1,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_send_parity0", &ECNaiveNative::submit_send_parity0,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_send_parity1", &ECNaiveNative::submit_send_parity1,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_recv_parity1", &ECNaiveNative::submit_recv_parity1,
             pybind11::arg("recv_addr"),
             pybind11::arg("size"))
        .def("submit_recv_parity0", &ECNaiveNative::submit_recv_parity0,
             pybind11::arg("recv_addr"),
             pybind11::arg("size"))
        .def("submit_recv_data1", &ECNaiveNative::submit_recv_data1,
             pybind11::arg("recv_addr"),
             pybind11::arg("size"))
        // Common functions
        .def("get_data_buffers_to_release", &ECNaiveNative::get_data_buffers_to_release)
        .def("get_parity_buffers_to_release", &ECNaiveNative::get_parity_buffers_to_release)
        .def("reset_encoding_completion_flags", &ECNaiveNative::reset_encoding_completion_flags)
        .def("wait_for_encoding_completion", &ECNaiveNative::wait_for_encoding_completion)
        // Save mode sentinels
        .def("submit_send_data1_sentinel", &ECNaiveNative::submit_send_data1_sentinel)
        .def("submit_send_parity0_sentinel", &ECNaiveNative::submit_send_parity0_sentinel)
        .def("submit_send_parity1_sentinel", &ECNaiveNative::submit_send_parity1_sentinel)
        .def("submit_recv_parity1_sentinel", &ECNaiveNative::submit_recv_parity1_sentinel)
        .def("submit_recv_parity0_sentinel", &ECNaiveNative::submit_recv_parity0_sentinel)
        .def("submit_recv_data1_sentinel", &ECNaiveNative::submit_recv_data1_sentinel)
        // Load mode functions (temporarily kept, will be removed later)
        .def("set_load_mode", &ECNaiveNative::set_load_mode,
             "Set load mode for recovery",
             pybind11::arg("is_load"),
             pybind11::arg("failed_rank") = -1)
        .def("init_load_connections", &ECNaiveNative::init_load_connections,
             "Initialize load mode connections (rank0 recv, rank1/2/3 send)",
             pybind11::arg("rank"),
             pybind11::arg("rank0_ip"),
             pybind11::arg("load_recv_rank1_data1_port"),
             pybind11::arg("load_recv_rank1_data2_port"),
             pybind11::arg("load_recv_rank2_data2_port"),
             pybind11::arg("load_recv_rank2_parity2_port"),
             pybind11::arg("load_recv_rank3_data1_port"),
             pybind11::arg("load_recv_rank3_parity1_port"))
        .def("wait_for_load_connections", &ECNaiveNative::wait_for_load_connections,
             "Wait for load mode connections to be established",
             pybind11::arg("timeout_seconds") = 30)
        .def("load_recover", &ECNaiveNative::load_recover,
             "Unified recovery interface for rank0 (parallel recv + parallel XOR)",
             pybind11::arg("rank1_data1_addr"),
             pybind11::arg("rank1_data2_addr"),
             pybind11::arg("rank2_data2_addr"),
             pybind11::arg("rank2_parity2_addr"),
             pybind11::arg("rank3_data1_addr"),
             pybind11::arg("rank3_parity1_addr"),
             pybind11::arg("recovered_data1_addr"),
             pybind11::arg("recovered_data2_addr"),
             pybind11::arg("recovered_parity1_addr"),
             pybind11::arg("recovered_parity2_addr"),
             pybind11::arg("size"))
        .def("load_send_blocks", &ECNaiveNative::load_send_blocks,
             "Send two blocks to rank2 in parallel (for rank0, rank1, rank3)",
             pybind11::arg("block1_name"),
             pybind11::arg("block1_addr"),
             pybind11::arg("block2_name"),
             pybind11::arg("block2_addr"),
             pybind11::arg("size"))
        .def("stop", &ECNaiveNative::stop);
}


