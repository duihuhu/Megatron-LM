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
    
    // Parity 1 sockets
    boost::asio::ip::tcp::socket parity1_send1_socket_;
    boost::asio::ip::tcp::socket parity1_send2_socket_;
    boost::asio::ip::tcp::socket parity1_recv1_socket_;
    boost::asio::ip::tcp::socket parity1_recv2_socket_;
    boost::asio::ip::tcp::acceptor parity1_recv1_acceptor_;
    boost::asio::ip::tcp::acceptor parity1_recv2_acceptor_;
    
    // Parity 2 sockets
    boost::asio::ip::tcp::socket parity2_send1_socket_;
    boost::asio::ip::tcp::socket parity2_send2_socket_;
    boost::asio::ip::tcp::socket parity2_recv1_socket_;
    boost::asio::ip::tcp::socket parity2_recv2_socket_;
    boost::asio::ip::tcp::acceptor parity2_recv1_acceptor_;
    boost::asio::ip::tcp::acceptor parity2_recv2_acceptor_;

    // Load mode sockets (rank0 as receiver)
    boost::asio::ip::tcp::socket load_recv_rank1_data1_socket_;
    boost::asio::ip::tcp::socket load_recv_rank1_data2_socket_;
    boost::asio::ip::tcp::socket load_recv_rank2_data2_socket_;
    boost::asio::ip::tcp::socket load_recv_rank2_parity2_socket_;
    boost::asio::ip::tcp::socket load_recv_rank3_data1_socket_;
    boost::asio::ip::tcp::socket load_recv_rank3_parity1_socket_;
    boost::asio::ip::tcp::acceptor load_recv_rank1_data1_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank1_data2_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank2_data2_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank2_parity2_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank3_data1_acceptor_;
    boost::asio::ip::tcp::acceptor load_recv_rank3_parity1_acceptor_;
    
    // Load mode sockets (other ranks as senders)
    boost::asio::ip::tcp::socket load_send_rank1_data1_socket_;
    boost::asio::ip::tcp::socket load_send_rank1_data2_socket_;
    boost::asio::ip::tcp::socket load_send_rank2_data2_socket_;
    boost::asio::ip::tcp::socket load_send_rank2_parity2_socket_;
    boost::asio::ip::tcp::socket load_send_rank3_data1_socket_;
    boost::asio::ip::tcp::socket load_send_rank3_parity1_socket_;

    std::atomic<bool> parity1_send1_connected_;
    std::atomic<bool> parity1_send2_connected_;
    std::atomic<bool> parity1_recv1_connected_;
    std::atomic<bool> parity1_recv2_connected_;
    std::atomic<bool> parity2_send1_connected_;
    std::atomic<bool> parity2_send2_connected_;
    std::atomic<bool> parity2_recv1_connected_;
    std::atomic<bool> parity2_recv2_connected_;
    
    // Load mode connection flags (rank0 receiver)
    std::atomic<bool> load_recv_rank1_data1_connected_{false};
    std::atomic<bool> load_recv_rank1_data2_connected_{false};
    std::atomic<bool> load_recv_rank2_data2_connected_{false};
    std::atomic<bool> load_recv_rank2_parity2_connected_{false};
    std::atomic<bool> load_recv_rank3_data1_connected_{false};
    std::atomic<bool> load_recv_rank3_parity1_connected_{false};
    
    // Load mode connection flags (other ranks sender)
    std::atomic<bool> load_send_rank1_data1_connected_{false};
    std::atomic<bool> load_send_rank1_data2_connected_{false};
    std::atomic<bool> load_send_rank2_data2_connected_{false};
    std::atomic<bool> load_send_rank2_parity2_connected_{false};
    std::atomic<bool> load_send_rank3_data1_connected_{false};
    std::atomic<bool> load_send_rank3_parity1_connected_{false};

    std::mutex connection_mutex_;
    std::condition_variable connection_cv_;

public:
    AsioConnectionManager()
        : io_context_(),
          parity1_send1_socket_(io_context_),
          parity1_send2_socket_(io_context_),
          parity1_recv1_socket_(io_context_),
          parity1_recv2_socket_(io_context_),
          parity1_recv1_acceptor_(io_context_),
          parity1_recv2_acceptor_(io_context_),
          parity2_send1_socket_(io_context_),
          parity2_send2_socket_(io_context_),
          parity2_recv1_socket_(io_context_),
          parity2_recv2_socket_(io_context_),
          parity2_recv1_acceptor_(io_context_),
          parity2_recv2_acceptor_(io_context_),
          load_recv_rank1_data1_socket_(io_context_),
          load_recv_rank1_data2_socket_(io_context_),
          load_recv_rank2_data2_socket_(io_context_),
          load_recv_rank2_parity2_socket_(io_context_),
          load_recv_rank3_data1_socket_(io_context_),
          load_recv_rank3_parity1_socket_(io_context_),
          load_recv_rank1_data1_acceptor_(io_context_),
          load_recv_rank1_data2_acceptor_(io_context_),
          load_recv_rank2_data2_acceptor_(io_context_),
          load_recv_rank2_parity2_acceptor_(io_context_),
          load_recv_rank3_data1_acceptor_(io_context_),
          load_recv_rank3_parity1_acceptor_(io_context_),
          load_send_rank1_data1_socket_(io_context_),
          load_send_rank1_data2_socket_(io_context_),
          load_send_rank2_data2_socket_(io_context_),
          load_send_rank2_parity2_socket_(io_context_),
          load_send_rank3_data1_socket_(io_context_),
          load_send_rank3_parity1_socket_(io_context_),
          parity1_send1_connected_(false),
          parity1_send2_connected_(false),
          parity1_recv1_connected_(false),
          parity1_recv2_connected_(false),
          parity2_send1_connected_(false),
          parity2_send2_connected_(false),
          parity2_recv1_connected_(false),
          parity2_recv2_connected_(false) {}

    // Parity 1 getters
    boost::asio::ip::tcp::socket& get_parity1_send1_socket() { return parity1_send1_socket_; }
    boost::asio::ip::tcp::socket& get_parity1_send2_socket() { return parity1_send2_socket_; }
    boost::asio::ip::tcp::socket& get_parity1_recv1_socket() { return parity1_recv1_socket_; }
    boost::asio::ip::tcp::socket& get_parity1_recv2_socket() { return parity1_recv2_socket_; }
    
    // Parity 2 getters
    boost::asio::ip::tcp::socket& get_parity2_send1_socket() { return parity2_send1_socket_; }
    boost::asio::ip::tcp::socket& get_parity2_send2_socket() { return parity2_send2_socket_; }
    boost::asio::ip::tcp::socket& get_parity2_recv1_socket() { return parity2_recv1_socket_; }
    boost::asio::ip::tcp::socket& get_parity2_recv2_socket() { return parity2_recv2_socket_; }
    
    // Load mode getters (rank0 receiver)
    boost::asio::ip::tcp::socket& get_load_recv_rank1_data1_socket() { return load_recv_rank1_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank1_data2_socket() { return load_recv_rank1_data2_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank2_data2_socket() { return load_recv_rank2_data2_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank2_parity2_socket() { return load_recv_rank2_parity2_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank3_data1_socket() { return load_recv_rank3_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_recv_rank3_parity1_socket() { return load_recv_rank3_parity1_socket_; }
    
    // Load mode getters (other ranks sender)
    boost::asio::ip::tcp::socket& get_load_send_rank1_data1_socket() { return load_send_rank1_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank1_data2_socket() { return load_send_rank1_data2_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank2_data2_socket() { return load_send_rank2_data2_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank2_parity2_socket() { return load_send_rank2_parity2_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank3_data1_socket() { return load_send_rank3_data1_socket_; }
    boost::asio::ip::tcp::socket& get_load_send_rank3_parity1_socket() { return load_send_rank3_parity1_socket_; }

    // Parity 1 connection checks
    bool is_parity1_send1_connected() const { return parity1_send1_connected_; }
    bool is_parity1_send2_connected() const { return parity1_send2_connected_; }
    bool is_parity1_recv1_connected() const { return parity1_recv1_connected_; }
    bool is_parity1_recv2_connected() const { return parity1_recv2_connected_; }
    
    // Parity 2 connection checks
    bool is_parity2_send1_connected() const { return parity2_send1_connected_; }
    bool is_parity2_send2_connected() const { return parity2_send2_connected_; }
    bool is_parity2_recv1_connected() const { return parity2_recv1_connected_; }
    bool is_parity2_recv2_connected() const { return parity2_recv2_connected_; }

    // Parity 1 init functions
    void init_parity1_send1(const std::string& partner_ip, uint16_t port);
    void init_parity1_send2(const std::string& partner_ip, uint16_t port);
    void init_parity1_recv1(const std::string& listen_ip, uint16_t port);
    void init_parity1_recv2(const std::string& listen_ip, uint16_t port);
    
    // Parity 2 init functions
    void init_parity2_send1(const std::string& partner_ip, uint16_t port);
    void init_parity2_send2(const std::string& partner_ip, uint16_t port);
    void init_parity2_recv1(const std::string& listen_ip, uint16_t port);
    void init_parity2_recv2(const std::string& listen_ip, uint16_t port);
    
    // Load mode init functions (rank0 as receiver)
    void init_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank1_data2(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank2_data2(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank2_parity2(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port);
    void init_load_recv_rank3_parity1(const std::string& listen_ip, uint16_t port);
    
    // Load mode init functions (other ranks as senders)
    void init_load_send_rank1_data1(const std::string& rank0_ip, uint16_t port);
    void init_load_send_rank1_data2(const std::string& rank0_ip, uint16_t port);
    void init_load_send_rank2_data2(const std::string& rank0_ip, uint16_t port);
    void init_load_send_rank2_parity2(const std::string& rank0_ip, uint16_t port);
    void init_load_send_rank3_data1(const std::string& rank0_ip, uint16_t port);
    void init_load_send_rank3_parity1(const std::string& rank0_ip, uint16_t port);
    
    // Load mode bind+listen helpers (for rank0, before accept)
    void bind_listen_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank1_data2(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank2_data2(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank2_parity2(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port);
    void bind_listen_load_recv_rank3_parity1(const std::string& listen_ip, uint16_t port);
    
    // Load mode accept helpers (for rank0, after bind+listen)
    void accept_load_recv_rank1_data1();
    void accept_load_recv_rank1_data2();
    void accept_load_recv_rank2_data2();
    void accept_load_recv_rank2_parity2();
    void accept_load_recv_rank3_data1();
    void accept_load_recv_rank3_parity1();
    
    void wait_for_connections(int timeout_seconds = 30);
    void wait_for_load_connections(int timeout_seconds = 30);
    void cleanup();
};

// Parity 1 init functions
void AsioConnectionManager::init_parity1_send1(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(parity1_send1_socket_, endpoints);
        parity1_send1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity1_send1 init error: " << e.what() << std::endl;
        parity1_send1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity1_send2(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(parity1_send2_socket_, endpoints);
        parity1_send2_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity1_send2 init error: " << e.what() << std::endl;
        parity1_send2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity1_recv1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        parity1_recv1_acceptor_.open(endpoint.protocol());
        parity1_recv1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        parity1_recv1_acceptor_.bind(endpoint);
        parity1_recv1_acceptor_.listen();
        parity1_recv1_acceptor_.accept(parity1_recv1_socket_);
        parity1_recv1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity1_recv1 init error: " << e.what() << std::endl;
        parity1_recv1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity1_recv2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        parity1_recv2_acceptor_.open(endpoint.protocol());
        parity1_recv2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        parity1_recv2_acceptor_.bind(endpoint);
        parity1_recv2_acceptor_.listen();
        parity1_recv2_acceptor_.accept(parity1_recv2_socket_);
        parity1_recv2_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity1_recv2 init error: " << e.what() << std::endl;
        parity1_recv2_connected_ = false;
        connection_cv_.notify_all();
    }
}

// Parity 2 init functions
void AsioConnectionManager::init_parity2_send1(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(parity2_send1_socket_, endpoints);
        parity2_send1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity2_send1 init error: " << e.what() << std::endl;
        parity2_send1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity2_send2(const std::string& partner_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(partner_ip, std::to_string(port));
        boost::asio::connect(parity2_send2_socket_, endpoints);
        parity2_send2_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity2_send2 init error: " << e.what() << std::endl;
        parity2_send2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity2_recv1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        parity2_recv1_acceptor_.open(endpoint.protocol());
        parity2_recv1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        parity2_recv1_acceptor_.bind(endpoint);
        parity2_recv1_acceptor_.listen();
        parity2_recv1_acceptor_.accept(parity2_recv1_socket_);
        parity2_recv1_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity2_recv1 init error: " << e.what() << std::endl;
        parity2_recv1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::init_parity2_recv2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        parity2_recv2_acceptor_.open(endpoint.protocol());
        parity2_recv2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        parity2_recv2_acceptor_.bind(endpoint);
        parity2_recv2_acceptor_.listen();
        parity2_recv2_acceptor_.accept(parity2_recv2_socket_);
        parity2_recv2_connected_ = true;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: parity2_recv2 init error: " << e.what() << std::endl;
        parity2_recv2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::wait_for_connections(int timeout_seconds) {
    std::unique_lock<std::mutex> lock(connection_mutex_);
    connection_cv_.wait_for(
        lock,
        std::chrono::seconds(timeout_seconds),
        [this]() {
            return parity1_send1_connected_ && parity1_send2_connected_ && 
                   parity1_recv1_connected_ && parity1_recv2_connected_ &&
                   parity2_send1_connected_ && parity2_send2_connected_ && 
                   parity2_recv1_connected_ && parity2_recv2_connected_;
        }
    );
}

void AsioConnectionManager::wait_for_load_connections(int timeout_seconds) {
    // For rank0: wait for all 6 recv connections
    // For rank1/2/3: wait for 2 send connections each
    int rank = -1;  // Will be determined by which connections are being waited
    if (load_recv_rank1_data1_connected_ || load_recv_rank1_data2_connected_) {
        // rank0: wait for all 6 recv connections
        int wait_count = 0;
        while (!(load_recv_rank1_data1_connected_ && load_recv_rank1_data2_connected_ &&
                 load_recv_rank2_data2_connected_ && load_recv_rank2_parity2_connected_ &&
                 load_recv_rank3_data1_connected_ && load_recv_rank3_parity1_connected_)) {
            if (wait_count % 100 == 0) {
                std::cout << "ECLATIN: [Rank 0] Waiting for load connections: "
                          << "r1_d1=" << (load_recv_rank1_data1_connected_ ? "true" : "false")
                          << ", r1_d2=" << (load_recv_rank1_data2_connected_ ? "true" : "false")
                          << ", r2_d2=" << (load_recv_rank2_data2_connected_ ? "true" : "false")
                          << ", r2_p2=" << (load_recv_rank2_parity2_connected_ ? "true" : "false")
                          << ", r3_d1=" << (load_recv_rank3_data1_connected_ ? "true" : "false")
                          << ", r3_p1=" << (load_recv_rank3_parity1_connected_ ? "true" : "false") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Rank 0] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    } else if (load_send_rank1_data1_connected_ || load_send_rank1_data2_connected_) {
        // rank1: wait for 2 send connections
        int wait_count = 0;
        while (!(load_send_rank1_data1_connected_ && load_send_rank1_data2_connected_)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Rank 1] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    } else if (load_send_rank2_data2_connected_ || load_send_rank2_parity2_connected_) {
        // rank2: wait for 2 send connections
        int wait_count = 0;
        while (!(load_send_rank2_data2_connected_ && load_send_rank2_parity2_connected_)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
            if (wait_count * 10 > timeout_seconds * 1000) {
                std::cerr << "ECLATIN: [Rank 2] Timeout waiting for load connections" << std::endl;
                break;
            }
        }
    } else if (load_send_rank3_data1_connected_ || load_send_rank3_parity1_connected_) {
        // rank3: wait for 2 send connections
        int wait_count = 0;
        while (!(load_send_rank3_data1_connected_ && load_send_rank3_parity1_connected_)) {
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

void AsioConnectionManager::init_load_recv_rank1_data2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank1_data2_acceptor_.open(endpoint.protocol());
        load_recv_rank1_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank1_data2_acceptor_.bind(endpoint);
        load_recv_rank1_data2_acceptor_.listen();
        load_recv_rank1_data2_acceptor_.accept(load_recv_rank1_data2_socket_);
        load_recv_rank1_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank1_data2 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank1_data2 init error: " << e.what() << std::endl;
        load_recv_rank1_data2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank2_data2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank2_data2_acceptor_.open(endpoint.protocol());
        load_recv_rank2_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank2_data2_acceptor_.bind(endpoint);
        load_recv_rank2_data2_acceptor_.listen();
        load_recv_rank2_data2_acceptor_.accept(load_recv_rank2_data2_socket_);
        load_recv_rank2_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank2_data2 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank2_data2 init error: " << e.what() << std::endl;
        load_recv_rank2_data2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_recv_rank2_parity2(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank2_parity2_acceptor_.open(endpoint.protocol());
        load_recv_rank2_parity2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank2_parity2_acceptor_.bind(endpoint);
        load_recv_rank2_parity2_acceptor_.listen();
        load_recv_rank2_parity2_acceptor_.accept(load_recv_rank2_parity2_socket_);
        load_recv_rank2_parity2_connected_ = true;
        std::cout << "ASIO: load_recv_rank2_parity2 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank2_parity2 init error: " << e.what() << std::endl;
        load_recv_rank2_parity2_connected_ = false;
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

void AsioConnectionManager::init_load_recv_rank3_parity1(const std::string& listen_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
        load_recv_rank3_parity1_acceptor_.open(endpoint.protocol());
        load_recv_rank3_parity1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        load_recv_rank3_parity1_acceptor_.bind(endpoint);
        load_recv_rank3_parity1_acceptor_.listen();
        load_recv_rank3_parity1_acceptor_.accept(load_recv_rank3_parity1_socket_);
        load_recv_rank3_parity1_connected_ = true;
        std::cout << "ASIO: load_recv_rank3_parity1 connected" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank3_parity1 init error: " << e.what() << std::endl;
        load_recv_rank3_parity1_connected_ = false;
        throw;
    }
}

// Load mode init functions (other ranks as senders)
void AsioConnectionManager::init_load_send_rank1_data1(const std::string& rank0_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank0_ip, std::to_string(port));
        boost::asio::connect(load_send_rank1_data1_socket_, endpoints);
        load_send_rank1_data1_connected_ = true;
        std::cout << "ASIO: load_send_rank1_data1 connected to rank0" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank1_data1 init error: " << e.what() << std::endl;
        load_send_rank1_data1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank1_data2(const std::string& rank0_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank0_ip, std::to_string(port));
        boost::asio::connect(load_send_rank1_data2_socket_, endpoints);
        load_send_rank1_data2_connected_ = true;
        std::cout << "ASIO: load_send_rank1_data2 connected to rank0" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank1_data2 init error: " << e.what() << std::endl;
        load_send_rank1_data2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank2_data2(const std::string& rank0_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank0_ip, std::to_string(port));
        boost::asio::connect(load_send_rank2_data2_socket_, endpoints);
        load_send_rank2_data2_connected_ = true;
        std::cout << "ASIO: load_send_rank2_data2 connected to rank0" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank2_data2 init error: " << e.what() << std::endl;
        load_send_rank2_data2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank2_parity2(const std::string& rank0_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank0_ip, std::to_string(port));
        boost::asio::connect(load_send_rank2_parity2_socket_, endpoints);
        load_send_rank2_parity2_connected_ = true;
        std::cout << "ASIO: load_send_rank2_parity2 connected to rank0" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank2_parity2 init error: " << e.what() << std::endl;
        load_send_rank2_parity2_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank3_data1(const std::string& rank0_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank0_ip, std::to_string(port));
        boost::asio::connect(load_send_rank3_data1_socket_, endpoints);
        load_send_rank3_data1_connected_ = true;
        std::cout << "ASIO: load_send_rank3_data1 connected to rank0" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank3_data1 init error: " << e.what() << std::endl;
        load_send_rank3_data1_connected_ = false;
        throw;
    }
}

void AsioConnectionManager::init_load_send_rank3_parity1(const std::string& rank0_ip, uint16_t port) {
    try {
        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(rank0_ip, std::to_string(port));
        boost::asio::connect(load_send_rank3_parity1_socket_, endpoints);
        load_send_rank3_parity1_connected_ = true;
        std::cout << "ASIO: load_send_rank3_parity1 connected to rank0" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_send_rank3_parity1 init error: " << e.what() << std::endl;
        load_send_rank3_parity1_connected_ = false;
        throw;
    }
}

// Load mode bind+listen helpers (for rank0, before accept)
void AsioConnectionManager::bind_listen_load_recv_rank1_data1(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank1_data1_acceptor_.open(endpoint.protocol());
    load_recv_rank1_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank1_data1_acceptor_.bind(endpoint);
    load_recv_rank1_data1_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank1_data2(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank1_data2_acceptor_.open(endpoint.protocol());
    load_recv_rank1_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank1_data2_acceptor_.bind(endpoint);
    load_recv_rank1_data2_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank2_data2(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank2_data2_acceptor_.open(endpoint.protocol());
    load_recv_rank2_data2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank2_data2_acceptor_.bind(endpoint);
    load_recv_rank2_data2_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank2_parity2(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank2_parity2_acceptor_.open(endpoint.protocol());
    load_recv_rank2_parity2_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank2_parity2_acceptor_.bind(endpoint);
    load_recv_rank2_parity2_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank3_data1(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank3_data1_acceptor_.open(endpoint.protocol());
    load_recv_rank3_data1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank3_data1_acceptor_.bind(endpoint);
    load_recv_rank3_data1_acceptor_.listen();
}

void AsioConnectionManager::bind_listen_load_recv_rank3_parity1(const std::string& listen_ip, uint16_t port) {
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(listen_ip), port);
    load_recv_rank3_parity1_acceptor_.open(endpoint.protocol());
    load_recv_rank3_parity1_acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
    load_recv_rank3_parity1_acceptor_.bind(endpoint);
    load_recv_rank3_parity1_acceptor_.listen();
}

// Load mode accept helpers (for rank0, after bind+listen)
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

void AsioConnectionManager::accept_load_recv_rank1_data2() {
    try {
        load_recv_rank1_data2_acceptor_.accept(load_recv_rank1_data2_socket_);
        load_recv_rank1_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank1_data2 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank1_data2 accept error: " << e.what() << std::endl;
        load_recv_rank1_data2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank2_data2() {
    try {
        load_recv_rank2_data2_acceptor_.accept(load_recv_rank2_data2_socket_);
        load_recv_rank2_data2_connected_ = true;
        std::cout << "ASIO: load_recv_rank2_data2 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank2_data2 accept error: " << e.what() << std::endl;
        load_recv_rank2_data2_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::accept_load_recv_rank2_parity2() {
    try {
        load_recv_rank2_parity2_acceptor_.accept(load_recv_rank2_parity2_socket_);
        load_recv_rank2_parity2_connected_ = true;
        std::cout << "ASIO: load_recv_rank2_parity2 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank2_parity2 accept error: " << e.what() << std::endl;
        load_recv_rank2_parity2_connected_ = false;
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

void AsioConnectionManager::accept_load_recv_rank3_parity1() {
    try {
        load_recv_rank3_parity1_acceptor_.accept(load_recv_rank3_parity1_socket_);
        load_recv_rank3_parity1_connected_ = true;
        std::cout << "ASIO: load_recv_rank3_parity1 connected" << std::endl;
        connection_cv_.notify_all();
    } catch (const std::exception& e) {
        std::cerr << "ASIO: load_recv_rank3_parity1 accept error: " << e.what() << std::endl;
        load_recv_rank3_parity1_connected_ = false;
        connection_cv_.notify_all();
    }
}

void AsioConnectionManager::cleanup() {
    // Parity 1 sockets
    if (parity1_send1_socket_.is_open()) parity1_send1_socket_.close();
    if (parity1_send2_socket_.is_open()) parity1_send2_socket_.close();
    if (parity1_recv1_socket_.is_open()) parity1_recv1_socket_.close();
    if (parity1_recv2_socket_.is_open()) parity1_recv2_socket_.close();
    if (parity1_recv1_acceptor_.is_open()) parity1_recv1_acceptor_.close();
    if (parity1_recv2_acceptor_.is_open()) parity1_recv2_acceptor_.close();
    
    // Parity 2 sockets
    if (parity2_send1_socket_.is_open()) parity2_send1_socket_.close();
    if (parity2_send2_socket_.is_open()) parity2_send2_socket_.close();
    if (parity2_recv1_socket_.is_open()) parity2_recv1_socket_.close();
    if (parity2_recv2_socket_.is_open()) parity2_recv2_socket_.close();
    if (parity2_recv1_acceptor_.is_open()) parity2_recv1_acceptor_.close();
    if (parity2_recv2_acceptor_.is_open()) parity2_recv2_acceptor_.close();
    
    // Load mode sockets (rank0 receiver)
    if (load_recv_rank1_data1_socket_.is_open()) load_recv_rank1_data1_socket_.close();
    if (load_recv_rank1_data2_socket_.is_open()) load_recv_rank1_data2_socket_.close();
    if (load_recv_rank2_data2_socket_.is_open()) load_recv_rank2_data2_socket_.close();
    if (load_recv_rank2_parity2_socket_.is_open()) load_recv_rank2_parity2_socket_.close();
    if (load_recv_rank3_data1_socket_.is_open()) load_recv_rank3_data1_socket_.close();
    if (load_recv_rank3_parity1_socket_.is_open()) load_recv_rank3_parity1_socket_.close();
    if (load_recv_rank1_data1_acceptor_.is_open()) load_recv_rank1_data1_acceptor_.close();
    if (load_recv_rank1_data2_acceptor_.is_open()) load_recv_rank1_data2_acceptor_.close();
    if (load_recv_rank2_data2_acceptor_.is_open()) load_recv_rank2_data2_acceptor_.close();
    if (load_recv_rank2_parity2_acceptor_.is_open()) load_recv_rank2_parity2_acceptor_.close();
    if (load_recv_rank3_data1_acceptor_.is_open()) load_recv_rank3_data1_acceptor_.close();
    if (load_recv_rank3_parity1_acceptor_.is_open()) load_recv_rank3_parity1_acceptor_.close();
    
    // Load mode sockets (other ranks sender)
    if (load_send_rank1_data1_socket_.is_open()) load_send_rank1_data1_socket_.close();
    if (load_send_rank1_data2_socket_.is_open()) load_send_rank1_data2_socket_.close();
    if (load_send_rank2_data2_socket_.is_open()) load_send_rank2_data2_socket_.close();
    if (load_send_rank2_parity2_socket_.is_open()) load_send_rank2_parity2_socket_.close();
    if (load_send_rank3_data1_socket_.is_open()) load_send_rank3_data1_socket_.close();
    if (load_send_rank3_parity1_socket_.is_open()) load_send_rank3_parity1_socket_.close();
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

struct RecvXorTask {
    uintptr_t recv1_addr{0};
    uintptr_t recv2_addr{0};
    uintptr_t parity_addr{0};
    size_t size{0};
};

class ECLATINNative {
public:
    ECLATINNative(const std::string& parity1_send1_ip, uint16_t parity1_send1_port,
                  const std::string& parity1_send2_ip, uint16_t parity1_send2_port,
                  const std::string& parity1_recv1_ip, uint16_t parity1_recv1_port,
                  const std::string& parity1_recv2_ip, uint16_t parity1_recv2_port,
                  const std::string& parity2_send1_ip, uint16_t parity2_send1_port,
                  const std::string& parity2_send2_ip, uint16_t parity2_send2_port,
                  const std::string& parity2_recv1_ip, uint16_t parity2_recv1_port,
                  const std::string& parity2_recv2_ip, uint16_t parity2_recv2_port)
        : stop_(false),
          parity1_send1_ip_(parity1_send1_ip),
          parity1_send1_port_(parity1_send1_port),
          parity1_send2_ip_(parity1_send2_ip),
          parity1_send2_port_(parity1_send2_port),
          parity1_recv1_ip_(parity1_recv1_ip),
          parity1_recv1_port_(parity1_recv1_port),
          parity1_recv2_ip_(parity1_recv2_ip),
          parity1_recv2_port_(parity1_recv2_port),
          parity2_send1_ip_(parity2_send1_ip),
          parity2_send1_port_(parity2_send1_port),
          parity2_send2_ip_(parity2_send2_ip),
          parity2_send2_port_(parity2_send2_port),
          parity2_recv1_ip_(parity2_recv1_ip),
          parity2_recv1_port_(parity2_recv1_port),
          parity2_recv2_ip_(parity2_recv2_ip),
          parity2_recv2_port_(parity2_recv2_port),
          parity1_send1_completed_(false),
          parity1_send2_completed_(false),
          parity1_recv_xor_completed_(false),
          parity2_send1_completed_(false),
          parity2_send2_completed_(false),
          parity2_recv_xor_completed_(false),
          parity1_send1_sentinel_received_(false),
          parity1_send2_sentinel_received_(false),
          parity1_recv_xor_sentinel_received_(false),
          parity2_send1_sentinel_received_(false),
          parity2_send2_sentinel_received_(false),
          parity2_recv_xor_sentinel_received_(false) {
        std::cout << "ECLATIN: Initializing connections..." << std::endl;
        init_connections();
        start_threads();
        std::cout << "ECLATIN: Pipeline started successfully" << std::endl;
    }

    ~ECLATINNative() {
        stop();
    }

    // Parity 1 pipelines
    void submit_parity1_send1(uintptr_t send_addr, size_t size) {
        std::cout << "ECLATIN: Submitting parity1_send1 task: send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_send1_mutex_);
            parity1_send1_q_.push({send_addr, size});
        }
        parity1_send1_cv_.notify_one();
    }

    void submit_parity1_send2(uintptr_t send_addr, size_t size) {
        std::cout << "ECLATIN: Submitting parity1_send2 task: send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_send2_mutex_);
            parity1_send2_q_.push({send_addr, size});
        }
        parity1_send2_cv_.notify_one();
    }

    void submit_parity1_recv_xor(uintptr_t recv1_addr,
                                  uintptr_t recv2_addr,
                                  uintptr_t parity_addr,
                                  size_t size) {
        std::cout << "ECLATIN: Submitting parity1_recv_xor task: recv1_addr=" << recv1_addr
                  << ", recv2_addr=" << recv2_addr
                  << ", parity_addr=" << parity_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_recv_xor_mutex_);
            parity1_recv_xor_q_.push({recv1_addr, recv2_addr, parity_addr, size});
        }
        parity1_recv_xor_cv_.notify_one();
    }

    // Parity 2 pipelines
    void submit_parity2_send1(uintptr_t send_addr, size_t size) {
        std::cout << "ECLATIN: Submitting parity2_send1 task: send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_send1_mutex_);
            parity2_send1_q_.push({send_addr, size});
        }
        parity2_send1_cv_.notify_one();
    }

    void submit_parity2_send2(uintptr_t send_addr, size_t size) {
        std::cout << "ECLATIN: Submitting parity2_send2 task: send_addr=" << send_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_send2_mutex_);
            parity2_send2_q_.push({send_addr, size});
        }
        parity2_send2_cv_.notify_one();
    }

    void submit_parity2_recv_xor(uintptr_t recv1_addr,
                                  uintptr_t recv2_addr,
                                  uintptr_t parity_addr,
                                  size_t size) {
        std::cout << "ECLATIN: Submitting parity2_recv_xor task: recv1_addr=" << recv1_addr
                  << ", recv2_addr=" << recv2_addr
                  << ", parity_addr=" << parity_addr
                  << ", size=" << size << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_recv_xor_mutex_);
            parity2_recv_xor_q_.push({recv1_addr, recv2_addr, parity_addr, size});
        }
        parity2_recv_xor_cv_.notify_one();
    }

    // Submit sentinels to signal pipeline completion
    void submit_parity1_send1_sentinel() {
        std::cout << "ECLATIN: Submitting sentinel to parity1_send1 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_send1_mutex_);
            parity1_send1_q_.push({0, 0});
        }
        parity1_send1_cv_.notify_one();
    }

    void submit_parity1_send2_sentinel() {
        std::cout << "ECLATIN: Submitting sentinel to parity1_send2 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_send2_mutex_);
            parity1_send2_q_.push({0, 0});
        }
        parity1_send2_cv_.notify_one();
    }

    void submit_parity1_recv_xor_sentinel() {
        std::cout << "ECLATIN: Submitting sentinel to parity1_recv_xor pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity1_recv_xor_mutex_);
            parity1_recv_xor_q_.push({0, 0, 0, 0});
        }
        parity1_recv_xor_cv_.notify_one();
    }

    void submit_parity2_send1_sentinel() {
        std::cout << "ECLATIN: Submitting sentinel to parity2_send1 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_send1_mutex_);
            parity2_send1_q_.push({0, 0});
        }
        parity2_send1_cv_.notify_one();
    }

    void submit_parity2_send2_sentinel() {
        std::cout << "ECLATIN: Submitting sentinel to parity2_send2 pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_send2_mutex_);
            parity2_send2_q_.push({0, 0});
        }
        parity2_send2_cv_.notify_one();
    }

    void submit_parity2_recv_xor_sentinel() {
        std::cout << "ECLATIN: Submitting sentinel to parity2_recv_xor pipeline" << std::endl;
        {
            std::lock_guard<std::mutex> lk(parity2_recv_xor_mutex_);
            parity2_recv_xor_q_.push({0, 0, 0, 0});
        }
        parity2_recv_xor_cv_.notify_one();
    }

    // Release helpers: Python can poll these to free buffers.
    // Data buffers are released after send operations complete
    std::vector<uintptr_t> get_data_buffers_to_release() {
        std::vector<uintptr_t> buffers;
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        while (!data_buffers_to_release_.empty()) {
            buffers.push_back(data_buffers_to_release_.front());
            data_buffers_to_release_.pop();
        }
        return buffers;
    }
    
    // Recv buffers are released after recv_xor operations complete (XOR done)
    std::vector<uintptr_t> get_recv_buffers_to_release() {
        std::vector<uintptr_t> buffers;
        std::lock_guard<std::mutex> lock(release_queue_mutex_);
        while (!recv_buffers_to_release_.empty()) {
            buffers.push_back(recv_buffers_to_release_.front());
            recv_buffers_to_release_.pop();
        }
        return buffers;
    }

    void reset_encoding_completion_flags() {
        // Parity 1 flags
        parity1_send1_completed_ = false;
        parity1_send2_completed_ = false;
        parity1_recv_xor_completed_ = false;
        parity1_send1_sentinel_received_ = false;
        parity1_send2_sentinel_received_ = false;
        parity1_recv_xor_sentinel_received_ = false;
        
        // Parity 2 flags
        parity2_send1_completed_ = false;
        parity2_send2_completed_ = false;
        parity2_recv_xor_completed_ = false;
        parity2_send1_sentinel_received_ = false;
        parity2_send2_sentinel_received_ = false;
        parity2_recv_xor_sentinel_received_ = false;

        // Clear queues
        {
            std::lock_guard<std::mutex> lock(parity1_send1_mutex_);
            while (!parity1_send1_q_.empty()) parity1_send1_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(parity1_send2_mutex_);
            while (!parity1_send2_q_.empty()) parity1_send2_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(parity1_recv_xor_mutex_);
            while (!parity1_recv_xor_q_.empty()) parity1_recv_xor_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(parity2_send1_mutex_);
            while (!parity2_send1_q_.empty()) parity2_send1_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(parity2_send2_mutex_);
            while (!parity2_send2_q_.empty()) parity2_send2_q_.pop();
        }
        {
            std::lock_guard<std::mutex> lock(parity2_recv_xor_mutex_);
            while (!parity2_recv_xor_q_.empty()) parity2_recv_xor_q_.pop();
        }

        std::cout << "ECLATIN: Reset encoding completion flags and cleared all queues" << std::endl;
    }

    void wait_for_encoding_completion() {
        // Wait for all 6 workers
        int wait_count = 0;
        while (!parity1_send1_completed_ || !parity1_send2_completed_ || !parity1_recv_xor_completed_ ||
               !parity2_send1_completed_ || !parity2_send2_completed_ || !parity2_recv_xor_completed_) {
            if (wait_count % 100 == 0) {
                std::cout << "ECLATIN: Waiting for workers: "
                          << "p1_s1=" << (parity1_send1_completed_ ? "true" : "false")
                          << ", p1_s2=" << (parity1_send2_completed_ ? "true" : "false")
                          << ", p1_rx=" << (parity1_recv_xor_completed_ ? "true" : "false")
                          << ", p2_s1=" << (parity2_send1_completed_ ? "true" : "false")
                          << ", p2_s2=" << (parity2_send2_completed_ ? "true" : "false")
                          << ", p2_rx=" << (parity2_recv_xor_completed_ ? "true" : "false") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
        }
        std::cout << "ECLATIN: All workers completed" << std::endl;
    }

    void stop() {
        bool expected = false;
        if (!stop_.compare_exchange_strong(expected, true)) {
            return;  // already stopped
        }
        parity1_send1_cv_.notify_all();
        parity1_send2_cv_.notify_all();
        parity1_recv_xor_cv_.notify_all();
        parity2_send1_cv_.notify_all();
        parity2_send2_cv_.notify_all();
        parity2_recv_xor_cv_.notify_all();
        if (parity1_recv_xor_thread_.joinable()) parity1_recv_xor_thread_.join();
        if (parity1_send1_thread_.joinable()) parity1_send1_thread_.join();
        if (parity1_send2_thread_.joinable()) parity1_send2_thread_.join();
        if (parity2_recv_xor_thread_.joinable()) parity2_recv_xor_thread_.join();
        if (parity2_send1_thread_.joinable()) parity2_send1_thread_.join();
        if (parity2_send2_thread_.joinable()) parity2_send2_thread_.join();
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
        const std::string& rank0_ip,
        uint16_t load_recv_rank1_data1_port,
        uint16_t load_recv_rank1_data2_port,
        uint16_t load_recv_rank2_data2_port,
        uint16_t load_recv_rank2_parity2_port,
        uint16_t load_recv_rank3_data1_port,
        uint16_t load_recv_rank3_parity1_port
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: init_load_connections called but not in load mode" << std::endl;
            return;
        }
        
        std::cout << "ECLATIN: [Rank " << rank << "] Initializing load connections..." << std::endl;
        
        if (rank == 0) {
            // rank0: Initialize 6 recv sockets (accept connections from rank1/2/3)
            // Step 1: First, bind and listen all acceptors synchronously (before accept)
            std::cout << "ECLATIN: [Rank 0] Binding and listening all acceptors..." << std::endl;
            try {
                conn_.bind_listen_load_recv_rank1_data1(rank0_ip, load_recv_rank1_data1_port);
                conn_.bind_listen_load_recv_rank1_data2(rank0_ip, load_recv_rank1_data2_port);
                conn_.bind_listen_load_recv_rank2_data2(rank0_ip, load_recv_rank2_data2_port);
                conn_.bind_listen_load_recv_rank2_parity2(rank0_ip, load_recv_rank2_parity2_port);
                conn_.bind_listen_load_recv_rank3_data1(rank0_ip, load_recv_rank3_data1_port);
                conn_.bind_listen_load_recv_rank3_parity1(rank0_ip, load_recv_rank3_parity1_port);
                
                std::cout << "ECLATIN: [Rank 0] All acceptors bound and listening" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "ECLATIN: [Rank 0] Failed to bind/listen acceptors: " << e.what() << std::endl;
                throw;
            }
            
            // Step 2: Start accept operations in separate threads (similar to EC-CHECK)
            // These threads will block on accept() until connections arrive
            std::thread recv_init_thread([this]() {
                std::thread r1_d1([this]() {
                    conn_.accept_load_recv_rank1_data1();
                });
                std::thread r1_d2([this]() {
                    conn_.accept_load_recv_rank1_data2();
                });
                std::thread r2_d2([this]() {
                    conn_.accept_load_recv_rank2_data2();
                });
                std::thread r2_p2([this]() {
                    conn_.accept_load_recv_rank2_parity2();
                });
                std::thread r3_d1([this]() {
                    conn_.accept_load_recv_rank3_data1();
                });
                std::thread r3_p1([this]() {
                    conn_.accept_load_recv_rank3_parity1();
                });
                r1_d1.join();
                r1_d2.join();
                r2_d2.join();
                r2_p2.join();
                r3_d1.join();
                r3_p1.join();
            });
            
            // Step 3: Small delay to ensure accept sockets are bound and listening (similar to EC-CHECK)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Step 4: Detach the recv_init_thread so it runs in background
            // The accept operations will block until connections arrive from rank1/2/3
            recv_init_thread.detach();
            
            std::cout << "ECLATIN: [Rank 0] Accept threads started, waiting for connections..." << std::endl;
        } else {
            // rank1/2/3: Initialize 2 send sockets each (connect to rank0)
            if (rank == 1) {
                std::cout << "ECLATIN: [Rank 1] Connecting load send sockets to rank0..." << std::endl;
                conn_.init_load_send_rank1_data1(rank0_ip, load_recv_rank1_data1_port);
                conn_.init_load_send_rank1_data2(rank0_ip, load_recv_rank1_data2_port);
                std::cout << "ECLATIN: [Rank 1] Load send sockets connected" << std::endl;
            } else if (rank == 2) {
                std::cout << "ECLATIN: [Rank 2] Connecting load send sockets to rank0..." << std::endl;
                conn_.init_load_send_rank2_data2(rank0_ip, load_recv_rank2_data2_port);
                conn_.init_load_send_rank2_parity2(rank0_ip, load_recv_rank2_parity2_port);
                std::cout << "ECLATIN: [Rank 2] Load send sockets connected" << std::endl;
            } else if (rank == 3) {
                std::cout << "ECLATIN: [Rank 3] Connecting load send sockets to rank0..." << std::endl;
                conn_.init_load_send_rank3_data1(rank0_ip, load_recv_rank3_data1_port);
                conn_.init_load_send_rank3_parity1(rank0_ip, load_recv_rank3_parity1_port);
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

    // Unified recovery interface for rank0 (parallel recv + parallel XOR)
    void load_recover(
        // Receive buffers (6 blocks from other ranks)
        uintptr_t rank1_data1_addr,
        uintptr_t rank1_data2_addr,
        uintptr_t rank2_data2_addr,
        uintptr_t rank2_parity2_addr,
        uintptr_t rank3_data1_addr,
        uintptr_t rank3_parity1_addr,
        // Recovered buffers (4 blocks to write results)
        uintptr_t recovered_data1_addr,
        uintptr_t recovered_data2_addr,
        uintptr_t recovered_parity1_addr,
        uintptr_t recovered_parity2_addr,
        size_t size
    ) {
        if (!is_load_mode_) {
            std::cerr << "ECLATIN: [Rank 0] load_recover called but not in load mode" << std::endl;
            return;
        }
        
        std::cout << "ECLATIN: [Rank 0] Starting recovery (size=" << size << ")" << std::endl;
        
        // Step 1: Parallel receive all 6 blocks using threads
        std::vector<std::exception_ptr> recv_exceptions(6);
        std::vector<std::thread> recv_threads;
        
        recv_threads.emplace_back([&]() {
            try {
                if (!recv_with_size_bool(conn_.get_load_recv_rank1_data1_socket(), 
                                        reinterpret_cast<void*>(rank1_data1_addr), size)) {
                    throw std::runtime_error("Failed to receive rank1_data1");
                }
            } catch (...) {
                recv_exceptions[0] = std::current_exception();
            }
        });
        
        recv_threads.emplace_back([&]() {
            try {
                if (!recv_with_size_bool(conn_.get_load_recv_rank1_data2_socket(), 
                                        reinterpret_cast<void*>(rank1_data2_addr), size)) {
                    throw std::runtime_error("Failed to receive rank1_data2");
                }
            } catch (...) {
                recv_exceptions[1] = std::current_exception();
            }
        });
        
        recv_threads.emplace_back([&]() {
            try {
                if (!recv_with_size_bool(conn_.get_load_recv_rank2_data2_socket(), 
                                        reinterpret_cast<void*>(rank2_data2_addr), size)) {
                    throw std::runtime_error("Failed to receive rank2_data2");
                }
            } catch (...) {
                recv_exceptions[2] = std::current_exception();
            }
        });
        
        recv_threads.emplace_back([&]() {
            try {
                if (!recv_with_size_bool(conn_.get_load_recv_rank2_parity2_socket(), 
                                        reinterpret_cast<void*>(rank2_parity2_addr), size)) {
                    throw std::runtime_error("Failed to receive rank2_parity2");
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
                if (!recv_with_size_bool(conn_.get_load_recv_rank3_parity1_socket(), 
                                        reinterpret_cast<void*>(rank3_parity1_addr), size)) {
                    throw std::runtime_error("Failed to receive rank3_parity1");
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
        
        std::cout << "ECLATIN: [Rank 0] All 6 blocks received" << std::endl;
        
        // Step 2: Parallel XOR recoveries using threads
        std::vector<std::exception_ptr> xor_exceptions(4);
        std::vector<std::thread> xor_threads;
        
        // data1 = rank2.data2 XOR rank3.parity1
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_data1_addr), 
                           reinterpret_cast<void*>(rank2_data2_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_data1_addr), 
                                     reinterpret_cast<void*>(rank3_parity1_addr)};
                xor_gen(2, static_cast<int>(size), xor_array);
            } catch (...) {
                xor_exceptions[0] = std::current_exception();
            }
        });
        
        // data2 = rank3.data1 XOR rank2.parity2
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_data2_addr), 
                           reinterpret_cast<void*>(rank3_data1_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_data2_addr), 
                                     reinterpret_cast<void*>(rank2_parity2_addr)};
                xor_gen(2, static_cast<int>(size), xor_array);
            } catch (...) {
                xor_exceptions[1] = std::current_exception();
            }
        });
        
        // parity1 = rank3.data1 XOR rank1.data2
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_parity1_addr), 
                           reinterpret_cast<void*>(rank3_data1_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_parity1_addr), 
                                     reinterpret_cast<void*>(rank1_data2_addr)};
                xor_gen(2, static_cast<int>(size), xor_array);
            } catch (...) {
                xor_exceptions[2] = std::current_exception();
            }
        });
        
        // parity2 = rank1.data1 XOR rank2.data2
        xor_threads.emplace_back([&]() {
            try {
                std::memcpy(reinterpret_cast<void*>(recovered_parity2_addr), 
                           reinterpret_cast<void*>(rank1_data1_addr), size);
                void* xor_array[2] = {reinterpret_cast<void*>(recovered_parity2_addr), 
                                     reinterpret_cast<void*>(rank2_data2_addr)};
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
        
        std::cout << "ECLATIN: [Rank 0] Recovery completed successfully" << std::endl;
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
                  << " and " << block2_name << " to rank0 (size=" << size << ")" << std::endl;
        
        // Helper function to get socket by block name
        auto get_socket = [this](const std::string& block_name) -> boost::asio::ip::tcp::socket* {
            if (block_name == "rank1_data1") {
                return &conn_.get_load_send_rank1_data1_socket();
            } else if (block_name == "rank1_data2") {
                return &conn_.get_load_send_rank1_data2_socket();
            } else if (block_name == "rank2_data2") {
                return &conn_.get_load_send_rank2_data2_socket();
            } else if (block_name == "rank2_parity2") {
                return &conn_.get_load_send_rank2_parity2_socket();
            } else if (block_name == "rank3_data1") {
                return &conn_.get_load_send_rank3_data1_socket();
            } else if (block_name == "rank3_parity1") {
                return &conn_.get_load_send_rank3_parity1_socket();
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
                
                std::cout << "ECLATIN: Sending " << block1_name << " to rank0 (size=" << size << ")" << std::endl;
                
                if (!send_with_size(*sock, block1_addr, size)) {
                    throw std::runtime_error("ECLATIN: load_send_blocks send failed for " + block1_name);
                }
                
                std::cout << "ECLATIN: Successfully sent " << block1_name << " to rank0" << std::endl;
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
                
                std::cout << "ECLATIN: Sending " << block2_name << " to rank0 (size=" << size << ")" << std::endl;
                
                if (!send_with_size(*sock, block2_addr, size)) {
                    throw std::runtime_error("ECLATIN: load_send_blocks send failed for " + block2_name);
                }
                
                std::cout << "ECLATIN: Successfully sent " << block2_name << " to rank0" << std::endl;
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
    
    // Parity 1 network config
    std::string parity1_send1_ip_;
    uint16_t parity1_send1_port_;
    std::string parity1_send2_ip_;
    uint16_t parity1_send2_port_;
    std::string parity1_recv1_ip_;
    uint16_t parity1_recv1_port_;
    std::string parity1_recv2_ip_;
    uint16_t parity1_recv2_port_;
    
    // Parity 2 network config
    std::string parity2_send1_ip_;
    uint16_t parity2_send1_port_;
    std::string parity2_send2_ip_;
    uint16_t parity2_send2_port_;
    std::string parity2_recv1_ip_;
    uint16_t parity2_recv1_port_;
    std::string parity2_recv2_ip_;
    uint16_t parity2_recv2_port_;

    // Parity 1 pipelines
    std::queue<SendTask> parity1_send1_q_;
    std::mutex parity1_send1_mutex_;
    std::condition_variable parity1_send1_cv_;
    std::queue<SendTask> parity1_send2_q_;
    std::mutex parity1_send2_mutex_;
    std::condition_variable parity1_send2_cv_;
    std::queue<RecvXorTask> parity1_recv_xor_q_;
    std::mutex parity1_recv_xor_mutex_;
    std::condition_variable parity1_recv_xor_cv_;

    // Parity 2 pipelines
    std::queue<SendTask> parity2_send1_q_;
    std::mutex parity2_send1_mutex_;
    std::condition_variable parity2_send1_cv_;
    std::queue<SendTask> parity2_send2_q_;
    std::mutex parity2_send2_mutex_;
    std::condition_variable parity2_send2_cv_;
    std::queue<RecvXorTask> parity2_recv_xor_q_;
    std::mutex parity2_recv_xor_mutex_;
    std::condition_variable parity2_recv_xor_cv_;

    // Separate release queues for data and recv buffers
    std::queue<uintptr_t> data_buffers_to_release_;
    std::queue<uintptr_t> recv_buffers_to_release_;
    std::mutex release_queue_mutex_;

    // Completion flags
    std::atomic<bool> parity1_send1_completed_;
    std::atomic<bool> parity1_send2_completed_;
    std::atomic<bool> parity1_recv_xor_completed_;
    std::atomic<bool> parity2_send1_completed_;
    std::atomic<bool> parity2_send2_completed_;
    std::atomic<bool> parity2_recv_xor_completed_;

    // Sentinel received flags
    std::atomic<bool> parity1_send1_sentinel_received_;
    std::atomic<bool> parity1_send2_sentinel_received_;
    std::atomic<bool> parity1_recv_xor_sentinel_received_;
    std::atomic<bool> parity2_send1_sentinel_received_;
    std::atomic<bool> parity2_send2_sentinel_received_;
    std::atomic<bool> parity2_recv_xor_sentinel_received_;

    std::thread parity1_send1_thread_;
    std::thread parity1_send2_thread_;
    std::thread parity1_recv_xor_thread_;
    std::thread parity2_send1_thread_;
    std::thread parity2_send2_thread_;
    std::thread parity2_recv_xor_thread_;
    
    // Load mode flags
    std::atomic<bool> is_load_mode_{false};
    int failed_rank_{-1};

    void start_threads() {
        std::cout << "ECLATIN: Starting worker threads..." << std::endl;
        parity1_send1_thread_ = std::thread(&ECLATINNative::parity1_send1_worker, this);
        parity1_send2_thread_ = std::thread(&ECLATINNative::parity1_send2_worker, this);
        parity1_recv_xor_thread_ = std::thread(&ECLATINNative::parity1_recv_xor_worker, this);
        parity2_send1_thread_ = std::thread(&ECLATINNative::parity2_send1_worker, this);
        parity2_send2_thread_ = std::thread(&ECLATINNative::parity2_send2_worker, this);
        parity2_recv_xor_thread_ = std::thread(&ECLATINNative::parity2_recv_xor_worker, this);
        std::cout << "ECLATIN: All worker threads started" << std::endl;
    }

    void init_connections() {
        std::cout << "ECLATIN: Initializing connections..." << std::endl;
        // Start acceptors in separate threads to avoid deadlock (mirror eccheck pattern)
        std::thread recv_init_thread([this]() {
            std::thread r1_1([this]() { conn_.init_parity1_recv1(parity1_recv1_ip_, parity1_recv1_port_); });
            std::thread r2_1([this]() { conn_.init_parity1_recv2(parity1_recv2_ip_, parity1_recv2_port_); });
            std::thread r1_2([this]() { conn_.init_parity2_recv1(parity2_recv1_ip_, parity2_recv1_port_); });
            std::thread r2_2([this]() { conn_.init_parity2_recv2(parity2_recv2_ip_, parity2_recv2_port_); });
            r1_1.join();
            r2_1.join();
            r1_2.join();
            r2_2.join();
        });

        // Small delay to ensure acceptors are listening
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Connect send sockets (blocking)
        std::cout << "ECLATIN: Connecting parity1_send1 socket..." << std::endl;
        conn_.init_parity1_send1(parity1_send1_ip_, parity1_send1_port_);
        std::cout << "ECLATIN: Connecting parity1_send2 socket..." << std::endl;
        conn_.init_parity1_send2(parity1_send2_ip_, parity1_send2_port_);
        std::cout << "ECLATIN: Connecting parity2_send1 socket..." << std::endl;
        conn_.init_parity2_send1(parity2_send1_ip_, parity2_send1_port_);
        std::cout << "ECLATIN: Connecting parity2_send2 socket..." << std::endl;
        conn_.init_parity2_send2(parity2_send2_ip_, parity2_send2_port_);

        recv_init_thread.join();
        std::cout << "ECLATIN: Waiting for all connections..." << std::endl;
        conn_.wait_for_connections();
        std::cout << "ECLATIN: All connections established" << std::endl;
    }


    // Parity 1 workers
    void parity1_recv_xor_worker() {
        std::cout << "ECLATIN: Parity1_RecvXor worker started" << std::endl;
        while (!stop_) {
            RecvXorTask task;
            {
                std::unique_lock<std::mutex> lk(parity1_recv_xor_mutex_);
                parity1_recv_xor_cv_.wait(lk, [this] { return stop_ || !parity1_recv_xor_q_.empty(); });
                if (stop_) break;
                task = parity1_recv_xor_q_.front();
                parity1_recv_xor_q_.pop();
            }
            // Check for sentinel
            if (task.recv1_addr == 0 && task.recv2_addr == 0 &&
                task.parity_addr == 0 && task.size == 0) {
                parity1_recv_xor_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity1_recv_xor_mutex_);
                    if (parity1_recv_xor_q_.empty()) {
                        parity1_recv_xor_completed_ = true;
                        std::cout << "ECLATIN: Parity1_RecvXor worker completed" << std::endl;
                        parity1_recv_xor_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.recv1_addr == 0 ||
                task.recv2_addr == 0 || task.parity_addr == 0) {
                continue;
            }

            // Ensure connections are ready
            if (!conn_.is_parity1_recv1_connected()) {
                std::cerr << "ECLATIN: ERROR: parity1_recv1 socket not connected!" << std::endl;
                throw std::runtime_error("ECLATIN: parity1_recv1 socket not connected");
            }
            if (!conn_.is_parity1_recv2_connected()) {
                std::cerr << "ECLATIN: ERROR: parity1_recv2 socket not connected!" << std::endl;
                throw std::runtime_error("ECLATIN: parity1_recv2 socket not connected");
            }

            bool recv1_success = false;
            bool recv2_success = false;
            std::string recv1_error_msg;
            std::string recv2_error_msg;
            std::exception_ptr recv1_exception = nullptr;
            std::exception_ptr recv2_exception = nullptr;

            std::thread recv1_thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_parity1_recv1_socket(),
                            reinterpret_cast<void*>(task.recv1_addr),
                            task.size)) {
                        recv1_error_msg = "ECLATIN: parity1_recv1_with_size_bool returned false";
                        recv1_success = false;
                    } else {
                        recv1_success = true;
                    }
                } catch (const std::exception& e) {
                    recv1_exception = std::current_exception();
                    recv1_error_msg = std::string("ECLATIN: parity1_recv1 exception: ") + e.what();
                } catch (...) {
                    recv1_exception = std::current_exception();
                    recv1_error_msg = "ECLATIN: parity1_recv1 unknown exception";
                }
            });

            std::thread recv2_thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_parity1_recv2_socket(),
                            reinterpret_cast<void*>(task.recv2_addr),
                            task.size)) {
                        recv2_error_msg = "ECLATIN: parity1_recv2_with_size_bool returned false";
                        recv2_success = false;
                    } else {
                        recv2_success = true;
                    }
                } catch (const std::exception& e) {
                    recv2_exception = std::current_exception();
                    recv2_error_msg = std::string("ECLATIN: parity1_recv2 exception: ") + e.what();
                } catch (...) {
                    recv2_exception = std::current_exception();
                    recv2_error_msg = "ECLATIN: parity1_recv2 unknown exception";
                }
            });

            recv1_thread.join();
            recv2_thread.join();

            // Bubble up failures
            if (recv1_exception) {
                std::cerr << recv1_error_msg << std::endl;
                std::rethrow_exception(recv1_exception);
            }
            if (recv2_exception) {
                std::cerr << recv2_error_msg << std::endl;
                std::rethrow_exception(recv2_exception);
            }
            if (!recv1_success) {
                std::cerr << recv1_error_msg << std::endl;
                throw std::runtime_error(recv1_error_msg);
            }
            if (!recv2_success) {
                std::cerr << recv2_error_msg << std::endl;
                throw std::runtime_error(recv2_error_msg);
            }

            // XOR after both recvs succeed
            unsigned char* recv1_ptr = reinterpret_cast<unsigned char*>(task.recv1_addr);
            unsigned char* recv2_ptr = reinterpret_cast<unsigned char*>(task.recv2_addr);
            unsigned char* parity_ptr = reinterpret_cast<unsigned char*>(task.parity_addr);

            void* xor_array[3];
            xor_array[0] = recv1_ptr;
            xor_array[1] = recv2_ptr;
            xor_array[2] = parity_ptr;
            xor_gen(3, static_cast<int>(task.size), xor_array);

            // Release recv buffers after XOR operation completes
            // Note: parity_addr is managed by Python, not released here
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                recv_buffers_to_release_.push(task.recv1_addr);
                recv_buffers_to_release_.push(task.recv2_addr);
            }

            if (parity1_recv_xor_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity1_recv_xor_mutex_);
                if (parity1_recv_xor_q_.empty()) {
                    parity1_recv_xor_completed_ = true;
                    std::cout << "ECLATIN: Parity1_RecvXor worker completed" << std::endl;
                    parity1_recv_xor_sentinel_received_ = false;
                }
            }
        }
    }

    void parity1_send1_worker() {
        std::cout << "ECLATIN: Parity1_Send1 worker started" << std::endl;
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(parity1_send1_mutex_);
                parity1_send1_cv_.wait(lk, [this] { return stop_ || !parity1_send1_q_.empty(); });
                if (stop_) break;
                task = parity1_send1_q_.front();
                parity1_send1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                parity1_send1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity1_send1_mutex_);
                    if (parity1_send1_q_.empty()) {
                        parity1_send1_completed_ = true;
                        std::cout << "ECLATIN: Parity1_Send1 worker completed" << std::endl;
                        parity1_send1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_parity1_send1_connected()) {
                send_with_size(conn_.get_parity1_send1_socket(), task.addr, task.size);
            }
            // Release data buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                data_buffers_to_release_.push(task.addr);
            }

            if (parity1_send1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity1_send1_mutex_);
                if (parity1_send1_q_.empty()) {
                    parity1_send1_completed_ = true;
                    std::cout << "ECLATIN: Parity1_Send1 worker completed" << std::endl;
                    parity1_send1_sentinel_received_ = false;
                }
            }
        }
    }

    void parity1_send2_worker() {
        std::cout << "ECLATIN: Parity1_Send2 worker started" << std::endl;
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(parity1_send2_mutex_);
                parity1_send2_cv_.wait(lk, [this] { return stop_ || !parity1_send2_q_.empty(); });
                if (stop_) break;
                task = parity1_send2_q_.front();
                parity1_send2_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                parity1_send2_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity1_send2_mutex_);
                    if (parity1_send2_q_.empty()) {
                        parity1_send2_completed_ = true;
                        std::cout << "ECLATIN: Parity1_Send2 worker completed" << std::endl;
                        parity1_send2_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_parity1_send2_connected()) {
                send_with_size(conn_.get_parity1_send2_socket(), task.addr, task.size);
            }
            // Release data buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                data_buffers_to_release_.push(task.addr);
            }

            if (parity1_send2_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity1_send2_mutex_);
                if (parity1_send2_q_.empty()) {
                    parity1_send2_completed_ = true;
                    std::cout << "ECLATIN: Parity1_Send2 worker completed" << std::endl;
                    parity1_send2_sentinel_received_ = false;
                }
            }
        }
    }

    // Parity 2 workers (same logic as parity1)
    void parity2_recv_xor_worker() {
        std::cout << "ECLATIN: Parity2_RecvXor worker started" << std::endl;
        while (!stop_) {
            RecvXorTask task;
            {
                std::unique_lock<std::mutex> lk(parity2_recv_xor_mutex_);
                parity2_recv_xor_cv_.wait(lk, [this] { return stop_ || !parity2_recv_xor_q_.empty(); });
                if (stop_) break;
                task = parity2_recv_xor_q_.front();
                parity2_recv_xor_q_.pop();
            }
            // Check for sentinel
            if (task.recv1_addr == 0 && task.recv2_addr == 0 &&
                task.parity_addr == 0 && task.size == 0) {
                parity2_recv_xor_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity2_recv_xor_mutex_);
                    if (parity2_recv_xor_q_.empty()) {
                        parity2_recv_xor_completed_ = true;
                        std::cout << "ECLATIN: Parity2_RecvXor worker completed" << std::endl;
                        parity2_recv_xor_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.recv1_addr == 0 ||
                task.recv2_addr == 0 || task.parity_addr == 0) {
                continue;
            }

            // Ensure connections are ready
            if (!conn_.is_parity2_recv1_connected()) {
                std::cerr << "ECLATIN: ERROR: parity2_recv1 socket not connected!" << std::endl;
                throw std::runtime_error("ECLATIN: parity2_recv1 socket not connected");
            }
            if (!conn_.is_parity2_recv2_connected()) {
                std::cerr << "ECLATIN: ERROR: parity2_recv2 socket not connected!" << std::endl;
                throw std::runtime_error("ECLATIN: parity2_recv2 socket not connected");
            }

            bool recv1_success = false;
            bool recv2_success = false;
            std::string recv1_error_msg;
            std::string recv2_error_msg;
            std::exception_ptr recv1_exception = nullptr;
            std::exception_ptr recv2_exception = nullptr;

            std::thread recv1_thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_parity2_recv1_socket(),
                            reinterpret_cast<void*>(task.recv1_addr),
                            task.size)) {
                        recv1_error_msg = "ECLATIN: parity2_recv1_with_size_bool returned false";
                        recv1_success = false;
                    } else {
                        recv1_success = true;
                    }
                } catch (const std::exception& e) {
                    recv1_exception = std::current_exception();
                    recv1_error_msg = std::string("ECLATIN: parity2_recv1 exception: ") + e.what();
                } catch (...) {
                    recv1_exception = std::current_exception();
                    recv1_error_msg = "ECLATIN: parity2_recv1 unknown exception";
                }
            });

            std::thread recv2_thread([&]() {
                try {
                    if (!recv_with_size_bool(
                            conn_.get_parity2_recv2_socket(),
                            reinterpret_cast<void*>(task.recv2_addr),
                            task.size)) {
                        recv2_error_msg = "ECLATIN: parity2_recv2_with_size_bool returned false";
                        recv2_success = false;
                    } else {
                        recv2_success = true;
                    }
        } catch (const std::exception& e) {
                    recv2_exception = std::current_exception();
                    recv2_error_msg = std::string("ECLATIN: parity2_recv2 exception: ") + e.what();
                } catch (...) {
                    recv2_exception = std::current_exception();
                    recv2_error_msg = "ECLATIN: parity2_recv2 unknown exception";
                }
            });

            recv1_thread.join();
            recv2_thread.join();

            // Bubble up failures
            if (recv1_exception) {
                std::cerr << recv1_error_msg << std::endl;
                std::rethrow_exception(recv1_exception);
            }
            if (recv2_exception) {
                std::cerr << recv2_error_msg << std::endl;
                std::rethrow_exception(recv2_exception);
            }
            if (!recv1_success) {
                std::cerr << recv1_error_msg << std::endl;
                throw std::runtime_error(recv1_error_msg);
            }
            if (!recv2_success) {
                std::cerr << recv2_error_msg << std::endl;
                throw std::runtime_error(recv2_error_msg);
            }

            // XOR after both recvs succeed
            unsigned char* recv1_ptr = reinterpret_cast<unsigned char*>(task.recv1_addr);
            unsigned char* recv2_ptr = reinterpret_cast<unsigned char*>(task.recv2_addr);
            unsigned char* parity_ptr = reinterpret_cast<unsigned char*>(task.parity_addr);

            void* xor_array[3];
            xor_array[0] = recv1_ptr;
            xor_array[1] = recv2_ptr;
            xor_array[2] = parity_ptr;
            xor_gen(3, static_cast<int>(task.size), xor_array);

            // Release recv buffers after XOR operation completes
            // Note: parity_addr is managed by Python, not released here
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                recv_buffers_to_release_.push(task.recv1_addr);
                recv_buffers_to_release_.push(task.recv2_addr);
            }

            if (parity2_recv_xor_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity2_recv_xor_mutex_);
                if (parity2_recv_xor_q_.empty()) {
                    parity2_recv_xor_completed_ = true;
                    std::cout << "ECLATIN: Parity2_RecvXor worker completed" << std::endl;
                    parity2_recv_xor_sentinel_received_ = false;
                }
            }
        }
    }

    void parity2_send1_worker() {
        std::cout << "ECLATIN: Parity2_Send1 worker started" << std::endl;
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(parity2_send1_mutex_);
                parity2_send1_cv_.wait(lk, [this] { return stop_ || !parity2_send1_q_.empty(); });
                if (stop_) break;
                task = parity2_send1_q_.front();
                parity2_send1_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                parity2_send1_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity2_send1_mutex_);
                    if (parity2_send1_q_.empty()) {
                        parity2_send1_completed_ = true;
                        std::cout << "ECLATIN: Parity2_Send1 worker completed" << std::endl;
                        parity2_send1_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_parity2_send1_connected()) {
                send_with_size(conn_.get_parity2_send1_socket(), task.addr, task.size);
            }
            // Release data buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                data_buffers_to_release_.push(task.addr);
            }

            if (parity2_send1_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity2_send1_mutex_);
                if (parity2_send1_q_.empty()) {
                    parity2_send1_completed_ = true;
                    std::cout << "ECLATIN: Parity2_Send1 worker completed" << std::endl;
                    parity2_send1_sentinel_received_ = false;
                }
            }
        }
    }

    void parity2_send2_worker() {
        std::cout << "ECLATIN: Parity2_Send2 worker started" << std::endl;
        while (!stop_) {
            SendTask task;
            {
                std::unique_lock<std::mutex> lk(parity2_send2_mutex_);
                parity2_send2_cv_.wait(lk, [this] { return stop_ || !parity2_send2_q_.empty(); });
                if (stop_) break;
                task = parity2_send2_q_.front();
                parity2_send2_q_.pop();
            }
            // Check for sentinel
            if (task.addr == 0 && task.size == 0) {
                parity2_send2_sentinel_received_ = true;
                {
                    std::lock_guard<std::mutex> lock(parity2_send2_mutex_);
                    if (parity2_send2_q_.empty()) {
                        parity2_send2_completed_ = true;
                        std::cout << "ECLATIN: Parity2_Send2 worker completed" << std::endl;
                        parity2_send2_sentinel_received_ = false;
                    }
                }
                continue;
            }
            if (task.size == 0 || task.addr == 0) {
                continue;
            }
            if (conn_.is_parity2_send2_connected()) {
                send_with_size(conn_.get_parity2_send2_socket(), task.addr, task.size);
            }
            // Release data buffer after send operation completes
            {
                std::lock_guard<std::mutex> lk(release_queue_mutex_);
                data_buffers_to_release_.push(task.addr);
            }

            if (parity2_send2_sentinel_received_.load()) {
                std::lock_guard<std::mutex> lock(parity2_send2_mutex_);
                if (parity2_send2_q_.empty()) {
                    parity2_send2_completed_ = true;
                    std::cout << "ECLATIN: Parity2_Send2 worker completed" << std::endl;
                    parity2_send2_sentinel_received_ = false;
                }
            }
        }
    }
};

}  // namespace

PYBIND11_MODULE(eclatin_native, m) {
    pybind11::class_<ECLATINNative>(m, "ECLATINNative")
        .def(pybind11::init<const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t,
                            const std::string&, uint16_t>())
        // Parity 1 submit functions
        .def("submit_parity1_send1", &ECLATINNative::submit_parity1_send1,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_parity1_send2", &ECLATINNative::submit_parity1_send2,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_parity1_recv_xor", &ECLATINNative::submit_parity1_recv_xor,
             pybind11::arg("recv1_addr"),
             pybind11::arg("recv2_addr"),
             pybind11::arg("parity_addr"),
             pybind11::arg("size"))
        // Parity 2 submit functions
        .def("submit_parity2_send1", &ECLATINNative::submit_parity2_send1,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_parity2_send2", &ECLATINNative::submit_parity2_send2,
             pybind11::arg("send_addr"),
             pybind11::arg("size"))
        .def("submit_parity2_recv_xor", &ECLATINNative::submit_parity2_recv_xor,
             pybind11::arg("recv1_addr"),
             pybind11::arg("recv2_addr"),
             pybind11::arg("parity_addr"),
             pybind11::arg("size"))
        // Common functions
        .def("get_data_buffers_to_release", &ECLATINNative::get_data_buffers_to_release)
        .def("get_recv_buffers_to_release", &ECLATINNative::get_recv_buffers_to_release)
        .def("reset_encoding_completion_flags", &ECLATINNative::reset_encoding_completion_flags)
        .def("wait_for_encoding_completion", &ECLATINNative::wait_for_encoding_completion)
        // Parity 1 sentinels
        .def("submit_parity1_send1_sentinel", &ECLATINNative::submit_parity1_send1_sentinel)
        .def("submit_parity1_send2_sentinel", &ECLATINNative::submit_parity1_send2_sentinel)
        .def("submit_parity1_recv_xor_sentinel", &ECLATINNative::submit_parity1_recv_xor_sentinel)
        // Parity 2 sentinels
        .def("submit_parity2_send1_sentinel", &ECLATINNative::submit_parity2_send1_sentinel)
        .def("submit_parity2_send2_sentinel", &ECLATINNative::submit_parity2_send2_sentinel)
        .def("submit_parity2_recv_xor_sentinel", &ECLATINNative::submit_parity2_recv_xor_sentinel)
        // Load mode functions
        .def("set_load_mode", &ECLATINNative::set_load_mode,
             "Set load mode for recovery",
             pybind11::arg("is_load"),
             pybind11::arg("failed_rank") = -1)
        .def("init_load_connections", &ECLATINNative::init_load_connections,
             "Initialize load mode connections (rank0 recv, rank1/2/3 send)",
             pybind11::arg("rank"),
             pybind11::arg("rank0_ip"),
             pybind11::arg("load_recv_rank1_data1_port"),
             pybind11::arg("load_recv_rank1_data2_port"),
             pybind11::arg("load_recv_rank2_data2_port"),
             pybind11::arg("load_recv_rank2_parity2_port"),
             pybind11::arg("load_recv_rank3_data1_port"),
             pybind11::arg("load_recv_rank3_parity1_port"))
        .def("wait_for_load_connections", &ECLATINNative::wait_for_load_connections,
             "Wait for load mode connections to be established",
             pybind11::arg("timeout_seconds") = 30)
        .def("load_recover", &ECLATINNative::load_recover,
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
        .def("load_send_blocks", &ECLATINNative::load_send_blocks,
             "Send two blocks to rank0 in parallel (for rank1, rank2, rank3)",
             pybind11::arg("block1_name"),
             pybind11::arg("block1_addr"),
             pybind11::arg("block2_name"),
             pybind11::arg("block2_addr"),
             pybind11::arg("size"))
        .def("stop", &ECLATINNative::stop);
}


