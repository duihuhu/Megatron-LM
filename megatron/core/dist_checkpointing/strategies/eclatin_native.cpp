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

    std::atomic<bool> parity1_send1_connected_;
    std::atomic<bool> parity1_send2_connected_;
    std::atomic<bool> parity1_recv1_connected_;
    std::atomic<bool> parity1_recv2_connected_;
    std::atomic<bool> parity2_send1_connected_;
    std::atomic<bool> parity2_send2_connected_;
    std::atomic<bool> parity2_recv1_connected_;
    std::atomic<bool> parity2_recv2_connected_;

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
    
    void wait_for_connections(int timeout_seconds = 30);
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
    std::vector<uintptr_t> get_buffers_to_release() {
        std::vector<uintptr_t> res;
        std::lock_guard<std::mutex> lk(release_mutex_);
        while (!release_q_.empty()) {
            res.push_back(release_q_.front());
            release_q_.pop();
        }
        return res;
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

    std::queue<uintptr_t> release_q_;
    std::mutex release_mutex_;

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

            // Release recv buffers (parity managed by Python)
            {
                std::lock_guard<std::mutex> lk(release_mutex_);
                release_q_.push(task.recv1_addr);
                release_q_.push(task.recv2_addr);
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
            std::lock_guard<std::mutex> lk(release_mutex_);
            release_q_.push(task.addr);

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
            std::lock_guard<std::mutex> lk(release_mutex_);
            release_q_.push(task.addr);

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

            // Release recv buffers (parity managed by Python)
            {
                std::lock_guard<std::mutex> lk(release_mutex_);
                release_q_.push(task.recv1_addr);
                release_q_.push(task.recv2_addr);
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
            std::lock_guard<std::mutex> lk(release_mutex_);
            release_q_.push(task.addr);

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
            std::lock_guard<std::mutex> lk(release_mutex_);
            release_q_.push(task.addr);

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
        .def("get_buffers_to_release", &ECLATINNative::get_buffers_to_release)
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
        .def("stop", &ECLATINNative::stop);
}


