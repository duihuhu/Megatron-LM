/**
 * RDMA Server - Latency Test Receiver
 * 
 * This server receives data packets from clients via RDMA and measures latency.
 * It echoes back a timestamp to allow round-trip latency measurement.
 * Uses RDMA reliable connected (RC) mode.
 * 
 * Compile:
 *   g++ -std=c++17 -O3 rdma_server.cpp -o rdma_server -libverbs -lrdmacm -pthread
 * 
 * Usage:
 *   ./rdma_server <port>
 */

#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cstring>
#include <thread>
#include <atomic>
#include <memory>
#include <unistd.h>
#include <cassert>

struct PacketHeader {
    uint64_t packet_id;
    uint64_t timestamp_ns;  // Client send timestamp in nanoseconds
    uint32_t data_size;     // Size of payload data in bytes
};

struct ConnectionContext {
    struct rdma_cm_id* id;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    struct ibv_mr* send_mr;
    struct ibv_mr* recv_mr;
    void* send_buf;
    void* recv_buf;
    size_t buf_size;
    int session_id;
    std::atomic<uint64_t> packets_received{0};
    std::atomic<uint64_t> total_latency_ns{0};
    std::atomic<bool> stop_polling{false};
    
    PacketHeader* recv_header() {
        return reinterpret_cast<PacketHeader*>(recv_buf);
    }
    
    PacketHeader* send_header() {
        return reinterpret_cast<PacketHeader*>(send_buf);
    }
    
    void* recv_data() {
        return reinterpret_cast<char*>(recv_buf) + sizeof(PacketHeader);
    }
    
    void* send_data() {
        return reinterpret_cast<char*>(send_buf) + sizeof(PacketHeader);
    }
};

class RDMAServer {
public:
    RDMAServer(const char* port) : port_(port), session_counter_(0) {}
    
    ~RDMAServer() {
        if (listen_id_) {
            rdma_destroy_id(listen_id_);
        }
        if (event_channel_) {
            rdma_destroy_event_channel(event_channel_);
        }
    }
    
    int run() {
        struct rdma_addrinfo hints = {};
        hints.ai_flags = RAI_PASSIVE;
        hints.ai_port_space = RDMA_PS_TCP;
        
        struct rdma_addrinfo* res;
        int ret = rdma_getaddrinfo(nullptr, port_, &hints, &res);
        if (ret) {
            std::cerr << "rdma_getaddrinfo failed: " << strerror(errno) << std::endl;
            return ret;
        }
        
        event_channel_ = rdma_create_event_channel();
        if (!event_channel_) {
            std::cerr << "rdma_create_event_channel failed: " << strerror(errno) << std::endl;
            rdma_freeaddrinfo(res);
            return -1;
        }
        
        ret = rdma_create_id(event_channel_, &listen_id_, nullptr, RDMA_PS_TCP);
        if (ret) {
            std::cerr << "rdma_create_id failed: " << strerror(errno) << std::endl;
            rdma_destroy_event_channel(event_channel_);
            rdma_freeaddrinfo(res);
            return ret;
        }
        
        ret = rdma_bind_addr(listen_id_, res->ai_src_addr);
        if (ret) {
            std::cerr << "rdma_bind_addr failed: " << strerror(errno) << std::endl;
            rdma_destroy_id(listen_id_);
            rdma_destroy_event_channel(event_channel_);
            rdma_freeaddrinfo(res);
            return ret;
        }
        
        ret = rdma_listen(listen_id_, 10);
        if (ret) {
            std::cerr << "rdma_listen failed: " << strerror(errno) << std::endl;
            rdma_destroy_id(listen_id_);
            rdma_destroy_event_channel(event_channel_);
            rdma_freeaddrinfo(res);
            return ret;
        }
        
        rdma_freeaddrinfo(res);
        
        std::cout << "=========================================" << std::endl;
        std::cout << "RDMA Server - Latency Test Receiver" << std::endl;
        std::cout << "Listening on port: " << port_ << std::endl;
        std::cout << "=========================================" << std::endl;
        
        // Handle connection events
        struct rdma_cm_event* event;
        while (rdma_get_cm_event(event_channel_, &event) == 0) {
            struct rdma_cm_event event_copy;
            memcpy(&event_copy, event, sizeof(*event));
            rdma_ack_cm_event(event);
            
            if (handle_event(&event_copy) != 0) {
                break;
            }
        }
        
        return 0;
    }

private:
    int handle_event(struct rdma_cm_event* event) {
        int ret = 0;
        
        switch (event->event) {
            case RDMA_CM_EVENT_CONNECT_REQUEST:
                ret = on_connect_request(event->id);
                break;
            case RDMA_CM_EVENT_ESTABLISHED:
                ret = on_connection_established(event->id);
                break;
            case RDMA_CM_EVENT_DISCONNECTED:
                ret = on_disconnect(event->id);
                break;
            default:
                std::cerr << "Unknown event: " << event->event << std::endl;
                break;
        }
        
        return ret;
    }
    
    int on_connect_request(struct rdma_cm_id* id) {
        struct rdma_conn_param conn_param = {};
        
        // Create context first (allocates PD, CQ, buffers)
        ConnectionContext* ctx = create_context(id);
        if (!ctx) {
            return -1;
        }
        
        // Now create QP using rdma_create_qp with context's PD and CQ
        struct ibv_qp_init_attr qp_attr = {};
        qp_attr.qp_context = ctx;
        qp_attr.cap.max_send_wr = 32;
        qp_attr.cap.max_recv_wr = 32;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.cap.max_inline_data = 0;
        qp_attr.sq_sig_all = 0;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.send_cq = ctx->cq;
        qp_attr.recv_cq = ctx->cq;
        
        int ret = rdma_create_qp(id, ctx->pd, &qp_attr);
        if (ret) {
            std::cerr << "rdma_create_qp failed: " << strerror(errno) << std::endl;
            destroy_context(ctx);
            return ret;
        }
        ctx->qp = id->qp;
        
        ret = rdma_accept(id, &conn_param);
        if (ret) {
            std::cerr << "rdma_accept failed: " << strerror(errno) << std::endl;
            rdma_destroy_qp(id);
            destroy_context(ctx);
            return ret;
        }
        
        return 0;
    }
    
    int on_connection_established(struct rdma_cm_id* id) {
        ConnectionContext* ctx = static_cast<ConnectionContext*>(id->context);
        if (!ctx) {
            return -1;
        }
        
        std::cout << "New client connected, session ID: " << ctx->session_id << std::endl;
        
        // Post multiple receive buffers to avoid RNR errors
        // Pre-post 4 receive buffers to handle rapid incoming packets
        for (int i = 0; i < 4; i++) {
            post_receive_only(ctx);
        }
        
        // Start a thread to handle completion polling for this connection
        std::thread poll_thread([this, ctx]() {
            this->completion_poll_loop(ctx);
        });
        poll_thread.detach();
        
        return 0;
    }
    
    int on_disconnect(struct rdma_cm_id* id) {
        ConnectionContext* ctx = static_cast<ConnectionContext*>(id->context);
        if (ctx) {
            std::cout << "Client disconnected, session ID: " << ctx->session_id << std::endl;
            ctx->stop_polling = true;
            destroy_context(ctx);
        }
        return 0;
    }
    
    ConnectionContext* create_context(struct rdma_cm_id* id) {
        ConnectionContext* ctx = new ConnectionContext();
        ctx->id = id;
        ctx->session_id = ++session_counter_;
        // Use large buffer to support large packet sizes (up to 128MB)
        // This should be large enough for most test scenarios
        ctx->buf_size = 128 * 1024 * 1024;  // 128MB buffer
        
        id->context = ctx;
        
        // Get protection domain
        ctx->pd = ibv_alloc_pd(id->verbs);
        if (!ctx->pd) {
            std::cerr << "ibv_alloc_pd failed: " << strerror(errno) << std::endl;
            delete ctx;
            return nullptr;
        }
        
        // Create completion queue
        ctx->cq = ibv_create_cq(id->verbs, 16, nullptr, nullptr, 0);
        if (!ctx->cq) {
            std::cerr << "ibv_create_cq failed: " << strerror(errno) << std::endl;
            ibv_dealloc_pd(ctx->pd);
            delete ctx;
            return nullptr;
        }
        
        // Allocate and register buffers
        ctx->send_buf = aligned_alloc(4096, ctx->buf_size);
        ctx->recv_buf = aligned_alloc(4096, ctx->buf_size);
        
        if (!ctx->send_buf || !ctx->recv_buf) {
            std::cerr << "Buffer allocation failed" << std::endl;
            if (ctx->send_buf) free(ctx->send_buf);
            if (ctx->recv_buf) free(ctx->recv_buf);
            ibv_destroy_cq(ctx->cq);
            ibv_dealloc_pd(ctx->pd);
            delete ctx;
            return nullptr;
        }
        
        std::cout << "[Session " << ctx->session_id << "] Allocating buffers: " 
                  << (ctx->buf_size / (1024 * 1024)) << " MB" << std::endl;
        
        ctx->send_mr = ibv_reg_mr(ctx->pd, ctx->send_buf, ctx->buf_size,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                   IBV_ACCESS_REMOTE_READ);
        ctx->recv_mr = ibv_reg_mr(ctx->pd, ctx->recv_buf, ctx->buf_size,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                   IBV_ACCESS_REMOTE_READ);
        
        if (!ctx->send_mr || !ctx->recv_mr) {
            std::cerr << "[Session " << ctx->session_id << "] ibv_reg_mr failed: " << strerror(errno) << std::endl;
            if (ctx->send_buf) free(ctx->send_buf);
            if (ctx->recv_buf) free(ctx->recv_buf);
            ibv_destroy_cq(ctx->cq);
            ibv_dealloc_pd(ctx->pd);
            delete ctx;
            return nullptr;
        }
        
        // QP will be created in on_connect_request using rdma_create_qp
        ctx->qp = nullptr;
        
        return ctx;
    }
    
    void destroy_context(ConnectionContext* ctx) {
        if (!ctx) return;
        
        // QP is destroyed with rdma_cm_id, don't destroy separately
        if (ctx->qp && ctx->id) {
            rdma_destroy_qp(ctx->id);
            ctx->qp = nullptr;
        }
        if (ctx->send_mr) {
            ibv_dereg_mr(ctx->send_mr);
        }
        if (ctx->recv_mr) {
            ibv_dereg_mr(ctx->recv_mr);
        }
        if (ctx->send_buf) {
            free(ctx->send_buf);
        }
        if (ctx->recv_buf) {
            free(ctx->recv_buf);
        }
        if (ctx->cq) {
            ibv_destroy_cq(ctx->cq);
        }
        if (ctx->pd) {
            ibv_dealloc_pd(ctx->pd);
        }
        if (ctx->id) {
            rdma_destroy_id(ctx->id);
        }
        
        delete ctx;
    }
    
    void post_receive_only(ConnectionContext* ctx) {
        struct ibv_recv_wr wr = {};
        struct ibv_sge sge = {};
        
        sge.addr = reinterpret_cast<uintptr_t>(ctx->recv_buf);
        sge.length = ctx->buf_size;
        sge.lkey = ctx->recv_mr->lkey;
        
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.wr_id = reinterpret_cast<uintptr_t>(ctx);
        
        struct ibv_recv_wr* bad_wr;
        if (ibv_post_recv(ctx->qp, &wr, &bad_wr)) {
            std::cerr << "[Session " << ctx->session_id << "] ibv_post_recv failed: " << strerror(errno) << std::endl;
        }
    }
    
    void completion_poll_loop(ConnectionContext* ctx) {
        struct ibv_wc wc;
        int iteration = 0;
        const int busy_wait_iterations = 10000;  // Busy-wait for first 10000 iterations
        auto last_poll_time = std::chrono::high_resolution_clock::now();
        int64_t total_poll_iterations = 0;
        int64_t total_poll_time_ns = 0;
        
        while (!ctx->stop_polling && ctx->qp) {
            auto poll_start = std::chrono::high_resolution_clock::now();
            int ret = ibv_poll_cq(ctx->cq, 1, &wc);
            auto poll_end = std::chrono::high_resolution_clock::now();
            auto poll_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                poll_end - poll_start).count();
            total_poll_time_ns += poll_duration_ns;
            total_poll_iterations++;
            
            if (ret > 0) {
                iteration = 0;  // Reset counter on successful poll
                if (wc.status != IBV_WC_SUCCESS) {
                    if (wc.status != IBV_WC_WR_FLUSH_ERR) {  // Ignore flush errors on disconnect
                        std::cerr << "[Session " << ctx->session_id << "] Completion error: " 
                                  << ibv_wc_status_str(wc.status) << std::endl;
                    }
                    if (wc.status == IBV_WC_WR_FLUSH_ERR) {
                        // Connection error or flush, stop polling
                        break;
                    }
                    continue;
                }
                
                if (wc.opcode & IBV_WC_RECV) {
                    // RECV completion means data has been received in server's receive buffer
                    // This is the server-side confirmation that data has arrived
                    auto recv_completion_time = std::chrono::high_resolution_clock::now();
                    
                    // Calculate time since last poll (approximate CQ polling interval)
                    auto time_since_last_poll = std::chrono::duration_cast<std::chrono::microseconds>(
                        poll_start - last_poll_time).count();
                    last_poll_time = poll_start;
                    
                    handle_recv(ctx);
                    auto after_handle_time = std::chrono::high_resolution_clock::now();
                    auto handle_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        after_handle_time - recv_completion_time).count();
                    
                    // Immediately post new receive buffer for next packet
                    post_receive_only(ctx);
                    auto after_post_time = std::chrono::high_resolution_clock::now();
                    auto post_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        after_post_time - after_handle_time).count();
                    
                    // Print polling breakdown for first few packets
                    if (ctx->packets_received <= 5 || ctx->packets_received % 10 == 0) {
                        double avg_poll_time_ns = (total_poll_iterations > 0) ? 
                            (static_cast<double>(total_poll_time_ns) / total_poll_iterations) : 0.0;
                        std::cout << "  [Server] RECV Completion Analysis:" << std::endl;
                        std::cout << "    RECV completion: Data confirmed received in server buffer" << std::endl;
                        std::cout << "    Time since last poll: " << time_since_last_poll << " us" << std::endl;
                        std::cout << "    Avg poll_cq duration: " << (avg_poll_time_ns / 1000.0) << " us" << std::endl;
                        std::cout << "    Total poll iterations: " << total_poll_iterations << std::endl;
                        std::cout << "    handle_recv duration: " << handle_duration_us << " us" << std::endl;
                        std::cout << "    post_receive duration: " << post_duration_us << " us" << std::endl;
                    }
                } else if (wc.opcode == IBV_WC_SEND) {
                    // Send completed (echo response sent, but we don't echo in one-way mode)
                }
            } else if (ret < 0) {
                std::cerr << "[Session " << ctx->session_id << "] ibv_poll_cq failed: " << strerror(errno) << std::endl;
                break;
            } else {
                // No completion yet
                // Use busy-wait for first iterations (low latency), then sleep to reduce CPU usage
                if (iteration > busy_wait_iterations) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                iteration++;
            }
        }
    }
    
    void handle_recv(ConnectionContext* ctx) {
        auto recv_start_time = std::chrono::high_resolution_clock::now();
        
        PacketHeader* header = ctx->recv_header();
        
        // Validate received packet size
        size_t expected_size = sizeof(PacketHeader) + header->data_size;
        if (expected_size > ctx->buf_size) {
            std::cerr << "[Session " << ctx->session_id << "] Error: Received packet size (" 
                      << expected_size << " bytes) exceeds buffer size (" << ctx->buf_size 
                      << " bytes)" << std::endl;
            return;
        }
        
        auto validation_time = std::chrono::high_resolution_clock::now();
        auto validation_us = std::chrono::duration_cast<std::chrono::microseconds>(
            validation_time - recv_start_time).count();
        
        auto receive_time = std::chrono::high_resolution_clock::now();
        auto receive_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            receive_time.time_since_epoch()).count();
        
        // Calculate one-way latency (from client send to server receive)
        int64_t network_latency_ns = 0;
        if (header->timestamp_ns > 0) {
            network_latency_ns = receive_time_ns - static_cast<int64_t>(header->timestamp_ns);
            ctx->total_latency_ns += network_latency_ns;
            ctx->packets_received++;
        }
        
        auto latency_calc_time = std::chrono::high_resolution_clock::now();
        auto latency_calc_us = std::chrono::duration_cast<std::chrono::microseconds>(
            latency_calc_time - receive_time).count();
        
        // Prepare echo packet with server timestamp
        // For one-way transmission test, only echo back header (no data)
        // This avoids unnecessary 64MB memcpy and reduces network traffic
        PacketHeader* send_header = ctx->send_header();
        *send_header = *header;
        send_header->timestamp_ns = receive_time_ns;
        send_header->data_size = 0;  // No data in echo response
        
        auto prep_time = std::chrono::high_resolution_clock::now();
        auto prep_us = std::chrono::duration_cast<std::chrono::microseconds>(
            prep_time - latency_calc_time).count();
        
        // Send only header as echo response (one-way transmission)
        // Total size = header size only (no data)
        size_t total_size = sizeof(PacketHeader);
        send_response(ctx, total_size);
        
        auto send_post_time = std::chrono::high_resolution_clock::now();
        auto send_post_us = std::chrono::duration_cast<std::chrono::microseconds>(
            send_post_time - prep_time).count();
        
        auto total_handle_us = std::chrono::duration_cast<std::chrono::microseconds>(
            send_post_time - recv_start_time).count();
        
        // Print detailed breakdown for first few packets
        if (ctx->packets_received <= 5 || ctx->packets_received % 10 == 0) {
            double data_size_mb = header->data_size / (1024.0 * 1024.0);
            double data_size_bytes = static_cast<double>(header->data_size);
            double theoretical_time_us = (data_size_bytes * 8.0) / (100.0 * 1000.0);  // 100Gbps in microseconds
            double actual_time_us = network_latency_ns / 1000.0;
            double effective_bandwidth_gbps = (data_size_bytes * 8.0) / (actual_time_us / 1e6) / 1e9;
            double bandwidth_utilization = (effective_bandwidth_gbps / 100.0) * 100.0;
            
            std::cout << "[Session " << ctx->session_id << "] Packet " << header->packet_id 
                      << " Time Breakdown (us):" << std::endl;
            std::cout << "  ========================================" << std::endl;
            std::cout << "  Total latency (client send -> server recv): " 
                      << actual_time_us << " us (" << (actual_time_us / 1000.0) << " ms)" << std::endl;
            std::cout << "  Data size: " << data_size_mb << " MB (" << header->data_size << " bytes)" << std::endl;
            std::cout << "  ========================================" << std::endl;
            std::cout << "  Bandwidth Analysis:" << std::endl;
            std::cout << "    Theoretical time @ 100Gbps: " << theoretical_time_us << " us" << std::endl;
            std::cout << "    Actual transmission time: " << actual_time_us << " us" << std::endl;
            std::cout << "    Effective bandwidth: " << std::fixed << std::setprecision(2) 
                      << effective_bandwidth_gbps << " Gbps" << std::endl;
            std::cout << "    Bandwidth utilization: " << std::fixed << std::setprecision(1) 
                      << bandwidth_utilization << "%" << std::endl;
            std::cout << "    Overhead ratio: " << std::fixed << std::setprecision(1) 
                      << ((actual_time_us / theoretical_time_us - 1.0) * 100.0) << "%" << std::endl;
            std::cout << "  ========================================" << std::endl;
            std::cout << "  Server-side processing breakdown:" << std::endl;
            std::cout << "    - Validation: " << validation_us << " us" << std::endl;
            std::cout << "    - Latency calculation: " << latency_calc_us << " us" << std::endl;
            std::cout << "    - Prepare echo: " << prep_us << " us" << std::endl;
            std::cout << "    - Post send: " << send_post_us << " us" << std::endl;
            std::cout << "    - Total handle_recv: " << total_handle_us << " us" << std::endl;
            std::cout << "  ========================================" << std::endl;
            std::cout << "  Time Distribution:" << std::endl;
            std::cout << "    Network transmission: ~" << (actual_time_us - total_handle_us) << " us (" 
                      << std::fixed << std::setprecision(1) 
                      << (((actual_time_us - total_handle_us) / actual_time_us) * 100.0) << "%)" << std::endl;
            std::cout << "    Server processing: " << total_handle_us << " us (" 
                      << std::fixed << std::setprecision(1) 
                      << ((total_handle_us / actual_time_us) * 100.0) << "%)" << std::endl;
        }
    }
    
    void send_response(ConnectionContext* ctx, size_t size) {
        // Send complete echo packet as a single contiguous buffer - one packet per send, no splitting
        struct ibv_send_wr wr = {};
        struct ibv_sge sge = {};
        
        sge.addr = reinterpret_cast<uintptr_t>(ctx->send_buf);
        sge.length = size;  // Total size = header + data
        sge.lkey = ctx->send_mr->lkey;
        
        wr.sg_list = &sge;
        wr.num_sge = 1;  // Single scatter-gather entry ensures no splitting
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr_id = reinterpret_cast<uintptr_t>(ctx);
        
        struct ibv_send_wr* bad_wr;
        if (ibv_post_send(ctx->qp, &wr, &bad_wr)) {
            std::cerr << "[Session " << ctx->session_id << "] ibv_post_send failed: " << strerror(errno) << std::endl;
            return;
        }
        
        // Send completion will be handled by completion_poll_loop
    }
    
    const char* port_;
    struct rdma_event_channel* event_channel_ = nullptr;
    struct rdma_cm_id* listen_id_ = nullptr;
    std::atomic<int> session_counter_;
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <port>" << std::endl;
        return 1;
    }
    
    const char* port = argv[1];
    
    RDMAServer server(port);
    return server.run();
}

