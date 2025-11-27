/**
 * RDMA Client - Latency Test Sender
 * 
 * This client sends data packets to server via RDMA and measures latency.
 * It measures both one-way and round-trip latencies.
 * Uses RDMA reliable connected (RC) mode.
 * 
 * Compile:
 *   g++ -std=c++17 -O3 rdma_client.cpp -o rdma_client -libverbs -lrdmacm -pthread
 * 
 * Usage:
 *   ./rdma_client <host> <port> <packet_size> <num_packets> [interval_us] [warmup_packets]
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
#include <cmath>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <cassert>

struct PacketHeader {
    uint64_t packet_id;
    uint64_t timestamp_ns;  // Send timestamp in nanoseconds
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
    bool connected;
    
    PacketHeader* send_header() {
        return reinterpret_cast<PacketHeader*>(send_buf);
    }
    
    PacketHeader* recv_header() {
        return reinterpret_cast<PacketHeader*>(recv_buf);
    }
    
    void* send_data() {
        return reinterpret_cast<char*>(send_buf) + sizeof(PacketHeader);
    }
    
    void* recv_data() {
        return reinterpret_cast<char*>(recv_buf) + sizeof(PacketHeader);
    }
};

class RDMAClient {
public:
    RDMAClient(const char* host, const char* port,
               size_t packet_size, size_t num_packets,
               int interval_us, size_t warmup_packets)
        : host_(host), port_(port),
          packet_size_(packet_size),
          num_packets_(num_packets),
          interval_us_(interval_us),
          warmup_packets_(warmup_packets) {
        // Atomic variables are initialized in-class, no need to initialize here
    }
    
    ~RDMAClient() {
        if (ctx_) {
            destroy_context();
        }
        if (event_channel_) {
            rdma_destroy_event_channel(event_channel_);
        }
    }
    
    int run() {
        struct rdma_addrinfo hints = {};
        hints.ai_port_space = RDMA_PS_TCP;
        
        struct rdma_addrinfo* res;
        int ret = rdma_getaddrinfo(host_, port_, &hints, &res);
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
        
        ret = rdma_create_id(event_channel_, &id_, nullptr, RDMA_PS_TCP);
        if (ret) {
            std::cerr << "rdma_create_id failed: " << strerror(errno) << std::endl;
            rdma_freeaddrinfo(res);
            return ret;
        }
        
        ret = rdma_resolve_addr(id_, nullptr, res->ai_dst_addr, 2000);
        if (ret) {
            std::cerr << "rdma_resolve_addr failed: " << strerror(errno) << std::endl;
            rdma_freeaddrinfo(res);
            return ret;
        }
        
        rdma_freeaddrinfo(res);
        
        // Handle connection events
        struct rdma_cm_event* event;
        while (rdma_get_cm_event(event_channel_, &event) == 0) {
            struct rdma_cm_event event_copy;
            memcpy(&event_copy, event, sizeof(*event));
            rdma_ack_cm_event(event);
            
            ret = handle_event(&event_copy);
            if (ret != 0) {
                std::cerr << "Event handling failed" << std::endl;
                break;
            }
            
            // Check if connection is established
            if (ctx_ && ctx_->connected) {
                break;
            }
        }
        
        if (!ctx_ || !ctx_->connected) {
            std::cerr << "Failed to establish connection" << std::endl;
            return -1;
        }
        
        std::cout << "Connected to server" << std::endl;
        std::cout << "Mode: One-way transmission (like ib_send) - only measuring SEND completion latency" << std::endl;
        
        // Pre-initialize data buffer before starting transmission
        // This ensures data preparation time is not included in transmission latency measurement
        std::cout << "Pre-initializing data buffer (" << (packet_size_ / (1024.0 * 1024.0)) << " MB)..." << std::endl;
        auto init_start = std::chrono::high_resolution_clock::now();
        void* data = ctx_->send_data();
        if (packet_size_ > 0) {
            std::memset(data, 0xAA, packet_size_);
        }
        auto init_end = std::chrono::high_resolution_clock::now();
        auto init_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            init_end - init_start).count();
        std::cout << "Data buffer initialized in " << init_duration_us << " us" << std::endl;
        
        // Small delay to ensure server receive buffer is ready
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Start sending packets
        start_sending();
        
        return 0;
    }

private:
    const char* event_to_string(enum rdma_cm_event_type event_type) {
        switch (event_type) {
            case RDMA_CM_EVENT_ADDR_RESOLVED: return "ADDR_RESOLVED";
            case RDMA_CM_EVENT_ADDR_ERROR: return "ADDR_ERROR";
            case RDMA_CM_EVENT_ROUTE_RESOLVED: return "ROUTE_RESOLVED";
            case RDMA_CM_EVENT_ROUTE_ERROR: return "ROUTE_ERROR";
            case RDMA_CM_EVENT_CONNECT_REQUEST: return "CONNECT_REQUEST";
            case RDMA_CM_EVENT_CONNECT_RESPONSE: return "CONNECT_RESPONSE";
            case RDMA_CM_EVENT_CONNECT_ERROR: return "CONNECT_ERROR";
            case RDMA_CM_EVENT_UNREACHABLE: return "UNREACHABLE";
            case RDMA_CM_EVENT_REJECTED: return "REJECTED";
            case RDMA_CM_EVENT_ESTABLISHED: return "ESTABLISHED";
            case RDMA_CM_EVENT_DISCONNECTED: return "DISCONNECTED";
            case RDMA_CM_EVENT_DEVICE_REMOVAL: return "DEVICE_REMOVAL";
            case RDMA_CM_EVENT_MULTICAST_JOIN: return "MULTICAST_JOIN";
            case RDMA_CM_EVENT_MULTICAST_ERROR: return "MULTICAST_ERROR";
            case RDMA_CM_EVENT_ADDR_CHANGE: return "ADDR_CHANGE";
            case RDMA_CM_EVENT_TIMEWAIT_EXIT: return "TIMEWAIT_EXIT";
            default: return "UNKNOWN";
        }
    }
    
    int handle_event(struct rdma_cm_event* event) {
        int ret = 0;
        
        std::cout << "Received event: " << event_to_string(event->event) 
                  << " (code: " << event->event << ")" << std::endl;
        
        // Check for errors first
        if (event->status) {
            std::cerr << "RDMA event error: " << event_to_string(event->event)
                      << " status: " << event->status << " (" << strerror(event->status) << ")" << std::endl;
            return -1;
        }
        
        switch (event->event) {
            case RDMA_CM_EVENT_ADDR_RESOLVED:
                std::cout << "Address resolved" << std::endl;
                ret = on_addr_resolved(event->id);
                break;
            case RDMA_CM_EVENT_ROUTE_RESOLVED:
                std::cout << "Route resolved" << std::endl;
                ret = on_route_resolved(event->id);
                break;
            case RDMA_CM_EVENT_ESTABLISHED:
                std::cout << "Connection established" << std::endl;
                ret = on_connection_established(event->id);
                break;
            case RDMA_CM_EVENT_DISCONNECTED:
                std::cout << "Disconnected" << std::endl;
                on_disconnect(event->id);
                break;
            case RDMA_CM_EVENT_ADDR_ERROR:
                std::cerr << "Address resolution error" << std::endl;
                ret = -1;
                break;
            case RDMA_CM_EVENT_ROUTE_ERROR:
                std::cerr << "Route resolution error" << std::endl;
                ret = -1;
                break;
            case RDMA_CM_EVENT_CONNECT_ERROR:
                std::cerr << "Connection error" << std::endl;
                ret = -1;
                break;
            case RDMA_CM_EVENT_UNREACHABLE:
                std::cerr << "Server unreachable" << std::endl;
                ret = -1;
                break;
            case RDMA_CM_EVENT_REJECTED:
                std::cerr << "Connection rejected" << std::endl;
                ret = -1;
                break;
            default:
                std::cerr << "Unknown or unhandled event: " << event_to_string(event->event)
                          << " (code: " << event->event << ")" << std::endl;
                ret = -1;
                break;
        }
        
        return ret;
    }
    
    int on_addr_resolved(struct rdma_cm_id* id) {
        if (create_context(id) != 0) {
            return -1;
        }
        
        int ret = rdma_resolve_route(id, 2000);
        if (ret) {
            std::cerr << "rdma_resolve_route failed: " << strerror(errno) << std::endl;
            return ret;
        }
        
        return 0;
    }
    
    int on_route_resolved(struct rdma_cm_id* id) {
        struct rdma_conn_param conn_param = {};
        conn_param.initiator_depth = 1;
        conn_param.responder_resources = 1;
        conn_param.retry_count = 7;
        conn_param.rnr_retry_count = 7;
        
        int ret = rdma_connect(id, &conn_param);
        if (ret) {
            std::cerr << "rdma_connect failed: " << strerror(errno) << std::endl;
            return ret;
        }
        
        return 0;
    }
    
    int on_connection_established(struct rdma_cm_id* /*id*/) {
        ctx_->connected = true;
        return 0;
    }
    
    void on_disconnect(struct rdma_cm_id* /*id*/) {
        if (ctx_) {
            ctx_->connected = false;
        }
    }
    
    int create_context(struct rdma_cm_id* id) {
        ctx_ = new ConnectionContext();
        ctx_->id = id;
        ctx_->connected = false;
        // Allocate buffer large enough for packet (header + data)
        // Add some extra space for safety
        ctx_->buf_size = packet_size_ + sizeof(PacketHeader) + 1024;
        // Ensure minimum buffer size of 1MB
        if (ctx_->buf_size < 1024 * 1024) {
            ctx_->buf_size = 1024 * 1024;
        }
        
        id->context = ctx_;
        
        // Get protection domain
        ctx_->pd = ibv_alloc_pd(id->verbs);
        if (!ctx_->pd) {
            std::cerr << "ibv_alloc_pd failed: " << strerror(errno) << std::endl;
            delete ctx_;
            ctx_ = nullptr;
            return -1;
        }
        
        // Create completion queue
        ctx_->cq = ibv_create_cq(id->verbs, 16, nullptr, nullptr, 0);
        if (!ctx_->cq) {
            std::cerr << "ibv_create_cq failed: " << strerror(errno) << std::endl;
            ibv_dealloc_pd(ctx_->pd);
            delete ctx_;
            ctx_ = nullptr;
            return -1;
        }
        
        // Create QP using rdma_cm API (instead of ibv_create_qp)
        struct ibv_qp_init_attr qp_attr = {};
        qp_attr.qp_context = ctx_;
        qp_attr.send_cq = ctx_->cq;
        qp_attr.recv_cq = ctx_->cq;
        qp_attr.srq = nullptr;
        qp_attr.cap.max_send_wr = 32;
        qp_attr.cap.max_recv_wr = 32;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.cap.max_inline_data = 0;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.sq_sig_all = 0;
        
        int ret = rdma_create_qp(id, ctx_->pd, &qp_attr);
        if (ret) {
            std::cerr << "rdma_create_qp failed: " << strerror(errno) << std::endl;
            ibv_destroy_cq(ctx_->cq);
            ibv_dealloc_pd(ctx_->pd);
            delete ctx_;
            ctx_ = nullptr;
            return -1;
        }
        ctx_->qp = id->qp;
        
        // Allocate and register buffers
        ctx_->send_buf = aligned_alloc(4096, ctx_->buf_size);
        ctx_->recv_buf = aligned_alloc(4096, ctx_->buf_size);
        
        if (!ctx_->send_buf || !ctx_->recv_buf) {
            std::cerr << "Buffer allocation failed" << std::endl;
            if (ctx_->send_buf) free(ctx_->send_buf);
            if (ctx_->recv_buf) free(ctx_->recv_buf);
            if (ctx_->qp && id) {
                rdma_destroy_qp(id);
            }
            ibv_destroy_cq(ctx_->cq);
            ibv_dealloc_pd(ctx_->pd);
            delete ctx_;
            ctx_ = nullptr;
            return -1;
        }
        
        std::cout << "Allocating buffers: " << (ctx_->buf_size / (1024 * 1024)) 
                  << " MB (packet size: " << (packet_size_ / (1024 * 1024)) << " MB)" << std::endl;
        
        ctx_->send_mr = ibv_reg_mr(ctx_->pd, ctx_->send_buf, ctx_->buf_size,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_REMOTE_READ);
        ctx_->recv_mr = ibv_reg_mr(ctx_->pd, ctx_->recv_buf, ctx_->buf_size,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_REMOTE_READ);
        
        if (!ctx_->send_mr || !ctx_->recv_mr) {
            std::cerr << "ibv_reg_mr failed: " << strerror(errno) << std::endl;
            if (ctx_->qp) {
                rdma_destroy_qp(id);
            }
            if (ctx_->send_buf) free(ctx_->send_buf);
            if (ctx_->recv_buf) free(ctx_->recv_buf);
            ibv_destroy_cq(ctx_->cq);
            ibv_dealloc_pd(ctx_->pd);
            delete ctx_;
            ctx_ = nullptr;
            return -1;
        }
        
        // Note: When using rdma_create_qp(), the QP state transition is handled
        // automatically by rdma_connect(). No need to manually modify QP state here.
        
        return 0;
    }
    
    void destroy_context() {
        if (!ctx_) return;
        
        // QP will be destroyed with rdma_cm_id, don't destroy separately
        if (ctx_->qp && ctx_->id) {
            rdma_destroy_qp(ctx_->id);
            ctx_->qp = nullptr;
        }
        if (ctx_->send_mr) {
            ibv_dereg_mr(ctx_->send_mr);
        }
        if (ctx_->recv_mr) {
            ibv_dereg_mr(ctx_->recv_mr);
        }
        if (ctx_->send_buf) {
            free(ctx_->send_buf);
        }
        if (ctx_->recv_buf) {
            free(ctx_->recv_buf);
        }
        if (ctx_->cq) {
            ibv_destroy_cq(ctx_->cq);
        }
        if (ctx_->pd) {
            ibv_dealloc_pd(ctx_->pd);
        }
        
        delete ctx_;
        ctx_ = nullptr;
    }
    
    // post_receive_initial removed - not needed for one-way transmission (like ib_send)
    
    void start_sending() {
        send_next_packet();
    }
    
    void send_next_packet() {
        if (packets_sent_ >= num_packets_) {
            rdma_disconnect(id_);
            print_statistics();
            return;
        }
        
        // Prepare packet header only (data buffer is pre-initialized)
        // Note: packet_size_ is the data size (payload only, excluding header)
        PacketHeader* header = ctx_->send_header();
        header->packet_id = packets_sent_;
        
        // Record send time right before posting send (excludes data preparation)
        auto send_time = std::chrono::high_resolution_clock::now();
        header->timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            send_time.time_since_epoch()).count();
        header->data_size = packet_size_;  // data_size equals packet_size_ (payload size)
        
        // Data buffer is already initialized, no need to fill it again
        // This ensures we only measure network transmission latency, not data preparation
        
        // Send complete packet as a single contiguous buffer (header + data)
        // Total packet size = header size + data size (packet_size_)
        // This ensures the entire packet is sent as one unit without splitting
        size_t total_size = sizeof(PacketHeader) + packet_size_;
        send_packet(total_size, send_time, header->packet_id);
        
        packets_sent_++;
    }
    
    void send_packet(size_t size, std::chrono::high_resolution_clock::time_point send_time,
                     uint64_t packet_id) {
        // Send complete packet as a single contiguous buffer - one packet per send, no splitting
        // Similar to ib_send: only measure one-way transmission latency (SEND completion)
        // Data buffer is pre-initialized, so we only measure network transmission time
        auto post_send_start = std::chrono::high_resolution_clock::now();
        
        struct ibv_send_wr wr = {};
        struct ibv_sge sge = {};
        
        sge.addr = reinterpret_cast<uintptr_t>(ctx_->send_buf);
        sge.length = size;  // Total size = header + data
        sge.lkey = ctx_->send_mr->lkey;
        
        wr.sg_list = &sge;
        wr.num_sge = 1;  // Single scatter-gather entry ensures no splitting
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr_id = reinterpret_cast<uintptr_t>(this);
        
        struct ibv_send_wr* bad_wr;
        if (ibv_post_send(ctx_->qp, &wr, &bad_wr)) {
            std::cerr << "ibv_post_send failed: " << strerror(errno) << std::endl;
            return;
        }
        
        auto post_send_end = std::chrono::high_resolution_clock::now();
        auto post_send_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            post_send_end - post_send_start).count();
        
        // Wait only for SEND completion (one-way transmission, like ib_send)
        // No need to wait for echo response
        wait_for_send_completion(send_time, packet_id, post_send_duration_us);
    }
    
    // post_receive removed - not needed for one-way transmission (like ib_send)
    
    void wait_for_send_completion(std::chrono::high_resolution_clock::time_point send_time,
                                  uint64_t packet_id, int64_t post_send_duration_us) {
        // Wait only for SEND completion (one-way transmission, like ib_send)
        struct ibv_wc wc;
        int ret;
        int send_completed = 0;
        int max_iterations = 1000000;  // Prevent infinite loop
        int iteration = 0;
        int64_t total_poll_iterations = 0;
        int64_t total_poll_time_ns = 0;
        
        auto poll_start_time = std::chrono::high_resolution_clock::now();
        auto send_completion_time = std::chrono::high_resolution_clock::time_point();
        
        // Optimized polling: use busy-wait for low latency, fallback to sleep only after many iterations
        const int busy_wait_iterations = 10000;  // Busy-wait for first 10000 iterations
        while (send_completed == 0 && iteration < max_iterations) {
            auto poll_start = std::chrono::high_resolution_clock::now();
            ret = ibv_poll_cq(ctx_->cq, 1, &wc);
            auto poll_end = std::chrono::high_resolution_clock::now();
            auto poll_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                poll_end - poll_start).count();
            total_poll_time_ns += poll_duration_ns;
            total_poll_iterations++;
            
            if (ret > 0) {
                if (wc.status != IBV_WC_SUCCESS) {
                    std::cerr << "Packet " << packet_id << " SEND completion error: " 
                              << ibv_wc_status_str(wc.status) << std::endl;
                    iteration++;
                    continue;
                }
                
                if (wc.opcode == IBV_WC_SEND) {
                    send_completed = 1;
                    send_completion_time = std::chrono::high_resolution_clock::now();
                    auto poll_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        send_completion_time - poll_start_time).count();
                    handle_send_completion(send_time, send_completion_time, packet_id, 
                                         post_send_duration_us, poll_duration_us,
                                         total_poll_iterations, total_poll_time_ns);
                } else if (wc.opcode & IBV_WC_RECV) {
                    // Ignore RECV completions (we don't wait for echo in one-way mode)
                    // Just continue polling for SEND
                }
            } else if (ret < 0) {
                std::cerr << "ibv_poll_cq failed: " << strerror(errno) << std::endl;
                break;
            } else {
                // No completion yet
                // Use busy-wait for first iterations (low latency), then sleep to reduce CPU usage
                if (iteration > busy_wait_iterations) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                // For busy-wait iterations, just continue without sleep
                iteration++;
            }
        }
        
        if (iteration >= max_iterations) {
            std::cerr << "Timeout waiting for SEND completion for packet " << packet_id << std::endl;
            return;
        }
        
        if (interval_us_ > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(interval_us_));
        }
        
        send_next_packet();
    }
    
    void handle_send_completion(std::chrono::high_resolution_clock::time_point send_time,
                                std::chrono::high_resolution_clock::time_point completion_time,
                                uint64_t packet_id, int64_t post_send_duration_us,
                                int64_t poll_duration_us, int64_t total_poll_iterations, int64_t total_poll_time_ns) {
        packets_sent_++;
        packets_received_++;  // Count as received for statistics (one-way completion)
        
        // Skip warmup packets in statistics
        if (packets_received_ <= warmup_packets_) {
            return;
        }
        
        // Calculate one-way latency (SEND completion time)
        auto one_way_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            completion_time - send_time).count();
        double one_way_us = one_way_ns / 1000.0;
        
        // Calculate actual transmission time (excluding post_send and polling overhead)
        // Data preparation is done before sending, so it's not included
        double transmission_time_us = one_way_us - post_send_duration_us - poll_duration_us;
        if (transmission_time_us < 0) transmission_time_us = one_way_us;  // Fallback if calculation is negative
        
        // Calculate effective bandwidth
        double data_size_bytes = static_cast<double>(packet_size_);
        double effective_bandwidth_gbps = (data_size_bytes * 8.0) / (transmission_time_us / 1e6) / 1e9;
        double bandwidth_utilization = (effective_bandwidth_gbps / 100.0) * 100.0;
        double theoretical_time_us = (data_size_bytes * 8.0) / (100.0 * 1000.0);  // 100Gbps in microseconds
        
        total_round_trip_ns_ += one_way_ns;  // Reuse for one-way latency
        total_round_trip_ns_squared_ += (one_way_ns * one_way_ns);
        if (one_way_ns < min_round_trip_ns_) min_round_trip_ns_ = one_way_ns;
        if (one_way_ns > max_round_trip_ns_) max_round_trip_ns_ = one_way_ns;
        
        packets_counted_++;
        
        // Print detailed breakdown for first few packets
        if (packets_counted_ <= 5 || packets_counted_ % 10 == 0) {
            double avg_poll_time_ns = (total_poll_iterations > 0) ? 
                (static_cast<double>(total_poll_time_ns) / total_poll_iterations) : 0.0;
            
            std::cout << "[Client] Packet " << packet_id << " SEND Completion Analysis:" << std::endl;
            std::cout << "  ========================================" << std::endl;
            std::cout << "  Total SEND completion latency: " << std::fixed << std::setprecision(2) 
                      << one_way_us << " us (" << (one_way_us / 1000.0) << " ms)" << std::endl;
            std::cout << "  ========================================" << std::endl;
            std::cout << "  Time Breakdown:" << std::endl;
            std::cout << "    Data preparation: 0 us (pre-initialized before sending)" << std::endl;
            std::cout << "    Post send overhead: " << post_send_duration_us << " us" << std::endl;
            std::cout << "    CQ polling time: " << poll_duration_us << " us" << std::endl;
            std::cout << "    Estimated transmission time: " << std::fixed << std::setprecision(2) 
                      << transmission_time_us << " us" << std::endl;
            std::cout << "  ========================================" << std::endl;
            std::cout << "  Bandwidth Analysis:" << std::endl;
            std::cout << "    Data size: " << (data_size_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
            std::cout << "    Theoretical time @ 100Gbps: " << std::fixed << std::setprecision(2) 
                      << theoretical_time_us << " us" << std::endl;
            std::cout << "    Actual transmission time: " << std::fixed << std::setprecision(2) 
                      << transmission_time_us << " us" << std::endl;
            std::cout << "    Effective bandwidth: " << std::fixed << std::setprecision(2) 
                      << effective_bandwidth_gbps << " Gbps" << std::endl;
            std::cout << "    Bandwidth utilization: " << std::fixed << std::setprecision(1) 
                      << bandwidth_utilization << "%" << std::endl;
            std::cout << "  ========================================" << std::endl;
            std::cout << "  Polling Statistics:" << std::endl;
            std::cout << "    Total poll iterations: " << total_poll_iterations << std::endl;
            std::cout << "    Avg poll_cq duration: " << std::fixed << std::setprecision(2) 
                      << (avg_poll_time_ns / 1000.0) << " us" << std::endl;
            std::cout << "  ========================================" << std::endl;
        }
    }
    
    // handle_response removed - not needed for one-way transmission (like ib_send)
    
    void print_statistics() {
        std::cout << "\n=========================================" << std::endl;
        std::cout << "RDMA Latency Test Statistics" << std::endl;
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
            
            // One-way latency statistics (like ib_send - SEND completion time)
            std::cout << "One-Way Latency Statistics (SEND completion, like ib_send):" << std::endl;
            std::cout << "  Packets measured: " << packets_counted_ << std::endl;
            std::cout << "  Average: " << std::fixed << std::setprecision(2) << avg_rt_us << " us" << std::endl;
            std::cout << "  Minimum: " << std::fixed << std::setprecision(2) << min_rt_us << " us" << std::endl;
            std::cout << "  Maximum: " << std::fixed << std::setprecision(2) << max_rt_us << " us" << std::endl;
            std::cout << "  Std Dev: " << std::fixed << std::setprecision(2) << stddev_us << " us" << std::endl;
            std::cout << std::endl;
            
            // Calculate throughput (one-way transmission only)
            double total_bytes = packets_counted_ * (sizeof(PacketHeader) + packet_size_);
            double total_time_sec = total_round_trip_ns_ / 1e9;
            double throughput_mbps = (total_bytes * 8.0) / (total_time_sec * 1e6);
            
            std::cout << "Throughput: " << std::fixed << std::setprecision(2)
                      << throughput_mbps << " Mbps" << std::endl;
        } else {
            std::cout << "No packets were counted for statistics." << std::endl;
        }
        std::cout << "=========================================" << std::endl;
    }
    
    const char* host_;
    const char* port_;
    size_t packet_size_;
    size_t num_packets_;
    int interval_us_;
    size_t warmup_packets_;
    
    struct rdma_event_channel* event_channel_ = nullptr;
    struct rdma_cm_id* id_ = nullptr;
    ConnectionContext* ctx_ = nullptr;
    
    std::atomic<size_t> packets_sent_{0};
    std::atomic<size_t> packets_received_{0};
    std::atomic<size_t> packets_counted_{0};
    
    std::atomic<int64_t> total_one_way_ns_{0};
    std::atomic<int64_t> total_round_trip_ns_{0};
    std::atomic<int64_t> total_round_trip_ns_squared_{0};  // For standard deviation calculation
    std::atomic<int64_t> min_one_way_ns_{INT64_MAX};
    std::atomic<int64_t> max_one_way_ns_{0};
    std::atomic<int64_t> min_round_trip_ns_{INT64_MAX};
    std::atomic<int64_t> max_round_trip_ns_{0};
};

int main(int argc, char* argv[]) {
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
    std::cout << "RDMA Client - Latency Test Sender" << std::endl;
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
    
    RDMAClient client(host.c_str(), port.c_str(), packet_size, num_packets, interval_us, warmup_packets);
    return client.run();
}

