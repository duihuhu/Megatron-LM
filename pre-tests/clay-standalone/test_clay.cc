/*
 * Clay 纠删码测试程序
 * 
 * 测试功能：
 * 1. 创建 Clay 编码器
 * 2. 编码测试数据
 * 3. 模拟数据块丢失
 * 4. 解码恢复丢失的数据
 * 5. 验证数据正确性
 */

#include "erasure_code_clay.hh"
#include "erasure_code_factory.hpp"
#include "erasure_code_intf.hpp"
#include "exception.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <map>
#include <cstring>
#include <cassert>
#include <random>
#include <numeric>
#include <sstream>
#include <chrono>
#include <algorithm>

using namespace ec;
using namespace ceph;
using namespace ceph::buffer;

// 辅助函数：打印 bufferlist 内容（前 N 个字节）
void print_bufferlist(const bufferlist &bl, const std::string &name, size_t max_bytes = 32) {
    std::cout << name << " (size=" << bl.length() << "): ";
    size_t to_print = std::min(static_cast<size_t>(bl.length()), max_bytes);
    // 使用迭代器安全访问 const bufferlist
    bufferlist::const_iterator it = bl.begin();
    for (size_t i = 0; i < to_print && it != bl.end(); i++, ++it) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (unsigned char)(*it) << " ";
    }
    if (static_cast<size_t>(bl.length()) > max_bytes) {
        std::cout << "...";
    }
    std::cout << std::dec << std::endl;
}

// 辅助函数：比较两个 bufferlist 是否相等
bool compare_bufferlist(const bufferlist &bl1, const bufferlist &bl2) {
    if (bl1.length() != bl2.length()) {
        return false;
    }
    // 使用迭代器访问 const bufferlist 的数据
    bufferlist::const_iterator it1 = bl1.begin();
    bufferlist::const_iterator it2 = bl2.begin();
    size_t len = bl1.length();
    for (size_t i = 0; i < len; i++) {
        if (*it1 != *it2) {
            return false;
        }
        ++it1;
        ++it2;
    }
    return true;
}

// 辅助函数：从 bufferlist 提取数据到 vector
std::vector<char> bufferlist_to_vector(const bufferlist &bl) {
    std::vector<char> result(bl.length());
    bufferlist::const_iterator it = bl.begin();
    for (size_t i = 0; i < bl.length(); i++) {
        result[i] = *it;
        ++it;
    }
    return result;
}

// 测试 1: 基本编码和解码测试
bool test_basic_encode_decode(int k, int m, size_t data_size) {
    std::cout << "\n=== 测试 1: 基本编码和解码 (k=" << k << ", m=" << m 
              << ", data_size=" << data_size << ") ===" << std::endl;

    try {
        // 1. 创建 Clay 编码器
        ErasureCodeProfile profile;
        profile["k"] = std::to_string(k);
        profile["m"] = std::to_string(m);
        profile["d"] = std::to_string(k + m - 1);  // d = k + m - 1 (默认值)
        profile["scalar_mds"] = "jerasure";
        profile["technique"] = "reed_sol_van";

        std::ostringstream errors;
        ErasureCodeClayFactory factory;
        auto encoder = factory.make(profile, errors);

        if (!encoder) {
            std::cerr << "创建编码器失败: " << errors.str() << std::endl;
            return false;
        }

        std::cout << "✓ 编码器创建成功" << std::endl;
        std::cout << "  - 数据块数 (k): " << encoder->get_data_chunk_count() << std::endl;
        std::cout << "  - 校验块数 (m): " << encoder->get_coding_chunk_count() << std::endl;
        std::cout << "  - 总块数: " << encoder->get_chunk_count() << std::endl;

        // 2. 准备测试数据
        std::vector<char> original_data(data_size);
        std::iota(original_data.begin(), original_data.end(), 0);  // 填充 0, 1, 2, ...
        
        // 创建 bufferlist - 使用原项目的方式
        bufferlist input;
        bufferptr input_ptr(buffer::create_page_aligned(data_size));
        input_ptr.zero();
        input_ptr.set_length(0);
        input_ptr.append(original_data.data(), data_size);
        input.push_back(input_ptr);

        std::cout << "✓ 测试数据准备完成 (大小: " << data_size << " 字节)" << std::endl;

        // 3. 编码
        std::set<int> want_to_encode;
        for (int i = 0; i < k + m; i++) {
            want_to_encode.insert(i);
        }

        std::map<int, bufferlist> encoded;
        int encode_result = encoder->encode(want_to_encode, input, &encoded);
        
        if (encode_result != 0) {
            std::cerr << "✗ 编码失败，错误码: " << encode_result << std::endl;
            return false;
        }

        std::cout << "✓ 编码成功，生成 " << encoded.size() << " 个块" << std::endl;

        // 验证编码后的块大小
        unsigned int chunk_size = encoder->get_chunk_size(data_size);
        std::cout << "  - 每个块大小: " << chunk_size << " 字节" << std::endl;

        // 4. 完整解码测试（所有块都可用）
        std::set<int> want_to_read;
        for (int i = 0; i < k; i++) {
            want_to_read.insert(i);
        }

        std::map<int, bufferlist> decoded;
        int decode_result = encoder->decode(want_to_read, encoded, &decoded, chunk_size);
        
        if (decode_result != 0) {
            std::cerr << "✗ 解码失败，错误码: " << decode_result << std::endl;
            return false;
        }

        std::cout << "✓ 解码成功" << std::endl;

        // 5. 验证解码后的数据
        bufferlist recovered;
        for (int i = 0; i < k; i++) {
            if (decoded.find(i) != decoded.end()) {
                recovered.claim_append(decoded[i]);
            }
        }

        // 只比较原始数据大小的部分（因为可能有填充）
        size_t compare_size = std::min(original_data.size(), static_cast<size_t>(recovered.length()));
        // 使用迭代器比较数据
        bool data_match = true;
        bufferlist::const_iterator it = recovered.begin();
        for (size_t i = 0; i < compare_size; i++) {
            if (original_data[i] != *it) {
                data_match = false;
                break;
            }
            ++it;
        }

        if (data_match) {
            std::cout << "✓ 数据验证通过！原始数据与恢复数据完全匹配" << std::endl;
            return true;
        } else {
            std::cerr << "✗ 数据验证失败！原始数据与恢复数据不匹配" << std::endl;
            return false;
        }

    } catch (const std::exception &e) {
        std::cerr << "✗ 异常: " << e.what() << std::endl;
        return false;
    }
}

// 测试 2: 数据丢失恢复测试
bool test_data_loss_recovery(int k, int m, size_t data_size, int lost_chunks) {
    std::cout << "\n=== 测试 2: 数据丢失恢复 (k=" << k << ", m=" << m 
              << ", 丢失 " << lost_chunks << " 个块) ===" << std::endl;

    try {
        // 1. 创建编码器
        ErasureCodeProfile profile;
        profile["k"] = std::to_string(k);
        profile["m"] = std::to_string(m);
        profile["d"] = std::to_string(k + m - 1);
        profile["scalar_mds"] = "jerasure";
        profile["technique"] = "reed_sol_van";

        std::ostringstream errors;
        ErasureCodeClayFactory factory;
        auto encoder = factory.make(profile, errors);

        if (!encoder) {
            std::cerr << "创建编码器失败: " << errors.str() << std::endl;
            return false;
        }

        // 2. 准备测试数据
        std::vector<char> original_data(data_size);
        std::iota(original_data.begin(), original_data.end(), 0);

        bufferlist input;
        bufferptr input_ptr(buffer::create_page_aligned(data_size));
        input_ptr.zero();
        input_ptr.set_length(0);
        input_ptr.append(original_data.data(), data_size);
        input.push_back(input_ptr);

        // 3. 编码
        std::set<int> want_to_encode;
        for (int i = 0; i < k + m; i++) {
            want_to_encode.insert(i);
        }

        std::map<int, bufferlist> encoded;
        int encode_result = encoder->encode(want_to_encode, input, &encoded);
        
        if (encode_result != 0) {
            std::cerr << "✗ 编码失败" << std::endl;
            return false;
        }

        std::cout << "✓ 编码成功，生成 " << encoded.size() << " 个块" << std::endl;

        // 4. 模拟数据丢失（随机丢失 lost_chunks 个数据块）
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, k - 1);
        
        std::set<int> lost_indices;
        while (lost_indices.size() < (size_t)lost_chunks && lost_indices.size() < (size_t)k) {
            lost_indices.insert(dis(gen));
        }

        std::map<int, bufferlist> available_chunks = encoded;
        for (int idx : lost_indices) {
            available_chunks.erase(idx);
            std::cout << "  - 丢失块: " << idx << std::endl;
        }

        std::cout << "✓ 模拟数据丢失完成，剩余 " << available_chunks.size() << " 个块" << std::endl;

        // 5. 计算需要哪些块来恢复
        std::set<int> want_to_read = lost_indices;
        std::set<int> available;
        for (const auto &pair : available_chunks) {
            available.insert(pair.first);
        }

        std::map<int, std::vector<std::pair<int, int>>> minimum;
        int min_result = encoder->minimum_to_decode(want_to_read, available, &minimum);
        
        if (min_result != 0) {
            std::cerr << "✗ 计算最小恢复集失败" << std::endl;
            return false;
        }

        std::cout << "✓ 最小恢复集计算完成，需要 " << minimum.size() << " 个块" << std::endl;

        // 6. 解码恢复
        unsigned int chunk_size = encoder->get_chunk_size(data_size);

        std::map<int, bufferlist> decoded;
        int decode_result = encoder->decode(want_to_read, available_chunks, &decoded, chunk_size);
        
        if (decode_result != 0) {
            std::cerr << "✗ 解码恢复失败，错误码: " << decode_result << std::endl;
            return false;
        }

        std::cout << "✓ 解码恢复成功" << std::endl;

        // 7. 验证恢复的数据
        bool all_recovered = true;
        for (int idx : lost_indices) {
            if (decoded.find(idx) == decoded.end()) {
                std::cerr << "✗ 块 " << idx << " 未能恢复" << std::endl;
                all_recovered = false;
                continue;
            }

            // 比较恢复的块与原始编码的块
            if (!compare_bufferlist(decoded[idx], encoded[idx])) {
                std::cerr << "✗ 块 " << idx << " 恢复的数据不匹配" << std::endl;
                all_recovered = false;
            }
        }

        if (all_recovered) {
            std::cout << "✓ 所有丢失的块都已成功恢复并验证！" << std::endl;
            return true;
        } else {
            std::cerr << "✗ 部分块恢复失败" << std::endl;
            return false;
        }

    } catch (const std::exception &e) {
        std::cerr << "✗ 异常: " << e.what() << std::endl;
        return false;
    }
}

// 测试 3: 不同数据大小的测试
bool test_different_sizes(int k, int m) {
    std::cout << "\n=== 测试 3: 不同数据大小测试 (k=" << k << ", m=" << m << ") ===" << std::endl;

    std::vector<size_t> test_sizes = {1024, 4096, 16384, 65536};
    bool all_passed = true;

    for (size_t size : test_sizes) {
        std::cout << "\n测试数据大小: " << size << " 字节" << std::endl;
        bool result = test_basic_encode_decode(k, m, size);
        if (!result) {
            all_passed = false;
        }
    }

    return all_passed;
}

// 辅助函数：格式化字节大小为可读格式
std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 3) {
        size /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

// 辅助函数：格式化吞吐量
std::string format_throughput(double throughput_mbps) {
    std::ostringstream oss;
    if (throughput_mbps >= 1000.0) {
        oss << std::fixed << std::setprecision(2) << (throughput_mbps / 1000.0) << " GB/s";
    } else {
        oss << std::fixed << std::setprecision(2) << throughput_mbps << " MB/s";
    }
    return oss.str();
}

// 测试 4/5: 编码吞吐量测试
bool test_encode_throughput(int k, int m) {
    std::string test_name = (k == 4 && m == 2) ? "测试 4" : (k == 2 && m == 2) ? "测试 5" : "编码吞吐量测试";
    std::cout << "\n=== " << test_name << ": 编码吞吐量测试 (k=" << k << ", m=" << m << ") ===" << std::endl;

    try {
        // 编码器配置（所有数据大小共享）
        ErasureCodeProfile base_profile;
        base_profile["k"] = std::to_string(k);
        base_profile["m"] = std::to_string(m);
        base_profile["d"] = std::to_string(k + m - 1);
        base_profile["scalar_mds"] = "jerasure";
        base_profile["technique"] = "reed_sol_van";

        std::cout << "✓ 编码器配置准备完成" << std::endl;

        // 2. 定义测试数据大小（KB 和 MB 级别）
        // 注意：小数据时固定开销占比大，吞吐量会较低
        // 建议关注 64KB 以上的性能，这才是 Clay Code 的优势场景
        std::vector<size_t> test_sizes = {
            // 1 * 1024,        // 1 KB（固定开销大，吞吐量低）
            // 4 * 1024,        // 4 KB
            // 16 * 1024,       // 16 KB
            // 64 * 1024,       // 64 KB（开始进入优势区间）
            // 256 * 1024,      // 256 KB
            // 512 * 1024,      // 512 KB
            1 * 1024 * 1024,    // 1 MB
            4 * 1024 * 1024,    // 4 MB
            16 * 1024 * 1024,   // 16 MB
            64 * 1024 * 1024    // 64 MB
        };

        // 每个大小测试的次数（用于计算平均值）
        const int num_iterations = 10;
        const int warmup_iterations = 3;  // 预热迭代，不计入统计

        std::cout << "\n开始编码吞吐量测试..." << std::endl;
        std::cout << "注意：Clay code 是分层编码方案：" << std::endl;
        if (k == 4 && m == 2) {
            std::cout << "  - k=4, m=2: sub_chunk_no = 8 (需要处理 8 个子块)" << std::endl;
        } else if (k == 2 && m == 2) {
            std::cout << "  - k=2, m=2: sub_chunk_no = 4 (需要处理 4 个子块)" << std::endl;
        } else {
            std::cout << "  - k=" << k << ", m=" << m << ": 需要处理多个子块" << std::endl;
        }
        std::cout << "  - 需要调用 MDS 和 PFT 编码器" << std::endl;
        std::cout << "  - 小数据时固定开销占比大，吞吐量会较低" << std::endl;
        std::cout << "每个大小测试 " << num_iterations << " 次（前 " << warmup_iterations 
                  << " 次为预热）" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        std::cout << std::left << std::setw(12) << "数据大小" 
                  << std::setw(15) << "编码时间(ms)" 
                  << std::setw(15) << "平均时间(ms)"
                  << std::setw(15) << "吞吐量"
                  << std::setw(15) << "状态" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        bool all_passed = true;

        for (size_t data_size : test_sizes) {
            // 为每个数据大小创建新的编码器实例，避免状态问题
            std::ostringstream errors;
            ErasureCodeClayFactory factory;
            ErasureCodeProfile profile = base_profile;  // 复制配置
            auto encoder = factory.make(profile, errors);

            if (!encoder) {
                std::cerr << "✗ 创建编码器失败 (" << format_bytes(data_size) 
                          << "): " << errors.str() << std::endl;
                std::cout << std::left << std::setw(12) << format_bytes(data_size)
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "✗ 创建失败" << std::endl;
                all_passed = false;
                continue;
            }


            // 准备测试数据
            std::vector<char> original_data(data_size);
            std::iota(original_data.begin(), original_data.end(), 0);

            std::set<int> want_to_encode;
            for (int i = 0; i < k + m; i++) {
                want_to_encode.insert(i);
            }

            bool warmup_failed = false;

            // 预热 - 每次创建新的 bufferlist，并捕获异常
            for (int i = 0; i < warmup_iterations; i++) {
                try {
                    bufferlist input;
                    // 使用原项目的方式创建 bufferlist
                    bufferptr input_ptr(buffer::create_page_aligned(data_size));
                    if (!input_ptr.c_str()) {
                        std::cerr << "✗ 预热阶段 bufferptr 创建失败 (迭代 " << i + 1 << ")" << std::endl;
                        warmup_failed = true;
                        break;
                    }
                    input_ptr.zero();
                    input_ptr.set_length(0);
                    
                    // 检查 append 是否成功
                    try {
                        input_ptr.append(original_data.data(), data_size);
                        if (input_ptr.length() != data_size) {
                            std::cerr << "✗ 预热阶段 append 失败，长度不匹配: " 
                                      << input_ptr.length() << " != " << data_size 
                                      << " (迭代 " << i + 1 << ")" << std::endl;
                            warmup_failed = true;
                            break;
                        }
                    } catch (const std::exception &e) {
                        std::cerr << "✗ 预热阶段 append 异常 (迭代 " << i + 1 << "): " << e.what() << std::endl;
                        warmup_failed = true;
                        break;
                    }
                    
                    input.push_back(input_ptr);
                    
                    // 验证 input 的长度
                    if (input.length() != data_size) {
                        std::cerr << "✗ 预热阶段 bufferlist 长度不匹配: " 
                                  << input.length() << " != " << data_size 
                                  << " (迭代 " << i + 1 << ")" << std::endl;
                        warmup_failed = true;
                        break;
                    }

                    std::map<int, bufferlist> encoded;
                    int encode_result = encoder->encode(want_to_encode, input, &encoded);
                    if (encode_result != 0) {
                        std::cerr << "✗ 预热阶段编码失败，错误码: " << encode_result 
                                  << " (迭代 " << i + 1 << ")" << std::endl;
                        warmup_failed = true;
                        break;
                    }
                    // 清理 encoded，释放内存
                    encoded.clear();
                    input.clear();  // 显式清理
                } catch (const std::exception &e) {
                    std::cerr << "✗ 预热阶段异常 (迭代 " << i + 1 << "): " << e.what() << std::endl;
                    warmup_failed = true;
                    break;
                } catch (...) {
                    std::cerr << "✗ 预热阶段未知异常 (迭代 " << i + 1 << ")" << std::endl;
                    warmup_failed = true;
                    break;
                }
            }

            // 如果预热失败，跳过这个数据大小的测试
            if (warmup_failed) {
                std::cout << std::left << std::setw(12) << format_bytes(data_size)
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "✗ 预热失败" << std::endl;
                all_passed = false;
                continue;
            }

            // 正式测试 - 每次迭代都创建新的 bufferlist
            std::vector<double> encode_times;
            bool encode_success = true;

            // 预先准备输入数据，避免在计时中包含内存分配时间
            bufferlist input;
            bufferptr input_ptr(buffer::create_page_aligned(data_size));
            input_ptr.zero();
            input_ptr.set_length(0);
            input_ptr.append(original_data.data(), data_size);
            input.push_back(input_ptr);


            for (int i = 0; i < num_iterations; i++) {
                try {
                    // 每次迭代创建新的 encoded map，但重用 input
                    std::map<int, bufferlist> encoded;
                    
                    // 只测量编码本身的时间，不包括内存分配
                    auto start = std::chrono::high_resolution_clock::now();
                    int encode_result = encoder->encode(want_to_encode, input, &encoded);
                    auto end = std::chrono::high_resolution_clock::now();

                    if (encode_result != 0) {
                        std::cerr << "✗ 编码失败，错误码: " << encode_result 
                                  << " (迭代 " << i + 1 << "/" << num_iterations << ")" << std::endl;
                        encode_success = false;
                        break;
                    }

                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    double time_ms = duration.count() / 1000.0;  // 转换为毫秒
                    encode_times.push_back(time_ms);

                    // 清理 encoded，释放内存
                    encoded.clear();
                } catch (const std::exception &e) {
                    std::cerr << "✗ 编码异常 (迭代 " << i + 1 << "/" << num_iterations 
                              << "): " << e.what() << std::endl;
                    encode_success = false;
                    break;
                } catch (...) {
                    std::cerr << "✗ 未知编码异常 (迭代 " << i + 1 << "/" << num_iterations << ")" << std::endl;
                    encode_success = false;
                    break;
                }
            }

            if (!encode_success) {
                std::cout << std::left << std::setw(12) << format_bytes(data_size)
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "✗ 失败" << std::endl;
                all_passed = false;
                // 如果小数据失败，可能大数据也会失败，但继续测试
                continue;
            }

            if (encode_times.empty()) {
                std::cout << std::left << std::setw(12) << format_bytes(data_size)
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "✗ 无数据" << std::endl;
                all_passed = false;
                continue;
            }

            // 计算统计信息
            double total_time = std::accumulate(encode_times.begin(), encode_times.end(), 0.0);
            double avg_time = total_time / num_iterations;
            double min_time = *std::min_element(encode_times.begin(), encode_times.end());
            double max_time = *std::max_element(encode_times.begin(), encode_times.end());

            // 计算吞吐量（MB/s）
            // 吞吐量 = 数据大小(MB) / 时间(秒)
            double data_size_mb = static_cast<double>(data_size) / (1024.0 * 1024.0);
            double throughput_mbps = data_size_mb / (avg_time / 1000.0);

            // 输出结果
            std::cout << std::left << std::setw(12) << format_bytes(data_size)
                      << std::setw(15) << std::fixed << std::setprecision(3) 
                      << "[" << min_time << "-" << max_time << "]"
                      << std::setw(15) << std::fixed << std::setprecision(3) << avg_time
                      << std::setw(15) << format_throughput(throughput_mbps)
                      << std::setw(15) << "✓ 成功" << std::endl;

        }

        std::cout << std::string(80, '-') << std::endl;

        if (all_passed) {
            std::cout << "✓ 所有编码吞吐量测试通过！" << std::endl;
            return true;
        } else {
            std::cerr << "✗ 部分编码吞吐量测试失败" << std::endl;
            return false;
        }

    } catch (const std::exception &e) {
        std::cerr << "✗ 异常: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char *argv[]) {
    (void)argc;  // 避免未使用参数警告
    (void)argv;  // 避免未使用参数警告
    std::cout << "========================================" << std::endl;
    std::cout << "    Clay 纠删码功能测试程序" << std::endl;
    std::cout << "========================================" << std::endl;

    int passed = 0;
    int total = 0;

    // 测试配置
    int k = 4;  // 数据块数
    int m = 2;  // 校验块数
    size_t data_size = 4096;  // 测试数据大小

    // 测试 1: 基本编码和解码
    // total++;
    // if (test_basic_encode_decode(k, m, data_size)) {
    //     passed++;
    // }

    // // 测试 2: 数据丢失恢复（丢失 1 个块）
    // total++;
    // if (test_data_loss_recovery(k, m, data_size, 1)) {
    //     passed++;
    // }

    // // 测试 2b: 数据丢失恢复（丢失 2 个块）
    // total++;
    // if (test_data_loss_recovery(k, m, data_size, 2)) {
    //     passed++;
    // }

    // // 测试 3: 不同数据大小
    // total++;
    // if (test_different_sizes(k, m)) {
    //     passed++;
    // }

    // // 测试 4: 编码吞吐量测试 (k=4, m=2)
    // total++;
    // if (test_encode_throughput(4, 2)) {
    //     passed++;
    // }

    // 测试 5: 编码吞吐量测试 (k=2, m=2)
    total++;
    if (test_encode_throughput(2, 2)) {
        passed++;
    }

    // 测试总结
    std::cout << "\n========================================" << std::endl;
    std::cout << "测试总结: " << passed << "/" << total << " 通过" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}

