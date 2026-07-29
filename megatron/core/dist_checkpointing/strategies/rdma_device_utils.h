// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

/*
 * Shared RDMA device selection utility.
 *
 * Selects an RDMA device from the existing per-local-rank NIC environment
 * variables, then falls back to IP-based GID-table matching.  A configured
 * NIC can be either an RDMA device name (for example, mlx5_0) or a Linux
 * network interface whose sysfs device exposes an InfiniBand device.
 */

#ifndef RDMA_DEVICE_UTILS_H
#define RDMA_DEVICE_UTILS_H

#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <infiniband/verbs.h>
#include <arpa/inet.h>

static inline int rdma_rank_from_env(
    std::initializer_list<const char*> variables,
    int fallback)
{
    for (const char* variable : variables) {
        const char* value = std::getenv(variable);
        if (value && *value) {
            char* end = nullptr;
            long rank = std::strtol(value, &end, 10);
            if (end != value && *end == '\0' && rank >= 0) {
                return static_cast<int>(rank);
            }
        }
    }
    return fallback;
}

static inline int rdma_local_rank()
{
    return rdma_rank_from_env(
        {"LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"}, 0);
}

static inline int rdma_global_rank()
{
    return rdma_rank_from_env(
        {"RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"}, -1);
}

static inline std::string rdma_trim(const std::string& value)
{
    const std::string whitespace = " \t\r\n";
    const std::string::size_type begin = value.find_first_not_of(whitespace);
    if (begin == std::string::npos) return "";
    const std::string::size_type end = value.find_last_not_of(whitespace);
    return value.substr(begin, end - begin + 1);
}

static inline std::string rdma_device_binding_from_env(
    std::initializer_list<const char*> prefixes,
    std::string* source_env)
{
    const int global_rank = rdma_global_rank();
    if (global_rank >= 0) {
        for (const char* prefix : prefixes) {
            if (!prefix || !*prefix) continue;
            const std::string env_name =
                std::string(prefix) + "_RANK_IP_" + std::to_string(global_rank);
            const char* value = std::getenv(env_name.c_str());
            if (value && *value) {
                return "";
            }
        }
    }

    const int local_rank = rdma_local_rank();
    for (const char* prefix : prefixes) {
        if (!prefix || !*prefix) continue;
        const std::string env_name =
            std::string(prefix) + "_LOCAL_RANK_NIC_" + std::to_string(local_rank);
        const char* value = std::getenv(env_name.c_str());
        if (value && *value) {
            if (source_env) *source_env = env_name;
            return rdma_trim(value);
        }
    }

    for (const char* prefix : prefixes) {
        if (!prefix || !*prefix) continue;
        const std::string list_env = std::string(prefix) + "_NIC_LIST";
        const std::string count_env = std::string(prefix) + "_RANKS_PER_NIC";
        const char* list_value = std::getenv(list_env.c_str());
        const char* count_value = std::getenv(count_env.c_str());
        if (!list_value || !*list_value || !count_value || !*count_value) continue;

        char* end = nullptr;
        long ranks_per_nic = std::strtol(count_value, &end, 10);
        if (end == count_value || *end != '\0' || ranks_per_nic <= 0) continue;

        std::vector<std::string> entries;
        std::stringstream stream(list_value);
        std::string entry;
        while (std::getline(stream, entry, ',')) {
            entry = rdma_trim(entry);
            if (!entry.empty()) entries.push_back(entry);
        }
        const size_t index = static_cast<size_t>(local_rank / ranks_per_nic);
        if (index < entries.size()) {
            if (source_env) *source_env = list_env;
            return entries[index];
        }
    }

    for (const char* prefix : prefixes) {
        if (!prefix || !*prefix) continue;
        const std::string env_name = std::string(prefix) + "_INTERFACE";
        const char* value = std::getenv(env_name.c_str());
        if (value && *value) {
            if (source_env) *source_env = env_name;
            return rdma_trim(value);
        }
    }
    return "";
}

static inline struct ibv_device* rdma_device_by_name(
    const std::string& device_name,
    struct ibv_device** device_list,
    int num_devices)
{
    for (int i = 0; i < num_devices; ++i) {
        if (device_name == ibv_get_device_name(device_list[i])) {
            return device_list[i];
        }
    }
    return nullptr;
}

static inline struct ibv_device* rdma_device_by_netdev(
    const std::string& interface_name,
    struct ibv_device** device_list,
    int num_devices)
{
    const std::string path =
        "/sys/class/net/" + interface_name + "/device/infiniband";
    DIR* directory = opendir(path.c_str());
    if (!directory) return nullptr;

    struct ibv_device* selected = nullptr;
    while (struct dirent* entry = readdir(directory)) {
        if (entry->d_name[0] == '.') continue;
        selected = rdma_device_by_name(entry->d_name, device_list, num_devices);
        if (selected) break;
    }
    closedir(directory);
    return selected;
}

static inline struct ibv_device* rdma_device_by_binding(
    const std::string& binding,
    struct ibv_device** device_list,
    int num_devices)
{
    std::string target = binding;
    const std::string::size_type separator = target.find(':');
    if (separator != std::string::npos) {
        target = rdma_trim(target.substr(0, separator));
    }
    if (target.empty()) return nullptr;

    struct ibv_device* selected =
        rdma_device_by_name(target, device_list, num_devices);
    if (selected) return selected;
    return rdma_device_by_netdev(target, device_list, num_devices);
}

static inline struct ibv_device* rdma_device_by_ip(
    const std::string& ip_str,
    struct ibv_device** device_list,
    int num_devices)
{
    /*
     * Compare the final four bytes of every GID with the IPv4 address.  This
     * handles RoCE IPv4-mapped GIDs while leaving native InfiniBand selection
     * to the explicit device/netdev path above.
     */
    struct in_addr target;
    if (inet_pton(AF_INET, ip_str.c_str(), &target) != 1) return nullptr;

    for (int i = 0; i < num_devices; i++) {
        struct ibv_context* ctx = ibv_open_device(device_list[i]);
        if (!ctx) continue;

        struct ibv_port_attr port_attr;
        if (ibv_query_port(ctx, 1, &port_attr) != 0) {
            ibv_close_device(ctx);
            continue;
        }

        bool matched = false;
        for (int g = 0; g < port_attr.gid_tbl_len; g++) {
            union ibv_gid gid;
            if (ibv_query_gid(ctx, 1, g, &gid) == 0 &&
                memcmp(&gid.raw[12], &target.s_addr, 4) == 0) {
                matched = true;
                break;
            }
        }
        ibv_close_device(ctx);

        if (matched) return device_list[i];
    }
    return nullptr;
}

/*
 * Return the explicitly configured RDMA device, the device whose GID table
 * contains *ip_str*, or the first device as a backward-compatible fallback.
 * The caller owns *device_list* and must call ibv_free_device_list().
 */
static inline struct ibv_device*
find_rdma_device_by_ip(const std::string&     ip_str,
                       struct ibv_device**    device_list,
                       int                    num_devices,
                       std::initializer_list<const char*> prefixes = {})
{
    if (!device_list || num_devices <= 0) return nullptr;

    std::string source_env;
    const std::string binding =
        rdma_device_binding_from_env(prefixes, &source_env);
    if (!binding.empty()) {
        struct ibv_device* selected =
            rdma_device_by_binding(binding, device_list, num_devices);
        if (selected) {
            std::cerr << "[RDMA] selected " << ibv_get_device_name(selected)
                      << " from " << source_env << "=" << binding << std::endl;
            return selected;
        }
        std::cerr << "[RDMA] cannot map " << source_env << "=" << binding
                  << " to an RDMA device; trying IP-based selection" << std::endl;
    }

    struct ibv_device* selected =
        rdma_device_by_ip(ip_str, device_list, num_devices);
    if (selected) return selected;

    std::cerr << "[RDMA] IP " << ip_str
              << " not found in any GID table, falling back to first device "
              << ibv_get_device_name(device_list[0]) << std::endl;
    return device_list[0];
}

#endif // RDMA_DEVICE_UTILS_H
