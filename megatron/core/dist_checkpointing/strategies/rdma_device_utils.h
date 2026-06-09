// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

/*
 * Shared RDMA device selection utility.
 *
 * Replaces hardcoded ``device_list[0]`` with IP-based GID-table matching
 * so that multi-NIC nodes route RDMA traffic through the same NIC that
 * ``resolve_ip()`` selected in Python.
 */

#ifndef RDMA_DEVICE_UTILS_H
#define RDMA_DEVICE_UTILS_H

#include <cstring>
#include <string>
#include <iostream>

#include <infiniband/verbs.h>
#include <arpa/inet.h>

/*
 * Return the RDMA device whose GID table contains *ip_str*, or fall back
 * to the first device.  *device_list* and *num_devices* come directly from
 * ``ibv_get_device_list()``.  The caller still owns the list and must call
 * ``ibv_free_device_list()`` afterwards.
 *
 * Matching:  for each device we open a temporary context, query the GID
 * table of port 1, and compare the last 4 bytes of each GID entry against
 * the IPv4 address.  This catches both RoCE v2 ``::ffff:x.x.x.x`` and any
 * other GID format that embeds the IPv4 address in the tail.
 */
static inline struct ibv_device*
find_rdma_device_by_ip(const std::string&     ip_str,
                       struct ibv_device**    device_list,
                       int                    num_devices)
{
    struct in_addr target;
    if (inet_pton(AF_INET, ip_str.c_str(), &target) != 1) {
        std::cerr << "[RDMA] cannot parse IP '" << ip_str
                  << "', falling back to first device" << std::endl;
        return device_list[0];
    }

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
            if (ibv_query_gid(ctx, 1, g, &gid) == 0) {
                // RoCE v2 IPv4 GID: last 4 bytes == IPv4 addr (big-endian)
                if (memcmp(&gid.raw[12], &target.s_addr, 4) == 0) {
                    matched = true;
                    break;
                }
            }
        }
        ibv_close_device(ctx);

        if (matched) {
            std::cout << "[RDMA] IP " << ip_str << " → device "
                      << ibv_get_device_name(device_list[i])
                      << " (index " << i << ")" << std::endl;
            return device_list[i];
        }
    }

    std::cerr << "[RDMA] IP " << ip_str
              << " not found in any GID table, falling back to first device "
              << ibv_get_device_name(device_list[0]) << std::endl;
    return device_list[0];
}

#endif // RDMA_DEVICE_UTILS_H
