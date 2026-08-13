# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Shared network utilities for IP resolution with multi-NIC support.

Phase 1: per-rank explicit IP via ``{PREFIX}_RANK_IP_{rank}``.
Phase 2: per-local-rank NIC binding via ``{PREFIX}_LOCAL_RANK_NIC_{local_rank}``
         and batched ``{PREFIX}_NIC_LIST`` + ``{PREFIX}_RANKS_PER_NIC``.
"""

import os
import socket
import struct
from logging import getLogger
from typing import List, Optional

logger = getLogger(__name__)


def _get_local_rank() -> int:
    """Auto-detect local rank from common launcher environment variables."""
    for var in ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 0


def _try_get_ip_from_interface(interface_name: str) -> Optional[str]:
    """Try to resolve an IPv4 address from a named Linux network interface."""
    if os.path.isdir(f'/sys/class/infiniband/{interface_name}'):
        logger.debug(
            "Interface value %s is an RDMA device name; skipping IPv4 lookup",
            interface_name,
        )
        return None

    try:
        import fcntl

        ifname = interface_name.encode('utf-8')
        if len(ifname) >= 16:
            logger.warning("Interface name %s is too long for IPv4 lookup", interface_name)
            return None

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            ifreq = struct.pack('256s', ifname)
            result = fcntl.ioctl(sock.fileno(), 0x8915, ifreq)  # SIOCGIFADDR
        ip = socket.inet_ntoa(result[20:24])
        logger.debug("Resolved IP %s from interface %s", ip, interface_name)
        return ip
    except OSError as e:
        logger.warning("Interface %s has no usable IPv4 address: %s", interface_name, e)
    except Exception as e:
        logger.warning("Failed to get IP from interface %s: %s", interface_name, e)
    return None


def _try_auto_detect_ip() -> Optional[str]:
    """Auto-detect IP via UDP socket connect to a public address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        logger.debug("Auto-detected IP address: %s", ip)
        return ip
    except Exception as e:
        logger.warning("Failed to auto-detect IP: %s", e)
    return None


def resolve_ip(
    prefix: str,
    rank: int = None,
    local_rank: int = None,
    fallback_prefixes: List[str] = None,
) -> str:
    """Resolve the IP address for a given rank with multi-NIC affinity support.

    Priority chain (highest to lowest):

    **Phase 1 — per-rank explicit IP** (fine-grained, any topology)::

        {PREFIX}_RANK_IP_0=10.0.0.1    # rank 0 → NIC 10.0.0.1
        {PREFIX}_RANK_IP_1=10.0.0.1    # rank 1 → same NIC

    **Phase 2 — per-local-rank NIC binding** (multi-node friendly,
    identical config on every node)::

        # All nodes share the same config — only depends on local_rank
        {PREFIX}_LOCAL_RANK_NIC_0=mlx5_0
        {PREFIX}_LOCAL_RANK_NIC_1=mlx5_0
        {PREFIX}_LOCAL_RANK_NIC_2=mlx5_1
        {PREFIX}_LOCAL_RANK_NIC_3=mlx5_1

    **Phase 2 — batched NIC list** (simpler, rule-based grouping)::

        {PREFIX}_NIC_LIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3
        {PREFIX}_RANKS_PER_NIC=2
        # → local_rank 0,1 use mlx5_0; 2,3 use mlx5_1; etc.

    **Legacy — global / single-NIC**::

        {PREFIX}_BASE_IP=10.0.0.1       # all ranks same NIC
        {PREFIX}_INTERFACE=ib0          # all ranks use ib0
        MASTER_ADDR                      # final fallback

    Args:
        prefix: Environment variable prefix (e.g. ``"FRCHECK"``, ``"GEMINI"``).
        rank: Current global rank. If None, auto-detected via
            ``torch.distributed.get_rank()``.
        local_rank: Local rank within the node. If None, auto-detected via
            ``LOCAL_RANK`` / ``OMPI_COMM_WORLD_LOCAL_RANK`` / ``SLURM_LOCALID``.
        fallback_prefixes: Additional prefixes to try for ``BASE_IP`` and
            ``INTERFACE`` env vars (e.g. a legacy configuration prefix).

    Returns:
        Resolved IP address string.
    """
    # Resolve rank
    if rank is None:
        try:
            import torch
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
        except Exception:
            pass

    # Resolve local_rank
    if local_rank is None:
        local_rank = _get_local_rank()

    prefixes = [prefix] + (fallback_prefixes or [])

    # ==== Phase 1: per-rank explicit IP ====
    if rank is not None:
        for p in prefixes:
            rank_ip = os.environ.get(f'{p}_RANK_IP_{rank}')
            if rank_ip:
                logger.debug("[Rank %d] Using IP from %s_RANK_IP_%d: %s", rank, p, rank, rank_ip)
                return rank_ip

    # ==== Phase 2: per-local-rank NIC name ====
    if local_rank is not None:
        for p in prefixes:
            nic_name = os.environ.get(f'{p}_LOCAL_RANK_NIC_{local_rank}')
            if nic_name:
                ip = _try_get_ip_from_interface(nic_name)
                if ip:
                    logger.debug(
                        "[Rank %d / local_rank %d] Using IP %s from %s_LOCAL_RANK_NIC_%d=%s",
                        rank, local_rank, ip, p, local_rank, nic_name,
                    )
                    return ip

    # ==== Phase 2: batched NIC list with ranks-per-nic ====
    for p in prefixes:
        nic_list_str = os.environ.get(f'{p}_NIC_LIST')
        ranks_per_nic_str = os.environ.get(f'{p}_RANKS_PER_NIC')
        if nic_list_str and ranks_per_nic_str:
            try:
                nics = [s.strip() for s in nic_list_str.split(',') if s.strip()]
                ranks_per_nic = int(ranks_per_nic_str)
                nic_idx = local_rank // ranks_per_nic
                if 0 <= nic_idx < len(nics):
                    entry = nics[nic_idx].strip()
                    # Each entry can be "interface_name" or "interface_name:ip"
                    if ':' in entry:
                        ip = entry.split(':', 1)[1].strip()
                        logger.debug(
                            "[Rank %d / local_rank %d] Using IP %s "
                            "from %s_NIC_LIST (nic_idx=%d)",
                            rank, local_rank, ip, p, nic_idx,
                        )
                    else:
                        ip = _try_get_ip_from_interface(entry)
                        if ip:
                            logger.debug(
                                "[Rank %d / local_rank %d] Using IP %s "
                                "from %s_NIC_LIST interface %s (nic_idx=%d)",
                                rank, local_rank, ip, p, entry, nic_idx,
                            )
                    if ip:
                        return ip
                else:
                    logger.warning(
                        "local_rank %d → nic_idx %d out of range for %s_NIC_LIST (len=%d)",
                        local_rank, nic_idx, p, len(nics),
                    )
            except Exception as e:
                logger.warning("Failed to parse %s_NIC_LIST / %s_RANKS_PER_NIC: %s", p, p, e)

    # ==== Legacy: global BASE_IP ====
    for p in prefixes:
        base_ip = os.environ.get(f'{p}_BASE_IP')
        if base_ip:
            logger.debug("Using IP from %s_BASE_IP: %s", p, base_ip)
            return base_ip

    # ==== Legacy: single INTERFACE name ====
    for p in prefixes:
        iface = os.environ.get(f'{p}_INTERFACE')
        if iface:
            ip = _try_get_ip_from_interface(iface)
            if ip:
                return ip

    # ==== Auto-detect ====
    ip = _try_auto_detect_ip()
    if ip:
        return ip

    # ==== MASTER_ADDR fallback ====
    fallback = os.environ.get('MASTER_ADDR', '127.0.0.1')
    logger.warning("All IP discovery methods failed, using MASTER_ADDR fallback: %s", fallback)
    return fallback
