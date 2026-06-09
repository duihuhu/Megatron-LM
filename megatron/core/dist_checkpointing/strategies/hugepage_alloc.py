import mmap
from typing import List

from logging import getLogger

import torch

_HUGEPAGE_SIZE = 2 * 1024 * 1024

logger = getLogger(__name__)

_success_logged = 0
_fail_logged = 0
_log_limit = int(__import__("os").environ.get("MEGATRON_HUGEPAGE_LOG_LIMIT", "5"))


def _maybe_log_success(msg: str) -> None:
    global _success_logged
    if _success_logged < _log_limit:
        logger.info(msg)
        _success_logged += 1


def _maybe_log_fail(msg: str) -> None:
    global _fail_logged
    if _fail_logged < _log_limit:
        logger.warning(msg)
        _fail_logged += 1


def _touch_tensor_pages(buffer: torch.Tensor, stride: int = _HUGEPAGE_SIZE) -> None:
    """Touch sparse offsets so pages are faulted before RDMA registration."""
    if buffer.numel() == 0:
        return
    view = buffer.view(-1)
    for off in range(0, view.numel(), stride):
        view[off] = 0
    view[-1] = 0


def allocate_hugepage_tensor(
    size_bytes: int,
    *,
    fallback_pin_memory: bool = False,
    touch_pages: bool = True,
) -> torch.Tensor:
    """Allocate uint8 CPU tensor on hugepages, fallback to torch.empty on failure."""
    aligned = ((size_bytes + _HUGEPAGE_SIZE - 1) // _HUGEPAGE_SIZE) * _HUGEPAGE_SIZE
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    # Python 的 mmap 模块在某些环境下不会暴露 MAP_HUGETLB 常量。
    # 在常见 x86_64 Linux 上该值通常为 0x40000。
    map_hugetlb = getattr(mmap, "MAP_HUGETLB", 0)
    if map_hugetlb == 0:
        map_hugetlb = 0x40000

    if map_hugetlb != 0:
        try:
            mm = mmap.mmap(
                -1,
                aligned,
                flags=flags | map_hugetlb,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            buffer = torch.frombuffer(mm, dtype=torch.uint8, count=aligned)[:size_bytes]
            if touch_pages:
                _touch_tensor_pages(buffer)
            _maybe_log_success(
                f"[HUGEPAGE_ALLOC] Using MAP_HUGETLB success: req={size_bytes} "
                f"aligned={aligned} hugepage_size={_HUGEPAGE_SIZE} pin={fallback_pin_memory}"
            )
            return buffer
        except (BufferError, OSError):
            _maybe_log_fail(
                f"[HUGEPAGE_ALLOC] MAP_HUGETLB failed, falling back: req={size_bytes} "
                f"aligned={aligned} hugepage_size={_HUGEPAGE_SIZE} pin={fallback_pin_memory}"
            )
            pass

    buffer = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=fallback_pin_memory)
    if touch_pages:
        _touch_tensor_pages(buffer, stride=4096)
    _maybe_log_fail(
        f"[HUGEPAGE_ALLOC] allocate_hugepage_tensor fallback to torch.empty: req={size_bytes} "
        f"aligned={aligned} hugepage_size={_HUGEPAGE_SIZE} pin={fallback_pin_memory}"
    )
    return buffer


def allocate_hugepage_slices(
    slice_size_bytes: int,
    count: int,
    *,
    fallback_pin_memory: bool = False,
    touch_pages: bool = True,
) -> List[torch.Tensor]:
    """Allocate one large hugepage-backed tensor and return non-overlapping slices."""
    total = slice_size_bytes * count
    base = allocate_hugepage_tensor(
        total,
        fallback_pin_memory=fallback_pin_memory,
        touch_pages=False,
    )
    slices = [
        base[i * slice_size_bytes : (i + 1) * slice_size_bytes]
        for i in range(count)
    ]
    if touch_pages:
        for chunk in slices:
            _touch_tensor_pages(chunk)
    return slices
