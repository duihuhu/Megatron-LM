import ctypes
import mmap
import weakref
from typing import Dict, List, Optional

from logging import getLogger

import torch

_HUGEPAGE_SIZE = 2 * 1024 * 1024

logger = getLogger(__name__)

_success_logged = 0
_fail_logged = 0
_log_limit = int(__import__("os").environ.get("MEGATRON_HUGEPAGE_LOG_LIMIT", "5"))
_cuda_host_register = int(__import__("os").environ.get("MEGATRON_HUGEPAGE_CUDA_REGISTER", "1"))

_CUDA_SUCCESS = 0
_CUDA_ERROR_ALREADY_MAPPED = 208
_REGISTERED_HOST: Dict[int, int] = {}


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


_CUDART = None


def _get_cudart():
    global _CUDART
    if _CUDART is not None:
        return _CUDART
    for lib_name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            _CUDART = ctypes.CDLL(lib_name)
            break
        except OSError:
            continue
    if _CUDART is None:
        return None
    _CUDART.cudaHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    _CUDART.cudaHostRegister.restype = ctypes.c_int
    _CUDART.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    _CUDART.cudaHostUnregister.restype = ctypes.c_int
    return _CUDART


def _try_cuda_host_unregister_ptr(ptr: int) -> bool:
    """Undo cudaHostRegister for a tracked host pointer."""
    if ptr not in _REGISTERED_HOST:
        return False
    if not torch.cuda.is_available():
        _REGISTERED_HOST.pop(ptr, None)
        return True
    cudart = _get_cudart()
    if cudart is None:
        _REGISTERED_HOST.pop(ptr, None)
        return False
    rc = cudart.cudaHostUnregister(ctypes.c_void_p(ptr))
    if rc not in (_CUDA_SUCCESS, _CUDA_ERROR_ALREADY_MAPPED):
        _maybe_log_fail(
            f"[HUGEPAGE_ALLOC] cudaHostUnregister failed rc={rc} ptr=0x{ptr:x}"
        )
        return False
    _REGISTERED_HOST.pop(ptr, None)
    return True


def release_hugepage_host_registration(buffer: torch.Tensor) -> bool:
    """Release CUDA host registration for a hugetlb-backed allocation."""
    if not torch.is_tensor(buffer):
        return False
    return _try_cuda_host_unregister_ptr(int(buffer.data_ptr()))


def release_all_hugepage_host_registrations() -> int:
    """Release every tracked cudaHostRegister region."""
    released = 0
    for ptr in list(_REGISTERED_HOST.keys()):
        if _try_cuda_host_unregister_ptr(ptr):
            released += 1
    return released


def _attach_host_unregister_finalizer(buffer: torch.Tensor, ptr: int) -> None:
    weakref.finalize(buffer, _try_cuda_host_unregister_ptr, ptr)


def _try_cuda_host_register(buffer: torch.Tensor) -> bool:
    """Register an existing host allocation with CUDA for async D2H."""
    if not _cuda_host_register or not torch.cuda.is_available():
        return False
    if buffer.is_pinned():
        return True
    cudart = _get_cudart()
    if cudart is None:
        return False
    ptr = int(buffer.data_ptr())
    size = int(buffer.numel() * buffer.element_size())
    if size <= 0:
        return True
    if ptr in _REGISTERED_HOST:
        return True
    rc = cudart.cudaHostRegister(
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(size),
        ctypes.c_uint(0),
    )
    if rc == _CUDA_ERROR_ALREADY_MAPPED:
        _REGISTERED_HOST[ptr] = size
        return True
    if rc != _CUDA_SUCCESS:
        _maybe_log_fail(
            f"[HUGEPAGE_ALLOC] cudaHostRegister failed rc={rc} size={size}"
        )
        return False
    _REGISTERED_HOST[ptr] = size
    _attach_host_unregister_finalizer(buffer, ptr)
    return True


def _try_allocate_hugetlb(
    size_bytes: int,
    aligned: int,
    touch_pages: bool,
) -> Optional[torch.Tensor]:
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    map_hugetlb = getattr(mmap, "MAP_HUGETLB", 0)
    if map_hugetlb == 0:
        map_hugetlb = 0x40000
    if map_hugetlb == 0:
        return None
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
        return buffer
    except (BufferError, OSError):
        return None


def allocate_hugepage_tensor(
    size_bytes: int,
    *,
    fallback_pin_memory: bool = False,
    touch_pages: bool = True,
) -> torch.Tensor:
    """Allocate a uint8 CPU tensor for RDMA / D2H checkpoint buffers."""
    aligned = ((size_bytes + _HUGEPAGE_SIZE - 1) // _HUGEPAGE_SIZE) * _HUGEPAGE_SIZE

    # Prefer MAP_HUGETLB for multi-GB buffers. torch pin_memory on every rank at
    # once can exhaust the CUDA pinned pool and surface as invalid resource handle
    # at later NCCL collectives.
    buffer = _try_allocate_hugetlb(size_bytes, aligned, touch_pages)
    if buffer is not None:
        registered = False
        if fallback_pin_memory:
            registered = _try_cuda_host_register(buffer)
        _maybe_log_success(
            f"[HUGEPAGE_ALLOC] Using MAP_HUGETLB success: req={size_bytes} "
            f"aligned={aligned} hugepage_size={_HUGEPAGE_SIZE} "
            f"pin={registered}"
        )
        return buffer

    _maybe_log_fail(
        f"[HUGEPAGE_ALLOC] MAP_HUGETLB failed, falling back: req={size_bytes} "
        f"aligned={aligned} hugepage_size={_HUGEPAGE_SIZE} "
        f"pin={fallback_pin_memory}"
    )

    if fallback_pin_memory:
        buffer = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
        if touch_pages:
            _touch_tensor_pages(buffer, stride=4096)
        _maybe_log_success(
            f"[HUGEPAGE_ALLOC] Using torch pinned memory: req={size_bytes} "
            f"aligned={aligned} hugepage_size={_HUGEPAGE_SIZE} pin=True"
        )
        return buffer

    buffer = torch.empty(size_bytes, dtype=torch.uint8)
    if touch_pages:
        _touch_tensor_pages(buffer, stride=4096)
    _maybe_log_fail(
        f"[HUGEPAGE_ALLOC] allocate_hugepage_tensor fallback to torch.empty: req={size_bytes} "
        f"aligned={aligned} hugepage_size={_HUGEPAGE_SIZE} pin=False"
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
