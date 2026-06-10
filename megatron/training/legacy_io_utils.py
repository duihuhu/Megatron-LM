# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Zero-copy raw file I/O for torch legacy checkpoint save/load.

Writes tensor data directly via memoryview + f.write, bypassing pickle
serialisation overhead.  Uses a short magic-number header so that old
``torch.save``-format ``.pt`` files remain readable (auto-detected on load).
"""

import os
import pickle
import struct
from logging import getLogger
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = getLogger(__name__)

# ---- file-format constants --------------------------------------------------

_HEADER_MAGIC_OFFSET = 0
_HEADER_MAGIC_LEN = 4

# Magic strings for each scheme's main checkpoint file
MAGIC_ECCHECK = b"ECCK"
MAGIC_ECNAIVE = b"ECNV"
MAGIC_ECLATIN = b"ECLT"
MAGIC_GEMINI = b"GEMR"

# Magic strings for block / replica files
MAGIC_BLOCK = b"ECBK"  # generic block: ECCHECK / ECNAIVE / ECLATIN
MAGIC_GEMINI_REPLICA = b"GMRP"

# ---- write helpers ----------------------------------------------------------

def write_raw_checkpoint(
    path: str,
    magic: bytes,
    non_tensor_data: Dict[str, Any],
    tensor_infos: List[Any],
    tensor_buffer: torch.Tensor,
    total_tensor_bytes: int,
    **extra_meta,
) -> None:
    """Write a legacy main checkpoint file using raw ``f.write``.

    File layout::

        [magic:4][meta1_len:8][meta2_len:8][data_len:8]
        [pickle(non_tensor_data)]
        [pickle(tensor_infos)]
        [raw tensor bytes (total_tensor_bytes)]
    """
    meta1 = pickle.dumps(non_tensor_data)
    meta2 = pickle.dumps(tensor_infos)
    extra = pickle.dumps(extra_meta) if extra_meta else b""

    # Ensure the tensor view is contiguous and on CPU
    buf = tensor_buffer[:total_tensor_bytes]
    if not buf.is_contiguous():
        buf = buf.contiguous()
    if buf.device.type != "cpu":
        buf = buf.to("cpu")
    # Write directly from the underlying storage – no clone needed
    mv = memoryview(buf.numpy())

    with open(path, "wb") as f:
        f.write(struct.pack("<4sQQQ", magic, len(meta1), len(meta2), total_tensor_bytes))
        f.write(struct.pack("<Q", len(extra)))
        f.write(meta1)
        f.write(meta2)
        f.write(extra)
        f.write(mv)


def write_raw_block(path: str, magic: bytes, tensor: torch.Tensor, size: int) -> None:
    """Write a pure-data block file.

    File layout::

        [magic:4][data_len:8]
        [raw bytes (size)]
    """
    buf = tensor[:size]
    if not buf.is_contiguous():
        buf = buf.contiguous()
    if buf.device.type != "cpu":
        buf = buf.to("cpu")
    mv = memoryview(buf.numpy())

    with open(path, "wb") as f:
        f.write(struct.pack("<4sQ", magic, size))
        f.write(mv)


def write_raw_simple(path: str, magic: bytes, tensor: torch.Tensor) -> None:
    """Alias for ``write_raw_block`` using the tensor's full size."""
    write_raw_block(path, magic, tensor, tensor.numel())


# ---- parallel write helpers (pre-serialized data, for ThreadPoolExecutor) ----

def write_main_prepared(
    path: str, magic: bytes,
    meta1: bytes, meta2: bytes, extra: bytes,
    mv: memoryview, data_len: int,
) -> None:
    """Write a main file from pre-serialized metadata + prepared memoryview."""
    with open(path, "wb") as f:
        f.write(struct.pack("<4sQQQ", magic, len(meta1), len(meta2), data_len))
        f.write(struct.pack("<Q", len(extra)))
        f.write(meta1)
        f.write(meta2)
        f.write(extra)
        f.write(mv)


def write_block_prepared(path: str, magic: bytes, mv: memoryview, size: int) -> None:
    """Write a block file from a pre-prepared memoryview."""
    with open(path, "wb") as f:
        f.write(struct.pack("<4sQ", magic, size))
        f.write(mv)


# ---- read helpers -----------------------------------------------------------

def pin_uint8_tensor_if_available(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a CPU uint8 tensor into pinned memory when CUDA is available."""
    if (
        not torch.is_tensor(tensor)
        or tensor.device.type != "cpu"
        or tensor.numel() == 0
        or not torch.cuda.is_available()
        or tensor.is_pinned()
    ):
        return tensor
    try:
        pinned = torch.empty(tensor.numel(), dtype=torch.uint8, pin_memory=True)
        pinned.copy_(tensor.contiguous().view(torch.uint8).reshape(-1))
        return pinned
    except Exception as exc:
        logger.warning(
            "Unable to pin raw checkpoint tensor buffer; using pageable CPU memory: %s",
            exc,
        )
        return tensor


def pin_payload_tensor_buffer_if_available(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pin payload['tensor_buffer'] in-place when present."""
    tensor_buffer = payload.get("tensor_buffer")
    if torch.is_tensor(tensor_buffer):
        payload["tensor_buffer"] = pin_uint8_tensor_if_available(
            tensor_buffer.detach().contiguous().reshape(-1).view(torch.uint8)
        )
    return payload

def _peek_magic(path: str) -> bytes:
    """Return the first 4 bytes of *path* without consuming the file."""
    with open(path, "rb") as f:
        return f.read(_HEADER_MAGIC_LEN)


def _read_header(f: BinaryIO) -> Tuple[int, int, int]:
    """Read the 3-size header after the magic bytes.  File position must be at
    offset ``_HEADER_MAGIC_LEN``."""
    meta1_len, meta2_len, data_len = struct.unpack("<QQQ", f.read(24))
    return meta1_len, meta2_len, data_len


def read_raw_checkpoint(
    path: str,
    expected_magic: bytes,
    pin_tensor_buffer: bool = False,
) -> Dict[str, Any]:
    """Read a legacy main checkpoint file written by ``write_raw_checkpoint``.

    Returns a dict with keys ``non_tensor_data``, ``tensor_infos``,
    ``tensor_buffer``, and any extra keyword metadata stored at save time.
    """
    with open(path, "rb") as f:
        magic = f.read(_HEADER_MAGIC_LEN)
        if magic != expected_magic:
            raise ValueError(
                f"Unexpected magic {magic!r} (expected {expected_magic!r}) in {path}"
            )
        meta1_len, meta2_len, data_len = _read_header(f)
        extra_len = struct.unpack("<Q", f.read(8))[0]

        non_tensor_data = pickle.loads(f.read(meta1_len))
        tensor_infos = pickle.loads(f.read(meta2_len))
        extra = pickle.loads(f.read(extra_len)) if extra_len else {}

        # Read directly into a torch tensor (zero-copy from read buffer)
        raw = f.read(data_len)
        tensor = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8))
        if pin_tensor_buffer:
            tensor = pin_uint8_tensor_if_available(tensor)

    result: Dict[str, Any] = {
        "non_tensor_data": non_tensor_data,
        "tensor_infos": tensor_infos,
        "tensor_buffer": tensor,
    }
    result.update(extra)
    return result


def read_raw_checkpoint_metadata(
    path: str,
    expected_magic: bytes,
) -> Dict[str, Any]:
    """Read only metadata from a raw legacy main checkpoint file."""
    with open(path, "rb") as f:
        magic = f.read(_HEADER_MAGIC_LEN)
        if magic != expected_magic:
            raise ValueError(
                f"Unexpected magic {magic!r} (expected {expected_magic!r}) in {path}"
            )
        meta1_len, meta2_len, _data_len = _read_header(f)
        extra_len = struct.unpack("<Q", f.read(8))[0]

        non_tensor_data = pickle.loads(f.read(meta1_len))
        tensor_infos = pickle.loads(f.read(meta2_len))
        extra = pickle.loads(f.read(extra_len)) if extra_len else {}

    result: Dict[str, Any] = {
        "non_tensor_data": non_tensor_data,
        "tensor_infos": tensor_infos,
        "tensor_buffer": None,
    }
    result.update(extra)
    return result


def read_raw_block(
    path: str,
    expected_magic: bytes,
    pin_tensor: bool = False,
) -> torch.Tensor:
    """Read a pure-data block file written by ``write_raw_block``."""
    with open(path, "rb") as f:
        magic = f.read(_HEADER_MAGIC_LEN)
        if magic != expected_magic:
            raise ValueError(
                f"Unexpected magic {magic!r} (expected {expected_magic!r}) in {path}"
            )
        size = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(size)
    tensor = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8))
    if pin_tensor:
        tensor = pin_uint8_tensor_if_available(tensor)
    return tensor


def is_raw_format(path: str, expected_magic: bytes) -> bool:
    """True if *path* exists and starts with *expected_magic*."""
    if not os.path.isfile(path):
        return False
    try:
        return _peek_magic(path) == expected_magic
    except OSError:
        return False


def smart_load_checkpoint(
    path: str,
    expected_magic: bytes,
    pin_tensor_buffer: bool = False,
) -> Dict[str, Any]:
    """Load a main checkpoint file — raw format preferred, torch.load fallback."""
    if is_raw_format(path, expected_magic):
        return read_raw_checkpoint(
            path, expected_magic, pin_tensor_buffer=pin_tensor_buffer,
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if pin_tensor_buffer:
        pin_payload_tensor_buffer_if_available(payload)
    return payload


def smart_load_block(
    path: str,
    expected_magic: bytes,
    pin_tensor: bool = False,
) -> torch.Tensor:
    """Load a block file — raw format preferred, torch.load fallback."""
    if is_raw_format(path, expected_magic):
        return read_raw_block(path, expected_magic, pin_tensor=pin_tensor)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload["tensor"].contiguous().view(torch.uint8).reshape(-1)
    if pin_tensor:
        tensor = pin_uint8_tensor_if_available(tensor)
    return tensor
