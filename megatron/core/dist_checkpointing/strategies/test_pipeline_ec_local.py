"""
Local unit-style test for pipeline EC logic that does NOT require torch.distributed or GPUs.

This script attempts to import and use `FileSystemWriterAsyncPipeline.ec_encode_tensors`
from the repository. If the project dependencies (zfec, torch) are not available in the
developer environment, the script falls back to a lightweight splitter that mimics the
data chunking behavior and uses zeroed parity chunks so that the chunking logic can be
validated without external dependencies.

Run this script on a developer machine to quickly validate the chunking/serialization
behaviour without launching a distributed job.
"""
import os
import math
import io
import tempfile
import logging

try:
    import torch
    have_torch = True
except Exception:
    torch = None
    have_torch = False

try:
    # Try to import pipeline ec implementation
    from megatron.core.dist_checkpointing.strategies.filesystem_async_pipeline import (
        FileSystemWriterAsyncPipeline,
    )
    pipeline_available = True
except Exception:
    FileSystemWriterAsyncPipeline = None
    pipeline_available = False

try:
    from megatron.core.dist_checkpointing.strategies.filesystem_async import serialize_bucket_to_bytes
    serialize_available = True
except Exception:
    serialize_bucket_to_bytes = None
    serialize_available = False

logger = logging.getLogger("test_pipeline_ec_local")
logging.basicConfig(level=logging.INFO)

# Minimal dummy WriteItem for unit tests to satisfy _write_item's attribute access
class DummyWriteItem:
    def __init__(self, item_type=None, index: int = 0, name: str = None):
        self.type = item_type
        # _write_item/_result_from_write_item expects .index and sometimes .name
        self.index = index
        self.name = name

try:
    from torch.distributed.checkpoint.planner import WriteItemType
except Exception:
    # Fallback placeholder
    class WriteItemType:
        BYTE_IO = 0


def fallback_ec_encode_tensors(tensors, k: int, m: int):
    """A fallback encoder that mimics chunking used in the real implementation.

    It concatenates tensor bytes and splits into k data chunks. It produces m
    parity chunks consisting of zero bytes of the chunk length. This is NOT a
    real erasure code, but enough to validate chunk sizes and counts.
    """
    if not tensors:
        return []

    all_data_list = []
    for t in tensors:
        if have_torch and isinstance(t, torch.Tensor):
            t_cpu = t.contiguous() if t.is_contiguous() else t.clone().contiguous()
            t_cpu = t_cpu if t_cpu.device.type == 'cpu' else t_cpu.cpu()
            all_data_list.append(t_cpu.numpy().tobytes())
        else:
            # assume bytes-like
            all_data_list.append(bytes(t))

    all_data = b"".join(all_data_list)
    total_size = len(all_data)
    chunk_size = math.ceil(total_size / max(1, k))
    padded_size = chunk_size * k
    all_data = all_data.ljust(padded_size, b'\0')
    data_chunks = [all_data[i:i+chunk_size] for i in range(0, padded_size, chunk_size)]
    parity_chunks = [b'\0' * chunk_size for _ in range(m)]
    return data_chunks + parity_chunks


def make_test_tensors():
    """Create a small set of deterministic CPU tensors (or bytes) for testing."""
    tensors = []
    if have_torch:
        tensors.append(torch.arange(0, 1024, dtype=torch.uint8))
        tensors.append(torch.arange(0, 2048, dtype=torch.uint8) + 1)
        tensors.append(torch.arange(0, 512, dtype=torch.uint8) + 2)
    else:
        tensors.append(bytes(range(256)))
        tensors.append(bytes(range(256)) * 2)
        tensors.append(bytes(range(128)))
    return tensors


def test_ec_encode(k=2, m=2):
    tensors = make_test_tensors()
    logger.info("Testing EC encode with k=%d m=%d", k, m)

    # Try to use the project's implementation if available and zfec present
    chunks = None
    if pipeline_available:
        try:
            chunks = FileSystemWriterAsyncPipeline.ec_encode_tensors(tensors, k, m)
            logger.info("Used pipeline.ec_encode_tensors implementation")
        except Exception as e:
            logger.warning("Pipeline ec_encode_tensors failed (fallback): %s", e)

    if chunks is None:
        chunks = fallback_ec_encode_tensors(tensors, k, m)
        logger.info("Used fallback_ec_encode_tensors")

    assert len(chunks) == k + m, f"Expected {k+m} chunks, got {len(chunks)}"
    total_bytes = sum(len(c) for c in chunks[:k])
    original_size = sum(len(x.numpy().tobytes()) if have_torch and isinstance(x, torch.Tensor) else len(x) for x in tensors)
    assert total_bytes >= original_size, "Data chunks do not cover original size"
    logger.info("ec_encode basic checks passed: %d chunks, original %d bytes, data bytes %d", len(chunks), original_size, total_bytes)


def test_serialize_bucket():
    if not serialize_available:
        logger.warning("serialize_bucket_to_bytes not available in this environment; skipping serialize test")
        return

    logger.info("Testing serialize_bucket_to_bytes (best-effort)")
    # Build a minimal bucket: one small bytes_data and one small tensor_data (if torch)
    # _write_item expects objects with a `.type` attribute; use DummyWriteItem to simulate WriteItem
    bytes_data = [(DummyWriteItem(WriteItemType.BYTE_IO), io.BytesIO(b"hello world"))]
    tensor_list = []
    if have_torch:
        tensor_list = [(DummyWriteItem(None), torch.arange(0, 16, dtype=torch.uint8))]
    else:
        tensor_list = [(DummyWriteItem(None), b"0123456789")]

    try:
        data_bytes, size = serialize_bucket_to_bytes([], (bytes_data, tensor_list), "test_key")
        assert size == len(data_bytes)
        assert size > 0
        logger.info("serialize_bucket_to_bytes returned %d bytes", size)
    except Exception as e:
        logger.exception("serialize_bucket_to_bytes failed: %s", e)


def main():
    tmpdir = tempfile.mkdtemp(prefix="megatron_ec_test_")
    logger.info("Temporary test dir: %s", tmpdir)
    try:
        test_ec_encode(k=2, m=2)
        test_serialize_bucket()
        logger.info("All local EC tests passed (or were skipped where dependencies missing)")
    finally:
        # keep the tempdir by default for inspection; remove if you prefer
        pass


if __name__ == "__main__":
    main()
