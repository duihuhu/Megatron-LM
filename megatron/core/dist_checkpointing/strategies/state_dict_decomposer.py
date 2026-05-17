# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""
StateDict decomposer for serialization-free checkpoint encoding.

This module provides utilities to decompose state_dict into three components:
1. Non-tensor key-value pairs (dict): training metadata like iteration count
2. Tensor keys (list): keys for all tensors in the state_dict
3. Tensor data (list): actual tensor data (model states, optimizer states, RNG states)

This decomposition enables serialization-free encoding and decoding for efficient
checkpointing.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Union, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class TensorInfo:
    """Information about a single tensor in the state_dict.
    
    Attributes:
        key (str): the key/name of this tensor in state_dict (FQN)
        shape (tuple): shape of the tensor
        dtype (torch.dtype): data type of the tensor
        device (torch.device): device where the tensor resides
        numel (int): number of elements in the tensor
        size_bytes (int): size in bytes
        offset (int): offset in the continuous tensor data buffer (for reconstruction)
        global_offset (tuple): global offset for sharded tensors (from WriteItem.index.offset)
        shard_index (int): shard index for multiple shards with same key+offset
    """
    key: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    numel: int
    size_bytes: int
    offset: int = 0
    global_offset: Tuple[int, ...] = None  # Store WriteItem.index.offset as tuple
    shard_index: int = None  # Store WriteItem.index.index


@dataclass
class TensorMetadata:
    """
    Serializable metadata for a single tensor (for Phase 2 broadcasting).
    
    Used for all-to-all metadata exchange in distributed encoding.
    
    Attributes:
        key (str): Tensor FQN
        shape (tuple): Tensor shape
        dtype (str): Data type as string (e.g., 'torch.float32')
        size_bytes (int): Size in bytes
        global_offset (tuple): Global offset for sharded tensors
        shard_index (int): Shard index
        chunk_type (str): 'data' or 'parity'
        target_rank (int): Which rank should receive this chunk
        source_rank (int): Which rank sends this chunk
        cpu_buffer_address (int): CPU memory address (set after allocation)
        cpu_buffer_size (int): Buffer size in bytes (set after allocation)
    """
    key: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    global_offset: Tuple[int, ...]
    shard_index: int
    chunk_type: str = 'data'
    target_rank: int = 0
    source_rank: int = 0
    cpu_buffer_address: Optional[int] = None
    cpu_buffer_size: Optional[int] = None


@dataclass
class GlobalMetadataRegistry:
    """
    Complete metadata from all ranks after all-to-all exchange.
    
    Attributes:
        rank_metadata: Mapping from rank to its metadata list
        rank_non_tensor_data: Mapping from rank to its non-tensor data
    """
    rank_metadata: Dict[int, List[TensorMetadata]] = field(default_factory=dict)
    rank_non_tensor_data: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    def get_send_list(self, my_rank: int) -> List[TensorMetadata]:
        """
        Get tensors I need to send (from my own metadata).
        
        Args:
            my_rank (int): Current rank
            
        Returns:
            List[TensorMetadata]: Metadata for tensors to send
        """
        return self.rank_metadata.get(my_rank, [])
    
    def get_recv_list(self, my_rank: int) -> List[TensorMetadata]:
        """
        Get tensors I need to receive (from other ranks' metadata).
        
        Args:
            my_rank (int): Current rank
            
        Returns:
            List[TensorMetadata]: Metadata for tensors to receive
        """
        recv_list = []
        for source_rank, metadata_list in self.rank_metadata.items():
            if source_rank == my_rank:
                continue  # Skip own metadata
            for meta in metadata_list:
                # Check if this chunk should be received by me
                if meta.target_rank == my_rank:
                    recv_list.append(meta)
        return recv_list
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about metadata distribution.
        
        Returns:
            Dict with statistics
        """
        import pickle
        
        stats = {
            'total_ranks': len(self.rank_metadata),
            'per_rank_tensor_items': {},      # Tensor items per rank
            'per_rank_non_tensor_items': {},  # Non-tensor items per rank
            'total_tensor_chunks': 0,
            'total_non_tensor_items': 0,
            'total_tensor_data_bytes': 0,  # Size of actual tensor data (not metadata)
            'total_metadata_bytes': 0,  # Size of metadata itself (tensor + non-tensor)
        }
        
        for rank, metadata_list in self.rank_metadata.items():
            stats['per_rank_tensor_items'][rank] = len(metadata_list)
            stats['total_tensor_chunks'] += len(metadata_list)
            stats['total_tensor_data_bytes'] += sum(m.size_bytes for m in metadata_list)
            
            # Calculate actual metadata size (tensor metadata)
            tensor_metadata_bytes = pickle.dumps(metadata_list)
            stats['total_metadata_bytes'] += len(tensor_metadata_bytes)
        
        # Add non-tensor data statistics
        for rank, non_tensor_data in self.rank_non_tensor_data.items():
            stats['per_rank_non_tensor_items'][rank] = len(non_tensor_data)
            stats['total_non_tensor_items'] += len(non_tensor_data)
            
            non_tensor_bytes = pickle.dumps(non_tensor_data)
            stats['total_metadata_bytes'] += len(non_tensor_bytes)
        
        return stats


@dataclass
class DecomposedStateDict:
    """Decomposed state_dict structure for serialization-free encoding.
    
    This structure separates state_dict into three components:
    1. non_tensor_data: small metadata (iteration count, config, etc.)
    2. tensor_keys: list of tensor information (keys, shapes, dtypes, etc.)
    3. tensor_data: list of actual tensor data
    
    Attributes:
        non_tensor_data (Dict[str, Any]): non-tensor key-value pairs
        tensor_infos (List[TensorInfo]): metadata about all tensors
        tensor_data (List[torch.Tensor]): actual tensor data
        total_tensor_size_bytes (int): total size of all tensors in bytes
    """
    non_tensor_data: Dict[str, Any]
    tensor_infos: List[TensorInfo]
    tensor_data: List[torch.Tensor]
    total_tensor_size_bytes: int = 0
    flat_key_roots: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Calculate total tensor size after initialization."""
        if self.total_tensor_size_bytes == 0:
            self.total_tensor_size_bytes = sum(
                info.size_bytes for info in self.tensor_infos
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the decomposed state_dict.
        
        Returns:
            Dict containing size statistics for each component
        """
        import pickle
        
        # Calculate approximate sizes
        non_tensor_size = len(pickle.dumps(self.non_tensor_data))
        tensor_keys_size = sum(
            len(info.key.encode('utf-8')) + 
            len(str(info.shape).encode('utf-8')) + 
            16  # approximate size for dtype, device, etc.
            for info in self.tensor_infos
        )
        
        return {
            'non_tensor_size_bytes': non_tensor_size,
            'tensor_keys_size_bytes': tensor_keys_size,
            'tensor_data_size_bytes': self.total_tensor_size_bytes,
            'num_tensors': len(self.tensor_data),
            'non_tensor_percentage': 100.0 * non_tensor_size / (
                non_tensor_size + tensor_keys_size + self.total_tensor_size_bytes
            ),
            'tensor_keys_percentage': 100.0 * tensor_keys_size / (
                non_tensor_size + tensor_keys_size + self.total_tensor_size_bytes
            ),
            'tensor_data_percentage': 100.0 * self.total_tensor_size_bytes / (
                non_tensor_size + tensor_keys_size + self.total_tensor_size_bytes
            ),
        }


def decompose_state_dict(
    state_dict: Dict[str, Any],
    sort_by_size: bool = False
) -> DecomposedStateDict:
    """Decompose state_dict into three components.
    
    This function analyzes the state_dict and separates it into:
    1. Non-tensor key-value pairs (metadata)
    2. Tensor keys with metadata
    3. Tensor data
    
    Args:
        state_dict (Dict[str, Any]): the state_dict to decompose
        sort_by_size (bool): if True, sort tensors by size in descending order
    
    Returns:
        DecomposedStateDict: decomposed structure
    """
    non_tensor_data = {}
    tensor_infos = []
    tensor_data = []
    flat_key_roots: Set[str] = set()

    # Recursively traverse state_dict
    def _traverse_dict(d: Dict, prefix: str = ""):
        """Recursively traverse dictionary to find tensors."""
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key

            # Detect flat key roots: if prefix is non-empty and key contains a dot,
            # the top-level key has a flat dict with dot-containing keys (like
            # model state_dict from state_dict_for_save_checkpoint()).
            if prefix and isinstance(key, str) and '.' in key:
                top_level = prefix.split('.')[0]
                flat_key_roots.add(top_level)

            if isinstance(value, torch.Tensor):
                # This is a tensor - extract metadata
                tensor_info = TensorInfo(
                    key=full_key,
                    shape=tuple(value.shape),
                    dtype=value.dtype,
                    device=value.device,
                    numel=value.numel(),
                    size_bytes=value.numel() * value.element_size(),
                )
                tensor_infos.append(tensor_info)
                tensor_data.append(value)
            elif isinstance(value, dict):
                # Recursively traverse nested dictionaries
                _traverse_dict(value, full_key)
            else:
                # Non-tensor data
                if prefix:
                    if prefix not in non_tensor_data:
                        non_tensor_data[prefix] = {}
                    non_tensor_data[prefix][key] = value
                else:
                    non_tensor_data[full_key] = value

    _traverse_dict(state_dict)
    
    # Sort tensors by size if requested (largest first for better packing)
    if sort_by_size and tensor_data:
        combined = list(zip(tensor_infos, tensor_data))
        combined.sort(key=lambda x: x[0].size_bytes, reverse=True)
        tensor_infos, tensor_data = zip(*combined)
        tensor_infos = list(tensor_infos)
        tensor_data = list(tensor_data)
    
    # Calculate offsets for reconstruction
    offset = 0
    for info in tensor_infos:
        info.offset = offset
        offset += info.size_bytes
    
    decomposed = DecomposedStateDict(
        non_tensor_data=non_tensor_data,
        tensor_infos=tensor_infos,
        tensor_data=tensor_data,
        flat_key_roots=flat_key_roots,
    )
    
    # Log statistics
    stats = decomposed.get_statistics()
    logger.debug(
        f"Decomposed state_dict statistics:\n"
        f"  Non-tensor data: {stats['non_tensor_size_bytes']} bytes "
        f"({stats['non_tensor_percentage']:.4f}%)\n"
        f"  Tensor keys: {stats['tensor_keys_size_bytes']} bytes "
        f"({stats['tensor_keys_percentage']:.4f}%)\n"
        f"  Tensor data: {stats['tensor_data_size_bytes']} bytes "
        f"({stats['tensor_data_percentage']:.4f}%)\n"
        f"  Total tensors: {stats['num_tensors']}"
    )
    
    return decomposed


def reconstruct_state_dict(decomposed: DecomposedStateDict) -> Dict[str, Any]:
    """Reconstruct original state_dict from decomposed structure.
    
    This function is used during checkpoint loading/decoding to reconstruct
    the original state_dict from the three components.
    
    Args:
        decomposed (DecomposedStateDict): decomposed structure
    
    Returns:
        Dict[str, Any]: reconstructed state_dict
    """
    state_dict = {}

    def _ensure_nested(current, keys):
        for k in keys:
            if k not in current:
                current[k] = {}
            current = current[k]
        return current

    # First, add non-tensor data
    for key, value in decomposed.non_tensor_data.items():
        if isinstance(value, dict):
            # key is a dot-separated prefix; navigate into the nested structure
            target = _ensure_nested(state_dict, key.split('.'))
            target.update(value)
        else:
            # Scalar value: if key contains dots, nest it; otherwise top-level
            keys = key.split('.')
            if len(keys) == 1:
                state_dict[key] = value
            else:
                target = _ensure_nested(state_dict, keys[:-1])
                target[keys[-1]] = value

    # Then, add tensor data with flat_key_roots awareness
    for info, tensor in zip(decomposed.tensor_infos, decomposed.tensor_data):
        # Check if this tensor's top-level key is a flat key root.
        # For flat key roots, the key after the first dot is a single flat key
        # (not multi-level nesting). This handles model state_dicts where
        # keys like "model.module.decoder.weight" must be stored as
        # state_dict["model"]["module.decoder.weight"], NOT nested.
        first_dot = info.key.find('.')
        if first_dot >= 0 and info.key[:first_dot] in decomposed.flat_key_roots:
            top_key = info.key[:first_dot]
            flat_sub_key = info.key[first_dot + 1:]
            if top_key not in state_dict:
                state_dict[top_key] = {}
            state_dict[top_key][flat_sub_key] = tensor
        else:
            keys = info.key.split('.')
            target = _ensure_nested(state_dict, keys[:-1])
            target[keys[-1]] = tensor

    return state_dict


# ---------------------------------------------------------------------------
# Optimizer fp32_from_fp16_params flatten / unflatten
# ---------------------------------------------------------------------------

_OPT_FP32_FLAT_PREFIX = "_fp32_group"


def flatten_optimizer_fp32_params(state_dict: Dict[str, Any]) -> bool:
    """Flatten ``fp32_from_fp16_params`` (list-of-lists of tensors) into a dict
    so that ``decompose_state_dict`` can extract the tensors into the
    EC-protected tensor buffer instead of burying them in non_tensor_data.

    Modifies *state_dict* in-place.  Returns True if flattening was performed.
    """
    optim_sd = state_dict.get("optimizer")
    if optim_sd is None:
        return False
    fp32_params = optim_sd.get("fp32_from_fp16_params")
    if fp32_params is None:
        return False

    structure = []       # [len(g0), len(g1), ...]
    flat_dict = {}       # "_fp32_group{gi}_param{pi}" → tensor
    for gi, group in enumerate(fp32_params):
        structure.append(len(group))
        for pi, tensor in enumerate(group):
            flat_dict[f"{_OPT_FP32_FLAT_PREFIX}{gi}_param{pi}"] = tensor

    optim_sd.pop("fp32_from_fp16_params")          # remove old list-of-lists
    optim_sd["fp32_params_flat"] = flat_dict       # dict form → decompose extracts tensors
    optim_sd["_fp32_structure"] = structure        # small metadata for unflatten
    return True


def unflatten_optimizer_fp32_params(reconstructed: Dict[str, Any]) -> bool:
    """Reverse of ``flatten_optimizer_fp32_params``: rebuild the original
    ``fp32_from_fp16_params`` list-of-lists from the flat dict produced
    during save.

    Modifies *reconstructed* in-place.  Returns True if unflattening was
    performed (i.e. the flat representation was present).
    """
    optim_sd = reconstructed.get("optimizer")
    if optim_sd is None:
        return False
    flat_dict = optim_sd.pop("fp32_params_flat", None)
    structure = optim_sd.pop("_fp32_structure", None)
    if flat_dict is None or structure is None:
        return False  # old-format checkpoint — fp32_from_fp16_params already present

    fp32_params: List[List[torch.Tensor]] = []
    for gi, group_size in enumerate(structure):
        group = []
        for pi in range(group_size):
            key = f"{_OPT_FP32_FLAT_PREFIX}{gi}_param{pi}"
            group.append(flat_dict[key])
        fp32_params.append(group)
    optim_sd["fp32_from_fp16_params"] = fp32_params
    return True


def organize_tensor_data_in_cpu_memory(
    tensor_data_list: List[torch.Tensor],
    use_continuous_buffer: bool = True,
    pin_memory: bool = False
) -> Union[List[torch.Tensor], torch.Tensor]:
    """Organize tensor data in CPU memory for efficient encoding.
    
    When tensors are transferred from GPU to CPU, this function organizes them
    in CPU memory in a way that facilitates subsequent encoding operations.
    
    Args:
        tensor_data_list (List[torch.Tensor]): list of tensors (should be on CPU)
        use_continuous_buffer (bool): if True, concatenate all tensors into a 
            single continuous buffer for better cache locality
        pin_memory (bool): if True, use pinned memory for faster GPU-CPU transfers
    
    Returns:
        Either a list of tensors or a single continuous tensor buffer
    """
    if not tensor_data_list:
        return [] if not use_continuous_buffer else torch.tensor([], dtype=torch.uint8)
    
    # Ensure all tensors are on CPU
    for i, tensor in enumerate(tensor_data_list):
        if tensor.device.type != 'cpu':
            logger.warning(
                f"Tensor {i} is on {tensor.device}, moving to CPU. "
                f"This should have been done earlier for better performance."
            )
            tensor_data_list[i] = tensor.cpu()
    
    if not use_continuous_buffer:
        # Clone tensors to ensure they are contiguous
        result = []
        for tensor in tensor_data_list:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            if pin_memory:
                tensor = tensor.pin_memory()
            result.append(tensor)
        return result
    
    # Create a continuous buffer
    total_size = sum(t.numel() * t.element_size() for t in tensor_data_list)
    
    # Allocate continuous buffer as bytes
    if pin_memory:
        buffer = torch.empty(total_size, dtype=torch.uint8).pin_memory()
    else:
        buffer = torch.empty(total_size, dtype=torch.uint8)
    
    # Copy tensors into the buffer
    offset = 0
    for tensor in tensor_data_list:
        tensor_bytes = tensor.numel() * tensor.element_size()
        # View tensor as bytes and copy
        # Must flatten first, then view as uint8 to get a 1D byte array
        tensor_flat = tensor.flatten().contiguous().view(torch.uint8)
        buffer[offset:offset + tensor_bytes].copy_(tensor_flat)
        offset += tensor_bytes
    
    logger.debug(
        f"Organized {len(tensor_data_list)} tensors into continuous buffer "
        f"of size {total_size} bytes ({total_size / (1024**3):.2f} GB)"
    )
    
    return buffer


def extract_tensors_from_continuous_buffer(
    buffer: torch.Tensor,
    tensor_infos: List[TensorInfo]
) -> List[torch.Tensor]:
    """Extract individual tensors from a continuous buffer.
    
    This is the reverse operation of organize_tensor_data_in_cpu_memory.
    Used during checkpoint loading to extract tensors from the buffer.
    
    Args:
        buffer (torch.Tensor): continuous buffer containing all tensor data
        tensor_infos (List[TensorInfo]): metadata about tensors
    
    Returns:
        List[torch.Tensor]: list of individual tensors
    """
    tensors = []
    
    for info in tensor_infos:
        # Extract bytes for this tensor
        start = info.offset
        end = start + info.size_bytes
        tensor_bytes = buffer[start:end]
        
        # Reshape to original tensor
        # First view as correct dtype, then reshape
        tensor = tensor_bytes.view(info.dtype).reshape(info.shape)
        tensors.append(tensor)
    
    logger.debug(f"Extracted {len(tensors)} tensors from continuous buffer")
    
    return tensors

