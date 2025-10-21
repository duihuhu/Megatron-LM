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
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class TensorInfo:
    """Information about a single tensor in the state_dict.
    
    Attributes:
        key (str): the key/name of this tensor in state_dict
        shape (tuple): shape of the tensor
        dtype (torch.dtype): data type of the tensor
        device (torch.device): device where the tensor resides
        numel (int): number of elements in the tensor
        size_bytes (int): size in bytes
        offset (int): offset in the continuous tensor data buffer (for reconstruction)
        metadata_index (Any): full metadata index from WriteItem (for proper reconstruction)
    """
    key: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    numel: int
    size_bytes: int
    offset: int = 0
    metadata_index: Any = None  # Store full WriteItem.index for proper key mapping


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
    
    # Recursively traverse state_dict
    def _traverse_dict(d: Dict, prefix: str = ""):
        """Recursively traverse dictionary to find tensors."""
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
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
    
    # First, add non-tensor data
    for key, value in decomposed.non_tensor_data.items():
        if isinstance(value, dict):
            # Nested dictionary
            if key not in state_dict:
                state_dict[key] = {}
            state_dict[key].update(value)
        else:
            state_dict[key] = value
    
    # Then, add tensor data
    for info, tensor in zip(decomposed.tensor_infos, decomposed.tensor_data):
        # Navigate to the correct position in nested dict
        keys = info.key.split('.')
        current = state_dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = tensor
    
    return state_dict


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

