"""
GPU to CPU Memory Pool Manager
===============================

A memory pool manager for efficiently transferring multiple GPU tensors to pre-allocated CPU memory.
Features:
- Pre-allocated memory pool
- Automatic allocation and deallocation tracking
- Support for multiple concurrent tensor transfers
- Memory alignment and fragmentation handling
- Thread-safe operations

Author: ECTrain Project
Date: 2025-10-15
"""

import torch
import numpy as np
import ctypes
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
from enum import Enum
import platform
import sys


class AllocationStrategy(Enum):
    """Memory allocation strategies."""
    FIRST_FIT = "first_fit"      # Find first block that fits
    BEST_FIT = "best_fit"        # Find smallest block that fits
    WORST_FIT = "worst_fit"      # Find largest block (reduce fragmentation)


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""
    offset: int          # Offset from pool base address
    size: int           # Size in bytes
    is_free: bool       # Whether block is free
    tensor_id: Optional[int] = None  # ID of tensor using this block
    

class CPUMemoryPool:
    """
    Pre-allocated CPU memory pool for GPU tensor transfers.
    
    Features:
    - Pre-allocate large memory pool at initialization
    - Efficiently allocate/deallocate memory for multiple tensors
    - Track memory usage and fragmentation
    - Thread-safe operations
    """
    
    def __init__(
        self,
        pool_size_bytes: int,
        alignment: int = 64,
        strategy: AllocationStrategy = AllocationStrategy.BEST_FIT,
        enable_alignment: bool = True,
        use_pinned_pool: bool = False
    ):
        """
        Initialize memory pool.
        
        Args:
            pool_size_bytes: Total size of memory pool in bytes
            alignment: Memory alignment boundary (default 64 bytes)
            strategy: Allocation strategy
            enable_alignment: Whether to align allocations (default True for performance)
            use_pinned_pool: Whether to use pinned memory for entire pool (faster transfer)
        """
        self.pool_size = pool_size_bytes
        self.alignment = alignment if enable_alignment else 1
        self.strategy = strategy
        self.enable_alignment = enable_alignment
        self.use_pinned_pool = use_pinned_pool
        
        # Allocate the memory pool
        self._allocate_pool()
        
        # Initialize with one large free block
        self.blocks: List[MemoryBlock] = [
            MemoryBlock(offset=0, size=self.pool_size, is_free=True)
        ]
        
        # Track allocations
        self.allocations: Dict[int, MemoryBlock] = {}  # tensor_id -> block
        self.next_tensor_id = 0
        
        # Thread safety - use RLock to allow re-entrant calls
        self.lock = threading.RLock()  # Changed from Lock to RLock
        
        # Statistics
        self.total_allocated = 0
        self.peak_allocated = 0
        self.num_allocations = 0
        self.num_deallocations = 0
        
    def _allocate_pool(self):
        """Allocate the memory pool with proper alignment."""
        if self.use_pinned_pool:
            # Use PyTorch's pinned memory (via cudaHostAlloc, more reliable)
            # PyTorch automatically handles permissions and system limits
            try:
                self._pinned_tensor = torch.empty(
                    self.pool_size,
                    dtype=torch.uint8,
                    device='cpu',
                    pin_memory=True  # Uses cudaHostAlloc internally
                )
                self.base_address = self._pinned_tensor.data_ptr()
                self._raw_buffer = None
                print(f"[MemoryPool] Allocated {self.pool_size / 1024 / 1024:.2f} MB (PINNED via PyTorch) at 0x{self.base_address:x}")
            except Exception as e:
                print(f"[MemoryPool] WARNING: Failed to allocate pinned memory: {e}")
                print(f"[MemoryPool] Falling back to regular memory")
                self.use_pinned_pool = False
                # Fall through to regular allocation
        
        if not self.use_pinned_pool:
            # Use ctypes for regular memory allocation
            self._raw_buffer = ctypes.create_string_buffer(self.pool_size + self.alignment)
            
            # Get aligned address
            raw_address = ctypes.addressof(self._raw_buffer)
            self.base_address = (raw_address + self.alignment - 1) & ~(self.alignment - 1)
            self._pinned_tensor = None
            
            print(f"[MemoryPool] Allocated {self.pool_size / 1024 / 1024:.2f} MB at 0x{self.base_address:x}")
    
    def allocate(self, size_bytes: int, tensor_id: Optional[int] = None) -> Tuple[int, int]:
        """
        Allocate memory from the pool.
        
        Args:
            size_bytes: Size to allocate
            tensor_id: Optional tensor ID (auto-generated if None)
            
        Returns:
            Tuple of (memory_address, tensor_id)
            
        Raises:
            MemoryError: If allocation fails
        """
        with self.lock:
            # Align size to alignment boundary
            aligned_size = (size_bytes + self.alignment - 1) & ~(self.alignment - 1)
            
            # Find suitable block
            block_idx = self._find_free_block(aligned_size)
            
            if block_idx is None:
                # Try to coalesce free blocks first
                self._coalesce_free_blocks()
                block_idx = self._find_free_block(aligned_size)
                
                if block_idx is None:
                    raise MemoryError(
                        f"Cannot allocate {aligned_size} bytes. "
                        f"Available: {self.get_free_memory()} bytes, "
                        f"Largest block: {self.get_largest_free_block()} bytes"
                    )
            
            # Allocate from the block
            block = self.blocks[block_idx]
            
            if tensor_id is None:
                tensor_id = self.next_tensor_id
                self.next_tensor_id += 1
            
            # Split block if necessary
            if block.size > aligned_size:
                # Create new free block for remaining space
                remaining_block = MemoryBlock(
                    offset=block.offset + aligned_size,
                    size=block.size - aligned_size,
                    is_free=True
                )
                self.blocks.insert(block_idx + 1, remaining_block)
            
            # Mark block as allocated
            block.size = aligned_size
            block.is_free = False
            block.tensor_id = tensor_id
            
            # Track allocation
            self.allocations[tensor_id] = block
            self.total_allocated += aligned_size
            self.peak_allocated = max(self.peak_allocated, self.total_allocated)
            self.num_allocations += 1
            
            address = self.base_address + block.offset
            
            return address, tensor_id
    
    def allocate_contiguous(self, sizes_bytes: List[int]) -> Tuple[List[int], List[int]]:
        """
        Allocate contiguous memory for multiple tensors (addresses are consecutive).
        
        Args:
            sizes_bytes: List of sizes for each tensor
            
        Returns:
            Tuple of (list of addresses, list of tensor IDs)
            
        Raises:
            MemoryError: If allocation fails
        """
        with self.lock:
            # Calculate total size needed
            aligned_sizes = [(size + self.alignment - 1) & ~(self.alignment - 1) 
                           for size in sizes_bytes]
            total_size = sum(aligned_sizes)
            
            # Find a single block that can fit all tensors
            block_idx = self._find_free_block(total_size)
            
            if block_idx is None:
                self._coalesce_free_blocks()
                block_idx = self._find_free_block(total_size)
                
                if block_idx is None:
                    raise MemoryError(
                        f"Cannot allocate contiguous {total_size} bytes for {len(sizes_bytes)} tensors. "
                        f"Available: {self.get_free_memory()} bytes, "
                        f"Largest block: {self.get_largest_free_block()} bytes"
                    )
            
            block = self.blocks[block_idx]
            addresses = []
            tensor_ids = []
            current_offset = block.offset
            
            # Allocate each tensor in sequence
            for i, aligned_size in enumerate(aligned_sizes):
                tensor_id = self.next_tensor_id
                self.next_tensor_id += 1
                
                # Create block for this tensor
                new_block = MemoryBlock(
                    offset=current_offset,
                    size=aligned_size,
                    is_free=False,
                    tensor_id=tensor_id
                )
                
                # Insert at the correct position
                self.blocks.insert(block_idx + i, new_block)
                
                # Track allocation
                self.allocations[tensor_id] = new_block
                self.total_allocated += aligned_size
                self.num_allocations += 1
                
                addresses.append(self.base_address + current_offset)
                tensor_ids.append(tensor_id)
                current_offset += aligned_size
            
            # Update the original block
            if block.size > total_size:
                # Remaining free space
                block.offset = current_offset
                block.size -= total_size
                # Block remains at block_idx + len(aligned_sizes)
            else:
                # Entire block used, remove it
                self.blocks.pop(block_idx + len(aligned_sizes))
            
            self.peak_allocated = max(self.peak_allocated, self.total_allocated)
            
            return addresses, tensor_ids
    
    def deallocate(self, tensor_id: int):
        """
        Deallocate memory for a tensor.
        
        Args:
            tensor_id: ID of tensor to deallocate
        """
        with self.lock:
            if tensor_id not in self.allocations:
                raise ValueError(f"Tensor ID {tensor_id} not found in allocations")
            
            block = self.allocations[tensor_id]
            block.is_free = True
            block.tensor_id = None
            
            self.total_allocated -= block.size
            self.num_deallocations += 1
            
            # Remove from allocations
            del self.allocations[tensor_id]
            
            # Coalesce adjacent free blocks
            self._coalesce_free_blocks()
    
    def _find_free_block(self, size: int) -> Optional[int]:
        """Find a free block that can fit the requested size."""
        if self.strategy == AllocationStrategy.FIRST_FIT:
            for idx, block in enumerate(self.blocks):
                if block.is_free and block.size >= size:
                    return idx
                    
        elif self.strategy == AllocationStrategy.BEST_FIT:
            best_idx = None
            best_size = float('inf')
            for idx, block in enumerate(self.blocks):
                if block.is_free and block.size >= size and block.size < best_size:
                    best_idx = idx
                    best_size = block.size
            return best_idx
            
        elif self.strategy == AllocationStrategy.WORST_FIT:
            worst_idx = None
            worst_size = 0
            for idx, block in enumerate(self.blocks):
                if block.is_free and block.size >= size and block.size > worst_size:
                    worst_idx = idx
                    worst_size = block.size
            return worst_idx
            
        return None
    
    def _coalesce_free_blocks(self):
        """Merge adjacent free blocks to reduce fragmentation."""
        if len(self.blocks) <= 1:
            return
        
        new_blocks = []
        i = 0
        while i < len(self.blocks):
            current = self.blocks[i]
            
            if current.is_free:
                # Merge with next free blocks
                merged_size = current.size
                j = i + 1
                while j < len(self.blocks) and self.blocks[j].is_free:
                    merged_size += self.blocks[j].size
                    j += 1
                
                if j > i + 1:
                    # Create merged block
                    new_blocks.append(MemoryBlock(
                        offset=current.offset,
                        size=merged_size,
                        is_free=True
                    ))
                    i = j
                else:
                    new_blocks.append(current)
                    i += 1
            else:
                new_blocks.append(current)
                i += 1
        
        self.blocks = new_blocks
    
    def get_free_memory(self) -> int:
        """Get total free memory in bytes."""
        with self.lock:
            return sum(block.size for block in self.blocks if block.is_free)
    
    def get_used_memory(self) -> int:
        """Get total used memory in bytes."""
        return self.total_allocated
    
    def get_largest_free_block(self) -> int:
        """Get size of largest free block."""
        with self.lock:
            free_blocks = [block.size for block in self.blocks if block.is_free]
            return max(free_blocks) if free_blocks else 0
    
    def get_fragmentation_ratio(self) -> float:
        """
        Get fragmentation ratio.
        
        Returns:
            Ratio between 0 (no fragmentation) and 1 (highly fragmented)
        """
        with self.lock:
            free_memory = self.get_free_memory()
            if free_memory == 0:
                return 0.0
            
            largest_free = self.get_largest_free_block()
            return 1.0 - (largest_free / free_memory)
    
    def get_statistics(self) -> Dict:
        """Get memory pool statistics."""
        with self.lock:
            return {
                'pool_size': self.pool_size,
                'used_memory': self.total_allocated,
                'free_memory': self.get_free_memory(),
                'largest_free_block': self.get_largest_free_block(),
                'fragmentation_ratio': self.get_fragmentation_ratio(),
                'num_blocks': len(self.blocks),
                'num_allocations': self.num_allocations,
                'num_deallocations': self.num_deallocations,
                'peak_allocated': self.peak_allocated,
                'utilization': self.total_allocated / self.pool_size * 100
            }
    
    def reset(self):
        """Reset the memory pool to initial state."""
        with self.lock:
            self.blocks = [MemoryBlock(offset=0, size=self.pool_size, is_free=True)]
            self.allocations.clear()
            self.total_allocated = 0
            self.num_allocations = 0
            self.num_deallocations = 0
    
    def print_memory_map(self):
        """Print visual representation of memory layout."""
        with self.lock:
            print("\n" + "=" * 80)
            print("Memory Pool Layout")
            print("=" * 80)
            
            for idx, block in enumerate(self.blocks):
                status = "FREE" if block.is_free else f"USED (tensor {block.tensor_id})"
                offset_mb = block.offset / 1024 / 1024
                size_mb = block.size / 1024 / 1024
                print(f"Block {idx:3d}: offset={offset_mb:8.2f} MB, size={size_mb:8.2f} MB, {status}")
            
            stats = self.get_statistics()
            print("\n" + "-" * 80)
            print(f"Total: {stats['pool_size'] / 1024 / 1024:.2f} MB")
            print(f"Used:  {stats['used_memory'] / 1024 / 1024:.2f} MB ({stats['utilization']:.1f}%)")
            print(f"Free:  {stats['free_memory'] / 1024 / 1024:.2f} MB")
            print(f"Fragmentation: {stats['fragmentation_ratio']:.2%}")
            print("=" * 80 + "\n")


class GPUToCPUPoolTransfer:
    """
    Manager for transferring multiple GPU tensors to a CPU memory pool.
    """
    
    def __init__(self, memory_pool: CPUMemoryPool):
        """
        Initialize transfer manager.
        
        Args:
            memory_pool: Pre-allocated CPU memory pool
        """
        self.pool = memory_pool
        self.tensor_metadata: Dict[int, Dict] = {}  # tensor_id -> {shape, dtype, address}
        self._cpu_tensor_cache: Dict[int, torch.Tensor] = {}  # Cache CPU tensor views
    
    def transfer_to_pool(
        self,
        gpu_tensor: torch.Tensor,
        tensor_id: Optional[int] = None
    ) -> int:
        """
        Transfer GPU tensor to memory pool.
        
        Args:
            gpu_tensor: GPU tensor to transfer
            tensor_id: Optional tensor ID
            
        Returns:
            Tensor ID
            
        Note:
            Transfer speed depends on pool.use_pinned_pool setting.
            If pool is pinned, transfer is automatically faster.
        """
        if not gpu_tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device")
        
        # Calculate required size
        gpu_tensor = gpu_tensor.contiguous()
        size_bytes = gpu_tensor.element_size() * gpu_tensor.nelement()
        
        # Allocate from pool
        address, tensor_id = self.pool.allocate(size_bytes, tensor_id)
        
        # Get the actual allocated size (aligned) from the pool
        actual_size = self.pool.allocations[tensor_id].size
        
        # Store metadata
        self.tensor_metadata[tensor_id] = {
            'shape': gpu_tensor.shape,
            'dtype': gpu_tensor.dtype,
            'address': address,
            'size': actual_size  # Use actual aligned size, not original size
        }
        
        # Transfer data (pass tensor_id for potential caching)
        self._do_transfer(gpu_tensor, address, tensor_id)
        
        return tensor_id
    
    def transfer_batch_to_pool(
        self,
        gpu_tensors: List[torch.Tensor],
        contiguous: bool = True
    ) -> List[int]:
        """
        Transfer multiple GPU tensors to pool in batch.
        
        Args:
            gpu_tensors: List of GPU tensors
            contiguous: If True, allocate contiguous memory for all tensors (addresses consecutive)
            
        Returns:
            List of tensor IDs
            
        Note:
            Transfer speed depends on pool.use_pinned_pool setting.
        """
        if not contiguous:
            # Original behavior: allocate separately
            tensor_ids = []
            for gpu_tensor in gpu_tensors:
                tid = self.transfer_to_pool(gpu_tensor)
                tensor_ids.append(tid)
            return tensor_ids
        
        # Contiguous allocation: all tensors in consecutive memory
        if not all(t.is_cuda for t in gpu_tensors):
            raise ValueError("All tensors must be on CUDA device")
        
        # Make all tensors contiguous and calculate sizes
        gpu_tensors_contig = [t.contiguous() for t in gpu_tensors]
        sizes_bytes = [t.element_size() * t.nelement() for t in gpu_tensors_contig]
        
        # Allocate contiguous memory from pool
        addresses, tensor_ids = self.pool.allocate_contiguous(sizes_bytes)
        
        # Transfer each tensor to its allocated address
        for i, (gpu_tensor, address, tensor_id) in enumerate(zip(gpu_tensors_contig, addresses, tensor_ids)):
            # Get the actual allocated size (aligned) from the pool
            actual_size = self.pool.allocations[tensor_id].size
            
            # Store metadata
            self.tensor_metadata[tensor_id] = {
                'shape': gpu_tensor.shape,
                'dtype': gpu_tensor.dtype,
                'address': address,
                'size': actual_size  # Use actual aligned size, not original size
            }
            
            # Transfer data (pass tensor_id for caching)
            self._do_transfer(gpu_tensor, address, tensor_id)
        
        return tensor_ids
    
    def transfer_batch_async(
        self,
        gpu_tensors: List[torch.Tensor],
        stream: Optional[torch.cuda.Stream] = None,
        contiguous: bool = True
    ) -> Tuple[List[int], torch.cuda.Event]:
        """
        Asynchronously transfer multiple GPU tensors to pool.
        
        Args:
            gpu_tensors: List of GPU tensors
            stream: CUDA stream for async transfer
            contiguous: If True, allocate contiguous memory for all tensors
            
        Returns:
            Tuple of (list of tensor IDs, CUDA event)
        """
        if stream is None:
            stream = torch.cuda.Stream()
        
        if not all(t.is_cuda for t in gpu_tensors):
            raise ValueError("All tensors must be on CUDA device")
        
        # Make all tensors contiguous
        gpu_tensors_contig = [t.contiguous() for t in gpu_tensors]
        sizes_bytes = [t.element_size() * t.nelement() for t in gpu_tensors_contig]
        
        tensor_ids = []
        
        with torch.cuda.stream(stream):
            if contiguous:
                # Allocate contiguous memory for all tensors
                addresses, tensor_ids = self.pool.allocate_contiguous(sizes_bytes)
            else:
                # Allocate separately
                addresses = []
                tensor_ids = []
                for size_bytes in sizes_bytes:
                    address, tensor_id = self.pool.allocate(size_bytes)
                    addresses.append(address)
                    tensor_ids.append(tensor_id)
            
            # Transfer all tensors asynchronously
            for i, (gpu_tensor, address, tensor_id) in enumerate(zip(gpu_tensors_contig, addresses, tensor_ids)):
                # Get the actual allocated size (aligned) from the pool
                actual_size = self.pool.allocations[tensor_id].size
                
                # Store metadata
                self.tensor_metadata[tensor_id] = {
                    'shape': gpu_tensor.shape,
                    'dtype': gpu_tensor.dtype,
                    'address': address,
                    'size': actual_size  # Use actual aligned size, not original size
                }
                
                # Async transfer (pass tensor_id for caching)
                self._do_transfer_async(gpu_tensor, address, stream, tensor_id)
            
            # Create event to mark completion
            event = torch.cuda.Event()
            event.record(stream)
        
        return tensor_ids, event
    
    def get_tensor_from_pool(self, tensor_id: int) -> torch.Tensor:
        """
        Get CPU tensor from pool (zero-copy view).
        
        Args:
            tensor_id: ID of tensor to retrieve
            
        Returns:
            CPU tensor backed by pool memory
        """
        if tensor_id not in self.tensor_metadata:
            raise ValueError(f"Tensor ID {tensor_id} not found")
        
        metadata = self.tensor_metadata[tensor_id]
        
        return self._create_cpu_tensor_at_address(
            metadata['address'],
            metadata['shape'],
            metadata['dtype']
        )
    
    def free_tensor(self, tensor_id: int):
        """
        Free tensor from pool.
        
        Args:
            tensor_id: ID of tensor to free
        """
        if tensor_id in self.tensor_metadata:
            del self.tensor_metadata[tensor_id]
        
        self.pool.deallocate(tensor_id)
    
    def free_batch(self, tensor_ids: List[int]):
        """Free multiple tensors."""
        for tid in tensor_ids:
            self.free_tensor(tid)
    
    def prepare_cpu_tensor_view(self, tensor_id: int) -> torch.Tensor:
        """
        Pre-create CPU tensor view for a tensor (public API for optimization).
        
        This allows callers to pre-build tensor views before transfer,
        improving performance in tight loops.
        
        Args:
            tensor_id: ID of the tensor
            
        Returns:
            CPU tensor view backed by pool memory
            
        Note: This is a lightweight view operation, very fast (~0.01ms).
        
        Example:
            # Pre-build views before transfer loop
            for tid in tensor_ids:
                mgr.prepare_cpu_tensor_view(tid)
            
            # Now transfers are faster (use cached views)
            for gpu_tensor, tid in zip(gpu_tensors, tensor_ids):
                mgr._do_transfer(gpu_tensor, address, tid)
        """
        if tensor_id not in self.tensor_metadata:
            raise ValueError(f"Tensor ID {tensor_id} not found")
        
        metadata = self.tensor_metadata[tensor_id]
        address = metadata['address']
        shape = metadata['shape']
        dtype = metadata['dtype']
        
        # Create and cache the view
        cpu_tensor_view = self._create_cpu_tensor_view_internal(
            address,
            shape,
            dtype
        )
        self._cpu_tensor_cache[tensor_id] = cpu_tensor_view
        
        return cpu_tensor_view
    
    def prepare_batch_cpu_tensor_views(self, tensor_ids: List[int]) -> List[torch.Tensor]:
        """
        Pre-create CPU tensor views for multiple tensors (batch optimization).
        
        Args:
            tensor_ids: List of tensor IDs
            
        Returns:
            List of CPU tensor views
            
        Example:
            # Pre-build all views at once
            views = mgr.prepare_batch_cpu_tensor_views(tensor_ids)
            
            # Now all transfers use cached views (faster)
        """
        return [self.prepare_cpu_tensor_view(tid) for tid in tensor_ids]
    
    def _create_cpu_tensor_view_internal(self, address: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """
        Internal method to create CPU tensor view at address.
        
        Note: This creates a lightweight view, not a copy. Very fast operation.
        """
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
        }
        
        numpy_dtype = dtype_map.get(dtype)
        if numpy_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        # Calculate total bytes
        numel = np.prod(shape)
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_bytes = numel * element_size
        
        # Create numpy array view at address (lightweight, no copy)
        byte_array = np.ctypeslib.as_array(
            ctypes.cast(address, ctypes.POINTER(ctypes.c_byte)),
            shape=(total_bytes,)
        )
        
        typed_array = np.frombuffer(byte_array, dtype=numpy_dtype).reshape(shape)
        return torch.from_numpy(typed_array)
    
    def _do_transfer(self, gpu_tensor: torch.Tensor, address: int, tensor_id: Optional[int] = None):
        """
        Perform the actual transfer from GPU to CPU memory address.
        
        Args:
            gpu_tensor: GPU tensor to transfer
            address: Target CPU memory address
            tensor_id: Optional tensor ID (if provided, will try to use cached view)
        
        Transfer speed automatically depends on pool.use_pinned_pool setting.
        """
        # Try to use cached CPU tensor view first (optimization)
        if tensor_id is not None and tensor_id in self._cpu_tensor_cache:
            cpu_tensor = self._cpu_tensor_cache[tensor_id]
        else:
            # Create CPU tensor view at target address
            cpu_tensor = self._create_cpu_tensor_view_internal(
                address,
                gpu_tensor.shape,
                gpu_tensor.dtype
            )
        
        # Transfer strategy depends on whether pool is pinned
        # Important: Flatten both tensors to match shapes
        cpu_tensor_flat = cpu_tensor.view(-1)
        gpu_tensor_flat = gpu_tensor.view(-1)
        
        if self.pool.use_pinned_pool:
            # Pool is pinned, use non-blocking copy for better performance
            cpu_tensor_flat.copy_(gpu_tensor_flat, non_blocking=True)
            torch.cuda.synchronize()
        else:
            # Pool is NOT pinned, use regular blocking copy
            cpu_tensor_flat.copy_(gpu_tensor_flat)
            torch.cuda.synchronize()
    
    def _do_transfer_async(self, gpu_tensor: torch.Tensor, address: int, stream: torch.cuda.Stream, tensor_id: Optional[int] = None):
        """
        Perform async transfer.
        
        Args:
            gpu_tensor: GPU tensor to transfer
            address: Target CPU memory address
            stream: CUDA stream for async operations
            tensor_id: Optional tensor ID (if provided, will try to use cached view)
        
        Strategy:
        - If pool is pinned: direct async copy (fastest)
        - If pool is NOT pinned: use temporary pinned buffer (like .to does)
        """
        # Try to use cached CPU tensor view first (optimization)
        if tensor_id is not None and tensor_id in self._cpu_tensor_cache:
            cpu_tensor = self._cpu_tensor_cache[tensor_id]
        else:
            # Create target CPU tensor at the specified address
            cpu_tensor = self._create_cpu_tensor_view_internal(
                address,
                gpu_tensor.shape,
                gpu_tensor.dtype
            )
        
        # Flatten both tensors to match shapes
        cpu_tensor_flat = cpu_tensor.view(-1)
        
        if self.pool.use_pinned_pool:
            # Pool is pinned, direct async copy (fastest)
            gpu_tensor_flat = gpu_tensor.view(-1)
            cpu_tensor_flat.copy_(gpu_tensor_flat, non_blocking=True)
        else:
            # Pool is NOT pinned, use temporary pinned buffer (like .to does)
            # Strategy: GPU → pinned buffer (async) → target memory (sync after stream.sync)
            
            # Create temporary pinned buffer
            temp_pinned = torch.empty(
                gpu_tensor.shape,
                dtype=gpu_tensor.dtype,
                device='cpu',
                pin_memory=True
            )
            
            # GPU → temp pinned buffer (truly async within stream)
            temp_pinned.copy_(gpu_tensor, non_blocking=True)
            
            # pinned buffer → target address
            # The main async benefit comes from GPU→pinned being non-blocking
            cpu_tensor_flat.copy_(temp_pinned.view(-1))
    
    def _create_cpu_tensor_at_address(self, address: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Create CPU tensor at specific address."""
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
        }
        
        numpy_dtype = dtype_map[dtype]
        numel = np.prod(shape)
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_bytes = numel * element_size
        
        byte_array = np.ctypeslib.as_array(
            ctypes.cast(address, ctypes.POINTER(ctypes.c_byte)),
            shape=(total_bytes,)
        )
        
        typed_array = np.frombuffer(byte_array, dtype=numpy_dtype).reshape(shape)
        return torch.from_numpy(typed_array)

