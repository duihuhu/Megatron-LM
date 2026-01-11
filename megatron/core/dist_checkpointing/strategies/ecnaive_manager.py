# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""EC-NAIVE manager for shared ecnaive_native initialization and buffer management."""

import os
import queue
import threading
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
from dataclasses import replace

from .state_dict_decomposer import GlobalMetadataRegistry, TensorMetadata

logger = getLogger(__name__)


class ECNAIVEManager:
    """Shared manager for EC-NAIVE C++ module initialization and buffer management.
    
    This class provides a singleton instance that manages:
    - EC-NAIVE C++ native module (_ecnaive_native)
    - Buffer allocation and management (data and parity buffers, pooled)
    - Buffer poller thread for releasing buffers
    
    Note: The 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1) are allocated
    in strategy after metadata exchange, not in manager.
    
    Both TorchDistSaveShardedStrategy and TorchDistLoadShardedStrategy
    can share the same manager instance to reuse initialized resources.
    """
    
    _instance: Optional['ECNAIVEManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the manager (only once due to singleton)."""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._ecnaive_native = None
        self.use_ecnaive = False
        
        # Buffer configuration
        self.ecnaive_data_buffers_count = 12
        self.ecnaive_parity_buffers_count = 12  # Pooled parity buffers
        self.ecnaive_buffer_size = 64 * 1024 * 1024  # 64MB
        self.ecnaive_pin_memory = True
        
        # Buffers (simplified: only data and parity pools)
        self.ecnaive_data_buffers: Optional[List[torch.Tensor]] = None
        self.ecnaive_parity_buffers: Optional[List[torch.Tensor]] = None  # Pooled parity buffers
        # Note: The 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1) are allocated in strategy
        
        # Free buffer queues (simplified)
        self._free_data_buffer_queue: Optional[queue.Queue] = None
        self._free_parity_buffer_queue: Optional[queue.Queue] = None
        
        # Buffer poller thread
        self._buffer_poller_thread: Optional[threading.Thread] = None
        self._buffer_poller_stop_event: Optional[threading.Event] = None
        self._buffer_poller_active_event: Optional[threading.Event] = None
        
        self._initialized = True

    def _get_round_robin_ranks(self, rank: int, world_size: int) -> dict:
        """Calculate round-robin partner ranks for EC-NAIVE.
        
        Round-robin distribution:
        - Rank i sends d_{i1} to rank (i+1) % world_size
        - Rank i sends p_{i0} to rank (i+2) % world_size
        - Rank i sends p_{i1} to rank (i+3) % world_size
        - Rank i receives p_{(i+1)%world_size,1} from rank (i+1) % world_size
        - Rank i receives p_{(i+2)%world_size,0} from rank (i+2) % world_size
        - Rank i receives d_{(i+3)%world_size,1} from rank (i+3) % world_size
        
        Args:
            rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            dict: Partner ranks for each connection with keys:
                - 'send_data1_to': int - Rank to send d_{i1} to
                - 'send_parity0_to': int - Rank to send p_{i0} to
                - 'send_parity1_to': int - Rank to send p_{i1} to
                - 'recv_parity1_from': int - Rank to receive p_{(i+1),1} from
                - 'recv_parity0_from': int - Rank to receive p_{(i+2),0} from
                - 'recv_data1_from': int - Rank to receive d_{(i+3),1} from
        """
        return {
            'send_data1_to': (rank + 1) % world_size,
            'send_parity0_to': (rank + 2) % world_size,
            'send_parity1_to': (rank + 3) % world_size,
            'recv_parity1_from': (rank + 1) % world_size,
            'recv_parity0_from': (rank + 2) % world_size,
            'recv_data1_from': (rank + 3) % world_size,
        }

    def _get_ecnaive_network_config(self, rank: int, world_size: int) -> dict:
        """
        Get network configuration for EC-NAIVE ASIO connections.
        
        This function:
        1. Gets base IP address (from ECNAIVE_BASE_IP env var, MASTER_ADDR, or auto-detect)
        2. Calculates ports for this rank (base_port + rank * 6 + offset)
        3. Exchanges IP addresses with all ranks via torch.distributed.all_gather
        4. Returns configuration dictionary
        
        Args:
            rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            dict: Network configuration with keys:
                - 'my_ip': str - This rank's IP address
                - 'base_port': int - Base port number
                - 'ports': dict - Port numbers for each connection type
                    - 'send_data1': int
                    - 'send_parity0': int
                    - 'send_parity1': int
                    - 'recv_parity1': int
                    - 'recv_parity0': int
                    - 'recv_data1': int
        """
        import socket
        
        # Step 1: Get base IP address
        # Priority: ECNAIVE_BASE_IP > ECNAIVE_INTERFACE > auto-detect > MASTER_ADDR (fallback)
        base_ip = os.environ.get('ECNAIVE_BASE_IP')
        
        # If ECNAIVE_BASE_IP is not set, try to auto-detect actual IP
        if not base_ip:
            # Check if specific network interface is requested
            interface_name = os.environ.get('ECNAIVE_INTERFACE')
            
            if interface_name:
                try:
                    import netifaces
                    addrs = netifaces.ifaddresses(interface_name)
                    if netifaces.AF_INET in addrs:
                        base_ip = addrs[netifaces.AF_INET][0]['addr']
                        logger.info(f"EC-NAIVE: Using IP from interface {interface_name}: {base_ip}")
                    else:
                        logger.warning(f"EC-NAIVE: Interface {interface_name} has no IPv4 address")
                        base_ip = None
                except ImportError:
                    logger.warning(
                        "EC-NAIVE: netifaces module not installed. "
                        "Install via 'pip install netifaces' to use ECNAIVE_INTERFACE. "
                        "Falling back to auto-detection."
                    )
                    base_ip = None
                except Exception as e:
                    logger.warning(f"EC-NAIVE: Failed to get IP from interface {interface_name}: {e}")
                    base_ip = None
            
            if not base_ip:
                try:
                    # Get IP of the interface used for distributed training
                    # Connect to a remote address (doesn't actually send data)
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    # Use a public DNS server IP to determine the default route interface
                    s.connect(('8.8.8.8', 80))
                    base_ip = s.getsockname()[0]
                    s.close()
                    logger.info(f"EC-NAIVE: Auto-detected IP address: {base_ip}")
                except Exception as e:
                    logger.warning(f"EC-NAIVE: Failed to auto-detect IP: {e}")
                    # Fallback to MASTER_ADDR or localhost
                    base_ip = os.environ.get('MASTER_ADDR', '127.0.0.1')
                    logger.warning(f"EC-NAIVE: Using fallback IP: {base_ip}")
        
        # Step 2: Get base port
        # Priority: ECNAIVE_BASE_PORT > MASTER_PORT + 10000 > default 16000
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(os.environ.get('ECNAIVE_BASE_PORT', master_port + 10000))
        
        # Step 3: Calculate ports for this rank
        # Port allocation: base_port + rank * 6 + offset
        # offset: 0=send_data1, 1=send_parity0, 2=send_parity1,
        #         3=recv_parity1, 4=recv_parity0, 5=recv_data1
        ports = {
            'send_data1': base_port + rank * 6 + 0,
            'send_parity0': base_port + rank * 6 + 1,
            'send_parity1': base_port + rank * 6 + 2,
            'recv_parity1': base_port + rank * 6 + 3,
            'recv_parity0': base_port + rank * 6 + 4,
            'recv_data1': base_port + rank * 6 + 5,
        }
        
        # Step 4: Exchange IP addresses via torch.distributed.all_gather
        rank_ips = {}
        
        if torch.distributed.is_initialized():
            try:
                # Convert IP to bytes, then to int list for tensor
                my_ip_bytes = socket.inet_aton(base_ip)
                my_ip_tensor = torch.tensor(
                    [int(b) for b in my_ip_bytes], 
                    dtype=torch.uint8
                )
                
                # Move to CUDA if available (for NCCL backend compatibility)
                if torch.cuda.is_available():
                    my_ip_tensor = my_ip_tensor.cuda()
                
                # Gather all IPs
                ip_list = [torch.zeros_like(my_ip_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(ip_list, my_ip_tensor)
                
                # Convert back to IP strings
                for r, ip_tensor in enumerate(ip_list):
                    ip_bytes = bytes(ip_tensor.cpu().tolist())
                    rank_ips[r] = socket.inet_ntoa(ip_bytes)
                
                logger.info(
                    f"EC-NAIVE: [Rank {rank}] IP exchange completed - "
                    f"All rank IPs: {rank_ips}"
                )
            except Exception as e:
                logger.warning(
                    f"EC-NAIVE: Failed to exchange IPs via all_gather, using local IP: {e}"
                )
                # Fallback to using local IP for all ranks
                for r in range(world_size):
                    rank_ips[r] = base_ip
        else:
            # Single rank mode - use local IP
            logger.info("EC-NAIVE: Distributed not initialized, using local IP for all ranks")
            rank_ips[0] = base_ip
        
        config = {
            'my_ip': base_ip,
            'base_port': base_port,
            'rank_ips': rank_ips,
            'ports': ports,
        }
        
        logger.info(
            f"EC-NAIVE: [Rank {rank}] Network config:\n"
            f"  My IP: {config['my_ip']}\n"
            f"  Base port: {config['base_port']}\n"
            f"  Ports: {config['ports']}\n"
            f"  All rank IPs: {config['rank_ips']}"
        )
        
        return config
    
    def init_ecnaive_if_enabled(self):
        """Initialize EC-NAIVE C++ module if enabled and distributed environment is ready."""
        if self._ecnaive_native is not None:
            logger.debug("EC-NAIVE: Already initialized, skipping")
            return
        
        try:
            from megatron.training import get_args as input_args
            args = input_args()
            self.use_ecnaive = args.use_ecnaive
            if not getattr(args, 'use_ecnaive', False):
                return
                
            # Check if distributed environment is initialized
            if not torch.distributed.is_initialized():
                logger.warning("EC-NAIVE: Distributed environment not initialized, skipping EC-NAIVE initialization")
                return
                
            # Initialize EC-NAIVE C++ module
            self._init_ecnaive_native()
            
            # Start persistent buffer poller thread
            # EC-NAIVE does not use layerwise mode
            self._start_buffer_poller_thread()
            
        except Exception as e:
            logger.warning(f"EC-NAIVE: Failed to initialize during manager initialization: {e}")
            self._ecnaive_native = None
    
    def _init_ecnaive_native(self):
        """Initialize EC-NAIVE C++ native module."""
        ecnaive_native = None
        try:
            # Direct import .so file without modifying sys.path or affecting other packages
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Find .so file
            import glob as _glob_module
            so_files = _glob_module.glob(os.path.join(current_dir, "ecnaive_native*.so"))
            
            if not so_files:
                raise ImportError(f"No ecnaive_native.so file found in {current_dir}")
            
            # Load .so file directly using importlib
            import importlib.util as _importlib_util
            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location("ecnaive_native", so_path)
            ecnaive_native = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(ecnaive_native)
            logger.debug(f"EC-NAIVE: Loaded .so file from {so_path}")
            
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            
            # EC-NAIVE only uses ASIO (no NCCL support)
            # Create instance with error handling
            try:
                # ===== ASIO Initialization Path =====
                logger.info(f"EC-NAIVE: [Rank {rank}] Using ASIO for communication")
                
                # Get network configuration
                net_config = self._get_ecnaive_network_config(rank, world_size)
                
                # Synchronize all ranks before creating C++ instances
                logger.info(f"EC-NAIVE: [Rank {rank}] Synchronizing all ranks before creating C++ native module (ASIO)...")
                torch.distributed.barrier()
                logger.info(f"EC-NAIVE: [Rank {rank}] All ranks synchronized, creating C++ native module with ASIO...")
                
                # Create C++ instance with ASIO parameters
                logger.info(f"EC-NAIVE: Creating C++ native module with ASIO (this will block until ASIO connections are established)...")
                print(f"EC-NAIVE: [Rank {rank}] Creating C++ native module with ASIO (blocking until ASIO initialization completes)...")
                
                # EC-NAIVE requires 12 parameters (6 pairs of ip:port):
                # send_data1_ip, send_data1_port, send_parity0_ip, send_parity0_port, 
                # send_parity1_ip, send_parity1_port, recv_parity1_ip, recv_parity1_port,
                # recv_parity0_ip, recv_parity0_port, recv_data1_ip, recv_data1_port
                
                # Calculate round-robin partner ranks
                partner_ranks = self._get_round_robin_ranks(rank, world_size)
                
                base_port = net_config['base_port']
                rank_ips = net_config['rank_ips']
                
                # Calculate partner ports (send connects to partner's recv port)
                # Port allocation: base_port + rank * 6 + offset
                # For send connections, we connect to the partner's recv port:
                # - send_data1 connects to partner's recv_parity1 (offset 3)
                # - send_parity0 connects to partner's recv_parity0 (offset 4)
                # - send_parity1 connects to partner's recv_data1 (offset 5)
                send_data1_partner_port = base_port + partner_ranks['send_data1_to'] * 6 + 3  # recv_parity1
                send_parity0_partner_port = base_port + partner_ranks['send_parity0_to'] * 6 + 4  # recv_parity0
                send_parity1_partner_port = base_port + partner_ranks['send_parity1_to'] * 6 + 5  # recv_data1
                
                # Create C++ instance with ASIO parameters (12 parameters: 6 pairs of ip:port)
                self._ecnaive_native = ecnaive_native.ECNaiveNative(
                    # Send connections
                    rank_ips.get(partner_ranks['send_data1_to'], net_config['my_ip']), send_data1_partner_port,
                    rank_ips.get(partner_ranks['send_parity0_to'], net_config['my_ip']), send_parity0_partner_port,
                    rank_ips.get(partner_ranks['send_parity1_to'], net_config['my_ip']), send_parity1_partner_port,
                    # Recv connections (listen on local IP)
                    net_config['my_ip'], net_config['ports']['recv_parity1'],
                    net_config['my_ip'], net_config['ports']['recv_parity0'],
                    net_config['my_ip'], net_config['ports']['recv_data1'],
                )
                
                # If we reach here, ASIO connections are ready and threads are running
                logger.info(f"EC-NAIVE: C++ native module initialized successfully with ASIO (rank={rank}, world_size={world_size})")
                print(f"EC-NAIVE: [Rank {rank}] C++ native module initialized - ASIO connections ready for data exchange")
                
                # Initialize EC-NAIVE buffers
                # EC-NAIVE does not use layerwise mode
                self._init_ecnaive_buffers()
        
            except Exception as e:
                logger.warning(f"EC-NAIVE: Failed to create C++ native module instance: {e}")
                # Try to stop the pipeline if it was partially created
                try:
                    if hasattr(self, '_ecnaive_native') and self._ecnaive_native is not None:
                        self._ecnaive_native.stop()
                except:
                    pass
                self._ecnaive_native = None
                raise e
            
        except ImportError as e:
            logger.warning(f"EC-NAIVE: C++ native module not available: {e}, EC-NAIVE functionality will not work")
            self._ecnaive_native = None
        except Exception as e:
            logger.warning(f"EC-NAIVE: Failed to initialize C++ native module: {e}, EC-NAIVE functionality will not work")
            self._ecnaive_native = None
    
    def _init_ecnaive_buffers(self):
        """Initialize EC-NAIVE buffers during C++ module initialization.
        
        Note: Only allocates data and parity buffers (pooled) at initialization.
        The 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1) will be allocated
        in strategy after metadata exchange.
        """
        rank = torch.distributed.get_rank()
        logger.info("EC-NAIVE: Initializing buffers for EC-NAIVE (data and parity pools only)")
        print(f"EC-NAIVE: Initializing buffers for EC-NAIVE (rank={rank}, data and parity pools only)")
        
        # Allocate data buffers for storing original tensor data
        self.ecnaive_data_buffers = self._allocate_data_buffers()
        
        # Allocate parity buffers (pooled) for parity blocks
        self.ecnaive_parity_buffers = self._allocate_parity_buffers()
        
        # Initialize free buffer queues
        self._free_data_buffer_queue = queue.Queue()
        for buffer in self.ecnaive_data_buffers:
            self._free_data_buffer_queue.put(int(buffer.data_ptr()))
        
        self._free_parity_buffer_queue = queue.Queue()
        for buffer in self.ecnaive_parity_buffers:
            self._free_parity_buffer_queue.put(int(buffer.data_ptr()))

        logger.info(f"EC-NAIVE: Buffer initialization completed - "
                   f"Data buffers: {len(self.ecnaive_data_buffers)}, "
                   f"Parity buffers: {len(self.ecnaive_parity_buffers)}")
        print(f"EC-NAIVE: Buffer initialization completed (rank={rank}) - "
              f"Data buffers: {len(self.ecnaive_data_buffers)}, "
              f"Parity buffers: {len(self.ecnaive_parity_buffers)}")
    
    def _allocate_data_buffers(self):
        """Allocate data buffers for storing original tensor data."""
        logger.info(f"EC-NAIVE: Allocating data buffers ({self.ecnaive_data_buffers_count} buffers, {self.ecnaive_buffer_size // (1024*1024)}MB each)")
        
        data_buffers = []
        for i in range(self.ecnaive_data_buffers_count):
            buffer = torch.empty(self.ecnaive_buffer_size, dtype=torch.uint8, pin_memory=self.ecnaive_pin_memory)
            data_buffers.append(buffer)
            logger.debug(f"EC-NAIVE: Allocated data buffer {i}: {self.ecnaive_buffer_size} bytes")
        
        logger.info(f"EC-NAIVE: Allocated {len(data_buffers)} data buffers")
        return data_buffers
    
    def _allocate_parity_buffers(self):
        """Allocate parity buffers (pooled) for parity blocks."""
        logger.info(f"EC-NAIVE: Allocating parity buffers ({self.ecnaive_parity_buffers_count} buffers, {self.ecnaive_buffer_size // (1024*1024)}MB each)")
        
        parity_buffers = []
        for i in range(self.ecnaive_parity_buffers_count):
            buffer = torch.empty(self.ecnaive_buffer_size, dtype=torch.uint8, pin_memory=self.ecnaive_pin_memory)
            parity_buffers.append(buffer)
            logger.debug(f"EC-NAIVE: Allocated parity buffer {i}: {self.ecnaive_buffer_size} bytes")
        
        logger.info(f"EC-NAIVE: Allocated {len(parity_buffers)} parity buffers")
        return parity_buffers
    
    
    def _poll_and_release_buffers(self):
        """Poll C++ for buffers ready to be released and put them back to queues."""
        if self._ecnaive_native is None:
            return
        
        # Get data buffers ready for release
        data_buffers = self._ecnaive_native.get_data_buffers_to_release()
        for data_addr in data_buffers:
            try:
                self._free_data_buffer_queue.put_nowait(data_addr)
                # logger.debug(f"EC-NAIVE: Released data buffer at address {data_addr}")
            except Exception:
                logger.error(f"EC-NAIVE: Data buffer queue is full, cannot release buffer {data_addr}")
        
        # Get parity buffers ready for release
        parity_buffers = self._ecnaive_native.get_parity_buffers_to_release()
        for parity_addr in parity_buffers:
            try:
                self._free_parity_buffer_queue.put_nowait(parity_addr)
                # logger.debug(f"EC-NAIVE: Released parity buffer at address {parity_addr}")
            except Exception:
                logger.error(f"EC-NAIVE: Parity buffer queue is full, cannot release buffer {parity_addr}")
    
    def _start_buffer_poller_thread(self):
        """Start a persistent background thread to poll and release buffers."""
        if hasattr(self, '_buffer_poller_thread') and self._buffer_poller_thread is not None:
            logger.warning("EC-NAIVE: Buffer poller thread already started")
            return
        
        # Create control events
        self._buffer_poller_stop_event = threading.Event()
        self._buffer_poller_active_event = threading.Event()
        
        def buffer_poller_worker():
            """Persistent background thread that polls for buffer releases."""
            logger.info("EC-NAIVE: Buffer poller thread started")
            poll_count = 0
            
            while not self._buffer_poller_stop_event.is_set():
                # Only poll when active
                if self._buffer_poller_active_event.is_set():
                    self._poll_and_release_buffers()
                    poll_count += 1
                    if poll_count % 1000 == 0:
                        logger.debug(f"EC-NAIVE: Buffer poller running (polled {poll_count} times)")
                
                # Sleep briefly to avoid busy waiting
                from time import sleep
                sleep(0.001)  # 1ms
            
            logger.info("EC-NAIVE: Buffer poller thread stopping")
        
        # Start the daemon thread
        self._buffer_poller_thread = threading.Thread(target=buffer_poller_worker, daemon=True)
        self._buffer_poller_thread.start()
        logger.info("EC-NAIVE: Buffer poller thread created and started")
    
    def _stop_buffer_poller_thread(self):
        """Stop the persistent buffer poller thread."""
        if not hasattr(self, '_buffer_poller_thread') or self._buffer_poller_thread is None:
            return
        
        logger.info("EC-NAIVE: Stopping buffer poller thread...")
        
        # Signal the thread to stop
        if self._buffer_poller_stop_event:
            self._buffer_poller_stop_event.set()
        
        # Wait for thread to finish
        if self._buffer_poller_thread.is_alive():
            self._buffer_poller_thread.join(timeout=2.0)
            if self._buffer_poller_thread.is_alive():
                logger.warning("EC-NAIVE: Buffer poller thread did not stop in time")
            else:
                logger.info("EC-NAIVE: Buffer poller thread stopped successfully")
        
        self._buffer_poller_thread = None
        self._buffer_poller_stop_event = None
        self._buffer_poller_active_event = None
    
    def get_ecnaive_buffers(self):
        """Get EC-NAIVE buffers for FileSystemWriterAsync.
        
        Note: Returns data and parity buffers (pooled).
        The 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1) will be allocated
        in strategy after metadata exchange.
        
        Returns:
            Dict containing all buffer information, or None if not initialized
        """
        if self.ecnaive_data_buffers is None:
            return None
        
        return {
            'data_buffers': self.ecnaive_data_buffers,
            'parity_buffers': self.ecnaive_parity_buffers,
            'free_data_buffer_queue': self._free_data_buffer_queue,
            'free_parity_buffer_queue': self._free_parity_buffer_queue,
            # Pass buffer poller control objects
            'buffer_poller_active_event': self._buffer_poller_active_event,
            'poll_and_release_buffers': self._poll_and_release_buffers,
            # Note: The 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1) 
            # will be allocated in strategy after metadata exchange
        }
    
    def cleanup(self):
        """Cleanup EC-NAIVE resources when manager is destroyed."""
        try:
            # Stop buffer poller thread
            self._stop_buffer_poller_thread()
            
            # Stop the C++ pipeline
            if hasattr(self, '_ecnaive_native') and self._ecnaive_native is not None:
                self._ecnaive_native.stop()
                logger.info("EC-NAIVE: C++ native module stopped in manager cleanup")
                
        except Exception as e:
            logger.warning(f"EC-NAIVE: Error during manager cleanup: {e}")
    
    def __del__(self):
        """Cleanup EC-NAIVE resources when manager is destroyed."""
        self.cleanup()

