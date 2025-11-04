import ctypes
import numpy as np
import os
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Helper: query CUDA runtime errors via libcudart (no need to recompile C code)
def _check_cuda_runtime_error(prefix="CUDA runtime check"):
    """Call cudaDeviceSynchronize and cudaGetLastError via libcudart and print any error string.

    Returns True if no error, False if an error was found (and printed).
    """
    try:
        from ctypes.util import find_library
        libname = find_library('cudart') or 'libcudart.so'
        libcudart = ctypes.CDLL(libname)
    except Exception as e:
        print(f"{prefix}: failed to load libcudart: {e}")
        return True

    # int cudaDeviceSynchronize(void)
    try:
        libcudart.cudaDeviceSynchronize.restype = ctypes.c_int
        rc = libcudart.cudaDeviceSynchronize()
    except Exception as e:
        print(f"{prefix}: cudaDeviceSynchronize call failed: {e}")
        rc = 0

    try:
        libcudart.cudaGetLastError.restype = ctypes.c_int
        err = libcudart.cudaGetLastError()
        if err != 0:
            libcudart.cudaGetErrorString.restype = ctypes.c_char_p
            msg = libcudart.cudaGetErrorString(err)
            try:
                s = msg.decode() if isinstance(msg, bytes) else str(msg)
            except Exception:
                s = str(msg)
            print(f"{prefix}: cuda error {err}: {s}")
            return False
    except Exception as e:
        print(f"{prefix}: cudaGetLastError failed: {e}")

    return True

class GCRSPCIEWrapper:
    """
    Python wrapper for G-CRSPCIE GPU Erasure Code library.
    Supports encoding data blocks into coding blocks using GPU acceleration.
    Handles both CPU (numpy) and GPU (torch.Tensor) data.
    """

    def __init__(self, lib_path=None):
        # Debug flag: enable by passing debug=True or setting env GCRS_DEBUG=1
        debug_env = os.environ.get("GCRS_DEBUG", None)
        if debug_env is not None:
            try:
                self.debug = bool(int(debug_env))
            except Exception:
                self.debug = debug_env.lower() in ("1", "true", "yes")
        else:
            self.debug = False

        """
        Initialize the wrapper by loading the shared library.

        Args:
            lib_path (str, optional): Path to libgcrspcie.so. Defaults to relative path.
        """
        if lib_path is None:
            lib_path = os.path.join(os.path.dirname(__file__), "../../G-CRSPCIE/libgcrspcie.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found: {lib_path}")
        try:
            self.lib = ctypes.cdll.LoadLibrary(lib_path)
        except OSError as e:
            raise OSError(f"Failed to load library: {e}")

        # Define function signatures
        self._setup_function_signatures()
        # Cached worker and parameters for reuse (initialized per-wrapper)
        self._worker = None
        self._worker_params = None  # tuple (k, m, w, whole_buf_size, task_size)

    def _dbg(self, *args, **kwargs):
        """Print debug messages only when debug flag is set."""
        if getattr(self, 'debug', False):
            print(*args, **kwargs)

    def _setup_function_signatures(self):
        """Set up ctypes function signatures for the C library functions."""

                # PErasureWorkerInit
        self.lib.PErasureWorkerInit.argtypes = [
            ctypes.c_size_t,  # k
            ctypes.c_size_t,  # m
            ctypes.c_size_t,  # w
            ctypes.c_size_t,  # wholeBufSize
            ctypes.c_size_t   # taskSize
        ]
        self.lib.PErasureWorkerInit.restype = ctypes.POINTER(ctypes.c_void_p)

        # fullDuplexRunEncode
        self.lib.fullDuplexRunEncode.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)  # struct PErasureWorker *
        ]
        self.lib.fullDuplexRunEncode.restype = None

        # PErasureWorkerDealloc
        self.lib.PErasureWorkerDealloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)  # struct PErasureWorker *
        ]
        self.lib.PErasureWorkerDealloc.restype = None

        # PErasureWorkerSetInputData
        self.lib.PErasureWorkerSetInputData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # struct PErasureWorker *
            ctypes.POINTER(ctypes.c_char),    # char *input_data
            ctypes.c_size_t                   # size_t data_size
        ]
        self.lib.PErasureWorkerSetInputData.restype = None

        # PErasureWorkerGetOutputData
        self.lib.PErasureWorkerGetOutputData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # struct PErasureWorker *
            ctypes.POINTER(ctypes.c_char),    # char *output_data
            ctypes.c_size_t                   # size_t data_size
        ]
        self.lib.PErasureWorkerGetOutputData.restype = None

        # PErasureWorkerGetOutputDevicePtr (new API for zero-copy GPU output)
        try:
            self.lib.PErasureWorkerGetOutputDevicePtr.argtypes = [
                ctypes.POINTER(ctypes.c_void_p)  # struct PErasureWorker *
            ]
            self.lib.PErasureWorkerGetOutputDevicePtr.restype = ctypes.c_void_p
            self._has_get_output_device_ptr = True
        except Exception:
            self._has_get_output_device_ptr = False

        # PErasureWorkerSetInputDevicePtr (optional zero-copy device pointer API)
        try:
            self.lib.PErasureWorkerSetInputDevicePtr.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),  # struct PErasureWorker *
                ctypes.c_void_p,                  # device pointer
                ctypes.c_size_t                   # size
            ]
            self.lib.PErasureWorkerSetInputDevicePtr.restype = None
            self._has_set_input_device_ptr = True
        except Exception:
            self._has_set_input_device_ptr = False

        # GCRSMCodingInit
        self.lib.GCRSMCodingInit.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # struct GCRSMCoding *
            ctypes.c_int,  # k
            ctypes.c_int,  # m
            ctypes.c_int   # w
        ]
        self.lib.GCRSMCodingInit.restype = None

        # GCRSMCodingSetBitmatrix
        self.lib.GCRSMCodingSetBitmatrix.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # struct GCRSMCoding *
            ctypes.POINTER(ctypes.c_int)      # int *bitmatrix
        ]
        self.lib.GCRSMCodingSetBitmatrix.restype = None

        # gcrs_create_bitmatrix
        self.lib.gcrs_create_bitmatrix.argtypes = [
            ctypes.c_int,  # k
            ctypes.c_int,  # m
            ctypes.c_int   # w
        ]
        self.lib.gcrs_create_bitmatrix.restype = ctypes.POINTER(ctypes.c_int)

        # PErasureWorkerResetDevice (explicit device reset)
        # void PErasureWorkerResetDevice(void)
        try:
            self.lib.PErasureWorkerResetDevice.argtypes = []
            self.lib.PErasureWorkerResetDevice.restype = None
        except Exception:
            # older libs may not expose this; it's okay
            pass

    def _prepare_data(self, data_blocks):
        """
        Prepare data for C library: convert to numpy uint8 if needed, flatten.

        Args:
            data_blocks: numpy.ndarray or torch.Tensor, shape (k, block_size)

        Returns:
            tuple: (flat_data, is_gpu, original_dtype)
        """
        if TORCH_AVAILABLE and isinstance(data_blocks, torch.Tensor):
            if data_blocks.dtype != torch.uint8:
                data_blocks = data_blocks.to(torch.uint8)
            flat_data = data_blocks.flatten()
            is_gpu = data_blocks.is_cuda
            original_dtype = data_blocks.dtype
            return flat_data, is_gpu, original_dtype
        elif isinstance(data_blocks, np.ndarray):
            if data_blocks.dtype != np.uint8:
                data_blocks = data_blocks.astype(np.uint8)
            flat_data = data_blocks.flatten()
            is_gpu = False
            original_dtype = data_blocks.dtype
            return flat_data, is_gpu, original_dtype
        else:
            raise ValueError("data_blocks must be numpy.ndarray or torch.Tensor")

    def encode(self, data_blocks, k, m, w=16, task_size=1, return_gpu=False, use_device_ptr=None):
        """
        Encode data blocks into coding blocks using G-CRSPCIE.

        Args:
            data_blocks (numpy.ndarray or torch.Tensor): Input data, shape (k, block_size), dtype uint8
            k (int): Number of data blocks
            m (int): Number of coding blocks
            w (int): Word size (default 8)
            task_size (int): Number of tasks (default 1)
            return_gpu (bool): If True and input is GPU tensor, return GPU tensor

        Returns:
            numpy.ndarray or torch.Tensor: Coding blocks, shape (m, block_size), dtype uint8
        """
        # Normalize and inspect input
        flat_data, is_gpu, original_dtype = self._prepare_data(data_blocks)
        self._dbg(f"Debug: input data_blocks shape {data_blocks.shape}, dtype {data_blocks.dtype if hasattr(data_blocks, 'dtype') else type(data_blocks)}")
        self._dbg(f"Debug: is_gpu {is_gpu}, original_dtype {original_dtype}")

        block_size = data_blocks.shape[1]
        whole_buf_size = k * block_size

        # Determine per-call device-pointer attempt behavior
        # Note: device_ptr optimization has issues, disabled by default for now
        # Default to False - can be enabled via env var or explicit parameter
        env_use = (os.environ.get('GCRS_USE_DEVICE_PTR', '0') == '1')
        lib_supports = getattr(self, '_has_set_input_device_ptr', False)
        if use_device_ptr is None:
            # Disable auto-enable for now due to stability issues
            attempt_device_ptr = False  # env_use and lib_supports and is_gpu
        else:
            attempt_device_ptr = bool(use_device_ptr) and lib_supports
        self._dbg(f"Device-pointer decision for this call: use_device_ptr_param={use_device_ptr}, env={env_use}, lib_supports={lib_supports}, is_gpu={is_gpu} -> attempt_device_ptr={attempt_device_ptr}")

        # Ensure worker exists (bitmatrix is already created in worker initialization)
        try:
            worker = self._ensure_worker(k, m, w, whole_buf_size, task_size)
        except Exception as e:
            raise RuntimeError(f"Failed to ensure PErasureWorker: {e}")

        # Prepare and set input data
        data_used_device_ptr = False
        if TORCH_AVAILABLE and isinstance(data_blocks, torch.Tensor) and data_blocks.is_cuda:
            # GPU tensor
            if attempt_device_ptr:
                device_ptr = ctypes.c_void_p(data_blocks.data_ptr())
                try:
                    self._dbg(f"Attempting device-pointer zero-copy path: device_ptr=0x{int(device_ptr.value):x}, nbytes={data_blocks.nbytes}")
                    self.lib.PErasureWorkerSetInputDevicePtr(worker, device_ptr, data_blocks.nbytes)
                    data_flat = None
                    data_used_device_ptr = True
                    self._dbg("Device-pointer path succeeded: using external device buffer in C library")
                except Exception as e:
                    self._dbg(f"Device-pointer path failed (falling back to pinned host copy): {e}")
                    try:
                        pinned = torch.empty_like(data_blocks, device='cpu', pin_memory=True)
                        pinned.copy_(data_blocks, non_blocking=True)
                        data_flat = pinned.flatten().numpy().astype(np.int8)
                    except Exception:
                        data_flat = data_blocks.flatten().cpu().numpy().astype(np.int8)
                    data_used_device_ptr = False
            else:
                try:
                    pinned = torch.empty_like(data_blocks, device='cpu', pin_memory=True)
                    pinned.copy_(data_blocks, non_blocking=True)
                    data_flat = pinned.flatten().numpy().astype(np.int8)
                except Exception:
                    data_flat = data_blocks.flatten().cpu().numpy().astype(np.int8)
                data_used_device_ptr = False
        elif TORCH_AVAILABLE and isinstance(data_blocks, torch.Tensor):
            # CPU tensor
            data_flat = data_blocks.flatten().cpu().numpy().astype(np.int8)
            data_used_device_ptr = False
        else:
            data_flat = data_blocks.flatten().astype(np.int8)
            data_used_device_ptr = False

        # If we used device pointer, skip SetInputData; otherwise pass host pointer
        if not data_used_device_ptr:
            if not isinstance(data_flat, np.ndarray):
                raise RuntimeError("Internal error: expected host data_flat numpy array when not using device pointer")
            self.lib.PErasureWorkerSetInputData(worker, data_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_char)), data_flat.nbytes)

        # Run encoding
        self._dbg("Running encoding...")
        try:
            self.lib.fullDuplexRunEncode(worker)
            self._dbg("Encoding completed")
            if os.environ.get("GCRS_CHECK_CUDA", "0") == "1":
                ok = _check_cuda_runtime_error(prefix="After fullDuplexRunEncode")
                if not ok:
                    raise RuntimeError("CUDA runtime error detected after fullDuplexRunEncode; see stderr for details")
        except Exception as e:
            raise RuntimeError(f"Failed to run encoding: {e}")

        # Get output coding blocks - optimize for GPU zero-copy path
        if is_gpu and return_gpu and TORCH_AVAILABLE and getattr(self, '_has_get_output_device_ptr', False):
            # Zero-copy GPU path: directly copy from C library's GPU buffer to PyTorch tensor
            self._dbg("Using zero-copy GPU output path")
            try:
                # Ensure encoding is complete before accessing output buffer
                if TORCH_AVAILABLE:
                    torch.cuda.synchronize()
                
                # Get device pointer from C library
                output_dev_ptr = self.lib.PErasureWorkerGetOutputDevicePtr(worker)
                if output_dev_ptr is None or output_dev_ptr == 0:
                    raise RuntimeError("Failed to get output device pointer")
                
                # Create output tensor directly on GPU
                coding_blocks_gpu = torch.empty((m, block_size), dtype=original_dtype, device='cuda')
                
                # Copy from C library's GPU buffer to PyTorch tensor (device-to-device copy)
                from ctypes.util import find_library
                libname = find_library('cudart') or 'libcudart.so'
                libcudart = ctypes.CDLL(libname)
                libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                libcudart.cudaMemcpy.restype = ctypes.c_int
                
                # Calculate output size in bytes (C library always uses int8/uint8 internally)
                output_size_bytes = m * block_size  # C library uses int8, so 1 byte per element
                
                # Use cudaMemcpyDeviceToDevice (2) for GPU-to-GPU copy
                cudaMemcpyDeviceToDevice = 2
                err = libcudart.cudaMemcpy(
                    ctypes.c_void_p(coding_blocks_gpu.data_ptr()),
                    ctypes.c_void_p(output_dev_ptr),
                    output_size_bytes,
                    cudaMemcpyDeviceToDevice
                )
                if err != 0:
                    raise RuntimeError(f"cudaMemcpy failed with error {err}")
                
                torch.cuda.synchronize()
                coding_blocks = coding_blocks_gpu
                self._dbg(f"Zero-copy GPU output succeeded: shape {coding_blocks.shape}, dtype {coding_blocks.dtype}")
            except Exception as e:
                self._dbg(f"Zero-copy GPU output failed (falling back to host path): {e}")
                # Fallback to host path
                coding_flat = np.zeros(m * block_size, dtype=np.int8)
                self.lib.PErasureWorkerGetOutputData(worker, coding_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_char)), coding_flat.nbytes)
                coding_blocks = coding_flat.reshape((m, block_size))
                coding_blocks_uint8 = np.ascontiguousarray(coding_blocks.astype(np.uint8))
                coding_blocks = torch.from_numpy(coding_blocks_uint8.copy()).to(device='cuda', dtype=original_dtype)
                torch.cuda.synchronize()
        else:
            # Host path: get output from CPU buffer
            coding_flat = np.zeros(m * block_size, dtype=np.int8)
            self.lib.PErasureWorkerGetOutputData(worker, coding_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_char)), coding_flat.nbytes)
            coding_blocks = coding_flat.reshape((m, block_size))
            
            if is_gpu and return_gpu and TORCH_AVAILABLE:
                # Convert to GPU tensor if requested
                self._dbg(f"Converting to torch tensor with dtype {original_dtype}")
                coding_blocks_uint8 = np.ascontiguousarray(coding_blocks.astype(np.uint8))
                try:
                    tensor_gpu = torch.tensor(coding_blocks_uint8, device='cuda', dtype=original_dtype)
                    torch.cuda.synchronize()
                    coding_blocks = tensor_gpu
                except Exception as e:
                    self._dbg(f"Error during direct GPU tensor creation: {e}")
                    tensor_cpu = torch.from_numpy(coding_blocks_uint8.copy())
                    try:
                        coding_blocks = tensor_cpu.to(device='cuda', dtype=original_dtype)
                        torch.cuda.synchronize()
                    except Exception as ee:
                        self._dbg(f"Fallback move to CUDA failed: {ee}")
                        raise
        
        # Convert dtype if needed (for non-GPU outputs)
        if isinstance(coding_blocks, np.ndarray) and original_dtype != np.int8:
            try:
                if TORCH_AVAILABLE and isinstance(original_dtype, type(torch.uint8)):
                    torch_to_np = {
                        getattr(torch, 'uint8', None): np.uint8,
                        getattr(torch, 'int8', None): np.int8,
                        getattr(torch, 'int16', None): np.int16,
                        getattr(torch, 'int32', None): np.int32,
                        getattr(torch, 'int64', None): np.int64,
                        getattr(torch, 'float32', None): np.float32,
                        getattr(torch, 'float64', None): np.float64,
                    }
                    npdtype = torch_to_np.get(original_dtype, None)
                    if npdtype is not None:
                        coding_blocks = coding_blocks.astype(npdtype)
                    else:
                        try:
                            coding_blocks = coding_blocks.astype(np.dtype(str(original_dtype)))
                        except Exception:
                            pass
                else:
                    coding_blocks = coding_blocks.astype(original_dtype)
            except Exception:
                pass
        
        self._dbg(f"Debug: coding_blocks shape {coding_blocks.shape}, dtype {coding_blocks.dtype}")
        try:
            if isinstance(coding_blocks, np.ndarray):
                self._dbg(f"Debug: coding_blocks min {coding_blocks.min()}, max {coding_blocks.max()}")
                self._dbg(f"Debug: first few values {coding_blocks.flatten()[:10]}")
        except Exception:
            pass

        # Note: Do NOT deallocate worker here; we reuse the cached worker across
        # multiple encode() calls. Deallocation will be handled in close()/__del__.

        return coding_blocks

    def _ensure_worker(self, k, m, w, whole_buf_size, task_size):
        """
        Ensure a PErasureWorker exists with compatible parameters. If a cached
        worker exists and its parameters match the requested ones, return it.
        Otherwise create a new worker and cache it.
        """
        params = (k, m, w, whole_buf_size, task_size)
        if self._worker is not None and self._worker_params == params:
            return self._worker

        # If an existing worker exists but parameters differ, dealloc it first
        if self._worker is not None and self._worker_params is not None:
            try:
                self.lib.PErasureWorkerDealloc(self._worker)
            except Exception:
                # best-effort dealloc
                pass
            self._worker = None
            self._worker_params = None

        # Create new worker (bitmatrix is already created inside PErasureWorkerInit)
        worker = self.lib.PErasureWorkerInit(k, m, w, whole_buf_size, task_size)
        if not worker:
            raise RuntimeError("Failed to allocate PErasureWorker")
        self._dbg(f"Worker initialized (cached): {worker}")

        # Note: bitmatrix is already created and set inside PErasureWorkerInit
        # No need to create it again here

        self._worker = worker
        self._worker_params = params
        return worker

    def close(self):
        """Free cached resources (worker)."""
        if getattr(self, '_worker', None) is not None:
            try:
                self._dbg("Closing: deallocating cached worker")
                self.lib.PErasureWorkerDealloc(self._worker)
            except Exception as e:
                self._dbg(f"Error deallocating worker in close(): {e}")
            finally:
                self._worker = None
                self._worker_params = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def reset_device(self):
        """
        Explicitly reset the CUDA device by calling into the native library's
        PErasureWorkerResetDevice() function if available. This is optional and
        should only be used when you want to release CUDA resources globally.
        """
        if not hasattr(self.lib, 'PErasureWorkerResetDevice'):
            self._dbg("PErasureWorkerResetDevice not available in the loaded library")
            return
        try:
            self.lib.PErasureWorkerResetDevice()
            self._dbg("Called PErasureWorkerResetDevice()")
        except Exception as e:
            raise RuntimeError(f"Failed to call PErasureWorkerResetDevice: {e}")

    def decode(self, data_blocks, coding_blocks, k, m, w=8, task_size=1, erasure_ids=None, return_gpu=False):
        """
        Decode data blocks from available data and coding blocks.

        Args:
            data_blocks (numpy.ndarray or torch.Tensor): Available data blocks, shape (k, block_size)
            coding_blocks (numpy.ndarray or torch.Tensor): Coding blocks, shape (m, block_size)
            k, m, w, task_size: Same as encode
            erasure_ids (list): List of erased block indices
            return_gpu (bool): Same as encode

        Returns:
            numpy.ndarray or torch.Tensor: Recovered data blocks
        """
        # Placeholder for decode implementation
        raise NotImplementedError("Decode not implemented yet")

# Example usage
if __name__ == "__main__":
    # CPU example
    k = 4
    m = 2
    block_size = 1024
    data_cpu = np.random.randint(0, 256, (k, block_size), dtype=np.uint8)

    wrapper = GCRSPCIEWrapper()
    coding_cpu = wrapper.encode(data_cpu, k, m)
    print(f"CPU: Data shape {data_cpu.shape}, Coding shape {coding_cpu.shape}")

    # GPU example (if torch available)
    if TORCH_AVAILABLE:
        data_gpu = torch.randint(0, 256, (k, block_size), dtype=torch.uint8, device='cuda')
        coding_gpu = wrapper.encode(data_gpu, k, m, return_gpu=True)
        print(f"GPU: Data shape {data_gpu.shape}, Coding shape {coding_gpu.shape}, on {coding_gpu.device}")
