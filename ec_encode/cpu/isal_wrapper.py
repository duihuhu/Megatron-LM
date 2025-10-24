"""isal_wrapper.py (tensor-first)

一个面向 ISA-L 的 Python 封装：
- 优先支持 torch.Tensor（CPU、dtype=uint8、C contiguous）
- 兼容 numpy.ndarray（dtype=uint8、C contiguous）
- 提供 gftbls 的预计算与复用接口，避免在热路径中反复初始化
- 可选择调用优化路径或 base 路径（若符号存在）
"""

import ctypes
import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - 环境可能没有 torch
    torch = None  # type: ignore
    _HAS_TORCH = False


TensorOrNd = Any  # 避免在无 torch 时的类型检查问题，运行时仍做严格校验


class _ECTables:
    """持有 ec_init_tables 的产物，保证底层内存不被 GC 回收。"""

    def __init__(self, k: int, rows: int, a_buf: np.ndarray, gftbls_ctypes: Any):
        self.k = k
        self.rows = rows
        self._a_np = a_buf  # np.uint8 扁平 a（保持生命周期）
        self._gftbls = gftbls_ctypes  # ctypes.c_ubyte * (32*k*rows)

    @property
    def gftbls_ptr(self) -> Any:
        return ctypes.cast(self._gftbls, ctypes.POINTER(ctypes.c_ubyte))


class ISAL_Lib:
    """ISA-L 的 ctypes 封装（Tensor 优先）。"""

    def __init__(self, lib_path: Optional[str] = None, prefer_base: bool = False):
        self.lib = self._load_library(lib_path)
        self._define_prototypes()
        self.prefer_base = prefer_base

    # -------------------------
    # 动态库加载与函数原型
    # -------------------------
    def _load_library(self, lib_path: Optional[str]):
        if lib_path:
            return ctypes.CDLL(lib_path)

        if os.name == "posix":
            lib_names = ["libisal.so.2", "libisal.so"]
        elif os.name == "nt":
            lib_names = ["isal.dll"]
        else:
            raise NotImplementedError(f"不支持的操作系统: {os.name}")

        last_err: Optional[BaseException] = None
        for name in lib_names:
            try:
                return ctypes.CDLL(name)
            except OSError as e:
                last_err = e
                continue
        raise FileNotFoundError(
            f"无法找到 ISA-L 库（尝试 {lib_names}）。请确保已安装，或通过 lib_path 指定路径。\n最后错误：{last_err}"
        )

    def _define_prototypes(self) -> None:
        # xor_gen / xor_gen_base
        def _set_proto_if_exists(symbol: str, restype, argtypes) -> Optional[Any]:
            try:
                fn = getattr(self.lib, symbol)
            except AttributeError:
                return None
            fn.restype = restype
            fn.argtypes = argtypes
            return fn

        arg_ptr_ptr = ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))

        self._xor_gen = _set_proto_if_exists(
            "xor_gen", ctypes.c_int, [ctypes.c_int, ctypes.c_int, arg_ptr_ptr]
        )
        self._xor_gen_base = _set_proto_if_exists(
            "xor_gen_base", ctypes.c_int, [ctypes.c_int, ctypes.c_int, arg_ptr_ptr]
        )

        # ec_init_tables
        self.lib.ec_init_tables.restype = None
        self.lib.ec_init_tables.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_ubyte),
        ]

        # ec_encode_data / ec_encode_data_base
        self._ec_encode_data = _set_proto_if_exists(
            "ec_encode_data",
            None,
            [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_ubyte),
                arg_ptr_ptr,
                arg_ptr_ptr,
            ],
        )
        self._ec_encode_data_base = _set_proto_if_exists(
            "ec_encode_data_base",
            None,
            [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_ubyte),
                arg_ptr_ptr,
                arg_ptr_ptr,
            ],
        )

    # -------------------------
    # 工具：指针与对齐
    # -------------------------
    @staticmethod
    def _is_uint8_c_contiguous(x: TensorOrNd) -> bool:
        if _HAS_TORCH and (torch is not None) and isinstance(x, torch.Tensor):  # type: ignore[attr-defined]
            torch_uint8 = getattr(torch, "uint8", None)
            torch_strided = getattr(torch, "strided", None)
            return (
                x.device.type == "cpu"
                and x.dtype == torch_uint8
                and x.is_contiguous()
                and x.layout == torch_strided
            )
        if isinstance(x, np.ndarray):
            return x.dtype == np.uint8 and x.flags.c_contiguous
        return False

    @staticmethod
    def _elem_count(x: TensorOrNd) -> int:
        if _HAS_TORCH and (torch is not None) and isinstance(x, torch.Tensor):  # type: ignore[attr-defined]
            return int(x.numel())
        if isinstance(x, np.ndarray):
            return int(x.size)
        raise TypeError("不支持的 buffer 类型")

    @staticmethod
    def _as_uint8_ptr(x: TensorOrNd) -> Any:
        if _HAS_TORCH and (torch is not None) and isinstance(x, torch.Tensor):  # type: ignore[attr-defined]
            if x.device.type != "cpu":
                raise TypeError("仅支持 CPU tensor")
            torch_uint8 = getattr(torch, "uint8", None)
            if x.dtype != torch_uint8:
                raise TypeError("tensor 必须为 torch.uint8")
            if not x.is_contiguous():
                raise TypeError("tensor 必须为 C contiguous")
            addr = x.data_ptr()
            return ctypes.cast(ctypes.c_void_p(addr), ctypes.POINTER(ctypes.c_ubyte))
        # numpy
        if not (isinstance(x, np.ndarray) and x.dtype == np.uint8 and x.flags.c_contiguous):
            raise TypeError("buffer 必须为 numpy.uint8 且 C contiguous")
        return x.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    def _get_pointer_array(self, buffers: Sequence[TensorOrNd]):
        num = len(buffers)
        arr = (ctypes.POINTER(ctypes.c_ubyte) * num)()
        for i, b in enumerate(buffers):
            if not self._is_uint8_c_contiguous(b):
                raise TypeError("所有 buffer 需为 CPU uint8 且 C contiguous")
            arr[i] = self._as_uint8_ptr(b)
        return arr

    @staticmethod
    def aligned_tensor(n_bytes: int, align: int = 32) -> Any:
        """返回对齐到 align 字节的 CPU uint8 Tensor（共享一个对齐的 numpy 缓冲区）。"""
        if not _HAS_TORCH:
            raise RuntimeError("未安装 torch，无法创建 tensor")
        # 过量分配 + 切片对齐（避免 posix_memalign/free 的繁琐生命周期管理）
        buf = np.empty(n_bytes + align, dtype=np.uint8)
        offset = (-buf.ctypes.data) % align
        view = buf[offset : offset + n_bytes]
        # from_numpy 与 view 共享内存，且保持对齐
        if torch is None:
            raise RuntimeError("torch 不可用")
        return torch.from_numpy(view)

    # -------------------------
    # 公开 API：XOR & EC 编码
    # -------------------------
    def perform_xor(self, data_blocks: Sequence[TensorOrNd], use_base: Optional[bool] = None) -> None:
        """
        对一组数据块执行 XOR 生成校验块。最后一个块作为目标校验块。
        支持 torch.Tensor 或 numpy.ndarray（uint8）。
        """
        if len(data_blocks) < 2:
            raise ValueError("XOR 至少需要 2 个块（k 数据 + 1 校验）")
        length = self._elem_count(data_blocks[0])
        for b in data_blocks:
            if self._elem_count(b) != length:
                raise ValueError("所有块长度必须相同（以字节为单位）")

        ptrs = self._get_pointer_array(data_blocks)
        vects = len(data_blocks)

        # 选择 base/opt
        call_base = self.prefer_base if use_base is None else use_base
        fn = self._xor_gen_base if call_base and self._xor_gen_base else self._xor_gen
        if fn is None:
            raise RuntimeError("未找到 xor_gen 符号（库可能缺失）")
        fn(vects, int(length), ptrs)

    # --- Galois Field (GF) 辅助：Vandermonde ---
    @staticmethod
    def gf_mul(a: int, b: int) -> int:
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hbit = a & 0x80
            a = (a << 1) & 0xFF
            if hbit:
                a ^= 0x1D  # x^8 + x^4 + x^3 + x + 1
            b >>= 1
        return p & 0xFF

    @classmethod
    def gf_pow(cls, a: int, b: int) -> int:
        if b == 0:
            return 1
        p = a & 0xFF
        for _ in range(1, b):
            p = cls.gf_mul(p, a)
        return p & 0xFF

    @classmethod
    def build_encode_matrix_vandermonde(cls, k: int, rows: int) -> np.ndarray:
        """返回 (rows x k) 的 uint8 编码矩阵（Vandermonde）。"""
        a = np.zeros((rows, k), dtype=np.uint8)
        for i in range(rows):
            base = (i + 1) & 0xFF
            for j in range(k):
                a[i, j] = cls.gf_pow(base, j)
        return a

    # --- 预计算与复用 ---
    def build_gftbls(self, k: int, rows: int, encode_matrix: Optional[np.ndarray] = None) -> _ECTables:
        """
        计算 gftbls 并返回可复用的表对象。仅调用一次，后续复用。
        encode_matrix: (rows x k) 的 np.uint8。若为 None，则使用 Vandermonde。
        """
        if encode_matrix is None:
            encode_matrix = self.build_encode_matrix_vandermonde(k, rows)
        if not (isinstance(encode_matrix, np.ndarray) and encode_matrix.dtype == np.uint8 and encode_matrix.flags.c_contiguous):
            raise TypeError("encode_matrix 需为 np.uint8 且 C contiguous，形状为 (rows, k)")
        if encode_matrix.shape != (rows, k):
            raise ValueError(f"encode_matrix 形状必须为 (rows, k)=({rows}, {k})")

        a_flat = encode_matrix.reshape(-1)
        a_ptr = a_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        gftbls_size = 32 * k * rows
        gftbls = (ctypes.c_ubyte * gftbls_size)()
        self.lib.ec_init_tables(k, rows, a_ptr, gftbls)
        return _ECTables(k, rows, a_flat, gftbls)

    def ec_encode_with_tables(
        self,
        data_blocks: Sequence[TensorOrNd],
        coding_blocks: Sequence[TensorOrNd],
        tables: _ECTables,
        use_base: Optional[bool] = None,
    ) -> None:
        """使用预计算的 gftbls 进行一次 EC 编码。"""
        k = tables.k
        rows = tables.rows
        if len(data_blocks) != k or len(coding_blocks) != rows:
            raise ValueError("data_blocks/coding_blocks 的数量与 k/rows 不匹配")
        length = self._elem_count(data_blocks[0])
        for b in list(data_blocks) + list(coding_blocks):
            if self._elem_count(b) != length:
                raise ValueError("所有块长度必须一致")

        src_ptrs = self._get_pointer_array(data_blocks)
        dest_ptrs = self._get_pointer_array(coding_blocks)

        call_base = self.prefer_base if use_base is None else use_base
        fn = self._ec_encode_data_base if call_base and self._ec_encode_data_base else self._ec_encode_data
        if fn is None:
            raise RuntimeError("未找到 ec_encode_data 符号（库可能缺失）")
        fn(int(length), int(k), int(rows), tables.gftbls_ptr, src_ptrs, dest_ptrs)

    # 仍保留一次性接口（向后兼容，内部自动构建 gftbls）
    def perform_ec_encode(
        self,
        data_blocks: Sequence[TensorOrNd],
        coding_blocks: Sequence[TensorOrNd],
        k: int,
        rows: int,
        encode_matrix: Optional[np.ndarray] = None,
        use_base: Optional[bool] = None,
    ) -> None:
        tables = self.build_gftbls(k, rows, encode_matrix)
        self.ec_encode_with_tables(data_blocks, coding_blocks, tables, use_base=use_base)
