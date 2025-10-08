# isal_wrapper.py

import ctypes
import numpy as np
from typing import List
import os

class ISAL_Lib:
    """
    一个用于加载和调用 ISA-L C 库函数的 Python 封装器。
    """
    def __init__(self, lib_path: str = None):
        self.lib = self._load_library(lib_path)
        self._define_prototypes()

    def _load_library(self, lib_path: str):
        """加载 ISA-L 共享库"""
        if lib_path:
            return ctypes.CDLL(lib_path)
        
        # 尝试在标准位置查找库
        if os.name == 'posix':
            lib_names = ['libisal.so.2', 'libisal.so']
        elif os.name == 'nt':
            lib_names = ['isal.dll']
        else:
            raise NotImplementedError(f"不支持的操作系统: {os.name}")
        
        for name in lib_names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        raise FileNotFoundError(
            "无法找到 ISA-L 库。请确保已安装，或通过 lib_path 参数指定路径。"
        )

    def _define_prototypes(self):
        """使用 ctypes 定义 C 函数原型"""
        
        # --- XOR 函数原型 ---
        # int xor_gen(int vects, int len, void **array);
        self.lib.xor_gen.restype = ctypes.c_int
        self.lib.xor_gen.argtypes = [
            ctypes.c_int, 
            ctypes.c_int, 
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))
        ]

        # --- EC 函数原型 ---
        # void ec_init_tables(int k, int rows, unsigned char *a, unsigned char *gftbls);
        self.lib.ec_init_tables.restype = None
        self.lib.ec_init_tables.argtypes = [
            ctypes.c_int, 
            ctypes.c_int, 
            ctypes.POINTER(ctypes.c_ubyte), 
            ctypes.POINTER(ctypes.c_ubyte)
        ]

        # void ec_encode_data(int len, int k, int rows, unsigned char *gftbls, 
        #                     unsigned char **data, unsigned char **coding);
        self.lib.ec_encode_data.restype = None
        self.lib.ec_encode_data.argtypes = [
            ctypes.c_int, 
            ctypes.c_int, 
            ctypes.c_int, 
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))
        ]

    def _get_pointer_array(self, buffers: List[np.ndarray]):
        """
        辅助函数：将一个 NumPy 数组列表转换为 C 的 `unsigned char**` 类型。
        """
        num_buffers = len(buffers)
        # 创建一个指针数组
        ptr_array = (ctypes.POINTER(ctypes.c_ubyte) * num_buffers)()
        for i, buf in enumerate(buffers):
            # 确保数据类型正确
            if buf.dtype != np.uint8:
                raise TypeError("所有 buffer 必须是 numpy.uint8 类型")
            # 获取每个 buffer 的数据指针并填充到指针数组中
            ptr_array[i] = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        return ptr_array

    def perform_xor(self, data_blocks: List[np.ndarray]):
        """
        对一组数据块执行 XOR 生成校验块。
        isa-l 的 xor_gen 会将所有块的异或结果存放到最后一个块中。
        
        :param data_blocks: NumPy 数组的列表，最后一个元素将作为目标校验块。
        """
        if len(data_blocks) < 2:
            raise ValueError("XOR 操作至少需要2个数据块 (k个数据 + 1个校验)")
            
        vects = len(data_blocks)
        length = len(data_blocks[0]) # 假设所有块长度相同

        # 转换为 C 指针数组
        ptr_array = self._get_pointer_array(data_blocks)
        
        # 调用 C 函数
        self.lib.xor_gen(vects, length, ptr_array)

    def perform_ec_encode(self, data_blocks: List[np.ndarray], coding_blocks: List[np.ndarray], k: int, m: int):
        """
        对数据块进行EC编码。
        
        :param data_blocks: k 个源数据块 (NumPy数组列表)
        :param coding_blocks: m 个用于存放结果的校验块 (NumPy数组列表)
        :param k: 数据块数量
        :param m: 校验块数量
        """
        if len(data_blocks) != k or len(coding_blocks) != m:
            raise ValueError(f"数据块数量({len(data_blocks)})或校验块数量({len(coding_blocks)})与k/m不匹配")

        length = len(data_blocks[0])

        # 1. 创建编码系数矩阵 (使用标准的 Vandermonde 矩阵)
        # isa-l 需要一个扁平化的 k*m 矩阵
        encode_matrix = np.zeros((m, k), dtype=np.uint8)
        for i in range(m):
            for j in range(k):
                encode_matrix[i, j] = self.gf_pow(i + 1, j)
        
        # 将矩阵扁平化并转换为 C 指针
        encode_matrix_flat = encode_matrix.flatten()
        c_encode_matrix = encode_matrix_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        # 2. 初始化 GF 表
        gftbls_size = 32 * k * m
        c_gftbls = (ctypes.c_ubyte * gftbls_size)() # 在内存中分配空间
        self.lib.ec_init_tables(k, m, c_encode_matrix, c_gftbls)

        # 3. 准备数据和编码区的指针数组
        data_ptr_array = self._get_pointer_array(data_blocks)
        coding_ptr_array = self._get_pointer_array(coding_blocks)
        
        # 4. 调用编码函数
        self.lib.ec_encode_data(length, k, m, c_gftbls, data_ptr_array, coding_ptr_array)

    # --- Galois Field (GF) 辅助函数, 用于生成矩阵 ---
    @staticmethod
    def gf_pow(a: int, b: int) -> int:
        """伽罗瓦域(2^8)内的幂运算"""
        p = 0
        for _ in range(b):
            p = ISAL_Lib.gf_mul(p, a) if p != 0 else a
        return p if b > 0 else 1

    @staticmethod
    def gf_mul(a: int, b: int) -> int:
        """伽罗瓦域(2^8)内的乘法"""
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hbit = a & 0x80
            a <<= 1
            if hbit:
                a ^= 0x1d  # 0x1d 是 GF(2^8) 的本原多项式 x^8+x^4+x^3+x+1
            b >>= 1
        return p