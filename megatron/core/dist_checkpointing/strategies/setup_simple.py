#!/usr/bin/env python3
"""
Simple setup script for EC-CHECK native C++ module (minimal dependencies).
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import torch
import os

print(f"PyTorch version: {torch.__version__}")

# Get include directories manually
torch_path = torch.__file__
torch_dir = os.path.dirname(torch_path)

# Try different possible include paths
possible_torch_includes = [
    os.path.join(torch_dir, "include"),
    os.path.join(torch_dir, "include", "torch", "csrc", "api", "include"),
    os.path.join(torch_dir, "..", "include"),
    os.path.join(torch_dir, "..", "include", "torch", "csrc", "api", "include"),
]

torch_include = []
for path in possible_torch_includes:
    if os.path.exists(path):
        torch_include.append(path)
        print(f"Found torch include: {path}")

if not torch_include:
    print("Warning: No torch include paths found, using fallback")
    torch_include = [torch_dir]

pybind11_include = pybind11.get_include()
print(f"PyBind11 include: {pybind11_include}")

# Check for NCCL availability
nccl_available = False
nccl_lib_dirs = []
nccl_libs = []
nccl_include_dirs = []

# Try to find NCCL library
nccl_lib_paths = [
    "/usr/local/nccl/lib",
    "/opt/nccl/lib", 
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/cuda/lib64",
    "/opt/cuda/lib64",
]

for path in nccl_lib_paths:
    if os.path.exists(os.path.join(path, "libnccl.so")):
        nccl_lib_dirs.append(path)
        nccl_libs.append("nccl")
        nccl_available = True
        print(f"Found NCCL library at: {path}")
        break

# Try to find NCCL headers
nccl_header_paths = [
    "/usr/local/nccl/include",
    "/opt/nccl/include",
    "/usr/include/nccl",
    "/usr/local/cuda/include",
    "/opt/cuda/include",
]

for path in nccl_header_paths:
    if os.path.exists(os.path.join(path, "nccl.h")):
        nccl_include_dirs.append(path)
        print(f"Found NCCL headers at: {path}")
        break

if not nccl_available:
    print("Warning: NCCL not found, building without NCCL support")

# Define the extension
ext_modules = [
    Pybind11Extension(
        "eccheck_native",
        sources=["eccheck_native.cpp"],
        include_dirs=[
            *torch_include,
            pybind11_include,
            *nccl_include_dirs,
        ],
        libraries=nccl_libs,
        library_dirs=nccl_lib_dirs,
        define_macros=[
            ("NCCL_AVAILABLE", "1") if nccl_available else ("NCCL_AVAILABLE", "0"),
        ],
        cxx_std=17,
        language='c++',
        extra_compile_args=[
            "-O3",
            "-std=c++17",
            "-fPIC",
        ],
        extra_link_args=[
            "-fPIC",
        ],
    ),
]

setup(
    name="eccheck_native",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
