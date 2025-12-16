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
isa_available = False
isa_lib_dirs = []
isa_libs = []
isa_include_dirs = []

boost_include_dirs = []
# Common library paths to check for libisal
isa_lib_paths = [
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib",
    "/lib",
    "/lib64",
    "/usr/lib64",
]

boost_header_paths = [
    "/usr/include/boost",
    "/usr/local/include/boost",
    "/usr/include",
    "/usr/local/include",
]

for path in boost_header_paths:
    if os.path.exists(os.path.join(path, "boost/filesystem.hpp")):
        boost_include_dirs.append(path)
        print(f"Found boost include at: {path}")
        break

for path in isa_lib_paths:
    if os.path.exists(os.path.join(path, "libisal.so")) or os.path.exists(os.path.join(path, "libisal.a")):
        isa_lib_dirs.append(path)
        isa_libs.append("isal")
        isa_available = True
        print(f"Found isa-l library at: {path}")
        break

# Common include paths for isa-l headers
isa_header_paths = [
    "/usr/include/isa-l",
    "/usr/include",
    "/usr/local/include",
]

for path in isa_header_paths:
    # prefer directory that contains isa-l headers
    if os.path.isdir(path) and (os.path.exists(os.path.join(path, "isa-l.h")) or os.path.exists(os.path.join(path, "isa-l", "isa-l.h"))):
        isa_include_dirs.append(path)
        print(f"Found isa-l include at: {path}")
        break

if not isa_available:
    print("Warning: isa-l not found, building without isa-l support")

# Wrap setuptools.setup so we can inject isa-l include/libs into ext_modules at call time.
_original_setup = setup

def setup(*args, **kwargs):
    ext_modules = kwargs.get("ext_modules")
    if ext_modules:
        for ext in ext_modules:
            # Pybind11Extension exposes include_dirs, libraries, library_dirs attributes
            try:
                if isa_include_dirs:
                    existing = list(getattr(ext, "include_dirs", []) or [])
                    # avoid duplicates
                    for p in isa_include_dirs:
                        if p not in existing:
                            existing.append(p)
                    ext.include_dirs = existing
                if isa_libs:
                    existing = list(getattr(ext, "libraries", []) or [])
                    for lib in isa_libs:
                        if lib not in existing:
                            existing.append(lib)
                    ext.libraries = existing
                if isa_lib_dirs:
                    existing = list(getattr(ext, "library_dirs", []) or [])
                    for d in isa_lib_dirs:
                        if d not in existing:
                            existing.append(d)
                    ext.library_dirs = existing
            except Exception:
                # If anything goes wrong, fall back to original behavior
                pass
    return _original_setup(*args, **kwargs)
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

cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
cuda_include_dir = os.path.join(cuda_home, 'include')
cuda_lib_dir = os.path.join(cuda_home, 'lib64')
cuda_include = []
cuda_libs = []
cuda_lib_dirs = []
if os.path.isdir(cuda_include_dir):
    cuda_include.append(cuda_include_dir)
    print(f"Found CUDA include: {cuda_include_dir}")
else:
    print(f"Warning: CUDA include directory not found at {cuda_include_dir}")

if os.path.isdir(cuda_lib_dir):
    cuda_lib_dirs.append(cuda_lib_dir)
    cuda_libs.append("cudart")
    print(f"Found CUDA lib: {cuda_lib_dir}")
else:
    # Try alternative paths
    alt_cuda_lib_dirs = ["/usr/local/cuda/lib64", "/opt/cuda/lib64", "/usr/lib/x86_64-linux-gnu"]
    for alt_dir in alt_cuda_lib_dirs:
        if os.path.isdir(alt_dir):
            cuda_lib_dirs.append(alt_dir)
            cuda_libs.append("cudart")
            print(f"Found CUDA lib: {alt_dir}")
            break

# Define the extension
ext_modules = [
    Pybind11Extension(
        "eclatin_native",
        sources=["eclatin_native.cpp"],
        include_dirs=[
            *torch_include,
            pybind11_include,
            *nccl_include_dirs,
            *cuda_include,
        ],
        libraries=nccl_libs + cuda_libs,
        library_dirs=nccl_lib_dirs + cuda_lib_dirs,
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
    name="eclatin_native",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
