#!/usr/bin/env python3
"""
Simple setup script for Gemini native C++ module (minimal dependencies).
No PyTorch dependency - uses raw memory addresses for buffer transfer.
Supports both ASIO and RDMA transports.
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os

pybind11_include = pybind11.get_include()
print(f"PyBind11 include: {pybind11_include}")

# Find Boost include directories
boost_include_dirs = []
boost_header_paths = [
    "/usr/include",
    "/usr/local/include",
    "/opt/homebrew/include",  # macOS homebrew
    "/usr/include/boost",
    "/usr/local/include/boost",
]

for path in boost_header_paths:
    # Check for boost/asio.hpp
    if os.path.exists(os.path.join(path, "boost", "asio.hpp")):
        boost_include_dirs.append(path)
        print(f"Found boost include at: {path}")
        break

if not boost_include_dirs:
    print("Warning: Boost headers not found, will try default system paths")

# Find Boost libraries
boost_lib_dirs = []
boost_libs = []
boost_lib_paths = [
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib",
    "/opt/homebrew/lib",  # macOS homebrew
    "/lib",
    "/lib64",
    "/usr/lib64",
]

for path in boost_lib_paths:
    if os.path.exists(os.path.join(path, "libboost_system.so")) or \
       os.path.exists(os.path.join(path, "libboost_system.a")) or \
       os.path.exists(os.path.join(path, "libboost_system.dylib")):
        boost_lib_dirs.append(path)
        boost_libs.append("boost_system")
        print(f"Found boost_system library at: {path}")
        break

if not boost_libs:
    print("Warning: boost_system library not found, will try default system paths")
    boost_libs.append("boost_system")  # Try anyway

# Find RDMA libraries (libibverbs and librdmacm)
rdma_include_dirs = []
rdma_lib_dirs = []
rdma_libs = []

# Check for RDMA headers
rdma_header_paths = [
    "/usr/include",
    "/usr/local/include",
    "/opt/mellanox/include",
]

for path in rdma_header_paths:
    if os.path.exists(os.path.join(path, "infiniband", "verbs.h")):
        rdma_include_dirs.append(path)
        print(f"Found RDMA (libibverbs) headers at: {path}")
        break

# Check for RDMA libraries
rdma_lib_paths = [
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib",
    "/usr/lib64",
    "/lib64",
    "/opt/mellanox/lib",
]

# Check for libibverbs
ibverbs_found = False
for path in rdma_lib_paths:
    # Check for .so, .so.1, .a files
    if os.path.exists(os.path.join(path, "libibverbs.so")) or \
       os.path.exists(os.path.join(path, "libibverbs.so.1")) or \
       os.path.exists(os.path.join(path, "libibverbs.a")):
        if path not in rdma_lib_dirs:
            rdma_lib_dirs.append(path)
        if "ibverbs" not in rdma_libs:
            rdma_libs.append("ibverbs")
        # List all found variants
        variants = []
        if os.path.exists(os.path.join(path, "libibverbs.so")):
            variants.append("libibverbs.so")
        if os.path.exists(os.path.join(path, "libibverbs.so.1")):
            variants.append("libibverbs.so.1")
        if os.path.exists(os.path.join(path, "libibverbs.a")):
            variants.append("libibverbs.a")
        print(f"Found libibverbs library at: {path} ({', '.join(variants)})")
        ibverbs_found = True
        break

# Check for librdmacm
rdmacm_found = False
for path in rdma_lib_paths:
    # Check for .so, .so.1, .a files
    if os.path.exists(os.path.join(path, "librdmacm.so")) or \
       os.path.exists(os.path.join(path, "librdmacm.so.1")) or \
       os.path.exists(os.path.join(path, "librdmacm.a")):
        if path not in rdma_lib_dirs:
            rdma_lib_dirs.append(path)
        if "rdmacm" not in rdma_libs:
            rdma_libs.append("rdmacm")
        # List all found variants
        variants = []
        if os.path.exists(os.path.join(path, "librdmacm.so")):
            variants.append("librdmacm.so")
        if os.path.exists(os.path.join(path, "librdmacm.so.1")):
            variants.append("librdmacm.so.1")
        if os.path.exists(os.path.join(path, "librdmacm.a")):
            variants.append("librdmacm.a")
        print(f"Found librdmacm library at: {path} ({', '.join(variants)})")
        rdmacm_found = True
        break

# If versioned .so files exist but not the plain .so symlink, warn user
if not ibverbs_found or not rdmacm_found:
    print("\nSearching for versioned library files...")
    for path in rdma_lib_paths:
        if not ibverbs_found and os.path.exists(os.path.join(path, "libibverbs.so.1")):
            print(f"⚠️  Found {path}/libibverbs.so.1 but missing libibverbs.so symlink")
            print(f"   Create symlink: cd {path} && sudo ln -sf libibverbs.so.1 libibverbs.so")
        if not rdmacm_found and os.path.exists(os.path.join(path, "librdmacm.so.1")):
            print(f"⚠️  Found {path}/librdmacm.so.1 but missing librdmacm.so symlink")
            print(f"   Create symlink: cd {path} && sudo ln -sf librdmacm.so.1 librdmacm.so")

if not rdma_libs:
    print("Warning: RDMA development libraries not found.")
    print("  Runtime libraries (.so.1) may exist but development symlinks (.so) are missing.")
    print("  To fix this, install development packages:")
    print("    Ubuntu/Debian: sudo apt-get install libibverbs-dev librdmacm-dev")
    print("    CentOS/RHEL: sudo yum install libibverbs-devel librdmacm-devel")
    print("  Or create symlinks manually:")
    print("    cd /lib/x86_64-linux-gnu")
    print("    sudo ln -sf libibverbs.so.1 libibverbs.so")
    print("    sudo ln -sf librdmacm.so.1 librdmacm.so")
    # Add libraries anyway to allow compilation (will fail at link time if symlinks don't exist)
    rdma_libs.extend(["ibverbs", "rdmacm"])

# Combine all include directories and library directories
all_include_dirs = [pybind11_include] + boost_include_dirs + rdma_include_dirs
all_lib_dirs = list(set(boost_lib_dirs + rdma_lib_dirs))  # Remove duplicates
all_libs = boost_libs + rdma_libs + ["pthread"]

print(f"\nCompilation configuration:")
print(f"  Include dirs: {all_include_dirs}")
print(f"  Library dirs: {all_lib_dirs}")
print(f"  Libraries: {all_libs}")
print()

# Define the extension
ext_modules = [
    Pybind11Extension(
        "gemini_native",
        sources=["gemini_native.cpp"],
        include_dirs=all_include_dirs,
        libraries=all_libs,
        library_dirs=all_lib_dirs,
        cxx_std=17,
        language='c++',
        extra_compile_args=[
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-DBOOST_ASIO_DISABLE_CONCEPTS",  # For compatibility
        ],
        extra_link_args=[
            "-fPIC",
            "-lpthread",
        ],
    ),
]

setup(
    name="gemini_native",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

