#!/usr/bin/env python3
"""
Simple setup script for Gemini native C++ module (minimal dependencies).
No PyTorch dependency - uses raw memory addresses for buffer transfer.
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

# Define the extension
ext_modules = [
    Pybind11Extension(
        "gemini_native",
        sources=["gemini_native.cpp"],
        include_dirs=[
            pybind11_include,
            *boost_include_dirs,
        ],
        libraries=boost_libs + ["pthread"],
        library_dirs=boost_lib_dirs,
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

