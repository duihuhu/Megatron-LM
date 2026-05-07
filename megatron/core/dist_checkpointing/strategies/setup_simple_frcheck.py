#!/usr/bin/env python3
"""Build script for FRCheck native module (pybind11 + Boost.Asio + RDMA verbs)."""

import os

import pybind11
import torch
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

print(f"PyTorch version: {torch.__version__}")

torch_path = torch.__file__
torch_dir = os.path.dirname(torch_path)

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

ext_modules = [
    Pybind11Extension(
        "frcheck_native",
        sources=["frcheck_native.cpp"],
        include_dirs=[*torch_include, pybind11_include],
        libraries=["ibverbs", "isal", "pthread"],
        library_dirs=[],
        cxx_std=17,
        language="c++",
        extra_compile_args=[
            "-O2",
            "-g",
            "-std=c++17",
            "-fPIC",
            "-pthread",
        ],
        extra_link_args=[
            "-fPIC",
            "-pthread",
        ],
    ),
]

setup(
    name="frcheck_native",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
