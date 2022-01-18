import torch
from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension, \
    BuildExtension, CppExtension
import os
import glob
import sys
import re
import warnings
import pkg_resources
from setuptools import find_packages, setup
# import versioneer


def omp_flags():
    if sys.platform == "win32":
        return ["/openmp"]
    if sys.platform == "darwin":
        # https://stackoverflow.com/questions/37362414/
        # return ["-fopenmp=libiomp5"]
        return []
    return ["-fopenmp"]

def torch_parallel_backend():
    try:
        match = re.search("^ATen parallel backend: (?P<backend>.*)$", torch._C._parallel_info(), re.MULTILINE)
        if match is None:
            return None
        backend = match.group("backend")
        if backend == "OpenMP":
            return "AT_PARALLEL_OPENMP"
        if backend == "native thread pool":
            return "AT_PARALLEL_NATIVE"
        if backend == "native thread pool and TBB":
            return "AT_PARALLEL_NATIVE_TBB"
    except (NameError, AttributeError):  # no torch or no binaries
        warnings.warn("Could not determine torch parallel_info.")
    return None


print(f"setup.py with torch {torch.__version__}")
print(f"Torch backend {torch_parallel_backend()}")
_pt_version = pkg_resources.parse_version(torch.__version__).release  # type: ignore[attr-defined]
if _pt_version is None or len(_pt_version) < 3:
    raise AssertionError("unknown torch version")
TORCH_VERSION = int(_pt_version[0]) * 10000 + int(_pt_version[1]) * 100 + int(_pt_version[2])
define_macros = [(f"{torch_parallel_backend()}", 1), ("TORCH_VERSION", TORCH_VERSION)]

extra_link_args = omp_flags()

this_dir = os.path.dirname(os.path.abspath(__file__))
ext_dir = this_dir
include_dirs = [ext_dir]

source_cpu = glob.glob(os.path.join(ext_dir, "**", "*.cpp"), recursive=True)
source_cuda = glob.glob(os.path.join(ext_dir, "**", "*.cu"), recursive=True)
sources = source_cpu

extension = CUDAExtension
sources += source_cuda
define_macros += [("WITH_CUDA", None)]
extra_compile_args = {"cxx": [], "nvcc": []}
if torch_parallel_backend() == "AT_PARALLEL_OPENMP":
            extra_compile_args["cxx"] += omp_flags()

ext_modules = [
        CUDAExtension(
            name="permuto",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
setup(name='permuto',
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)})