import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# FIX 1: Modern GPU Architectures
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.1;7.0;7.5;8.0;8.6;8.9;9.0"

setup(name='chamfer',
      version='2.0.0',
      ext_modules=[
          CUDAExtension('chamfer', [
              'chamfer_cuda.cpp',
              'chamfer.cu',
          ],
          extra_compile_args={
              'cxx': ['-O3'],
              'nvcc': [
                  '-O3',
                  # FIX 2: The "Master Key" for Visual Studio 18
                  '-allow-unsupported-compiler' 
              ]
          }),
      ],
      cmdclass={'build_ext': BuildExtension})