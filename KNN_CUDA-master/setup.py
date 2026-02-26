import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Set the architecture list to avoid the compute_37 error we saw earlier
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.1;7.0;7.5;8.0;8.6;8.9;9.0"

setup(
    name='knn_cuda',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='knn_cuda._ext',
            sources=[
                'knn_cuda/csrc/cuda/knn_main.cpp', # Updated name
                'knn_cuda/csrc/cuda/knn.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-allow-unsupported-compiler'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)