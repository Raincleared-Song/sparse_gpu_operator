from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ffn_4",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "ffn_4",
            ["pytorch/formula_4.cpp", "kernel/formula_4_kernel.cu"],
            define_macros=[('USE_CONSTANT', None)],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)

setup(
    name="ffn_23",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "ffn_23",
            ["pytorch/formula_23.cpp", "kernel/formula_23_kernel.cu"],
            define_macros=[('USE_CONSTANT', None)],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
