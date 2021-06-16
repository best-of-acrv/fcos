from glob import glob
import os
from pkg_resources import resource_filename
from setuptools import find_packages, setup
import torch
import torch.utils.cpp_extension as tcpp

ROOT = os.path.join('fcos', 'core', 'csrc')


def get_extensions():
    is_cuda = ((torch.cuda.is_available() and tcpp.CUDA_HOME is not None) or
               os.getenv("FORCE_CUDA", "0") == "1")

    ext = tcpp.CUDAExtension if is_cuda else tcpp.CppExtension

    source_main = glob(os.path.join(ROOT, '*.cpp'))[0]
    sources_cpu = glob(os.path.join(ROOT, 'cpu', '*.cpp'))
    sources = [source_main] + sources_cpu + (glob(
        os.path.join(ROOT, 'cuda', '*.cu')) if is_cuda else [])

    define_macros = [("WITH_CUDA", None)] if is_cuda else []
    extra_compile_args = {
        "cxx": [],
        **({
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        } if is_cuda else {})
    }

    return [
        ext("fcos.core._C",
            sources,
            include_dirs=[ROOT],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)
    ]


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='fcos',
      version='0.9.3',
      author='Ben Talbot',
      author_email='b.talbot@qut.edu.au',
      url='https://github.com/best-of-acrv/fcos',
      description='Fully convolutional one-stage object detection (FCOS)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      package_data={'fcos': ['configs/*.yaml']},
      install_requires=[
          'acrv_datasets', 'numpy', 'opencv-python', 'pycocotools', 'torch',
          'torchvision', 'yacs'
      ],
      ext_modules=get_extensions(),
      cmdclass={"build_ext": tcpp.BuildExtension},
      classifiers=(
          "Development Status :: 4 - Beta",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ))
