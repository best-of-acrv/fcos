from glob import glob
import os
from pkg_resources import resource_filename
from setuptools import find_packages, setup
import torch
import torch.utils.cpp_extension as tcpp


def get_extensions():
    # This code originally offered non-CUDA compilation... this seemed to be a
    # lie though as crucial methods like "modulated_deform_conv" are currently
    # unimplemented for CPU. Instead, we bite the bullet and make CUDA an
    # explicit requirement for this package. It matches the typical use-case at
    # the end of the day.
    root = os.path.join('fcos', 'core', 'csrc')
    return [
        tcpp.CUDAExtension("fcos.core._C",
                           [glob(os.path.join(root, '*.cpp'))[0]] +
                           glob(os.path.join(root, 'cpu', '*.cpp')) +
                           glob(os.path.join(root, 'cuda', '*.cu')),
                           include_dirs=[root],
                           define_macros=[("WITH_CUDA", None)],
                           extra_compile_args={
                               "cxx": [],
                               "nvcc": [
                                   "-DCUDA_HAS_FP16=1",
                                   "-D__CUDA_NO_HALF_OPERATORS__",
                                   "-D__CUDA_NO_HALF_CONVERSIONS__",
                                   "-D__CUDA_NO_HALF2_OPERATORS__",
                               ]
                           })
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
