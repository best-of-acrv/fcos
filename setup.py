from glob import glob
import os
from pkg_resources import resource_filename
from setuptools import find_packages, setup
import sys
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

# Hack to deal with opencv's diabolical naming conventions (conda installs
# 'py-opencv', pip installs 'opencv-python'... pip check errors in conda as it
# can't find 'opencv-python'... solution -> only require 'opencv-python' for
# installsoutside of conda)
install_requires_list = [
    'acrv_datasets',
    'numpy',
    'pycocotools==2.0.1',  # 2.0.2 causes: https://github.com/tensorflow/models/issues/9749 
    'torch',
    'torchvision',
    'yacs'
]
if not os.path.exists(os.path.join(sys.prefix, 'conda-meta')):
    install_requires_list.append('opencv-python')

setup(name='fcos',
      version='0.9.7',
      author='Ben Talbot',
      author_email='b.talbot@qut.edu.au',
      url='https://github.com/best-of-acrv/fcos',
      description='Fully convolutional one-stage object detection (FCOS)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      package_data={'fcos': ['configs/*.yaml']},
      install_requires=install_requires_list,
      ext_modules=get_extensions(),
      cmdclass={"build_ext": tcpp.BuildExtension},
      entry_points={'console_scripts': ['fcos=fcos.__main__:main']},
      classifiers=(
          "Development Status :: 4 - Beta",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ))
