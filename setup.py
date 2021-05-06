from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='fcos',
      version='0.9.0',
      author='Ben Talbot',
      author_email='b.talbot@qut.edu.au',
      description='Fully convolutional one-stage object detection (FCOS)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=['acrv_datasets'],
      classifiers=(
          "Development Status :: 4 - Beta",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ))
