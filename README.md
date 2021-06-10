<p align=center><strong>~Please note this is only a <em>beta</em> release at this stage~</strong></p>

# FCOS: fully convolutional one-stage object detection

[![Best of ACRV Repository](https://img.shields.io/badge/collection-best--of--acrv-%23a31b2a)](https://roboticvision.org/best-of-acrv)
![Primary language](https://img.shields.io/github/languages/top/best-of-acrv/fcos)
[![PyPI package](https://img.shields.io/pypi/pyversions/fcos)](https://pypi.org/project/fcos/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/fcos.svg)](https://anaconda.org/conda-forge/fcos)
[![Conda Recipe](https://img.shields.io/badge/recipe-fcos-green.svg)](https://anaconda.org/conda-forge/fcos)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/fcos.svg)](https://anaconda.org/conda-forge/fcos)
[![License](https://img.shields.io/github/license/best-of-acrv/fcos)](./LICENSE.txt)

Fully convolutional one-stage object detection (FCOS) is a framework for per-pixel prediction of objects in images. FCOS doesn't rely on expensive anchor box calculations and their hyper-parameters, which is in contrast to state-of-the-art object detectors like RetinaNet, YOLOv3, and Faster R-CNN.

TODO: image of the system's output

This repository contains an open-source implementation of FCOS in Python, with access to pre-trained weights for a number of different models. The package provides PyTorch implementations for using training, evaluation, and prediction in your own systems. The package is easily installable with `conda`, and can also be installed via `pip` if you'd prefer to manually manage dependencies.

Our code is free to use, and licensed under BSD-3. We simply ask that you [cite our work](#citing-our-work) if you use FCOS in your own research.

## Related resources

This repository brings the work from a number of sources together. Please see the links below for further details:

- our original paper: ["FCOS: Fully convolutional one-stage object detection"](#citing-our-work)
- our latest paper: ["FCOS: A Simple and Strong Anchor-free Object Detector"](#citing-our-work)
- the original FCOS implementation: [https://github.com/tianzhi0549/FCOS](https://github.com/tianzhi0549/FCOS)
- implementation in the AdelaiDet toolbox: [https://github.com/aim-uofa/AdelaiDet/blob/master/configs/FCOS-Detection/README.md#fcos-real-time-models](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/FCOS-Detection/README.md#fcos-real-time-models)

## Installing FCOS

We offer three methods for installing FCOS:

1. [Through our Conda package](#conda): single command installs everything including system dependencies (recommended)
2. [Through our pip package](#pip): single command installs FCOS and Python dependences, you take care of system dependencies
3. [Directly from source](#from-source): allows easy editing and extension of our code, but you take care of building and all dependencies

### Conda

The only requirement is that you have [Conda installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) on your system, and [NVIDIA drivers installed](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&=Ubuntu&target_version=20.04&target_type=deb_network) if you want CUDA acceleration. We provide Conda packages through [Conda Forge](https://conda-forge.org/), which recommends adding their channel globally with strict priority:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Once you have access to the `conda-forge` channel, FCOS is installed by running the following from inside a [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```
u@pc:~$ conda install fcos
```

We don't explicitly lock the PyTorch installation to a CUDA-enabled version to maximise compatibility with our users' possible setups. If you wish to ensure a CUDA-enabled PyTorch is installed, please use the following installation line instead:

```
u@pc:~$ conda install pytorch=*=*cuda* fcos
```

You can see a list of our Conda dependencies in the [FCOS feedstock's recipe](https://github.com/conda-forge/fcos-feedstock/blob/master/recipe/meta.yaml).

### Pip

Before installing via `pip`, you must have the following system dependencies installed if you want CUDA acceleration:

- NVIDIA drivers
- CUDA

Then FCOS, its custom CUDA code, and all of its Python dependencies, can be installed via:

```
u@pc:~$ pip install fcos
```

### From source

Installing from source is very similar to the `pip` method above, accept we install from a local copy. Simply clone the repository, enter the directory, and install via `pip`:

```
u@pc:~$ pip install -e .
```

_Note: the editable mode flag (`-e`) is optional, but allows you to immediately use any changes you make to the code in your local Python ecosystem._

We also include scripts in the `./scripts` directory to support running FCOS without any `pip` installation, but this workflow means you need to handle all system and Python dependencies manually.

## Using FCOS

TODO

### FCOS from the command line

### FCOS Python API

## Citing our work

If using FCOS in your work, please cite [our original ICVV paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.pdf):

```bibtex
@inproceedings{tian2019fcos,
  title={FCOS: Fully convolutional one-stage object detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9627--9636},
  year={2019}
}
```

Or our [more recent TPAMI journal](https://arxiv.org/pdf/2006.09214.pdf) with further details of our work:

```bibtex
@article{tian2021fcos,
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={FCOS: A Simple and Strong Anchor-free Object Detector},
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2020.3032166}}
```
