# README #

This repository provides is intended to produce the results from the paper **FCOS: Fully Convolutional One-Stage Object Detection**, 
which is available [here](https://arxiv.org/abs/1904.01355)

> Tian, Zhi, et al. "Fcos: Fully convolutional one-stage object detection." 
> Proceedings of the IEEE international conference on computer vision. 2019.

This repository is designed to provide out-of-the-box functionality for evaluation and training of
FCOS models as specified in its paper, with as little overhead as possible. Models were adapted from
the official [FCOS](https://github.com/tianzhi0549/FCOS) repository.

## Setup ##
To create the Conda environment to run code from this repository:

```
$ conda config --set channel_priority strict
$ conda env create -f requirements.yml
```
This should set up the conda environment with all prerequisites for running this code. Activate this Conda
environment using the following command:
```
$ conda activate pytorch-fcos
```

### Install COCO API ###
Clone and install the official COCO API Git Repository:
```
$ git clone https://github.com/cocodataset/cocoapi
$ cd cocoapi/PythonAPI
$ make
$ python setup.py install
```

### Install FCOS core ###
```
$ python setup.py build develop --no-deps
```

### Download MS COCO ###
Download the MS COCO dataset using the `download_data.sh` located in the `data_utils`folder.
```
$ sh data_utils/download_data.sh
```
After downloading and unzipping, you will see a folder named `dataset` which contains the following folders:
* `train2014`: training dataset containing 118287 JPEG images
* `val2014`: validation dataset containing 5000 JPEG images
* `annotations`: contains 8 json files containing corresponding label annotations

We will use the annotations from the object detection task; these are the files labelled ``instances`` in the annotations
folder (e.g. ``instances_train2014`` and ``instances_val2014``). For FCOS, the custom training and validation dataset split
is required.

## Evaluation ##
To evaluate with one of the pretrained models, run ```eval.py```.
 
You can specify the desired model CNN backbone (ResNet-50 or ResNet-101)
For example, to evaluate using a ResNet-50 backbone and generate sample detection images, run the following command from the root directory:
```python eval.py --sample_images True --config-file configs/fcos/FCOS_imprv_R_50_FPN_1x.yaml MODEL.WEIGHT FCOS_imprv_R_50_FPN_1x.pth```

Please check the ```configs``` folder for currently supported models. Pretrained FCOS models will be automatically downloaded and stored in the ```pretrained/models``` directory.
Alternatively, if you wish to load your own pretrained model, you can do this by specifying a load directory (e.g.):
```python eval.py --load_directory=runs/mymodel```

## Training ##
To train your own FCOS model, run ```train.py```. 

By default to assist with training, models will be preloaded with ImageNet weights for the backbone ResNet encoder. 
For example, to train using a RefineNet-50 model, run the following command from the root directory:
```python train.py --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml```