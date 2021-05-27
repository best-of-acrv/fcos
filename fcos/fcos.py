import acrv_datasets
import numpy as np
import os
import PIL.Image as Image
import torch

from .core.config import cfg
from .core.data import make_data_loader
from .core.modeling.detector import build_detection_model
from .core.utils.checkpoint import DetectronCheckpointer
from .core.solver import make_lr_scheduler, make_optimizer
from .core.utils.checkpoint import DetectronCheckpointer
from .core.utils.collect_env import collect_env_info
from .core.utils.logger import setup_logger
from .core.utils.miscellaneous import mkdir
from .evaluator import Evaluator
from .helpers import config_by_name
from .helpers import download_model

PRETRAINED_MODELS = {
    'FCOS_imprv_R_50_FPN_1x':
        'https://cloudstor.aarnet.edu.au/plus/s/ZSAqNJB96hA71Yf/download',
    'FCOS_imprv_dcnv2_R_50_FPN_1x':
        'https://cloudstor.aarnet.edu.au/plus/s/plKgHuykjiilzWr/download',
    'FCOS_imprv_R_101_FPN_2x':
        'https://cloudstor.aarnet.edu.au/plus/s/hTeMuRa4pwtCemq/download',
    'FCOS_imprv_dcnv2_R_101_FPN_2x':
        'https://cloudstor.aarnet.edu.au/plus/s/xq2Ll7s0hpaQycO/download',
    'FCOS_imprv_X_101_32x8d_FPN_2x':
        'https://cloudstor.aarnet.edu.au/plus/s/WZ0i7RZW5BRpJu6/download',
    'FCOS_imprv_dcnv2_X_101_32x8d_FPN_2x':
        'https://cloudstor.aarnet.edu.au/plus/s/08UK0OP67TogLCU/download',
    'FCOS_imprv_X_101_64x4d_FPN_2x':
        'https://cloudstor.aarnet.edu.au/plus/s/rKOJtwvJwcKVOz8/download',
    'FCOS_imprv_dcnv2_X_101_64x4d_FPN_2x':
        'https://cloudstor.aarnet.edu.au/plus/s/jdtVmG7MlugEXB7/download'
}


class Fcos(object):
    # TODO add dataset list
    DATASETS = ['coco/minival2014']

    def __init__(
            self,
            *,
            config_file=config_by_name('FCOS_imprv_dcnv2_R_50_FPN_1x.yaml'),
            config_list=None,
            gpu_id=0,
            load_pretrained='FCOS_imprv_dcnv2_R_50_FPN_1x',
            load_snapshot=None,
            model_seed=0,
            name='fcos'):
        # Apply sanitised args
        self.config_file = config_file
        self.config_list = config_list
        self.gpu_id = gpu_id
        self.model_seed = model_seed
        self.name = name

        self.load_pretrained = (None if load_pretrained is None else
                                _sanitise_arg(load_pretrained,
                                              'load_pretrained',
                                              PRETRAINED_MODELS.keys(),
                                              lower=False))
        self.load_snapshot = load_snapshot

        # Try setting up GPU integration
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.manual_seed(self.model_seed)

        # Load and merge configuration, then freeze it
        if self.config_file is not None:
            cfg.merge_from_file(self.config_file)
        if self.config_list is not None and self.config_list:
            cfg.merge_from_list(self.config_list)
        cfg.freeze()

        # Load model based on the specified parameters
        self.model = build_detection_model(cfg)
        self.checkpointer = DetectronCheckpointer(cfg, self.model)
        print("\nLOADING PRE-TRAINED WEIGHTS INTO DETECTRON MODEL:")
        _load_pretrained(self.load_pretrained, self.checkpointer)
        if self.load_snapshot:
            print("\nLOADING SNAPSHOT INTO DETECTRON MODEL:")
            _load_snapshot(self.load_snapshot, self.checkpointer)
        self.model.to(cfg.MODEL.DEVICE)

    def evaluate(self,
                 *,
                 dataset_name=None,
                 dataset_dir=None,
                 output_directory='./eval_output',
                 output_images=False):
        # Perform argument validation
        if dataset_name is not None:
            dataset_name = _sanitise_arg(dataset_name, 'dataset_name',
                                         Fcos.DATASETS)

        # Load in the dataset
        cfg.defrost()
        cfg.DATASETS.TEST = ('coco/minival2014',)
        cfg.freeze()
        data_loader = _load_datasets(dataset_name, dataset_dir)

        # Perform the requested evaluation

        pass

    def predict(self,
                *,
                image=None,
                image_file=None,
                output_file=None,
                show_mask_heatmaps_if_available=False):
        # Handle input arguments
        if image is None and image_file is None:
            raise ValueError("Only one of 'image' or 'image_file' can be "
                             "used in a call, not both.")
        elif image is not None and image_file is not None:
            raise ValueError("Either 'image' or 'image_file' must be provided")
        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Obtain the input image
        img = (np.array(Image.open(image_file))[:, :, ::-1]
               if image_file else image)

        # Perform the forward pass
        self.model.eval()
        out_img, out_boxes = Evaluator(
            cfg, show_mask_heatmaps=show_mask_heatmaps_if_available
        ).run_on_opencv_image(img, self.model)

        # Save the file if requested, & return the output
        if output_file:
            Image.fromarray(out_img[:, :, ::-1]).save(output_file)
        return out_boxes

    def train(self,
              dataset_name,
              *,
              output_directory=os.path.expanduser('~/fcos-output')):
        # Perform argument validation / set defaults
        dataset_name = _sanitise_arg(dataset_name, 'dataset_name',
                                     Fcos.DATASETS)
        pass


def _load_datasets(dataset_dir, is_train=False, quiet=False):
    # Print some verbose information
    if not quiet:
        print("\nGETTING DATASET:")
    if dataset_dir is None:
        # TODO translate voc into all the required datasets (i.e. this
        # should handle multiple dataset_dirs)
        dataset_dir = acrv_datasets.get_datasets_directory()
    if not quiet:
        print("Using 'dataset_dir': %s" % dataset_dir)

    return make_data_loader(cfg,
                            datasets_dir=dataset_dir,
                            is_train=is_train,
                            is_distributed=False)


def _load_pretrained(pretrained_name, checkpointer):
    checkpointer.load(
        download_model(pretrained_name, PRETRAINED_MODELS[pretrained_name]))


def _load_snapshot(snapshot_path, checkpointer):
    checkpointer.load(snapshot_path)


def _sanitise_arg(value, name, supported_list, lower=True):
    ret = value.lower() if lower and type(value) is str else value
    if ret not in supported_list:
        raise ValueError("Invalid '%s' provided. Supported values are one of:"
                         "\n\t%s" % (name, supported_list))
    return ret
