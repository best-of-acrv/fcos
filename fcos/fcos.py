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
from .trainer import Trainer


class Fcos(object):
    # TODO add rest of datasets from ./core/config/paths_catalog.py
    DATASETS = {
        'coco/train2017': [],
        'coco/val2017': [],
        'coco/train2014': [],
        'coco/val2014': [],
        'coco/minival2014': ['coco/val2014'],
        'coco/valminusminival2014': ['coco/val2014'],
    }

    PRETRAINED_MODELS = {
        'FCOS_imprv_R_50_FPN_1x':
            'https://cloudstor.aarnet.edu.au/plus/s/6yBHy9eXMp8kxXE/download',
        'FCOS_imprv_dcnv2_R_50_FPN_1x':
            'https://cloudstor.aarnet.edu.au/plus/s/kWpZoUvgTpn7u6A/download',
        'FCOS_imprv_R_101_FPN_2x':
            'https://cloudstor.aarnet.edu.au/plus/s/jgfmAliI86BSzHc/download',
        'FCOS_imprv_dcnv2_R_101_FPN_2x':
            'https://cloudstor.aarnet.edu.au/plus/s/Z8GKgxTbVzQKd9D/download',
        'FCOS_imprv_X_101_32x8d_FPN_2x':
            'https://cloudstor.aarnet.edu.au/plus/s/RmpDdqJRNFy6eDm/download',
        'FCOS_imprv_dcnv2_X_101_32x8d_FPN_2x':
            'https://cloudstor.aarnet.edu.au/plus/s/feJzyj4OgYl3vlq/download',
        'FCOS_imprv_X_101_64x4d_FPN_2x':
            'https://cloudstor.aarnet.edu.au/plus/s/gfZWtxG8qyFBrtf/download',
        'FCOS_imprv_dcnv2_X_101_64x4d_FPN_2x':
            'https://cloudstor.aarnet.edu.au/plus/s/eqbxb8owOu5J97O/download'
    }

    def __init__(self,
                 *,
                 config_file=None,
                 config_list=None,
                 gpu_id=0,
                 load_checkpoint=None,
                 load_pretrained='FCOS_imprv_R_50_FPN_1x',
                 model_seed=0,
                 name='fcos'):
        # Apply sanitised args
        self.config_list = config_list
        self.gpu_id = gpu_id
        self.model_seed = model_seed
        self.name = name

        self.load_pretrained = (None if load_pretrained is None else
                                _sanitise_arg(load_pretrained,
                                              'load_pretrained',
                                              Fcos.PRETRAINED_MODELS.keys(),
                                              lower=False))
        self.load_checkpoint = (None if load_checkpoint is None else
                                os.path.expanduser(load_checkpoint))

        if self.load_pretrained is None and config_file is None:
            self.config_file = config_by_name('FCOS_imprv_R_50_FPN_1x')
        elif self.load_pretrained is not None:
            self.config_file = config_by_name(self.load_pretrained)
            print("\nOverriding 'config_file' selection to match "
                  "'load_pretrained':\n%s" % self.config_file)
        else:
            self.config_file = config_file

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
        if self.load_checkpoint:
            print("\nLOADING SNAPSHOT INTO DETECTRON MODEL:")
            _load_checkpoint(self.load_checkpoint, self.checkpointer)
        self.model.to(cfg.MODEL.DEVICE)

    def evaluate(self,
                 *,
                 dataset_name=None,
                 dataset_dir=None,
                 output_directory='./eval_output'):
        # Perform argument validation
        if dataset_name is not None:
            dataset_name = _sanitise_arg(dataset_name, 'dataset_name',
                                         Fcos.DATASETS.keys())
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)

        # Apply configuration settings
        cfg.defrost()
        if dataset_name is not None:
            cfg.DATASETS.TEST = (dataset_name,)
        cfg.OUTPUT_DIR = output_directory
        cfg.freeze()

        # Load evaluation datasets
        data_loader = _load_datasets(dataset_dir, is_train=False)

        # Start the logging service
        print("\nEVALUATING PERFORMANCE:")
        l = setup_logger("fcos_core", cfg.OUTPUT_DIR, distributed_rank=0)
        l.info("Dumping env info (may take some time):")
        l.info("\n" + collect_env_info())
        l.info("Running with config:\n%s" % cfg)

        # Perform the requested evaluation
        e = Evaluator(cfg)
        e.inference(
            self.model,
            data_loader[0],
            dataset_name=cfg.DATASETS.TEST[0],
            iou_types=('bbox',),
            box_only=(False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else
                      cfg.MODEL.RPN_ONLY),
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=cfg.OUTPUT_DIR)

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
            output_file = os.path.expanduser(output_file)
            dn = os.path.dirname(output_file)
            if dn:
                os.makedirs(dn, exist_ok=True)

        # Obtain the input image
        img = (np.array(Image.open(os.path.expanduser(image_file)))[:, :, ::-1]
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
              *,
              checkpoint_period=None,
              dataset_name=None,
              dataset_dir=None,
              output_directory=os.path.expanduser('~/fcos-output')):
        # Perform argument validation / set defaults
        if dataset_name is not None:
            dataset_name = _sanitise_arg(dataset_name, 'dataset_name',
                                         Fcos.DATASETS.keys())
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)

        # Apply configuration settings
        cfg.defrost()
        if checkpoint_period is not None:
            cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
        if dataset_name is not None:
            cfg.DATASETS.TRAIN = (dataset_name,)
        cfg.OUTPUT_DIR = output_directory
        cfg.freeze()

        # Configure the model for training
        data_loader = _load_datasets(dataset_dir, is_train=True)
        self.checkpointer.cfg = cfg
        self.checkpointer.optimizer = make_optimizer(cfg, self.model)
        self.checkpointer.scheduler = make_lr_scheduler(
            cfg, self.checkpointer.optimizer)
        self.checkpointer.save_dir = cfg.OUTPUT_DIR
        self.checkpointer.save_to_disk = True

        # Start the logging service
        print("\nPERFORMING TRAINING:")
        l = setup_logger("fcos_core", cfg.OUTPUT_DIR, distributed_rank=0)
        l.info("Dumping env info (may take some time):")
        l.info("\n" + collect_env_info())
        l.info("Running with config:\n%s" % cfg)

        # Start a model trainer
        return Trainer(self.checkpointer, cfg.MODEL.DEVICE,
                       cfg.SOLVER.CHECKPOINT_PERIOD, {
                           'iteration': 0
                       }).train(self.model, data_loader)


def _load_datasets(dataset_dir, is_train=False, quiet=False):
    # Print some verbose information
    if not quiet:
        print("\nGETTING DATASET:")
    if dataset_dir is None:
        dataset_dir = acrv_datasets.get_datasets_directory()
    if not quiet:
        print("Using 'dataset_dir': %s" % dataset_dir)

    # Build a list of requested datasets (including dependencies), and ensure
    # they are all available locally
    datasets = [
        [d] + Fcos.DATASETS[d]
        for d in (cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST)
    ]
    acrv_datasets.get_datasets([d for ds in datasets for d in ds],
                               datasets_directory=dataset_dir)
    if not quiet:
        print("\n")

    # Return a data loader for the requested dataset
    return make_data_loader(cfg,
                            datasets_dir=dataset_dir,
                            is_train=is_train,
                            is_distributed=False)


def _load_checkpoint(checkpoint_path, checkpointer):
    checkpointer.load(checkpoint_path)


def _load_pretrained(pretrained_name, checkpointer):
    checkpointer.load(
        download_model(pretrained_name,
                       Fcos.PRETRAINED_MODELS[pretrained_name]))


def _sanitise_arg(value, name, supported_list, lower=True):
    ret = value.lower() if lower and type(value) is str else value
    if ret not in supported_list:
        raise ValueError("Invalid '%s' provided. Supported values are one of:"
                         "\n\t%s" % (name, supported_list))
    return ret
