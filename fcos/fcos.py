import os
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

from .helpers import config_by_name

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
    DATASETS = []

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
        # TODO sanitising if needed?
        self.config_file = config_file
        self.config_list = config_list
        self.gpu_id = gpu_id
        self.model_seed = model_seed
        self.name = name

        # TODO handle loading snapshots and pre-trained models...
        self.load_pretrained = (
            None if load_pretrained is None else _sanitise_arg(
                load_pretrained, 'load_pretrained', PRETRAINED_MODELS.keys()))
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
        if self.load_snapshot:
            print("\nLOADING SNAPSHOT INTO DETECTRON MODEL:")
            _load_snapshot(self.load_snapshot, self.checkpointer)
        else:
            print("\nLOADING PRE-TRAINED WEIGHTS INTO DETECTRON MODEL")
            _load_pretrained(self.load_pretrained, self.checkpointer)
        self.model.to(cfg.MODEL.DEVICE)

    def evaluate(self,
                 *,
                 dataset_dir=None,
                 output_directory='./eval_output',
                 output_images=False):
        # TODO merge from eval.py
        pass

    def predict(self, *, image=None, image_file=None, output_file=None):
        # TODO create using the output_image methodology of eval.py
        pass

    def train(self,
              dataset_name,
              *,
              output_directory=os.path.expanduser('~/fcos-output')):
        # Perform argument validation / set defaults
        dataset_name = _sanitise_arg(dataset_name, 'dataset_name',
                                     Fcos.DATASETS)
        pass


def _load_pretrained(pretrained_path, checkpointer):
    checkpointer.load(os.path.join(pretrained_path, 'models',
                                   cfg.MODEL.WEIGHT))


def _load_snapshot(snapshot_path, checkpointer):
    checkpointer.load(snapshot_path)


def _sanitise_arg(value, name, supported_list):
    ret = value.lower() if type(value) is str else value
    if ret not in [s.lower() for s in supported_list]:
        raise ValueError("Invalid '%s' provided. Supported values are one of:"
                         "\n\t%s" % (name, supported_list))
    return ret
