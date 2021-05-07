import os
import torch

from .core.config import cfg
from .core.data import make_data_loader
from .core.modeling.detector import build_detection_model
from .core.utils.checkpoint import DetectronCheckpointer
from .core.solver import make_lr_scheduler, make_optimizer
from .core.utils.collect_env import collect_env_info
from .core.utils.logger import setup_logger
from .core.utils.miscellaneous import mkdir


class Fcos(object):

    def __init__(self,
                 *,
                 config_file=None,
                 config_list=None,
                 gpu_id=0,
                 load_snapshot=None,
                 model_seed=0,
                 name='fcos'):
        # Apply sanitised args
        # TODO sanitising...
        self.config_file = config_file
        self.config_list = config_list
        self.gpu_id = gpu_id
        self.model_seed = model_seed
        self.name = name

        # TODO handle loading arguments...

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
        # TODO

    def evaluate(self,
                 *,
                 dataset_dir=None,
                 output_directory='./eval_output',
                 output_images=False):
        pass

    def predict(self, *, image=None, image_file=None, output_file=None):
        pass

    def train(self,
              dataset_name,
              *,
              output_directory=os.path.expanduser('~/fcos-output')):
        pass
