import os

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
        pass

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
