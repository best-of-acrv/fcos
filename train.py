import os
import argparse
import torch

# helper imports
from helpers.trainer import Trainer
from helpers.model_helper import find_checkpoint

# fcos core imports
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir

# get arguments
parser = argparse.ArgumentParser(description='PyTorch FCOS')
parser.add_argument('--name', type=str, default='fcos', help='custom prefix for naming model')
parser.add_argument('--model', type=str, default='FCOS_imprv_R_50_FPN_1x', help='name of model to use')
parser.add_argument("--config-file", default="configs/fcos/fcos_imprv_R_50_FPN_1x.yaml", metavar="FILE", help="path to config file")
parser.add_argument('--save_directory', type=str, default='runs', help='save model directory')
parser.add_argument('--load_directory', type=str, default=None, help='load model directory')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
args.save_directory = os.path.join(args.save_directory, args.name)

# GPU settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.manual_seed(args.seed)

if __name__ == '__main__':

    # load in config file
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # setup logger
    output_dir = cfg.OUTPUT_DIR
    output_dir = os.path.join(cfg.OUTPUT_DIR, args.save_directory)
    if output_dir:
        mkdir(output_dir)
    logger = setup_logger("fcos_core", output_dir, distributed_rank=0)
    logger.info(args)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # build detectron backbone model
    device = torch.device(cfg.MODEL.DEVICE)
    model = build_detection_model(cfg)
    model.to(device)

    # create training optimizer and scheduler
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # setup checkpointer
    arguments = {}
    arguments["iteration"] = 0
    checkpointer = DetectronCheckpointer(cfg,
                                         model=model,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         save_dir=output_dir,
                                         save_to_disk=True)

    # load previous model (if any)
    if args.load_directory:
        args.load_directory = os.path.join(args.load_directory, args.name)
        weight_dir = os.path.join(args.load_directory)
        model_name = find_checkpoint(weight_dir)
        _ = checkpointer.load(model_name)
    else:
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
        arguments.update(extra_checkpoint_data)

    # create data loader
    iou_types = ('bbox',)
    data_loader = make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=arguments["iteration"])
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # Initialise model trainer and train
    trainer = Trainer(optimizer, scheduler, checkpointer, device, checkpoint_period, arguments)
    trainer.train(model, data_loader)