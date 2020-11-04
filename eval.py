import os
import argparse
import torch
import cv2

# helper imports
from helpers.download_helper import download_model
from helpers.evaluator import Evaluator

# fcos core imports
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir

# get arguments
parser = argparse.ArgumentParser(description='PyTorch FCOS')
parser.add_argument('--name', type=str, default='fcos', help='custom prefix for naming model')
parser.add_argument('--model', type=str, default='FCOS_imprv_R_50_FPN_1x', help='name of model to use')
parser.add_argument("--config-file", default="configs/fcos/fcos_imprv_R_50_FPN_1x.yaml", metavar="FILE", help="path to config file")
parser.add_argument('--save_directory', type=str, default='runs', help='save model directory')
parser.add_argument('--load_directory', type=str, default='pretrained', help='load directory of model')
parser.add_argument('--download_model', type=bool, default=True, help='download a pretrained model')
parser.add_argument('--sample_images', type=bool, default=True, help='create sample detection images')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
args.save_directory = os.path.join(args.save_directory, args.name)

# GPU settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.manual_seed(args.seed)

pretrained_urls = {
    'FCOS_imprv_R_50_FPN_1x': 'https://cloudstor.aarnet.edu.au/plus/s/Jn3WqLvpr2fIxP8/download',
}

if __name__ == '__main__':

    # load in config file
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # setup logger
    save_dir = ""
    logger = setup_logger("fcos_core", save_dir, distributed_rank=0)
    logger.info(cfg)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    # build detectron backbone model
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    # download model
    if args.download_model:
        map_location = None
        if not torch.cuda.is_available():
            map_location = torch.device('cpu')
        url = pretrained_urls[args.model]
        _ = download_model(args.model, url, map_location=map_location)

    # load model checkpoint
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=args.save_directory)
    if args.load_directory:
        weight_dir = os.path.join(args.load_directory, 'models')
        _ = checkpointer.load(os.path.join(weight_dir, cfg.MODEL.WEIGHT))

    # create data loader
    iou_types = ('bbox',)
    data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=False)

    # create output folder
    dataset_name = 'coco_2014_minival'
    output_folder = os.path.join(args.save_directory, "inference", dataset_name)
    mkdir(output_folder)

    # evaluate model
    evaluator = Evaluator(cfg)

    # sample image
    if args.sample_images:

        # check samples folder for images
        sample_dir = 'samples'
        sample_files = os.listdir(sample_dir)

        for sample_file in sample_files:
            img_path = os.path.join(sample_dir, sample_file)
            img = cv2.imread(img_path)
            composite = evaluator.run_on_opencv_image(img, model)
            _ = cv2.imwrite(os.path.join(output_folder, sample_file), composite)

    # inference for eval metrics
    evaluator.inference(model,
                        data_loader_val[0],
                        dataset_name=dataset_name,
                        iou_types=iou_types,
                        box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                        device=cfg.MODEL.DEVICE,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=output_folder
                        )


