import argparse
import os
import re
import sys
import textwrap

from .fcos import Fcos
from .helpers import config_by_name


class ShowNewlines(argparse.ArgumentDefaultsHelpFormatter,
                   argparse.RawDescriptionHelpFormatter):

    def _fill_text(self, text, width, indent):
        return ''.join([
            indent + i for ii in [
                textwrap.fill(
                    s, width, drop_whitespace=False, replace_whitespace=False)
                for s in text.splitlines(keepends=True)
            ] for i in ii
        ])


def main():
    # Parse command line arguments
    p = argparse.ArgumentParser(
        prog='fcos',
        formatter_class=ShowNewlines,
        description="Fully convolutional one-stage object detection (FCOS).\n\n"
        "Dataset interaction is performed through the acrv_datasets package. "
        "Please see it for details on downloading datasets, accessing them, "
        "and changing where they are stored.\n\n"
        "For full documentation of FCOS, plea see "
        "https://github.com/best-of-acrv/fcos.")

    p_parent = argparse.ArgumentParser(add_help=False)
    p_parent.add_argument(
        '--config-file',
        default=None,
        help='YAML file from which to load FCOS configuration')
    p_parent.add_argument(
        '--config-list',
        default=None,
        help='Comma separated list of key-value config pairs '
        '(e.g. MODEL.WEIGHTS=weights.pth,SOLVER.WEIGHT_DECAY=0.001)')
    p_parent.add_argument('--gpu-id',
                          default=0,
                          type=int,
                          help="ID of GPU to use for model")
    p_parent.add_argument('--load-checkpoint',
                          default=None,
                          help="Checkpoint location from which to load weights"
                          " (overrides --load-pretrained)")
    p_parent.add_argument('--load-pretrained',
                          default='FCOS_imprv_R_50_FPN_1x',
                          choices=Fcos.PRETRAINED_MODELS.keys(),
                          help="Load these pre-trained weights in at startup")
    p_parent.add_argument('--model-seed',
                          default=0,
                          type=int,
                          help="Seed used for model training")
    p_parent.add_argument('--name',
                          default='fcos',
                          help="Name to give FCOS model")
    sp = p.add_subparsers(dest='mode')

    p_eval = sp.add_parser('evaluate',
                           parents=[p_parent],
                           formatter_class=ShowNewlines,
                           help="Evaluate a model's performance against a "
                           "specific dataset")
    p_eval.add_argument('--dataset-name',
                        default=None,
                        help="Name of the dataset to use from 'acrv_datasets "
                        "--supported_datasets' (value in config will be used "
                        "if not supplied)")
    p_eval.add_argument('--dataset-dir',
                        default=None,
                        help="Search this directory for datasets instead "
                        "of the current default in 'acrv_datasets'")
    p_eval.add_argument('--output-directory',
                        default='.',
                        help="Directory to save evaluation results")

    p_pred = sp.add_parser('predict',
                           parents=[p_parent],
                           formatter_class=ShowNewlines,
                           help="Use a model to detect objects in a given "
                           "input image")
    p_pred.add_argument('image_file', help="Filename for input image")
    p_pred.add_argument('--output-file',
                        default='./output.jpg',
                        help="Filename used for saving the output image")
    p_pred.add_argument('--show-mask-heatmaps-if-available',
                        default=False,
                        action='store_true',
                        help="Overlay heatmaps on the output image "
                        "(only when the model provides them)")

    p_train = sp.add_parser('train',
                            parents=[p_parent],
                            formatter_class=ShowNewlines,
                            help="Train a model from a previous starting "
                            "point using a specific dataset")
    p_train.add_argument('--checkpoint-period',
                         default=None,
                         type=int,
                         help="Frequency with which to make checkpoints in # "
                         "of epochs (value in config will be used if not "
                         "supplied)")
    p_train.add_argument('--dataset-name',
                         help="Name of the dataset to use from 'acrv_datasets "
                         "--supported_datasets' (value in config will be used "
                         "if not supplied)")
    p_train.add_argument('--dataset-dir',
                         default=None,
                         help="Search this directory for datasets instead "
                         "of the current default in 'acrv_datasets'")
    p_train.add_argument('--output-directory',
                         default=os.path.expanduser('~/fcos-output'),
                         help="Location where checkpoints and training "
                         "progress will be stored")

    args = p.parse_args()

    # Print help if no args provided
    if len(sys.argv) == 1:
        p.print_help()
        return

    # Run requested FCOS operations
    f = Fcos(config_file=args.config_file,
             config_list=None if args.config_list is None else re.split(
                 ',|=', args.config_list),
             gpu_id=args.gpu_id,
             load_checkpoint=args.load_checkpoint,
             load_pretrained=args.load_pretrained)
    if args.mode == 'evaluate':
        f.evaluate(dataset_name=args.dataset_name,
                   dataset_dir=args.dataset_dir,
                   output_directory=args.output_directory)
    elif args.mode == 'predict':
        f.predict(image_file=args.image_file,
                  output_file=args.output_file,
                  show_mask_heatmaps_if_available=args.
                  show_mask_heatmaps_if_available)
    elif args.mode == 'train':
        f.train(checkpoint_period=args.checkpoint_period,
                dataset_name=args.dataset_name,
                dataset_dir=args.dataset_dir,
                output_directory=args.output_directory)
    else:
        raise ValueError("Unsupported mode: %s" % args.mode)


if __name__ == '__main__':
    main()
