import os
import pkg_resources
import sys
from six.moves import urllib
import torch

_CACHE_LOCATION = '.cache'
_CONFIGS_LOCATION = 'configs'


def cache_location():
    return pkg_resources.resource_filename(__name__, _CACHE_LOCATION)


def config_by_name(name, must_exist=True):
    fn = pkg_resources.resource_filename(
        __name__, os.path.join(_CONFIGS_LOCATION, '%s.yaml' % name))
    if must_exist and not os.path.exists(fn):
        raise ValueError('No config exists with the filepath:\n\t%s' % fn)
    return fn


def download_model(model_name, model_url, model_dir=None, map_location=None):
    print("Fetching pre-trained model '%s' from:\n\t%s" %
          (model_name, model_url))
    if model_dir is None:
        # Try use PyTorch's recommendations by default, otherwise fallback to
        # our package level cache
        model_dir = os.getenv('TORCH_MODEL_ZOO')
        if model_dir is None and os.getenv('TORCH_HOME') is not None:
            model_dir = os.path.join(
                os.path.expanduser(str(os.getenv('TORCH_HOME'))), 'models')
        if model_dir is None:
            model_dir = os.path.join(cache_location(), 'pretrained')
            print("Falling back to package cache for locating cached "
                  "pretrained models:\n\t%s" % model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading to {}\n'.format(cached_file))
        urllib.request.urlretrieve(url, cached_file)
    # return torch.load(cached_file, map_location=map_location)
    return cached_file
