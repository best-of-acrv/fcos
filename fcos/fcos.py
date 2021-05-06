import os


class Fcos(object):

    def __init__(self, *, gpu_id=0):
        pass

    def evaluate(self, dataset_name, *, dataset_dir=None):
        pass

    def predict(self, *, image=None):
        pass

    def train(self,
              dataset_name,
              *,
              output_directory=os.path.expanduser('~/fcos-output')):
        pass
