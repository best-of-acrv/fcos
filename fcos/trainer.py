import torch.nn as nn
from .core.engine.trainer import do_train


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(nn.Module):

    def __init__(self, checkpointer, device, checkpoint_period, arguments):
        super().__init__()

        self.optimizer = checkpointer.optimizer
        self.scheduler = checkpointer.scheduler
        self.checkpointer = checkpointer
        self.device = device
        self.checkpoint_period = checkpoint_period
        self.arguments = arguments

    def train(self, model, data_loader):

        do_train(model, data_loader, self.optimizer, self.scheduler,
                 self.checkpointer, self.device, self.checkpoint_period,
                 self.arguments)
