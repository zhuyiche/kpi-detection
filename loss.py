import torch.nn as nn
import torch
import torchvision
__all__ = ['SquareErrorLoss']


class SquareErrorLoss(nn.Module):
    def __init__(self):
        super(SquareErrorLoss, self).__init__()

    def forward(self, input, targets=None):
        dist = input - targets
        loss = torch.pow(dist, 2)
        return loss