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


class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z*z)/(2*self.sigma*self.sigma) - log_s_total - log_det_W_total
        return loss/(z.size(0)*z.size(1)*z.size(2))


class IsolateLoss(nn.Module):
    def __init__(self):
        super(IsolateLoss, self).__init__()

    def forward(self, input, output, probability_weights):
        ###
        # Ideally we g
        residual = input - output
        loss = probability_weights * torch.pow(residual, 2)
        return loss


class SphereLoss(nn.Module):
    def __init__(self):
        super(SphereLoss ,self).__init__()

    def forward(self, *input):
        return None