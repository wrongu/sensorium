import torch
import warnings
from torch import nn
from neuralpredictors.measures.modules import Corr, AvgCorr, PoissonLoss, PoissonLoss3d, ExponentialLoss, AnscombeMSE


class PoissonLikeGaussianLoss(nn.Module):
    def __init__(self, bias=1e-1, per_neuron=False, avg=True):
        """
        Computes Poisson-like Gaussian loss (squared error normalized by variance, where variance = mean like in a
        Poisson)

        Implemented by Richard Lange but largely copied from PoissonLoss

        Args:
            bias (float, optional): Value used to numerically stabilize evalution of the log-likelihood. Added to variance (denominator of log loss)
            per_neuron (bool, optional): If set to True, the average/total Poisson loss is returned for each entry of the last dimension (assumed to be enumeration neurons) separately. Defaults to False.
            avg (bool, optional): If set to True, return mean loss. Otherwise returns the sum of loss. Defaults to True.
        """
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron
        self.avg = avg
        if self.avg:
            warnings.warn("PoissonLikeGausianLoss is averaged per batch. It's recommended to use `sum` instead")

    def forward(self, output, target):
        target = target.detach()
        variance = torch.clip(output, 0., None) + self.bias
        # loss is negative log probability under a gaussian with mean 'output' and variance 'output+bias', but with
        # output clipped so that variance is at least 'bias'
        loss = 1/2*(output - target)**2 / variance + 1/2*torch.log(variance).sum()
        if not self.per_neuron:
            return loss.mean() if self.avg else loss.sum()
        else:
            loss = loss.view(-1, loss.shape[-1])
            return loss.mean(dim=0) if self.avg else loss.sum(dim=0)


__all__ = ["Corr", "AvgCorr", "PoissonLoss", "PoissonLoss3d", "ExponentialLoss", "AnscombeMSE", "PoissonLikeGaussianLoss"]