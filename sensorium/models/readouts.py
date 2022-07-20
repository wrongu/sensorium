import torch
from neuralpredictors.layers.readouts import MultiReadoutSharedParametersBase, FullGaussian2d


def pairwise_squared_euclidean_f_by_n(x):
    xx = torch.sum(x*x, dim=0)
    xy = torch.einsum('...i,...j->ij', x, x)
    return xx.view(1, -1) + xx.view(-1 , 1) - 2*xy


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d

    def __init__(self, *args, **kwargs):
        """Optional 'spatial_similiarity' argument is a #neurons x #neurons matrix, where sim[i,j] is a weight on
        how strongly we regularize that weight[i] == weight[j]
        """
        self.spatial_sim = kwargs.pop('spatial_similarity', None)
        super(MultipleFullGaussian2d, self).__init__(*args, **kwargs)

        if self.spatial_sim is not None:
            for key in self.keys():
                assert key in self.spatial_sim, f"key {key} not in spatial_similarity dict!"

    def regularizer(self, data_key=None, reduction="sum", average=None):
        reg = super(MultipleFullGaussian2d, self).regularizer(data_key, reduction, average)
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]

        # TODO (?) move to per-readout regularizer?
        if self.spatial_sim is not None:
            sim = self.spatial_sim[data_key]
            readout = self[data_key]
            readout_features_f_by_n = readout.features.squeeze()
            diff2 = pairwise_squared_euclidean_f_by_n(readout_features_f_by_n)
            diff2_loss = torch.sum(torch.triu(sim * diff2, 1))
            if average:
                n_neurons = readout_features_f_by_n.shape[-1]
                n_pairs = n_neurons * (n_neurons - 1) / 2
                diff2_loss = diff2_loss / n_pairs
            reg += diff2_loss

        return reg
