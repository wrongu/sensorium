from torch import nn
from layers.spatial_transformer import SpatialTransformerNetwork


class SimpleEyeLNP(nn.Module):
    def __init__(self, n_neurons, crop_resolution=100):
        super(SimpleEyeLNP, self).__init__()
        self._res = crop_resolution
        self._dim = self._res * self._res
        self.saccade_net = nn.Linear(in_features=2, out_features=6)

        self.cropper = SpatialTransformerNetwork(major_resolution=self._res, sample_resolution=1)
        self.lnp = nn.Sequential(
            nn.Linear(in_features=self._dim, out_features=n_neurons),
            nn.Softplus()
        )

    def forward(self, im, eye_xy):
        crop_theta = self.saccade_net(eye_xy).view((-1, 2, 3))
        cropped = self.cropper(im, crop_theta)
        return self.lnp(cropped.view(-1, self._dim))
