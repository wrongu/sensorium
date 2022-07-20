import torch
from torch import nn
import torch.nn.functional as F


class SpatialTransformerNetwork(nn.Module):
    """Module implementing the STN, following code from
    https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

    Bonus feature: we sample grids at two resolutions:
        - Major resolution = output size.
        - Sample resolution: how many samples per output pixel, allowing for blur.
    """

    def __init__(self,  major_resolution=100, sample_resolution=11, sample_scale=1.):
        """
        major_resolution: size of output width/height
        sample_resolution: number of input pixels to average/blur together per output pixel, in each dimension
        sample_scale: number of major grid spaces that each average/blur window averages over. Overlap between windows
            only if sample_scale > 1.
        """
        super(SpatialTransformerNetwork, self).__init__()

        if major_resolution < 2:
            raise ValueError(f"Bare minimum, major_resolution must be >= 2 but was {major_resolution}")

        if sample_resolution < 1:
            raise ValueError(f"Bare minimum, sample_resolution must be >= 1 but was {sample_resolution}")

        self._major = major_resolution
        self._sample = sample_resolution
        self._sample_scale = sample_scale

    def forward(self, x, theta):
        """Sample image x in bounding box defined by theta. Size of theta must be (batch, 2, 3)

        Theta is an affine-transformation matrix. Think of the coordinates of the four corners of the input image as
        having (x,y) coordinates of (-1,-1)...(+1,+1). Then, the sample grid coordinates are theta @ [x, y, 1].T.

        If grid coordinates are outside the [-1,+1]^2 bounding box, returned values are zeros.

        Returns: [batch, self._major, self._major) size image patches cropped from x_image
        """
        b, c, _, _ = x.size()
        out = x.new_zeros((b, c, self._major, self._major))
        # Construct many grids - one per point in the blur grid
        blur_xy, blur_w = _gaussian_blur_grid(self._sample, self._sample)
        # bx and by are in [-1, 1] but we would like them to span the width of major grid points. With n major
        # grid points, there are n-1 spans between points. So, the [-1,1] range of bx should be scaled by a factor
        # of 1/(n-1)
        scale_blur_factor = self._sample_scale / (self._major - 1)
        for w, (bx, by) in zip(blur_w, blur_xy):
            major_grid = F.affine_grid(_shift_affine_grid(theta, bx * scale_blur_factor, by * scale_blur_factor),
                                       size=[b, c, self._major, self._major],
                                       align_corners=False)
            out += w * F.grid_sample(x, major_grid,
                                     mode="bilinear",
                                     align_corners=False,
                                     padding_mode="zeros")
        return out


def _gaussian_blur_grid(nx, ny, n_sigma=4.):
    # In case nx = 1, we want xx to be 0. So, use nx+2 to create the grid but retain only [1:-1] indices.
    xx, yy = torch.meshgrid([torch.linspace(-1.0, +1.0, nx+2)[1:-1],
                             torch.linspace(-1.0, +1.0, ny+2)[1:-1]])
    ww = torch.exp(-1/2*(xx*xx + yy*yy)*n_sigma**2)
    ww = ww / torch.sum(ww)
    xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return xy, ww.flatten()


def _shift_affine_grid(theta, dx, dy, in_place=False):
    """Given 'theta', a (?, 2, 3) sized descriptor of an affine grid, return a new 'theta' shifted by dx and dy
    units *in coordinates of theta*, so a shift of dx=1 would result in a new grid abutting the original one to the
    right, with the same rotation/shear/etc.
    """
    if not in_place:
        theta = theta.clone()
    # theta[:, :2, :2] is a linear transform (shear, rotate, scale, etc) and theta[:, :2, 2] is the translation. We
    # would like to add to the translation part to shift the grid. The amount we shift by in 'x' is the first column
    # of the linear transform, and the amount we shift by in 'y' is the second column.
    shift_x, shift_y = theta[:, :, 0] * dx, theta[:, :, 1] * dy
    theta[:, :, 2] = theta[:, :, 2] + shift_x + shift_y
    return theta
