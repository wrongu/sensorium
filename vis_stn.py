#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from layers.spatial_transformer import SpatialTransformerNetwork
from PIL import Image

#%%
theta = torch.tensor([[.5, -.3, .05], 
                      [.1, .5, .2]]).view((1, 2, 3))

stn1 = SpatialTransformerNetwork(major_resolution=200, sample_resolution=1)
stn11 = SpatialTransformerNetwork(major_resolution=200, sample_resolution=5, sample_scale=1.5)

im = torch.tensor(np.array(Image.open("/Users/richard/Pictures/Wedding photos/Adina + Richard-1.jpg"))).permute((2, 0, 1)).float()/255
im1 = stn1.forward(im.unsqueeze(0), theta)[0]
im11 = stn11.forward(im.unsqueeze(0), theta)[0]


def crop_gridlines(ax, theta, *args, **kwargs):
    xy1 = torch.tensor([[-1, -1, 1], 
                       [+1, -1, 1], 
                       [+1, +1, 1], 
                       [-1, +1, 1], 
                       [-1, -1, 1]]).float()
    coord = theta[0] @ xy1.T
    # Inverted y!
    ax.plot(coord[0], -coord[1], *args, **kwargs)

#%%
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[0].imshow(im.permute((1, 2, 0)), extent=(-1, +1, -1, +1))
crop_gridlines(ax[0], theta, '-r')
ax[1].imshow(im1.permute((1, 2, 0)))
ax[2].imshow(im11.permute((1, 2, 0)))
plt.show()