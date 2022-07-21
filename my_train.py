import torch
import numpy as np
import warnings
from pathlib import Path
# pick dataset loader, model, and trainer by importing one
from sensorium.datasets import static_loaders as loader_builder
from sensorium.models import stacked_core_full_gauss_readout as model_builder
from sensorium.training import standard_trainer as trainer
import argparse

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--tikhanov-readout-by-xyz", default=0., type=float)
parser.add_argument("--loss", default="PoissonLoss", type=str, choices=["PoissonLoss", "PoissonLikeGaussianLoss"])
parser.add_argument("--seed", default=18913674, type=int)
parser.add_argument("--model-name", default="baseline", type=str)
parser.add_argument("--init-model", default=None)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

if args.device.startswith("cuda:"):
    # inside the sensorium package, there are various calls to x.cuda() without specifying a number. Ensure that
    # devices match here by setting the default cuda device number for all of torch.
    torch.cuda.set_device(int(args.device[-1]))

MICE = ["21067-10-18", "23964-4-22", "22846-10-16", "26872-17-20", "23343-5-17", "27204-5-13", "23656-14-22"]

filenames = [f"notebooks/data/static{m}-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip" for m in MICE]

pairwise_neuron_similarities = {}
if args.tikhanov_readout_by_xyz > 0:
    print("Loading distance metadata...")
    # Construct dict of mouse: pairwise distances between neurons
    def pairwise_neuron_dist(mouse_id):
        data_root = Path(f"notebooks/data/static{mouse_id}-GrayImageNet-94c6ff995dac583098847cfecd43e7b6")
        coords_file = data_root  / "meta" / "neurons" / "cell_motor_coordinates.npy"
        coordinates_xyz = torch.tensor(np.load(str(coords_file.resolve())), device=args.device)
        diff_dist = coordinates_xyz[:, None, :] - coordinates_xyz[None, :, :]
        euclidean_dist = torch.sqrt(torch.sum(diff_dist**2, dim=-1))
        return euclidean_dist
    pairwise_neuron_distances = {m: pairwise_neuron_dist(m) for m in MICE}

    # Now convert pairwise distances into a scaled similarity score for each pair of neurons. This score will penalize
    # differences-in-weights.
    def dist2reg(distances, tau=None, scale=1e-1):
        # default lenght scale to median of pairwise distances
        if tau is None:
            i, j = torch.triu_indices(*distances.shape, offset=1)
            tau = torch.median(distances[i, j])
        return scale * torch.exp(-distances / tau)

    pairwise_neuron_similarities = {k: dist2reg(dist, scale=args.tikhanov_readout_by_xyz)
                                    for k, dist in pairwise_neuron_distances.items()}


print("Creating dataloaders...")
dataloaders = loader_builder(paths=filenames,
                             normalize=True,
                             include_behavior=True,
                             include_eye_position=True,
                             batch_size=128,
                             scale=.25)

print("Creating model...")
model = model_builder(dataloaders=dataloaders,
                      seed=args.seed,
                      pad_input=False,
                      stack=-1,
                      layers=4,
                      input_kern=9,
                      gamma_input=6.3831,
                      gamma_readout=0.0076,
                      hidden_kern=7,
                      hidden_channels=64,
                      depth_separable=True,
                      grid_mean_predictor={
                        'type': 'cortex',
                        'input_dimensions': 2,
                        'hidden_layers': 1,
                        'hidden_features': 30,
                        'final_tanh': True},
                      init_sigma=0.1,
                      init_mu_range=0.3,
                      gauss_type='full',
                      shifter=True,
                      spatial_similarity=pairwise_neuron_similarities if args.tikhanov_readout_by_xyz > 0 else None)

if args.init_model is not None:
    print("Attempting to restore model from existing checkpoint...")
    data = torch.load(args.init_model)
    model.load_state_dict(data)

print("Training...")
validation_score, trainer_output, state_dict = trainer(
    loss_function=args.loss,
    model=model,
    dataloaders=dataloaders,
    seed=args.seed,
    max_iter=args.epochs,
    verbose=True,
    track_training=True,
    lr_decay_steps=4,
    avg_loss=False,
    lr_init=0.009,
    device=args.device)

print("Saving...")
save_file = Path(f'./model_checkpoints/sensorium_p_{args.model_name}_{args.seed}.pth')
i = 1
while save_file.exists():
    save_file = Path(f'./model_checkpoints/sensorium_p_{args.model_name}_{args.seed}_{i}.pth')
    i += 1
torch.save(model.state_dict(), str(save_file))
