import torch
import warnings
# pick dataset loader, model, and trainer by importing one
from sensorium.datasets import static_loaders as loader_builder
from sensorium.models import stacked_core_full_gauss_readout as model_builder
from sensorium.training import standard_trainer as trainer

warnings.filterwarnings('ignore')

MODEL_NAME = 'sota_model'
SEED = 18913674  # chosen by keyboard-mashing

# NOTE: using any other cuda device is tricky. The dataloaders call x.cuda() on the data, but this always defaults to
# cuda:0 unless some other environment config is done
DEVICE = 'cuda:0'

MICE = ["21067-10-18", "23964-4-22", "22846-10-16", "26872-17-20", "23343-5-17", "27204-5-13", "23656-14-22"]

filenames = [f"notebooks/data/static{m}-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip" for m in MICE]

print("Creating dataloaders...")
dataloaders = loader_builder(paths=filenames,
                             normalize=True,
                             include_behavior=True,
                             include_eye_position=True,
                             batch_size=128,
                             scale=.25)

print("Creating model...")
model = model_builder(dataloaders=dataloaders,
                      seed=SEED,
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
                      shifter=True)

print("Training...")
validation_score, trainer_output, state_dict = trainer(
    model=model,
    dataloaders=dataloaders,
    seed=SEED,
    max_iter=200,
    verbose=True,
    track_training=True,
    lr_decay_steps=4,
    avg_loss=False,
    lr_init=0.009,
    device=DEVICE)

print("Saving...")
torch.save(model.state_dict(), f'./model_checkpoints/sensorium_p_{MODEL_NAME}_{SEED}.pth')
