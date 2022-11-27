import argparse
import os
import torch


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    default="models/sd-v1-4-full-ema.ckpt",
    help="path to checkpoint of model",
)

parser.add_argument(
    "--outdir",
    type=str,
    default="models",
    help="the directory to put converted models"
)


args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)


ckpt = torch.load(args.ckpt, map_location="cpu")
state = ckpt['state_dict']


autoencoder_state = dict()
for key, val in state.items():
    if key.startswith('first_stage_model'):
        autoencoder_state[key[len('first_stage_model.'):]] = val

torch.save(autoencoder_state, os.path.join(args.outdir, 'autoencoder_kl.pth'))


nnet_state = dict()
for key, val in state.items():
    if key.startswith('model.diffusion_model'):
        nnet_state[key[len('model.diffusion_model.'):]] = val

torch.save(nnet_state, os.path.join(args.outdir, 'sd_nnet.pth'))


nnet_ema_state = dict()
for key in state.keys():
    if key.startswith('model.diffusion_model'):
        ema_key = 'model_ema.' + key[len('model.'):].replace('.', '')
        nnet_ema_state[key[len('model.diffusion_model.'):]] = state[ema_key]

torch.save(nnet_ema_state, os.path.join(args.outdir, 'sd_nnet_ema.pth'))
