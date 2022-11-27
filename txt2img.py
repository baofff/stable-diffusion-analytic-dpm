import torch
import utils_ as utils
import einops
import numpy as np
from diffusion.schedule import Schedule
from diffusion.dtdpm import DDPM, DDIM
from diffusion.sample import sample_dtdpm


class Wrapper(object):
    def __init__(self, model):
        super().__init__()
        self.typ = 'eps'
        self.model = model

    def __call__(self, xn, n):
        if np.isscalar(n):
            n = np.array([n] * xn.size(0))
        n = torch.tensor(n).to(xn)
        out = self.model(xn, n)
        return out


def txt2img(nnet, autoencoder, clip, prompt: str, H: int, W: int, n_samples: int,
            sample_steps: int, scale: float, seed: int, sampler: str) -> np.ndarray:
    r"""

    Args:
        nnet: the noise prediction network in a diffusion model
        autoencoder: the image autoencoder
        clip: the clip text encoder
        prompt: the input prompt
        H: the height of the image
        W: the width of the image
        n_samples: the number of samples to generate
        sample_steps: the number denoising iterations during sampling
        scale: the classifier free guidance scale
        seed: the random seed
        sampler: ddim / ddpm / addim / addpm
    Returns:
        images of shape (B H W C) with range {0, ..., 255}
    """
    utils.set_seed(seed)

    prompts = [prompt] * n_samples
    contexts = clip.encode(prompts)
    empty_context = clip.encode([''])
    empty_context = einops.repeat(empty_context, '1 L D -> B L D', B=n_samples)

    _betas = utils.stable_diffusion_beta_schedule()
    betas = np.append(0., _betas)
    schedule = Schedule(betas)

    def model_fn(z, timesteps):
        cond = nnet(z, timesteps, context=contexts)
        uncond = nnet(z, timesteps, context=empty_context)
        return uncond + scale * (cond - uncond)

    wrapper = Wrapper(model_fn)
    z_init = torch.randn(n_samples, 4, H // 8, W // 8, device=contexts.device)

    if sampler == 'ddim':
        dtdpm = DDIM(wrapper, schedule, clip_x0=False, eta=0, clip_cov_x0=True, avg_cov=True)
        z_out = sample_dtdpm(dtdpm, z_init, rev_var_type='small', sample_steps=sample_steps, clip_sigma_idx=1)
    elif sampler == 'ddpm':
        dtdpm = DDPM(wrapper, schedule, clip_x0=False, clip_cov_x0=True, avg_cov=True)
        z_out = sample_dtdpm(dtdpm, z_init, rev_var_type='small', sample_steps=sample_steps, clip_sigma_idx=1)
    elif sampler == 'addim':
        dtdpm = DDIM(wrapper, schedule, clip_x0=False, eta=0, clip_cov_x0=True, avg_cov=True)
        z_out = sample_dtdpm(dtdpm, z_init, rev_var_type='optimal', sample_steps=sample_steps, clip_sigma_idx=1)
    elif sampler == 'addpm':
        dtdpm = DDPM(wrapper, schedule, clip_x0=False, clip_cov_x0=True, avg_cov=True)
        z_out = sample_dtdpm(dtdpm, z_init, rev_var_type='optimal', sample_steps=sample_steps, clip_sigma_idx=1)
    else:
        raise NotImplementedError
    return utils.unpreprocess(autoencoder.decode(z_out))


def main():
    import argparse
    import libs.autoencoder
    import libs.clip
    from PIL import Image
    import os


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photograph of an astronaut riding a horse",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        help="dir to write results to",
        default="outputs"
    )

    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="number of sampling steps",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--nnet",
        type=str,
        default="stable_diffusion_v1",
        help="specify the noise prediction model"
    )

    parser.add_argument(
        "--nnet_path",
        type=str,
        default="models/sd_nnet_ema.pth",
        help="the path to the noise prediction network"
    )

    parser.add_argument(
        "--autoencoder_path",
        type=str,
        default="models/autoencoder_kl.pth",
        help="the path to the image autoencoder"
    )

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prepare the clip text encoder
    clip = libs.clip.FrozenCLIPEmbedder()

    # prepare the diffusion backbone
    nnet = utils.get_nnet(args.nnet)
    nnet.load_state_dict(torch.load(args.nnet_path, map_location='cpu'))

    # prepare the autoencoder
    autoencoder = libs.autoencoder.get_model(args.autoencoder_path)

    for m in [clip, nnet, autoencoder]:
        m.eval()
        m.to(device)
        m.requires_grad_(False)

    if args.nnet == 'stable_diffusion_v1':
        # t=0 in stable diffusion code corresponds to t=1 in paper
        # we use the paper time instead of the code time
        _nnet = nnet
        nnet = lambda x, timesteps, context: _nnet(x, timesteps - 1, context)

    os.makedirs(args.outdir, exist_ok=True)

    for sampler in ['ddim', 'ddpm', 'addim', 'addpm']:
        for sample_steps in [10, 15, 20, 25, 30, 40, 50]:
            with torch.autocast(device_type=device, enabled=args.precision == 'autocast'):
                images = txt2img(nnet=nnet, autoencoder=autoencoder, clip=clip, prompt=args.prompt, H=args.H, W=args.W,
                                 n_samples=args.n_samples, sample_steps=sample_steps, scale=args.scale, seed=args.seed, sampler=sampler)

            for idx, image in enumerate(images):
                os.makedirs(os.path.join(args.outdir, f'{idx}'), exist_ok=True)
                Image.fromarray(image).save(os.path.join(args.outdir, f'{idx}', f'{sampler}{sample_steps}.png'))


if __name__ == "__main__":
    main()
