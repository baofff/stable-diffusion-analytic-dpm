import random
import einops
import numpy as np
import torch
import argparse
from PIL import Image


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def preprocess(x: np.array) -> torch.Tensor:
    r"""

    Args:
        x: images of shape (B H W C) with range {0, ..., 255}

    Returns:
        images of shape (B C H W) with range [-1, 1]
    """

    x = einops.rearrange(x, 'B H W C -> B C H W')
    x = x.astype(np.float32) / 255.0
    x = 2. * x - 1.
    x = torch.tensor(x)
    return x


def unpreprocess(x: torch.Tensor) -> np.ndarray:
    r"""

    Args:
        x: images of shape (B C H W) with range [-1, 1]

    Returns:
        images of shape (B H W C) with range {0, ..., 255}
    """

    x = 0.5 * (x + 1)  # [-1, 1] to [0, 1]
    x = x.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()  # [0, 1] to {0, ..., 255}
    x = einops.rearrange(x, 'B C H W -> B H W C')
    return x


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


def sos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).sum(dim=-1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_img(path):
    img = Image.open(path).convert("RGB")
    H, W = img.size
    print(f"loaded input image of size ({H}, {W}) from {path}")
    H, W = map(lambda x: x - x % 32, (H, W))  # resize to integer multiple of 32
    img = img.resize((H, W), resample=Image.Resampling.LANCZOS)
    img = np.array(img)
    return img


def get_nnet(name='stable_diffusion_v1'):
    if name == 'stable_diffusion_v1':
        from libs.ldm.openaimodel import UNetModel
        return UNetModel(
            image_size=32,  # unused
            in_channels=4,
            out_channels=4,
            model_channels=320,
            attention_resolutions=[ 4, 2, 1 ],  # down sampling
            num_res_blocks=2,
            channel_mult=[ 1, 2, 4, 4 ],
            num_heads=8,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=768,
            use_checkpoint=False,
            legacy=False
        )
    else:
        raise NotImplementedError(name)
