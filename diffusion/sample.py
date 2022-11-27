import torch
import logging
import math
from .dtdpm import DTDPM
from .trajectory import _choice_steps


def statistics2str(statistics):
    return str({k: f'{v:.6g}' for k, v in statistics.items()})


def report_statistics(s, t, statistics):
    print(f'[(s, r): ({s:.6g}, {t:.6g})] [{statistics2str(statistics)}]')


@ torch.no_grad()
def sample_dtdpm(dtdpm, x_init, rev_var_type, trajectory='linear', sample_steps=None, clip_sigma_idx=0, clip_pixel=2, ms_eps=None):
    r"""
    Sample from the reverse model p(x0|x1)...p(xN-1|xN)p(xN) proposed in DDPM, DDIM and Analytic-DPM
    """
    assert isinstance(dtdpm, DTDPM)
    N = dtdpm.N
    sample_steps = sample_steps or N
    ns = _choice_steps(N, sample_steps, trajectory, ms_eps=ms_eps, betas=dtdpm.betas)
    timesteps = [0] + ns
    logging.info("sample_dtdpm with rev_var_type={}, trajectory={}, sample_steps={}, clip_sigma_idx={}, clip_pixel={}"
                 .format(rev_var_type, trajectory, sample_steps, clip_sigma_idx, clip_pixel))
    return _sample_dtdpm(dtdpm, x_init, rev_var_type, timesteps, clip_sigma_idx, clip_pixel, ms_eps)


@ torch.no_grad()
def _sample_dtdpm(dtdpm, x_init, rev_var_type, timesteps, clip_sigma_idx=0, clip_pixel=2, ms_eps=None):
    assert isinstance(dtdpm, DTDPM)
    assert timesteps[0] == 0
    x = x_init
    for s, t in list(zip(timesteps, timesteps[1:]))[::-1]:
        dtdpm.statistics = {}
        x_mean, sigma2 = dtdpm.predict_xprev_cov_xprev(x, s, t, rev_var_type, ms_eps)
        if s != 0:
            if s <= timesteps[clip_sigma_idx]:  # clip_sigma_idx = 0 <=> not clip
                dtdpm.statistics['sigma2_unclip'] = sigma2.mean().item()
                sigma2_threshold = (clip_pixel * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
                sigma2 = sigma2.clamp(0., sigma2_threshold)
                dtdpm.statistics['sigma2_threshold'] = sigma2_threshold
            x = x_mean + sigma2 ** 0.5 * torch.randn_like(x)
            dtdpm.statistics['sigma2'] = sigma2.mean().item()
        else:
            x = x_mean
        report_statistics(s, t, dtdpm.statistics)
    return x
