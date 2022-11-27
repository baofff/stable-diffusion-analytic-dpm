## Generate samples on Stable Diffusion with Analytic-DPM

<img src="outputs/ddpm.png" alt="drawing" width="1600"/>

<img src="outputs/ddim.png" alt="drawing" width="1600"/>

I briefly try Analytic-DPM on Stable Diffusion. It looks that Analytic-DPM has more details in the image background than DDPM/DDIM.

## Run
```
python convert_ckpt.py  # split the Stable Diffusion checkpoint sd-v1-4-full-ema.ckpt
python txt2img.py  # see generated samples in "outputs" directory
```


