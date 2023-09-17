import argparse
import os
import pathlib
import random

import numpy as np
import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.multiprocessing import cpu_count, spawn
from torchvision.io import write_png
from tqdm import trange


def load_model_from_config(config, ckpt, verbose=False):
    ckpt = pathlib.Path(ckpt).absolute()
    print(f"Loading model from {ckpt}")
    with ckpt.open("rb") as f:
        pl_sd = torch.load(f, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


@torch.inference_mode()
def _main(opt, data, model, sampler, rank):
    batch_size = len(data)
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    sample_path = outpath
    os.makedirs(sample_path, exist_ok=True)

    start_code = None

    outer_loop = trange(opt.n_iter, desc="Sampling") if rank == 0 else range(opt.n_iter)
    with model.ema_scope():
        for _ in outer_loop:
            idx = [i for i, d in data]
            prompts = [d for i, d in data]
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])

            c = model.get_learned_conditioning(prompts)
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            samples_ddim, _ = sampler.sample(
                S=opt.ddim_steps,
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                unconditional_conditioning=uc,
                eta=opt.ddim_eta,
                x_T=start_code,
            )

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            for id, x_sample in zip(idx, x_samples_ddim):
                write_png((255 * x_sample.cpu()).to(torch.uint8), os.path.join(sample_path, f"{id:05}.png"))


def main(rank):
    print(f"setup {rank=}")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render",
    )
    parser.add_argument(
        "--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
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
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
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
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast"
    )
    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.set_device(rank)

    torch.set_num_threads(2 * cpu_count() // torch.cuda.device_count())

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    with open("coco_caption_train.txt", "r") as f:
        captions = f.read().splitlines()

    size = len(captions) // torch.cuda.device_count()
    captions = list(enumerate(captions))
    captions = captions[rank * size : (rank + 1) * size]
    for idx in range(len(captions) // opt.n_samples):
        data = captions[idx * opt.n_samples : (idx + 1) * opt.n_samples]
        _main(opt, data, model, sampler, rank)


if __name__ == "__main__":
    spawn(main, nprocs=torch.cuda.device_count())
