import argparse
import os
import pathlib
import random
from contextlib import nullcontext

import numpy as np
import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch import autocast
from torch.multiprocessing import cpu_count, spawn
from torchvision.io import write_png
from tqdm import trange


def _generate_caption(category: str) -> str:
    names = category.split(",")
    name = np.random.choice(names, 1)[0]
    _0 = [
        "",
        "high quality",
        "low quality",
        "monochrome",
        "blured",
        "atmospheric",
        "rendered",
        "zoomed",
        "wide-angle",
        "hdr",
        "high resolution",
    ]
    _1 = ["photo", "picture", "realistic photo", "realistic drawing"]
    _2 = ["", "taken with iPhone", "inside", "outside", "without background"]

    _0 = np.random.choice(_0, None)
    _1 = np.random.choice(_1, None)
    _2 = np.random.choice(_2, None)
    return f"{_0} {_1} of {name} {_2}".strip()


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
def _main(opt, model, sampler, rank):
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    all_prompts = []

    batch_size = opt.n_samples

    sample_path = outpath
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    start_code = None

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    outer_loop = trange(opt.n_iter, desc="Sampling") if rank == 0 else range(opt.n_iter)
    with model.ema_scope():
        for _ in outer_loop:
            prompts = [_generate_caption(opt.prompt) for _ in range(batch_size)]
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])

            c = model.get_learned_conditioning(prompts)
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            samples_ddim, _ = sampler.sample(
                S=opt.ddim_steps,
                conditioning=c,
                batch_size=opt.n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                unconditional_conditioning=uc,
                eta=opt.ddim_eta,
                x_T=start_code,
            )

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            for x_sample in x_samples_ddim:
                write_png((255 * x_sample.cpu()).to(torch.uint8), os.path.join(sample_path, f"{base_count:05}.png"))
                base_count += 1
            all_prompts.extend(prompts)
    with open(f"{sample_path}/prompts.txt", "w") as f:
        f.writelines([f"{prompt}\n" for prompt in all_prompts])


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
        default=2,
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

    torch.manual_seed(opt.seed + rank)
    random.seed(opt.seed + rank)
    np.random.seed(opt.seed + rank)
    torch.cuda.set_device(rank)

    torch.set_num_threads(2 * cpu_count() // torch.cuda.device_count())

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    with open("imagenet.categories", "r") as f:
        cat_id_list = f.read().splitlines()

    with open("imagenet_category.txt", "r") as f:
        category_names = f.read().splitlines()

    size = 1000 // torch.cuda.device_count()
    ids = list(range(1000))[rank * size : (rank + 1) * size]
    outdir = opt.outdir
    for i in ids:
        id = cat_id_list[i].split(",")[1]
        opt.prompt = category_names[i]
        opt.outdir = f"{outdir}/{id}"
        _main(opt, model, sampler, rank)


if __name__ == "__main__":
    spawn(main, nprocs=torch.cuda.device_count())
