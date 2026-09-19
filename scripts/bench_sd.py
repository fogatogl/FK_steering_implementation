"""Chronomètre SD v1.5 nu (bloc C1) : secondes par image et VRAM, avant tout FK.

Suppose HF_HOME=/home/onyxia/work/hf_cache (les poids y sont déjà).
"""
import argparse, time, torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="a photo of an astronaut riding a horse on mars")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slicing", action="store_true")
    p.add_argument("--out", default="samples/sd_bench")
    args = p.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        REPO, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    if args.slicing:
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()

    import os
    os.makedirs(args.out, exist_ok=True)
    print(f"VRAM après chargement : {torch.cuda.memory_allocated()/2**30:.2f} Gio")

    for n in args.batch:
        torch.cuda.reset_peak_memory_stats()
        g = torch.Generator("cuda").manual_seed(args.seed)
        torch.cuda.synchronize()
        t0 = time.time()
        imgs = pipe([args.prompt] * n, num_inference_steps=args.steps,
                    guidance_scale=args.guidance, eta=args.eta, generator=g).images
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"batch={n:2d}  {dt:6.1f} s  ({dt/n:5.1f} s/image)  pic VRAM {peak:.2f} Gio")
        for i, im in enumerate(imgs):
            im.save(f"{args.out}/b{n}_s{args.seed}_{i}.png")


if __name__ == "__main__":
    main()
