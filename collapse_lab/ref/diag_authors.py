"""Diagnostic de R0 : que voit le guide du code des auteurs a chaque pas de reechantillonnage ?

Meme pipeline et memes reglages que run_authors.py (configuration du papier), un prompt.
On enveloppe FKD.resample pour enregistrer, a chaque pas planifie : les rewards brutes que
leur ImageReward rend sur les x0 decodes, la statistique max planchee, les poids, l'ESS et
les indices tires. Et on rescore les memes images decodees avec l'ImageReward officiel
(`ImageReward.load`, celui de smc/rewards_sd.py) pour savoir si les deux scorers coincident.
Sortie : collapse_lab/out/diag_authors_<i>.json et le tableau imprime.

  /home/onyxia/work/.venvs/sd/bin/python collapse_lab/ref/diag_authors.py --i 0
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FKD_DIR = Path("/home/onyxia/work/fkd_ref/Fk-Diffusion-Steering/text_to_image")
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--i", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ir-cache", default="/home/onyxia/work/ir_cache")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "shim"))
    sys.path.insert(0, str(FKD_DIR))
    sys.path.insert(0, str(FKD_DIR / "fkd_diffusers"))
    from diffusers import DDIMScheduler
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
    import fkd_diffusers.rewards as rewards
    import rewards as rewards_plat
    from fkd_diffusers.image_reward_utils import rm_load
    import fkd_class
    import ImageReward as RM

    prompts = json.loads((FKD_DIR / "prompt_files" / "benchmark_ir.json").read_text())
    pid, prompt = prompts[args.i]["id"], prompts[args.i]["prompt"]
    print(f"prompt {pid} : {prompt}")

    pipe = FKDStableDiffusion.from_pretrained(REPO, torch_dtype=torch.float16, variant="fp16",
                                              safety_checker=None).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    leur_ir = rm_load("ImageReward-v1.0", device="cuda", download_root=args.ir_cache)
    rewards.REWARDS_DICT["ImageReward"] = rewards_plat.REWARDS_DICT["ImageReward"] = leur_ir
    officiel = RM.load("ImageReward-v1.0", device="cuda", download_root=args.ir_cache)

    journal = []
    original = fkd_class.FKD.resample

    def espion(self, *, sampling_idx, latents, x0_preds):
        interval = np.append(np.arange(self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency), self.time_steps - 1)
        if sampling_idx not in interval:
            return original(self, sampling_idx=sampling_idx, latents=latents, x0_preds=x0_preds)
        imgs = self.latent_to_decode_fn(x0_preds)
        brut = self.reward_fn(imgs)                                   # leur scorer, leurs images
        pil = pipe.image_processor.postprocess(imgs, output_type="pil")
        off = officiel.score(prompt, pil)                             # le scorer officiel, memes images
        off = off if isinstance(off, list) else [off]
        avant = self.population_rs.clone()
        stat = torch.max(brut, self.population_rs)
        w = torch.exp(self.lmbda * stat)
        if sampling_idx == self.time_steps - 1:
            w = torch.exp(self.lmbda * brut) / self.product_of_potentials
        wn = w / w.sum()
        ess = float(1 / (wn ** 2).sum())
        # on laisse leur code faire le tirage, puis on lit ce qu'il a garde
        lat_out, _ = original(self, sampling_idx=sampling_idx, latents=latents, x0_preds=x0_preds)
        garde = [int(torch.nonzero((latents == lat_out[j]).all(dim=(1, 2, 3)))[0]) for j in range(lat_out.shape[0])]
        row = {"idx": int(sampling_idx), "r_leur": [round(float(v), 3) for v in brut], "r_officiel": [round(float(v), 3) for v in off],
               "population_rs_avant": [round(float(v), 3) for v in avant], "stat_max_plancher": [round(float(v), 3) for v in stat],
               "w_norm": [round(float(v), 3) for v in wn], "ess": round(ess, 3), "indices": garde,
               "img_min_max": [round(float(imgs.min()), 2), round(float(imgs.max()), 2)]}
        journal.append(row)
        print(f"  idx {sampling_idx:3d} | r leur {row['r_leur']} | r officiel {row['r_officiel']} | max plancher {row['stat_max_plancher']} "
              f"| w {row['w_norm']} | ESS {ess:.2f} | garde {garde} | images dans [{row['img_min_max'][0]}, {row['img_min_max'][1]}]")
        return lat_out, None

    fkd_class.FKD.resample = espion
    effective = args.seed * 1000 + args.i
    torch.manual_seed(effective); torch.cuda.manual_seed_all(effective)
    fkd_args = dict(lmbda=10.0, num_particles=4, use_smc=True, adaptive_resampling=False, time_steps=100,
                    guidance_reward_fn="ImageReward", potential_type="max", resample_frequency=20,
                    resampling_t_start=20, resampling_t_end=80)
    images = pipe([prompt] * 4, num_inference_steps=100, eta=1.0, fkd_args=fkd_args)[0]
    fin_leur = leur_ir.score_batched([prompt] * 4, images)
    fin_off = officiel.score(prompt, images)
    print(f"  finales : leur scorer {[round(v, 3) for v in fin_leur]} | officiel {[round(float(v), 3) for v in fin_off]}")
    out = ROOT / "collapse_lab" / "out" / f"diag_authors_{args.i}.json"
    out.write_text(json.dumps({"prompt_id": pid, "prompt": prompt, "seed_effective": effective, "pas": journal,
                               "finales_leur": fin_leur, "finales_officiel": [float(v) for v in fin_off]}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
