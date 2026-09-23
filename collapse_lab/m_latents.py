"""M. Le test des latents du constat 11 : la case j de bon4 est-elle la continuation de la racine j ?

Un prompt, le seed_effective de sa case dans sd_baseline.json. Deux trajectoires :
  - le pipeline diffusers tel que sample_independent l'appelle (bon4), latents captures
    par callback_on_step_end ;
  - smc.models.StableDiffusion : initial_state puis step, comme fk_steer les enchaine.
On compare les latents aux memes pas. Si l'ecart est nul au pas 0 et reste petit : les
racines sont partagees et bon4[j] continue bien la racine j (constat 11 faux). Si l'ecart
est de l'ordre de l'arrondi fp16 au pas 0 et croit jusqu'a O(1) : x_T est partage mais la
trajectoire diverge numeriquement, et la case j de bon4 n'est pas la continuation de la
racine j (seconde lecture du constat 11, precisee : c'est le chaos numerique, pas x_T).

  /home/onyxia/work/.venvs/sd/bin/python collapse_lab/m_latents.py
"""
import argparse, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PAS = (0, 1, 10, 50, 99)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default=str(ROOT / "data" / "imagereward-benchmark-prompts.json"))
    p.add_argument("--i", type=int, default=0, help="indice du prompt dans le fichier")
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()

    data = json.loads(Path(args.prompts).read_text())
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    pid, prompt = str(data[args.i].get("id", args.i)), data[args.i]["prompt"]
    eff = args.seed * 1000 + args.i
    ref = [r for r in json.loads((ROOT / "results" / "sd_baseline.json").read_text())["runs"]
           if r["prompt_id"] == pid and r["sampler"] == "bon4" and r["seed"] == args.seed]
    assert ref and ref[0]["seed_effective"] == eff, "appariement rompu"
    print(f"prompt {pid} ({prompt[:60]}...), seed_effective {eff}")

    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from smc.models import StableDiffusion

    pipe = StableDiffusionPipeline.from_pretrained(
        REPO, torch_dtype=torch.float16, variant="fp16", safety_checker=None).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # --- 1. le pipeline, comme sample_independent ---------------------------------
    pipeline = {}

    def cb(pipe_, i, t, kw):
        if i in PAS:
            pipeline[i] = kw["latents"].detach().clone()
        return kw

    g = torch.Generator("cuda").manual_seed(eff)
    pipe([prompt] * 4, num_inference_steps=100, guidance_scale=7.5, eta=1.0, height=512,
         width=512, generator=g, output_type="latent", callback_on_step_end=cb,
         callback_on_step_end_tensor_inputs=["latents"])

    # --- 2. smc.models.StableDiffusion, comme fk_steer ----------------------------
    g = torch.Generator("cuda").manual_seed(eff)
    model = StableDiffusion(pipe, guidance_scale=7.5, num_steps=100, eta=1.0)
    model.set_prompt(prompt)
    state = model.initial_state(4, g)
    x_T = state["x"].clone()
    ours = {}
    for i, t in enumerate(model.timesteps):
        state = model.step(state, t, g)
        if i in PAS:
            ours[i] = state["x"].clone()

    # --- 3. les embeddings de texte : batch 4 contre 1 etendu -------------------------
    cond4, _ = pipe.encode_prompt([prompt] * 4, "cuda", num_images_per_prompt=1,
                                  do_classifier_free_guidance=True)
    cond1 = model.text_emb[1]
    d_emb = (cond4.float() - cond1.float().expand_as(cond4)).abs().max().item()
    print(f"\nembeddings de texte, batch 4 contre 1 etendu : ecart max {d_emb:.2e}")

    # --- 4. x_T partage ? on rejoue le tirage du pipeline -----------------------------
    from diffusers.utils.torch_utils import randn_tensor
    g = torch.Generator("cuda").manual_seed(eff)
    xT_pipe = randn_tensor((4, 4, 64, 64), generator=g, device=torch.device("cuda"), dtype=torch.float16)
    xT_pipe = xT_pipe * pipe.scheduler.init_noise_sigma
    print(f"x_T : ecart max entre le tirage du pipeline et initial_state : {(xT_pipe - x_T).abs().max().item():.2e}"
          f"  (std de x_T {x_T.float().std().item():.3f})")

    # --- 5. la trajectoire, pas a pas ----------------------------------------------------
    print(f"\n{'pas i':>6s} {'t':>5s} | {'max|diff|':>10s} {'mean|diff|':>10s} {'std(x)':>8s} | {'corr par case':>40s}")
    lignes = []
    for i in PAS:
        a, b = pipeline[i].float(), ours[i].float()
        t = int(model.timesteps[i])
        corr = [torch.corrcoef(torch.stack([a[j].flatten(), b[j].flatten()]))[0, 1].item() for j in range(4)]
        lignes.append({"i": i, "t": t, "max_diff": (a - b).abs().max().item(),
                       "mean_diff": (a - b).abs().mean().item(), "std": b.std().item(), "corr": corr})
        print(f"{i:6d} {t:5d} | {(a - b).abs().max().item():10.3e} {(a - b).abs().mean().item():10.3e} "
              f"{b.std().item():8.3f} | " + " ".join(f"{c:+.4f}" for c in corr))
    # --- 6. la meme reward sur les deux finales, et la valeur enregistree dans bon4 --------------
    # Une correlation de 0.98 entre latents finaux, est-ce la meme image pour ImageReward ?
    import ImageReward as RM
    ir = RM.load("ImageReward-v1.0", device="cuda", download_root="/home/onyxia/work/ir_cache")
    with torch.no_grad():
        dec = lambda z: pipe.image_processor.postprocess(
            pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample, output_type="pil")
        im_pipe, im_ours = dec(pipeline[99]), dec(ours[99])
    ir_pipe = [float(v) for v in ir.score(prompt, im_pipe)]
    ir_ours = [float(v) for v in ir.score(prompt, im_ours)]
    ir_bon4 = ref[0]["ir"]
    print(f"\nImageReward des finales : pipeline (aujourd'hui) {[round(v, 3) for v in ir_pipe]}")
    print(f"                          notre modele (lam = 0)  {[round(v, 3) for v in ir_ours]}")
    print(f"                          bon4 enregistre (20/09)  {[round(v, 3) for v in ir_bon4]}")
    print(f"  |pipeline - bon4| max {max(abs(a - b) for a, b in zip(ir_pipe, ir_bon4)):.3f} : le fichier est-il rejouable ;"
          f"  |notre modele - pipeline| max {max(abs(a - b) for a, b in zip(ir_ours, ir_pipe)):.3f} : la sensibilite d'IR a l'ecart fp16")
    out = ROOT / "collapse_lab" / "out" / f"latents_{args.i}.json"
    out.write_text(json.dumps({"prompt_id": pid, "seed_effective": eff, "d_emb": d_emb,
                               "d_xT": (xT_pipe - x_T).abs().max().item(), "pas": lignes,
                               "ir_pipeline": ir_pipe, "ir_ours": ir_ours, "ir_bon4": ir_bon4}, indent=1))
    print(f"\n-> {out}")
    print("\nlecture : corr ~ 1 a tous les pas = meme trajectoire, la case j de bon4 continue la racine j ;"
          "\n          corr qui decroche = les deux samplers partagent x_T et le bruit, et divergent quand meme.")


if __name__ == "__main__":
    main()
