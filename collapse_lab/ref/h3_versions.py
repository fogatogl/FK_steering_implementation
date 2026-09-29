"""Session H, check H3: do the released code's pinned versions change the reward or the sampler?

Run once per venv (--tag current / pinned): the versions, the DDIM scheduler the pipeline builds, and
ImageReward rescored through their `rm_load` / `score_batched` on session D's saved `ctl` finals.
--compare reads both files and the two best-of-4 files of `run_authors.py --no-smc`, and writes the
verdict that `nuitH.sh` reads to pick the venv of H2.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "collapse_lab" / "out" / "session_H"
FKD = Path(os.environ.get("FKD_ROOT", "/home/onyxia/work/fkd_ref/prefix_6726324")) / "text_to_image"
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGES = Path("/home/onyxia/work/diffusion-models/collapse_lab/out/session_D/images/ctl")


def measure(tag, n, ir_cache):
    import torch, diffusers, transformers
    from importlib.metadata import version
    from PIL import Image
    from diffusers import DDIMScheduler, PNDMScheduler
    sys.path.insert(0, str(Path(__file__).parent / "shim"))
    sys.path.insert(0, str(FKD))
    sys.path.insert(0, str(FKD / "fkd_diffusers"))
    from fkd_diffusers.image_reward_utils import rm_load

    sched = DDIMScheduler.from_config(PNDMScheduler.from_pretrained(REPO, subfolder="scheduler").config)
    sched.set_timesteps(100)
    rm = rm_load("ImageReward-v1.0", device="cuda", download_root=ir_cache)
    runs = [r for r in json.loads((ROOT / "collapse_lab/out/session_D/probe_D.json").read_text())["runs"]
            if r["arm"] == "ctl"][:n]
    rescored = []
    with torch.no_grad():
        for r in runs:
            ims = [Image.open(IMAGES / f"{r['prompt_id']}_{j}.png").convert("RGB") for j in range(4)]
            s = rm.score_batched([r["prompt"]] * 4, ims)
            rescored.append({"prompt_id": r["prompt_id"], "recorded": r["ir"],
                             "rescored": [round(float(v), 4) for v in s]})
    res = {"tag": tag, "python": sys.version.split()[0],
           "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                        "transformers": transformers.__version__, "image-reward": version("image-reward")},
           "scheduler": {k: v for k, v in sched.config.items() if not k.startswith("_")},
           "timesteps": sched.timesteps.tolist(), "rescored": rescored}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"h3_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["versions"]), f"{len(rescored)} prompts rescored")


def compare(pinned_venv):
    a, b = (json.loads((OUT / f"h3_{t}.json").read_text()) for t in ("current", "pinned"))
    reward = max(abs(x - y) for ra, rb in zip(a["rescored"], b["rescored"], strict=True)
                 for x, y in zip(ra["rescored"], rb["rescored"], strict=True))
    sched = sorted(k for k in set(a["scheduler"]) | set(b["scheduler"]) if a["scheduler"].get(k) != b["scheduler"].get(k))
    ba, bb = ({r["prompt_id"]: r["ir"] for r in json.loads((OUT / f"h3_bon4_{t}.json").read_text())["runs"]}
              for t in ("current", "pinned"))
    gen = max(abs(x - y) for pid in ba for x, y in zip(ba[pid], bb[pid], strict=True))
    agree = reward <= 1e-3 and not sched and a["timesteps"] == b["timesteps"] and gen <= 1e-3
    verdict = {"reward_max_diff": round(reward, 5), "scheduler_keys_differ": sched,
               "timesteps_equal": a["timesteps"] == b["timesteps"], "bon4_max_diff": round(gen, 4),
               "n_bon4_prompts": len(ba), "agree": agree,
               "h2_venv": "/home/onyxia/work/.venvs/sd" if agree else pinned_venv}
    (OUT / "h3_verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", choices=["current", "pinned"])
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--ir-cache", default="/home/onyxia/work/ir_cache")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--pinned-venv", default="/home/onyxia/work/.venvs/fkd_pinned")
    args = p.parse_args()
    compare(args.pinned_venv) if args.compare else measure(args.tag, args.n, args.ir_cache)
