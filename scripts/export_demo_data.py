"""Write docs/demo/data.js, the one data file the interactive demo (docs/demo/index.html) loads.

Reads collapse_lab/out/probe_C.json (session C: ctl, lam0, floor2 on 100 prompts, with the
ancestor matrix and the guide rewards at each scheduled step), results/sd_baseline.json
(k1, bon4, fk4 on 100 prompts x 3 seeds) and data/visual_selection.json (the W1 and W2
prompt lists). Copies to docs/demo/img/<arm>/<prompt_id>_<slot>.webp, at 256 px, any final
image of a W1/W2 prompt listed in collapse_lab/out/images/index.json.

Two checks are printed on the way: the per-step weights the demo recomputes from
r_at_schedule (running max, increment form, terminal step on r itself, smc/fk.py) match the
recorded logG_at_schedule, and the roots propagated along ancestors_at_schedule match
root_slots. Nothing under collapse_lab/ or results/ is modified.
"""
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "collapse_lab" / "out" / "probe_C.json"
BASELINE = ROOT / "results" / "sd_baseline.json"
SELECTION = ROOT / "data" / "visual_selection.json"
OUT = ROOT / "docs" / "demo"
ARMS = ["ctl", "floor2", "lam0"]
SAMPLERS = ["k1", "bon4", "fk4"]
THUMB = 256


def logG_from_rewards(r, anc, lam, floor):
    """The increments smc/fk.py writes for the `max` potential in increment form: at a
    non-terminal step lam * (max(r, M_prev) - M_prev) with M_prev the running max of the
    parent (the gate starts at -inf, so the first step is lam * r); at the terminal step
    lam * (r - M_prev). Floor: r <- max(r, 0) first."""
    r = np.array(r, float)
    if floor:
        r = np.maximum(r, 0.0)
    anc = np.array(anc, int)
    k = r.shape[1]
    out = []
    M = None
    for m in range(len(r)):
        prev = np.zeros(k) if m == 0 else M[anc[m - 1]]
        last = m == len(r) - 1
        curr = r[m] if (last or m == 0) else np.maximum(r[m], prev)
        out.append(lam * (curr - prev))
        M = curr
    return np.array(out)


def roots_from_ancestors(anc):
    roots = list(range(len(anc[0])))
    for row in anc[:-1]:          # the terminal row is the identity, no resampling at t = 0
        roots = [roots[i] for i in row]
    return roots


def round_list(x, nd=4):
    return [round_list(v, nd) if isinstance(v, list) else round(float(v), nd) for v in x]


def main():
    sel = json.loads(SELECTION.read_text())
    # the session the selection was ranked on, and its images next to it
    probe_path = ROOT / sel["source"] if "source" in sel else PROBE
    images_dir = probe_path.parent / "images"
    probe = json.loads(probe_path.read_text())["runs"]
    base = json.loads(BASELINE.read_text())["runs"]
    index = json.loads((images_dir / "index.json").read_text())
    by = {(r["arm"], r["prompt_id"]): r for r in probe}

    # --- checks --------------------------------------------------------------------------
    worst = {}
    bad_roots = 0
    for r in probe:
        lg = logG_from_rewards(r["r_at_schedule"], r["ancestors_at_schedule"], r["lam"], r["plancher"])
        worst[r["arm"]] = max(worst.get(r["arm"], 0.0), float(np.abs(lg - np.array(r["logG_at_schedule"])).max()))
        bad_roots += roots_from_ancestors(r["ancestors_at_schedule"]) != r["root_slots"]
        bad_roots += len(set(r["root_slots"])) != r["n_lineages"]
    print("check: max |recomputed logG - recorded| per arm:",
          {a: round(v, 4) for a, v in worst.items()}, "(r is stored at 4 dp, so ~1e-3 at lambda 10)")
    print(f"check: roots propagated along ancestors differ from root_slots, or n_lineages from the distinct roots, in {bad_roots} of {len(probe)} runs")
    if max(worst.values()) > 5e-3 or bad_roots:
        raise SystemExit("the reconstruction does not match the recorded run; fix before exporting")

    # --- images of W1/W2 prompts ------------------------------------------------------
    from PIL import Image
    wanted = set(sel["W1"]) | set(sel["W2"])
    images = {}
    for e in index:
        arm = e["arm"].split("_")[0]
        if e["prompt_id"] not in wanted or arm not in ARMS:
            continue
        run = by[(arm, e["prompt_id"])]
        if abs(run["ir"][e["slot"]] - e["ir"]) > 1e-3:
            print(f"skip {e['path']}: ir {e['ir']} is not the record's {run['ir'][e['slot']]}")
            continue
        dst = OUT / "img" / arm / f"{e['prompt_id']}_{e['slot']}.webp"
        dst.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(images_dir / Path(e["path"]).relative_to("images")).convert("RGB")
        im.resize((THUMB, THUMB), Image.LANCZOS).save(dst, quality=85, method=6)
        images.setdefault(e["prompt_id"], {}).setdefault(arm, []).append(e["slot"])

    # --- W1 / W2 prompts, per arm -----------------------------------------------------
    prompts = {}
    for pid in sorted(wanted):
        d = {"prompt": by[("ctl", pid)]["prompt"], "arms": {}, "images": images.get(pid, {})}
        for arm in ARMS:
            r = by[(arm, pid)]
            d["arms"][arm] = {
                "lam": r["lam"], "floor": r["plancher"],
                "ess": r["ess_at_schedule"],
                "anc": r["ancestors_at_schedule"],
                "r": r["r_at_schedule"],
                "ir": r["ir"], "ir_max": r["ir_max"],
                "root_slots": r["root_slots"], "n_lineages": r["n_lineages"],
                "n_resamplings": r["n_resamplings"],
            }
        prompts[pid] = d

    # --- W3: the recorded rewards of ctl, all 100 prompts ------------------------------
    ctl_ids = sorted(r["prompt_id"] for r in probe if r["arm"] == "ctl")
    w3 = {pid: {"prompt": by[("ctl", pid)]["prompt"], "r": by[("ctl", pid)]["r_at_schedule"],
                "anc": by[("ctl", pid)]["ancestors_at_schedule"], "n_lineages": by[("ctl", pid)]["n_lineages"]}
          for pid in ctl_ids}

    # --- W5: k1, bon4, fk4 x 100 prompts x 3 seeds -------------------------------------
    w5 = {"seeds": sorted({r["seed"] for r in base}), "prompts": sorted({r["prompt_id"] for r in base}), "runs": []}
    for r in base:
        if r["sampler"] not in SAMPLERS:
            continue
        w5["runs"].append({"s": r["sampler"], "pid": r["prompt_id"], "seed": r["seed"],
                           "ir_max": round(r["ir_max"], 4), "ir": round_list(r["ir"]), "hps": round_list(r["hps"])})

    ctl0 = by[("ctl", ctl_ids[0])]
    demo = {
        "schedule_t": sorted(ctl0["schedule_t"], reverse=True),   # index m of the arrays is t = 80, 60, 40, 20, 0
        "k": ctl0["k"], "threshold": ctl0["threshold"], "seed": ctl0["seed"],
        "arms": ARMS,
        "w1": sel["W1"], "w2": sel["W2"],
        "prompts": prompts, "w3": w3, "w5": w5,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    text = "window.DEMO = " + json.dumps(demo, separators=(",", ":")) + ";\n"
    (OUT / "data.js").write_text(text)

    print(f"wrote {OUT / 'data.js'}: {len(text) / 1024:.0f} KB")
    print(f"  W1 prompts: {len(sel['W1'])}, W2 prompts: {len(sel['W2'])}, union: {len(wanted)}, arms: {ARMS}")
    print(f"  W3: r and ancestors of ctl for {len(w3)} prompts")
    print(f"  W5: {len(w5['runs'])} runs, samplers {SAMPLERS}, seeds {w5['seeds']}, {len(w5['prompts'])} prompts")
    if images:
        for pid, arms in images.items():
            print(f"  images: {pid} -> " + ", ".join(f"{a} slots {sorted(s)}" for a, s in arms.items()))
    else:
        print("  images: none of the W1/W2 prompts has an image in collapse_lab/out/images yet")
    missing = [pid for pid in sel["W1"] if pid not in images]
    print(f"  W1 prompts without images: {len(missing)} of {len(sel['W1'])}")


if __name__ == "__main__":
    main()
