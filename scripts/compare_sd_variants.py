"""The screen of the FK variants, read paired against results/sd_baseline.json.

Each JSON in results/sd_variants/ shares (prompt_id, seed) with the main
file, hence the same x_T: we compare variant minus the reference fk4, and
variant minus bon4, prompt by prompt. Standard error over the prompts, number of
prompts won. For K8 the best-of-N reference is bon8 from the same file.

From the repository root: `python scripts/compare_sd_variants.py`.
"""
import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def par_cle(runs, sampler):
    return {(r["prompt_id"], r["seed"]): r for r in runs if r["sampler"] == sampler}


def apparie(a, b, key):
    """Differences a - b over the common (prompt, seed) pairs: (mean, standard error, won, n)."""
    communs = sorted(set(a) & set(b))
    d = [a[k][key] - b[k][key] for k in communs]
    if len(d) < 2:
        return float("nan"), float("nan"), 0, len(d)
    return st.mean(d), st.stdev(d) / len(d) ** .5, sum(1 for v in d if v > 0), len(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=str(ROOT / "results" / "sd_baseline.json"))
    p.add_argument("--variants", default=str(ROOT / "results" / "sd_variants"))
    p.add_argument("--baseline-div", default=str(ROOT / "results" / "sd_baseline_div20.json"),
                   help="reference bon4 and fk4 rerun with n_lineages / div_pix")
    args = p.parse_args()

    base = json.loads(Path(args.baseline).read_text())["runs"]
    ref_fk, ref_bon = par_cle(base, "fk4"), par_cle(base, "bon4")

    print(f"{'variant':8s} {'n':>3s}  {'IR vs fk4':>18s} {'won':>7s}  {'IR vs best-of-N':>18s} {'won':>7s}  "
          f"{'HPS vs fk4':>18s}  {'ESS 1st':>9s} {'nres':>5s} {'lin.':>5s} {'div_pix':>7s} {'s/run':>6s}")
    for f in sorted(Path(args.variants).glob("*.json")):
        runs = json.loads(f.read_text())["runs"]
        fk_name = "fk8" if any(r["sampler"] == "fk8" for r in runs) else "fk4"
        var = par_cle(runs, fk_name)
        if not var:
            continue
        bon = par_cle(runs, "bon8") if fk_name == "fk8" else ref_bon
        m1, s1, w1, n = apparie(var, ref_fk, "ir_max")
        m2, s2, w2, _ = apparie(var, bon, "ir_max")
        m3, s3, w3, _ = apparie(var, ref_fk, "hps_at_ir_max")
        # index 0 of ess_at_schedule = the smallest loop index = the largest t.
        # S60, S40, S80 and D10 do not start at t = 80: the actual t is shown.
        ess1 = st.median(r["ess_at_schedule"][0] for r in var.values())
        t1 = max(next(iter(var.values()))["schedule_t"])
        nres = st.mean(r["n_resamplings"] for r in var.values())
        sec = st.mean(r["seconds"] for r in var.values())
        # diversity of the k final images: distinct x_T roots and mean pixel RMSE over pairs;
        # absent from the JSON files written before the evening of 20/09
        lign = [r["n_lineages"] for r in var.values() if "n_lineages" in r]
        div = [r["div_pix"] for r in var.values() if r.get("div_pix") is not None]
        lign_s = f"{st.mean(lign):5.2f}" if lign else "    -"
        div_s = f"{st.mean(div):7.4f}" if div else "      -"
        print(f"{f.stem:8s} {n:3d}  {m1:+.4f} ± {s1:.4f}  {w1:3d}/{n:<3d}  {m2:+.4f} ± {s2:.4f}  {w2:3d}/{n:<3d}  "
              f"{m3:+.4f} ± {s3:.4f}  {ess1:5.2f}@{t1:<3d} {nres:5.2f} {lign_s} {div_s} {sec:6.1f}")

    # the reference itself, on the same prompts, to read the screen at the right scale
    communs = {k for f in Path(args.variants).glob("*.json")
               for k in par_cle(json.loads(f.read_text())["runs"], "fk4")}
    if communs:
        sub_fk = {k: v for k, v in ref_fk.items() if k in communs}
        sub_bon = {k: v for k, v in ref_bon.items() if k in communs}
        m, s, w, n = apparie(sub_fk, sub_bon, "ir_max")
        print(f"\nreference fk4 - bon4 on these {n} prompts: {m:+.4f} ± {s:.4f}, {w}/{n} won "
              f"(on all 100: +0.0619 ± 0.0259)")

    # the reference's diversity comes from a separate regeneration (same seeds, hence same x_T)
    if Path(args.baseline_div).exists():
        div_runs = json.loads(Path(args.baseline_div).read_text())["runs"]
        for name in ("bon4", "fk4"):
            rs = [r for r in div_runs if r["sampler"] == name and r.get("div_pix") is not None]
            if rs:
                lign = [r["n_lineages"] for r in rs if "n_lineages" in r]
                lign_s = f"{st.mean(lign):.2f} lineages, " if lign else ""
                print(f"reference {name} ({len(rs)} prompts): {lign_s}div_pix {st.mean(r['div_pix'] for r in rs):.4f}")


if __name__ == "__main__":
    main()
