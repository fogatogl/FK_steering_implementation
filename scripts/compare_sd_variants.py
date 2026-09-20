"""L'écran des variantes FK, lu en apparié contre results/sd_baseline.json.

Chaque JSON de results/sd_variants/ partage (prompt_id, seed) avec le fichier
principal, donc les mêmes x_T : on compare variante moins fk4 de référence, et
variante moins bon4, prompt par prompt. Erreur-type sur les prompts, nombre de
prompts gagnés. Pour K8 la référence best-of-N est bon8 du même fichier.

Depuis la racine : `python scripts/compare_sd_variants.py`.
"""
import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def par_cle(runs, sampler):
    return {(r["prompt_id"], r["seed"]): r for r in runs if r["sampler"] == sampler}


def apparie(a, b, key):
    """Différences a - b sur les (prompt, seed) communs : (moyenne, erreur-type, gagnés, n)."""
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
                   help="bon4 et fk4 de référence refaits avec n_lineages / div_pix")
    args = p.parse_args()

    base = json.loads(Path(args.baseline).read_text())["runs"]
    ref_fk, ref_bon = par_cle(base, "fk4"), par_cle(base, "bon4")

    print(f"{'variante':8s} {'n':>3s}  {'IR vs fk4':>18s} {'gagnés':>7s}  {'IR vs best-of-N':>18s} {'gagnés':>7s}  "
          f"{'HPS vs fk4':>18s}  {'ESS t=80':>8s} {'nres':>5s} {'lign.':>5s} {'div_pix':>7s} {'s/run':>6s}")
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
        ess80 = st.median(r["ess_at_schedule"][0] for r in var.values())
        nres = st.mean(r["n_resamplings"] for r in var.values())
        sec = st.mean(r["seconds"] for r in var.values())
        # diversité des k finales : racines x_T distinctes et RMSE pixel moyenne entre paires ;
        # absents des JSON écrits avant le 20/09 au soir
        lign = [r["n_lineages"] for r in var.values() if "n_lineages" in r]
        div = [r["div_pix"] for r in var.values() if r.get("div_pix") is not None]
        lign_s = f"{st.mean(lign):5.2f}" if lign else "    -"
        div_s = f"{st.mean(div):7.4f}" if div else "      -"
        print(f"{f.stem:8s} {n:3d}  {m1:+.4f} ± {s1:.4f}  {w1:3d}/{n:<3d}  {m2:+.4f} ± {s2:.4f}  {w2:3d}/{n:<3d}  "
              f"{m3:+.4f} ± {s3:.4f}  {ess80:8.2f} {nres:5.2f} {lign_s} {div_s} {sec:6.1f}")

    # la référence elle-même, sur les mêmes prompts, pour lire l'écran à la bonne échelle
    communs = {k for f in Path(args.variants).glob("*.json")
               for k in par_cle(json.loads(f.read_text())["runs"], "fk4")}
    if communs:
        sub_fk = {k: v for k, v in ref_fk.items() if k in communs}
        sub_bon = {k: v for k, v in ref_bon.items() if k in communs}
        m, s, w, n = apparie(sub_fk, sub_bon, "ir_max")
        print(f"\nréférence fk4 - bon4 sur ces {n} prompts : {m:+.4f} ± {s:.4f}, {w}/{n} gagnés "
              f"(sur les 100 : +0.0619 ± 0.0259)")

    # la diversité de la référence vient d'une régénération séparée (mêmes seeds, donc mêmes x_T)
    if Path(args.baseline_div).exists():
        div_runs = json.loads(Path(args.baseline_div).read_text())["runs"]
        for name in ("bon4", "fk4"):
            rs = [r for r in div_runs if r["sampler"] == name and r.get("div_pix") is not None]
            if rs:
                lign = [r["n_lineages"] for r in rs if "n_lineages" in r]
                lign_s = f"{st.mean(lign):.2f} lignées, " if lign else ""
                print(f"référence {name} ({len(rs)} prompts) : {lign_s}div_pix {st.mean(r['div_pix'] for r in rs):.4f}")


if __name__ == "__main__":
    main()
