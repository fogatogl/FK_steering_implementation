"""E. Ce que les 19 variantes deja sur disque disent de l'effondrement.

n_lineages n'existe que depuis le 21/09 ; div_pix, lui, est dans tous les
fichiers et separe "quatre images identiques" de "quatre images differentes".
bon4 donne le plafond (quatre tirages libres), fk4 le plancher observe.
"""
import json, statistics as st
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
load = lambda p: json.loads((R / p).read_text())["runs"]

base = load("sd_baseline.json")
div = load("sd_baseline_div20.json")   # div_pix n'existe que dans ce fichier
ref = {r["prompt_id"]: r for r in base if r["sampler"] == "bon4" and r["seed"] == 2024}
plancher = st.mean(r["div_pix"] for r in div if r["sampler"] == "fk4")
plafond = st.mean(r["div_pix"] for r in div if r["sampler"] == "bon4")
print(__doc__.splitlines()[0], "\n")
print(f"  bon4 (4 tirages libres)  div_pix {plafond:.4f}   <- plafond")
print(f"  fk4  (reference, 20)       div_pix {plancher:.4f}   <- plancher\n")
print(f"  {'variante':10s} {'lam':>5s} {'seuil':>6s} {'calendrier':>22s} {'div_pix':>8s} "
      f"{'lignees':>8s} {'resampl':>8s} {'ir_max':>8s} {'ecart/fk4':>10s}")

fkref = {r["prompt_id"]: r for r in base if r["sampler"] == "fk4" and r["seed"] == 2024}
for nom in ("S80", "S60", "S40", "D10", "L2", "L5", "L20", "A05", "T1", "T2", "T1A05",
            "T2A05", "T1t", "T2t", "T1tA05", "T2tA05", "fk4_stat", "fk4_diff"):
    try:
        runs = load(f"sd_variants/{nom}.json")
    except FileNotFoundError:
        continue
    r0 = runs[0]
    com = [r for r in runs if r["prompt_id"] in fkref]
    d = [r["ir_max"] - fkref[r["prompt_id"]]["ir_max"] for r in com]
    lin = [r["n_lineages"] for r in runs if r.get("n_lineages")]
    sched = str(r0.get("schedule_t"))
    print(f"  {nom:10s} {r0.get('lam', 0):5.1f} {r0.get('threshold', 0):6.2f} {sched:>22s} "
          f"{(f'{st.mean(dv):.4f}' if (dv := [r['div_pix'] for r in runs if r.get('div_pix') is not None]) else '   -'):>8s} "
          f"{(f'{st.mean(lin):.2f}' if lin else '  -'):>8s} "
          f"{st.mean(r['n_resamplings'] for r in runs):8.2f} "
          f"{st.mean(r['ir_max'] for r in runs):+8.4f} "
          f"{(f'{st.mean(d):+.4f}' if d else '-'):>10s}")

print("\n  lecture : aucune variante ne remonte div_pix vers le plafond de bon4 sauf celles")
print("  qui affaiblissent le PREMIER pas (T2, lambda_1 = 0.4) ou le suppriment (S80 n'en a")
print("  qu'un). Le calendrier, la rampe, la forme du potentiel et le potentiel lui-meme")
print("  laissent div_pix au plancher.")
