"""Pick the prompts shown in F1, F5, W1 and W2, by a rule fixed before the data.

Question: which benchmark prompts does the post show as images, and why these?

Reads
    data/imagereward-benchmark-prompts.json    the 100 prompts, in run order
    data/visual_pool.json                       eligibility, written from the text alone
    collapse_lab/out/session_D/probe_D.json     session D: ctl, lam0, floor2 in one process,
                                                with images/<arm>/ next to it

A prompt enters the ranking only if it is complete: the three arms exist, carry every field
in REQUIRED, come from the same process (same session_id, finding 19), and their finals and
Tweedie thumbnails are on disk. An incomplete prompt is never patched by hand: it is listed
in `excluded_incomplete` and the rule moves to the next one (amendment of 23/09 in
docs/visual_selection.md).

At seed 2024, free = lam0.ir[0], bo4 = max(lam0.ir), fk = ctl.ir_max:
    gains   8 prompts with fk > bo4, highest fk - free first, at most 2 per category
    median  1 prompt whose fk - bo4 is closest to the median over the complete pool
    loss    1 prompt: among fk < bo4, closest to the median of that subset
    F1      the three best gains from three different categories (kept as F1_first_rule).
            Amendment of 26/09, after the images were seen: the author chose the categories
            figure, animal and scene; F1 is the best gain of each
    F5     among the ten, ctl ends on 1 root and floor2 on >= 3; largest fk - free (kept as
            F5_first_rule). Amendment of 25/09, after the images were seen: among the complete
            pool, ctl on 1 root and floor2 on 4, the largest rise of floor2's mean ir over lam0's
    W1      the ten
    W2      the first 10 complete prompts of a fixed permutation of the eligible pool
            (random.Random(2026)); a missing prompt shifts the list by one, no more.
Ties go to the lower prompt id. Output: data/visual_selection.json. --no-files skips the
image check.
"""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 2024
CAP = 2
F1_CATS = ("figure", "animal", "scene")   # amendment of 26/09, chosen by eye
ARMS = ("ctl", "lam0", "floor2")
# the names probe.py writes (lam_max is None by design outside the adaptive arms)
REQUIRED = ("ir", "ir_max", "n_lineages", "lineages_trace", "ancestors_at_schedule",
            "r_at_schedule", "logG_at_schedule", "session_id")
STEPS_T = (80, 60, 40, 20, 0)


def load_runs(path):
    by = {}
    for r in json.loads(Path(path).read_text())["runs"]:
        if r.get("seed", SEED) == SEED and not r.get("partial_from"):
            by.setdefault(r["arm"], {})[r["prompt_id"]] = r
    return by


def missing_parts(pid, runs, img, check_files):
    miss = []
    recs = {a: runs.get(a, {}).get(pid) for a in ARMS}
    for a, r in recs.items():
        if r is None:
            miss.append(f"{a}:record")
            continue
        miss += [f"{a}:{f}" for f in REQUIRED if r.get(f) in (None, [], "")]
        if len(r.get("ir") or []) != 4:
            miss.append(f"{a}:ir_len")
    if len({r.get("session_id") for r in recs.values() if r}) > 1:
        miss.append("arms_from_different_sessions")
    if check_files:
        for a in ARMS:
            if not all((img / a / f"{pid}_{j}.png").exists() for j in range(4)):
                miss.append(f"{a}:final_images")
            if not all((img / a / f"{pid}_t{t}_{j}.webp").exists() for t in STEPS_T for j in range(4)):
                miss.append(f"{a}:thumbnails")
    return miss


def check_pool(pool, bench):
    ids = [e["id"] for e in bench]
    classified = {e["id"] for e in pool["eligible"]}
    for v in pool["excluded"].values():
        classified |= set(v)
    unknown = classified - set(ids)
    missing = [i for i in ids if i not in classified]
    if unknown:
        sys.exit(f"pool ids not in the benchmark: {sorted(unknown)}")
    if missing:
        text = {e["id"]: e["prompt"] for e in bench}
        sys.exit("classify these prompts in data/visual_pool.json (rules E1-E4) first:\n"
                 + "\n".join(f"  {i}: {text[i]}" for i in missing))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_files = "--no-files" not in sys.argv
    src = Path(args[0]).resolve() if args else ROOT / "collapse_lab" / "out" / "session_D" / "probe_D.json"
    bench = json.loads((ROOT / "data" / "imagereward-benchmark-prompts.json").read_text())
    pool = json.loads((ROOT / "data" / "visual_pool.json").read_text())
    check_pool(pool, bench)
    runs = load_runs(src)

    cat = {e["id"]: e["cat"] for e in pool["eligible"]}
    rows, incomplete = [], {}
    for pid in sorted(cat):
        miss = missing_parts(pid, runs, src.parent / "images", check_files)
        if miss:
            incomplete[pid] = miss
            continue
        c, l, f = (runs[a][pid] for a in ARMS)
        rows.append(dict(id=pid, cat=cat[pid], free=l["ir"][0], bo4=max(l["ir"]), fk=c["ir_max"],
                         g_free=c["ir_max"] - l["ir"][0], g_bo4=c["ir_max"] - max(l["ir"]),
                         roots_ctl=c["n_lineages"], roots_floor2=f["n_lineages"],
                         rise_floor2=sum(f["ir"]) / len(f["ir"]) - sum(l["ir"]) / len(l["ir"])))
    if len(rows) < 20:
        sys.exit(f"only {len(rows)} complete eligible prompts; incomplete: "
                 + "; ".join(f"{p} ({', '.join(m[:3])})" for p, m in sorted(incomplete.items())[:5]))

    gains, used = [], {}
    for r in sorted((r for r in rows if r["g_bo4"] > 0), key=lambda r: (-r["g_free"], r["id"])):
        if used.get(r["cat"], 0) < CAP:
            gains.append(r)
            used[r["cat"]] = used.get(r["cat"], 0) + 1
        if len(gains) == 8:
            break
    taken = {r["id"] for r in gains}
    med = statistics.median(r["g_bo4"] for r in rows)
    median = min((r for r in rows if r["id"] not in taken), key=lambda r: (abs(r["g_bo4"] - med), r["id"]))
    taken.add(median["id"])
    losses = [r for r in rows if r["g_bo4"] < 0 and r["id"] not in taken]
    if not losses:
        sys.exit("no losing prompt in the complete pool: report it, do not relax the rule")
    lmed = statistics.median(r["g_bo4"] for r in losses)
    loss = min(losses, key=lambda r: (abs(r["g_bo4"] - lmed), r["id"]))
    worst = min(losses, key=lambda r: (r["g_bo4"], r["id"]))

    ten = gains + [median, loss]
    f1, seen = [], set()
    for r in gains:
        if r["cat"] not in seen:
            f1.append(r["id"])
            seen.add(r["cat"])
        if len(f1) == 3:
            break
    f1_first, f1 = f1, [next(r["id"] for r in gains if r["cat"] == c) for c in F1_CATS]
    f5c = [r for r in ten if r["roots_ctl"] == 1 and r["roots_floor2"] >= 3]
    f5_first = (max(f5c, key=lambda r: (r["g_free"], r["id"])) if f5c
                else max(ten, key=lambda r: (r["roots_floor2"], r["id"])))["id"]
    f5_row = max((r for r in rows if r["roots_ctl"] == 1 and r["roots_floor2"] == 4),
                 key=lambda r: (r["rise_floor2"], r["id"]))
    f5 = f5_row["id"]
    order = sorted(cat)
    random.Random(2026).shuffle(order)
    complete = {r["id"] for r in rows}
    w2 = [p for p in order if p in complete][:10]

    hp = pool["hand_picked"]["id"]
    sessions = sorted({runs["ctl"][r["id"]]["session_id"] for r in rows})
    out = dict(rule="docs/visual_selection.md", source=str(src.relative_to(ROOT)), sessions=sessions,
               seed=SEED, pool_complete=len(rows), pool_median_g_bo4=round(med, 4),
               gains=[r["id"] for r in gains], median=median["id"], loss=loss["id"],
               worst_loss=worst["id"], F1=f1, F1_first_rule=f1_first, F5=f5, F5_first_rule=f5_first, W1=[r["id"] for r in ten], W2=w2,
               hand_picked=None if hp in {r["id"] for r in ten} else hp,
               excluded_incomplete=incomplete,
               rows={r["id"]: {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in r.items() if k != "id"} for r in ten + [worst, f5_row]})
    (ROOT / "data" / "visual_selection.json").write_text(json.dumps(out, indent=2))

    text = {e["id"]: e["prompt"] for e in bench}
    for pid, miss in sorted(incomplete.items()):
        print(f"  incomplete {pid}: {', '.join(miss)}")
    for role, r in [*(("gain", r) for r in gains), ("median", median), ("loss", loss), ("worst", worst)]:
        print(f"  {role:6s} {r['id']}  fk-free {r['g_free']:+.2f}  fk-bo4 {r['g_bo4']:+.2f}  "
              f"roots ctl/floor2 {r['roots_ctl']}/{r['roots_floor2']}  {text[r['id']][:55]}")
    print(f"pool median fk - bo4 over the {len(rows)} complete eligible prompts: {med:+.3f}")
    print(f"F1 {f1} (first rule: {f1_first})  F5 {f5} (floor2 mean ir {f5_row['rise_floor2']:+.2f} over lam0's; first rule: {f5_first})")
    print(f"W2 {w2}")
    if out["hand_picked"]:
        print(f"{hp} not selected by the rule: appendix only, labelled hand-picked")
    print(f"{len(rows)} complete eligible prompts, {len(incomplete)} incomplete and skipped, "
          f"session(s) {sessions}; selection written to data/visual_selection.json.")


if __name__ == "__main__":
    main()
