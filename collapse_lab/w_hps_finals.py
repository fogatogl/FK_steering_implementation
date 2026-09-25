"""W. HPS v2.1 of every saved final of a session: the second number under F1's images.

Scores images/<arm>/<prompt_id>_<slot>.png next to the records with
hpsv2.score(images, prompt, hps_version="v2.1"), one call per prompt for its three arms (the
call reloads its checkpoint), as scripts/run_sd_baseline.py does on the same images in memory.
Writes hps_finals.json next to the records, {arm: {prompt_id: [k floats]}}, after every
prompt, and skips the prompts already there. The run records are not rewritten (R2).

    HF_HOME=/home/onyxia/work/hf_cache /home/onyxia/work/.venvs/sd/bin/python \
        collapse_lab/w_hps_finals.py collapse_lab/out/session_D/probe_D.json
"""
import json, sys
from pathlib import Path

import hpsv2
from PIL import Image

src = Path(sys.argv[1])
out = src.parent / "hps_finals.json"
runs = json.loads(src.read_text())["runs"]
done = json.loads(out.read_text()) if out.exists() else {}
by = {}
for r in runs:
    by.setdefault(r["prompt_id"], []).append(r)
for i, (pid, rs) in enumerate(by.items(), 1):
    if all(pid in done.get(r["arm"], {}) for r in rs):
        continue
    paths = [src.parent / "images" / r["arm"] / f"{pid}_{j}.png" for r in rs for j in range(r["k"])]
    scores = [float(s) for s in hpsv2.score([Image.open(p) for p in paths], rs[0]["prompt"], hps_version="v2.1")]
    for n, r in enumerate(rs):
        done.setdefault(r["arm"], {})[pid] = [round(s, 5) for s in scores[n * r["k"]:(n + 1) * r["k"]]]
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(done, indent=1))
    tmp.replace(out)
    print(f"[{i:3d}/{len(by)}] {pid} " + " ".join(f"{r['arm']}={max(done[r['arm']][pid]):.4f}" for r in rs), flush=True)
print(out)
