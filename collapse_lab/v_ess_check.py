"""V. Does the recorded ESS equal the ESS the recorded log-weights imply? (plan, control 2.5)

For every run, logW is rebuilt from logG_at_schedule alone: it adds logG at each scheduled
step, the ESS is read there, and logW restarts from zero when that ESS falls under
threshold * k (smc/weights.py, should_resample), as fk_steer does. The recorded
ess_at_schedule must agree up to the 4-decimal rounding of logG. A disagreement would mean
the weights on disk are not the weights the resampler saw.

    python3 collapse_lab/v_ess_check.py [collapse_lab/out/probe_C.json]
"""
import json, sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "out" / "probe_C.json"
worst, n, flips, edge = {}, 0, 0, 0
for r in json.loads(path.read_text())["runs"]:
    if "logG_at_schedule" not in r:
        continue
    k, logW = r["k"], np.zeros(r["k"])
    for lg, rec in zip(r["logG_at_schedule"], r["ess_at_schedule"]):
        logW = logW + np.array(lg)
        w = np.exp(logW - logW.max())
        w /= w.sum()
        e = 1 / (w ** 2).sum()
        d = abs(e - rec)
        worst[r["arm"]] = max(worst.get(r["arm"], 0.0), d)
        n += 1
        if (e < r["threshold"] * k) != (rec < r["threshold"] * k):
            # the recorded ESS is rounded to 4 decimals: 3.99998 is written 4.0
            edge += abs(e - r["threshold"] * k) < 1e-4
            flips += abs(e - r["threshold"] * k) >= 1e-4
        if e < r["threshold"] * k:
            logW = np.zeros(k)
print(f"{path.name}: {n} scheduled steps; max |ESS rebuilt - ESS recorded| per arm: "
      + ", ".join(f"{a} {v:.1e}" for a, v in sorted(worst.items()))
      + f"; resampling decisions that differ: {flips}, plus {edge} within the rounding of the record")
