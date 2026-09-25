"""Regenerate every number the post may quote, from the committed records, into one index.

Usage: /home/onyxia/work/.venvs/ddpm/bin/python scripts/post_numbers.py   (CPU, about 10 min)

Runs each analysis script below from its own folder, keeps its console output in
results/post_numbers/<name>.txt, and writes results/post_numbers.json: for each script, the
command, the output file, the record files it reads, and every number of its output.
scripts/check_numbers.py then looks each number of the post up in that index. A number the
post quotes that no script prints has no provenance.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "post_numbers"
SCRIPTS = [f"collapse_lab/{n}.py" for n in (
    "a_step0_ess", "b_comb", "c_information", "d_forms", "e_variants", "f_probe", "h_invariances",
    "i_root100", "j_decomposition", "l_target_ess", "n_coalescence", "p_two_rewards", "r_solutions",
    "t_sessionC", "u_determinism", "v_ess_check", "y_sessionD")] + [
    "collapse_lab/ref/parse_authors.py", "scripts/compare_sd_variants.py", "scripts/make_table_sd.py",
    "scripts/select_visual_prompts.py",   # deterministic: rewrites data/visual_selection.json identically
    "scripts/fig_three_scales.py"]        # F8's values; redraws figures/f8_three_scales identically
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "python": sys.executable, "scripts": {}}
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for s in SCRIPTS:
        path = ROOT / s
        name = path.stem
        p = subprocess.run([sys.executable, path.name], cwd=path.parent, env=env,
                           capture_output=True, text=True, timeout=1200)
        if p.returncode:
            sys.exit(f"{s} failed:\n{p.stderr[-2000:]}")
        (OUT / f"{name}.txt").write_text(p.stdout)
        reads = sorted(set(re.findall(r"[\w./-]+\.json", path.read_text())))
        index["scripts"][name] = {"command": f"cd {path.parent.relative_to(ROOT)} && python {path.name}",
                                  "output": str((OUT / f"{name}.txt").relative_to(ROOT)),
                                  "reads": reads, "numbers": NUMBER.findall(p.stdout)}
        print(f"{s}: {len(index['scripts'][name]['numbers'])} numbers", flush=True)
    (ROOT / "results" / "post_numbers.json").write_text(json.dumps(index, indent=1))
    print("results/post_numbers.json written")


if __name__ == "__main__":
    main()
