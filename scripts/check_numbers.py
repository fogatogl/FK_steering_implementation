"""Does every decimal and percentage of the post's main text come out of a script?

Usage: python3 scripts/check_numbers.py docs/paper.md

Looks each number of the main text (before "## Appendix", captions included) up in results/post_numbers.json,
the index scripts/post_numbers.py writes from the analysis scripts' outputs. A number matches
if some script prints a value that rounds to it at the precision the post shows (a percentage
also matches a fraction, 96 % against 0.96; a sign the prose turns into words, "0.35 under",
matches with a note). Hyperparameters and dates are whitelisted. Numbers found only in
docs/results.md are reported as "doc only": a script must print them before the freeze.
Prints one line per unmatched number with its line, then a count. Exit 1 if any number is
matched nowhere.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# settings of the runs, not results: guidance, steps, resolution, thresholds, k, lambda, seeds
WHITELIST = {"7.5", "0.5", "1.0", "1.5", "2.0", "10.0", "100", "512", "0.05", "2.1", "1.5"}
# values quoted from the FK paper's source (arXiv 2501.06848v5), outside the repository's records
PAPER = {"0.348": "appendix_experiments.tex:71, CLIP diversity of the base model, SD v1.4",
         "0.091": "appendix_experiments.tex:73, CLIP diversity of FK, lambda 10, 20-80-20",
         "0.104": "appendix_experiments.tex:39, SD v1.5, FK lambda 10, 20-80-20, CLIP diversity 0.1038",
         "0.225": "appendix_experiments.tex:41, SD v1.5, FK lambda 2, 20-80-20, CLIP diversity 0.2252",
         "0.312": "appendix_experiments.tex:43, SD v1.5 base, CLIP diversity 0.3115",
         "0.811": "appendix_experiments.tex:73, SD v1.4 FK lambda 10 20-80-20, IR mean",
         "0.927": "appendix_experiments.tex:73, SD v1.4 FK lambda 10 20-80-20, IR max",
         "0.783": "appendix_experiments.tex:72, SD v1.4 FK lambda 10 5-30-5, IR max"}
NUM = re.compile(r"(?<![\w.])([-+−]?)(\d+(?:\.\d+)?)(\s?%)?")


def values(strings):
    out = []
    for s in strings:
        pct = s.endswith("%")
        try:
            v = float(s.rstrip("%"))
        except ValueError:
            continue
        out.append(v)
        if pct:
            out.append(v / 100)
    return out


def main():
    post = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "paper.md")
    idx_path = ROOT / "results" / "post_numbers.json"
    if not idx_path.exists():
        sys.exit("results/post_numbers.json missing: run scripts/post_numbers.py first")
    idx = json.loads(idx_path.read_text())
    scripted = values(n for s in idx["scripts"].values() for n in s["numbers"])
    doc = values(re.findall(r"[-+]?\d+(?:\.\d+)?%?", (ROOT / "docs" / "results.md").read_text()))

    lines = post.read_text().split("## Appendix")[0].splitlines()
    missing, doc_only, n = [], [], 0
    in_math = False
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("$$"):
            in_math = not in_math if line.strip() == "$$" else in_math
            continue
        if in_math:
            continue
        text = re.sub(r"`[^`]*`|\$[^$]*\$|\]\([^)]*\)|\d{2}/\d{2}(/\d{4})?|20\d\d-\d\d-\d\d|\b(19|20)\d\d\b", " ", line)
        for sign, num, pct in NUM.findall(text):
            if "." not in num and not pct:
                continue
            n += 1
            if (num in WHITELIST or num in PAPER) and not pct:
                continue
            d = len(num.split(".")[1]) if "." in num else 0
            v = float(num) * (-1 if sign in "-−" and sign else 1)
            cands = [v / 100, v] if pct else [v]
            tol = 0.5 * 10 ** -d + 1e-9

            def hit(pool, c):
                return any(abs(x - c) <= tol for x in pool) or (pct and any(abs(x - c) <= tol / 100 for x in pool))
            if any(hit(scripted, c) for c in cands):
                continue
            if any(hit(scripted, -c) for c in cands):
                continue                       # the sign is carried by the words
            where = "doc only" if any(hit(doc, c) or hit(doc, -c) for c in cands) else "nowhere"
            (doc_only if where == "doc only" else missing).append((i, sign + num + (pct or "")))
    for i, x in missing:
        print(f"MISSING line {i}: {x}")
    for i, x in doc_only:
        print(f"doc only line {i}: {x}")
    print(f"{n} numbers in the main text; {len(missing)} found in no output, "
          f"{len(doc_only)} only in docs/results.md; index of {idx['generated']}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
