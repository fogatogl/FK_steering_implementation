"""Turn docs/paper.md into the three files the conference PR accepts.

Usage: python scripts/build_post.py [--slug 2026-10-07-four-particles-one-image] [--out docs/post/build]

Writes  <out>/_posts/<slug>.md            (distill front matter, figure.liquid includes, d-cite)
        <out>/assets/img/<slug>/*.png     (the figures the post includes, copied from figures/)
        <out>/assets/bibliography/<slug>.bib
from    docs/paper.md, whose figures are Markdown images with the caption as alt text and whose
citations are [@key] or [@a; @b], and docs/post/four-particles-one-image.bib.
Then checks: every cited key is in the .bib, every figure file exists, no author name, school or
personal URL in the post, and prints the main-text word count. Exit 1 if a check fails.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_NAMES = ["giulio", "fogato", "ensae", "onyxia"]
TITLE = "Four particles, one image: reproducing FK Steering on Stable Diffusion, and what its gain over best-of-N buys"
DESCRIPTION = ("I reimplement FK Steering (ICML 2025) from its equations and rerun its Stable Diffusion "
               "experiment. Best-of-4 lands on the paper, and the steering gain over it is about 40 % of the published "
               "one, while the four images a run returns descend from a single initial noise in 93 to 96 runs out of "
               "100. Replaying the recorded weights through the resampler recovers that count, and the post measures "
               "what keeping the lineages costs.")


def convert(md, slug, img_dir):
    body = md.split("\n", 1)[1]                       # drop the H1, the layout prints the title
    body = re.sub(r"^\*Draft, pass.*?\*\n", "", body, count=1, flags=re.S | re.M)
    figs, missing = [], []

    def fig(m):
        caption, path = m.group(1), Path(m.group(2))
        src = (ROOT / "docs" / path).resolve()
        figs.append(src)
        if not src.exists():
            missing.append(str(path))
        return ('{% include figure.liquid path="assets/img/' + slug + "/" + src.name +
                '" class="img-fluid" zoomable=true caption="' + caption.replace('"', "'") + '" %}')
    body = re.sub(r"!\[(.*?)\]\((.*?)\)", fig, body)
    keys = set()

    def cite(m):
        ks = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        keys.update(ks)
        return '<d-cite key="' + ",".join(ks) + '"></d-cite>'
    body = re.sub(r"\[(@[^\]]+)\]", cite, body)
    body = body.replace("## Appendix\n", "## Appendix\n\n*The reviewers are not asked to read past this point.*\n")
    toc = [re.sub(r"^#+\s*", "", l) for l in body.splitlines() if l.startswith("## ")]
    front = ["---", "layout: distill", f"title: \"{TITLE}\"", f"description: \"{DESCRIPTION}\"",
             "date: 2026-10-07", "future: true", "htmlwidgets: true", "hidden: false",
             "authors:", "  - name: Anonymous", f"bibliography: {slug}.bib", "toc:"]
    front += [f"  - name: \"{t}\"" for t in toc]
    front += ["---", ""]
    return "\n".join(front) + body, figs, missing, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="2026-10-07-four-particles-one-image")
    ap.add_argument("--out", default=str(ROOT / "docs" / "post" / "build"))
    a = ap.parse_args()
    out = Path(a.out)
    md = (ROOT / "docs" / "paper.md").read_text()
    bib_src = ROOT / "docs" / "post" / "four-particles-one-image.bib"
    post, figs, missing, keys = convert(md, a.slug, out / "assets" / "img" / a.slug)
    (out / "_posts").mkdir(parents=True, exist_ok=True)
    (out / "assets" / "img" / a.slug).mkdir(parents=True, exist_ok=True)
    for old in (out / "assets" / "img" / a.slug).iterdir():   # a figure the post dropped must not ship
        old.unlink()
    (out / "assets" / "bibliography").mkdir(parents=True, exist_ok=True)
    (out / "_posts" / f"{a.slug}.md").write_text(post)
    shutil.copy(bib_src, out / "assets" / "bibliography" / f"{a.slug}.bib")
    for f in figs:
        if f.exists():
            shutil.copy(f, out / "assets" / "img" / a.slug / f.name)
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_src.read_text()))
    bad = []
    if missing:
        bad.append("figures missing: " + ", ".join(missing))
    if keys - bib_keys:
        bad.append("citations not in the .bib: " + ", ".join(sorted(keys - bib_keys)))
    low = post.lower()
    hits = [n for n in FORBIDDEN_NAMES if n in low]
    if hits:
        bad.append("names in the post: " + ", ".join(hits))
    main_text = post.split("## Appendix")[0]
    words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'%.-]*", re.sub(r"\{%.*?%\}|<d-cite.*?</d-cite>|\$\$.*?\$\$|\$[^$]*\$|^---.*?---", "", main_text, flags=re.S | re.M)))
    print(f"{out / '_posts' / (a.slug + '.md')}: {words} words in the main text, {len(figs)} figures, {len(keys)} citation keys")
    for b in bad:
        print("FAILED " + b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
