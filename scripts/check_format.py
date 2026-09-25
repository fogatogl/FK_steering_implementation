"""Check the built post against the conference's submission rules, file by file.

Usage: python scripts/check_format.py [docs/post/build]

Reads the three things the PR will carry: _posts/<slug>.md, assets/img/<slug>/,
assets/bibliography/<slug>.bib (and assets/html/<slug>/ if present). The rules are those of
the conference repository, read on 23/09/2026: bin/filter_paths.py (slug template
2026-\\d\\d-\\d\\d-.+, only those paths), REQUIREMENTS.md (front matter fields, description
without math, links or images, every ## heading in toc, d-cite keys in the .bib, date shown
as 2026-04-28) and the example post _posts/2026-04-28-my-blog-post.md.
Prints one line per check, FAIL or ok, then the word counts. Exit 1 on any FAIL.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLUG_TEMPLATE = r"2026-\d\d-\d\d-.+"
TEMPLATE_DATE = "2026-04-28"
FORBIDDEN = ["giulio", "fogato", "ensae", "onyxia", "sspcloud", "insee", "gfogato", "fogatogl"]
MAX_WORDS, MAX_IMG_MB = 3500, 15


def words(text):
    text = re.sub(r"\{%.*?%\}|<d-cite.*?</d-cite>|\$\$.*?\$\$|\$[^$]*\$|<[^>]+>|```.*?```", " ", text, flags=re.S)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'%.-]*", text))


def main():
    build = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "post" / "build"
    posts = sorted((build / "_posts").glob("*.md"))
    results = []

    def check(ok, what):
        results.append(ok)
        print(("ok   " if ok else "FAIL ") + what)

    check(len(posts) == 1, f"one post in {build / '_posts'} ({len(posts)} found)")
    if not posts:
        return 1
    post = posts[0]
    slug = post.stem
    check(re.match(SLUG_TEMPLATE, slug) is not None and slug == slug.lower() and " " not in slug,
          f"slug {slug} matches {SLUG_TEMPLATE}, lower case, no space")
    img, bib = build / "assets" / "img" / slug, build / "assets" / "bibliography" / f"{slug}.bib"
    check(img.is_dir(), f"image folder assets/img/{slug}/")
    check(bib.exists(), f"bibliography assets/bibliography/{slug}.bib")
    allowed = [f"_posts/{slug}.md", f"assets/img/{slug}/", f"assets/html/{slug}/", f"assets/bibliography/{slug}.bib"]
    stray = [str(p.relative_to(build)) for p in build.rglob("*") if p.is_file()
             and not any(str(p.relative_to(build)).startswith(a) for a in allowed)]
    check(not stray, "no file outside the four accepted paths" + (f": {stray}" if stray else ""))

    text = post.read_text()
    m = re.match(r"---\n(.*?)\n---\n(.*)", text, flags=re.S)
    check(m is not None, "front matter between two --- lines")
    if not m:
        return 1
    front, body = m.group(1), m.group(2)

    def field(name):
        f = re.search(rf"^{name}:\s*(.*)$", front, flags=re.M)
        return f.group(1).strip().strip('"') if f else None
    check(field("layout") == "distill", f"layout: distill ({field('layout')})")
    for name in ("title", "description", "date", "bibliography"):
        check(bool(field(name)), f"{name} present")
    check(re.search(r"^authors:\s*\n\s*-\s*name:\s*\"?Anonymous\"?\s*$", front, flags=re.M) is not None
          and len(re.findall(r"^\s*-\s*name:", front.split("toc:")[0], flags=re.M)) == 1,
          "authors: Anonymous, and only Anonymous")
    check(field("bibliography") == f"{slug}.bib", f"bibliography: {slug}.bib ({field('bibliography')})")
    date = field("date") or ""
    check(slug.startswith(date + "-"), f"slug starts with the date field ({date})")
    if date != TEMPLATE_DATE:
        print(f"note date is {date}; REQUIREMENTS.md shows {TEMPLATE_DATE} in the date field (ask the organisers)")
    desc = field("description") or ""
    check(not re.search(r"\$|https?://|\]\(|!\[|<img|\{%", desc), "description without math, link or image")
    n_sent = len(re.findall(r"[.!?](\s|$)", desc))
    check(2 <= n_sent <= 3, f"description in 2-3 sentences ({n_sent})")

    toc = re.findall(r"^\s*-\s*name:\s*\"?(.*?)\"?\s*$", front.split("toc:", 1)[1], flags=re.M) if "toc:" in front else []
    heads = [h.strip() for h in re.findall(r"^## (.*)$", body, flags=re.M)]
    check(toc == heads, "toc equals the ## headings, in order"
          + ("" if toc == heads else f": toc-only {sorted(set(toc) - set(heads))}, text-only {sorted(set(heads) - set(toc))}"))

    keys = {k.strip() for c in re.findall(r'<d-cite key="([^"]+)"', body) for k in c.split(",")}
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib.read_text())) if bib.exists() else set()
    check(keys <= bib_keys, f"{len(keys)} cited keys all in the .bib" + (f", missing {sorted(keys - bib_keys)}" if keys - bib_keys else ""))
    check("[@" not in body, "no Pandoc citation left ([@key])")
    paths = re.findall(r'path="([^"]+)"', body)
    missing = [p for p in paths if not (build / p).exists()]
    check(not missing, f"{len(paths)} included figures exist" + (f", missing {missing}" if missing else ""))
    outside = [p for p in paths if not p.startswith(f"assets/img/{slug}/") and not p.startswith(f"assets/html/{slug}/")]
    check(not outside, "every include points inside the post's folders" + (f": {outside}" if outside else ""))
    bad_ext = [p.name for p in img.glob("*") if p.suffix.lower() not in (".png", ".jpg", ".jpeg")] if img.is_dir() else []
    check(not bad_ext, "images are PNG or JPG" + (f" (test the others in the local build: {bad_ext})" if bad_ext else ""))
    mb = sum(p.stat().st_size for p in img.rglob("*") if p.is_file()) / 1e6 if img.is_dir() else 0
    check(mb < MAX_IMG_MB, f"image folder {mb:.1f} MB (< {MAX_IMG_MB})")

    low = text.lower()
    hits = [n for n in FORBIDDEN if n in low]
    check(not hits, "no author, school or platform name in the post" + (f": {hits}" if hits else ""))

    main_text = body.split("## Appendix")[0]
    captions = re.findall(r'caption="([^"]*)"', main_text)
    n_prose, n_capt = words(main_text), sum(words(c) for c in captions)
    check(n_prose + n_capt <= MAX_WORDS, f"{n_prose + n_capt} words in the main text (<= {MAX_WORDS}): "
          f"{n_prose} of prose and {n_capt} in {len(captions)} captions; appendix {words(body) - n_prose}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
