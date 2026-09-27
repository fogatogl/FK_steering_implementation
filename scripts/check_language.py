"""Find French left in what the post and its readers will see.

Usage: python scripts/check_language.py [paths...]
Default paths: README.md collapse_lab/README.md figures/ notebooks/ scripts/ and the
collapse_lab scripts the post cites. For .py files only strings, docstrings and comments
are read; for .ipynb the Markdown cells, comments and strings of code cells, and text
outputs; for .md and .txt the whole file; for .svg the text elements. A hit is a frequent
French word on a word boundary, or an accented letter. One line per hit, a count per
file, exit 1 if anything was found.
"""
import io
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ["README.md", "collapse_lab/README.md", "figures", "notebooks", "scripts",
           "collapse_lab/n_coalescence.py", "collapse_lab/r_solutions.py",
           "collapse_lab/p_two_rewards.py", "collapse_lab/o_ancestry_fig.py",
           "collapse_lab/q_image_grid.py", "collapse_lab/s_coalescence_fig.py",
           "docs/demo", "collapse_lab/FINDINGS.md", "collapse_lab/ASSESSMENT.md", "docs/paper.md"]
# Frequent French words that are not English words. Kept short on purpose: a false
# positive costs a look, a missing word costs a French label in a published figure.
FRENCH_WORDS = """
le la les des une du au aux est sont pas avec dans sur que qui cette ces mais ou
donc tres aussi comme etre avoir fait faire nous vous ils elles leur leurs notre votre
ce cet celui celle ceux celles chaque tous toutes tout toute moins entre apres avant
racine racines lignee lignees bras poids reech reechantillonnage etape etapes pas planifie
planifies seuil plancher cible gauche droite haut bas moyenne ecart ecarts erreur nombre
valeur valeurs courbe courbes legende bleu rouge vert gris noir blanc
apparie appariee libre premiere dernier derniere seule seul
""".split()
ACCENTS = re.compile(r"[àâäéèêëîïôöùûüçœÀÂÄÉÈÊËÎÏÔÖÙÛÜÇŒ]")
WORD = re.compile(r"(?<![A-Za-z_])(" + "|".join(sorted(FRENCH_WORDS, key=len, reverse=True)) + r")(?![A-Za-z_])",
                  re.I)
# English homographs (plus, pour, premier, meme, figure, axes) are left out of the list.
# Record keys of probe.py's JSON that scripts have to spell to read them.
ALLOW = {"plancher", "adaptatif"}
ACCENTED_OK = ["Fréchet", "Björn"]   # proper names an English text spells with their accent


def hits(text):
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        ws = [m.group(1) for m in WORD.finditer(line) if m.group(1).lower() not in ALLOW]
        clean = line
        for w in ACCENTED_OK:
            clean = clean.replace(w, "")
        acc = ACCENTS.findall(clean)
        if ws or acc:
            out.append((i, ws, "".join(acc), line.strip()[:90]))
    return out


def py_text(src):
    """Strings, docstrings and comments of a Python source, with their line numbers kept."""
    keep = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                for k, l in enumerate(tok.string.splitlines()):
                    keep[tok.start[0] + k] = keep.get(tok.start[0] + k, "") + " " + l
    except (tokenize.TokenError, SyntaxError):
        return src
    n = max(keep) if keep else 0
    return "\n".join(keep.get(i, "") for i in range(1, n + 1))


def nb_text(src):
    nb = json.loads(src)
    lines = []
    for c in nb.get("cells", []):
        body = "".join(c.get("source", []))
        if c.get("cell_type") == "markdown":
            lines += body.splitlines()
        else:
            lines += py_text(body).splitlines()
            for o in c.get("outputs", []):
                t = o.get("text") or o.get("data", {}).get("text/plain") or []
                lines += "".join(t).splitlines()
    return "\n".join(lines)


def svg_text(src):
    return "\n".join(re.findall(r">([^<>]{2,})<", src))


def scan(path):
    src = path.read_text(errors="replace")
    if path.suffix == ".py":
        text = py_text(src)
    elif path.suffix == ".ipynb":
        text = nb_text(src)
    elif path.suffix == ".svg":
        text = svg_text(src)
    elif path.suffix in (".md", ".txt", ".sh", ".json", ".html", ".js", ".css"):
        text = src
    else:
        return None
    return hits(text)


def main(argv):
    targets = argv or DEFAULT
    files = []
    for t in targets:
        p = ROOT / t if not Path(t).is_absolute() else Path(t)
        if p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.is_file() and "__pycache__" not in q.parts)
        elif p.exists():
            files.append(p)
        else:
            print(f"missing: {t}")
    total = 0
    for f in files:
        if f.name in ("check_language.py", "check_prose.py", "check_figures.py"):
            continue
        h = scan(f)
        if not h:
            continue
        total += len(h)
        rel = f.relative_to(ROOT) if f.is_relative_to(ROOT) else f
        print(f"{rel}: {len(h)} line(s)")
        for i, ws, acc, line in h[:12]:
            tag = " ".join(ws) + (f" accents:{acc}" if acc else "")
            print(f"  {i:4d}  [{tag}]  {line}")
        if len(h) > 12:
            print(f"  ... {len(h) - 12} more")
    print("clean" if total == 0 else f"FRENCH: {total} line(s) in {sum(1 for _ in files)} file(s) scanned")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
