"""Check the figure scripts and their exports against the figure rules of the post's editorial plan (section 2.2).

Usage: python scripts/check_figures.py [scripts or PNGs...]
Default: every scripts/plot_*.py, scripts/fig_*.py, collapse_lab/*_fig.py,
collapse_lab/{k_figure,p_two_rewards,q_image_grid}.py and every PNG under figures/.

On a script: no set_title / suptitle / fig.text title (rule 4), no twinx / pie / 3d
(rule 9), fontsize literals >= 9 (rule 6), every hex colour in scripts/figstyle.PALETTE
(rule 7), string literals in ASCII apart from the maths symbols below and free of
FRENCH_WORDS (rule 10). On a PNG: width >= 1400 px (rule 6). One line per finding,
exit 1 if anything failed.
"""
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

MATH_OK = set("λφσμ×−±≥≤→←Σ°…′²∈∞αβγητ")
FRENCH_WORDS = """la le les des une du est sont pas pour avec dans sur que qui racine racines
lignee lignees bras reech poids etape planifie seuil plancher cible moyenne ecart erreur
nombre valeur courbe legende bleu rouge vert gris noir apparie libre premier dernier
seule memes chaque tous toutes""".split()
FRENCH = re.compile(r"(?<![A-Za-z_])(" + "|".join(FRENCH_WORDS) + r")(?![A-Za-z_])", re.I)
FORBIDDEN = [
    (re.compile(r"\.set_title\s*\("), "set_title: no title inside the image (rule 4)"),
    (re.compile(r"\.suptitle\s*\("), "suptitle: no title inside the image (rule 4)"),
    (re.compile(r"\.twinx\s*\(|\.twiny\s*\("), "twinx: no double axis (rule 9)"),
    (re.compile(r"\.pie\s*\("), "pie chart (rule 9)"),
    (re.compile(r"projection\s*=\s*['\"]3d['\"]"), "3D axes (rule 9)"),
    (re.compile(r"\.set_facecolor\s*\((?!\s*['\"](?:none|white|#fff(?:fff)?)['\"])"), "coloured background (rule 9)"),
]
HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")
FONTSIZE = re.compile(r"(?<![A-Za-z_])(?:fontsize|labelsize|size)\s*=\s*(\d+(?:\.\d+)?)")
DEFAULT_SCRIPTS = (sorted((ROOT / "scripts").glob("plot_*.py")) + sorted((ROOT / "scripts").glob("fig_*.py"))
                   + sorted((ROOT / "collapse_lab").glob("*_fig.py"))
                   + [ROOT / "collapse_lab" / n for n in ("k_figure.py", "p_two_rewards.py", "q_image_grid.py")])
DEFAULT_PNGS = sorted((ROOT / "figures").glob("*.png"))


def palette():
    try:
        import figstyle
        return {c.upper() for c in figstyle.PALETTE}
    except Exception as e:  # matplotlib missing in this interpreter, say so once
        print(f"note: figstyle not importable ({e}); hex colours not checked")
        return None


def check_script(path, pal):
    src = path.read_text()
    out = []
    for i, line in enumerate(src.splitlines(), 1):
        for rx, msg in FORBIDDEN:
            if rx.search(line):
                out.append((i, msg, line.strip()[:80]))
        for m in FONTSIZE.finditer(line):
            if float(m.group(1)) < 9:
                out.append((i, f"fontsize {m.group(1)} < 9 pt (rule 6)", line.strip()[:80]))
        if pal is not None:
            for h in HEX.findall(line):
                if h.upper() not in pal:
                    out.append((i, f"colour {h} not in figstyle.PALETTE (rule 7)", line.strip()[:80]))
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError):
        toks = []
    for t in toks:
        if t.type != tokenize.STRING:
            continue
        s = t.string
        bad = sorted({c for c in s if ord(c) > 127 and c not in MATH_OK})
        if bad:
            out.append((t.start[0], f"non-ASCII {''.join(bad)!r} in a string (rule 10)", s.strip()[:80]))
        fr = sorted({m.group(1).lower() for m in FRENCH.finditer(s)})
        if fr:
            out.append((t.start[0], f"French {fr} in a string (rule 10)", s.strip()[:80]))
    return out


def check_png(path):
    try:
        from PIL import Image
    except ImportError:
        return [(0, "Pillow missing: width not checked", "")]
    w, h = Image.open(path).size
    return [] if w >= 1400 else [(0, f"width {w} px < 1400 (rule 6)", "")]


def rel(p):
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def main(argv):
    targets = [Path(a) if Path(a).is_absolute() else ROOT / a for a in argv] or DEFAULT_SCRIPTS + DEFAULT_PNGS
    pal = palette()
    total = 0
    for p in targets:
        if not p.exists():
            print(f"missing: {p}")
            continue
        found = check_png(p) if p.suffix == ".png" else check_script(p, pal)
        if found:
            total += len(found)
            print(f"{rel(p)}: {len(found)}")
            for i, msg, line in found:
                print(f"  {i:4d}  {msg}" + (f"    {line}" if line else ""))
    print("clean" if total == 0 else f"FAILED: {total} finding(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
