"""Grep the post for the tics a reader recognises as generated text.

Usage: python scripts/check_prose.py docs/paper.md [--appendix-marker "## Appendix"]

Applies the part of the post's style rules (editorial plan, section 4.2) that a grep can see: the blocking
lexicon, sentence-opening connectors (warning above three per section), em-dashes
outside code (blocking), bold outside the repo's convention (warning), Title Case
headings (blocking), and the spread of paragraph lengths. One line per section, then
a verdict: `clean` or `BLOCKING`. Exit code 1 on BLOCKING. The structures the plan
lists (rule of three, negative parallelism, trailing participles, summary sentences)
are for the rereading, not for this script.
"""
import argparse
import re
import sys
from pathlib import Path

# the editorial plan's section 4.2, plus the words found zero times in docs/.
LEXICON = [
    "delve", "tapestry", "testament", "pivotal", "crucial", "robust", "leverage",
    "landscape", "navigate", "underscore", "underscoring", "foster", "showcase", "seamless",
    "intricate", "realm", "harness", "elevate", "unlock", "journey", "game-changer",
    "cutting-edge", "state-of-the-art", "it is worth noting", "it is important to note",
    "in conclusion", "overall", "in summary", "ultimately", "in today's", "plays a role",
    "serves as", "stands as", "genuinely", "honestly", "straightforward",
    "however", "note that", "comprehensive", "meticulous", "we will explore", "let us dive",
]
CONNECTORS = ["Moreover", "Furthermore", "Additionally", "Notably", "Importantly",
              "Interestingly", "Crucially"]
# Bold allowed mid-paragraph, once each: defined terms. Fill in as the post defines them.
BOLD_ALLOWED = {"lineage"}  # defined terms, once each; glossary labels are checked by heading
SMALL = {"a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "by",
         "with", "from", "as", "is", "vs", "but", "not", "its", "what", "where", "how"}


def strip_code(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    text = re.sub(r"\$[^$\n]*\$", "", text)
    text = re.sub(r"<d-cite[^>]*>.*?</d-cite>", "", text, flags=re.S)
    text = re.sub(r"\{%.*?%\}", "", text, flags=re.S)
    return text


def sections(text):
    """[(heading, body)], with the front matter and the preamble under heading ''."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            text = text[end + 4:]
    out, head, buf = [], "", []
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if line.startswith("#") and not fenced:
            out.append((head, "\n".join(buf)))
            head, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    out.append((head, "\n".join(buf)))
    return out


def title_case(heading):
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", heading)]
    if len(words) < 3:
        return False
    cand = [w for w in words[1:] if w.lower() not in SMALL and not w.isupper()]
    caps = [w for w in cand if w[0].isupper()]
    return len(cand) >= 2 and len(caps) == len(cand)


def words_of(text):
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'%.-]*", text))


def check(path, appendix_marker):
    raw = Path(path).read_text()
    blocking, warnings = [], []
    main_words = appendix_words = 0
    in_appendix = False
    lines = []
    for head, body in sections(raw):
        if head and appendix_marker and appendix_marker.lstrip("# ").lower() in head.lower():
            in_appendix = True
        clean = strip_code(body)
        n = words_of(clean)
        if in_appendix:
            appendix_words += n
        else:
            main_words += n
        found = []
        for w in LEXICON:
            hits = re.findall(r"(?<![A-Za-z-])" + re.escape(w) + r"(?![A-Za-z-])", clean, flags=re.I)
            if hits:
                found.append(f"{w}x{len(hits)}")
        if found:
            blocking.append(f"LEXICON in '{head or '(preamble)'}': {', '.join(found)}")
        conn = sum(len(re.findall(r"(?:^|[.!?]\s+)" + c + r"\b", clean, flags=re.M)) for c in CONNECTORS)
        if conn > 3:
            warnings.append(f"connectors in '{head or '(preamble)'}': {conn} sentence-opening (max 3)")
        dashes = clean.count("—")
        if dashes:
            blocking.append(f"EM-DASH in '{head or '(preamble)'}': {dashes}")
        if head and title_case(head):
            blocking.append(f"Title Case heading: '{head}'")
        paras = [p for p in re.split(r"\n\s*\n", clean) if p.strip() and not p.lstrip().startswith("|")]
        bolds, results = [], []
        for p in paras:
            for m in re.finditer(r"\*\*(.+?)\*\*", p):
                inner = m.group(1).strip()
                label = p[:m.start()].strip() == ""
                if label and inner.endswith((".", ":")):
                    continue                      # the repo's paragraph-opening label
                if inner in BOLD_ALLOWED or head.startswith("A.1"):
                    continue
                if re.search(r"\d", inner) and len(inner.split()) <= 6:
                    results.append(inner[:40])    # the one number that is the result
                else:
                    bolds.append(inner[:40])
        if bolds:
            warnings.append(f"bold in prose in '{head or '(preamble)'}': {bolds}")
        if len(results) > 1:
            warnings.append(f"bold results in '{head or '(preamble)'}': {len(results)}, the rules allow one per block: {results}")
        plen = sorted(words_of(p) for p in paras)
        spread = f"{plen[0]}-{plen[-1]} words/para, median {plen[len(plen) // 2]}" if plen else "no prose"
        lines.append(f"  {head or '(preamble)':55.55s} {n:5d} w  {len(paras):2d} paras  {spread}"
                     f"{'  conn ' + str(conn) if conn else ''}{'  ' + str(dashes) + ' em-dash' if dashes else ''}")
    print(f"{path}: {main_words} words in the main text, {appendix_words} in the appendix")
    print("\n".join(lines))
    if main_words > 3500:
        warnings.append(f"WARNING length: {main_words} words in the main text, above the 3,500 the call recommends")
    for w in warnings:
        print("WARNING " + w if not w.startswith("WARNING") else w)
    for b in blocking:
        print("BLOCKING " + b)
    print("BLOCKING: %d issue(s)" % len(blocking) if blocking else "clean")
    return 1 if blocking else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("post", nargs="?", default="docs/paper.md")
    ap.add_argument("--appendix-marker", default="Appendix",
                    help="heading text from which words stop counting toward the main text")
    a = ap.parse_args()
    sys.exit(check(a.post, a.appendix_marker))
