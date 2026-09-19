"""Tire le sous-ensemble de prompts du protocole C6 : sans la liste, personne ne reproduit.

`random.Random(seed).sample` et pas torch : l'algorithme de `random` est figé par la
spec CPython, celui de torch peut bouger d'une version à l'autre.

Depuis la racine : `python scripts/sample_prompts.py --n 40 --out data/prompts_subset_40.json`.
"""
import argparse, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--input", default=str(ROOT / "data" / "imagereward-benchmark-prompts.json"))
    p.add_argument("--out", default=str(ROOT / "data" / "prompts_subset_40.json"))
    args = p.parse_args()

    pool = json.loads(Path(args.input).read_text())
    # Tri sur l'id : l'ordre du fichier amont ne doit pas décider du tirage.
    pool.sort(key=lambda e: e["id"])
    prompts = random.Random(args.seed).sample(pool, args.n)

    out = Path(args.out)
    out.write_text(json.dumps({
        "seed": args.seed,
        "n": args.n,
        "source": str(Path(args.input).relative_to(ROOT)),
        "source_size": len(pool),
        "sampler": "random.Random(seed).sample sur la source triée par id",
        "prompts": prompts,
    }, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
