"""Test de classification zero-shot NLI sur les demandes d'une lettre.

Usage:
    uv run python scripts/test_classification.py --n 10

Premier appel : ~1-2 min (telechargement du modele mDeBERTa, ~280 Mo).
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.classification import classify_theme  # noqa: E402
from ml.parsing import parse_letter  # noqa: E402

DEFAULT_PDF = ROOT / "data" / "raw" / "asnr" / "INSSN-CAE-2026-0206.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Parsing : {args.pdf.name}")
    result = parse_letter(args.pdf)
    demandes = [i for i in result["items"] if i["type"] == "demande"]
    items_to_test = demandes[: args.n]
    print(f"  {len(items_to_test)}/{len(demandes)} demandes a classifier\n")

    print("Chargement du modele mDeBERTa (1er appel)...")
    t0 = time.perf_counter()
    first_theme, first_score = classify_theme(items_to_test[0]["texte"])
    first_elapsed = time.perf_counter() - t0
    print(f"  1er appel : {first_elapsed:.1f}s (inclut chargement)\n")

    print("Classification :")
    print(f"   {'ms':>6}  {'identifiant':<10}  {'score':<7}  theme")
    print(f"   {'-' * 6}  {'-' * 10}  {'-' * 7}  {'-' * 50}")

    predictions: list[tuple[str, float, float]] = [(first_theme, first_score, first_elapsed * 1000)]
    print(
        f"   {first_elapsed * 1000:>6.0f}  "
        f"{items_to_test[0]['identifiant']:<10}  "
        f"{first_score:.3f}    "
        f"{first_theme}"
    )

    for item in items_to_test[1:]:
        t0 = time.perf_counter()
        theme, score = classify_theme(item["texte"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        predictions.append((theme, score, elapsed_ms))
        print(f"   {elapsed_ms:>6.0f}  {item['identifiant']:<10}  {score:.3f}    {theme}")

    # Stats (hors 1er appel qui inclut le chargement)
    times_ms = [p[2] for p in predictions[1:]]
    avg_ms = sum(times_ms) / max(len(times_ms), 1)
    distribution = Counter(p[0] for p in predictions)

    print("\n=== STATS ===")
    print(f"  Temps moyen (hors 1er) : {avg_ms:.0f} ms / demande")
    print(
        f"  Extrapolation 934 items : {avg_ms * 934 / 1000:.1f} s = {avg_ms * 934 / 60_000:.1f} min"
    )
    print("\n  Distribution des themes :")
    for theme, count in distribution.most_common():
        pct = count / len(predictions) * 100
        print(f"    {theme:50s} : {count:3d} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
