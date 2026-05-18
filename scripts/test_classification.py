"""Test rapide de classification zero-shot NLI sur les demandes du corpus.

Usage:
    uv run python scripts/test_classification.py
    uv run python scripts/test_classification.py --n 20

But : valider que mDeBERTa zero-shot classifie correctement le thème
des demandes ASNR, mesurer le temps moyen par demande (~50-200ms attendu
sur CPU), et comparer la distribution des thèmes prédits avec celle
qu'on a obtenue via Phi-3/Mistral (qui mettait tout en "sûreté").

⚠️ Le PREMIER appel déclenche le téléchargement automatique du modèle
mDeBERTa (~280 Mo). Compte ~1-2 min de plus uniquement la première fois.
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
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Nombre de demandes à classifier (toutes par défaut: 10).",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="PDF à parser. Défaut : Penly (15 demandes).",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    # 1. Parser la lettre
    print(f"📄 Parsing : {args.pdf.name}")
    result = parse_letter(args.pdf)
    demandes = [i for i in result["items"] if i["type"] == "demande"]
    print(f"   → {len(demandes)} demandes disponibles")

    items_to_test = demandes[: args.n]
    print(f"   → {len(items_to_test)} à classifier\n")

    # 2. Premier appel : chargement du modèle (~5-10s) + 1er forward pass (~1-2s)
    #    Au premier appel TOUT court (première utilisation depuis l'installation),
    #    il y a aussi le DOWNLOAD du modèle (~280 Mo, peut prendre 1-2 min).
    print("🤖 Chargement du modèle mDeBERTa (1er appel, ~5-10s en cache, ~1-2 min sinon)...")
    t0 = time.perf_counter()
    first_theme, first_score = classify_theme(items_to_test[0]["texte"])
    first_elapsed = time.perf_counter() - t0
    print(f"   → 1er appel : {first_elapsed:.1f}s (inclut chargement + 1er forward pass)\n")

    # 3. Classification rapide de tout le reste
    print("🔬 Classification :")
    print(f"   {'ms':>6}  {'identifiant':<10}  {'score':<7}  thème prédit")
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

    # 4. Stats agrégées
    times_ms = [p[2] for p in predictions[1:]]  # exclure le 1er (qui inclut chargement)
    avg_ms = sum(times_ms) / max(len(times_ms), 1)
    distribution = Counter(p[0] for p in predictions)

    print("\n=== STATISTIQUES ===")
    print(f"  Temps moyen (hors 1er) : {avg_ms:.0f} ms / demande")
    print(
        f"  Extrapolation 934 items : {avg_ms * 934 / 1000:.1f} s = {avg_ms * 934 / 60_000:.1f} min"
    )
    print("\n  Distribution des thèmes prédits :")
    for theme, count in distribution.most_common():
        pct = count / len(predictions) * 100
        print(f"    • {theme:50s} : {count:3d} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
