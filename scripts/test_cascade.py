"""Test de la cascade NLI + LLM extract_anomalie_cascade() sur quelques demandes.

Usage:
    uv run python scripts/test_cascade.py
    uv run python scripts/test_cascade.py --n 5
    uv run python scripts/test_cascade.py --model mistral:7b-instruct-q4_K_M

Pipeline testé :
    Parse PDF → items[i] → cascade NLI + LLM → AnomalieEnrichie

But : valider que la cascade tourne sur de vraies demandes ASNR, mesurer le
temps total par item (NLI + LLM combinés) et inspecter visuellement le JSON
produit (theme_nli, theme_llm, equipements, etc.).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.extraction import extract_anomalie_cascade, setup_llm_client  # noqa: E402
from ml.parsing import parse_letter  # noqa: E402

DEFAULT_PDF = ROOT / "data" / "raw" / "asnr" / "INSSN-CAE-2026-0206.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Nombre de demandes à traiter (défaut: 3).",
    )
    parser.add_argument(
        "--model",
        default="phi3:mini",
        help="Modèle Ollama (phi3:mini | mistral:7b-instruct-q4_K_M).",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="PDF à parser. Défaut : INSSN-CAE-2026-0206.pdf (Penly).",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    # 1. Parser la lettre
    print(f"📄 Parsing : {args.pdf.name}")
    result = parse_letter(args.pdf)
    demandes = [i for i in result["items"] if i["type"] == "demande"]
    print(f"   → {len(demandes)} demandes disponibles")

    # 2. Setup client LLM
    print(f"\n🤖 Setup client Instructor → Ollama (modèle: {args.model})")
    client = setup_llm_client()

    # 3. Cascade sur les N premières demandes
    items_to_test = demandes[: args.n]
    print(f"\n🔬 Cascade NLI + LLM sur {len(items_to_test)} demande(s)\n")

    elapsed_per_item: list[float] = []

    for item in items_to_test:
        print(f"{'=' * 70}")
        print(f"Demande {item['identifiant']} (criticité: {item['criticite']})")
        print(f"Texte: {item['texte'][:150]}{'…' if len(item['texte']) > 150 else ''}")
        print()

        # step 1 : appel à la cascade avec mesure du temps
        # Tu dois compléter ces 3 lignes :
        #   3. Calculer le temps    → différence avec t0
        # Stocke le résultat dans `anomalie` (de type AnomalieEnrichie) et
        # le temps écoulé en secondes dans `elapsed_s`.

        t0 = time.perf_counter()
        anomalie = extract_anomalie_cascade(item, client, model=args.model)
        elapsed_s = time.perf_counter() - t0

        # step 2 : afficher le résultat

        print(f"✓ extraite en {elapsed_s:.3f}")
        print(anomalie.model_dump_json(indent=2))

        elapsed_per_item.append(elapsed_s)
        print()

    # 4. Stats agrégées (déjà fait, pas un TODO)
    if elapsed_per_item:
        avg = sum(elapsed_per_item) / len(elapsed_per_item)
        print(f"{'=' * 70}")
        print(f"⏱️  Temps moyen par item (cascade complète) : {avg:.1f}s")
        print(f"   Extrapolation 934 items : {avg * 934 / 60:.0f} min = {avg * 934 / 3600:.1f} h")


if __name__ == "__main__":
    main()
