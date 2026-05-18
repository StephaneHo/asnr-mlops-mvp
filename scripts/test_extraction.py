"""Test rapide de l'extraction d'anomalies sur quelques demandes du corpus.

Usage:
    uv run python scripts/test_extraction.py
    uv run python scripts/test_extraction.py --model mistral:7b-instruct-q4_K_M
    uv run python scripts/test_extraction.py --n 5

Prend le corpus déjà téléchargé dans data/raw/asnr/, parse une lettre,
puis envoie les N premières demandes au LLM local via Instructor pour
extraire des Anomalie typées. Affiche le résultat structuré.

But : valider que l'intégration Pydantic + Instructor + Ollama tourne,
mesurer le temps moyen par demande, et calibrer la qualité du prompt.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.extraction import Anomalie, extract_anomalie, setup_llm_client  # noqa: E402
from ml.parsing import parse_letter  # noqa: E402

# 👉 TODO (toi) : changer cette lettre si tu veux tester sur un autre exploitant
#    (Penly = EDF). Bonnes alternatives parmi les 217 du corpus :
#    - INSSN-LYO-2026-0507.pdf (Orano)
#    - INSSN-MRS-2025-0689.pdf (CEA)
DEFAULT_PDF = ROOT / "data" / "raw" / "asnr" / "INSSN-CAE-2026-0206.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="phi3:mini",
        help="Modèle Ollama (phi3:mini rapide | mistral:7b-instruct-q4_K_M plus précis)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Nombre de demandes à extraire (limite pour ne pas attendre trop longtemps).",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="Chemin du PDF à parser. Défaut : INSSN-CAE-2026-0206.pdf",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    # 1. Parser la lettre pour obtenir les items
    print(f"Parsing : {args.pdf.name}")
    result = parse_letter(args.pdf)
    items = result["items"]
    print(f"   → {len(items)} items extraits par le parser")

    # 2. Setup du client Ollama via Instructor
    print(f"\nSetup client Instructor → Ollama (modèle: {args.model})")
    client = setup_llm_client()

    # 3. Pour chacune des N premières demandes : extraction structurée
    items_to_test = [i for i in items if i["type"] == "demande"][: args.n]
    print(f"\n Test sur {len(items_to_test)} demande(s)\n")

    for item in items_to_test:
        print(f"{'─' * 70}")
        print(f"Demande {item['identifiant']} (criticité: {item['criticite']})")
        print(f"Texte: {item['texte'][:200]}{'…' if len(item['texte']) > 200 else ''}")
        print()

        t0 = time.perf_counter()
        try:
            anomalie: Anomalie = extract_anomalie(item["texte"], client, model=args.model)
            elapsed = time.perf_counter() - t0
            print(f"✓ Anomalie extraite en {elapsed:.1f}s :")
            print(anomalie.model_dump_json(indent=2))
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"✗ Échec après {elapsed:.1f}s : {type(exc).__name__}: {exc}")

        print()


if __name__ == "__main__":
    main()
