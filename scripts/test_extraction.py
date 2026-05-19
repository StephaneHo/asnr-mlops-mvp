"""Test de l'extraction LLM (Pydantic + Instructor + Ollama) sur N demandes.

Usage:
    uv run python scripts/test_extraction.py --n 3
    uv run python scripts/test_extraction.py --model mistral:7b-instruct-q4_K_M
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.extraction import Anomalie, extract_anomalie, setup_llm_client  # noqa: E402
from ml.parsing import parse_letter  # noqa: E402

DEFAULT_PDF = ROOT / "data" / "raw" / "asnr" / "INSSN-CAE-2026-0206.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="phi3:mini")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Parsing : {args.pdf.name}")
    result = parse_letter(args.pdf)
    items = result["items"]
    print(f"  {len(items)} items extraits")

    print(f"\nSetup client Instructor + Ollama (modele: {args.model})")
    client = setup_llm_client()

    items_to_test = [i for i in items if i["type"] == "demande"][: args.n]
    print(f"\nTest sur {len(items_to_test)} demande(s)\n")

    for item in items_to_test:
        print("-" * 70)
        print(f"Demande {item['identifiant']} (criticite: {item['criticite']})")
        print(f"Texte: {item['texte'][:200]}{'...' if len(item['texte']) > 200 else ''}\n")

        t0 = time.perf_counter()
        try:
            anomalie: Anomalie = extract_anomalie(item["texte"], client, model=args.model)
            elapsed = time.perf_counter() - t0
            print(f"[OK] {elapsed:.1f}s")
            print(anomalie.model_dump_json(indent=2))
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"[FAIL] {elapsed:.1f}s : {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()
