"""Test de la cascade NLI + LLM (extract_anomalie_cascade) sur N demandes.

Usage:
    uv run python scripts/test_cascade.py --n 3
    uv run python scripts/test_cascade.py --model mistral:7b-instruct-q4_K_M
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
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--model", default="phi3:mini")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Parsing : {args.pdf.name}")
    result = parse_letter(args.pdf)
    demandes = [i for i in result["items"] if i["type"] == "demande"]
    print(f"  {len(demandes)} demandes disponibles")

    print(f"\nSetup client Instructor + Ollama (modele: {args.model})")
    client = setup_llm_client()

    items_to_test = demandes[: args.n]
    print(f"\nCascade NLI + LLM sur {len(items_to_test)} demande(s)\n")

    elapsed_per_item: list[float] = []
    for item in items_to_test:
        print("=" * 70)
        print(f"Demande {item['identifiant']} (criticite: {item['criticite']})")
        print(f"Texte: {item['texte'][:150]}{'...' if len(item['texte']) > 150 else ''}\n")

        t0 = time.perf_counter()
        anomalie = extract_anomalie_cascade(item, client, model=args.model)
        elapsed_s = time.perf_counter() - t0

        print(f"[OK] {elapsed_s:.1f}s")
        print(anomalie.model_dump_json(indent=2))
        elapsed_per_item.append(elapsed_s)
        print()

    if elapsed_per_item:
        avg = sum(elapsed_per_item) / len(elapsed_per_item)
        print("=" * 70)
        print(f"Temps moyen par item : {avg:.1f}s")
        print(f"Extrapolation 934 items : {avg * 934 / 60:.0f} min = {avg * 934 / 3600:.1f} h")


if __name__ == "__main__":
    main()
