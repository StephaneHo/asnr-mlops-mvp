"""Pipeline end-to-end : PDF -> AnomalieEnrichie + embedding -> JSON sur disque.

Usage:
    uv run python scripts/pipeline_e2e.py --n 5
    uv run python scripts/pipeline_e2e.py --pdf data/raw/asnr/INSSN-LYO-2026-0507.pdf --n 3

Premier appel : ~30-60s (telechargement E5) + ~70-150s par demande (cascade NLI + LLM).
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.embeddings import embed_anomalie  # noqa: E402
from ml.extraction import extract_anomalie_cascade, setup_llm_client  # noqa: E402
from ml.parsing import parse_letter  # noqa: E402

DEFAULT_PDF = ROOT / "data" / "raw" / "asnr" / "INSSN-CAE-2026-0206.pdf"
DEFAULT_OUT = ROOT / "data" / "processed" / "anomalies.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="phi3:mini")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing : {args.pdf.name}")
    result = parse_letter(args.pdf)
    demandes = [i for i in result["items"] if i["type"] == "demande"][: args.n]
    print(f"  {len(demandes)} demandes a traiter")

    print(f"\nSetup client Instructor + Ollama (modele: {args.model})")
    client = setup_llm_client()

    print(f"\nPipeline end-to-end sur {len(demandes)} demandes\n")

    output_records: list[dict] = []

    # Boucle d'orchestration
    for item in demandes:
        t0 = time.perf_counter()
        anomalie = extract_anomalie_cascade(item, client, model=args.model)
        vector = embed_anomalie(anomalie)
        elapsed_s = time.perf_counter() - t0

        record = {
            "anomalie": anomalie.model_dump(),  # Pydantic -> dict
            "embedding": vector,
        }
        output_records.append(record)
        print(f"  [OK] {anomalie.identifiant} en {elapsed_s:.1f}s ({len(vector)}D)")

    # Sauvegarde JSON
    args.out.write_text(
        json.dumps(output_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"\n[OK] {len(output_records)} anomalies enrichies + embeddings sauvegardees "
        f"dans {args.out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
