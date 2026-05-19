"""Pipeline complet : PDF -> cascade NLI+LLM -> embedding -> insertion Postgres+pgvector.

Usage:
    docker compose up -d                          # demarrer la DB
    uv run python scripts/run_full_pipeline.py --n 5
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.embeddings import embed_anomalie  # noqa: E402
from ml.extraction import extract_anomalie_cascade, setup_llm_client  # noqa: E402
from ml.parsing import parse_letter  # noqa: E402
from ml.store import get_connection, insert_anomalie  # noqa: E402

DEFAULT_PDF = ROOT / "data" / "raw" / "asnr" / "INSSN-CAE-2026-0206.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--model", default="phi3:mini")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    # Identifiant de la lettre source (= nom du fichier sans extension)
    lettre = args.pdf.stem.strip()

    print(f"Parsing : {args.pdf.name}")
    result = parse_letter(args.pdf)
    demandes = [i for i in result["items"] if i["type"] == "demande"][: args.n]
    print(f"  {len(demandes)} demandes a traiter (lettre={lettre})")

    print(f"\nSetup client Instructor + Ollama (modele: {args.model})")
    client = setup_llm_client()

    print("\nConnexion a la base Postgres")
    conn = get_connection()
    try:
        n_inserted = 0
        n_skipped = 0
        n_failed = 0
        for item in demandes:
            t0 = time.perf_counter()
            try:
                anomalie = extract_anomalie_cascade(item, client, model=args.model)
                vector = embed_anomalie(anomalie)
                inserted = insert_anomalie(conn, anomalie, vector, lettre)
                elapsed_s = time.perf_counter() - t0

                tag = "[INS]" if inserted else "[DUP]"
                if inserted:
                    n_inserted += 1
                else:
                    n_skipped += 1
                print(f"  {tag} {anomalie.identifiant} en {elapsed_s:.1f}s")
            except Exception as exc:
                # Graceful degradation : on logue et on continue sur l'item suivant
                # plutot que de planter tout le batch. Phi-3 sur CPU est non-deterministe
                # et echoue parfois la validation Pydantic (max_retries=0 dans extract_anomalie).
                elapsed_s = time.perf_counter() - t0
                n_failed += 1
                print(f"  [FAIL] {item['identifiant']} en {elapsed_s:.1f}s : {type(exc).__name__}")

        print(
            f"\n[OK] {n_inserted} inserees, {n_skipped} skippees (deja en base), {n_failed} echecs"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
