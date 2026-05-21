"""Pipeline complet : PDF -> cascade NLI+LLM -> embedding -> insertion Postgres+pgvector.
Avec tracking MLflow pour comparer les configs entre runs.

Usage:
    docker compose up -d                          # demarrer DB + MLflow + MinIO
    uv run python scripts/run_full_pipeline.py --n 5
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.embeddings import embed_anomalie  # noqa: E402
from ml.extraction import extract_anomalie_cascade, setup_llm_client  # noqa: E402
from ml.parsing import parse_letter  # noqa: E402
from ml.store import get_connection, insert_anomalie  # noqa: E402

DEFAULT_PDF = ROOT / "data" / "raw" / "asnr" / "INSSN-CAE-2026-0206.pdf"

# Constantes du pipeline (loggees comme params dans MLflow pour tracage).
NLI_SEUIL = 0.5  # cf. ml/classification.py - fallback "autre" si score < seuil
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Setup MLflow au niveau module : tracking server + credentials MinIO.
# setdefault permet de surcharger via env shell sans casser le default local.
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "asnr_minio")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "asnr_minio_password")
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("asnr-pipeline-ingestion")


def _git_short_sha() -> str:
    """Sha court du HEAD git (ou 'unknown' si hors d'un repo)."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--model", default="phi3:mini")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    lettre = args.pdf.stem.strip()

    print(f"Parsing : {args.pdf.name}")
    result = parse_letter(args.pdf)
    demandes = [i for i in result["items"] if i["type"] == "demande"][: args.n]
    print(f"  {len(demandes)} demandes a traiter (lettre={lettre})")

    with mlflow.start_run(run_name=f"pipeline-{lettre}-n{args.n}"):
        mlflow.log_params(
            {
                "pdf_filename": args.pdf.name,
                "lettre": lettre,
                "n_demandes_max": args.n,
                "llm_model": args.model,
                "nli_seuil": NLI_SEUIL,
                "embedding_model": EMBEDDING_MODEL,
            }
        )

        mlflow.set_tags({"git_sha": _git_short_sha(), "host": socket.gethostname()})

        print(f"\nSetup client Instructor + Ollama (modele: {args.model})")
        client = setup_llm_client()

        print("\nConnexion a la base Postgres")
        conn = get_connection()
        t_total_start = time.perf_counter()

        n_fallback_nli = 0  # items ou theme_nli == "autre" (sous seuil)
        items_summary: list[dict] = []  # pour log_artifact en M5
        n_divergences = 0  # items ou theme_nli != theme_llm

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

                    if anomalie.theme_nli != anomalie.theme_llm:
                        n_divergences += 1
                    if anomalie.theme_nli == "autre":
                        n_fallback_nli += 1
                    items_summary.append(
                        {
                            "identifiant": anomalie.identifiant,
                            "theme_nli": anomalie.theme_nli,
                            "theme_nli_score": anomalie.theme_nli_score,
                            "theme_llm": anomalie.theme_llm,
                            "duree_s": round(elapsed_s, 2),
                        }
                    )

                except Exception as exc:
                    # Graceful degradation : on logue et on continue sur l'item suivant
                    # plutot que de planter tout le batch.
                    elapsed_s = time.perf_counter() - t0
                    n_failed += 1
                    print(
                        f"  [FAIL] {item['identifiant']} en {elapsed_s:.1f}s : {type(exc).__name__}"
                    )

            duree_totale_s = time.perf_counter() - t_total_start
            n_traites = n_inserted + n_skipped  # items qui n'ont pas plante a la cascade
            duree_moyenne_s = duree_totale_s / n_traites if n_traites else 0.0

            mlflow.log_metrics(
                {
                    "n_demandes_total": len(demandes),
                    "n_inserted": n_inserted,
                    "n_skipped": n_skipped,
                    "n_failed": n_failed,
                    "duree_totale_s": duree_totale_s,
                    "duree_moyenne_par_item_s": duree_moyenne_s,
                    "taux_divergence_nli_llm": n_divergences / n_traites if n_traites else 0,
                    "taux_fallback_nli_autre": n_fallback_nli / n_traites if n_traites else 0,
                }
            )

            path = ROOT / "items_summary.json"
            path.write_text(
                json.dumps(items_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(path))
            path.unlink()  # nettoyage local apres upload S3

            print(f"\n[OK] {n_inserted} inserees, {n_skipped} skippees, {n_failed} echecs")
            print(f"     duree totale {duree_totale_s:.1f}s, moyenne {duree_moyenne_s:.1f}s/item")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
