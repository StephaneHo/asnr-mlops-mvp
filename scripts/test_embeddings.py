"""Test des embeddings sentence-transformers sur quelques demandes.

Usage:
    uv run python scripts/test_embeddings.py --n 3

Premier appel : ~30-60s (téléchargement du modèle E5, ~120 Mo).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.embeddings import embed_text  # noqa: E402
from ml.parsing import parse_letter  # noqa: E402
from sentence_transformers.util import cos_sim  # noqa: E402

DEFAULT_PDF = ROOT / "data" / "raw" / "asnr" / "INSSN-CAE-2026-0206.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=3, help="Nombre de demandes.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Parsing : {args.pdf.name}")
    result = parse_letter(args.pdf)
    demandes = [i for i in result["items"] if i["type"] == "demande"][: args.n]
    print(f"  {len(demandes)} demandes a encoder\n")

    print("Embedding (1er appel = chargement modele, ~30-60s)")
    vectors: list[list[float]] = []
    for item in demandes:
        t0 = time.perf_counter()
        vector = embed_text(item["texte"])
        elapsed_s = time.perf_counter() - t0
        vectors.append(vector)
        print(f"  [{elapsed_s:.1f}s] {item['identifiant']} -> {len(vector)}D")

    print("\nMatrice de similarite cosinus :")
    print(f"   {'':10}", *[f"{d['identifiant']:>10}" for d in demandes])
    sim_matrix = cos_sim(vectors, vectors)
    for i, item in enumerate(demandes):
        scores = [f"{sim_matrix[i][j].item():>10.3f}" for j in range(len(demandes))]
        print(f"   {item['identifiant']:<10}", *scores)

    print("\nLecture : diagonale = 1.0. Hors diagonale : >0.85 tres proche, <0.5 different.")


if __name__ == "__main__":
    main()
