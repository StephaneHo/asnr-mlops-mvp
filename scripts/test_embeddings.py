"""Test rapide des embeddings sentence-transformers sur quelques demandes.

Usage:
    uv run python scripts/test_embeddings.py
    uv run python scripts/test_embeddings.py --n 5

Pipeline testé :
    Parse PDF → items → embed_text → vecteurs 384D → matrice de similarité

But : valider que embed_text() tourne, mesurer le temps moyen et visualiser
la similarité cosinus entre paires d'anomalies (matrice carrée). On s'attend
à ce que les anomalies sémantiquement proches aient un score > 0.7-0.8.

⚠️ Le 1er appel déclenche le téléchargement du modèle E5 (~120 Mo).
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
    parser.add_argument("--n", type=int, default=3, help="Nombre de demandes à encoder.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    # 1. Parsing
    print(f"Parsing : {args.pdf.name}")
    result = parse_letter(args.pdf)
    demandes = [i for i in result["items"] if i["type"] == "demande"][: args.n]
    print(f"   → {len(demandes)} demandes à encoder\n")

    # 2. Embedding de chaque demande, avec chronométrage
    print(" Embedding (le 1er appel inclut le chargement du modèle, ~30-60s)\n")
    vectors: list[list[float]] = []
    for _, item in enumerate(demandes):
        t0 = time.perf_counter()
        vector = embed_text(item["texte"])
        elapsed_s = time.perf_counter() - t0
        vectors.append(vector)
        print(f"  [{elapsed_s:.1f}s] {item['identifiant']} → vecteur de {len(vector)}D")

    # 3. Matrice de similarité cosinus
    print("\n Matrice de similarité cosinus :")
    print(f"   {'':10}", *[f"{d['identifiant']:>10}" for d in demandes])
    sim_matrix = cos_sim(vectors, vectors)
    for i, item in enumerate(demandes):
        scores = [f"{sim_matrix[i][j].item():>10.3f}" for j in range(len(demandes))]
        print(f"   {item['identifiant']:<10}", *scores)

    # 4. Interprétation visuelle
    print("\n Interprétation :")
    print("   • Diagonale = 1.000 (chaque demande avec elle-même)")
    print("   • Hors diagonale : > 0.85 = très proches, 0.6-0.85 = proches, < 0.5 = différents")


if __name__ == "__main__":
    main()
