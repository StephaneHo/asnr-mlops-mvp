"""Test de discrimination des embeddings sur des lettres ASNR variees.

Prend la 1ere demande de chaque PDF, encode, et affiche la matrice de
similarite cosinus. Permet de visualiser que les anomalies semantiquement
proches (memes exploitants, memes sujets) ont des vecteurs proches.

Usage:
    uv run python scripts/test_embeddings_multi.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.embeddings import embed_text  # noqa: E402
from ml.parsing import parse_letter  # noqa: E402
from sentence_transformers.util import cos_sim  # noqa: E402

PDF_DIR = ROOT / "data" / "raw" / "asnr"

# 6 lettres variees (exploitants + sujets differents)
LETTERS: list[tuple[Path, str]] = [
    (PDF_DIR / "INSSN-CAE-2026-0206.pdf", "Penly EDF"),
    (PDF_DIR / "INSSN-LYO-2026-0507.pdf", "Orano GBII"),
    (PDF_DIR / "INSSN-LYO-2025-0627 .pdf", "Orano TU5"),  # nom avec espace, voulu
    (PDF_DIR / "INSSN-MRS-2025-0689.pdf", "CEA Rapsodie"),
    (PDF_DIR / "INSSN-MRS-2026-0738.pdf", "CEA Cadarache"),
    (PDF_DIR / "INSSN-MRS-2026-0754.pdf", "ITER"),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Test discrimination sur {len(LETTERS)} lettres differentes\n")

    labels: list[str] = []
    textes_courts: list[str] = []
    vectors: list[list[float]] = []

    # Pour chaque (pdf_path, label) dans LETTERS :
    for pdf_path, label in LETTERS:
        result = parse_letter(pdf_path)
        demandes = [i for i in result["items"] if i["type"] == "demande"]
        if len(demandes) == 0:
            continue
        item = demandes[0]
        vector = embed_text(item["texte"])
        labels.append(label)
        textes_courts.append(item["texte"][:80])
        vectors.append(vector)

    # Affichage des demandes utilisees
    print("Demandes utilisees (1er texte de chaque lettre) :")
    for label, texte in zip(labels, textes_courts, strict=False):
        print(f"  {label:14s} : {texte}...")

    # Matrice de similarite cosinus
    print("\nMatrice de similarite cosinus :")
    header = "                 " + " ".join(f"{lbl:>14}" for lbl in labels)
    print(header)
    sim_matrix = cos_sim(vectors, vectors)
    for i, lbl in enumerate(labels):
        scores = " ".join(f"{sim_matrix[i][j].item():>14.3f}" for j in range(len(labels)))
        print(f"  {lbl:14s} {scores}")

    print("\nLecture : >0.85 tres proche, 0.6-0.85 proche, <0.5 different.")


if __name__ == "__main__":
    main()
