"""Explore le contenu texte d'une lettre ASNR via pdfplumber.
Exemple:
    uv run python scripts/explore_pdf.py data/raw/asnr/INSSN-CAE-2026-0206.pdf
"""

import sys
from pathlib import Path

import pdfplumber


def explore(pdf_path: Path) -> None:
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Nombre de pages:  {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            print(f"\n page {i}")
            print(text)


if __name__ == "__main__":
    # Force UTF-8 on stdout — sinon, sous PowerShell Windows, Python écrit en
    # cp1252 et les accents/apostrophes typographiques sont corrompus à la
    # redirection ('>') ou à la lecture par un outil supposant de l'UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    explore(Path(sys.argv[1]))
