"""Valide parse_letter() sur tous les PDFs présents dans data/raw/asnr/.

Usage:
    uv run python scripts/validate_parser.py

Affiche un récapitulatif par lettre (métadonnées + comptage d'items) et signale
les anomalies (champs None, exceptions levées). Utile pour vérifier que le
parser tient sur des lettres d'exploitants/sites/dates variés avant de l'utiliser
sur les 200 lettres à scraper.
"""

import sys
from collections import Counter
from pathlib import Path

# Hack sys.path pour pouvoir importer le module ml/parsing sans installer le projet.
# Quand on créera un vrai package installable, on supprimera ces 2 lignes.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.parsing import TextExtractionError, parse_letter  # noqa: E402

PDF_DIR = ROOT / "data" / "raw" / "asnr"


def _truncate(text: str | None, n: int = 80) -> str:
    if text is None:
        return "None"
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"Aucun PDF trouvé dans {PDF_DIR}")
        return

    print(f"Validation du parser sur {len(pdf_paths)} lettre(s)\n")

    n_ok = 0
    n_to_ocr = 0
    n_errors = 0
    n_missing_fields = 0

    for pdf_path in pdf_paths:
        print(f"{'=' * 78}")
        print(f"📄 {pdf_path.name}")
        print(f"{'=' * 78}")

        try:
            result = parse_letter(pdf_path)
        except TextExtractionError as exc:
            print(f"  ⚠️  À ROUTER VERS OCR : {exc}")
            n_to_ocr += 1
            continue
        except Exception as exc:
            print(f"  ❌ ERREUR : {type(exc).__name__}: {exc}")
            n_errors += 1
            continue

        meta = result["metadata"]
        items = result["items"]

        print(f"  Référence courrier : {_truncate(meta['reference_courrier'])}")
        print(f"  N° dossier         : {_truncate(meta['n_dossier'])}")
        print(f"  Date               : {_truncate(meta['date_lettre'])}")
        print(f"  Objet              : {_truncate(meta['objet'])}")
        print(f"  Références         : {len(meta['references'])} entrée(s)")
        print(f"  Synthèse           : {len(result['synthese'])} caractères")
        print(f"  Items extraits     : {len(items)}")

        if items:
            by_criticite = Counter(item["criticite"] for item in items)
            print(f"  Par criticité      : {dict(by_criticite)}")
        else:
            print("Aucun item extrait")

        missing = [k for k, v in meta.items() if v is None or (isinstance(v, list) and not v)]
        if missing:
            print(f"Champs manquants : {missing}")
            n_missing_fields += 1

        n_ok += 1
        print()

    print(f"{'=' * 78}")
    print("RÉCAPITULATIF")
    print(f"{'=' * 78}")
    print(f"  Lettres parsées avec succès  : {n_ok}/{len(pdf_paths)}")
    print(f"  Lettres à router vers OCR    : {n_to_ocr}")
    print(f"  Lettres en erreur            : {n_errors}")
    print(f"  Lettres avec champs absents  : {n_missing_fields}")


if __name__ == "__main__":
    main()
