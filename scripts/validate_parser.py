"""Valide parse_letter() sur tous les PDFs presents dans data/raw/asnr/.

Usage:
    uv run python scripts/validate_parser.py             # detail + stats
    uv run python scripts/validate_parser.py --summary   # stats seules
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.parsing import TextExtractionError, parse_letter  # noqa: E402

PDF_DIR = ROOT / "data" / "raw" / "asnr"
INDEX_FILE = PDF_DIR / "index.json"
DIVISION_RE = re.compile(r"^INSSN-([A-Z]{3})-")


def _truncate(text: str | None, n: int = 80) -> str:
    if text is None:
        return "None"
    return text if len(text) <= n else text[: n - 1] + "..."


def _load_listing_index() -> dict[str, dict]:
    """Charge index.json (produit par scrape_asnr.py) pour croiser avec l'exploitant."""
    if not INDEX_FILE.exists():
        return {}
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        return {e["reference"]: e for e in data if e.get("reference")}
    except Exception:
        return {}


def main() -> None:
    parser_arg = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser_arg.add_argument("--summary", action="store_true")
    args = parser_arg.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"Aucun PDF trouve dans {PDF_DIR}")
        return

    listing_index = _load_listing_index()

    print(f"Validation du parser sur {len(pdf_paths)} lettre(s)")
    if listing_index:
        print(f"Index listing charge : {len(listing_index)} entrees")
    print()

    n_ok = 0
    n_to_ocr = 0
    n_errors = 0
    n_with_missing_fields = 0

    n_items_total = 0
    items_by_criticite: Counter[str] = Counter()
    missing_by_field: Counter[str] = Counter()
    letters_by_division: Counter[str] = Counter()
    letters_by_exploitant: Counter[str] = Counter()

    for pdf_path in pdf_paths:
        if not args.summary:
            print("=" * 78)
            print(pdf_path.name)
            print("=" * 78)

        try:
            result = parse_letter(pdf_path)
        except TextExtractionError as exc:
            if not args.summary:
                print(f"  [OCR] {exc}")
            n_to_ocr += 1
            continue
        except Exception as exc:
            if not args.summary:
                print(f"  [FAIL] {type(exc).__name__}: {exc}")
            n_errors += 1
            continue

        meta = result["metadata"]
        items = result["items"]

        if not args.summary:
            print(f"  Reference courrier : {_truncate(meta['reference_courrier'])}")
            print(f"  N dossier          : {_truncate(meta['n_dossier'])}")
            print(f"  Date               : {_truncate(meta['date_lettre'])}")
            print(f"  Objet              : {_truncate(meta['objet'])}")
            print(f"  References         : {len(meta['references'])} entree(s)")
            print(f"  Synthese           : {len(result['synthese'])} caracteres")
            print(f"  Items extraits     : {len(items)}")
            if items:
                by_criticite = Counter(item["criticite"] for item in items)
                print(f"  Par criticite      : {dict(by_criticite)}")

        n_items_total += len(items)
        for item in items:
            items_by_criticite[item["criticite"]] += 1

        for field in ["reference_courrier", "objet", "n_dossier", "date_lettre"]:
            if meta.get(field) is None:
                missing_by_field[field] += 1
        if not meta.get("references"):
            missing_by_field["references"] += 1

        ref = meta.get("n_dossier") or pdf_path.stem
        m_div = DIVISION_RE.match(ref)
        if m_div:
            letters_by_division[m_div.group(1)] += 1

        if listing_index:
            listing_entry = listing_index.get(ref)
            if listing_entry:
                exploitant = (listing_entry.get("exploitant") or "?").strip() or "?"
                letters_by_exploitant[exploitant] += 1

        missing = [k for k, v in meta.items() if v is None or (isinstance(v, list) and not v)]
        if missing:
            if not args.summary:
                print(f"  Champs manquants : {missing}")
            n_with_missing_fields += 1

        n_ok += 1
        if not args.summary:
            print()

    print("=" * 78)
    print("RECAPITULATIF")
    print("=" * 78)
    print(f"  Lettres parsees avec succes  : {n_ok}/{len(pdf_paths)}")
    print(f"  Lettres a router vers OCR    : {n_to_ocr}")
    print(f"  Lettres en erreur            : {n_errors}")
    print(f"  Lettres avec champs absents  : {n_with_missing_fields}")

    print()
    print("=" * 78)
    print("STATISTIQUES AGREGEES")
    print("=" * 78)

    avg_items = n_items_total / max(n_ok, 1)
    print("\nVolume :")
    print(f"  {n_ok} lettres parsees")
    print(f"  {n_items_total} items extraits (demandes + observations)")
    print(f"  moyenne {avg_items:.1f} items/lettre")

    print("\nItems par criticite :")
    if n_items_total > 0:
        for crit in ["haute", "normale", "faible"]:
            count = items_by_criticite.get(crit, 0)
            pct = count / n_items_total * 100
            print(f"  {crit:8} : {count:5}  ({pct:5.1f}%)")
    else:
        print("  (aucun item)")

    print("\nCompletude des metadonnees (% lettres avec ce champ rempli) :")
    for field in ["reference_courrier", "n_dossier", "date_lettre", "objet", "references"]:
        missing = missing_by_field.get(field, 0)
        present = n_ok - missing
        pct = present / max(n_ok, 1) * 100
        print(f"  {field:20} : {pct:5.1f}%  ({present}/{n_ok})")

    if letters_by_division:
        print("\nCouverture par division ASNR :")
        for div, count in sorted(letters_by_division.items(), key=lambda x: -x[1]):
            pct = count / max(n_ok, 1) * 100
            print(f"  {div:5} : {count:4} lettres ({pct:4.1f}%)")

    if letters_by_exploitant:
        print("\nCouverture par exploitant (top 10) :")
        top = sorted(letters_by_exploitant.items(), key=lambda x: -x[1])[:10]
        for exp, count in top:
            pct = count / max(n_ok, 1) * 100
            label = exp[:50] + ("..." if len(exp) > 50 else "")
            print(f"  {label:50} : {count:4} ({pct:4.1f}%)")


if __name__ == "__main__":
    main()
