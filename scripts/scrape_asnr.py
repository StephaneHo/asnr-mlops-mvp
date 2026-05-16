"""Scraper des lettres d'inspection ASNR (installations nucléaires).

Télécharge les PDFs des N pages les plus récentes du listing officiel
(reglementation-controle.asnr.fr) dans data/raw/asnr/, et sauvegarde
les métadonnées extraites du listing (date, site, exploitant, thème,
référence) dans data/raw/asnr/index.json.

Respecte le Crawl-delay de 10s déclaré dans robots.txt entre chaque
requête HTTP.

Usage:
    uv run python scripts/scrape_asnr.py --max-pages 7
    uv run python scripts/scrape_asnr.py --max-pages 1   # test rapide (~30 PDFs)

Le scraper est idempotent : si un PDF est déjà téléchargé, il est skippé.
On peut donc relancer en cas d'interruption sans tout retélécharger.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = "https://reglementation-controle.asnr.fr"
LISTING_URL = (
    f"{BASE_URL}/controle/actualites-du-controle/installations-nucleaires"
    "/lettres-de-suite-d-inspection-des-installations-nucleaires"
)
USER_AGENT = "asnr-mvp-scraper/0.1 (educational; +https://github.com/StephaneHo/asnr-mlops-mvp)"
CRAWL_DELAY_SECONDS = 10  # cf. robots.txt: "Crawl-delay: 10"

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "raw" / "asnr"
INDEX_FILE = PDF_DIR / "index.json"

DATE_RE = re.compile(r"Inspection du (\d{2}/\d{2}/\d{4})")
REF_RE = re.compile(r"(INSSN-[A-Z]{3}-\d{4}-\d{4})")


@dataclass
class LetterEntry:
    """Une entrée du listing ASNR avec ses métadonnées extraites du HTML."""

    reference: str  # ex: INSSN-CAE-2026-0206
    pdf_url: str  # URL absolue du PDF
    date_inspection: str | None  # format DD/MM/YYYY
    site: str | None  # ex: "Centrale nucléaire de Penly"
    exploitant: str | None  # ex: "Réacteurs de 1300 MWe - EDF"
    theme: str | None  # ex: "Transports internes de substances radioactives"
    pdf_path: str | None = None  # rempli après téléchargement
    downloaded: bool = False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
def _http_get(client: httpx.Client, url: str) -> httpx.Response:
    """GET avec retry/backoff exponentiel sur les erreurs réseau ou 5xx."""
    resp = client.get(url, timeout=30)
    resp.raise_for_status()
    return resp


def _parse_listing(html: str) -> list[LetterEntry]:
    """Extrait toutes les entrées du listing à partir du HTML d'une page.

    On repère chaque entrée via son lien PDF (`/content/download/.../INSSN-*.pdf`)
    et on remonte au bloc grand-parent pour récupérer les métadonnées
    (date, site, exploitant, thème) qui apparaissent côte à côte.
    """
    soup = BeautifulSoup(html, "html.parser")
    entries: list[LetterEntry] = []

    pdf_links = soup.find_all(
        "a", href=lambda h: h and "/content/download/" in h and h.endswith(".pdf")
    )

    for link in pdf_links:
        href = link["href"]
        pdf_url = BASE_URL + href if not href.startswith("http") else href

        # Le grand-parent regroupe les 5 champs (date/site/type/thème/référence).
        block = link.find_parent().find_parent() if link.find_parent() else None
        block_text = block.get_text(separator="|", strip=True) if block else ""

        # Référence depuis le nom du fichier (le plus fiable).
        ref_match = REF_RE.search(href)
        if not ref_match:
            continue  # lien étrange, on saute
        reference = ref_match.group(1)

        # Date d'inspection.
        date_match = DATE_RE.search(block_text)
        date_inspection = date_match.group(1) if date_match else None

        # Décompose le bloc texte en parts pour récupérer site / exploitant / thème.
        # Format attendu : "Inspection du DD/MM/YYYY|<site>|<exploitant>|<theme>|<filename>.pdf|(PDF - ...)"
        parts = [p for p in block_text.split("|") if p.strip()]
        site = parts[1] if len(parts) >= 2 else None
        exploitant = parts[2] if len(parts) >= 3 else None
        theme = parts[3] if len(parts) >= 4 else None

        entries.append(
            LetterEntry(
                reference=reference,
                pdf_url=pdf_url,
                date_inspection=date_inspection,
                site=site,
                exploitant=exploitant,
                theme=theme,
            )
        )

    return entries


def _download_pdf(client: httpx.Client, entry: LetterEntry, dest_dir: Path) -> bool:
    """Télécharge le PDF d'une entrée. Skip s'il existe déjà.

    Retourne True si le fichier a été téléchargé (ou déjà présent), False sur erreur.
    """
    dest = dest_dir / f"{entry.reference}.pdf"
    entry.pdf_path = str(dest.relative_to(ROOT))
    if dest.exists():
        entry.downloaded = True
        print(f"  • {entry.reference} déjà présent, skip")
        return True

    try:
        resp = _http_get(client, entry.pdf_url)
        dest.write_bytes(resp.content)
        entry.downloaded = True
        print(f"  ✓ {entry.reference} ({len(resp.content) // 1024} Ko)")
        return True
    except Exception as exc:
        print(f"  ✗ {entry.reference} : {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--max-pages",
        type=int,
        default=7,
        help="Nombre de pages du listing à scrapper (30 lettres/page). Défaut: 7 (~210 lettres).",
    )
    parser.add_argument(
        "--crawl-delay",
        type=float,
        default=CRAWL_DELAY_SECONDS,
        help=f"Délai en secondes entre requêtes HTTP (défaut: {CRAWL_DELAY_SECONDS}s, cf. robots.txt).",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # On utilise un dict indexé par référence : si plusieurs liens HTML pointent
    # vers le même PDF (icône PDF + titre cliquable, par exemple), on ne garde
    # qu'une seule entrée.
    entries_by_ref: dict[str, LetterEntry] = {}
    headers = {"User-Agent": USER_AGENT}

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        # Étape 1 : récupérer toutes les entrées des N premières pages du listing.
        for page_num in range(1, args.max_pages + 1):
            url = f"{LISTING_URL}?page={page_num}"
            print(f"\n[Listing] page {page_num}/{args.max_pages} — {url}")
            try:
                resp = _http_get(client, url)
            except Exception as exc:
                print(f"  ✗ échec page {page_num}: {exc}")
                continue
            page_entries = _parse_listing(resp.text)
            new_count = 0
            for entry in page_entries:
                if entry.reference not in entries_by_ref:
                    entries_by_ref[entry.reference] = entry
                    new_count += 1
            print(f"  → {len(page_entries)} liens HTML, {new_count} entrées uniques nouvelles")
            time.sleep(args.crawl_delay)

        all_entries = list(entries_by_ref.values())

        # Étape 2 : télécharger chaque PDF (skip si déjà téléchargé).
        print(f"\n[Téléchargement] {len(all_entries)} PDFs uniques à traiter")
        for entry in all_entries:
            _download_pdf(client, entry, PDF_DIR)
            time.sleep(args.crawl_delay)

    # Étape 3 : sauver l'index JSON (toutes les entrées, même celles en échec).
    INDEX_FILE.write_text(
        json.dumps([asdict(e) for e in all_entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_downloaded = sum(1 for e in all_entries if e.downloaded)
    print("\n=== Bilan ===")
    print(f"  Entrées listées      : {len(all_entries)}")
    print(f"  PDFs téléchargés OK  : {n_downloaded}")
    print(f"  Index sauvegardé     : {INDEX_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
