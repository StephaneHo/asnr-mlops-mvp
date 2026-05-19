"""Scraper des lettres d'inspection ASNR (installations nucleaires).

Telecharge les PDFs des N pages les plus recentes du listing officiel dans
data/raw/asnr/, et sauvegarde les metadonnees dans data/raw/asnr/index.json.
Respecte le Crawl-delay de 10s declare dans robots.txt.

Usage:
    uv run python scripts/scrape_asnr.py --max-pages 7
    uv run python scripts/scrape_asnr.py --max-pages 1   # test rapide (~30 PDFs)

Idempotent : un PDF deja present est skippe sans nouvel appel HTTP.
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
CRAWL_DELAY_SECONDS = 10  # cf. robots.txt

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "raw" / "asnr"
INDEX_FILE = PDF_DIR / "index.json"

DATE_RE = re.compile(r"Inspection du (\d{2}/\d{2}/\d{4})")
REF_RE = re.compile(r"(INSSN-[A-Z]{3}-\d{4}-\d{4})")


@dataclass
class LetterEntry:
    reference: str
    pdf_url: str
    date_inspection: str | None
    site: str | None
    exploitant: str | None
    theme: str | None
    pdf_path: str | None = None
    downloaded: bool = False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
def _http_get(client: httpx.Client, url: str) -> httpx.Response:
    """GET avec retry/backoff exponentiel sur erreurs reseau ou 5xx."""
    resp = client.get(url, timeout=30)
    resp.raise_for_status()
    return resp


def _parse_listing(html: str) -> list[LetterEntry]:
    """Extrait les entrees du listing depuis le HTML d'une page."""
    soup = BeautifulSoup(html, "html.parser")
    entries: list[LetterEntry] = []

    pdf_links = soup.find_all(
        "a", href=lambda h: h and "/content/download/" in h and h.endswith(".pdf")
    )

    for link in pdf_links:
        href = link["href"]
        pdf_url = BASE_URL + href if not href.startswith("http") else href

        # Le grand-parent regroupe les 5 champs (date/site/type/theme/reference)
        block = link.find_parent().find_parent() if link.find_parent() else None
        block_text = block.get_text(separator="|", strip=True) if block else ""

        ref_match = REF_RE.search(href)
        if not ref_match:
            continue
        reference = ref_match.group(1)

        date_match = DATE_RE.search(block_text)
        date_inspection = date_match.group(1) if date_match else None

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
    """Telecharge le PDF. Renvoie True si appel HTTP fait, False si fichier deja present."""
    dest = dest_dir / f"{entry.reference}.pdf"
    entry.pdf_path = str(dest.relative_to(ROOT))
    if dest.exists():
        entry.downloaded = True
        print(f"  [skip] {entry.reference} deja present")
        return False

    try:
        resp = _http_get(client, entry.pdf_url)
        dest.write_bytes(resp.content)
        entry.downloaded = True
        print(f"  [OK]   {entry.reference} ({len(resp.content) // 1024} Ko)")
    except Exception as exc:
        print(f"  [FAIL] {entry.reference} : {type(exc).__name__}: {exc}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-pages", type=int, default=7)
    parser.add_argument("--crawl-delay", type=float, default=CRAWL_DELAY_SECONDS)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    entries_by_ref: dict[str, LetterEntry] = {}
    headers = {"User-Agent": USER_AGENT}

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for page_num in range(1, args.max_pages + 1):
            url = f"{LISTING_URL}?page={page_num}"
            print(f"\n[Listing] page {page_num}/{args.max_pages} - {url}")
            try:
                resp = _http_get(client, url)
            except Exception as exc:
                print(f"  [FAIL] page {page_num}: {exc}")
                continue
            page_entries = _parse_listing(resp.text)
            new_count = 0
            for entry in page_entries:
                if entry.reference not in entries_by_ref:
                    entries_by_ref[entry.reference] = entry
                    new_count += 1
            print(f"  {len(page_entries)} liens HTML, {new_count} entrees uniques nouvelles")
            time.sleep(args.crawl_delay)

        all_entries = list(entries_by_ref.values())

        print(f"\n[Telechargement] {len(all_entries)} PDFs uniques a traiter")
        for entry in all_entries:
            made_http_request = _download_pdf(client, entry, PDF_DIR)
            if made_http_request:
                time.sleep(args.crawl_delay)

    INDEX_FILE.write_text(
        json.dumps([asdict(e) for e in all_entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_downloaded = sum(1 for e in all_entries if e.downloaded)
    print("\n=== Bilan ===")
    print(f"  Entrees listees      : {len(all_entries)}")
    print(f"  PDFs telecharges OK  : {n_downloaded}")
    print(f"  Index sauvegarde     : {INDEX_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
