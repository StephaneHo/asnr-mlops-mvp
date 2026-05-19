"""Parser de lettres d'inspection ASNR : PDF -> dict structure (header + sections + items)."""

import re
import sys
from pathlib import Path

import pdfplumber

# --- split_sections : titres des 4 sections principales ---
# Patterns case-insensitive (avec (?i)) pour accepter MAJUSCULES, "Synthèse" mixed case, etc.
PATTERN_SYNTHESE = r"(?i)synth[èe]se\s+de\s+l[’']inspection"
PATTERN_SECTION_I = r"(?i)I\.\s+demandes\s+[aà]\s+traiter\s+prioritairement"
PATTERN_SECTION_II = r"(?i)II\.\s+autres\s+demandes"
# Le "." entre N et APPELANT tolère apostrophe typo/droite/perdue dans l'encodage PDF
PATTERN_SECTION_III = r"III\.\s+CONSTATS\s+OU\s+OBSERVATIONS\s+N.APPELANT[^\n]*"

# --- extract_header : champs du bloc d'en-tete ---
PATTERN_REFERENCE_COURRIER = r"Référence courrier\s*:\s*(\S+)"
PATTERN_OBJET = r"Objet\s*:\s*(.+?)\nLettre de suite"
# On cherche le motif INSSN-XXX-YYYY-NNNN directement, indépendant de l'étiquette
PATTERN_N_DOSSIER = r"\b(INSSN-[A-Z]{3}-\d{4}-\d{4})\b"
# "À " optionnel ("À Caen, le …" vs "Lyon, le …")
PATTERN_DATE_LETTRE = r"(?:À\s+)?\w+(?:[-\s]\w+)*\s*,\s*le\s+(.+)"
PATTERN_REFERENCES = r"\[\d+\]\s*-?\s*[^\n]+"

# --- extract_items : items dans les sections ---
# `\.?` accepte "II.1." (MRS) ou "II.1" (EDF). `(?=[A-Z])` exige majuscule
# après pour éviter les faux positifs en milieu de phrase.
PATTERN_DEMANDE = r"Demande\s+([IVX]+\.\d+(?:\.[a-z])?)\.?\s*:?\s+(?=[A-Z])"
PATTERN_OBSERVATION = r"Observation\s+n°(\d+)\.?\s*:?\s+(?=[A-Z])"


def clean_text(raw_text: str) -> str:
    """Retire numeros de page, adresses, telephone, marqueurs et notes de bas de page."""
    kept_lines = []
    in_footnote = False
    for line in raw_text.splitlines():
        # Changement de page : reset du flag note
        if line.startswith(" page "):
            in_footnote = False
            kept_lines.append(line)
            continue
        # Debut d'une note (chiffre + maj) : skip ligne + lignes suivantes jusqu'au prochain " page"
        if re.match(r"^[1-9]\s+[A-Z]", line):
            in_footnote = True
            continue
        if in_footnote:
            continue
        if line.startswith("Téléphone"):
            continue
        if line.startswith("Adresse postale"):
            continue
        if line.strip() in {"*", "* *"}:
            continue
        if re.fullmatch(r"\d+/\d+", line.strip()):
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines)


def split_sections(cleaned_text: str) -> dict[str, str]:
    """Decoupe le texte en {header, synthese, section_I, section_II, section_III}.

    Robuste aux marqueurs absents : section non trouvee = chaine vide.
    """
    m_synthese = re.search(PATTERN_SYNTHESE, cleaned_text)
    m_i = re.search(PATTERN_SECTION_I, cleaned_text)
    m_ii = re.search(PATTERN_SECTION_II, cleaned_text)
    m_iii = re.search(PATTERN_SECTION_III, cleaned_text)

    def slice_between(start_match, end_match) -> str:
        if start_match is None:
            return ""
        start = start_match.end()
        end = end_match.start() if end_match is not None else len(cleaned_text)
        return cleaned_text[start:end].strip()

    header = cleaned_text[: m_synthese.start()].strip() if m_synthese else ""
    next_after_synthese = next((m for m in [m_i, m_ii, m_iii] if m is not None), None)
    next_after_i = next((m for m in [m_ii, m_iii] if m is not None), None)

    return {
        "header": header,
        "synthese": slice_between(m_synthese, next_after_synthese),
        "section_I": slice_between(m_i, next_after_i),
        "section_II": slice_between(m_ii, m_iii),
        "section_III": slice_between(m_iii, None),
    }


def extract_header(header_text: str) -> dict[str, str | list[str] | None]:
    """Renvoie {reference_courrier, objet, n_dossier, date_lettre, references}."""
    m = re.search(PATTERN_REFERENCE_COURRIER, header_text)
    reference_courrier = m.group(1).strip() if m else None

    m = re.search(PATTERN_OBJET, header_text, re.DOTALL)
    objet = m.group(1).strip() if m else None

    m = re.search(PATTERN_N_DOSSIER, header_text)
    n_dossier = m.group(1).strip() if m else None

    m = re.search(PATTERN_DATE_LETTRE, header_text)
    date_lettre = m.group(1).strip() if m else None

    references = re.findall(PATTERN_REFERENCES, header_text)

    return {
        "reference_courrier": reference_courrier,
        "objet": objet,
        "n_dossier": n_dossier,
        "date_lettre": date_lettre,
        "references": references,
    }


def _extract_items_from_section(
    section_text: str,
    pattern: str,
    type_item: str,
    criticite: str,
) -> list[dict]:
    """Extrait tous les items d'une section. Texte d'un item = fin de son match -> debut du match suivant."""
    matches = list(re.finditer(pattern, section_text))
    items = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        items.append(
            {
                "identifiant": m.group(1),
                "type": type_item,
                "criticite": criticite,
                "texte": section_text[start:end].strip(),
            }
        )
    return items


def extract_items(sections: dict[str, str]) -> list[dict]:
    """Extrait les demandes I/II et observations III avec criticite heritee de la section."""
    items = []
    items += _extract_items_from_section(sections["section_I"], PATTERN_DEMANDE, "demande", "haute")
    items += _extract_items_from_section(
        sections["section_II"], PATTERN_DEMANDE, "demande", "normale"
    )
    items += _extract_items_from_section(
        sections["section_III"], PATTERN_OBSERVATION, "observation", "faible"
    )
    return items


class TextExtractionError(Exception):
    """Texte PDF extrait trop corrompu (encoding casse). A router vers OCR."""


def _text_is_corrupted(text: str, threshold: float = 0.005) -> bool:
    """True si le ratio de U+FFFD ('�') dans le texte depasse `threshold`."""
    if not text:
        return True
    return text.count("�") / len(text) > threshold


def parse_letter(pdf_path: Path) -> dict:
    """Pipeline complet : PDF -> {metadata, synthese, items}.

    Raises:
        TextExtractionError: si la couche texte du PDF est corrompue (a router vers OCR).
    """
    # On injecte " page X" entre les pages : clean_text() s'en sert pour
    # reset le flag in_footnote a chaque nouvelle page.
    with pdfplumber.open(pdf_path) as pdf:
        parts = []
        for i, page in enumerate(pdf.pages, start=1):
            parts.append(f" page {i}")
            parts.append(page.extract_text() or "")
        raw = "\n".join(parts)

    if _text_is_corrupted(raw):
        n_replacements = raw.count("�")
        raise TextExtractionError(
            f"Texte corrompu ({n_replacements} '�' sur {len(raw)} chars). A router vers OCR."
        )

    cleaned = clean_text(raw)
    sections = split_sections(cleaned)

    return {
        "metadata": extract_header(sections["header"]),
        "synthese": sections["synthese"],
        "items": extract_items(sections),
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pdf_path = Path("data/raw/asnr/INSSN-CAE-2026-0206.pdf")
    result = parse_letter(pdf_path)

    print("=== Metadonnees ===")
    for k, v in result["metadata"].items():
        print(f"  {k}: {v}")

    print(f"\n=== Synthese ({len(result['synthese'])} chars) ===")
    print(result["synthese"][:300] + "...")

    print(f"\n=== {len(result['items'])} items ===")
    for item in result["items"]:
        print(f"  [{item['criticite']:8}] {item['type']} {item['identifiant']}")
        print(f"     -> {item['texte'][:100]}{'...' if len(item['texte']) > 100 else ''}")
