import re
import sys
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------------------
# Patterns regex utilisés par les différentes étapes du parser.
# Regroupés en tête de module pour faciliter la maintenance et la relecture.
# ---------------------------------------------------------------------------

# --- split_sections() : titres des 4 sections principales ------------------
# Patterns tolérants aux variations entre :
#   - ancien style ASN/EDF (sans accents : SYNTHESE, A TRAITER)
#   - nouveau style ASNR (avec accents : SYNTHÈSE, À TRAITER)
# `(?i)` rend les patterns case-insensitive : on accepte "SYNTHÈSE" (Penly),
# "SYNTHESE" (ancien EDF) ET "Synthèse" (lettres MRS en mixed case).
PATTERN_SYNTHESE = r"(?i)synth[èe]se\s+de\s+l[’']inspection"
PATTERN_SECTION_I = r"(?i)I\.\s+demandes\s+[aà]\s+traiter\s+prioritairement"
PATTERN_SECTION_II = r"(?i)II\.\s+autres\s+demandes"
# Pour la section III on raccourcit le pattern à un préfixe stable :
# "III. CONSTATS OU OBSERVATIONS N.APPELANT" suffit à identifier le titre
# sans dépendre de la fin exacte. Le "." tolère l'apostrophe sous toutes ses
# formes : ’ (typographique), ' (droite) ou � (lettre où l'encodage PDF a perdu
# l'apostrophe).
PATTERN_SECTION_III = r"III\.\s+CONSTATS\s+OU\s+OBSERVATIONS\s+N.APPELANT[^\n]*"

# --- extract_header() : champs du bloc d'en-tête ---------------------------
PATTERN_REFERENCE_COURRIER = r"Référence courrier\s*:\s*(\S+)"
PATTERN_OBJET = r"Objet\s*:\s*(.+?)\nLettre de suite"
# N° dossier : on cherche directement le motif INSSN-XXX-YYYY-NNNN, présent
# dans toutes les lettres quelle que soit l'étiquette ("Inspection n°",
# "N° dossier (à rappeler dans toute correspondance) :", etc.).
PATTERN_N_DOSSIER = r"\b(INSSN-[A-Z]{3}-\d{4}-\d{4})\b"
# Date : le préfixe "À " est optionnel (ancien style "À Caen, le …" vs
# nouveau style "Lyon, le …"). On capture la date en sortie.
PATTERN_DATE_LETTRE = r"(?:À\s+)?\w+(?:[-\s]\w+)*\s*,\s*le\s+(.+)"
# Références : le tiret entre [N] et le texte est optionnel.
PATTERN_REFERENCES = r"\[\d+\]\s*-?\s*[^\n]+"

# --- extract_items() : items à l'intérieur des sections --------------------
# Séparateur entre l'identifiant et le texte : ":" ancien style EDF, simple
# espace nouveau style Orano. `\.?` accepte un point final optionnel (MRS écrit
# "II.1." au lieu de "II.1"). Le lookahead `(?=[A-Z])` exige que le texte
# commence par une majuscule (le verbe de la demande), pour éviter les faux
# positifs où "Demande X.N" apparaît dans le corps d'une phrase.
PATTERN_DEMANDE = r"Demande\s+([IVX]+\.\d+(?:\.[a-z])?)\.?\s*:?\s+(?=[A-Z])"
PATTERN_OBSERVATION = r"Observation\s+n°(\d+)\.?\s*:?\s+(?=[A-Z])"


def clean_text(raw_text: str) -> str:
    """Nettoie le texte brut extrait d'une lettre ASNR par pdfplumber.

    Retire :

    - les notes de bas de page (chiffre 1-9 + espace + majuscule)
    - les marqueurs de séparation '*' et '* *'

    Args:
        raw_text: Texte brut tel que retourné par pdfplumber (toutes pages concaténées).

    Returns:
        Texte nettoyé, prêt pour le découpage en sections.
    """
    kept_lines = []
    in_footnote = False
    for line in raw_text.splitlines():
        # Changement de page: on reset le flag
        if line.startswith(" page "):
            in_footnote = False
            kept_lines.append(line)
            continue
        # Début d'une note de bas de page : on active le flag pour skipper aussi
        # les lignes suivantes (continuations) jusqu'au prochain changement de page.
        if re.match(r"^[1-9]\s+[A-Z]", line):
            in_footnote = True
            continue
        # Déjà à l'intérieur d'une note : on skip jusqu'au prochain " page X".
        if in_footnote:
            continue
        if line.startswith("Téléphone"):
            continue
        if line.startswith("Adresse postale"):
            continue
        if line.strip() in {"*", "* *"}:
            continue
        # On skip les numéros de page seul
        if re.fullmatch(r"\d+/\d+", line.strip()):
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines)


def split_sections(cleaned_text: str) -> dict[str, str]:
    """Découpe le texte nettoyé en 5 sections logiques.

    Robuste aux marqueurs absents : si un titre n'est pas trouvé, la section
    correspondante (et celles qui en dépendent comme borne) renvoient "".

    Returns:
        dict avec les clés "header", "synthese", "section_I",
        "section_II", "section_III". Une section absente est une chaîne vide.
    """
    m_synthese = re.search(PATTERN_SYNTHESE, cleaned_text)
    m_i = re.search(PATTERN_SECTION_I, cleaned_text)
    m_ii = re.search(PATTERN_SECTION_II, cleaned_text)
    m_iii = re.search(PATTERN_SECTION_III, cleaned_text)

    def slice_between(start_match, end_match) -> str:
        """Renvoie cleaned_text[start_match.end() : end_match.start()] avec fallback."""
        if start_match is None:
            return ""
        start = start_match.end()
        end = end_match.start() if end_match is not None else len(cleaned_text)
        return cleaned_text[start:end].strip()

    # Header = tout ce qui précède SYNTHESE (ou rien si SYNTHESE n'est pas trouvée).
    header = cleaned_text[: m_synthese.start()].strip() if m_synthese else ""

    # Pour chaque section, le 'next marker' est le premier marqueur trouvé après.
    next_after_synthese = next((m for m in [m_i, m_ii, m_iii] if m is not None), None)
    next_after_i = next((m for m in [m_ii, m_iii] if m is not None), None)

    synthese = slice_between(m_synthese, next_after_synthese)
    section1 = slice_between(m_i, next_after_i)
    section2 = slice_between(m_ii, m_iii)
    section3 = slice_between(m_iii, None)

    return {
        "header": header,
        "synthese": synthese,
        "section_I": section1,
        "section_II": section2,
        "section_III": section3,
    }


def extract_header(header_text: str) -> dict[str, str | list[str] | None]:
    """Extrait les métadonnées du bloc header d'une lettre ASNR.

    Args:
        header_text: Le bloc 'header' renvoyé par split_sections().

    Returns:
        dict avec les clés:
        - 'reference_courrier' (str | None)
        - 'objet' (str | None)
        - 'n_dossier' (str | None)
        - 'date_lettre' (str | None)
        - 'references' (list[str], peut être [])
    """

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
    type_item: str,  # "demande" ou "observation"
    criticite: str,  # "haute", "normale", "faible"
) -> list[dict]:
    """Helper : extrait tous les items d'UNE section."""
    matches = list(re.finditer(pattern, section_text))
    items = []
    for i, m in enumerate(matches):
        # Position de fin du match = début du texte de l'item
        start = m.end()
        # Position de début du match suivant = fin du texte de l'item
        # (ou la fin du texte si c'est le dernier)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)

        texte = section_text[start:end].strip()
        identifiant = m.group(1)

        items.append(
            {
                "identifiant": identifiant,
                "type": type_item,
                "criticite": criticite,
                "texte": texte,
            }
        )
    return items


def extract_items(sections: dict[str, str]) -> list[dict]:
    """Extrait tous les items (demandes + observations) des 3 sections.

    Args:
        sections: dict renvoyé par split_sections().

    Returns:
        liste plate d'items avec leur identifiant, type, criticité, texte.
    """
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
    """Le texte extrait par pdfplumber est trop corrompu pour être parsé.

    Levée quand la couche texte du PDF est cassée (police custom sans
    ToUnicode CMap valide → les accents sortent en U+FFFD '�'). Ces PDFs
    devront être traités par un pipeline OCR séparé (docTR / PaddleOCR).
    """


def _text_is_corrupted(text: str, threshold: float = 0.005) -> bool:
    """Détecte une couche texte PDF corrompue.

    Le critère : ratio de caractères de remplacement Unicode U+FFFD ('�')
    par rapport à la longueur totale du texte. Au-delà du seuil (0.5 % par
    défaut), on considère que l'extraction a échoué et qu'il faut router
    cette lettre vers un pipeline OCR.

    Args:
        text: texte extrait par pdfplumber.
        threshold: ratio de '�' au-dessus duquel on considère le texte cassé.

    Returns:
        True si le texte est corrompu (= à OCR), False sinon.
    """
    if not text:
        return True
    return text.count("�") / len(text) > threshold


def parse_letter(pdf_path: Path) -> dict:
    """Pipeline complet : PDF d'une lettre ASNR → dict structuré.

    Args:
        pdf_path: chemin vers un fichier PDF de lettre d'inspection.

    Returns:
        dict avec :
        - 'metadata' : dict renvoyé par extract_header()
        - 'synthese' : str, contenu de la section SYNTHESE
        - 'items'    : list[dict], demandes + observations

    Raises:
        TextExtractionError: si la couche texte du PDF est corrompue
            (trop de caractères de remplacement '�'). Ces PDFs doivent
            être traités par un pipeline OCR séparé.
    """

    # On injecte un marqueur " page X" entre les pages : clean_text() s'en sert
    # pour reset le flag in_footnote à chaque nouvelle page (sans ce marqueur,
    # une note de bas de page de la page 1 ferait passer toutes les pages
    # suivantes en mode "skip" jusqu'à la fin du document).
    with pdfplumber.open(pdf_path) as pdf:
        parts = []
        for i, page in enumerate(pdf.pages, start=1):
            parts.append(f" page {i}")
            parts.append(page.extract_text() or "")
        raw = "\n".join(parts)

    if _text_is_corrupted(raw):
        n_replacements = raw.count("�")
        raise TextExtractionError(
            f"Texte extrait corrompu ({n_replacements} caractères '�' "
            f"sur {len(raw)} = {n_replacements / max(len(raw), 1):.1%}). "
            "PDF à traiter par pipeline OCR (docTR / PaddleOCR)."
        )

    cleaned = clean_text(raw)
    sections = split_sections(cleaned)

    return {
        "metadata": extract_header(sections["header"]),
        "synthese": sections["synthese"],
        "items": extract_items(sections),
    }


if __name__ == "__main__":
    # Force UTF-8 sur stdout (sinon la redirection '>' sous PowerShell corrompt les accents).
    sys.stdout.reconfigure(encoding="utf-8")

    pdf_path = Path("data/raw/asnr/INSSN-CAE-2026-0206.pdf")
    result = parse_letter(pdf_path)

    print("=== Métadonnées ===")
    for k, v in result["metadata"].items():
        print(f"  {k}: {v}")

    print(f"\n=== Synthèse ({len(result['synthese'])} chars) ===")
    print(result["synthese"][:300] + "…")

    print(f"\n=== {len(result['items'])} items ===")
    for item in result["items"]:
        print(f"  [{item['criticite']:8}] {item['type']} {item['identifiant']}")
        print(f"     → {item['texte'][:100]}{'…' if len(item['texte']) > 100 else ''}")
