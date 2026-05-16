import re
import sys
from pathlib import Path


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


PATTERN_SYNTHESE = r"SYNTHESE DE L[’']INSPECTION"
PATTERN_SECTION_I = r"I\.\s+DEMANDES A TRAITER PRIORITAIREMENT"
PATTERN_SECTION_II = r"II\.\s+AUTRES DEMANDES"
PATTERN_SECTION_III = r"III\.\s+CONSTATS OU OBSERVATIONS N[’']APPELANT PAS DE REPONSE A L[’']ASNR"


def split_sections(cleaned_text: str) -> dict[str, str]:
    """Découpe le texte nettoyé en 5 sections logiques.

    Returns:
        dict avec les clés "header", "synthese", "section_I",
        "section_II", "section_III". Une section absente est une chaîne vide.
    """

    m_synthese = re.search(PATTERN_SYNTHESE, cleaned_text)
    m_I = re.search(PATTERN_SECTION_I, cleaned_text)
    m_II = re.search(PATTERN_SECTION_II, cleaned_text)
    m_III = re.search(PATTERN_SECTION_III, cleaned_text)

    # u travailles sur du contenu pur, donc on utilise .end() pour la borne gauche
    header = cleaned_text[: m_synthese.start()].strip()
    synthese = cleaned_text[m_synthese.end() : m_I.start()].strip()

    section1 = cleaned_text[m_I.end() : m_II.start()].strip()
    section2 = cleaned_text[m_II.end() : m_III.start()].strip()
    section3 = cleaned_text[m_III.end() :].strip()

    return {
        "header": header,
        "synthese": synthese,
        "section_I": section1,
        "section_II": section2,
        "section_III": section3,
    }


PATTERN_REFERENCE_COURRIER = r"Référence courrier\s*:\s*(\S+)"
PATTERN_OBJET = r"Objet\s*:\s*(.+?)\nLettre de suite"
PATTERN_N_DOSSIER = r"Inspection n°\s*(\S+)"
PATTERN_DATE_LETTRE = r"À\s+\w+\s*,\s*le\s+(.+)"
PATTERN_REFERENCES = r"\[\d+\]\s*-\s*[^\n]+"


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


if __name__ == "__main__":
    # Force UTF-8 sur stdout (sinon la redirection '>' sous PowerShell corrompt les accents).
    sys.stdout.reconfigure(encoding="utf-8")
    raw = Path("sample_output.txt").read_text(encoding="utf-8")

    cleaned = clean_text(raw)
    sections = split_sections(cleaned)
    metadata = extract_header(sections["header"])
    print("\n=== Métadonnées extraites ===")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    for name, content in sections.items():
        print(f"\n{'=' * 20} {name.upper()} ({len(content)} chars) {'=' * 20}")
        print(content[:300] + ("..." if len(content) > 300 else ""))
