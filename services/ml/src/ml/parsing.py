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


def split_sections(cleaned_text: str) -> dict[str, str]:
    """Découpe le texte nettoyé en 5 sections logiques.

    Returns:
        dict avec les clés "header", "synthese", "section_I",
        "section_II", "section_III". Une section absente est une chaîne vide.
    """

    PATTERN_SYNTHESE = r"SYNTHESE DE L[’']INSPECTION"
    PATTERN_SECTION_I = r"I\.\s+DEMANDES A TRAITER PRIORITAIREMENT"
    PATTERN_SECTION_II = r"II\.\s+AUTRES DEMANDES"
    PATTERN_SECTION_III = (
        r"III\.\s+CONSTATS OU OBSERVATIONS N[’']APPELANT PAS DE REPONSE A L[’']ASNR"
    )

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


if __name__ == "__main__":
    # Force UTF-8 sur stdout (sinon la redirection '>' sous PowerShell corrompt les accents).
    sys.stdout.reconfigure(encoding="utf-8")
    raw = Path("sample_output.txt").read_text(encoding="utf-8")

    cleaned = clean_text(raw)
    sections = split_sections(cleaned)

    for name, content in sections.items():
        print(f"\n{'=' * 20} {name.upper()} ({len(content)} chars) {'=' * 20}")
        print(content[:300] + ("..." if len(content) > 300 else ""))
