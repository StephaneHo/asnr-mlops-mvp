import re
import sys


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


if __name__ == "__main__":
    from pathlib import Path

    # Force UTF-8 sur stdout (sinon la redirection '>' sous PowerShell corrompt les accents).
    sys.stdout.reconfigure(encoding="utf-8")
    raw = Path("sample_output.txt").read_text(encoding="utf-8")
    print(clean_text(raw))
