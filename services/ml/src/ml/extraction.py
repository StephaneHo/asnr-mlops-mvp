"""Extraction structurée d'anomalies depuis le texte d'une demande d'inspection ASNR.

Pipeline : texte brut de Demande → JSON structuré (Anomalie typée Pydantic).

L'idée centrale est d'utiliser Instructor + Ollama pour forcer le LLM local
(Phi-3 ou Mistral) à répondre selon un schéma Pydantic strict. Si la sortie
du LLM n'est pas valide, Instructor relance automatiquement avec un message
d'erreur jusqu'à obtenir un JSON conforme.

Ollama expose une API OpenAI-compatible sur http://localhost:11434/v1, ce qui
permet de réutiliser le SDK `openai` standard, sans dépendance à un cloud.
"""

from __future__ import annotations

from typing import Literal

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Schéma Pydantic
# ---------------------------------------------------------------------------

# 👉 TODO (toi) : enrichir la liste des thèmes possibles d'après ton corpus.
#    Inspire-toi des sous-titres récurrents de la section II observés dans
#    les lettres ASNR : "Transport interne", "Arrimage", "Maintenance",
#    "Préparation du colis", "Organisation", "Radioprotection", etc.
ThemeAnomalie = Literal[
    "sûreté",
    "radioprotection",
    "transport",
    "maintenance",
    "organisation",
    "autre",
]


class Anomalie(BaseModel):
    """Représentation structurée d'une demande/observation extraite d'une lettre ASNR.

    C'est le schéma que le LLM doit produire à partir du texte brut d'une demande.
    Instructor s'assure que la sortie respecte exactement ces types et contraintes.
    """

    theme: ThemeAnomalie = Field(
        ...,
        description="Thème principal de l'anomalie, parmi la liste prédéfinie",
    )
    action_attendue: str = Field(
        ...,
        description=(
            "Verbe d'action que l'exploitant doit accomplir, à l'infinitif. "
            "Exemples : 'vérifier', 'mettre en place', 'transmettre', 'clarifier', "
            "'compléter', 'justifier'."
        ),
    )
    equipements_concernes: list[str] = Field(
        default_factory=list,
        description="Liste des équipements / processus / documents mentionnés. Liste vide si rien.",
    )
    delai_mentionne: str | None = Field(
        None,
        description="Délai explicitement indiqué dans la demande (ex: 'sous deux mois'). None si absent.",
    )


# ---------------------------------------------------------------------------
# 2. Setup du client Instructor branché sur Ollama
# ---------------------------------------------------------------------------


def setup_llm_client() -> instructor.Instructor:
    """Construit un client Instructor configuré pour parler à Ollama local.

    Ollama expose une API HTTP **compatible OpenAI** sur localhost:11434/v1.
    Cela signifie qu'on peut réutiliser le SDK `openai` officiel, en lui
    pointant simplement vers Ollama au lieu de https://api.openai.com.

    Le `api_key="ollama"` est ignoré par Ollama mais reste obligatoire pour
    que le SDK OpenAI ne plante pas au démarrage.
    """
    return instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        ),
        mode=instructor.Mode.JSON,
    )


# ---------------------------------------------------------------------------
# 3. Fonction d'extraction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Tu es un assistant d'analyse de lettres d'inspection nucléaire française.
À partir du texte d'une demande émise par l'ASNR (Autorité de Sûreté Nucléaire
et de Radioprotection), extrais les champs structurés demandés.
RÈGLES STRICTES :
- Réponds UNIQUEMENT en JSON, sans préambule ni commentaire.
- `action_attendue` : un verbe à l'infinitif suivi du complément, en moins de 15 mots.
- `equipements_concernes` : une LISTE de strings, même si un seul élément.
- `delai_mentionne` : null si aucun délai explicite (sous X mois, avant date, etc.).
"""


def extract_anomalie(
    demande_text: str,
    client: instructor.Instructor,
    model: str = "phi3:mini",
) -> Anomalie:
    """Extrait une Anomalie structurée à partir du texte brut d'une demande ASNR.

    Args:
        demande_text: Texte complet de la demande, tel que renvoyé par
            `extract_items()` (champ `texte`).
        client: Client Instructor configuré (cf. setup_llm_client).
        model: Nom du modèle Ollama. "phi3:mini" est rapide (~2-5s/demande),
            "mistral:7b-instruct-q4_K_M" est plus précis (~15-30s/demande).

    Returns:
        Anomalie typée selon le schéma Pydantic. Si le LLM produit un JSON
        invalide, Instructor relance automatiquement avec un message d'erreur
        (configurable via `max_retries`).
    """
    return client.chat.completions.create(
        model=model,
        response_model=Anomalie,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": demande_text},
        ],
        max_retries=1,
    )
