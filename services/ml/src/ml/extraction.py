"""Extraction structuree d'anomalies via Instructor + Ollama, et cascade NLI + LLM."""

from __future__ import annotations

from typing import Literal

import instructor
from ml.classification import classify_theme
from openai import OpenAI
from pydantic import BaseModel, Field

# Themes ajustables selon le corpus (sous-titres recurrents de la section II)
ThemeAnomalie = Literal[
    "sûreté",
    "radioprotection",
    "transport interne",
    "maintenance",
    "préparation des colis",
    "organisation",
    "contrôle technique",
    "réglementation",
    "environnement",
    "autre",
]


class Anomalie(BaseModel):
    """Schema attendu en sortie du LLM (1 demande ou observation extraite)."""

    theme: ThemeAnomalie = Field(
        ...,
        description="Theme principal de l'anomalie.",
        examples=["sûreté", "radioprotection", "transport interne"],
    )
    action_attendue: str = Field(
        ...,
        description="Verbe d'action a l'infinitif + complement.",
        examples=[
            "Vérifier le couple de serrage des vis",
            "Compléter le plan qualité",
            "Transmettre la justification",
        ],
    )
    equipements_concernes: list[str] = Field(
        default_factory=list,
        description="Equipements / processus / documents cites. Liste vide si rien.",
        examples=[["bouchon biologique", "coques béton C1 et C4", "vis"]],
    )
    delai_mentionne: str | None = Field(
        None,
        description="Delai explicite dans la demande. None si absent.",
        examples=["sous deux mois", None],
    )


class AnomalieEnrichie(BaseModel):
    """Resultat de la cascade NLI + LLM, pret pour stockage.

    `theme_nli` (mDeBERTa) et `theme_llm` (Phi-3) sont conserves separement :
    leur divergence est un signal de doute (items a reviewer en priorite).
    """

    identifiant: str = Field(
        ...,
        description="ID dans la lettre.",
        examples=["II.4.b", "n°1"],
    )
    criticite: str = Field(
        ...,
        description="Niveau hérité de la section : I/II/III.",
        examples=["haute", "normale", "faible"],
    )
    texte_source: str = Field(
        ...,
        description="Texte original de la demande.",
        examples=["Vérifier que le verrouillage du bouchon biologique..."],
    )

    theme_nli: str = Field(
        ...,
        description="Theme predit par NLI mDeBERTa (libellé descriptif).",
        examples=[
            "radioprotection et irradiation",
            "transport interne de substances radioactives",
            "autre",
        ],
    )
    theme_nli_score: float = Field(
        ...,
        description="Confiance NLI (0-1). Fallback 'autre' si < 0.5.",
        ge=0.0,
        le=1.0,
        examples=[0.746, 0.41],
    )
    theme_llm: str = Field(
        ...,
        description="Theme predit par le LLM Phi-3 (libellé court).",
        examples=["sûreté", "radioprotection", "transport interne"],
    )

    action_attendue: str = Field(
        ...,
        description="Verbe d'action a l'infinitif + complement.",
        examples=["Vérifier le couple de serrage des vis", "Compléter le plan qualité"],
    )
    equipements_concernes: list[str] = Field(
        default_factory=list,
        examples=[["bouchon biologique", "coques béton C1 et C4", "vis"]],
    )
    delai_mentionne: str | None = Field(
        None,
        examples=["sous deux mois", None],
    )


def setup_llm_client() -> instructor.Instructor:
    """Renvoie un client Instructor pointant sur Ollama local (API OpenAI-compat)."""
    return instructor.from_openai(
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
        mode=instructor.Mode.JSON,
    )


SYSTEM_PROMPT = """\
Tu es un assistant d'analyse de lettres d'inspection nucléaire française.
À partir du texte d'une demande émise par l'ASNR, extrais les 4 champs au
format JSON exact.

RÈGLES STRICTES :
- Produis une INSTANCE de données (pas un schéma JSON : pas de clés
  "properties", "type", "description" ou "enum" dans ta réponse).
- `theme` : exactement une des 10 valeurs autorisées par le schéma.
- `action_attendue` : un verbe à l'infinitif suivi du complément, max 15 mots.
- `equipements_concernes` : une LISTE de strings, même si un seul élément. [] si rien.
- `delai_mentionne` : null si aucun délai explicite mentionné dans la demande.

EXEMPLE D'INPUT :
Vérifier que le verrouillage du bouchon biologique des coques béton C1 et C4
est bien assuré par des vis serrées au couple indiqué. Assurer dans le temps
la bonne mise en œuvre du couple de serrage.

EXEMPLE D'OUTPUT (réponse attendue exactement de cette forme) :
{
  "theme": "sûreté",
  "action_attendue": "Vérifier le couple de serrage des vis de verrouillage",
  "equipements_concernes": ["bouchon biologique", "coques béton C1 et C4", "vis"],
  "delai_mentionne": null
}
"""


def extract_anomalie(
    demande_text: str,
    client: instructor.Instructor,
    model: str = "phi3:mini",
) -> Anomalie:
    """Appelle le LLM via Instructor pour produire une Anomalie typee."""
    return client.chat.completions.create(
        model=model,
        response_model=Anomalie,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": demande_text},
        ],
        max_retries=0,  # un retry coute ~80s sur Phi-3 CPU, on prefere echouer vite
    )


def extract_anomalie_cascade(
    item: dict,
    client: instructor.Instructor,
    model: str = "phi3:mini",
) -> AnomalieEnrichie:
    """Cascade NLI + LLM : item du parser -> AnomalieEnrichie."""
    theme_nli, theme_nli_score = classify_theme(item["texte"])

    # Tronque a 1000 chars : limite le cout CPU et evite que Phi-3 renvoie
    # `action_attendue` sous forme de list[str] sur les demandes longues.
    anomalie_llm = extract_anomalie(item["texte"][:1000], client=client, model=model)

    return AnomalieEnrichie(
        identifiant=item["identifiant"],
        criticite=item["criticite"],
        texte_source=item["texte"],
        theme_nli=theme_nli,
        theme_nli_score=theme_nli_score,
        theme_llm=anomalie_llm.theme,
        action_attendue=anomalie_llm.action_attendue,
        equipements_concernes=anomalie_llm.equipements_concernes,
        delai_mentionne=anomalie_llm.delai_mentionne,
    )
