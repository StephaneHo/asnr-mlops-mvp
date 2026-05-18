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
from ml.classification import classify_theme
from openai import OpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Schéma Pydantic
# ---------------------------------------------------------------------------

# Liste enrichie à partir des sous-titres thématiques récurrents observés
# dans la section II des lettres ASNR du corpus. À ajuster selon les retours
# d'extraction (si une catégorie est sur-utilisée → la subdiviser).
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


class AnomalieEnrichie(BaseModel):
    """Résultat de la cascade NLI + LLM pour un item du parser.

    Représentation 'production-ready' : contient tout ce qu'il faut pour
    stocker en base et tracer le pipeline.

    Le `theme_nli` (NLI mDeBERTa) et `theme_llm` (Phi-3 via Instructor) sont
    conservés tous les deux pour permettre de comparer / choisir / fallback
    en aval. NLI a une meilleure diversité de classification, LLM est utile
    quand NLI tombe sur "autre" (score < 0.5).
    """

    # --- Identité et traçabilité ---
    identifiant: str = Field(
        ...,
        description="Identifiant unique de la demande/observation dans la lettre (ex: 'II.4.b', 'n°1').",
    )
    criticite: str = Field(
        ...,
        description="Criticité héritée de la section : 'haute' (I), 'normale' (II), 'faible' (III).",
    )
    texte_source: str = Field(
        ...,
        description="Texte original complet de la demande, pour traçabilité et debug.",
    )

    # --- Classification thème (deux sources gardées séparément) ---
    theme_nli: str = Field(
        ...,
        description="Thème prédit par mDeBERTa zero-shot NLI (descriptif, ex: 'radioprotection et irradiation').",
    )
    theme_nli_score: float = Field(
        ...,
        description="Confiance du modèle NLI sur theme_nli (0-1).",
        ge=0.0,
        le=1.0,
    )
    theme_llm: str = Field(
        ...,
        description="Thème prédit par le LLM Phi-3 (libellé court, ex: 'sûreté').",
    )

    # --- Champs extraits par le LLM ---
    action_attendue: str = Field(
        ...,
        description="Verbe d'action que l'exploitant doit accomplir, à l'infinitif.",
    )
    equipements_concernes: list[str] = Field(
        default_factory=list,
        description="Liste des équipements / processus / documents cités.",
    )
    delai_mentionne: str | None = Field(
        None,
        description="Délai explicite mentionné dans la demande (ex: 'sous deux mois'). None sinon.",
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
        # max_retries=0 : on plante au 1er échec pour mesurer le taux de succès
        # brut de cette config. Un retry coûte 80s+ avec Phi-3 sur CPU.
        max_retries=0,
    )


def extract_anomalie_cascade(
    item: dict,
    client: instructor.Instructor,
    model: str = "phi3:mini",
) -> AnomalieEnrichie:
    """Cascade NLI + LLM : transforme un item du parser en AnomalieEnrichie.

    Pipeline :
      1. NLI (classify_theme)  → theme_nli, theme_nli_score
      2. LLM (extract_anomalie) → action_attendue, equipements, delai, theme_llm
      3. Fusion + héritage de la criticité du parser → AnomalieEnrichie

    Args:
        item: dict renvoyé par parse_letter()["items"][i]. Doit contenir
              les clés "identifiant", "criticite", "texte".
        client: client Instructor configuré (cf. setup_llm_client).
        model: nom du modèle Ollama pour le LLM. Défaut: phi3:mini.

    Returns:
        AnomalieEnrichie prête à être stockée.
    """

    # Étape 1 — NLI pour le thème
    (theme_nli, theme_nli_score) = classify_theme(item["texte"])

    # Étape 2 — LLM pour les champs en texte libre.
    # On tronque à ~1000 chars pour la cohérence avec NLI ET pour réduire le
    # risque de réponse en liste : sur les demandes longues, Phi-3 a tendance
    # à décomposer en plusieurs actions et à produire `action_attendue` sous
    # forme de list[str] au lieu de str (ValidationError).
    texte_tronque = item["texte"][:1000]
    anomalie_llm = extract_anomalie(texte_tronque, client=client, model=model)

    # Étape 3 — Fusion en AnomalieEnrichie
    # TODO 3 : construis un AnomalieEnrichie(...) en passant chaque champ.
    #          Sources :
    #            - identifiant, criticite, texte_source → depuis `item`
    #            - theme_nli, theme_nli_score → depuis les variables de l'étape 1
    #            - theme_llm, action_attendue, equipements_concernes, delai_mentionne
    #              → depuis l'objet `anomalie_llm` (accès par attributs, pas par clés)
    identifiant = item["identifiant"]

    criticite = item["criticite"]

    texte_source = item["texte"]

    action_attendue = anomalie_llm.action_attendue

    equipements_concernes = anomalie_llm.equipements_concernes

    delai_mentionne = anomalie_llm.delai_mentionne

    theme_llm = anomalie_llm.theme

    return AnomalieEnrichie(
        identifiant=identifiant,
        criticite=criticite,
        texte_source=texte_source,
        action_attendue=action_attendue,
        equipements_concernes=equipements_concernes,
        delai_mentionne=delai_mentionne,
        theme_nli=theme_nli,
        theme_nli_score=theme_nli_score,
        theme_llm=theme_llm,
    )
