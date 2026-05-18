"""Classification zero-shot via NLI (mDeBERTa) pour les anomalies ASNR.

Utilise un modèle NLI pré-entraîné (mDeBERTa-v3-base-mnli-xnli) pour classifier
le thème d'une demande sans aucune donnée annotée. Beaucoup plus rapide que
le LLM Phi-3/Mistral pour cette tâche (~100ms vs ~70s), avec une qualité
souvent comparable ou supérieure.

Le modèle est téléchargé automatiquement au premier appel (~280 Mo) puis
mis en cache localement par HuggingFace dans ~/.cache/huggingface/.

Architecture : NLI sert pour les champs avec un *espace fermé* de valeurs
(theme, criticité…). Le LLM reste utilisé pour les champs ouverts
(action_attendue en texte libre, equipements_concernes en liste de spans
extraits du texte).
"""

from __future__ import annotations

from functools import lru_cache

from transformers import pipeline

# ---------------------------------------------------------------------------
# Catégories candidates
# ---------------------------------------------------------------------------

# 👉 TODO : ajuster ces labels au fil du temps selon ce que tu vois en
#    pratique sur le corpus. Garde-les en français descriptif (plutôt que
#    juste "transport") pour aider le modèle NLI à mieux distinguer.
THEMES: list[str] = [
    "sûreté nucléaire",
    "radioprotection et irradiation",
    "transport interne de substances radioactives",
    "maintenance des équipements et engins",
    "préparation et confinement des colis",
    "organisation et procédures internes",
    "contrôle technique et inspection",
    "réglementation et conformité",
    "environnement et déchets",
    "autre",
]


# ---------------------------------------------------------------------------
# Modèle NLI multilingue, déjà fine-tuné sur MNLI + XNLI
# ---------------------------------------------------------------------------

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


@lru_cache(maxsize=1)
def get_classifier():
    """Charge le pipeline HuggingFace 'zero-shot-classification'.

    Mis en cache via @lru_cache pour ne charger le modèle qu'**une seule fois**
    pendant la vie du process (charger un modèle de 280 Mo prend ~5-10s sur CPU).

    Le `device=-1` force le calcul sur CPU. Si on a un GPU plus tard, on peut
    paramétrer ça.
    """
    return pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=-1,  # CPU
    )


def classify_theme(
    text: str,
    threshold: float = 0.5,
    candidate_labels: list[str] | None = None,
) -> tuple[str, float]:
    """Classifie un texte de demande dans un thème prédéfini via NLI zero-shot.

    Args:
        text: Texte de la demande à classifier.
        threshold: Seuil de confiance en-dessous duquel on retourne "autre".
        candidate_labels: Liste de catégories candidates (défaut: THEMES).

    Returns:
        Tuple (theme, score) où theme est l'une des candidate_labels (ou "autre"
        si la confiance est < threshold) et score est la probabilité (0-1).
    """
    # Tronquer à ~1000 caractères (≈ 256 tokens en français) pour limiter
    # le coût d'inférence sur CPU. Le thème d'une demande ASNR est presque
    # toujours déterminé par ses 1-2 premiers paragraphes.
    text = text[:1000]
    labels = candidate_labels or THEMES
    classifier = get_classifier()
    result = classifier(
        text,
        candidate_labels=labels,
        multi_label=False,  # une seule catégorie attendue (sinon True pour multi)
        # Template d'hypothèse en français pour que le modèle raisonne en français.
        # NLI compare la prémisse (`text`) à chaque hypothèse "Ce texte parle de {label}".
        hypothesis_template="Ce texte parle de {}.",
    )
    top_label = result["labels"][0]
    top_score = float(result["scores"][0])
    if top_score < threshold:
        return ("autre", top_score)
    return (top_label, top_score)
