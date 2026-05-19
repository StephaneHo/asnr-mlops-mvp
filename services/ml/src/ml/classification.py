"""Classification zero-shot du theme d'une demande via NLI mDeBERTa multilingue."""

from __future__ import annotations

from functools import lru_cache

from transformers import pipeline

# Labels descriptifs en francais : NLI distingue mieux que des labels mono-mot.
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

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


@lru_cache(maxsize=1)
def get_classifier():
    """Charge le pipeline HuggingFace zero-shot une seule fois (cache @lru_cache)."""
    return pipeline("zero-shot-classification", model=MODEL_NAME, device=-1)


def classify_theme(
    text: str,
    threshold: float = 0.5,
    candidate_labels: list[str] | None = None,
) -> tuple[str, float]:
    """Renvoie (theme, score). Fallback 'autre' si score < threshold."""
    # Tronque a 1000 chars (~256 tokens) : limite cout CPU, le theme est dans le debut.
    text = text[:1000]
    labels = candidate_labels or THEMES
    classifier = get_classifier()
    result = classifier(
        text,
        candidate_labels=labels,
        multi_label=False,
        hypothesis_template="Ce texte parle de {}.",
    )
    top_label = result["labels"][0]
    top_score = float(result["scores"][0])
    if top_score < threshold:
        return ("autre", top_score)
    return (top_label, top_score)
