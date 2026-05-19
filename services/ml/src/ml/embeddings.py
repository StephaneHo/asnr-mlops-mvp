"""Embeddings sémantiques des anomalies via sentence-transformers (E5 multilingue).

Pipeline : texte d'une anomalie → vecteur 384D normalisé prêt pour pgvector.

Choix du modèle : `intfloat/multilingual-e5-small`
- 384 dimensions (compact, indexable rapidement)
- Multilingue (français OK)
- Optimisé pour la recherche sémantique (asymmetric retrieval) — battu MiniLM
  sur les benchmarks BEIR / MTEB
- ~120 Mo, rapide sur CPU

⚠️ Subtilité E5 : les documents indexés ("passages") doivent être préfixés
par "passage: ". Les requêtes par "query: ". Sans ce préfixe, les performances
chutent de ~10-15%. Au stade actuel on n'a que des passages.
"""

from __future__ import annotations

from functools import lru_cache

from ml.extraction import AnomalieEnrichie
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

MODEL_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Charge le modèle E5 une seule fois (cache mémoïsé par @lru_cache).

    Le premier appel télécharge le modèle (~120 Mo) depuis HuggingFace si
    absent du cache local, puis le charge en RAM. Les appels suivants
    retournent l'instance déjà chargée.
    """
    return SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------------------------
# Fonctions d'embedding
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float]:
    """Encode un texte en vecteur 384D normalisé.

    Args:
        text: texte brut à encoder.

    Returns:
        Liste de 384 floats. Vecteur normalisé (norme L2 = 1), donc le produit
        scalaire avec un autre vecteur normalisé donne directement la similarité
        cosinus (entre -1 et 1, typiquement 0-1 sur des textes en français).
    """
    embedder = get_embedder()

    # E5 attend un préfixe "passage: " pour les documents indexés.
    prefixed = "passage: " + text
    vector = embedder.encode(prefixed, normalize_embeddings=True)
    return vector.tolist()
    #    (encode() renvoie un numpy.ndarray, on veut une list pour pouvoir
    #     sérialiser en JSON et stocker en pgvector via Pydantic.)


def embed_anomalie(anomalie: AnomalieEnrichie) -> list[float]:
    """Encode une AnomalieEnrichie en vecteur 384D.

    Au stade MVP, on encode UNIQUEMENT le `texte_source` (le texte original
    de la demande), qui porte l'essentiel de la sémantique. Plus tard on
    pourra essayer un encodage hybride (concat theme + action + equipements).

    Args:
        anomalie: l'AnomalieEnrichie produite par la cascade.

    Returns:
        Vecteur 384D normalisé.
    """
    return embed_text(anomalie["texte_source"])
