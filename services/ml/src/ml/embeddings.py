"""Embeddings semantiques via sentence-transformers (E5 multilingue, 384D)."""

from __future__ import annotations

from functools import lru_cache

from ml.extraction import AnomalieEnrichie
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Charge le modele E5 une seule fois (cache @lru_cache)."""
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """Encode un texte en vecteur 384D normalise (norme L2 = 1).

    Le prefixe 'passage: ' est exige par E5 pour les documents indexes
    (asymmetric retrieval). Pour une query utilisateur, utiliser 'query: '.
    """
    embedder = get_embedder()
    vector = embedder.encode("passage: " + text, normalize_embeddings=True)
    return vector.tolist()


def embed_anomalie(anomalie: AnomalieEnrichie) -> list[float]:
    """Encode le texte source d'une AnomalieEnrichie."""
    return embed_text(anomalie.texte_source)
