"""API FastAPI : recherche semantique sur les anomalies ASNR."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Permet d'importer ml.* sans installer le package (cohérent avec les scripts).
# main.py vit dans services/api/src/api/, parents[4] remonte a la racine du repo.
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.embeddings import embed_query  # noqa: E402
from ml.store import get_connection, search_similar  # noqa: E402

# ----------------------------------------------------------------------
# Schemas de reponse
# ----------------------------------------------------------------------


class SearchHit(BaseModel):
    """UN resultat. Forme JSON : {"identifiant": "II.1.a", "distance": 0.117, ...}."""

    identifiant: str = Field(
        ...,
        description="ID dans la lettre.",
        examples=["II.4.b", "n°1"],
    )
    lettre: str = Field(
        ...,
        description="référence.",
        examples=["INSSN-CAE-2026-0206"],
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
    equipements: list[str] = Field(
        default_factory=list,
        description="Equipements / processus / documents cites. Liste vide si rien.",
        examples=[["bouchon biologique", "coques béton C1 et C4", "vis"]],
    )
    delai_mentionne: str | None = Field(
        None,
        description="Delai explicite dans la demande. None si absent.",
        examples=["sous deux mois", None],
    )
    criticite: str = Field(
        ...,
        description="Niveau hérité de la section : I/II/III.",
        examples=["haute", "normale", "faible"],
    )
    extrait: str = Field(
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
    theme_llm: str = Field(
        ...,
        description="Theme predit par le LLM Phi-3 (libellé court).",
        examples=["sûreté", "radioprotection", "transport interne"],
    )
    theme_nli_score: float = Field(
        ...,
        description="Confiance NLI (0-1). Fallback 'autre' si < 0.5.",
        ge=0.0,
        le=1.0,
        examples=[0.746, 0.41],
    )
    distance: float = Field(
        ...,
        description="Distance cosinus",
        examples=[0.1, 0.3],
    )


class SearchResponse(BaseModel):
    """Reponse complete. Forme JSON : {"results": [hit, hit, ...], "tassement": 0.087, "elapsed_ms": 45}."""

    results: list[SearchHit] = Field(
        ...,
        description="Les hits de la requête",
    )
    tassement: float | None = Field(
        ...,
        description="Distance entre top K et top 1",
        examples=[0.1, 0.3],
    )
    elapsed_ms: int = Field(
        ...,
        description="Temps mis par la requête",
        examples=[300],
    )


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

app = FastAPI(title="ASNR Anomalies API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe : utilise par Docker/healthcheck plus tard."""
    return {"status": "ok"}


def _to_hit(row: dict) -> SearchHit:
    """Convertit un dict DB en SearchHit (mismatch : texte_source -> extrait tronque)."""

    texte = row["texte_source"]
    extrait: str = texte if len(texte) <= 200 else texte[:200] + "…"

    # construire SearchHit en passant chaque champ explicitement.
    return SearchHit(
        identifiant=row["identifiant"],
        lettre=row["lettre"],
        action_attendue=row["action_attendue"],
        delai_mentionne=row["delai_mentionne"],
        criticite=row["criticite"],
        extrait=extrait,
        theme_llm=row["theme_llm"],
        theme_nli=row["theme_nli"],
        theme_nli_score=row["theme_nli_score"],
        distance=row["distance"],
        equipements=row["equipements"],
    )


@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=1, description="Question en langage naturel."),
    k: int = Query(3, ge=1, le=20, description="Nombre de resultats."),
) -> SearchResponse:
    """Recherche semantique : E5 query embedding + cosinus pgvector."""

    if query.strip() == "":
        raise HTTPException(422, "la query est vide")

    t0 = time.perf_counter()

    # encoder la query + chercher en base.
    rows: list[dict] = []
    conn = get_connection()
    try:
        rows = search_similar(conn, embed_query(query), limit=k)
    finally:
        conn.close()

    # convertir chaque dict DB en SearchHit.
    hits: list[SearchHit] = [_to_hit(row) for row in rows]  # remplace par la liste convertie

    # calcul du tassement (gap entre top-1 et top-K).
    tassement = hits[-1].distance - hits[0].distance if len(hits) >= 2 else None

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return SearchResponse(results=hits, tassement=tassement, elapsed_ms=elapsed_ms)
