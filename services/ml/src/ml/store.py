"""Acces Postgres + pgvector pour stocker et rechercher les anomalies enrichies."""

from __future__ import annotations

import json
import os

import numpy as np
import psycopg
from ml.extraction import AnomalieEnrichie
from pgvector.psycopg import register_vector

# Connexion par defaut (cf. docker-compose.yml). Override possible via env vars.
DSN = os.environ.get(
    "ASNR_DB_DSN",
    "host=localhost port=5433 dbname=asnr user=asnr password=asnr_dev_password",
)


def get_connection() -> psycopg.Connection:
    """Ouvre une connexion Postgres + enregistre le type VECTOR de pgvector."""
    conn = psycopg.connect(DSN)
    register_vector(conn)  # active la conversion list[float] <-> VECTOR
    return conn


def insert_anomalie(
    conn: psycopg.Connection,
    anomalie: AnomalieEnrichie,
    embedding: list[float],
    lettre: str,
) -> bool:
    """Insere une anomalie. Idempotent : retourne False si deja presente.

    Args:
        conn:       connexion Postgres ouverte (cf. get_connection).
        anomalie:   AnomalieEnrichie produite par la cascade.
        embedding:  vecteur 384D (embed_anomalie).
        lettre:     reference de la lettre source (ex: 'INSSN-CAE-2026-0206').

    Returns:
        True si insere, False si la paire (lettre, identifiant) existait deja.
    """
    sql = """
        INSERT INTO anomalies (
            identifiant, lettre, criticite,
            theme_nli, theme_nli_score, theme_llm,
            action_attendue, equipements, delai_mentionne,
            texte_source, embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (lettre, identifiant) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                anomalie.identifiant,
                lettre,
                anomalie.criticite,
                anomalie.theme_nli,
                anomalie.theme_nli_score,
                anomalie.theme_llm,
                anomalie.action_attendue,
                json.dumps(anomalie.equipements_concernes),
                anomalie.delai_mentionne,
                anomalie.texte_source,
                embedding,
            ),
        )
        inserted = cur.rowcount > 0
    conn.commit()
    return inserted


def search_similar(
    conn: psycopg.Connection,
    query_vector: list[float],
    limit: int = 10,
) -> list[dict]:
    """Renvoie les `limit` anomalies les plus proches du vecteur fourni (cosinus).

    Args:
        conn:         connexion Postgres ouverte.
        query_vector: vecteur 384D a comparer (utiliser embed_text + prefixe 'query: ').
        limit:        nombre max de resultats.

    Returns:
        Liste de dicts triee par distance cosinus croissante (1er = plus proche).
        Chaque dict contient les champs metier + 'distance' (0 = identique, 2 = oppose).
    """

    # Operateur de distance cosinus pgvector : <=>  (0 = identique, 2 = oppose)
    sql = """
        SELECT
            id,
            identifiant,
            lettre,
            criticite,
            theme_nli,
            theme_nli_score,
            theme_llm,
            action_attendue,
            equipements,
            delai_mentionne,
            texte_source,
            created_at,
            embedding <=> %s AS distance
        FROM anomalies
        ORDER BY embedding <=> %s ASC
        LIMIT %s
    """

    # pgvector-python convertit numpy.ndarray -> VECTOR mais pas list[float].
    # Sans ce cast, Postgres reçoit un double precision[] et l'operateur
    # <=> rejette ("operator does not exist: vector <=> double precision[]").
    vec = np.asarray(query_vector, dtype=np.float32)

    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, (vec, vec, limit))
        return cur.fetchall()
