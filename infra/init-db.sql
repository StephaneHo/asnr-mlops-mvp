-- Schema initial de la base asnr.
-- Charge automatiquement par Postgres au 1er demarrage du conteneur
-- (montage sur /docker-entrypoint-initdb.d/, cf. docker-compose.yml).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS anomalies (
    id              SERIAL PRIMARY KEY,
    identifiant     TEXT NOT NULL,                    -- ex: 'II.4.b'
    lettre          TEXT NOT NULL,                    -- ex: 'INSSN-CAE-2026-0206'
    criticite       TEXT NOT NULL,                    -- 'haute' / 'normale' / 'faible'
    theme_nli       TEXT,
    theme_nli_score REAL,
    theme_llm       TEXT,
    action_attendue TEXT,
    equipements     JSONB,                            -- list[str] serialisee
    delai_mentionne TEXT,
    texte_source    TEXT NOT NULL,
    embedding       VECTOR(384) NOT NULL,             -- E5-small renvoie 384 dimensions
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index HNSW pour la recherche par similarite cosinus (state-of-the-art en 2024).
-- Operateur de distance cosinus pgvector : <=>
CREATE INDEX IF NOT EXISTS anomalies_embedding_idx
    ON anomalies USING hnsw (embedding vector_cosine_ops);

-- Une demande donnee (lettre + identifiant) ne peut etre inseree qu'une fois.
-- Permet l'idempotence : on peut relancer le pipeline sans creer de doublons.
CREATE UNIQUE INDEX IF NOT EXISTS anomalies_unique_idx
    ON anomalies (lettre, identifiant);
