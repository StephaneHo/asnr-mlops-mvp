# asnr-mlops-mvp

Pipeline MLOps d'ingestion et d'extraction d'anomalies depuis les lettres
d'inspection de l'ASNR (Autorité de Sûreté Nucléaire et de Radioprotection).

**Statut :** pipeline complet validé bout-en-bout — parser PDF, scraper,
cascade NLI/LLM, embeddings E5, stockage Postgres + pgvector, recherche
sémantique. API et frontend à venir.

## Idée

Les lettres d'inspection ASNR sont publiques et structurées :

- **Section I — Demandes à traiter prioritairement** → criticité haute
- **Section II — Autres demandes** → criticité normale
- **Section III — Constats ou observations n'appelant pas de réponse** → criticité faible

Cette structure tient lieu de **supervision faible** : les sections I/II/III
servent de vérité terrain sans annotation manuelle. Chaque item porte un
identifiant (`Demande II.4.b`, `Observation n°1`), une criticité héritée de
la section, et son texte intégral.

Le projet vise à démontrer une chaîne MLOps complète sur ce cas d'usage :

- Ingestion de PDF (parsing structuré, OCR pour les cas pathologiques).
- Cascade de classification : zero-shot NLI (rapide) → LLM local (enrichissement structuré).
- Extraction d'anomalies (type, criticité, passage source) avec identifiant
  traçable jusqu'à la lettre source.
- Recherche sémantique et clustering inter-rapports.
- Tracking d'expériences, versionnement des données, CI avec tests de
  régression, observabilité.

## Résultats actuels (parser PDF)

Validation sur un corpus de **217 lettres** scrappées depuis le site officiel
(`reglementation-controle.asnr.fr`, robots.txt respecté, Crawl-delay 10s) :

| Métrique | Valeur |
|---|---|
| Lettres parsées avec succès | **217 / 217** (100 %) |
| Items extraits (demandes + observations) | **934** |
| Moyenne items par lettre | **4.3** |
| Distribution criticité | normale 96.9 % / haute 2.5 % / faible 0.6 % |
| Complétude `reference_courrier` | 98.6 % |
| Complétude `n_dossier` | 98.6 % |
| Complétude `date_lettre` | 99.1 % |
| Complétude `objet` | 93.5 % |
| Complétude `references` | 95.4 % |

Le corpus couvre **13 divisions ASNR** (LYO, OLS, CAE, MRS, BDX, STR, CHA, LIL,
DRC, DCN, DEP, NAN, DEU) et 4 grands exploitants (EDF, Orano, CEA, ITER+).

## Pipeline parser (état actuel)

```
PDF (data/raw/asnr/*.pdf)
        │
        ▼
   pdfplumber.extract_text()            ─── injection " page X " entre pages
        │
        ▼
   clean_text()                         ─── filtre num pages, adresses,
        │                                    téléphone, notes de bas de page
        ▼                                    (avec flag in_footnote pour les
   split_sections()                          continuations multilignes)
        │
        ▼ {header, synthese, section_I, section_II, section_III}
        │
        ├─► extract_header()            ─► metadata: reference_courrier,
        │                                          objet, n_dossier, date_lettre,
        │                                          references
        │
        └─► extract_items()             ─► items: [{identifiant, type,
                                                     criticité, texte}, ...]
```

Le point d'entrée unique est `parse_letter(pdf_path)` qui orchestre tout
([services/ml/src/ml/parsing.py](services/ml/src/ml/parsing.py)).

## Pipeline ML (cascade NLI + LLM)

Chaque item du parser est enrichi par une cascade à deux étages : NLI
zero-shot pour la classification thématique, LLM local pour les champs en
texte libre.

```
item du parser
        │
        ▼
   classify_theme()                      ─── NLI zero-shot mDeBERTa
        │                                    (texte tronqué à 1000 chars)
        ├─► theme_nli + score
        │
   extract_anomalie()                    ─── LLM Phi-3 via Instructor
        │                                    (texte tronqué à 1000 chars)
        ├─► theme_llm
        ├─► action_attendue (str)
        ├─► equipements_concernes (list)
        └─► delai_mentionne (str | None)
        │
        ▼
   AnomalieEnrichie (Pydantic, 9 champs)
```

Le point d'entrée est `extract_anomalie_cascade(item, client)` dans
[services/ml/src/ml/extraction.py](services/ml/src/ml/extraction.py).

### Validation sur 3 demandes de la lettre Penly

| Demande | Temps | theme_nli (score) | theme_llm | Convergence |
|---|---|---|---|---|
| II.1.a | 138s | autre (0.41) | sûreté | **Divergent** : NLI hésite (< seuil 0.5), LLM tranche |
| II.1.b | 278s | radioprotection et irradiation (0.75) | radioprotection | **Convergent** |
| II.2.a | 54s | radioprotection et irradiation (0.57) | réglementation | **Divergent** : NLI précis, LLM moins |

- Temps moyen : **156 s / item** (CPU only). Acceptable pour quelques dizaines
  d'items en démo ; trop pour traiter les 934 sans batching/GPU.
- **3 / 3 succès** Pydantic après troncature à 1000 caractères (sans
  troncature, Phi-3 renvoie parfois `action_attendue: list[str]` → échec).

### Décision MLOps : garder les deux thèmes

`theme_nli` et `theme_llm` sont conservés en parallèle dans `AnomalieEnrichie`.
Justification :

- NLI a classé II.2.a en `radioprotection`, là où le LLM a renvoyé
  `réglementation` (probablement biaisé par le mot "régulations" du texte
  source).
- À l'inverse, sur II.1.a, NLI est tombé en fallback `autre` (score 0.41 <
  seuil 0.5) tandis que le LLM a renvoyé `sûreté`.
- La divergence est utilisable comme indicateur de doute : ces items peuvent
  être priorisés pour une relecture humaine.

### Limites connues du LLM Phi-3 sur CPU

Limites observées au stade MVP :

- `action_attendue` contient parfois des erreurs typographiques générées par
  Phi-3 (ex : `"réviser l0e l'étiquetage"`).
- La consigne "max 15 mots" du prompt n'est pas toujours respectée.
- Le LLM utilisé seul classe la plupart des demandes en `sûreté` — c'est pour
  ça que la cascade délègue la classification à NLI.

## Stockage et recherche sémantique (Postgres + pgvector)

Chaque `AnomalieEnrichie` est complétée par un **embedding 384D** (E5
multilingue, prefixe `passage:`) puis stockée dans Postgres avec l'extension
pgvector. Un index HNSW sur la colonne `embedding` permet la recherche par
similarité cosinus en quelques millisecondes.

```
PDF -> parse_letter -> extract_anomalie_cascade -> embed_anomalie
                                                       │
                                                       ▼
                                      INSERT anomalies (... embedding VECTOR(384))
                                                       │
                                                       ▼ (operateur <=>)
                                              recherche cosinus indexee HNSW
```

Le schéma SQL est dans [infra/init-db.sql](infra/init-db.sql), chargé
automatiquement au premier `docker compose up -d` via
`/docker-entrypoint-initdb.d/`. Le module d'accès est
[services/ml/src/ml/store.py](services/ml/src/ml/store.py).

### Validation de la recherche

Test sur les 3 anomalies de la lettre Penly, avec 3 requêtes en langage
naturel via [scripts/test_search.py](scripts/test_search.py) :

| Requête | Distance #1 | Distance #2 | Distance #3 | Gap #1→#3 |
|---|---|---|---|---|
| "couple de serrage des vis" | **0.117** | 0.192 | 0.204 | 0.086 |
| "irradiation et radioprotection" | **0.141** | 0.149 | 0.182 | 0.041 |
| "incident sismique sur une centrale" (hors sujet) | 0.206 | 0.212 | 0.220 | **0.014** |

Deux signaux exploitables en aval :

- **Distance #1 absolue** : reflète la pertinence du meilleur match
  (0.117 = match fort, 0.206 = aucun match vraiment proche).
- **Gap #1→#3 (tassement)** : reflète la confiance dans le tri. Un gap
  inférieur à ~0.03 indique que la requête n'a pas de cible claire dans
  le corpus — on peut alors choisir de répondre "aucun résultat pertinent"
  plutôt que de remonter du bruit.

### Idempotence et robustesse

- Index unique `(lettre, identifiant)` + `ON CONFLICT DO NOTHING` :
  le pipeline est rejouable sans créer de doublons.
- Le batch utilise un `try/except` autour de la cascade pour ne pas planter
  tout le lot quand Phi-3 échoue ponctuellement la validation Pydantic.
  La sortie logue `[INS]` / `[DUP]` / `[FAIL]` par item.

## Stack cible

100 % open-source, 100 % on-prem, CPU-only. Aucune dépendance à une API
externe payante.

| Couche | Choix | État |
|---|---|---|
| Parsing PDF | pdfplumber + regex | ✅ implémenté |
| OCR (fallback) | docTR / PaddleOCR | ⏳ à venir (cas pathologiques) |
| NLI zero-shot | mDeBERTa-v3-base-mnli-xnli | ✅ implémenté |
| Typage structuré | Pydantic + Instructor | ✅ implémenté |
| LLM local | Phi-3-mini / Mistral-7B Q4 via Ollama | ✅ implémenté |
| Embeddings | sentence-transformers (intfloat/multilingual-e5-small) | ✅ implémenté |
| Vector store | pgvector dans PostgreSQL 16 | ✅ implémenté |
| Storage objets | MinIO | ⏳ à venir |
| API | FastAPI | ⏳ à venir |
| Workers asynchrones | Celery + Redis | ⏳ à venir |
| Frontend | React + Vite | ⏳ à venir |
| Orchestration | Prefect (self-hosted) | ⏳ à venir |
| Tracking ML | MLflow (self-hosted) | ⏳ à venir |
| Versioning données | DVC | ⏳ à venir |
| Observabilité | Evidently + Prometheus + Grafana | ⏳ à venir |
| CI/CD | GitHub Actions + ruff + pre-commit | ✅ ruff/pre-commit configurés |

## Démarrage rapide

Pré-requis : Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker Desktop,
[Ollama](https://ollama.com/) avec `phi3:mini` (`ollama pull phi3:mini`).

```powershell
# 1. Installer les dépendances dans .venv
uv sync

# 2. Scraper le corpus (~38 min pour 200 lettres avec Crawl-delay 10s)
uv run python scripts/scrape_asnr.py --max-pages 15

# 3. Évaluer le parser sur tout le corpus téléchargé
uv run python scripts/validate_parser.py --summary

# 4. Démarrer Postgres + pgvector (init-db.sql joué au premier up)
docker compose up -d

# 5. Pipeline complet : parser -> cascade -> embedding -> INSERT
uv run python scripts/run_full_pipeline.py --n 5

# 6. Recherche sémantique sur la base
uv run python scripts/test_search.py --query "couple de serrage des vis"
```

## Structure du repo

```
asnr-mlops-mvp/
├── data/raw/asnr/              # PDFs téléchargés (ignoré par git)
├── docs/                       # ADRs, architecture, model cards
├── infra/
│   └── init-db.sql             # schema anomalies + extension pgvector + index HNSW
├── pipelines/                  # flows Prefect (à venir)
├── scripts/
│   ├── scrape_asnr.py          # télécharge listing + PDFs ASNR
│   ├── validate_parser.py      # rapport d'évaluation du parser
│   ├── explore_pdf.py          # debug : extraction texte d'un PDF
│   ├── test_classification.py  # test NLI sur quelques items
│   ├── test_extraction.py      # test LLM (extract_anomalie) sur un item
│   ├── test_cascade.py         # test extract_anomalie_cascade end-to-end
│   ├── test_embeddings.py      # similarité cosinus E5 sur 3 demandes
│   ├── pipeline_e2e.py         # parse -> cascade -> embed (sortie JSON, sans DB)
│   ├── run_full_pipeline.py    # pipeline complet avec insertion Postgres
│   └── test_search.py          # recherche sémantique sur la base
├── services/
│   ├── api/                    # FastAPI (à venir)
│   ├── frontend/               # React + Vite (à venir)
│   ├── ml/src/ml/
│   │   ├── parsing.py          # parser PDF → dict structuré
│   │   ├── classification.py   # NLI zero-shot (mDeBERTa) pour le thème
│   │   ├── extraction.py       # LLM + Instructor → AnomalieEnrichie
│   │   ├── embeddings.py       # E5 multilingue (384D), prefixes passage:/query:
│   │   └── store.py            # accès Postgres + pgvector (insert, search)
│   └── worker/                 # Celery (à venir)
├── docker-compose.yml          # service postgres (pgvector/pgvector:pg16, port 5433)
├── tests/                      # unit / integration / regression (à venir)
└── .pre-commit-config.yaml     # ruff lint + format avant commit
```

## Documentation

- [Architecture](docs/architecture.md)
- [ADRs (décisions d'architecture)](docs/adr/)
- [Model cards](docs/model_cards/)

## Licence

MIT — voir [LICENSE](LICENSE).
