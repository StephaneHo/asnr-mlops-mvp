# asnr-mlops-mvp

Pipeline MLOps d'ingestion et d'extraction d'anomalies depuis les lettres
d'inspection de l'ASNR (Autorité de Sûreté Nucléaire et de Radioprotection).

**Statut :** parser PDF + scraper validés sur 217 lettres réelles ; pipeline
ML (Ollama / NLI / embeddings) en cours de mise en place.

## Idée

Les lettres d'inspection ASNR sont publiques et structurées :

- **Section I — Demandes à traiter prioritairement** → criticité haute
- **Section II — Autres demandes** → criticité normale
- **Section III — Constats ou observations n'appelant pas de réponse** → criticité faible

Cette structure fournit une **supervision faible naturelle** : on peut
entraîner et évaluer un pipeline d'extraction d'anomalies **sans annotation
manuelle**, en utilisant les sections I/II/III comme vérité terrain. Chaque
item porte un identifiant unique (`Demande II.4.b`, `Observation n°1`), une
criticité héritée de sa section, et son texte intégral.

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

## Stack cible

100 % open-source, 100 % on-prem, CPU-only. Aucune dépendance à une API
externe payante.

| Couche | Choix | État |
|---|---|---|
| Parsing PDF | pdfplumber + regex | ✅ implémenté |
| OCR (fallback) | docTR / PaddleOCR | ⏳ à venir (cas pathologiques) |
| NLI zero-shot | mDeBERTa-v3-base-mnli-xnli | ⏳ à venir |
| Embeddings | sentence-transformers (paraphrase-multilingual-MiniLM) | ⏳ à venir |
| LLM local | Phi-3-mini / Mistral-7B Q4 via Ollama | ⏳ à venir |
| Vector store | pgvector dans PostgreSQL | ⏳ à venir |
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

Pré-requis : Python 3.11+, [uv](https://docs.astral.sh/uv/) installé.

```powershell
# 1. Installer les dépendances dans .venv
uv sync

# 2. Scraper le corpus (~38 min pour 200 lettres avec Crawl-delay 10s)
uv run python scripts/scrape_asnr.py --max-pages 15

# 3. Évaluer le parser sur tout le corpus téléchargé
uv run python scripts/validate_parser.py --summary
```

À la fin de la validation, tu auras le bilan exact (volume, complétude,
couverture) sur ton corpus.

## Structure du repo

```
asnr-mlops-mvp/
├── data/raw/asnr/              # PDFs téléchargés (ignoré par git)
├── docs/                       # ADRs, architecture, model cards
├── infra/                      # configs Prometheus, Grafana (à venir)
├── pipelines/                  # flows Prefect (à venir)
├── scripts/
│   ├── scrape_asnr.py          # télécharge listing + PDFs ASNR
│   ├── validate_parser.py      # rapport d'évaluation du parser
│   └── explore_pdf.py          # debug : extraction texte d'un PDF
├── services/
│   ├── api/                    # FastAPI (à venir)
│   ├── frontend/               # React + Vite (à venir)
│   ├── ml/src/ml/
│   │   └── parsing.py          # parser PDF → dict structuré
│   └── worker/                 # Celery (à venir)
├── tests/                      # unit / integration / regression (à venir)
└── .pre-commit-config.yaml     # ruff lint + format avant commit
```

## Documentation

- [Architecture](docs/architecture.md)
- [ADRs (décisions d'architecture)](docs/adr/)
- [Model cards](docs/model_cards/)

## Licence

MIT — voir [LICENSE](LICENSE).
