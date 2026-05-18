# asnr-mlops-mvp

Pipeline MLOps d'ingestion et d'extraction d'anomalies depuis les lettres
d'inspection de l'ASNR (Autorité de Sûreté Nucléaire et de Radioprotection).

**Statut :** parser PDF + scraper + cascade NLI/LLM validés ; embeddings et
stockage à venir.

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

## Pipeline ML (cascade NLI + LLM)

Chaque item du parser est ensuite enrichi par une **cascade à deux étages** qui
combine la rapidité du NLI zero-shot et la richesse du LLM local :

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

- NLI est **plus discriminant** sur les vraies catégories (II.2.a a été classé
  correctement en `radioprotection` par NLI alors que LLM s'est laissé piéger
  par le mot "régulations").
- LLM est **plus tolérant** quand le texte ne correspond à aucune catégorie
  claire (II.1.a : NLI tombe en `"autre"`, LLM tranche en `"sûreté"`).
- La **divergence elle-même est un signal** : items à reviewer en priorité.

### Limites connues du LLM Phi-3 sur CPU

Tolérées au stade MVP, à raffiner ensuite :

- Hallucinations légères dans `action_attendue` (ex : `"réviser l0e
  l'étiquetage"`, fusion `0`/`o`).
- Verbose : `action_attendue` dépasse parfois la consigne "max 15 mots" du
  prompt.
- Tout est mis en `"sûreté"` quand le LLM est utilisé seul (corrigé par la
  cascade : c'est NLI qui porte le vrai signal de thème).

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
| Embeddings | sentence-transformers (paraphrase-multilingual-MiniLM) | ⏳ à venir |
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
│   │   ├── parsing.py          # parser PDF → dict structuré
│   │   ├── classification.py   # NLI zero-shot (mDeBERTa) pour le thème
│   │   └── extraction.py       # LLM + Instructor → AnomalieEnrichie
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
