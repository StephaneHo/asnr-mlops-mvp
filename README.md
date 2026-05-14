# asnr-mlops-mvp

Pipeline MLOps d'ingestion et d'extraction d'anomalies depuis les lettres d'inspection de l'ASNR (Autorité de Sûreté Nucléaire et de Radioprotection).

**Statut :** MVP en construction.

## Idée

Les lettres d'inspection ASNR sont publiques et structurées (sections A : écarts, B : constats, C : synthèse). Cette structure fournit une **supervision faible naturelle** : on peut entraîner et évaluer un pipeline d'extraction d'anomalies sans annotation manuelle, en utilisant la section A comme vérité terrain et en demandant au modèle de la retrouver à partir du reste du document.

Le projet vise à démontrer une chaîne MLOps complète sur ce cas d'usage :

- Ingestion de PDF (OCR si nécessaire, parsing structuré).
- Cascade de classification : zero-shot NLI (rapide) → LLM local (enrichissement).
- Extraction structurée d'anomalies (type, criticité, passage source).
- Recherche sémantique et clustering inter-rapports.
- Tracking d'expériences, versionnement des données, CI avec tests de régression, observabilité.

## Stack

100 % open-source, 100 % on-prem, CPU-only. Aucune dépendance à une API externe payante.

| Couche | Choix |
|---|---|
| OCR | docTR / PaddleOCR |
| NLI zero-shot | mDeBERTa-v3-base-mnli-xnli |
| Embeddings | sentence-transformers (paraphrase-multilingual-MiniLM) |
| LLM local | Phi-3-mini / Mistral-7B Q4 via Ollama |
| Vector store | pgvector dans PostgreSQL |
| Storage objets | MinIO |
| API | FastAPI |
| Workers asynchrones | Celery + Redis |
| Frontend | React + Vite |
| Orchestration | Prefect (self-hosted) |
| Tracking ML | MLflow (self-hosted) |
| Versioning données | DVC |
| Observabilité | Evidently + Prometheus + Grafana |
| CI/CD | GitHub Actions |

## Démarrage rapide

> À venir : `docker compose up` + `ollama pull phi3:mini` suffiront à faire tourner la stack complète.

## Documentation

- [Architecture](docs/architecture.md)
- [ADRs (décisions d'architecture)](docs/adr/)
- [Model cards](docs/model_cards/)

## Licence

MIT — voir [LICENSE](LICENSE).
