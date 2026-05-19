"""Test de recherche semantique sur les anomalies stockees en Postgres+pgvector.

Usage:
    uv run python scripts/test_search.py --query "couple de serrage des vis"
    uv run python scripts/test_search.py --query "irradiation" --k 5

Prerequis : docker compose up -d + au moins une insertion via run_full_pipeline.py.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ml" / "src"))

from ml.embeddings import embed_query  # noqa: E402
from ml.store import get_connection, search_similar  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", required=True, help="Question utilisateur en langage naturel.")
    parser.add_argument("--k", type=int, default=3, help="Nombre de resultats a retourner.")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Query : {args.query!r}\n")

    # Encode la requete (prefixe E5 'query: ' applique par embed_query).
    vector = embed_query(args.query)

    # Recherche les K anomalies les plus proches via la distance cosinus.
    conn = get_connection()
    try:
        results = search_similar(conn, vector, limit=args.k)
    finally:
        conn.close()

    if not results:
        print("Aucun resultat (base vide ?). Lance d'abord run_full_pipeline.py.")
        return

    print(f"Top {len(results)} resultats :\n")
    for i, r in enumerate(results, start=1):
        texte = r["texte_source"]
        extrait = texte[:120] + ("..." if len(texte) > 120 else "")
        print(
            f"  {i}. [distance={r['distance']:.4f}] {r['lettre']} {r['identifiant']} ({r['criticite']})"
        )
        print(f"     theme NLI : {r['theme_nli']} (score {r['theme_nli_score']:.2f})")
        print(f"     theme LLM : {r['theme_llm']}")
        print(f"     action    : {r['action_attendue']}")
        print(f"     extrait   : {extrait}")
        print()


if __name__ == "__main__":
    main()
