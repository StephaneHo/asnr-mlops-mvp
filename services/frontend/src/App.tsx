import { useState, Suspense } from "react";
import { useSuspenseQuery } from "@tanstack/react-query";
import { ErrorBoundary } from "react-error-boundary";

const API_URL = "http://127.0.0.1:8000";

// ---------------------------------------------------------------
// Types qui matchent les schemas Pydantic backend.
// ---------------------------------------------------------------

interface SearchHit {
  identifiant: string;
  lettre: string;
  action_attendue: string;
  equipements: string[];
  delai_mentionne: string | null;
  criticite: string;
  extrait: string;
  theme_nli: string;
  theme_llm: string;
  theme_nli_score: number;
  distance: number;
}

interface SearchResponse {
  results: SearchHit[];
  tassement: number | null;
  elapsed_ms: number;
}

// ---------------------------------------------------------------
// Fonction de fetch pure (queryFn).
// ---------------------------------------------------------------

// TODO F3a : implementer fetchSearch.
// Etapes :
//   1. await fetch(`${API_URL}/search?query=${encodeURIComponent(query)}&k=3`)
//   2. if (!response.ok) throw new Error(`HTTP ${response.status}`)
//   3. return response.json() (typee SearchResponse grace au return type).
async function fetchSearch(query: string): Promise<SearchResponse> {
  // TODO F3a
  throw new Error("not implemented");
}

// ---------------------------------------------------------------
// Composant <Results> : consomme useSuspenseQuery, assume data presente.
// ---------------------------------------------------------------

function Results({ query }: { query: string }) {
  // TODO F3b : useSuspenseQuery.
  // Indice :
  //   const { data } = useSuspenseQuery<SearchResponse>({
  //     queryKey: ["search", query],
  //     queryFn: () => fetchSearch(query),
  //   });
  // queryKey est cle de cache : memes args -> meme cache, retour instantane.

  // TODO F4c : rendu des hits.
  // Structure suggeree :
  //   - 1 ligne de resume : "{N} resultats - tassement: X - {Y}ms"
  //   - data.results.map((hit, i) => (...)) avec identifiant, lettre,
  //     criticite (badge), extrait, distance, theme_nli, theme_llm.
  //   - Tailwind : space-y-3, chaque carte p-4 bg-white border rounded-md.
  return (
    <div className="text-slate-500">TODO F4c : afficher les resultats</div>
  );
}

// ---------------------------------------------------------------
// App : etat UI + composition Suspense/ErrorBoundary autour de <Results>.
// ---------------------------------------------------------------

function App() {
  // TODO F2 : 2 useState.
  //   - query: string ("" au depart) - ce qui est dans l'input
  //   - submittedQuery: string | null (null au depart) - ce qu'on a soumis.
  //   submittedQuery declenche le rendu de <Results> (et donc le fetch Suspense).

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold">Recherche d'anomalies ASNR</h1>

        {/* TODO F4a : barre de recherche.
            - <input value={query} onChange={(e) => setQuery(e.target.value)}>
              placeholder "Tapez votre requete..."
              Tailwind : flex-1 px-4 py-2 border border-slate-300 rounded-md
            - <button onClick={() => setSubmittedQuery(query)}>
              Tailwind : px-6 py-2 bg-blue-600 text-white rounded-md
            - parent <div className="flex gap-2"> */}

        {/* TODO F4b : rendu conditionnel sur submittedQuery.
            Pattern :
              {submittedQuery !== null && (
                <ErrorBoundary fallbackRender={({ error }) => <ErrorBox msg={...} />}>
                  <Suspense fallback={<div>Recherche en cours...</div>}>
                    <Results query={submittedQuery} />
                  </Suspense>
                </ErrorBoundary>
              )}
            Indice : la prop `fallbackRender` recoit {error, resetErrorBoundary}.
            Tu peux soit creer un sous-composant <ErrorBox/>, soit ecrire le JSX inline. */}
      </div>
    </div>
  );
}

export default App;
