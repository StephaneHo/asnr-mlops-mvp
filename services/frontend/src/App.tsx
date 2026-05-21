import { useState, Suspense } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
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

// Schéma de validation du formulaire (cote front, equivalent du Pydantic backend).
const searchSchema = z.object({
  query: z.string().trim().min(1, "La query ne peut pas etre vide"),
});

type SearchInput = z.infer<typeof searchSchema>;

async function fetchSearch(query: string): Promise<SearchResponse> {
  const response = await fetch(`${API_URL}/search?query=${encodeURIComponent(query)}&k=3`)
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json() as Promise<SearchResponse>;

}

function Results({ query }: { query: string }) {
  const { data } = useSuspenseQuery<SearchResponse>({
    queryKey: ["search", query],
    queryFn: () => fetchSearch(query),
  });
  // TODO F4c : rendu des hits.
  // - 1 ligne resume : "{N} resultats - tassement: X - {Y}ms"
  const { results, tassement, elapsed_ms } = data

  const updatedResults = results.map((hit, _) => (<div key={`${hit.lettre}-${hit.identifiant}`}>{hit.extrait}</div>))
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600">
        {results.length} résultats · tassement {tassement?.toFixed(3) ?? "N/A"} · {elapsed_ms} ms
      </p>
      {updatedResults}
    </div>
  );
}

// ---------------------------------------------------------------
// App : form RHF + Zod, etat submittedQuery, composition Suspense/ErrorBoundary.
// ---------------------------------------------------------------

function App() {
  const [submittedQuery, setSubmittedQuery] = useState<string | null>(null);


  const { register, handleSubmit, formState: { errors } } = useForm<SearchInput>({
    resolver: zodResolver(searchSchema),
  });
  // - register("query") : a splatter sur l'input pour le brancher au form
  // - handleSubmit(onValid) : wrapper qui valide AVANT d'appeler onValid
  // - errors.query : objet ZodIssue si la validation a echoue (ou undefined)

  const onValid = (data: SearchInput) => {
    setSubmittedQuery(data.query);
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold">Recherche d'anomalies ASNR</h1>


        <form onSubmit={handleSubmit(onValid)} className="space-y-2">
          <div className="flex gap-2">
            <input {...register("query")} placeholder="..."
              className="flex-1 px-4 py-2 border border-slate-300 rounded-md" />
            <button type="submit"
              className="px-6 py-2 bg-blue-600 text-white rounded-md">
              Rechercher
            </button>
          </div>
          {errors.query && <p className="text-red-600 text-sm">{errors.query.message}</p>}
        </form>



        {submittedQuery !== null && (
          <ErrorBoundary fallbackRender={({ error }) => (
            <div className="p-4 bg-red-50 text-red-700 rounded-md">
              Erreur : {String(error)}
            </div>
          )}>
            <Suspense fallback={
              <div className="p-4 text-slate-500">Recherche en cours...</div>
            }>
              <Results query={submittedQuery} />
            </Suspense>
          </ErrorBoundary>
        )}

      </div>
    </div>
  );
}

export default App;
