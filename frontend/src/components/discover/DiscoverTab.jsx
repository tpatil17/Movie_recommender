import { useState, useCallback, useRef } from "react"
import { searchMovies, getRecommendations, getSimilar } from "../../api/client"
import MovieCard from "../forYou/MovieCard"

// Seed-anchored recommendations: "movies like X".
//
// Two modes on the same seed, because they are different questions and the
// evaluation treats them separately:
//   hybrid  — content neighbours re-ranked by the user's predicted rating
//   similar — content similarity only, no personalisation

const MODES = [
  { id: "hybrid", label: "Personalised", hint: "content retrieval → CF re-ranking" },
  { id: "similar", label: "Content only", hint: "cosine similarity, no personalisation" },
]

export default function DiscoverTab({ user }) {
  const [query, setQuery] = useState("")
  const [suggestions, setSuggestions] = useState([])
  const [selected, setSelected] = useState("")
  const [mode, setMode] = useState("hybrid")
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const timer = useRef(null)

  const lookup = useCallback(async (q) => {
    if (q.length < 2) return setSuggestions([])
    try {
      const data = await searchMovies(q)
      setSuggestions(data.results || [])
    } catch {
      setSuggestions([])
    }
  }, [])

  const onChange = (e) => {
    const val = e.target.value
    setQuery(val)
    setSelected("")
    clearTimeout(timer.current)
    timer.current = setTimeout(() => lookup(val), 300)
  }

  const run = async (title, nextMode = mode) => {
    if (!title) return setError("Pick a movie from the suggestions first.")
    setLoading(true)
    setError("")
    setResults([])
    try {
      const data =
        nextMode === "hybrid"
          ? await getRecommendations(user.id, title, 10)
          : await getSimilar(title, 10)
      // get_similar returns raw content matches without predicted_rating,
      // so normalise both shapes into what MovieCard expects.
      setResults(
        (data.results || []).map((r) => ({
          ...r,
          reason: r.reason || "Content similarity",
        })),
      )
    } catch (err) {
      setError(err.status === 0 ? "Can't reach the backend on port 8000." : err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <div className="mb-6 rounded-2xl border border-gray-800 bg-gray-900 p-5">
        <h2 className="mb-4 text-lg font-semibold text-gray-200">Movies like…</h2>

        <div className="relative mb-4">
          <input
            value={query}
            onChange={onChange}
            placeholder="Search a movie you liked…"
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-4 py-3 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
          />
          {suggestions.length > 0 && (
            <div className="absolute inset-x-0 top-full z-10 mt-1 max-h-60 overflow-y-auto rounded-lg border border-gray-700 bg-gray-800">
              {suggestions.map((title) => (
                <button
                  key={title}
                  onClick={() => {
                    setQuery(title)
                    setSelected(title)
                    setSuggestions([])
                    run(title)
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-200 hover:bg-gray-700"
                >
                  {title}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          {MODES.map((m) => (
            <button
              key={m.id}
              onClick={() => {
                setMode(m.id)
                if (selected) run(selected, m.id)
              }}
              title={m.hint}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                mode === m.id
                  ? "bg-blue-600 text-white"
                  : "border border-gray-700 text-gray-400 hover:text-gray-200"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <p className="mt-2 text-[11px] text-gray-600">
          {MODES.find((m) => m.id === mode)?.hint}
          {mode === "hybrid" && " · offline evaluation shows this path is retrieval-bound"}
        </p>

        {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
      </div>

      {loading && <p className="text-sm text-gray-500">Loading…</p>}

      {results.length > 0 && (
        <>
          <h3 className="mb-3 text-sm text-gray-400">
            Based on <span className="text-blue-400">{selected}</span>
          </h3>
          <div className="space-y-3">
            {results.map((movie, i) => (
              <MovieCard
                key={`${movie.title}-${i}`}
                movie={movie}
                rank={i + 1}
                showScore={mode === "hybrid"}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
