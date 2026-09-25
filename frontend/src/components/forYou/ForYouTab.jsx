import { useState, useEffect, useCallback } from "react"
import { getForYou } from "../../api/client"
import MovieCard from "./MovieCard"

// Pure collaborative filtering: no seed movie, the whole candidate pool ranked
// by this user's predicted rating.
//
// This is the best-performing path in offline evaluation (precision@10 0.078
// versus the hybrid's 0.026), so it gets its own surface rather than hiding
// behind the seed-anchored flow.

export default function ForYouTab({ user }) {
  const [results, setResults] = useState([])
  const [coldStart, setColdStart] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const load = useCallback(async () => {
    setLoading(true)
    setError("")
    try {
      const data = await getForYou(user.id, 10)
      setResults(data.results || [])
      setColdStart(Boolean(data.cold_start))
    } catch (err) {
      setError(
        err.status === 0
          ? "Can't reach the backend on port 8000."
          : err.message,
      )
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [user.id])

  useEffect(() => {
    load()
  }, [load])

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-200">
          Recommended for {user.label.toLowerCase()}
        </h2>
        <p className="mt-1 text-sm text-gray-500">
          Pure collaborative filtering — every candidate ranked by this user's predicted
          rating, with no seed movie. Movies they have already rated are excluded.
        </p>
      </div>

      {/* Cold start is a real product state, labelled rather than disguised.
          An unknown user has no SVD factors, so any "personalised" ordering
          would be the global mean in disguise. */}
      {coldStart && (
        <div className="mb-5 rounded-xl border border-amber-800/60 bg-amber-950/30 p-4">
          <div className="text-sm font-semibold text-amber-200">
            Cold start — these are not personalised
          </div>
          <p className="mt-1 text-xs leading-relaxed text-amber-300/70">
            This user has no rating history, so the collaborative model has no learned
            factors for them and every prediction would collapse to the global mean.
            Showing popular titles instead, and saying so.
          </p>
        </div>
      )}

      {loading && <p className="text-sm text-gray-500">Ranking candidate pool…</p>}

      {error && (
        <div className="rounded-xl border border-red-900/60 bg-red-950/30 p-4 text-sm text-red-300">
          {error}
        </div>
      )}

      {!loading && !error && (
        <div className="space-y-3">
          {results.map((movie, i) => (
            <MovieCard
              key={`${movie.title}-${i}`}
              movie={movie}
              rank={i + 1}
              showScore={!coldStart}
            />
          ))}
        </div>
      )}

      {!loading && !error && results.length === 0 && (
        <p className="text-sm text-gray-500">No recommendations returned.</p>
      )}
    </div>
  )
}
