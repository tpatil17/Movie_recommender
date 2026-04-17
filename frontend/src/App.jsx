import { useState, useCallback } from "react"
const API = import.meta.env.VITE_API_URL || ""

function App() {
  const [query, setQuery] = useState("")
  const [suggestions, setSuggestions] = useState([])
  const [selectedTitle, setSelectedTitle] = useState("")
  const [userId, setUserId] = useState(1)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")


  // Debounced search for autocomplete
  const searchMovies = useCallback(async (q) => {
    if (q.length < 2) { setSuggestions([]); return }
    try {
      const res = await fetch(`${API}/api/movies/search?q=${encodeURIComponent(q)}`)
      const data = await res.json()
      setSuggestions(data.results || [])
    } catch {
      setSuggestions([])
    }
  }, [API])

  const handleQueryChange = (e) => {
    const val = e.target.value
    setQuery(val)
    searchMovies(val)
  }

  const handleSelectSuggestion = (title) => {
    setQuery(title)
    setSelectedTitle(title)
    setSuggestions([])
  }

  const handleRecommend = async () => {
    if (!selectedTitle) { setError("Please select a movie from the suggestions"); return }
    setLoading(true)
    setError("")
    setResults([])
    try {
      const res = await fetch(`${API}/api/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, title: selectedTitle, top_n: 10 })
      })
      if (!res.ok) throw new Error("Movie not found")
      const data = await res.json()
      setResults(data.results)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <h1 className="text-2xl font-bold text-white">🎬 Movie Recommender</h1>
        <p className="text-gray-400 text-sm mt-1">Hybrid content + collaborative filtering</p>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-10">
        {/* Search Section */}
        <div className="bg-gray-900 rounded-2xl p-6 mb-8 border border-gray-800">
          <h2 className="text-lg font-semibold mb-4 text-gray-200">Find Recommendations</h2>

          <div className="flex gap-4 mb-4">
            {/* Movie Search */}
            <div className="relative flex-1">
              <input
                type="text"
                value={query}
                onChange={handleQueryChange}
                placeholder="Search a movie you like..."
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
              />
              {/* Autocomplete dropdown */}
              {suggestions.length > 0 && (
                <div className="absolute top-full left-0 right-0 bg-gray-800 border border-gray-700 rounded-lg mt-1 z-10 max-h-60 overflow-y-auto">
                  {suggestions.map((title) => (
                    <button
                      key={title}
                      onClick={() => handleSelectSuggestion(title)}
                      className="w-full text-left px-4 py-2 hover:bg-gray-700 text-gray-200 text-sm"
                    >
                      {title}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* User ID input */}
            <input
              type="number"
              value={userId}
              onChange={(e) => setUserId(parseInt(e.target.value))}
              placeholder="User ID"
              min={1}
              className="w-28 bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
            />
          </div>

          <button
            onClick={handleRecommend}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-900 disabled:text-blue-400 text-white font-semibold py-3 rounded-lg transition-colors"
          >
            {loading ? "Finding recommendations..." : "Get Recommendations"}
          </button>

          {error && (
            <p className="text-red-400 text-sm mt-3">{error}</p>
          )}
        </div>

        {/* Results Grid */}
        {results.length > 0 && (
          <div>
            <h2 className="text-lg font-semibold mb-4 text-gray-200">
              Recommendations based on <span className="text-blue-400">{selectedTitle}</span>
            </h2>
            <div className="grid grid-cols-1 gap-4">
              {results.map((movie, idx) => (
                <div
                  key={idx}
                  className="bg-gray-900 border border-gray-800 rounded-xl p-5 flex items-center justify-between hover:border-gray-600 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    {/* Rank */}
                    <span className="text-2xl font-bold text-gray-600 w-8">
                      {idx + 1}
                    </span>
                    <div>
                      <h3 className="font-semibold text-white">{movie.title}</h3>
                      {/* Reason tag */}
                      <span className="inline-block mt-1 text-xs bg-gray-800 text-gray-400 border border-gray-700 rounded-full px-3 py-0.5">
                        {movie.reason}
                      </span>
                    </div>
                  </div>
                  {/* Predicted rating */}
                  <div className="text-right">
                    <div className="text-yellow-400 font-bold text-lg">
                      ⭐ {movie.predicted_rating}
                    </div>
                    <div className="text-gray-500 text-xs">predicted</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App