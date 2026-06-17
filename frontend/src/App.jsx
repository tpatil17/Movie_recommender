import { useState, useCallback, useEffect, useRef } from "react"

const API = import.meta.env.VITE_API_URL || ""
const AGENT_API = import.meta.env.VITE_AGENT_URL || "http://localhost:8002"

// ─── Chat Tab ────────────────────────────────────────────────────────────────

function ChatTab() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi! I'm your movie recommendation assistant. Tell me what kind of movies you're in the mood for, or ask me to recommend something based on a film you love.",
    },
  ])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const sessionId = useRef(`session-${Date.now()}`)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return
    setInput("")
    setMessages((prev) => [...prev, { role: "user", content: text }])
    setLoading(true)
    try {
      const res = await fetch(`${AGENT_API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId.current }),
      })
      const data = await res.json()
      setMessages((prev) => [...prev, { role: "assistant", content: data.response }])
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I couldn't reach the agent. Make sure it's running on port 8002." },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearChat = () => {
    sessionId.current = `session-${Date.now()}`
    setMessages([
      {
        role: "assistant",
        content: "Fresh start! What kind of movies are you looking for?",
      },
    ])
  }

  return (
    <div className="flex flex-col h-[calc(100vh-120px)] max-w-3xl mx-auto px-4 py-6">
      {/* Chat history */}
      <div className="flex-1 overflow-y-auto space-y-4 pb-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {msg.role === "assistant" && (
              <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-sm mr-3 mt-1 shrink-0">
                🎬
              </div>
            )}
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap leading-relaxed ${
                msg.role === "user"
                  ? "bg-blue-600 text-white rounded-br-sm"
                  : "bg-gray-800 text-gray-100 border border-gray-700 rounded-bl-sm"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-sm mr-3 shrink-0">
              🎬
            </div>
            <div className="bg-gray-800 border border-gray-700 rounded-2xl rounded-bl-sm px-4 py-3">
              <span className="flex gap-1">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
              </span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-gray-800 pt-4">
        <div className="flex gap-3 items-end">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask for a recommendation, e.g. 'What should I watch if I loved Interstellar?'"
            rows={2}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none text-sm"
          />
          <div className="flex flex-col gap-2">
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white px-4 py-3 rounded-xl transition-colors text-sm font-medium"
            >
              Send
            </button>
            <button
              onClick={clearChat}
              className="text-gray-500 hover:text-gray-300 text-xs text-center transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
        <p className="text-gray-600 text-xs mt-2">Press Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  )
}

// ─── Recommend Tab ────────────────────────────────────────────────────────────

function RecommendTab() {
  const [query, setQuery] = useState("")
  const [suggestions, setSuggestions] = useState([])
  const [selectedTitle, setSelectedTitle] = useState("")
  const [userId, setUserId] = useState(1)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const debounceTimer = useRef(null)

  useEffect(() => {
    fetch(`${API}/health`).catch(() => {})
  }, [])

  const searchMovies = useCallback(async (q) => {
    if (q.length < 2) { setSuggestions([]); return }
    try {
      const res = await fetch(`${API}/api/movies/search?q=${encodeURIComponent(q)}`)
      const data = await res.json()
      setSuggestions(data.results || [])
    } catch {
      setSuggestions([])
    }
  }, [])

  const handleQueryChange = (e) => {
    const val = e.target.value
    setQuery(val)
    clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => searchMovies(val), 300)
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
        body: JSON.stringify({ user_id: userId, title: selectedTitle, top_n: 10 }),
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
    <div className="max-w-4xl mx-auto px-6 py-10">
      <div className="bg-gray-900 rounded-2xl p-6 mb-8 border border-gray-800">
        <h2 className="text-lg font-semibold mb-4 text-gray-200">Find Recommendations</h2>
        <div className="flex gap-4 mb-4">
          <div className="relative flex-1">
            <input
              type="text"
              value={query}
              onChange={handleQueryChange}
              placeholder="Search a movie you like..."
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
            />
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
        {error && <p className="text-red-400 text-sm mt-3">{error}</p>}
      </div>

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
                  <span className="text-2xl font-bold text-gray-600 w-8">{idx + 1}</span>
                  <div>
                    <h3 className="font-semibold text-white">{movie.title}</h3>
                    <span className="inline-block mt-1 text-xs bg-gray-800 text-gray-400 border border-gray-700 rounded-full px-3 py-0.5">
                      {movie.reason}
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-yellow-400 font-bold text-lg">⭐ {movie.predicted_rating}</div>
                  <div className="text-gray-500 text-xs">predicted</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── App Shell ────────────────────────────────────────────────────────────────

const TABS = [
  { id: "chat", label: "AI Chat" },
  { id: "recommend", label: "Quick Search" },
]

export default function App() {
  const [activeTab, setActiveTab] = useState("chat")

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between max-w-4xl mx-auto">
          <div>
            <h1 className="text-2xl font-bold text-white">🎬 Movie Recommender</h1>
            <p className="text-gray-400 text-sm mt-0.5">Hybrid content + collaborative filtering</p>
          </div>
          {/* Tabs */}
          <div className="flex bg-gray-800 rounded-lg p-1 gap-1">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? "bg-blue-600 text-white"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {activeTab === "chat" ? <ChatTab /> : <RecommendTab />}
    </div>
  )
}
