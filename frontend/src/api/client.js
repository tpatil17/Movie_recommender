// Single place for base URLs, fetch mechanics and error shape.
//
// Every surface calls through here so one service being down degrades only the
// tab that needs it, rather than blanking the page. That matters for a live
// demo: the agent needs an OpenAI key and can fail independently of the
// backend, and a dead agent should not take the recommender down with it.

// Empty by default so backend calls stay relative and go through the `/api`
// proxy in vite.config.js during development. VITE_API_URL overrides it in
// production (see .env.production). The agent has no proxy entry, so it is
// addressed absolutely — its CORS is open, so that is fine.
const API = import.meta.env.VITE_API_URL || ""
const AGENT = import.meta.env.VITE_AGENT_URL || "http://localhost:8002"

export const ENDPOINTS = { API, AGENT }

class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(url, options = {}) {
  let res
  try {
    res = await fetch(url, options)
  } catch {
    // Network-level failure: service down, CORS, DNS. No status to report.
    throw new ApiError("unreachable", 0)
  }
  if (!res.ok) {
    let detail = `request failed (${res.status})`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      // Non-JSON error body; keep the generic message.
    }
    throw new ApiError(detail, res.status)
  }
  return res.json()
}

// ─── backend ──────────────────────────────────────────────────────────────

export const searchMovies = (q) =>
  request(`${API}/api/movies/search?q=${encodeURIComponent(q)}`)

export const getRecommendations = (userId, title, topN = 10) =>
  request(`${API}/api/recommendations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, title, top_n: topN }),
  })

export const getForYou = (userId, topN = 10) =>
  request(`${API}/api/recommendations/for-you?user_id=${userId}&top_n=${topN}`)

export const getSimilar = (title, topN = 10) =>
  request(`${API}/api/movies/similar?title=${encodeURIComponent(title)}&top_n=${topN}`)

export const backendHealth = () => request(`${API}/health`)

// ─── agent ────────────────────────────────────────────────────────────────

export const sendChat = (message, sessionId, userId) =>
  request(`${AGENT}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // user_id is sent even though the current ChatRequest schema ignores it.
    // Once identity is threaded through the agent this starts working with no
    // frontend change; until then the agent falls back to DEFAULT_USER_ID.
    body: JSON.stringify({ message, session_id: sessionId, user_id: userId }),
  })

export const resetSession = (sessionId) =>
  fetch(`${AGENT}/sessions/${sessionId}`, { method: "DELETE" }).catch(() => {})

export const agentMeta = () => request(`${AGENT}/meta`)
export const agentHealth = () => request(`${AGENT}/health`)

export { ApiError }
