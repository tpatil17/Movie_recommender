import { useState, useEffect, useRef } from "react"
import { sendChat, resetSession, agentMeta, agentHealth, ENDPOINTS } from "../../api/client"
import ToolTrace from "./ToolTrace"

const GREETING = {
  role: "assistant",
  content:
    "Ask me for a recommendation. Naming a film you liked routes to the seed-anchored hybrid; " +
    "an open-ended request like \"what should I watch?\" routes to pure collaborative filtering. " +
    "Every tool call I make is shown below my reply.",
  toolCalls: [],
}

const SUGGESTIONS = [
  "What should I watch tonight?",
  "I loved Interstellar, what next?",
  "What's most similar to The Matrix?",
  "Recommend movies like Zzyzx Blorptastic 9000",
]

export default function ChatTab({ userId }) {
  const [messages, setMessages] = useState([GREETING])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [meta, setMeta] = useState(null)
  const [agentDown, setAgentDown] = useState(false)
  const [probe, setProbe] = useState("checking")
  const sessionId = useRef(`demo-${Date.now()}`)
  const bottomRef = useRef(null)

  // Only a status of 0 means the service is unreachable. /meta is a newer
  // endpoint, so an agent started before it existed returns 404 — that agent
  // still serves /chat perfectly well and must not be reported as offline.
  useEffect(() => {
    let cancelled = false
    agentMeta()
      .then((m) => {
        if (cancelled) return
        setMeta(m)
        setAgentDown(false)
        setProbe("online")
      })
      .catch(async (err) => {
        if (cancelled) return
        if (err.status === 0) {
          setAgentDown(true)
          setProbe("unreachable")
          return
        }
        // Reachable but /meta is missing or erroring. Fall back to /health.
        try {
          await agentHealth()
          setProbe("online-no-meta")
        } catch (e) {
          setAgentDown(e.status === 0)
          setProbe(e.status === 0 ? "unreachable" : "error")
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  const send = async (text) => {
    const body = (text ?? input).trim()
    if (!body || loading) return
    setInput("")
    setMessages((m) => [...m, { role: "user", content: body }])
    setLoading(true)
    try {
      const data = await sendChat(body, sessionId.current, userId)
      setAgentDown(false)
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.response,
          toolCalls: data.tool_calls || [],
          latencyMs: data.latency_ms,
        },
      ])
    } catch (err) {
      if (err.status === 0) {
        setAgentDown(true)
        setProbe("unreachable")
      }
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            err.status === 0
              ? `Can't reach the agent at ${ENDPOINTS.AGENT}. Start it with ` +
                `"cd services/agent && python main.py" — it needs OPENAI_API_KEY. ` +
                `The For You, Discover and Evaluation tabs work without it.`
              : `Agent returned ${err.status}: ${err.message}`,
          toolCalls: [],
          isError: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const clear = async () => {
    await resetSession(sessionId.current)
    sessionId.current = `demo-${Date.now()}`
    setMessages([GREETING])
  }

  return (
    <div className="mx-auto flex h-[calc(100vh-150px)] max-w-3xl flex-col px-4 py-5">
      {/* Agent configuration, so what is being demonstrated is never ambiguous */}
      <div className="mb-3 flex flex-wrap items-center gap-3 text-[11px] text-gray-500">
        {probe === "checking" && <span>connecting…</span>}

        {probe === "unreachable" && (
          <span className="text-amber-500">
            ● agent unreachable at {ENDPOINTS.AGENT} — start it on port 8002
          </span>
        )}

        {probe === "online" && meta && (
          <>
            <span className="text-emerald-500">● agent online</span>
            <span className="font-mono">tag {meta.agent_tag}</span>
            <span className="font-mono">temp {meta.temperature}</span>
            {meta.tools && <span className="font-mono">{meta.tools.length} tools</span>}
          </>
        )}

        {probe === "online-no-meta" && (
          <>
            <span className="text-emerald-500">● agent online</span>
            <span className="text-gray-600">
              /meta unavailable — restart the agent to show tag and temperature
            </span>
          </>
        )}

        {probe === "error" && (
          <span className="text-amber-500">● agent reachable but erroring</span>
        )}
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto pb-4">
        {messages.map((msg, i) => (
          <div key={i}>
            <div className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "rounded-br-sm bg-blue-600 text-white"
                    : msg.isError
                      ? "rounded-bl-sm border border-amber-800/60 bg-amber-950/40 text-amber-200"
                      : "rounded-bl-sm border border-gray-700 bg-gray-800 text-gray-100"
                }`}
              >
                {msg.content}
              </div>
            </div>
            {msg.role === "assistant" && (
              <ToolTrace toolCalls={msg.toolCalls} latencyMs={msg.latencyMs} />
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-1 pl-1">
            {[0, 150, 300].map((d) => (
              <span
                key={d}
                className="h-2 w-2 animate-bounce rounded-full bg-gray-500"
                style={{ animationDelay: `${d}ms` }}
              />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {messages.length <= 1 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => send(s)}
              className="rounded-full border border-gray-700 px-3 py-1.5 text-xs text-gray-400 transition-colors hover:border-gray-500 hover:text-gray-200"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <div className="border-t border-gray-800 pt-3">
        <div className="flex items-end gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                send()
              }
            }}
            placeholder="Ask for a recommendation…"
            rows={2}
            className="flex-1 resize-none rounded-xl border border-gray-700 bg-gray-800 px-4 py-3 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
          />
          <div className="flex flex-col gap-2">
            <button
              onClick={() => send()}
              disabled={loading || !input.trim()}
              className="rounded-xl bg-blue-600 px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500"
            >
              Send
            </button>
            <button onClick={clear} className="text-xs text-gray-500 hover:text-gray-300">
              Clear
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
