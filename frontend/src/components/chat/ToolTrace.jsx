import { useState } from "react"

// Renders the tool calls the agent made during one turn.
//
// This is the point of the whole UI for an agentic project: without it, a
// reviewer sees a chat box and has to take the architecture on faith. The
// agent already returns `tool_calls` on every /chat response, so this is pure
// presentation of data that was previously thrown away.

const TOOL_COLORS = {
  search_movies: "text-sky-300 border-sky-500/40 bg-sky-500/10",
  get_recommendations: "text-violet-300 border-violet-500/40 bg-violet-500/10",
  get_similar: "text-amber-300 border-amber-500/40 bg-amber-500/10",
  get_for_you: "text-emerald-300 border-emerald-500/40 bg-emerald-500/10",
}

const FAILED = "text-red-300 border-red-500/50 bg-red-500/10"

function ToolCallChip({ call, index, expanded, onToggle }) {
  const failed = call.status !== "success"
  const palette = failed ? FAILED : (TOOL_COLORS[call.name] || "text-gray-300 border-gray-600 bg-gray-700/30")

  return (
    <div>
      <button
        onClick={onToggle}
        className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[11px] transition-opacity hover:opacity-80 ${palette}`}
        title={failed ? "Tool call failed — click for details" : "Click for arguments and result"}
      >
        <span className="opacity-50">{index + 1}</span>
        {call.name}
        <span>{failed ? "✕" : "✓"}</span>
      </button>

      {expanded && (
        <div className="mt-2 mb-1 space-y-2 rounded-lg border border-gray-700 bg-gray-950/80 p-3 font-mono text-[11px]">
          <div>
            <span className="text-gray-500">args</span>
            <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-gray-300">
              {JSON.stringify(call.args ?? {}, null, 2)}
            </pre>
          </div>
          <div>
            <span className="text-gray-500">status</span>{" "}
            <span className={failed ? "text-red-400" : "text-emerald-400"}>{call.status}</span>
          </div>
          {call.result && (
            <div>
              <span className="text-gray-500">result</span>
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all text-gray-400">
                {String(call.result).slice(0, 600)}
                {String(call.result).length > 600 ? "\n… truncated" : ""}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function ToolTrace({ toolCalls, latencyMs }) {
  const [openIndex, setOpenIndex] = useState(null)

  // A turn with no tool calls renders nothing rather than an empty container —
  // the agent answering directly is normal, not a gap to explain.
  if (!toolCalls?.length) return null

  return (
    <div className="mt-2 border-l-2 border-gray-800 pl-3">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[10px] uppercase tracking-wider text-gray-600">
          agent trace · {toolCalls.length} tool {toolCalls.length === 1 ? "call" : "calls"}
        </span>
        {latencyMs ? (
          <span className="font-mono text-[10px] text-gray-600">
            {(latencyMs / 1000).toFixed(2)}s
          </span>
        ) : null}
      </div>

      <div className="flex flex-wrap items-start gap-1.5">
        {toolCalls.map((call, i) => (
          <ToolCallChip
            key={i}
            call={call}
            index={i}
            expanded={openIndex === i}
            onToggle={() => setOpenIndex(openIndex === i ? null : i)}
          />
        ))}
      </div>
    </div>
  )
}
