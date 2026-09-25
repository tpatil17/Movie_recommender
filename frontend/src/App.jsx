import { useState } from "react"
import ProfileSwitcher from "./components/layout/ProfileSwitcher"
import ChatTab from "./components/chat/ChatTab"
import ForYouTab from "./components/forYou/ForYouTab"
import DiscoverTab from "./components/discover/DiscoverTab"
import EvaluationTab from "./components/evaluation/EvaluationTab"
import { DEFAULT_USER } from "./data/demoUsers"

const TABS = [
  { id: "chat", label: "Chat", hint: "LangChain agent over MCP tools" },
  { id: "forYou", label: "For You", hint: "pure collaborative filtering" },
  { id: "discover", label: "Discover", hint: "seed-anchored recommendations" },
  { id: "evaluation", label: "Evaluation", hint: "offline results" },
]

export default function App() {
  const [tab, setTab] = useState("chat")
  const [user, setUser] = useState(DEFAULT_USER)

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="border-b border-gray-800 bg-gray-900">
        <div className="mx-auto flex max-w-4xl items-start justify-between gap-4 px-6 py-4">
          <div>
            <h1 className="text-xl font-bold">🎬 Movie Recommendation Agent</h1>
            <p className="mt-0.5 text-xs text-gray-500">
              Hybrid recommender · MCP tool server · LangChain agent
            </p>
          </div>
          <ProfileSwitcher user={user} onChange={setUser} />
        </div>

        <div className="mx-auto max-w-4xl px-6">
          <nav className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                title={t.hint}
                className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
                  tab === t.id
                    ? "border-blue-500 text-white"
                    : "border-transparent text-gray-500 hover:text-gray-300"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {tab === "chat" && <ChatTab userId={user.id} />}
      {tab === "forYou" && <ForYouTab user={user} />}
      {tab === "discover" && <DiscoverTab user={user} />}
      {tab === "evaluation" && <EvaluationTab />}
    </div>
  )
}
