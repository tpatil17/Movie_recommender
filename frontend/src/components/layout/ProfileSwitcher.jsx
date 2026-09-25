import { useState, useRef, useEffect } from "react"
import { DEMO_USERS } from "../../data/demoUsers"

// Replaces the raw user_id number input.
//
// The point is demonstrable personalisation: switching profiles visibly changes
// recommendations, so a viewer verifies it rather than taking it on trust. The
// cold-start profile is included so that state is shown, not hidden.

export default function ProfileSwitcher({ user, onChange }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    const onClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener("mousedown", onClick)
    return () => document.removeEventListener("mousedown", onClick)
  }, [])

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-left transition-colors hover:border-gray-600"
      >
        <div>
          <div className="text-xs font-medium text-white">{user.label}</div>
          <div className="font-mono text-[10px] text-gray-500">
            user {user.id} · {user.ratings.toLocaleString()} ratings
          </div>
        </div>
        <span className="text-gray-500">{open ? "▴" : "▾"}</span>
      </button>

      {open && (
        <div className="absolute right-0 top-full z-20 mt-1 w-80 overflow-hidden rounded-xl border border-gray-700 bg-gray-800 shadow-xl">
          <div className="border-b border-gray-700 px-4 py-2 text-[10px] uppercase tracking-wider text-gray-500">
            MovieLens profiles
          </div>
          {DEMO_USERS.map((u) => (
            <button
              key={u.id}
              onClick={() => {
                onChange(u)
                setOpen(false)
              }}
              className={`block w-full border-b border-gray-700/50 px-4 py-2.5 text-left transition-colors last:border-0 hover:bg-gray-700 ${
                u.id === user.id ? "bg-gray-700/50" : ""
              }`}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-xs font-medium text-white">{u.label}</span>
                <span className="shrink-0 font-mono text-[10px] text-gray-500">
                  {u.ratings.toLocaleString()} ratings
                </span>
              </div>
              <div className="mt-0.5 text-[10px] text-gray-500">{u.genres}</div>
              {u.samples.length > 0 && (
                <div className="mt-1 truncate text-[10px] text-gray-600">
                  liked: {u.samples.join(", ")}
                </div>
              )}
              {u.coldStart && (
                <div className="mt-1 text-[10px] text-amber-500">
                  no history — demonstrates cold start
                </div>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
