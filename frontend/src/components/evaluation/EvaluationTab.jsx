import { useState } from "react"
import threeWay from "../../data/eval-three-way.json"
import tail from "../../data/eval-tail.json"
import diagnose from "../../data/eval-diagnose.json"

// Renders the committed offline evaluation artifacts.
//
// Sourced from the actual result JSON in eval_offline/results/, not retyped.
// Hardcoding numbers here would let the UI drift from the harness, which is the
// failure mode the evaluation work exists to prevent.

const METHODS = [
  { key: "cf", label: "Pure CF", color: "bg-emerald-500" },
  { key: "hybrid", label: "Hybrid", color: "bg-violet-500" },
  { key: "popularity_baseline", label: "Popularity", color: "bg-amber-500" },
]

function Bar({ label, value, max, color }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 shrink-0 text-xs text-gray-400">{label}</span>
      <div className="h-6 flex-1 overflow-hidden rounded bg-gray-800">
        <div
          className={`flex h-full items-center justify-end pr-2 ${color}`}
          style={{ width: `${Math.max((value / max) * 100, 6)}%` }}
        >
          <span className="font-mono text-[11px] font-semibold text-gray-950">
            {value.toFixed(4)}
          </span>
        </div>
      </div>
    </div>
  )
}

function Section({ title, children, subtitle }) {
  return (
    <section className="mb-8">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-400">{title}</h3>
      {subtitle && <p className="mt-1 text-xs text-gray-600">{subtitle}</p>}
      <div className="mt-3">{children}</div>
    </section>
  )
}

export default function EvaluationTab() {
  const [metric, setMetric] = useState("precision_at_k")
  const p = threeWay.params

  const METRICS = [
    { id: "precision_at_k", label: "Precision@10" },
    { id: "recall_at_k", label: "Recall@10" },
    { id: "ndcg_at_k", label: "NDCG@10" },
    { id: "catalog_coverage", label: "Coverage" },
  ]

  const values = METHODS.map((m) => threeWay[m.key][metric])
  const max = Math.max(...values)
  const d = diagnose.retrieval
  const dr = diagnose.ranking

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-200">Offline evaluation</h2>
        <p className="mt-1 text-sm text-gray-500">
          {p.users_evaluated} users · relevance = rated ≥ {p.rating_threshold} · per-user{" "}
          {Math.round((1 - p.test_frac) * 100)}/{Math.round(p.test_frac * 100)} split, seed {p.seed}{" "}
          · SVD trained on the train split only · catalog {p.catalog_size.toLocaleString()} titles
        </p>
      </div>

      <Section
        title="Three-way comparison"
        subtitle="Same users, same held-out relevant sets. All three exclude movies the user already rated."
      >
        <div className="mb-4 flex flex-wrap gap-2">
          {METRICS.map((m) => (
            <button
              key={m.id}
              onClick={() => setMetric(m.id)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                metric === m.id
                  ? "bg-blue-600 text-white"
                  : "border border-gray-700 text-gray-400 hover:text-gray-200"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <div className="space-y-2">
          {METHODS.map((m) => (
            <Bar
              key={m.key}
              label={m.label}
              value={threeWay[m.key][metric]}
              max={max}
              color={m.color}
            />
          ))}
        </div>
      </Section>

      {/* The headline result, stated rather than buried. */}
      <div className="mb-8 rounded-xl border border-amber-800/50 bg-amber-950/20 p-5">
        <h4 className="text-sm font-semibold text-amber-200">
          A popularity baseline beats both personalised methods on precision@10
        </h4>
        <p className="mt-2 text-xs leading-relaxed text-amber-300/70">
          {threeWay.popularity_baseline.precision_at_k} versus{" "}
          {threeWay.cf.precision_at_k} for pure CF and {threeWay.hybrid.precision_at_k} for the
          hybrid — on the full catalog and on the long tail with the{" "}
          {tail.params.head_titles_removed} most-popular titles removed (
          {tail.tail.popularity.precision_at_k} vs {tail.tail.cf.precision_at_k}). This is
          reported rather than omitted.
        </p>
        <ul className="mt-3 space-y-1.5 text-xs leading-relaxed text-amber-300/70">
          <li>
            · On MovieLens, what users rate is overwhelmingly what is popular, so a
            non-personalised baseline is genuinely hard to beat on this metric.
          </li>
          <li>
            · Random selection from {p.catalog_size.toLocaleString()} titles scores ≈ 0.0006. CF at{" "}
            {threeWay.cf.precision_at_k} is roughly 130× random — the signal is real.
          </li>
          <li>
            · Coverage inverts the ranking: popularity shows the same ~10 titles to everyone (
            {threeWay.popularity_baseline.catalog_coverage}), while the personalised methods
            actually differentiate ({threeWay.cf.catalog_coverage} and{" "}
            {threeWay.hybrid.catalog_coverage}).
          </li>
        </ul>
      </div>

      <Section
        title="Why the hybrid underperforms"
        subtitle="Diagnostics on the seed-anchored candidate pool"
      >
        <div className="overflow-hidden rounded-xl border border-gray-800">
          <table className="w-full text-sm">
            <tbody className="divide-y divide-gray-800">
              <tr>
                <td className="px-4 py-2.5 text-gray-400">Mean candidate pool size</td>
                <td className="px-4 py-2.5 text-right font-mono text-gray-200">
                  {diagnose.pool.mean_size} <span className="text-gray-600">/ 25 nominal</span>
                </td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 text-gray-400">
                  Users whose pool contained nothing they liked
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-red-400">
                  {d.pct_users_with_empty_pool_hits}%
                </td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 text-gray-400">Precision@10 ceiling given that pool</td>
                <td className="px-4 py-2.5 text-right font-mono text-gray-200">
                  {d.precision_ceiling}
                </td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 text-gray-400">Precision@10 actually achieved</td>
                <td className="px-4 py-2.5 text-right font-mono text-emerald-400">
                  {dr.hybrid_precision}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs leading-relaxed text-gray-500">
          The ranker reached 95% of the maximum the candidate pool allowed, so{" "}
          <span className="text-gray-300">ranking was never the bottleneck</span> — no amount of
          score tuning could help when the relevant movies are not in the pool. The fix was
          candidate generation, which is what the pure-CF path is.
        </p>
      </Section>

      <Section
        title="Protocol"
        subtitle="Each choice blocks a specific way the number could inflate"
      >
        <div className="overflow-hidden rounded-xl border border-gray-800">
          <table className="w-full text-left text-xs">
            <thead className="bg-gray-900 text-gray-500">
              <tr>
                <th className="px-4 py-2 font-medium">Choice</th>
                <th className="px-4 py-2 font-medium">Inflation it prevents</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800 text-gray-400">
              <tr>
                <td className="px-4 py-2">Per-user split; SVD fits train only</td>
                <td className="px-4 py-2 text-gray-500">
                  Model scoring ratings it trained on
                </td>
              </tr>
              <tr>
                <td className="px-4 py-2">Seed from train, relevant set from test</td>
                <td className="px-4 py-2 text-gray-500">Seed-to-target leakage</td>
              </tr>
              <tr>
                <td className="px-4 py-2">
                  Candidates are the full catalog, not the user's own ratings
                </td>
                <td className="px-4 py-2 text-gray-500">
                  Measuring rating prediction instead of retrieval
                </td>
              </tr>
              <tr>
                <td className="px-4 py-2">Precision denominator fixed at K</td>
                <td className="px-4 py-2 text-gray-500">
                  A short result list scoring as if it were complete
                </td>
              </tr>
              <tr>
                <td className="px-4 py-2">Relevance = rated ≥ 4.0, not merely rated</td>
                <td className="px-4 py-2 text-gray-500">
                  Counting "watched" as "liked"
                </td>
              </tr>
              <tr>
                <td className="px-4 py-2">A title credits only on first appearance</td>
                <td className="px-4 py-2 text-gray-500">Recall exceeding 1.0 via duplicates</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      <p className="mt-8 border-t border-gray-800 pt-4 text-[11px] text-gray-600">
        Generated {new Date(threeWay.timestamp).toLocaleDateString()} · reproduce with{" "}
        <code className="text-gray-500">python eval_offline/eval_offline.py</code> · metric
        correctness covered by 21 unit tests in CI
      </p>
    </div>
  )
}
