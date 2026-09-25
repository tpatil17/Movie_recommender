export default function MovieCard({ movie, rank, showScore = true }) {
  return (
    <div className="flex items-center justify-between rounded-xl border border-gray-800 bg-gray-900 p-4 transition-colors hover:border-gray-600">
      <div className="flex min-w-0 items-center gap-4">
        <span className="w-6 shrink-0 text-xl font-bold text-gray-700">{rank}</span>
        <div className="min-w-0">
          <h3 className="truncate font-semibold text-white">{movie.title}</h3>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            <span className="rounded-full border border-gray-700 bg-gray-800 px-2.5 py-0.5 text-[11px] text-gray-400">
              {movie.reason}
            </span>
            {movie.genres?.slice(0, 2).map((g) => (
              <span key={g} className="text-[11px] capitalize text-gray-600">
                {g}
              </span>
            ))}
          </div>
        </div>
      </div>
      {showScore && (
        <div className="ml-3 shrink-0 text-right">
          <div className="font-mono text-lg font-bold text-yellow-400">
            {movie.predicted_rating?.toFixed(2)}
          </div>
          <div className="text-[10px] text-gray-500">predicted</div>
        </div>
      )}
    </div>
  )
}
