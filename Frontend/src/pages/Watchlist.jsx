// src/pages/Watchlist.jsx
import React, { useEffect, useState, useMemo } from "react";
import { Film, Star, Trash2, Play, Info, Bookmark, Filter, SortAsc } from "lucide-react";
import { readWatchlist, removeFromWatchlist, writeWatchlist } from "../utils/watchlist";
import { fetchMovie } from "../api";

export default function Watchlist() {
  const [loading, setLoading] = useState(false);
  const [sortBy, setSortBy] = useState("added");
  const [filterGenre, setFilterGenre] = useState("");
  const [movies, setMovies] = useState([]);

  // load watchlist ids and fetch details
  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      const ids = await readWatchlist(); // <-- await here (was missing)
      if (!ids || ids.length === 0) {
        if (!cancelled) setMovies([]);
        setLoading(false);
        return;
      }
      // fetch movie details in parallel but preserve id order via mapping
      const promises = ids.map((id) =>
        fetchMovie(id)
          .then((res) => ({ ok: true, res, id }))
          .catch((err) => ({ ok: false, err, id }))
      );
      const results = await Promise.all(promises);
      if (cancelled) return;
      const ok = results.filter((r) => r.ok).map((r) => r.res);
      // attach added index so we can sort by "added"
      const withAdded = ok.map((m) => {
        // find original index from ids to preserve added order
        const idx = ids.indexOf(Number(m.id ?? m.movie_id ?? m.id));
        return { ...m, addedDate: undefined, __idx: idx >= 0 ? idx : 0 };
      });
      setMovies(withAdded);
      setLoading(false);
    }
    load();
    return () => { cancelled = true; };
  }, []);

  // compute genres for filter UI (simple reliable extraction)
  const genres = useMemo(() => {
    const set = new Set();
    movies.forEach((m) => {
      const list = m.genres ?? [];
      list.forEach((g) => {
        if (!g) return;
        const name = typeof g === "string" ? g : g.name ?? g;
        if (name) set.add(name);
      });
    });
    return ["All", ...Array.from(set)];
  }, [movies]);

  async function handleRemove(id) {
    if (!window.confirm("Remove this movie from your watchlist?")) return;
    // optimistic UI: remove immediately
    setMovies((s) => s.filter((m) => Number(m.id) !== Number(id)));
    try {
      const nextIds = await removeFromWatchlist(id);
      // if server/client returned canonical list, update local storage representation
      if (Array.isArray(nextIds)) {
        writeWatchlist(nextIds);
      } else {
        // ensure local storage sync if server didn't give list
        const fallback = (await readWatchlist()) || [];
        writeWatchlist(fallback);
      }
    } catch (err) {
      // on error we already removed optimistically; you could re-add or show message
      console.error("Failed to remove from server watchlist", err);
    }
  }

  function getSortedMovies() {
    let filtered = filterGenre && filterGenre !== "All"
      ? movies.filter((m) => (m.genres ?? []).some((g) => (g?.name ?? g) === filterGenre))
      : movies;

    return [...filtered].sort((a, b) => {
      if (sortBy === "rating") return (b.vote_average ?? 0) - (a.vote_average ?? 0);
      if (sortBy === "title") return String(a.title || "").localeCompare(String(b.title || ""));
      if (sortBy === "year") return (Number(b.release_date?.slice(0, 4) || 0) - Number(a.release_date?.slice(0, 4) || 0));
      // default: use __idx (higher index = added later if we preserved order)
      return (b.__idx ?? 0) - (a.__idx ?? 0);
    });
  }

  const sortedMovies = getSortedMovies();
  const totalRuntime = movies.reduce((sum, m) => sum + (m.runtime ?? 0), 0);
  const hours = Math.floor(totalRuntime / 60);
  const minutes = totalRuntime % 60;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-purple-50 to-pink-50">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-3 rounded-xl">
              <Bookmark className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold text-gray-800">My Watchlist</h1>
              <p className="text-gray-600 mt-1">
                {movies.length} {movies.length === 1 ? 'movie' : 'movies'} • {hours}h {minutes}m total runtime
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white rounded-xl p-4 shadow-md border border-purple-100">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Total Movies</p>
                  <p className="text-3xl font-bold text-purple-600">{movies.length}</p>
                </div>
                <Film className="w-12 h-12 text-purple-200" />
              </div>
            </div>
            <div className="bg-white rounded-xl p-4 shadow-md border border-yellow-100">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Avg Rating</p>
                  <p className="text-3xl font-bold text-yellow-600">{movies.length ? ((movies.reduce((s, m) => s + (m.vote_average ?? 0), 0) / movies.length).toFixed(1)) : "—"}</p>
                </div>
                <Star className="w-12 h-12 text-yellow-200 fill-current" />
              </div>
            </div>
            <div className="bg-white rounded-xl p-4 shadow-md border border-pink-100">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Watch Time</p>
                  <p className="text-3xl font-bold text-pink-600">{hours}h {minutes}m</p>
                </div>
                <Play className="w-12 h-12 text-pink-200" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl p-4 shadow-md border border-gray-200 flex flex-wrap gap-4 items-center">
            <div className="flex items-center gap-2">
              <Filter className="w-5 h-5 text-gray-600" />
              <span className="text-sm font-semibold text-gray-700">Filter:</span>
              <select value={filterGenre} onChange={(e) => setFilterGenre(e.target.value)} className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 outline-none">
                {genres.map((g) => <option key={g} value={g}>{g}</option>)}
              </select>
            </div>

            <div className="flex items-center gap-2">
              <SortAsc className="w-5 h-5 text-gray-600" />
              <span className="text-sm font-semibold text-gray-700">Sort by:</span>
              <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 outline-none">
                <option value="added">Date Added</option>
                <option value="rating">Rating</option>
                <option value="title">Title</option>
                <option value="year">Year</option>
              </select>
            </div>

            {movies.length > 0 && (
              <button onClick={async () => {
                if (!window.confirm("Clear all movies from watchlist?")) return;
                writeWatchlist([]);
                setMovies([]);
                // attempt server-side clear if logged in (best-effort)
                try {
                  // if you want server-side clear API later, call it here
                } catch (err) { /* ignore */ }
              }} className="ml-auto text-sm text-red-600 hover:text-red-700 font-medium">Clear All</button>
            )}
          </div>
        </div>

        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block w-12 h-12 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin mb-4"></div>
            <p className="text-gray-600 font-medium">Loading watchlist...</p>
          </div>
        ) : sortedMovies.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-32 h-32 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <Bookmark className="w-16 h-16 text-purple-400" />
            </div>
            <h3 className="text-2xl font-bold text-gray-800 mb-2">{filterGenre && filterGenre !== "All" ? `No ${filterGenre} movies in your watchlist` : "Your watchlist is empty"}</h3>
            <p className="text-gray-600 mb-6">{filterGenre && filterGenre !== "All" ? "Try selecting a different genre or add more movies" : "Start adding movies you want to watch"}</p>
            <a href="/" className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all shadow-lg">Discover Movies</a>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sortedMovies.map((movie) => (
              <div key={movie.id} className="bg-white rounded-xl shadow-md hover:shadow-2xl transition-all duration-300 overflow-hidden border border-gray-200 group">
                <div className="flex gap-4 p-4">
                  <div className="w-24 flex-shrink-0">
                    <div className="aspect-[2/3] bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg flex items-center justify-center group-hover:scale-105 transition-transform duration-300">
                      <Film className="w-12 h-12 text-white opacity-50" />
                    </div>
                  </div>

                  <div className="flex-1 min-w-0">
                    <h3 className="font-bold text-gray-800 text-lg mb-1 line-clamp-2 group-hover:text-purple-600 transition-colors">{movie.title}</h3>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="flex items-center gap-1 text-yellow-500">
                        <Star className="w-4 h-4 fill-current" />
                        <span className="text-sm font-semibold">{movie.vote_average}</span>
                      </div>
                      <span className="text-gray-400">•</span>
                      <span className="text-sm text-gray-600">{movie.release_date?.slice(0,4)}</span>
                    </div>
                    <div className="flex items-center gap-2 mb-3">
                      <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-medium rounded">{(movie.genres && movie.genres[0]?.name) || "—"}</span>
                      <span className="text-xs text-gray-500">{movie.runtime ?? "—"} min</span>
                    </div>

                    <div className="flex gap-2">
                      <a href={`/movie/${movie.id}`} className="flex-1 flex items-center justify-center gap-1 px-3 py-1.5 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 transition-colors">
                        <Play className="w-4 h-4" /> Watch
                      </a>
                      <a href={`/movie/${movie.id}`} className="p-1.5 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                        <Info className="w-4 h-4 text-gray-600" />
                      </a>
                      <button onClick={() => handleRemove(movie.id)} className="p-1.5 border border-red-200 rounded-lg hover:bg-red-50 transition-colors">
                        <Trash2 className="w-4 h-4 text-red-600" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {movies.length > 0 && (
          <div className="mt-8 bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl p-6 text-white">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <div>
                <h3 className="text-xl font-bold mb-1">Ready for a movie marathon?</h3>
                <p className="text-purple-100">You have {hours} hours and {minutes} minutes of entertainment waiting!</p>
              </div>
              <button className="px-6 py-3 bg-white text-purple-600 rounded-lg font-semibold hover:bg-purple-50 transition-colors shadow-lg whitespace-nowrap">Start Watching</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
