// src/pages/MovieDetail.jsx
import React, { useEffect, useMemo, useState } from "react";
import { Star, Clock, Calendar, Plus, Check, Play, Heart, Share2, Film } from "lucide-react";
import { useParams, useNavigate } from "react-router-dom";
import { useMovie, useRecommendations } from "../hooks/useMovies";
import { addToWatchlist, removeFromWatchlist, readWatchlist } from "../utils/watchlist";
import MovieCard from "../components/MovieCard";

export default function MovieDetail() {
  const { id } = useParams();
  const numericId = Number(id);
  const navigate = useNavigate();

  const { data, isLoading, isError } = useMovie(numericId);
  const recs = useRecommendations(numericId);

  const [adding, setAdding] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [watchlistIds, setWatchlistIds] = useState([]);

  // Load watchlist IDs on mount and when `id` changes (so UI reflects latest server/local state)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const ids = await readWatchlist();
        if (!cancelled) setWatchlistIds((ids || []).map((x) => Number(x)));
      } catch (e) {
        if (!cancelled) setWatchlistIds([]);
      }
    })();
    return () => { cancelled = true; };
  }, [id]);

  // compute membership from the state-managed list
  const inWatchlist = watchlistIds.includes(numericId);

  async function handleAddToWatchlist() {
    // if not logged in, send to login
    const token = localStorage.getItem("token");
    if (!token) {
      navigate("/login");
      return;
    }

    if (inWatchlist) return;
    setAdding(true);
    try {
      const next = await addToWatchlist(numericId);
      // normalize result:
      let ids;
      if (Array.isArray(next)) {
        // next is likely an array of ids or array of objects
        if (next.length > 0 && typeof next[0] === "object") {
          // objects with movie_id or id
          ids = next.map((it) => Number(it.movie_id ?? it.id ?? it));
        } else {
          ids = next.map(Number);
        }
      } else {
        // not an array, re-read from storage/server
        ids = await readWatchlist();
      }
      setWatchlistIds(Array.from(new Set((ids || []).map(Number))));
    } catch (e) {
      console.error("addToWatchlist error:", e);
      alert("Failed to add to watchlist");
    } finally {
      setAdding(false);
    }
  }

  async function handleRemoveFromWatchlist() {
    const token = localStorage.getItem("token");
    if (!token) {
      navigate("/login");
      return;
    }

    if (!inWatchlist) return;
    setRemoving(true);
    try {
      const next = await removeFromWatchlist(numericId);
      let ids;
      if (Array.isArray(next)) {
        if (next.length > 0 && typeof next[0] === "object") {
          ids = next.map((it) => Number(it.movie_id ?? it.id ?? it));
        } else {
          ids = next.map(Number);
        }
      } else {
        ids = await readWatchlist();
      }
      setWatchlistIds(Array.from(new Set((ids || []).map(Number))));
    } catch (e) {
      console.error("removeFromWatchlist error:", e);
      alert("Failed to remove from watchlist");
    } finally {
      setRemoving(false);
    }
  }

  if (isLoading) return <div className="p-4">Loading movie...</div>;
  if (isError) return <div className="p-4 text-red-600">Failed to load movie.</div>;
  if (!data) return <div className="p-4">No data available.</div>;

  const movie = data;
  const posterBase = "https://image.tmdb.org/t/p/w500";
  const trailer = movie.videos?.results?.find((v) => v.site === "YouTube" && v.type === "Trailer");

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-gray-900 to-gray-900 z-10"></div>
        <div className="h-[420px] md:h-[500px] bg-[url('https://image.tmdb.org/t/p/original' + (movie.backdrop_path || movie.poster_path))] bg-cover bg-center opacity-30" />
        <div className="absolute inset-0 z-20 container mx-auto px-4 flex items-end pb-12">
          <div className="flex flex-col md:flex-row gap-8 w-full">
            <div className="w-64 flex-shrink-0">
              {movie.poster_path ? (
                <img src={`${posterBase}${movie.poster_path}`} alt={movie.title} className="w-full rounded-xl shadow-2xl" loading="lazy" />
              ) : (
                <div className="aspect-[2/3] bg-gradient-to-br from-purple-600 to-blue-600 rounded-xl shadow-2xl flex items-center justify-center">
                  <Film className="w-24 h-24 text-white opacity-50" />
                </div>
              )}
            </div>

            <div className="flex-1">
              <h1 className="text-4xl md:text-5xl font-bold mb-3">{movie.title}</h1>
              {movie.tagline && <p className="text-xl text-gray-300 italic mb-4">"{movie.tagline}"</p>}

              <div className="flex flex-wrap items-center gap-4 mb-6">
                <div className="flex items-center gap-2 bg-yellow-500 text-black px-3 py-1.5 rounded-full font-bold">
                  <Star className="w-5 h-5 fill-current" />
                  <span>{Number(movie.vote_average ?? movie.vote_average).toFixed(1)}</span>
                </div>
                <div className="flex items-center gap-2 text-gray-300">
                  <Clock className="w-5 h-5" />
                  <span>{movie.runtime ?? movie.runtime} min</span>
                </div>
                <div className="flex items-center gap-2 text-gray-300">
                  <Calendar className="w-5 h-5" />
                  <span>{movie.release_date ? new Date(movie.release_date).getFullYear() : ""}</span>
                </div>
              </div>

              <div className="flex gap-3 mb-6">
                <button
                  onClick={inWatchlist ? handleRemoveFromWatchlist : handleAddToWatchlist}
                  disabled={adding || removing}
                  className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all duration-300 ${
                    inWatchlist ? "bg-green-600 text-white" : "bg-yellow-500 text-black hover:bg-yellow-400"
                  } disabled:opacity-50`}
                >
                  {inWatchlist ? (
                    <>
                      <Check className="w-5 h-5" />
                      {removing ? "Removing..." : "In Watchlist"}
                    </>
                  ) : adding ? (
                    <>
                      <div className="w-5 h-5 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
                      Adding...
                    </>
                  ) : (
                    <>
                      <Plus className="w-5 h-5" />
                      Add to Watchlist
                    </>
                  )}
                </button>

                <button className="flex items-center gap-2 px-6 py-3 bg-white/10 backdrop-blur-sm rounded-lg font-semibold hover:bg-white/20 transition-all">
                  <Play className="w-5 h-5" />
                  Watch Trailer
                </button>

                <button className="p-3 bg-white/10 backdrop-blur-sm rounded-lg hover:bg-white/20 transition-all" title="Favorite">
                  <Heart className="w-5 h-5" />
                </button>

                <button className="p-3 bg-white/10 backdrop-blur-sm rounded-lg hover:bg-white/20 transition-all" title="Share">
                  <Share2 className="w-5 h-5" />
                </button>
              </div>

              <div className="flex gap-2">
                {movie.genres?.map((genre) => (
                  <span key={genre.id} className="px-3 py-1 bg-white/10 backdrop-blur-sm rounded-full text-sm border border-white/20">
                    {genre.name}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-12">
        <div className="grid md:grid-cols-3 gap-8">
          <div className="md:col-span-2 space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <div className="w-1 h-6 bg-purple-500 rounded" />
                Overview
              </h2>
              <p className="text-gray-300 text-lg leading-relaxed">{movie.overview}</p>
            </section>

            <section>
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <div className="w-1 h-6 bg-blue-500 rounded" />
                Cast
              </h2>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {movie.credits?.cast?.slice(0, 8).map((actor) => (
                  <div key={actor.cast_id ?? actor.credit_id ?? actor.id} className="bg-white/5 backdrop-blur-sm rounded-lg p-4 hover:bg-white/10 transition-all cursor-pointer border border-white/10">
                    <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full mx-auto mb-3 flex items-center justify-center">
                      <span className="text-2xl font-bold">{(actor.name || " ")[0]}</span>
                    </div>
                    <p className="font-semibold text-center text-sm mb-1">{actor.name}</p>
                    <p className="text-gray-400 text-center text-xs">{actor.character}</p>
                  </div>
                ))}
              </div>
            </section>

            {trailer && (
              <section>
                <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                  <div className="w-1 h-6 bg-red-500 rounded" />
                  Trailer
                </h2>
                <div className="aspect-video bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl flex items-center justify-center border border-white/10">
                  <iframe
                    loading="lazy"
                    src={`https://www.youtube.com/embed/${trailer.key}`}
                    title={`${movie.title} trailer`}
                    allowFullScreen
                    className="w-full h-full rounded"
                  />
                </div>
              </section>
            )}
          </div>

          <aside className="space-y-6">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <h3 className="font-bold text-lg mb-4">Movie Stats</h3>
              <div className="space-y-3">
                <div>
                  <p className="text-gray-400 text-sm">Status</p>
                  <p className="font-semibold">{movie.status ?? "Released"}</p>
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Original Language</p>
                  <p className="font-semibold">{movie.original_language ?? "en"}</p>
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Budget</p>
                  <p className="font-semibold">{movie.budget ? `$${movie.budget.toLocaleString()}` : "—"}</p>
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Revenue</p>
                  <p className="font-semibold">{movie.revenue ? `$${movie.revenue.toLocaleString()}` : "—"}</p>
                </div>
              </div>
            </div>

            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <h3 className="font-bold text-lg mb-4">Keywords</h3>
              <div className="flex flex-wrap gap-2">
                {(movie.keywords?.keywords ?? movie.keywords ?? []).slice(0, 12).map((kw) => {
                  const text = typeof kw === "string" ? kw : kw.name ?? kw.keyword ?? "";
                  return (
                    <span key={text} className="px-3 py-1 bg-white/10 rounded-full text-sm hover:bg-white/20 transition-colors cursor-pointer">
                      {text}
                    </span>
                  );
                })}
              </div>
            </div>
          </aside>
        </div>

        <section className="mt-12">
          <h2 className="text-3xl font-bold mb-6 flex items-center gap-2">
            <div className="w-1 h-8 bg-gradient-to-b from-purple-500 to-blue-500 rounded"></div>
            You May Also Like
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {(recs.data?.results ?? []).map((rec) => (
              <div key={rec.id} className="group cursor-pointer">
                <div className="aspect-[2/3] bg-gradient-to-br from-purple-600 to-blue-600 rounded-lg flex items-center justify-center group-hover:scale-105 transition-transform duration-300 shadow-lg">
                  <Film className="w-16 h-16 text-white opacity-50" />
                </div>
                <div className="mt-2">
                  <h3 className="font-semibold group-hover:text-purple-400 transition-colors">{rec.title}</h3>
                  <div className="flex items-center gap-1 text-sm text-yellow-400">
                    <Star className="w-4 h-4 fill-current" />
                    <span>{rec.vote_average}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}