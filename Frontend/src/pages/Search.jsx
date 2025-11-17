// src/pages/Search.jsx
import React, { useEffect, useState } from "react";
import { Search, Film, Star, X, TrendingUp, SlidersHorizontal } from "lucide-react";
import { useSearch } from "../hooks/useMovies";
import { useLocation, useNavigate } from "react-router-dom";
import MovieCard from "../components/MovieCard";

function useQueryString() {
  const { search } = useLocation();
  return React.useMemo(() => new URLSearchParams(search), [search]);
}

export default function SearchPage() {
  const queryParams = useQueryString();
  const navigate = useNavigate();
  const initialQ = queryParams.get("q") || "";
  const initialPage = Number(queryParams.get("page") || 1);

  const [query, setQuery] = useState(initialQ);
  const [submitted, setSubmitted] = useState(initialQ);
  const [page, setPage] = useState(initialPage);
  const [isLoading, setIsLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedGenre, setSelectedGenre] = useState("");
  const [selectedYear, setSelectedYear] = useState("");

  useEffect(() => {
    setQuery(initialQ);
    setSubmitted(initialQ);
    setPage(initialPage);
  }, [initialQ, initialPage]);

  const { data, isLoading: qLoading, isError } = useSearch(submitted || null, page);

  useEffect(() => {
    setIsLoading(qLoading);
  }, [qLoading]);

  function goToPage(p) {
    setPage(p);
    const params = new URLSearchParams();
    if (submitted) params.set("q", submitted);
    params.set("page", String(p));
    navigate({ pathname: "/search", search: params.toString() });
  }

  function handleSearch() {
    const cleaned = query.trim();
    if (!cleaned) return;
    setSubmitted(cleaned);
    setPage(1);
    const params = new URLSearchParams();
    params.set("q", cleaned);
    params.set("page", "1");
    navigate({ pathname: "/search", search: params.toString() });
  }

  function clearSearch() {
    setQuery("");
    setSubmitted("");
    setPage(1);
    navigate({ pathname: "/search" });
  }

  const results = data?.results ?? [];

  const totalPages = data?.total_pages ?? 1;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-6 text-center">Find Your Perfect Movie</h1>

          <div className="relative mb-4">
            <div className="flex gap-2">
              <div className="flex-1 relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") handleSearch(); }}
                  placeholder="Search for movies, actors, directors..."
                  className="w-full pl-12 pr-12 py-4 rounded-xl border-2 border-gray-200 focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-all outline-none text-lg"
                  aria-label="Search movies"
                />
                {query && (
                  <button onClick={clearSearch} className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors">
                    <X className="w-5 h-5" />
                  </button>
                )}
              </div>
              <button onClick={handleSearch} className="px-8 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-semibold hover:from-purple-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-xl">
                Search
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <button onClick={() => setShowFilters((s) => !s)} className="flex items-center gap-2 text-gray-600 hover:text-purple-600 transition-colors">
              <SlidersHorizontal className="w-5 h-5" />
              <span className="font-medium">Filters</span>
            </button>
            {(selectedGenre || selectedYear) && (
              <button onClick={() => { setSelectedGenre(""); setSelectedYear(""); }} className="text-sm text-purple-600 hover:text-purple-700 font-medium">Clear filters</button>
            )}
          </div>

          {showFilters && (
            <div className="mt-4 p-6 bg-white rounded-xl shadow-lg border border-gray-200">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Genre</label>
                  <select value={selectedGenre} onChange={(e) => setSelectedGenre(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 outline-none">
                    <option value="">All Genres</option>
                    <option>Action</option>
                    <option>Comedy</option>
                    <option>Drama</option>
                    <option>Horror</option>
                    <option>Sci-Fi</option>
                    <option>Thriller</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Year</label>
                  <select value={selectedYear} onChange={(e) => setSelectedYear(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 outline-none">
                    <option value="">All Years</option>
                    <option>2024</option>
                    <option>2023</option>
                    <option>2022</option>
                    <option>2021</option>
                  </select>
                </div>
              </div>
            </div>
          )}

        </div>

        <div className="max-w-6xl mx-auto">
          {isLoading ? (
            <div className="text-center py-12">
              <div className="inline-block w-12 h-12 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin mb-4"></div>
              <p className="text-gray-600 font-medium">Searching movies...</p>
            </div>
          ) : submitted && results.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Search className="w-12 h-12 text-gray-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-700 mb-2">No results found</h3>
              <p className="text-gray-500 mb-4">Try a different search term or adjust your filters</p>
              <button onClick={clearSearch} className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700">Clear Search</button>
            </div>
          ) : submitted ? (
            <>
              <div className="mb-6">
                <p className="text-gray-600">Found <span className="font-semibold text-purple-600">{results.length}</span> results for "{submitted}"</p>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                {results.map((movie) => (
                  <div key={movie.id} className="group cursor-pointer">
                    <a href={`/movie/${movie.id}`} className="block">
                      <div className="aspect-[2/3] bg-gradient-to-br from-purple-500 to-blue-600 rounded-lg shadow-md group-hover:shadow-2xl transition-all duration-300 group-hover:scale-105 flex items-center justify-center relative overflow-hidden">
                        <Film className="w-16 h-16 text-white opacity-50" />
                        <div className="absolute top-2 right-2 bg-black/70 backdrop-blur-sm px-2 py-1 rounded-full flex items-center gap-1">
                          <Star className="w-3 h-3 text-yellow-400 fill-current" />
                          <span className="text-white text-xs font-semibold">{movie.vote_average}</span>
                        </div>
                      </div>
                    </a>
                    <div className="mt-2">
                      <h3 className="font-semibold text-gray-800 line-clamp-1 group-hover:text-purple-600 transition-colors">{movie.title}</h3>
                      <p className="text-sm text-gray-500">{movie.release_date}</p>
                    </div>
                  </div>
                ))}
              </div>

              {totalPages > 1 && (
                <div className="flex items-center justify-center gap-2">
                  <button onClick={() => goToPage(Math.max(1, page - 1))} disabled={page === 1} className="px-4 py-2 rounded-lg border-2 border-gray-300 font-medium hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-all">Previous</button>

                  <div className="flex gap-2">
                    {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                      const pageNum = i + 1;
                      return (
                        <button key={pageNum} onClick={() => goToPage(pageNum)} className={`w-10 h-10 rounded-lg font-semibold transition-all ${page === pageNum ? "bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg" : "border-2 border-gray-300 hover:bg-gray-50"}`}>{pageNum}</button>
                      );
                    })}
                  </div>

                  <button onClick={() => goToPage(Math.min(totalPages, page + 1))} disabled={page === totalPages} className="px-4 py-2 rounded-lg border-2 border-gray-300 font-medium hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-all">Next</button>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <div className="w-24 h-24 bg-gradient-to-br from-purple-100 to-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Search className="w-12 h-12 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-700 mb-2">Start Your Search</h3>
              <p className="text-gray-500">Enter a movie title, actor, or director to begin</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}