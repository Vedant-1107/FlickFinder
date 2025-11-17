// src/pages/Home.jsx
import React from "react";
import { useNavigate } from 'react-router-dom';
import { Film, TrendingUp, Star, Calendar } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useTrending, useTop, useUpcoming } from "../hooks/useMovies";
import { fetchMovie, fetchRecommendations } from "../api";
import Carousel from "../components/Carousel";
import MovieCard from "../components/MovieCard";

export default function Home() {
  const trending = useTrending();
  const top = useTop();
  const upcoming = useUpcoming();
  const qc = useQueryClient();
  const navigate = useNavigate();

  const prefetchMovie = async (id) => {
    if (!id) return;
    qc.prefetchQuery(["movie", Number(id)], () => fetchMovie(Number(id)), {
      staleTime: 1000 * 60 * 30,
    });
    qc.prefetchQuery(["rec", Number(id)], () => fetchRecommendations(Number(id)), {
      staleTime: 1000 * 60 * 30,
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-purple-50 to-blue-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 text-white">
        <div className="container mx-auto px-4 py-16">
          <div className="max-w-3xl">
            <h1 className="text-5xl font-bold mb-4 animate-fade-in">
              Discover Your Next Favorite Movie
            </h1>
            <p className="text-xl text-purple-100 mb-8">
              Explore trending films, top-rated classics, and upcoming releases all in one place
            </p>
            <div className="flex gap-4">
              <a href="#trending" className="bg-white text-purple-600 px-6 py-3 rounded-full font-semibold hover:bg-purple-50 transition-colors shadow-lg">
                Browse Movies
              </a>
              <a
                onClick={(e) => {                  
                  const token = localStorage.getItem("token");
                  if (!token) return navigate("/login");
                  navigate("/watchlist");
                }}
                className="bg-purple-500/30 backdrop-blur-sm text-white px-6 py-3 rounded-full font-semibold hover:bg-purple-500/50 transition-colors border border-white/30 cursor-pointer"
              >
                My Watchlist
              </a>
            </div>
          </div>
        </div>
      </div>

      <main className="container mx-auto px-4 py-8">
        {/* Trending Section */}
        <section id="trending" className="mb-12">
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-gradient-to-r from-red-500 to-orange-500 p-2 rounded-lg">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-3xl font-bold text-gray-800">Trending Now</h2>
          </div>

          {trending.isLoading ? (
            <div className="flex gap-4 overflow-x-auto pb-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="flex-shrink-0 w-48 h-72 bg-gray-200 animate-pulse rounded-lg" />
              ))}
            </div>
          ) : trending.isError ? (
            <div className="py-8 px-6 bg-red-50 border border-red-200 rounded-lg text-red-600">
              Failed to load trending movies. Please try again later.
            </div>
          ) : (
            <Carousel
              items={trending.data?.results ?? []}
              title="Trending"
              // prefetch on hover of MovieCard inside Carousel via container's mouse events
            />
          )}
        </section>

        {/* Top Rated Section */}
        <section className="mb-12">
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-gradient-to-r from-yellow-500 to-amber-500 p-2 rounded-lg">
              <Star className="w-6 h-6 text-white fill-current" />
            </div>
            <h2 className="text-3xl font-bold text-gray-800">Top Rated</h2>
          </div>

          {top.isLoading ? (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {Array.from({ length: 10 }).map((_, i) => (
                <div key={i} className="h-64 bg-gray-200 animate-pulse rounded-lg" />
              ))}
            </div>
          ) : top.isError ? (
            <div className="py-8 px-6 bg-red-50 border border-red-200 rounded-lg text-red-600">
              Failed to load top rated movies.
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {(top.data?.results ?? []).slice(0, 10).map((m) => (
                <div key={m.id} onMouseEnter={() => prefetchMovie(m.id)}>
                  <MovieCard movie={m} />
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Upcoming Section */}
        <section className="mb-12">
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-gradient-to-r from-blue-500 to-cyan-500 p-2 rounded-lg">
              <Calendar className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-3xl font-bold text-gray-800">Coming Soon</h2>
          </div>

          {upcoming.isLoading ? (
            <div className="flex gap-4 overflow-x-auto pb-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="flex-shrink-0 w-48 h-72 bg-gray-200 animate-pulse rounded-lg" />
              ))}
            </div>
          ) : upcoming.isError ? (
            <div className="py-8 px-6 bg-red-50 border border-red-200 rounded-lg text-red-600">
              Failed to load upcoming movies.
            </div>
          ) : (
            <Carousel items={upcoming.data?.results ?? []} title="Upcoming" />
          )}
        </section>
      </main>
    </div>
  );
}