// src/components/MovieCard.jsx
import React from "react";
import { Link } from "react-router-dom";

/**
 * MovieCard (presentational)
 *
 * Props:
 *  - movie: object (required)
 *  - onAdd?: (id) => void    // optional: called when watchlist/add button clicked
 *  - onRemove?: (id) => void // optional: remove handler
 *  - onMouseEnter?: (id) => void // optional: prefetch hook
 *  - linkTo?: string | ((id) => string) // optional override for link target
 */
export default function MovieCard({ movie, onAdd, onRemove, onMouseEnter, linkTo }) {
  const posterBase = "https://image.tmdb.org/t/p/";
  const posterPath = movie.poster_path ?? movie.poster ?? null;
  const title = movie.title ?? movie.name ?? "Untitled";
  const year = movie.release_date
    ? String(movie.release_date).slice(0, 4)
    : movie.first_air_date
    ? String(movie.first_air_date).slice(0, 4)
    : "";

  const id = movie.id ?? movie.movie_id ?? null;
  const target = typeof linkTo === "function" ? linkTo(id) : linkTo ?? `/movie/${id}`;

  function handleAdd(e) {
    // prevent navigation when clicking the small action button
    e.preventDefault();
    e.stopPropagation();
    onAdd?.(id);
  }
  function handleRemove(e) {
    e.preventDefault();
    e.stopPropagation();
    onRemove?.(id);
  }

  return (
    <Link
      to={target}
      aria-label={`Open details for ${title}`}
      className="block"
      onMouseEnter={() => onMouseEnter?.(id)}
    >
      <article className="bg-white rounded-md shadow-sm overflow-hidden hover:shadow-lg transition-transform transform hover:-translate-y-0.5 relative">
        <div className="relative bg-gray-100">
          {posterPath ? (
            <picture>
              <source srcSet={`${posterBase}w300${posterPath}`} media="(max-width: 640px)" />
              <source srcSet={`${posterBase}w500${posterPath}`} media="(min-width: 641px)" />
              <img
                src={`${posterBase}w300${posterPath}`}
                alt={title}
                loading="lazy"
                decoding="async"
                className="w-full h-64 object-cover"
                style={{ aspectRatio: "2/3" }}
              />
            </picture>
          ) : (
            <div className="w-full h-64 flex items-center justify-center text-sm text-gray-500 bg-gray-200">
              No image
            </div>
          )}

          {/* rating badge */}
          {movie.vote_average !== undefined && movie.vote_average !== null && (
            <div className="absolute top-2 left-2 bg-black/70 text-white text-xs px-2 py-0.5 rounded">
              {Number(movie.vote_average).toFixed(1)}
            </div>
          )}

          {/* small action button (add/remove) */}
          {(onAdd || onRemove) && (
            <div className="absolute top-2 right-2 flex gap-2">
              {onAdd && (
                <button
                  onClick={handleAdd}
                  aria-label="Add to watchlist"
                  title="Add to watchlist"
                  className="bg-white/90 text-gray-800 px-2 py-1 rounded text-xs shadow-sm hover:bg-white"
                >
                  + Watch
                </button>
              )}
              {onRemove && (
                <button
                  onClick={handleRemove}
                  aria-label="Remove from watchlist"
                  title="Remove from watchlist"
                  className="bg-red-50 text-red-600 px-2 py-1 rounded text-xs shadow-sm hover:bg-red-100"
                >
                  Remove
                </button>
              )}
            </div>
          )}
        </div>

        <div className="p-2">
          <h4 className="font-semibold text-sm leading-tight truncate">{title}</h4>
          <p className="text-xs text-gray-600">{year}</p>
        </div>
      </article>
    </Link>
  );
}