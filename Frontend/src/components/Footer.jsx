// src/components/Footer.jsx
import React from "react";
import { Link } from "react-router-dom";

export default function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="bg-white border-t mt-8">
      <div className="container mx-auto px-4 py-6 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-md bg-gradient-to-r from-indigo-500 to-pink-500 flex items-center justify-center text-white font-bold">MR</div>
          <div>
            <div className="font-semibold">FlickFinder</div>
            <div className="text-xs text-gray-500">All movie data via TMDB • Cached with Redis</div>
          </div>
        </div>

        <nav className="flex gap-4 text-sm" aria-label="Footer">
          <Link to="/" className="text-gray-600 hover:text-gray-900">Home</Link>
          <Link to="/search" className="text-gray-600 hover:text-gray-900">Search</Link>
          <Link to="/watchlist" className="text-gray-600 hover:text-gray-900">Watchlist</Link>
        </nav>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <a
              href="https://www.themoviedb.org/"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="The Movie Database (TMDB)"
              className="text-gray-500 hover:text-gray-800"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L2 7v7c0 5 4 9 10 9s10-4 10-9V7l-10-5zM9 13H7V9h2v4zm8 0h-7V9h7v4z"/></svg>
            </a>
            <a
              href="https://github.com/"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Project repository on GitHub"
              className="text-gray-500 hover:text-gray-800"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56v-2.08c-3.2.7-3.88-1.37-3.88-1.37-.52-1.33-1.27-1.69-1.27-1.69-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.68 1.25 3.33.96.1-.74.4-1.25.72-1.54-2.55-.29-5.23-1.28-5.23-5.71 0-1.26.45-2.29 1.18-3.1-.12-.29-.51-1.47.11-3.06 0 0 .96-.31 3.14 1.18a10.9 10.9 0 0 1 2.86-.39c.97 0 1.95.13 2.86.39C18.9 5.9 19.86 6.21 19.86 6.21c.62 1.59.23 2.77.11 3.06.74.81 1.18 1.84 1.18 3.1 0 4.44-2.69 5.42-5.25 5.7.41.36.77 1.08.77 2.18v3.24c0 .31.21.68.8.56A10.5 10.5 0 0 0 23.5 12c0-6.35-5.15-11.5-11.5-11.5z"/></svg>
            </a>
          </div>
          <div className="text-xs text-gray-500">© {year} FlickFinder</div>
        </div>
      </div>
    </footer>
  );
}