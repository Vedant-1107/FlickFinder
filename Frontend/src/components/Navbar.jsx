// src/components/Navbar.jsx
import React, { useState, useEffect, useRef } from "react";
import { NavLink, useNavigate, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Menu, X, ChevronDown } from "lucide-react";

import logo from '../assets/logo.png';

/**
 * Polished Navbar for FlickFinder (lucide icons)
 * - preserves existing behavior (search -> /search?q=..., login/logout, mobile menu)
 * - uses lucide-react for icons
 */
export default function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();
  const token = localStorage.getItem("token");
  const username = localStorage.getItem("username") || "";
  const [menuOpen, setMenuOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [q, setQ] = useState(new URLSearchParams(location.search).get("q") || "");
  const mobileRef = useRef(null);
  const profileRef = useRef(null);
  const firstMobileLinkRef = useRef(null);

  function handleLogout() {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    setProfileOpen(false);
    navigate("/");
  }

  function onSearchSubmit(e) {
    e?.preventDefault?.();
    const query = q.trim();
    if (!query) return;
    navigate(`/search?q=${encodeURIComponent(query)}`);
    setMenuOpen(false);
  }

  // keep q in sync with URL when location changes
  useEffect(() => {
    setQ(new URLSearchParams(location.search).get("q") || "");
    setMenuOpen(false);
    setProfileOpen(false);
  }, [location.pathname, location.search]);

  // close on outside click & Escape key
  useEffect(() => {
    function onDocClick(e) {
      if (menuOpen && mobileRef.current && !mobileRef.current.contains(e.target)) setMenuOpen(false);
      if (profileOpen && profileRef.current && !profileRef.current.contains(e.target)) setProfileOpen(false);
    }
    function onEsc(e) {
      if (e.key === "Escape") {
        setMenuOpen(false);
        setProfileOpen(false);
      }
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onEsc);
    };
  }, [menuOpen, profileOpen]);

  // focus first mobile link when opening
  useEffect(() => {
    if (menuOpen && firstMobileLinkRef.current) {
      try { firstMobileLinkRef.current.focus(); } catch {}
    }
  }, [menuOpen]);

  const activeClass = "text-indigo-600 font-medium";
  const baseLink = "text-sm text-gray-600 hover:text-gray-900";

  return (
    <header className="bg-white/70 backdrop-blur-md sticky top-0 z-40 border-b border-gray-100">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Left: logo + nav */}
          <div className="flex items-center gap-4">
            <NavLink to="/" className="flex items-center gap-3">
              <img src={logo} alt="FlickFinder Logo" className="w-10 h-10" />
              <span className="font-semibold text-lg text-gray-800">FlickFinder</span>
            </NavLink>

            <nav className="hidden md:flex items-center gap-3" aria-label="Primary">
              <NavLink to="/" className={({ isActive }) => isActive ? activeClass : baseLink}>Home</NavLink>
              <NavLink to="/search" className={({ isActive }) => isActive ? activeClass : baseLink}>Search</NavLink>
              {/* <NavLink to="/watchlist" className={({ isActive }) => isActive ? activeClass : baseLink}>Watchlist</NavLink> */}
              <button
                onClick={() => {
                  const token = localStorage.getItem("token");
                  if (!token) return navigate("/login");
                  navigate("/watchlist");
                }}
                className={baseLink + " bg-transparent border-0 cursor-pointer"}
              >
                Watchlist
              </button>
            </nav>
          </div>

          {/* Center: search (desktop) */}
          <div className="hidden md:flex flex-1 justify-center px-4">
            <form onSubmit={onSearchSubmit} className="w-full max-w-2xl">
              <div className="flex items-center bg-white border border-gray-200 rounded-full shadow-sm overflow-hidden focus-within:ring-2 focus-within:ring-indigo-300">
                <div className="pl-3 pr-2 text-gray-400 flex items-center">
                  <Search className="w-5 h-5" />
                </div>

                <input
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  className="flex-1 px-3 py-2 text-sm outline-none"
                  placeholder="Search movies, actors, keywords..."
                  aria-label="Search movies"
                />

                <button
                  type="submit"
                  className="ml-2 mr-1 px-4 py-2 rounded-full bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 transition"
                >
                  Search
                </button>
              </div>
            </form>
          </div>

          {/* Right: auth / profile / mobile */}
          <div className="flex items-center gap-3">
            {token ? (
              <>
                {/* profile / small avatar */}
                <div className="relative" ref={profileRef}>
                  <button
                    onClick={() => setProfileOpen((s) => !s)}
                    aria-haspopup="true"
                    aria-expanded={profileOpen}
                    className="flex items-center gap-2 px-3 py-1 rounded-full bg-white shadow-sm hover:shadow-md border border-gray-100"
                    title="Account"
                  >
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-500 to-pink-500 text-white flex items-center justify-center font-semibold">
                      {username ? username[0].toUpperCase() : "U"}
                    </div>
                    <span className="hidden sm:inline text-sm text-gray-700">{username || "User"}</span>
                    <ChevronDown className={`w-4 h-4 text-gray-500 transform transition ${profileOpen ? "rotate-180" : ""}`} />
                  </button>

                  <AnimatePresence>
                    {profileOpen && (
                      <motion.div
                        initial={{ opacity: 0, y: -6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        transition={{ duration: 0.12 }}
                        className="absolute right-0 mt-2 w-44 bg-white rounded-lg shadow-lg border border-gray-100 overflow-hidden"
                      >
                        <div className="py-2">
                          <NavLink to="/profile" onClick={() => setProfileOpen(false)} className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50">Profile</NavLink>
                          <NavLink to="/settings" onClick={() => setProfileOpen(false)} className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50">Settings</NavLink>
                          <button onClick={handleLogout} className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50">Logout</button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </>
            ) : (
              <div className="hidden sm:flex items-center gap-3">
                <NavLink to="/login" className={({ isActive }) => isActive ? "text-indigo-600 font-medium" : "text-sm text-gray-600 hover:text-gray-900"}>Login</NavLink>
                <NavLink to="/signup" className="text-sm px-3 py-1 rounded-md bg-green-600 text-white hover:bg-green-700">Signup</NavLink>
              </div>
            )}

            {/* Mobile menu button */}
            <button
              onClick={() => setMenuOpen((s) => !s)}
              aria-label="Open menu"
              aria-expanded={menuOpen}
              aria-controls="mobile-menu"
              className="md:hidden p-2 rounded-lg bg-white border border-gray-100 shadow-sm hover:shadow-md"
            >
              {menuOpen ? <X className="w-5 h-5 text-gray-700" /> : <Menu className="w-5 h-5 text-gray-700" />}
            </button>
          </div>
        </div>

        {/* Mobile dropdown */}
        <AnimatePresence>
          {menuOpen && (
            <motion.div
              id="mobile-menu"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.18 }}
              className="md:hidden pb-4"
            >
              <div ref={mobileRef} className="flex flex-col gap-3 py-3 px-2">
                <form onSubmit={onSearchSubmit} className="px-2">
                  <div className="flex items-center gap-2">
                    <input
                      value={q}
                      onChange={(e) => setQ(e.target.value)}
                      className="flex-1 px-3 py-2 bg-gray-100 rounded border border-transparent text-sm outline-none"
                      placeholder="Search movies..."
                    />
                    <button type="submit" className="px-3 py-2 bg-indigo-600 text-white rounded">Go</button>
                  </div>
                </form>

                <NavLink to="/" ref={firstMobileLinkRef} onClick={() => setMenuOpen(false)} className="px-4 py-2 rounded hover:bg-gray-50 text-gray-700">Home</NavLink>
                <NavLink to="/search" onClick={() => setMenuOpen(false)} className="px-4 py-2 rounded hover:bg-gray-50 text-gray-700">Search</NavLink>
                {/* <NavLink to="/watchlist" onClick={() => setMenuOpen(false)} className="px-4 py-2 rounded hover:bg-gray-50 text-gray-700">Watchlist</NavLink> */}
                <button
                  onClick={() => {
                    const token = localStorage.getItem("token");
                    setMenuOpen(false);
                    if (!token) navigate("/login");
                    else navigate("/watchlist");
                  }}
                  className="px-4 py-2 rounded hover:bg-gray-50 text-gray-700"
                  >
                  Watchlist
                  </button>

                <div className="border-t pt-3 px-4">
                  {token ? (
                    <>
                      <div className="mb-2 text-sm text-gray-700">Signed in as <strong>{username}</strong></div>
                      <NavLink to="/profile" onClick={() => setMenuOpen(false)} className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50">Profile</NavLink>
                      <button
                        onClick={() => { handleLogout(); setMenuOpen(false); }}
                        className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50"
                      >
                        Sign out
                      </button>
                    </>
                  ) : (
                    <div className="flex gap-2 items-center">
                      <NavLink onClick={() => setMenuOpen(false)} to="/login" className="text-sm text-gray-700">Login</NavLink>
                      <NavLink onClick={() => setMenuOpen(false)} to="/signup" className="ml-auto text-sm bg-green-600 text-white px-3 py-1 rounded">Signup</NavLink>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
}
