// src/api/index.js
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 12_000, // 12s timeout to avoid hanging requests
  headers: { "Content-Type": "application/json" },
});

// Attach auth token automatically from localStorage (same behaviour as before)
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token && config.headers) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

/**
 * Normalize responses and errors so callers can always expect `res.data`
 * If network error happens, do a single automatic retry.
 */
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    // If cancelled via AbortSignal, rethrow early
    if (axios.isCancel?.(error)) return Promise.reject({ message: "Request cancelled" });

    // Retry once for network errors (no response)
    const config = error.config || {};
    if (!error.response && !config._retry) {
      config._retry = true;
      try {
        return api(config);
      } catch (e) {
        /* fall through to normalized error */
      }
    }

    // Normalize error shape similar to: { message, status, data }
    const normalized = {
      message: "An error occurred",
      status: error?.response?.status ?? null,
      data: error?.response?.data ?? null,
    };

    // Prefer server message if present
    if (normalized.data) {
      if (typeof normalized.data === "string") normalized.message = normalized.data;
      else if (normalized.data.message) normalized.message = normalized.data.message;
    } else if (error.message) {
      normalized.message = error.message;
    }

    return Promise.reject(normalized);
  }
);

function unwrap(res) {
  // keep same behavior: return res.data
  return res?.data;
}

// Exported API functions (same names as your original file)
export async function fetchTrending() {
  const res = await api.get("/api/trending");
  return unwrap(res);
}

export async function fetchTop() {
  const res = await api.get("/api/top");
  return unwrap(res);
}

export async function fetchUpcoming() {
  const res = await api.get("/api/upcoming");
  return unwrap(res);
}

export async function fetchMovie(id) {
  const res = await api.get(`/api/movie/${id}`);
  return unwrap(res);
}

export async function fetchSearch(q, page = 1) {
  // allow optional AbortSignal via last arg in case callers pass config later
  const res = await api.get("/api/search", { params: { q, page } });
  return unwrap(res);
}

export async function fetchRecommendations(id) {
  const res = await api.get(`/api/recommendations/${id}`);
  return unwrap(res);
}

// Auth
export async function signup(data) {
  const res = await api.post("/auth/signup", data);
  return unwrap(res);
}
export async function login(credentials) {
  const res = await api.post("/auth/login", credentials);
  return unwrap(res);
}
export async function me() {
  const res = await api.get("/auth/me");
  return unwrap(res);
}
export async function fetchWatchlist() {
  const res = await api.get("/api/watchlist");
  return unwrap(res); // expects array of items or empty array
}

/**
 * Add a movie id to the current user's watchlist.
 * Accepts numeric id or string that can be parsed to a number.
 */
export async function addToWatchlist(movieId) {
  const payload = { movie_id: Number(movieId) };
  const res = await api.post("/api/watchlist", payload);
  return unwrap(res);
}

/**
 * Remove a movie from the current user's watchlist.
 * movieId can be number or string.
 */
export async function removeFromWatchlist(movieId) {
  const res = await api.delete(`/api/watchlist/${Number(movieId)}`);
  return unwrap(res);
}

export default api;