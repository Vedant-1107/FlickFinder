// src/utils/watchlist.js
import api from "../api"; // default axios instance from earlier

// localStorage helpers
function readLocal() {
  try {
    const raw = localStorage.getItem("watchlist");
    const arr = raw ? JSON.parse(raw) : [];
    return Array.from(new Set(arr.map((x) => Number(x)).filter(Boolean)));
  } catch {
    return [];
  }
}
function writeLocal(ids) {
  try {
    localStorage.setItem("watchlist", JSON.stringify(Array.from(new Set(ids.map(Number)))));
  } catch {}
}

// Server helpers (expects JWT in api instance headers)
// These return the axios response.data where possible
async function fetchServer() {
  const res = await api.get(`/api/watchlist`);
  return res?.data ?? [];
}

async function addServer(movieId) {
  const res = await api.post(`/api/watchlist`, { movie_id: Number(movieId) });
  // server may return { message, movie_id } or an updated list; normalize below in callers
  return res?.data ?? res;
}

async function removeServer(movieId) {
  const res = await api.delete(`/api/watchlist/${Number(movieId)}`);
  return res?.data ?? res;
}

// exported unified API
export async function readWatchlist() {
  const token = localStorage.getItem("token");
  if (!token) return readLocal();
  try {
    const data = await fetchServer(); // now returns data (array) reliably
    // data might be:
    // - array of { movie_id, added_at }  -> normalize to ordered ids
    // - array of full movie objects with id property -> normalize to ids
    // - array of ids -> normalize to numbers
    if (!Array.isArray(data)) return readLocal();
    if (data.length === 0) return [];
    const first = data[0];
    if (typeof first === "object") {
      if (first.movie_id !== undefined) return data.map((d) => Number(d.movie_id));
      if (first.id !== undefined) return data.map((d) => Number(d.id));
      // fallback: look for numeric-like values in objects
      return data.map((d) => Number(d.movie_id ?? d.id ?? NaN)).filter(Boolean);
    }
    // assume primitive list (ids)
    return data.map(Number).filter(Boolean);
  } catch (e) {
    // on failure fallback to local
    return readLocal();
  }
}

export async function addToWatchlist(id) {
  const token = localStorage.getItem("token");
  if (!token) {
    const next = Array.from(new Set([...readLocal(), Number(id)]));
    writeLocal(next);
    return next;
  }
  try {
    const r = await addServer(id);
    // server may return updated list or a small message
    if (Array.isArray(r)) {
      // assume list of ids or list-of-objects
      // reuse normalization by calling readWatchlist-like logic
      if (r.length === 0) return [];
      const first = r[0];
      if (typeof first === "object") {
        if (first.movie_id !== undefined) return r.map((d) => Number(d.movie_id));
        if (first.id !== undefined) return r.map((d) => Number(d.id));
      }
      return r.map(Number).filter(Boolean);
    }
    // If server returned a single object message, fallback to re-fetch
    return await readWatchlist();
  } catch (err) {
    // fallback: update local optimistic
    const next = Array.from(new Set([...readLocal(), Number(id)]));
    writeLocal(next);
    return next;
  }
}

export async function removeFromWatchlist(id) {
  const token = localStorage.getItem("token");
  if (!token) {
    const next = readLocal().filter((x) => x !== Number(id));
    writeLocal(next);
    return next;
  }
  try {
    const r = await removeServer(id);
    if (Array.isArray(r)) {
      if (r.length === 0) return [];
      const first = r[0];
      if (typeof first === "object") {
        if (first.movie_id !== undefined) return r.map((d) => Number(d.movie_id));
        if (first.id !== undefined) return r.map((d) => Number(d.id));
      }
      return r.map(Number).filter(Boolean);
    }
    // server returned message -> re-fetch canonical list
    return await readWatchlist();
  } catch (err) {
    // fallback local
    const next = readLocal().filter((x) => x !== Number(id));
    writeLocal(next);
    return next;
  }
}

// utility to allow Watchlist.jsx to clear local store or write custom list
export function writeWatchlist(ids) {
  writeLocal(ids || []);
}
