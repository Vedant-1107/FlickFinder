// src/hooks/useMovies.js
import { useQuery } from "@tanstack/react-query";
import {
  fetchTrending,
  fetchTop,
  fetchUpcoming,
  fetchMovie,
  fetchSearch,
  fetchRecommendations,
} from "../api";

/**
 * Small consistent options applied to all queries:
 * - retry: 1 (one automatic retry)
 * - refetchOnWindowFocus: false (safer for UX)
 */
const COMMON_QUERY_OPTIONS = {
  retry: 1,
  refetchOnWindowFocus: false,
};

export function useTrending() {
  return useQuery({
    queryKey: ["trending"],
    queryFn: async () => await fetchTrending(),
    staleTime: 1000 * 60 * 10, // 10 minutes
    ...COMMON_QUERY_OPTIONS,
  });
}

export function useTop() {
  return useQuery({
    queryKey: ["top"],
    queryFn: async () => await fetchTop(),
    staleTime: 1000 * 60 * 30, // 30 minutes
    ...COMMON_QUERY_OPTIONS,
  });
}

export function useUpcoming() {
  return useQuery({
    queryKey: ["upcoming"],
    queryFn: async () => await fetchUpcoming(),
    staleTime: 1000 * 60 * 30,
    ...COMMON_QUERY_OPTIONS,
  });
}

export function useMovie(id) {
  return useQuery({
    queryKey: ["movie", id],
    queryFn: async () => await fetchMovie(id),
    staleTime: 1000 * 60 * 60 * 24, // 24 hours
    enabled: !!id,
    ...COMMON_QUERY_OPTIONS,
  });
}

/**
 * useSearch:
 * - keepPreviousData true helps pagination UI remain smooth when changing pages
 * - enabled only when q is a non-empty string to avoid accidental queries
 */
export function useSearch(q, page = 1) {
  const isEnabled = typeof q === "string" && q.trim().length > 0;

  return useQuery({
    queryKey: ["search", q, page],
    queryFn: async () => await fetchSearch(q, page),
    enabled: isEnabled,
    keepPreviousData: true,
    ...COMMON_QUERY_OPTIONS,
  });
}

export function useRecommendations(id) {
  return useQuery({
    queryKey: ["rec", id],
    queryFn: async () => await fetchRecommendations(id),
    enabled: !!id,
    ...COMMON_QUERY_OPTIONS,
  });
}