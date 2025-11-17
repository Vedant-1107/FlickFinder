// src/components/Carousel.jsx
import React, { useRef, useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import MovieCard from "./MovieCard";

export default function Carousel({
  items = [],
  title = "",
  onCardHover,   // optional function: (movieId) => void
  onCardClick,   // optional function: (movie) => void
}) {
  const containerRef = useRef(null);
  const itemRefs = useRef([]);
  const [index, setIndex] = useState(0);
  const [autoplay, setAutoplay] = useState(false);

  // keep refs in sync
  itemRefs.current = items.map((_, i) => itemRefs.current[i] ?? React.createRef());

  const scrollToIndex = useCallback(
    (i) => {
      if (!containerRef.current || !itemRefs.current[i]) return;
      const container = containerRef.current;
      const el = itemRefs.current[i];
      const node = el.current ?? el;
      if (!node) return;
      const left =
        node.offsetLeft - container.offsetLeft - Math.max(0, (container.clientWidth - node.clientWidth) / 2);
      container.scrollTo({ left, behavior: "smooth" });
      setIndex(i);
    },
    [setIndex]
  );

  const next = () => setIndex((s) => Math.min(items.length - 1, s + 1));
  const prev = () => setIndex((s) => Math.max(0, s - 1));

  // sync scroll when index changes
  useEffect(() => {
    scrollToIndex(index);
  }, [index, scrollToIndex]);

  // update index when user scrolls (detect which item is mostly visible)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let raf = null;
    const onScroll = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const cx = container.scrollLeft + container.clientWidth / 2;
        let best = 0;
        let bestDist = Infinity;
        itemRefs.current.forEach((r, i) => {
          const node = r.current ?? r;
          if (!node) return;
          const mid = node.offsetLeft + node.clientWidth / 2;
          const dist = Math.abs(cx - mid);
          if (dist < bestDist) {
            bestDist = dist;
            best = i;
          }
        });
        setIndex(best);
      });
    };
    container.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      container.removeEventListener("scroll", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [items.length]);

  // keyboard navigation
  useEffect(() => {
    const handler = (e) => {
      if (e.key === "ArrowRight") next();
      if (e.key === "ArrowLeft") prev();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [items.length]);

  // autoplay
  useEffect(() => {
    if (!autoplay || items.length <= 1) return;
    const id = setInterval(() => {
      setIndex((s) => (s >= items.length - 1 ? 0 : s + 1));
    }, 3500);
    return () => clearInterval(id);
  }, [autoplay, items.length]);

  if (!items || items.length === 0) return null;

  return (
    <section aria-labelledby={title ? `carousel-${title}` : undefined} className="relative">
      <div className="flex items-center justify-between mb-2">
        {title ? <h3 id={`carousel-${title}`} className="text-lg font-semibold">{title}</h3> : null}
        <div className="flex items-center gap-2">
          <button
            aria-pressed={autoplay}
            onClick={() => setAutoplay((s) => !s)}
            className="text-xs px-2 py-1 border rounded text-gray-600 bg-white hover:bg-gray-50"
            title={autoplay ? "Stop autoplay" : "Start autoplay"}
          >
            {autoplay ? "Autoplay On" : "Autoplay Off"}
          </button>
        </div>
      </div>

      <div className="relative">
        {/* prev/next buttons */}
        <button
          onClick={() => { prev(); }}
          aria-label="Previous"
          className="hidden md:flex items-center justify-center absolute left-0 top-1/2 -translate-y-1/2 z-20 w-10 h-10 rounded-full bg-white shadow text-gray-700 hover:bg-gray-100"
        >
          ‹
        </button>

        <div
          ref={containerRef}
          className="no-scrollbar flex gap-4 overflow-x-auto scroll-smooth py-2 px-1 snap-x snap-mandatory"
          role="list"
          aria-roledescription="carousel"
        >
          {items.map((item, i) => (
            <motion.div
              key={item.id ?? i}
              ref={itemRefs.current[i]}
              className="min-w-[160px] md:min-w-[200px] snap-start"
              whileHover={{ scale: 1.03 }}
              role="listitem"
            >
              <div
                onMouseEnter={() => onCardHover?.(item.id)}
                onClick={() => onCardClick?.(item)}
              >
                <MovieCard
                  movie={item}
                  onMouseEnter={() => onCardHover?.(item.id)}
                  // MovieCard will navigate by default; onCardClick is optional override
                />
              </div>
            </motion.div>
          ))}
        </div>

        <button
          onClick={() => { next(); }}
          aria-label="Next"
          className="hidden md:flex items-center justify-center absolute right-0 top-1/2 -translate-y-1/2 z-20 w-10 h-10 rounded-full bg-white shadow text-gray-700 hover:bg-gray-100"
        >
          ›
        </button>
      </div>

      {/* dots */}
      <div className="mt-3 flex items-center gap-2 justify-center">
        {items.map((_, i) => (
          <button
            key={i}
            onClick={() => setIndex(i)}
            aria-label={`Go to slide ${i + 1}`}
            aria-current={i === index}
            className={`w-2.5 h-2.5 rounded-full ${i === index ? "bg-indigo-600" : "bg-gray-300"}`}
          />
        ))}
      </div>
    </section>
  );
}