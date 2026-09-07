"use client";

import { useEffect } from "react";
import Lenis from "lenis";

// Smooth scrolling for the story page. Anyone who asks for reduced motion gets
// the browser's own scrolling, untouched.
export default function SmoothScroll() {
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (query.matches) return;

    const lenis = new Lenis({
      // How long the page keeps gliding after you stop pushing. This is the
      // slipperiness dial: higher feels floatier, lower feels like it is
      // tracking your hand. Keep it under a second or it stops feeling yours.
      duration: 0.72,
      // Gentle exponential ease out. No overshoot, nothing springy.
      easing: (t: number) => 1 - Math.pow(1 - t, 4),
      touchMultiplier: 1.2,
    });

    let frame = 0;
    const raf = (time: number) => {
      lenis.raf(time);
      frame = requestAnimationFrame(raf);
    };
    frame = requestAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(frame);
      lenis.destroy();
    };
  }, []);

  return null;
}
