"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

// Reveal-on-scroll and count-up, set up once for the whole page.
// Content is visible by default (see globals.css); this only enhances it,
// and it stays fully static when the reader prefers reduced motion.
function useMotion() {
  useEffect(() => {
    const root = document.getElementById("about-root");
    if (!root) return;

    const prefersReduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (prefersReduced) return;

    root.classList.add("motion-ready");

    const runCount = (el: HTMLElement) => {
      const target = Number(el.dataset.count);
      const decimals = Number(el.dataset.decimals ?? "0");
      const suffix = el.dataset.suffix ?? "";
      const start = performance.now();
      const duration = 1100;
      const tick = (now: number) => {
        const t = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - t, 4); // ease-out-quart
        const value = target * eased;
        el.textContent = value.toFixed(decimals) + suffix;
        if (t < 1) requestAnimationFrame(tick);
        else el.textContent = target.toFixed(decimals) + suffix;
      };
      requestAnimationFrame(tick);
    };

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const el = entry.target as HTMLElement;
          el.classList.add("in");
          if (el.dataset.count) runCount(el);
          observer.unobserve(el);
        }
      },
      { threshold: 0.2, rootMargin: "0px 0px -8% 0px" },
    );

    root.querySelectorAll<HTMLElement>("[data-reveal]").forEach((el) => {
      observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);
}

function ScanGraphic() {
  // A phone frame with a sweeping scan line and detected spots popping in.
  // Abstract on purpose: no real face, just the idea of an on-device scan.
  const ref = useRef<HTMLDivElement>(null);

  // Only animate while the graphic is on screen, to save the compositor and
  // battery when it is scrolled away.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => el.classList.toggle("paused", !entry.isIntersecting),
      { threshold: 0.05 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div className="scan-graphic" ref={ref} aria-hidden>
      <svg viewBox="0 0 260 300" role="img">
        <defs>
          <linearGradient id="scanline" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(109,152,134,0)" />
            <stop offset="50%" stopColor="rgba(109,152,134,0.55)" />
            <stop offset="100%" stopColor="rgba(109,152,134,0)" />
          </linearGradient>
          <clipPath id="frame">
            <rect x="30" y="20" width="200" height="260" rx="26" />
          </clipPath>
        </defs>

        <rect
          x="30"
          y="20"
          width="200"
          height="260"
          rx="26"
          fill="#ffffff"
          stroke="var(--border)"
          strokeWidth="1.5"
        />

        <g clipPath="url(#frame)">
          <circle cx="130" cy="128" r="52" fill="var(--bg-soft)" />
          <path d="M60 280 Q130 190 200 280 Z" fill="var(--bg-soft)" />

          <g className="scan-spot" style={{ ["--d" as string]: "0.2s" }}>
            <circle cx="108" cy="118" r="5" fill="none" stroke="var(--accent)" strokeWidth="2" />
          </g>
          <g className="scan-spot" style={{ ["--d" as string]: "0.5s" }}>
            <circle cx="150" cy="110" r="5" fill="none" stroke="var(--accent)" strokeWidth="2" />
          </g>
          <g className="scan-spot" style={{ ["--d" as string]: "0.8s" }}>
            <circle cx="140" cy="148" r="5" fill="none" stroke="var(--accent)" strokeWidth="2" />
          </g>
          <g className="scan-spot" style={{ ["--d" as string]: "1.1s" }}>
            <circle cx="116" cy="152" r="5" fill="none" stroke="var(--accent)" strokeWidth="2" />
          </g>

          <rect className="scan-line" x="30" y="0" width="200" height="46" fill="url(#scanline)" />
        </g>

        <g stroke="var(--accent)" strokeWidth="2.5" fill="none" strokeLinecap="round">
          <path d="M46 40 v-2 a6 6 0 0 1 6 -6 h2" />
          <path d="M208 32 h2 a6 6 0 0 1 6 6 v2" />
          <path d="M46 260 v2 a6 6 0 0 0 6 6 h2" />
          <path d="M208 268 h2 a6 6 0 0 0 6 -6 v-2" />
        </g>
      </svg>
    </div>
  );
}

// A live, privacy-safe count of scans run. Hidden entirely until the counter is
// configured and returns a number, so the page never shows a lonely "0".
function ScanCount() {
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/visits")
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (!cancelled && data && typeof data.count === "number") {
          setCount(data.count);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  if (count === null || count <= 0) return null;
  return (
    <p className="scan-count" data-reveal>
      <strong>{count.toLocaleString("en-CA")}</strong> scans run privately so far
    </p>
  );
}

const PIPELINE = [
  {
    title: "Skin type",
    detail: "Dry, normal or oily. A MobileNetV2 network fine-tuned on skin.",
  },
  {
    title: "Acne type",
    detail: "Whiteheads, blackheads, papules, pustules or cysts. Each needs different care.",
  },
  {
    title: "Severity",
    detail: "A YOLOv8 detector finds and counts every spot, then grades how severe it is.",
  },
];

function PipelineIcon({ index }: { index: number }) {
  const common = {
    fill: "none",
    stroke: "var(--accent-dark)",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };
  if (index === 0)
    return (
      <svg viewBox="0 0 24 24" width="26" height="26" {...common}>
        <circle cx="12" cy="12" r="8" />
        <path d="M12 4 a8 8 0 0 1 0 16" fill="var(--bg-soft)" stroke="none" />
      </svg>
    );
  if (index === 1)
    return (
      <svg viewBox="0 0 24 24" width="26" height="26" {...common}>
        <circle cx="8" cy="9" r="2.4" />
        <circle cx="16" cy="8" r="1.6" />
        <circle cx="14" cy="16" r="2.8" />
      </svg>
    );
  return (
    <svg viewBox="0 0 24 24" width="26" height="26" {...common}>
      <rect x="4" y="4" width="7" height="7" rx="1.5" />
      <rect x="13" y="13" width="7" height="7" rx="1.5" />
      <path d="M9 15 h3 M15 6 h4 M17 4 v4" />
    </svg>
  );
}

function Stat({
  count,
  decimals = 0,
  suffix = "",
  prefix = "",
  label,
}: {
  count: number;
  decimals?: number;
  suffix?: string;
  prefix?: string;
  label: string;
}) {
  const initial = count.toFixed(decimals) + suffix;
  return (
    <div className="stat" data-reveal>
      <div className="stat-value">
        {prefix}
        <span data-count={count} data-decimals={decimals} data-suffix={suffix}>
          {initial}
        </span>
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

export default function Home() {
  useMotion();

  return (
    <main id="about-root" className="about">
      <section className="about-hero">
        <div className="about-hero-text" data-reveal>
          <h1>Skin answers should not need an appointment.</h1>
          <p className="lede">
            A dermatologist is expensive, slow to book, and for a lot of teens
            just too embarrassing to face. So most of us guess, following whatever
            skincare trend is loudest online. Acno is a free, private first answer
            that runs entirely on your own device.
          </p>
          <Link href="/scan" className="button about-cta">
            Try a scan
          </Link>
          <ScanCount />
        </div>
        <div data-reveal className="about-hero-visual">
          <ScanGraphic />
        </div>
      </section>

      <section className="about-band" data-reveal>
        <Stat count={85} suffix="%" label="of people aged 12 to 24 deal with acne" />
        <Stat count={0} label="photos uploaded, stored, or used for training" />
        <Stat count={3} label="AI models reading a single photo" />
      </section>

      {/* ---------- Part one: the story ---------- */}

      <div className="part-intro" data-reveal>
        <span className="part-kicker">Part one</span>
        <h2 className="part-title">Why Acno exists</h2>
      </div>

      <section className="about-section story">
        <p className="story-lead" data-reveal>
          Acno started with my own skin.
        </p>
        <p className="about-p" data-reveal>
          When my acne got bad, I did not know what to do, and I worked it out
          slowly, mostly alone, with very little support. A lot of that was fear.
          The stigma around acne made me too embarrassed to ask anyone for help in
          person. What I wanted back then was something genuinely private and safe
          to ask, an answer I could get without anyone watching or judging me.
        </p>
        <p className="about-p" data-reveal>
          I also watched friends go through worse. People were bullied and shamed
          for their skin, even though acne is mostly just puberty doing its work,
          not anything they did or could control. Seeing people I care about get
          hurt for something that was not their fault is what pushed me to build
          this.
        </p>
        <p className="story-emphasis" data-reveal>
          Acno is the private, judgment-free companion I wish I had had.
        </p>
      </section>

      <section className="about-section">
        <h2 className="about-h2" data-reveal>
          Built to help, not to judge
        </h2>
        <p className="about-p" data-reveal>
          That is why Acno is free, private, and open to anyone. Your photo never
          leaves your device, so there is nothing to be embarrassed about and no
          one to face. It is built first for the people who need it most and can
          ask for it least: teens who cannot afford a dermatologist, newcomers, and
          anyone kept away by fear or stigma.
        </p>
        <p className="about-p" data-reveal>
          Instead of the loud, and often wrong, skincare advice online, it gives a
          calm and honest first answer, and it always points serious cases toward a
          real dermatologist. The plan for this grant is to launch Acno publicly,
          translate the guide for newcomer families, and run a school-wide
          awareness campaign about skin health and skincare misinformation. As a
          student government member, I also hope to team up with our school
          government to spread the word further.
        </p>
      </section>

      {/* ---------- Part two: the tech ---------- */}

      <div className="part-intro" data-reveal>
        <span className="part-kicker">Part two</span>
        <h2 className="part-title">How Acno works</h2>
      </div>

      <section className="about-section">
        <h2 className="about-h2" data-reveal>
          One photo, read three ways
        </h2>
        <p className="about-p" data-reveal>
          When you scan, the picture never leaves your browser. Three models look
          at it on your device and hand back one report.
        </p>

        <div className="pipeline">
          {PIPELINE.map((step, index) => (
            <div
              className="pipeline-step"
              data-reveal
              style={{ ["--i" as string]: String(index) }}
              key={step.title}
            >
              <div className="pipeline-icon">
                <PipelineIcon index={index} />
              </div>
              <h3>{step.title}</h3>
              <p>{step.detail}</p>
            </div>
          ))}
        </div>

        <p className="about-p subtle" data-reveal>
          A fourth model, Claude, turns those results into a plain-language guide.
          It only ever sees the numbers, never your photo.
        </p>
      </section>

      <section className="about-section">
        <h2 className="about-h2" data-reveal>
          What the models actually score
        </h2>
        <p className="about-p" data-reveal>
          Measured on faces the models never saw during training. We report the
          weak number too, because pretending it is perfect would be the dishonest
          part.
        </p>
        <div className="metric-row">
          <Stat count={59} suffix="%" label="acne type, across 5 classes (guessing is 20%)" />
          <Stat count={0.67} decimals={2} label="mAP for the spot detector" />
          <Stat count={43} suffix="%" label="skin type, the one we are still fixing" />
        </div>
      </section>

      <section className="about-section">
        <div className="honesty" data-reveal>
          <h2 className="about-h2">The hard part was never the model</h2>
          <p className="about-p">
            It was the data. Rare acne types had four times fewer photos, so early
            models ignored them until we weighted them. The skin type model
            memorized its training set instead of learning, and the graphs caught
            it. And dermatology datasets rarely document skin tone coverage, a
            known bias risk we flag openly and are testing against. Knowing what a
            model cannot do turned out to be most of the work.
          </p>
        </div>
      </section>

      <section className="about-section privacy" data-reveal>
        <div className="privacy-mark" aria-hidden>
          <svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="var(--accent-dark)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3 4 6 v6 c0 5 3.5 7.5 8 9 4.5 -1.5 8 -4 8 -9 V6 Z" />
            <path d="M9 12 l2 2 4 -4" />
          </svg>
        </div>
        <div>
          <h2 className="about-h2">Your photo stays yours</h2>
          <p className="about-p">
            Acno has no server that receives your image. Everything runs in the
            page you are looking at, and when you close the tab the photo is gone.
            Nothing is saved, and nothing is ever used to train anything.
          </p>
        </div>
      </section>

      <section className="about-section">
        <h2 className="about-h2" data-reveal>
          Where this goes next
        </h2>
        <div className="future" data-reveal>
          <span>Weekly re-scans that show a real progress trend</span>
          <span>Products matched to your detected skin type</span>
          <span>A clear handoff to a dermatologist for severe cases</span>
          <span>The same pipeline for eczema and rosacea</span>
        </div>
      </section>

      <section className="about-section team" data-reveal>
        <h2 className="about-h2">Made by five students</h2>
        <p className="about-p">
          Acno was built for an Inspirit AI project by a team who wanted the first
          answer about your own skin to be free, private, and honest about its
          limits.
        </p>
        <div className="team-names">
          {["Travis", "Matthew", "Alan", "Tanner", "Erwin"].map((name) => (
            <span key={name}>{name}</span>
          ))}
        </div>
      </section>

      <section className="about-closing" data-reveal>
        <p className="about-motto">Scan smarter. Skin clearer.</p>
        <p className="about-p">
          A first answer, never a diagnosis. Anything severe is always pointed to a
          real dermatologist.
        </p>
        <Link href="/scan" className="button about-cta">
          Scan your skin
        </Link>
      </section>
    </main>
  );
}
