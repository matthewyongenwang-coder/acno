"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

/* --------------------------------------------------------------------------
   Motion setup

   Everything here enhances content that is already visible. If the observer
   never fires (no JS, a headless renderer, a background tab), the page still
   reads correctly; it just does not animate.
   -------------------------------------------------------------------------- */

function useReveals() {
  useEffect(() => {
    const root = document.getElementById("story-root");
    if (!root) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const targets = Array.from(
      root.querySelectorAll<HTMLElement>(
        "[data-reveal], [data-stagger], .line-reveal, .number-bar, [data-count]",
      ),
    );

    // Anything already on screen at load is simply there. Only what is below
    // the fold gets an entrance, so nothing the reader can already see depends
    // on an observer firing.
    for (const el of targets) {
      if (el.getBoundingClientRect().top < window.innerHeight * 0.9) {
        el.classList.add("in", "lit", "filled");
      }
    }

    root.classList.add("motion-ready");

    // Last resort: if the observer never fires (background tab, headless
    // renderer, a browser that pauses transitions), show everything anyway.
    const failsafe = window.setTimeout(() => {
      for (const el of targets) el.classList.add("in", "lit", "filled");
    }, 4000);

    const countUp = (el: HTMLElement) => {
      const target = Number(el.dataset.count);
      const decimals = Number(el.dataset.decimals ?? "0");
      const suffix = el.dataset.suffix ?? "";
      const start = performance.now();
      const duration = 1300;
      const tick = (now: number) => {
        const t = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - t, 5);
        el.textContent = (target * eased).toFixed(decimals) + suffix;
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
          el.classList.add("in", "lit", "filled");
          if (el.dataset.count) countUp(el);
          observer.unobserve(el);
        }
      },
      { threshold: 0.25, rootMargin: "0px 0px -10% 0px" },
    );

    for (const el of targets) {
      if (!el.classList.contains("in")) observer.observe(el);
      else if (el.dataset.count) countUp(el);
    }

    return () => {
      observer.disconnect();
      window.clearTimeout(failsafe);
    };
  }, []);
}

// The header borrows the palette of whatever half of the page is behind it.
function useHeaderTheme() {
  useEffect(() => {
    const header = document.querySelector(".site-header");
    const sentinel = document.getElementById("daybreak");
    if (!header || !sentinel) return;

    header.classList.add("night");

    const observer = new IntersectionObserver(
      ([entry]) => {
        header.classList.toggle("night", entry.boundingClientRect.top > 0);
        header.classList.toggle("scrolled", entry.boundingClientRect.top < 200);
      },
      { threshold: 0, rootMargin: "-64px 0px 0px 0px" },
    );
    observer.observe(sentinel);

    return () => {
      observer.disconnect();
      header.classList.remove("night", "scrolled");
    };
  }, []);
}

/* --------------------------------------------------------------------------
   The device: a phone scanning in a dark room
   -------------------------------------------------------------------------- */

function Device() {
  const ref = useRef<HTMLDivElement>(null);

  // Stop the loop while it is off screen, to spare the compositor.
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
    <div className="device" ref={ref} aria-hidden>
      <svg viewBox="0 0 280 320" role="img">
        <defs>
          <linearGradient id="sweep" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="oklch(0.8 0.11 162 / 0)" />
            <stop offset="50%" stopColor="oklch(0.8 0.11 162 / 0.7)" />
            <stop offset="100%" stopColor="oklch(0.8 0.11 162 / 0)" />
          </linearGradient>
          <clipPath id="screen">
            <rect x="40" y="20" width="200" height="280" rx="30" />
          </clipPath>
        </defs>

        <rect
          x="40"
          y="20"
          width="200"
          height="280"
          rx="30"
          fill="oklch(0.22 0.014 64)"
          stroke="oklch(0.36 0.02 70)"
          strokeWidth="1.5"
        />

        <g clipPath="url(#screen)">
          {/* an abstract face, never a real one */}
          <circle cx="140" cy="140" r="58" fill="oklch(0.27 0.016 64)" />
          <path d="M70 300 Q140 200 210 300 Z" fill="oklch(0.27 0.016 64)" />

          {[
            { x: 116, y: 128, d: "0.1s" },
            { x: 163, y: 120, d: "0.45s" },
            { x: 152, y: 162, d: "0.8s" },
            { x: 124, y: 166, d: "1.15s" },
          ].map((spot) => (
            <g
              className="scan-spot"
              key={`${spot.x}-${spot.y}`}
              style={{ ["--d" as string]: spot.d }}
            >
              <circle
                cx={spot.x}
                cy={spot.y}
                r="6"
                fill="none"
                stroke="oklch(0.85 0.1 162)"
                strokeWidth="2"
              />
            </g>
          ))}

          <rect
            className="scan-line"
            x="40"
            y="0"
            width="200"
            height="52"
            fill="url(#sweep)"
          />
        </g>

        <g
          stroke="oklch(0.8 0.095 162)"
          strokeWidth="2.5"
          fill="none"
          strokeLinecap="round"
        >
          <path d="M58 44 v-4 a8 8 0 0 1 8 -8 h4" />
          <path d="M214 32 h4 a8 8 0 0 1 8 8 v4" />
          <path d="M58 276 v4 a8 8 0 0 0 8 8 h4" />
          <path d="M214 288 h4 a8 8 0 0 0 8 -8 v-4" />
        </g>
      </svg>
    </div>
  );
}

/* --------------------------------------------------------------------------
   One photo, read three ways
   -------------------------------------------------------------------------- */

const READS = [
  {
    title: "Skin type",
    detail:
      "Dry, normal, or oily. This is the reading we are still fighting with, and we say so below.",
  },
  {
    title: "Acne type",
    detail:
      "Whiteheads, blackheads, papules, pustules, or cysts. Each one wants a different ingredient.",
  },
  {
    title: "How much",
    detail:
      "A detector finds every individual spot, counts them, and turns the count into a severity.",
  },
];

function FaceRead({ active }: { active: number }) {
  return (
    <div className="read-face" aria-hidden>
      <svg viewBox="0 0 260 300" role="img">
        <rect
          x="10"
          y="10"
          width="240"
          height="280"
          rx="26"
          fill="var(--surface)"
          stroke="var(--border)"
          strokeWidth="1.5"
        />
        <circle cx="130" cy="132" r="62" fill="var(--bg-soft)" />
        <path d="M56 290 Q130 190 204 290 Z" fill="var(--bg-soft)" />

        {/* 0: skin type, a wash across the whole face */}
        <g className={`read-layer ${active === 0 ? "on" : ""}`}>
          <circle
            cx="130"
            cy="132"
            r="62"
            fill="var(--accent)"
            opacity="0.22"
          />
          <text
            x="130"
            y="240"
            textAnchor="middle"
            fontSize="15"
            fontWeight="600"
            fill="var(--accent-strong)"
          >
            oily
          </text>
        </g>

        {/* 1: acne type, each spot named */}
        <g className={`read-layer ${active === 1 ? "on" : ""}`}>
          {[
            { x: 104, y: 118 },
            { x: 156, y: 112 },
            { x: 146, y: 158 },
            { x: 112, y: 160 },
          ].map((p) => (
            <circle
              key={`${p.x}-${p.y}`}
              cx={p.x}
              cy={p.y}
              r="7"
              fill="none"
              stroke="var(--accent-strong)"
              strokeWidth="2.2"
            />
          ))}
          <text
            x="130"
            y="240"
            textAnchor="middle"
            fontSize="15"
            fontWeight="600"
            fill="var(--accent-strong)"
          >
            papules
          </text>
        </g>

        {/* 2: severity, boxed and counted */}
        <g className={`read-layer ${active === 2 ? "on" : ""}`}>
          {[
            { x: 96, y: 108 },
            { x: 148, y: 102 },
            { x: 138, y: 148 },
            { x: 104, y: 150 },
            { x: 128, y: 176 },
          ].map((p) => (
            <rect
              key={`${p.x}-${p.y}`}
              x={p.x}
              y={p.y}
              width="18"
              height="18"
              rx="3"
              fill="none"
              stroke="var(--accent-strong)"
              strokeWidth="2"
            />
          ))}
          <text
            x="130"
            y="240"
            textAnchor="middle"
            fontSize="15"
            fontWeight="600"
            fill="var(--accent-strong)"
          >
            14 spots, moderate
          </text>
        </g>
      </svg>
    </div>
  );
}

function PipelineScene() {
  const sceneRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(0);

  // How far we are through the pinned frame decides which read is showing.
  //
  // This measures the element every frame rather than listening for scroll.
  // Smooth-scrolling libraries drive the page from their own loop and do not
  // reliably emit scroll events, so a listener here silently never fires. The
  // loop only runs while the scene is actually on screen.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    let frame = 0;
    let last = -1;

    const measure = () => {
      const rect = scene.getBoundingClientRect();
      const range = rect.height - window.innerHeight;
      if (range > 0) {
        const progress = Math.min(1, Math.max(0, -rect.top / range));
        const next = Math.min(
          READS.length - 1,
          Math.floor(progress * READS.length),
        );
        if (next !== last) {
          last = next;
          setActive(next);
        }
      }
      frame = requestAnimationFrame(measure);
    };

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !frame) {
          frame = requestAnimationFrame(measure);
        } else if (!entry.isIntersecting && frame) {
          cancelAnimationFrame(frame);
          frame = 0;
        }
      },
      { threshold: 0 },
    );
    observer.observe(scene);

    return () => {
      observer.disconnect();
      if (frame) cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <div className="pipeline-scene" ref={sceneRef}>
      <div className="pipeline-sticky shell">
        <div className="pipeline-visual">
          <FaceRead active={active} />
        </div>
        <div className="pipeline-steps">
          {READS.map((read, index) => (
            <div
              className={`pipeline-step ${index === active ? "active" : ""}`}
              key={read.title}
            >
              <h3>
                <span className="pipeline-index">
                  {String(index + 1).padStart(2, "0")}
                </span>
                {read.title}
              </h3>
              <p>{read.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------------
   Live visit count. Privacy-safe: one integer, no ids, no cookies.
   -------------------------------------------------------------------------- */

function ScanCount() {
  const [count, setCount] = useState<number | null>(null);
  const recorded = useRef(false);

  useEffect(() => {
    if (recorded.current) return;
    recorded.current = true;
    fetch("/api/visits", { method: "POST" })
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (data && typeof data.visits === "number") setCount(data.visits);
      })
      .catch(() => {});
  }, []);

  if (count === null || count <= 0) return null;
  return (
    <p className="scan-count">
      <strong>{count.toLocaleString("en-CA")}</strong> visits so far
    </p>
  );
}

/* --------------------------------------------------------------------------
   Numbers: accuracy always sits next to the score you would get by guessing.
   Figures from docs/RESULTS.md, strict leak-free split.
   -------------------------------------------------------------------------- */

const NUMBERS = [
  {
    value: 98.8,
    decimals: 1,
    suffix: "%",
    fill: 0.988,
    label: "acne type, across five classes. Guessing the most common one gets 26.8%.",
  },
  {
    value: 66.6,
    decimals: 1,
    suffix: "%",
    fill: 0.666,
    label: "mAP50 for the spot detector, measured on images it never trained on.",
  },
  {
    value: 44.4,
    decimals: 1,
    suffix: "%",
    fill: 0.444,
    label: "skin type, against a 37.7% baseline. This one is barely better than guessing.",
  },
];

export default function Story() {
  useReveals();
  useHeaderTheme();

  return (
    <main id="story-root" className="story-page">
      {/* ================= night ================= */}

      <div className="night-zone night">
        <section className="opening shell">
          <div data-reveal>
            <h1>It is 11pm and you are looking at your face again.</h1>
            <p className="lede">
              Not in front of anyone. Just you, the mirror, and the part of
              growing up nobody wants to ask about out loud. acno gives you a
              straight answer about your skin, on your own phone, without
              sending your photo anywhere.
            </p>
            <div className="opening-actions">
              <Link href="/scan" className="button">
                Scan your skin
              </Link>
              <ScanCount />
            </div>
          </div>
          <Device />

          <div className="scroll-cue" aria-hidden>
            <span className="scroll-cue-line" />
            scroll
          </div>
        </section>

        <section className="story shell-narrow">
          <p className="story-lead" data-reveal>
            acno started with my own skin.
          </p>
          <p className="line-reveal">
            When my acne got bad I did not know what to do, and I worked it out
            slowly, mostly alone, with very little help. A lot of that was fear.
            The stigma around acne made me too embarrassed to ask anyone in
            person. What I wanted back then was somewhere genuinely private to
            ask, an answer I could get without anyone watching or judging me.
          </p>
          <p className="line-reveal">
            I also watched friends go through worse. People were bullied and
            shamed for their skin, even though acne is mostly just puberty doing
            its work, not anything they did or could control. Seeing people I
            care about get hurt for something that was not their fault is what
            pushed me to build this.
          </p>
          <p className="story-emphasis" data-reveal>
            acno is the private, judgment free companion I wish I had had.
          </p>
          <p className="story-byline">Matthew, who started acno</p>
        </section>

        <section className="turn shell-narrow">
          <h2 className="turn-title" data-reveal>
            So here is exactly how it works.
          </h2>
          <p className="turn-sub" data-reveal>
            No mystery, no black box. Three models read one photo, and none of
            them ever see the internet.
          </p>
        </section>
      </div>

      {/* ================= daybreak ================= */}

      <div className="dawn" id="daybreak" aria-hidden />

      {/* ================= day ================= */}

      <div className="day-zone">
        <PipelineScene />

        <section className="section shell-narrow">
          <h2 className="h2" data-reveal>
            A fourth model writes it up
          </h2>
          <p className="prose" data-reveal>
            Once the three models finish, Gemini turns their output into plain
            language and looks up current, well reviewed products that match what
            was found. It only ever receives the numbers. Your photo stays on
            your device, where it started.
          </p>
        </section>

        <section className="section shell-narrow">
          <h2 className="h2" data-reveal>
            What the models actually score
          </h2>
          <p className="prose" data-reveal>
            Measured on faces the models never saw while training, on a split we
            cleaned after finding that the original one leaked. We publish the
            weak number too, because hiding it would be the dishonest part.
          </p>

          <div className="numbers">
            {NUMBERS.map((n) => (
              <div className="number" key={n.label} data-reveal>
                <div className="number-value">
                  <span
                    data-count={n.value}
                    data-decimals={n.decimals}
                    data-suffix={n.suffix}
                  >
                    {n.value.toFixed(n.decimals)}
                    {n.suffix}
                  </span>
                </div>
                <div className="number-bar" style={{ ["--fill" as string]: n.fill }}>
                  <span />
                </div>
                <p className="number-label">{n.label}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="section-tight shell-narrow">
          <div className="honesty" data-reveal>
            <h2 className="h2">The hard part was never the model</h2>
            <p className="prose">
              It was the data. We found that both of our datasets were leaking
              images between training and testing, which had inflated every
              accuracy we published before July. We rebuilt the splits and our
              acne numbers survived. Skin type did not, and it sits near its own
              baseline because that dataset cannot carry it. Almost none of these
              photos show dark skin, so we test for that gap openly and we do not
              claim acno works equally well for everyone yet.
            </p>
          </div>
        </section>

        <section className="section-tight shell-narrow">
          <div className="privacy" data-reveal>
            <div className="privacy-mark">
              <svg
                viewBox="0 0 24 24"
                width="30"
                height="30"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden
              >
                <path d="M12 3 4 6 v6 c0 5 3.5 7.5 8 9 4.5 -1.5 8 -4 8 -9 V6 Z" />
                <path d="M9 12 l2 2 4 -4" />
              </svg>
            </div>
            <div>
              <h2 className="h2">Your photo stays yours</h2>
              <p className="prose">
                There is no server waiting to receive your image. Everything runs
                inside the page you are looking at, and when you close the tab it
                is gone. Nothing is stored, and nothing is used to train anything.
              </p>
            </div>
          </div>
        </section>

        <section className="section shell-narrow">
          <h2 className="h2" data-reveal>
            Where this goes next
          </h2>
          <div className="next-list" data-stagger>
            {[
              "Weekly re-scans that show a real progress trend",
              "A guide translated for newcomer families",
              "An evaluation set that covers every skin tone",
              "A clear handoff to a dermatologist for severe cases",
            ].map((item, index) => (
              <div className="next-item" key={item}>
                <span className="next-marker">
                  {String(index + 1).padStart(2, "0")}
                </span>
                {item}
              </div>
            ))}
          </div>
        </section>

        <section className="section-tight shell-narrow">
          <h2 className="h2" data-reveal>
            Made by a team of students
          </h2>
          <p className="prose" data-reveal>
            acno is led by Matthew and built by a team of students who care about
            skin health and self esteem as much as he does, who wanted the first
            answer about your own skin to be free, private, and honest about its
            limits.
          </p>
          <div className="team-names" data-stagger>
            {["Matthew", "Travis", "Sophia", "Jocelyn"].map((name) => (
              <span key={name}>{name}</span>
            ))}
          </div>
          <p className="prose small" data-reveal>
            With earlier contributions from Alan, Tanner, and Erwin.
          </p>
        </section>

        <section className="closing shell-narrow">
          <p className="motto" data-reveal>
            Scan smarter. Skin clearer.
          </p>
          <p className="prose" data-reveal>
            A first answer, never a diagnosis. Anything severe is always pointed
            to a real dermatologist.
          </p>
          <div data-reveal>
            <Link href="/scan" className="button">
              Scan your skin
            </Link>
          </div>
        </section>
      </div>
    </main>
  );
}
