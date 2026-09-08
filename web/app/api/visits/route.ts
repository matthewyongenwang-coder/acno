// A privacy-safe counter. It stores two integers and nothing else: no photos,
// no user ids, no cookies, no IP addresses.
//
//   acno:visits  landing-page views
//   acno:scans   analyses that actually completed
//
// These used to be one key, which meant an ordinary page view and a real scan
// both incremented the same number while the page labelled it "scans run so
// far". The stored total was therefore mostly traffic, not usage. The visits
// key keeps its history (so the existing total stays meaningful as a visit
// count) and scans starts fresh, counting only completed analyses.
//
// Storage is a Vercel KV / Upstash Redis store, reached over its REST API so we
// need no extra dependency. If the store is not configured (the two env vars
// below are absent), every call returns nulls and the feature simply hides
// itself, exactly like the AI guide does without an API key.
//
//   KV_REST_API_URL    the store's REST endpoint
//   KV_REST_API_TOKEN  a token with read/write access

import { NextResponse } from "next/server";

// This is a live counter; it must never be served from a cache.
export const dynamic = "force-dynamic";

// Vercel's KV integration injects KV_REST_API_*; the newer Upstash marketplace
// integration injects UPSTASH_REDIS_REST_*. Accept either, same REST protocol.
const KV_URL = process.env.KV_REST_API_URL ?? process.env.UPSTASH_REDIS_REST_URL;
const KV_TOKEN =
  process.env.KV_REST_API_TOKEN ?? process.env.UPSTASH_REDIS_REST_TOKEN;

const VISITS_KEY = "acno:visits";
const SCANS_KEY = "acno:scans";

// People who visited Acno before the counter existed (demos, early testers).
// The stored value counts from zero; we add this so the public number starts
// here. It applies to visits only: the scan counter starts honestly at zero.
const VISITS_BASELINE = 20;

async function command(name: "get" | "incr", key: string): Promise<number | null> {
  if (!KV_URL || !KV_TOKEN) return null;
  try {
    const response = await fetch(`${KV_URL}/${name}/${encodeURIComponent(key)}`, {
      headers: { Authorization: `Bearer ${KV_TOKEN}` },
      cache: "no-store",
    });
    if (!response.ok) return null;
    const data = (await response.json()) as { result?: unknown };
    // An unset key reads back as null; that genuinely means zero.
    if (data.result === null || data.result === undefined) return 0;
    const value = Number(data.result);
    // Anything else non-numeric means the store is misconfigured or returning
    // something we do not understand. Treat that as a failure and hide the
    // number, rather than coercing it to 0 and publishing a plausible lie.
    return Number.isFinite(value) ? value : null;
  } catch {
    return null;
  }
}

function withBaseline(raw: number | null): number | null {
  return raw === null ? null : raw + VISITS_BASELINE;
}

export async function GET() {
  const [visits, scans] = await Promise.all([
    command("get", VISITS_KEY),
    command("get", SCANS_KEY),
  ]);
  return NextResponse.json({ visits: withBaseline(visits), scans });
}

// POST /api/visits            records a landing-page view
// POST /api/visits?kind=scan  records one completed analysis
export async function POST(request: Request) {
  const isScan = new URL(request.url).searchParams.get("kind") === "scan";
  if (isScan) {
    return NextResponse.json({ scans: await command("incr", SCANS_KEY) });
  }
  return NextResponse.json({ visits: withBaseline(await command("incr", VISITS_KEY)) });
}
