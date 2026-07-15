// A privacy-safe visit counter. It stores a single integer, the number of
// scans that have been run, and nothing else: no photos, no user ids, no
// cookies, no IP addresses. A completed scan sends one "+1" and gets back the
// running total, which the app can show as honest social proof and use for the
// project's impact report.
//
// Storage is a Vercel KV / Upstash Redis store, reached over its REST API so we
// need no extra dependency. If the store is not configured (the two env vars
// below are absent), every call returns { count: null } and the feature simply
// hides itself, exactly like the AI guide does without an API key.
//
//   KV_REST_API_URL    the store's REST endpoint
//   KV_REST_API_TOKEN  a token with read/write access

import { NextResponse } from "next/server";

// Vercel's KV integration injects KV_REST_API_*; the newer Upstash marketplace
// integration injects UPSTASH_REDIS_REST_*. Accept either, same REST protocol.
const KV_URL = process.env.KV_REST_API_URL ?? process.env.UPSTASH_REDIS_REST_URL;
const KV_TOKEN =
  process.env.KV_REST_API_TOKEN ?? process.env.UPSTASH_REDIS_REST_TOKEN;
const KEY = "acno:visits";

// People who used Acno before the counter existed (demos, early testers). The
// stored value counts from zero; we add this so the public number starts here.
const BASELINE = 20;

function withBaseline(raw: number | null): number | null {
  return raw === null ? null : raw + BASELINE;
}

async function command(name: "get" | "incr"): Promise<number | null> {
  if (!KV_URL || !KV_TOKEN) return null;
  try {
    const response = await fetch(`${KV_URL}/${name}/${KEY}`, {
      headers: { Authorization: `Bearer ${KV_TOKEN}` },
      cache: "no-store",
    });
    if (!response.ok) return null;
    const data = (await response.json()) as { result?: unknown };
    // An unset key reads back as null; treat that as zero.
    if (data.result === null || data.result === undefined) return 0;
    const value = Number(data.result);
    return Number.isFinite(value) ? value : 0;
  } catch {
    return null;
  }
}

export async function GET() {
  return NextResponse.json({ count: withBaseline(await command("get")) });
}

export async function POST() {
  return NextResponse.json({ count: withBaseline(await command("incr")) });
}
