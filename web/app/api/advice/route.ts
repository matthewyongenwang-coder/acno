// AI-written skin analysis and product suggestions.
//
// The browser sends only the scan RESULTS (skin type, acne type, spot count,
// severity) - never the photo. The AI writes a personalized explanation and
// suggests over-the-counter product types.
//
// Two phases:
//   1. RESEARCH - the model searches the web (Google Search grounding) for
//      current, well-reviewed OTC products that fit the scan profile, across
//      a range of brands. Best effort: if search comes back empty, formatting
//      still proceeds without it.
//   2. FORMAT   - the model turns the scan (plus any research it found) into a
//      structured report, drawing on the researched products so the advice is
//      varied and not the same two brands every time.
//
// Provider: Gemini (GEMINI_API_KEY). With no key set, the route returns 503
// and the app quietly falls back to its built-in rules-based routine.

import { GoogleGenAI } from "@google/genai";
import { NextResponse } from "next/server";

export const maxDuration = 60; // research + writing can take a moment

const MODEL = "gemini-3.8-flash";

const SKIN_TYPES = new Set(["dry", "normal", "oily"]);
const ACNE_TYPES = new Set(["Blackheads", "Cyst", "Papules", "Pustules", "Whiteheads"]);
const SEVERITIES = new Set(["clear", "mild", "moderate", "severe"]);

interface AdviceRequest {
  skinType: string;
  skinTypeConfidence: number;
  acneType: string;
  acneTypeConfidence: number;
  lesionCount: number;
  severity: string;
}

const SYSTEM_PROMPT = `You are the friendly skin guide inside acno, an app built by high school students that helps teens understand their skin. You receive the results of an on-device AI scan and write a short personalized report.

Rules you never break:
- You give educational guidance, not medical diagnosis or treatment. Never claim to diagnose.
- Recommend only widely available over-the-counter products by category and active ingredient (cleansers, moisturizers, serums, toners, sunscreen, spot treatments). Naming well-known examples is fine. Never recommend prescription products.
- For severe findings or cystic acne, the first and clearest advice is always to see a dermatologist; products are supportive care only.
- Keep it kind and calm. Acne is normal, especially during puberty. Never use shaming language.
- Write for a teenager: plain words, short sentences, explain any skincare term you use.
- Do not use emojis, decorative symbols, or em dashes.
- The scan has known limits: it only knows dry/normal/oily (not combination or sensitive), and its confidence can be low. If confidence is under 60 percent, say the reading is uncertain and the advice is general.

Variety matters. Do not default to the same one or two brands (for example Cetaphil and CeraVe) on every report. When you are given research on current products, prefer those specific, well-reviewed options and suggest a range of different brands that genuinely fit this person's skin type and acne type. Match the active ingredient to the concern, not the brand name.`;

const RESEARCH_SYSTEM_PROMPT = `You are a skincare research assistant. You search the web for current, widely available over-the-counter skincare products and report concise findings for another assistant to use.

Focus on: cleansers, moisturizers, serums, toners, sunscreen, and spot treatments that suit the given skin type and acne type, matched by active ingredient (for example salicylic acid, benzoyl peroxide, adapalene, niacinamide, ceramides, azelaic acid).

Give a diverse spread of brands and price points that are genuinely available at drugstores and beauty retailers. Do not limit yourself to the most obvious two brands. Never recommend prescription products. Note any product that is well reviewed and why it fits. Keep the whole brief under 250 words, as plain notes (no emojis, no em dashes).`;

const OUTPUT_SCHEMA = {
  type: "object",
  properties: {
    analysis: {
      type: "string",
      description:
        "Three to five sentences speaking directly to the user about what their scan means, in plain teen-friendly language.",
    },
    products: {
      type: "array",
      description:
        "Three to five over-the-counter product suggestions ordered morning routine first.",
      items: {
        type: "object",
        properties: {
          category: {
            type: "string",
            description:
              "Product category, e.g. Cleanser, Moisturizer, Serum, Toner, Sunscreen, Spot treatment",
          },
          lookFor: {
            type: "string",
            description:
              "The key ingredient or label to look for, with a plain explanation of what it does",
          },
          example: {
            type: "string",
            description: "One or two well-known affordable drugstore examples",
          },
          howToUse: {
            type: "string",
            description: "One sentence on when and how to use it",
          },
        },
        required: ["category", "lookFor", "example", "howToUse"],
      },
    },
    encouragement: {
      type: "string",
      description:
        "One or two warm closing sentences. If severity is severe or acne type is cystic, this must center seeing a dermatologist.",
    },
  },
  required: ["analysis", "products", "encouragement"],
} as const;

export interface AdviceResponse {
  analysis: string;
  products: {
    category: string;
    lookFor: string;
    example: string;
    howToUse: string;
  }[];
  encouragement: string;
}

function validate(body: unknown): AdviceRequest | null {
  if (typeof body !== "object" || body === null) return null;
  const b = body as Record<string, unknown>;
  if (
    typeof b.skinType !== "string" || !SKIN_TYPES.has(b.skinType) ||
    typeof b.acneType !== "string" || !ACNE_TYPES.has(b.acneType) ||
    typeof b.severity !== "string" || !SEVERITIES.has(b.severity) ||
    typeof b.lesionCount !== "number" || b.lesionCount < 0 || b.lesionCount > 1000 ||
    typeof b.skinTypeConfidence !== "number" ||
    typeof b.acneTypeConfidence !== "number"
  ) {
    return null;
  }
  return {
    skinType: b.skinType,
    skinTypeConfidence: b.skinTypeConfidence,
    acneType: b.acneType,
    acneTypeConfidence: b.acneTypeConfidence,
    lesionCount: b.lesionCount,
    severity: b.severity,
  };
}

function scanSummary(scan: AdviceRequest): string {
  return (
    `Skin type: ${scan.skinType} (${Math.round(scan.skinTypeConfidence * 100)}% confidence)\n` +
    `Main acne type detected: ${scan.acneType} (${Math.round(scan.acneTypeConfidence * 100)}% confidence)\n` +
    `Individual spots found: ${scan.lesionCount}\n` +
    `Severity level: ${scan.severity}`
  );
}

function researchPrompt(scan: AdviceRequest): string {
  return (
    `Research current over-the-counter skincare products for this profile:\n\n` +
    `${scanSummary(scan)}\n\n` +
    `Search the web and report a diverse set of specific, well-reviewed products ` +
    `(different brands, matched by active ingredient) that would suit this skin type ` +
    `and acne type. Return only your notes.`
  );
}

function formatPrompt(scan: AdviceRequest, research: string | null): string {
  const base =
    `Scan results for this user:\n${scanSummary(scan)}\n\n`;
  const withResearch = research
    ? `Current product research to draw from (prefer these specific, varied options ` +
      `over defaulting to the same one or two brands):\n${research}\n\n`
    : "";
  return `${base}${withResearch}Write their personalized report.`;
}

async function research(scan: AdviceRequest): Promise<string | null> {
  try {
    const client = new GoogleGenAI({});
    const interaction = await client.interactions.create({
      model: MODEL,
      system_instruction: RESEARCH_SYSTEM_PROMPT,
      input: researchPrompt(scan),
      tools: [{ type: "google_search" }],
    });
    const text = interaction.output_text?.trim();
    return text || null;
  } catch (error) {
    console.error("advice research (gemini):", error);
    return null; // research is best effort - never block the report on it
  }
}

async function advice(scan: AdviceRequest): Promise<AdviceResponse | null> {
  const notes = await research(scan);
  const client = new GoogleGenAI({});
  const interaction = await client.interactions.create({
    model: MODEL,
    system_instruction: SYSTEM_PROMPT,
    input: formatPrompt(scan, notes),
    response_format: {
      type: "text",
      mime_type: "application/json",
      schema: OUTPUT_SCHEMA,
    },
    generation_config: {
      thinking_level: "high",
    },
  });

  const text = interaction.output_text?.trim();
  if (!text) return null;
  return JSON.parse(text) as AdviceResponse;
}

export async function POST(request: Request) {
  if (!process.env.GEMINI_API_KEY) {
    return NextResponse.json({ error: "ai_not_configured" }, { status: 503 });
  }

  const scan = validate(await request.json().catch(() => null));
  if (!scan) {
    return NextResponse.json({ error: "invalid_scan" }, { status: 400 });
  }

  try {
    const result = await advice(scan);
    if (!result) {
      return NextResponse.json({ error: "ai_declined" }, { status: 502 });
    }
    return NextResponse.json(result);
  } catch (error) {
    console.error("advice route:", error);
    return NextResponse.json({ error: "ai_error" }, { status: 502 });
  }
}
