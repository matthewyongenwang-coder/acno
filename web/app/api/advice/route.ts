// AI-written skin analysis and product suggestions.
//
// The browser sends only the scan RESULTS (skin type, acne type, spot count,
// severity) - never the photo. Claude writes a personalized explanation and
// suggests over-the-counter product types. If no ANTHROPIC_API_KEY is
// configured, the route returns 503 and the app quietly falls back to its
// built-in rules-based routine.

import Anthropic from "@anthropic-ai/sdk";
import { NextResponse } from "next/server";

export const maxDuration = 60; // Claude can take a moment to write

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

const SYSTEM_PROMPT = `You are the friendly skin guide inside Acno, an app built by high school students that helps teens understand their skin. You receive the results of an on-device AI scan and write a short personalized report.

Rules you never break:
- You give educational guidance, not medical diagnosis or treatment. Never claim to diagnose.
- Recommend only widely available over-the-counter products by category and active ingredient (cleansers, moisturizers, serums, toners, sunscreen, spot treatments). Naming well-known drugstore examples is fine. Never recommend prescription products.
- For severe findings or cystic acne, the first and clearest advice is always to see a dermatologist; products are supportive care only.
- Keep it kind and calm. Acne is normal, especially during puberty. Never use shaming language.
- Write for a teenager: plain words, short sentences, explain any skincare term you use.
- Do not use emojis, decorative symbols, or em dashes.
- The scan has known limits: it only knows dry/normal/oily (not combination or sensitive), and its confidence can be low. If confidence is under 60 percent, say the reading is uncertain and the advice is general.`;

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
            description: "Product category, e.g. Cleanser, Moisturizer, Serum, Toner, Sunscreen, Spot treatment",
          },
          lookFor: {
            type: "string",
            description: "The key ingredient or label to look for, with a plain explanation of what it does",
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
        additionalProperties: false,
      },
    },
    encouragement: {
      type: "string",
      description: "One or two warm closing sentences. If severity is severe or acne type is cystic, this must center seeing a dermatologist.",
    },
  },
  required: ["analysis", "products", "encouragement"],
  additionalProperties: false,
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

export async function POST(request: Request) {
  if (!process.env.ANTHROPIC_API_KEY) {
    return NextResponse.json(
      { error: "ai_not_configured" },
      { status: 503 },
    );
  }

  const scan = validate(await request.json().catch(() => null));
  if (!scan) {
    return NextResponse.json({ error: "invalid_scan" }, { status: 400 });
  }

  const client = new Anthropic();

  try {
    const response = await client.messages.create({
      model: "claude-opus-4-8",
      max_tokens: 2048,
      thinking: { type: "adaptive" },
      system: [
        {
          type: "text",
          text: SYSTEM_PROMPT,
          cache_control: { type: "ephemeral" },
        },
      ],
      output_config: {
        format: { type: "json_schema", schema: OUTPUT_SCHEMA },
      },
      messages: [
        {
          role: "user",
          content:
            `Scan results for this user:\n` +
            `Skin type: ${scan.skinType} (${Math.round(scan.skinTypeConfidence * 100)}% confidence)\n` +
            `Main acne type detected: ${scan.acneType} (${Math.round(scan.acneTypeConfidence * 100)}% confidence)\n` +
            `Individual spots found: ${scan.lesionCount}\n` +
            `Severity level: ${scan.severity}\n\n` +
            `Write their personalized report.`,
        },
      ],
    });

    if (response.stop_reason === "refusal") {
      return NextResponse.json({ error: "ai_declined" }, { status: 502 });
    }

    const text = response.content.find((block) => block.type === "text");
    if (!text || text.type !== "text") {
      return NextResponse.json({ error: "ai_empty" }, { status: 502 });
    }
    const advice: AdviceResponse = JSON.parse(text.text);
    return NextResponse.json(advice);
  } catch (error) {
    if (error instanceof Anthropic.RateLimitError) {
      return NextResponse.json({ error: "ai_busy" }, { status: 429 });
    }
    if (error instanceof Anthropic.APIError) {
      console.error("advice route:", error.status, error.message);
      return NextResponse.json({ error: "ai_error" }, { status: 502 });
    }
    console.error("advice route:", error);
    return NextResponse.json({ error: "ai_error" }, { status: 502 });
  }
}
