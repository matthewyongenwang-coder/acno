// Advice text, ported from data/guide_rules.csv and data/acne_type_info.csv in the
// repo root. Those CSVs stay the editable source of truth for the team; if you
// change them, mirror the change here.

export type SkinType = "dry" | "normal" | "oily";
export type Severity = "clear" | "mild" | "moderate" | "severe";
export type AcneType = "Blackheads" | "Cyst" | "Papules" | "Pustules" | "Whiteheads";

export interface Routine {
  morning: string;
  evening: string;
  avoid: string;
}

export interface AcneInfo {
  plainName: string;
  explanation: string;
  careTip: string;
}

export const DISCLAIMER =
  "acno gives educational guidance, not a medical diagnosis. " +
  "For severe or persistent acne, please see a dermatologist.";

export const SEVERITY_TEXT: Record<Severity, string> = {
  clear: "Your skin looks clear. Whatever you are doing, it is working.",
  mild: "A few spots, which is completely normal, especially during puberty.",
  moderate: "A fair number of active spots. A steady routine can really help here.",
  severe: "Quite a lot of active spots. This level deserves professional care.",
};

export function severityFromCount(count: number): Severity {
  if (count <= 1) return "clear";
  if (count <= 10) return "mild";
  if (count <= 25) return "moderate";
  return "severe";
}

export function needsDermatologist(severity: Severity, acneType: AcneType): boolean {
  return severity === "severe" || acneType === "Cyst";
}

export const ACNE_INFO: Record<AcneType, AcneInfo> = {
  Whiteheads: {
    plainName: "Whiteheads",
    explanation:
      "Small bumps where a pore got clogged with oil and dead skin and closed over. " +
      "The white tip is the trapped material under the surface.",
    careTip:
      "Gentle exfoliating ingredients like salicylic acid help unclog pores. Do not squeeze them.",
  },
  Blackheads: {
    plainName: "Blackheads",
    explanation:
      "Clogged pores that stayed open. The dark color is not dirt. " +
      "It is the clog reacting with air.",
    careTip:
      "Salicylic acid cleansers work well over time. Pore strips feel satisfying " +
      "but the clog usually comes back.",
  },
  Papules: {
    plainName: "Red bumps",
    explanation:
      "Small firm red bumps where a clogged pore got inflamed. They have no white tip " +
      "and can feel tender.",
    careTip:
      "Be gentle. Benzoyl peroxide in low strength calms them. Scrubbing makes inflammation worse.",
  },
  Pustules: {
    plainName: "Pimples with a white center",
    explanation:
      "Inflamed bumps filled with fluid. This is what most people picture when they hear pimple.",
    careTip:
      "A small dab of benzoyl peroxide helps. Popping them pushes the inflammation " +
      "deeper and can scar.",
  },
  Cyst: {
    plainName: "Cystic acne",
    explanation:
      "Deep painful lumps under the skin. This is the most serious type and the most " +
      "likely to scar.",
    careTip:
      "This type really needs a dermatologist. Home products alone usually cannot reach it.",
  },
};

export const ROUTINES: Record<SkinType, Record<Severity, Routine>> = {
  dry: {
    clear: {
      morning: "Rinse with lukewarm water and apply a gentle fragrance-free moisturizer.",
      evening: "Use a mild hydrating cleanser and a richer moisturizer before bed.",
      avoid: "Hot showers on your face and foaming cleansers that leave skin feeling tight.",
    },
    mild: {
      morning: "Gentle hydrating cleanser then a light moisturizer with ceramides.",
      evening:
        "Cleanse and spot-treat blemishes with a low strength benzoyl peroxide " +
        "(2.5 percent) then moisturize.",
      avoid: "Scrubbing and over-washing. Dry skin with acne still needs moisture.",
    },
    moderate: {
      morning: "Hydrating cleanser and an oil-free moisturizer with SPF if you are heading out.",
      evening:
        "Cleanse then apply adapalene 0.1 percent gel (a pea sized amount for the " +
        "whole face) and moisturize after.",
      avoid: "Layering several acne products at once. Start with one and give it 6 to 8 weeks.",
    },
    severe: {
      morning: "Keep the routine simple: gentle cleanser and moisturizer only.",
      evening: "Same as morning. Do not stack strong products on irritated skin.",
      avoid: "Picking or squeezing. It spreads inflammation and causes scarring.",
    },
  },
  normal: {
    clear: {
      morning: "Rinse or use a gentle cleanser and a light moisturizer.",
      evening: "Gentle cleanser then moisturizer. That is genuinely enough.",
      avoid: "Buying products for problems you do not have. Simple routines win.",
    },
    mild: {
      morning: "Gentle cleanser then light moisturizer.",
      evening:
        "Cleanse and spot-treat with benzoyl peroxide 2.5 percent or salicylic acid " +
        "2 percent then moisturize.",
      avoid: "Touching your face through the day and sleeping in sweat after sports.",
    },
    moderate: {
      morning: "Gentle cleanser and an oil-free moisturizer.",
      evening:
        "Cleanse then a thin layer of adapalene 0.1 percent gel across acne-prone " +
        "areas and moisturize after.",
      avoid: "Adding new products every week. Give each change 6 to 8 weeks to show results.",
    },
    severe: {
      morning: "Gentle cleanser and moisturizer only. Keep irritation low.",
      evening: "Same as morning until you can talk to a professional.",
      avoid: "Harsh scrubs and DIY remedies like toothpaste or lemon juice. They make it worse.",
    },
  },
  oily: {
    clear: {
      morning: "Gentle foaming cleanser then an oil-free (non-comedogenic) moisturizer.",
      evening: "Cleanse to remove the day's oil then apply a light oil-free moisturizer.",
      avoid: "Skipping moisturizer. Stripped skin often makes more oil to compensate.",
    },
    mild: {
      morning: "Foaming cleanser with salicylic acid then an oil-free moisturizer.",
      evening: "Cleanse and spot-treat with benzoyl peroxide 2.5 percent then moisturize.",
      avoid: "Blotting and washing more than twice a day. Extra washing does not mean less oil.",
    },
    moderate: {
      morning: "Salicylic acid cleanser and an oil-free gel moisturizer.",
      evening:
        "Cleanse then apply adapalene 0.1 percent gel across acne-prone areas and " +
        "moisturize after.",
      avoid: "Heavy comedogenic products and hair oils that touch your forehead.",
    },
    severe: {
      morning: "Gentle cleanser and oil-free moisturizer. Do not attack your skin.",
      evening: "Same as morning. Strong home treatment is not the answer at this level.",
      avoid: "Aggressive treatment stacking. Severe acne needs professional care not more products.",
    },
  },
};
