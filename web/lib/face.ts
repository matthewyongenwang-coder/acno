// Face gate.
//
// The three skin models will happily return a confident label for a wall, a
// pet, or a screenshot: they are classifiers with no "none of the above" class,
// so argmax always names something. Before this, the app printed that as a real
// report. YuNet tells us whether there is a face in the frame at all.
//
// This is a GATE, not a crop. The shipped classifiers are trained on whole
// images to match what the browser sends, so cropping to the detected face here
// would break train/serve parity and quietly change every prediction.
//
// It is deliberately a soft gate. Measured over 300 random images per dataset
// (results/face_gate_rates.json), YuNet finds a face in 92% of the face photos
// but only 4.3% of the acne close-ups, which are real, useful photos of a cheek
// or a chin. Refusing outright would reject most legitimate close-ups, so a
// miss warns and lets the person continue with the result marked unverified.
//
// Model: YuNet (face_detection_yunet_2023mar), OpenCV Zoo, MIT licensed, 232KB.
// Fixed 1x3x640x640 input, BGR, 0-255, NCHW. Anchor-free head at strides 8, 16
// and 32 over 80x80, 40x40 and 20x20 grids.

import * as ort from "onnxruntime-web";

export const FACE_SIZE = 640;
const STRIDES = [8, 16, 32] as const;

// OpenCV's own default for this model. Raising it loses faces at an angle;
// lowering it starts calling patches of skin a face.
export const FACE_SCORE_THRESHOLD = 0.6;

export interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FaceCheck {
  found: boolean;
  confidence: number;
  box: FaceBox | null;
}

interface Letterbox {
  scale: number;
  padX: number;
  padY: number;
}

/**
 * Letterbox onto a black 640x640 canvas as BGR 0-255 NCHW.
 *
 * BGR because OpenCV's FaceDetectorYN feeds the network straight from its own
 * BGR images with no mean subtraction or scaling, and the weights were trained
 * that way. Sending RGB silently costs accuracy rather than failing.
 */
export function faceTensor(image: HTMLImageElement): {
  tensor: ort.Tensor;
  letterbox: Letterbox;
} {
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  const scale = Math.min(FACE_SIZE / width, FACE_SIZE / height);
  const newW = Math.round(width * scale);
  const newH = Math.round(height * scale);
  const padX = Math.floor((FACE_SIZE - newW) / 2);
  const padY = Math.floor((FACE_SIZE - newH) / 2);

  const canvas = document.createElement("canvas");
  canvas.width = FACE_SIZE;
  canvas.height = FACE_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("canvas 2d context unavailable");
  // black bars, matching how OpenCV pads for this model
  ctx.fillStyle = "rgb(0,0,0)";
  ctx.fillRect(0, 0, FACE_SIZE, FACE_SIZE);
  ctx.drawImage(image, padX, padY, newW, newH);
  const { data } = ctx.getImageData(0, 0, FACE_SIZE, FACE_SIZE);

  const area = FACE_SIZE * FACE_SIZE;
  const out = new Float32Array(area * 3);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    out[p] = data[i + 2]; // B
    out[p + area] = data[i + 1]; // G
    out[p + 2 * area] = data[i]; // R
  }
  return {
    tensor: new ort.Tensor("float32", out, [1, 3, FACE_SIZE, FACE_SIZE]),
    letterbox: { scale, padX, padY },
  };
}

/**
 * Pick the highest-scoring face across the three stride heads.
 *
 * We only need to know whether a face is there, so this takes the argmax rather
 * than running non-maximum suppression over every anchor: the top box is the
 * same either way, and NMS is only needed when you want all of the faces.
 */
export function decodeFace(
  outputs: Record<string, ort.Tensor>,
  letterbox: Letterbox,
  imageW: number,
  imageH: number,
): FaceCheck {
  let bestScore = 0;
  let bestBox: FaceBox | null = null;

  for (const stride of STRIDES) {
    const cls = outputs[`cls_${stride}`]?.data as Float32Array | undefined;
    const obj = outputs[`obj_${stride}`]?.data as Float32Array | undefined;
    const bbox = outputs[`bbox_${stride}`]?.data as Float32Array | undefined;
    if (!cls || !obj || !bbox) continue;

    const cols = FACE_SIZE / stride;
    for (let i = 0; i < cls.length; i++) {
      // YuNet's score is the geometric mean of the class and objectness heads,
      // matching OpenCV's face_detect.cpp.
      const c = Math.min(Math.max(cls[i], 0), 1);
      const o = Math.min(Math.max(obj[i], 0), 1);
      const score = Math.sqrt(c * o);
      if (score <= bestScore) continue;

      const col = i % cols;
      const row = Math.floor(i / cols);
      const cx = (col + bbox[i * 4]) * stride;
      const cy = (row + bbox[i * 4 + 1]) * stride;
      const w = Math.exp(bbox[i * 4 + 2]) * stride;
      const h = Math.exp(bbox[i * 4 + 3]) * stride;

      bestScore = score;
      // undo the letterbox, back into the original photo's coordinates
      bestBox = {
        x: (cx - w / 2 - letterbox.padX) / letterbox.scale,
        y: (cy - h / 2 - letterbox.padY) / letterbox.scale,
        width: w / letterbox.scale,
        height: h / letterbox.scale,
      };
    }
  }

  const found = bestScore >= FACE_SCORE_THRESHOLD;
  if (!found) return { found: false, confidence: bestScore, box: null };

  // clamp to the frame; a box can run off the edge when a face is half out
  const box = bestBox as FaceBox;
  const x = Math.max(0, Math.min(box.x, imageW));
  const y = Math.max(0, Math.min(box.y, imageH));
  return {
    found: true,
    confidence: bestScore,
    box: {
      x,
      y,
      width: Math.min(box.width, imageW - x),
      height: Math.min(box.height, imageH - y),
    },
  };
}

/* ---------------------------------------------------------------------------
   Second signal: how much of the frame is skin.

   The face detector alone is not a usable gate. Measured over 300 random images
   per dataset it fires on 92% of the face photos but only 4.3% of the acne
   close-ups, which are real photos of a cheek or a chin. Requiring a face would
   reject almost every close-up.

   So we also measure the fraction of the frame passing the same YCrCb + CIELAB
   skin test the fairness review uses (scripts/estimate_skin_tone.py). The two
   thresholds are calibrated in scripts/calibrate_face_gate.py, which writes
   results/face_gate_calibration.json.
   --------------------------------------------------------------------------- */

// Calibrated: at 0.10 the combined gate passes 100% of the face photos and
// 98.7% of the acne close-ups, while every non-skin negative tested (solid
// colours, plot images, random noise) scores 0.069 or below.
export const SKIN_FRACTION_THRESHOLD = 0.1;

// Downsample before measuring. The fraction is a whole-image statistic, so a
// 160px thumbnail gives the same answer as the full photo for a fraction of the
// work, and this runs on phones.
const SKIN_SAMPLE_SIZE = 160;

/**
 * Fraction of pixels that look like skin.
 *
 * Kept numerically identical to skin_mask() in scripts/estimate_skin_tone.py:
 * the same YCrCb box, the same luminance limits, and the same requirement that
 * CIELAB b* be positive (skin is yellowish, so pink backdrops and red arrows,
 * which is what broke the first fairness attempt, are excluded).
 */
export function skinFraction(image: HTMLImageElement): number {
  const canvas = document.createElement("canvas");
  canvas.width = SKIN_SAMPLE_SIZE;
  canvas.height = SKIN_SAMPLE_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return 0;
  ctx.drawImage(image, 0, 0, SKIN_SAMPLE_SIZE, SKIN_SAMPLE_SIZE);
  const { data } = ctx.getImageData(0, 0, SKIN_SAMPLE_SIZE, SKIN_SAMPLE_SIZE);

  let skin = 0;
  const total = SKIN_SAMPLE_SIZE * SKIN_SAMPLE_SIZE;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];

    // ITU-R BT.601, the conversion cv2.COLOR_BGR2YCrCb uses
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const cr = (r - y) * 0.713 + 128;
    const cb = (b - y) * 0.564 + 128;
    if (cr < 133 || cr > 180 || cb < 77 || cb > 127) continue;
    if (y < 30 || y > 240) continue;

    const [aStar, bStar] = labAB(r, g, b);
    if (bStar < 6) continue;
    if (aStar < 3 || aStar > 45) continue;
    skin++;
  }
  return skin / total;
}

/** CIELAB a* and b* from sRGB, on OpenCV's 0-255 encoded scale (offset 128). */
function labAB(r8: number, g8: number, b8: number): [number, number] {
  const lin = (c: number) => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  };
  const r = lin(r8);
  const g = lin(g8);
  const b = lin(b8);

  // sRGB to XYZ (D65), then normalise by the D65 white point
  const x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.950456;
  const y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
  const z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.088754;

  const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(x);
  const fy = f(y);
  const fz = f(z);

  // OpenCV stores a* and b* as the true value plus 128, so subtracting 128
  // (as estimate_skin_tone.py does) recovers the signed value we compare here.
  return [500 * (fx - fy), 200 * (fy - fz)];
}

/** The gate itself: a face, or enough skin in frame. */
export function passesGate(face: FaceCheck, skin: number): boolean {
  return face.found || skin >= SKIN_FRACTION_THRESHOLD;
}
