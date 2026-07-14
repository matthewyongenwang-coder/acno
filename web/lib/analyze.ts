// Browser-side inference: the TypeScript port of src/pipeline.py.
// All three ONNX models run locally through onnxruntime-web, so the photo
// never leaves the device.

import * as ort from "onnxruntime-web";

import {
  ACNE_INFO,
  DISCLAIMER,
  ROUTINES,
  type AcneInfo,
  type AcneType,
  type Routine,
  type Severity,
  type SkinType,
  needsDermatologist,
  severityFromCount,
} from "./guide";

const SKIN_CLASSES: SkinType[] = ["dry", "normal", "oily"];
// alphabetical, matching Keras image_dataset_from_directory class order
const ACNE_CLASSES: AcneType[] = ["Blackheads", "Cyst", "Papules", "Pustules", "Whiteheads"];

const CLASSIFIER_SIZE = 224;
const YOLO_SIZE = 640;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.7;

export interface Box {
  x: number; // top-left, in source image pixels
  y: number;
  width: number;
  height: number;
  confidence: number;
}

export interface Report {
  skinType: SkinType;
  skinTypeConfidence: number;
  acneType: AcneType;
  acneTypeConfidence: number;
  acneInfo: AcneInfo;
  lesionCount: number;
  severity: Severity;
  seeDermatologist: boolean;
  routine: Routine;
  boxes: Box[];
  disclaimer: string;
}

let sessionsPromise: Promise<{
  skin: ort.InferenceSession;
  acne: ort.InferenceSession;
  yolo: ort.InferenceSession;
}> | null = null;

function loadSessions() {
  if (!sessionsPromise) {
    // the ort wasm runtime is served from public/ort, next to the models
    ort.env.wasm.wasmPaths = "/ort/";
    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: ["wasm"],
    };
    sessionsPromise = (async () => {
      const [skin, acne, yolo] = await Promise.all([
        ort.InferenceSession.create("/models/skin_type.onnx", options),
        ort.InferenceSession.create("/models/acne_type.onnx", options),
        ort.InferenceSession.create("/models/acne_yolo.onnx", options),
      ]);
      return { skin, acne, yolo };
    })();
    sessionsPromise.catch(() => {
      sessionsPromise = null; // allow a retry after a failed load
    });
  }
  return sessionsPromise;
}

/** Kick off the model download early so the first analysis feels fast. */
export function warmUp(): void {
  void loadSessions();
}

function drawToCanvas(image: HTMLImageElement, width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("canvas 2d context unavailable");
  return { canvas, ctx };
}

/** Plain stretch-resize to 224x224, raw 0-255 RGB, NHWC. Matches Keras eval exactly. */
function classifierTensor(image: HTMLImageElement): ort.Tensor {
  const { ctx } = drawToCanvas(image, CLASSIFIER_SIZE, CLASSIFIER_SIZE);
  ctx.drawImage(image, 0, 0, CLASSIFIER_SIZE, CLASSIFIER_SIZE);
  const { data } = ctx.getImageData(0, 0, CLASSIFIER_SIZE, CLASSIFIER_SIZE);
  const out = new Float32Array(CLASSIFIER_SIZE * CLASSIFIER_SIZE * 3);
  for (let i = 0, j = 0; i < data.length; i += 4) {
    out[j++] = data[i];
    out[j++] = data[i + 1];
    out[j++] = data[i + 2];
  }
  return new ort.Tensor("float32", out, [1, CLASSIFIER_SIZE, CLASSIFIER_SIZE, 3]);
}

interface Letterbox {
  scale: number;
  padX: number;
  padY: number;
}

/** Letterbox to 640x640 on grey, 0-1 RGB, NCHW. Matches the ultralytics pipeline. */
function yoloTensor(image: HTMLImageElement): { tensor: ort.Tensor; letterbox: Letterbox } {
  const scale = Math.min(YOLO_SIZE / image.naturalWidth, YOLO_SIZE / image.naturalHeight);
  const newW = Math.round(image.naturalWidth * scale);
  const newH = Math.round(image.naturalHeight * scale);
  const padX = Math.floor((YOLO_SIZE - newW) / 2);
  const padY = Math.floor((YOLO_SIZE - newH) / 2);

  const { ctx } = drawToCanvas(image, YOLO_SIZE, YOLO_SIZE);
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, YOLO_SIZE, YOLO_SIZE);
  ctx.drawImage(image, padX, padY, newW, newH);
  const { data } = ctx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE);

  const area = YOLO_SIZE * YOLO_SIZE;
  const out = new Float32Array(area * 3);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    out[p] = data[i] / 255;
    out[p + area] = data[i + 1] / 255;
    out[p + 2 * area] = data[i + 2] / 255;
  }
  return {
    tensor: new ort.Tensor("float32", out, [1, 3, YOLO_SIZE, YOLO_SIZE]),
    letterbox: { scale, padX, padY },
  };
}

function softmaxTop(scores: Float32Array): { index: number; confidence: number } {
  // the models already end in softmax; just take the max
  let index = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[index]) index = i;
  }
  return { index, confidence: scores[index] };
}

function iou(a: Box, b: Box): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

function nonMaxSuppression(boxes: Box[]): Box[] {
  const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
  const kept: Box[] = [];
  for (const candidate of sorted) {
    if (kept.every((k) => iou(candidate, k) < IOU_THRESHOLD)) {
      kept.push(candidate);
    }
  }
  return kept;
}

/** Decode the (1, 5, 8400) YOLO output into boxes in source-image pixels. */
function decodeYolo(output: ort.Tensor, letterbox: Letterbox, imageW: number, imageH: number): Box[] {
  const data = output.data as Float32Array;
  const numAnchors = output.dims[2];
  const raw: Box[] = [];
  for (let i = 0; i < numAnchors; i++) {
    const confidence = data[4 * numAnchors + i];
    if (confidence < CONF_THRESHOLD) continue;
    const cx = data[i];
    const cy = data[numAnchors + i];
    const w = data[2 * numAnchors + i];
    const h = data[3 * numAnchors + i];
    const x = (cx - w / 2 - letterbox.padX) / letterbox.scale;
    const y = (cy - h / 2 - letterbox.padY) / letterbox.scale;
    raw.push({
      x: Math.max(0, Math.min(x, imageW)),
      y: Math.max(0, Math.min(y, imageH)),
      width: Math.min(w / letterbox.scale, imageW),
      height: Math.min(h / letterbox.scale, imageH),
      confidence,
    });
  }
  return nonMaxSuppression(raw);
}

export async function analyze(image: HTMLImageElement): Promise<Report> {
  const sessions = await loadSessions();

  const clsTensor = classifierTensor(image);
  const { tensor: detTensor, letterbox } = yoloTensor(image);

  // the wasm backend runs one inference at a time; keep these sequential
  const skinOut = await sessions.skin.run({ [sessions.skin.inputNames[0]]: clsTensor });
  const acneOut = await sessions.acne.run({ [sessions.acne.inputNames[0]]: clsTensor });
  const yoloOut = await sessions.yolo.run({ [sessions.yolo.inputNames[0]]: detTensor });

  const skin = softmaxTop(skinOut[sessions.skin.outputNames[0]].data as Float32Array);
  const acne = softmaxTop(acneOut[sessions.acne.outputNames[0]].data as Float32Array);
  const boxes = decodeYolo(
    yoloOut[sessions.yolo.outputNames[0]],
    letterbox,
    image.naturalWidth,
    image.naturalHeight,
  );

  const skinType = SKIN_CLASSES[skin.index];
  const acneType = ACNE_CLASSES[acne.index];
  const severity = severityFromCount(boxes.length);

  return {
    skinType,
    skinTypeConfidence: skin.confidence,
    acneType,
    acneTypeConfidence: acne.confidence,
    acneInfo: ACNE_INFO[acneType],
    lesionCount: boxes.length,
    severity,
    seeDermatologist: needsDermatologist(severity, acneType),
    routine: ROUTINES[skinType][severity],
    boxes,
    disclaimer: DISCLAIMER,
  };
}
