"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { analyze, warmUp, type Report } from "@/lib/analyze";
import { SEVERITY_TEXT } from "@/lib/guide";
import type { AdviceResponse } from "@/app/api/advice/route";

type AdviceState =
  | { kind: "loading" }
  | { kind: "ready"; advice: AdviceResponse }
  | { kind: "unavailable" };

function AdviceSection({ report }: { report: Report }) {
  const [state, setState] = useState<AdviceState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ kind: "loading" });
    fetch("/api/advice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        skinType: report.skinType,
        skinTypeConfidence: report.skinTypeConfidence,
        acneType: report.acneType,
        acneTypeConfidence: report.acneTypeConfidence,
        lesionCount: report.lesionCount,
        severity: report.severity,
      }),
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(String(response.status));
        return (await response.json()) as AdviceResponse;
      })
      .then((advice) => {
        if (!cancelled) setState({ kind: "ready", advice });
      })
      .catch(() => {
        if (!cancelled) setState({ kind: "unavailable" });
      });
    return () => {
      cancelled = true;
    };
  }, [report]);

  if (state.kind === "unavailable") return null;

  return (
    <section aria-label="AI skin guide">
      <h3 className="section-title">What our AI makes of it</h3>
      {state.kind === "loading" ? (
        <div className="status">
          <div className="spinner" aria-hidden />
          <p className="sub">Writing your personalized guide...</p>
        </div>
      ) : (
        <>
          <div className="card">
            <p>{state.advice.analysis}</p>
          </div>
          <h3 className="section-title">Products worth looking at</h3>
          {state.advice.products.map((product) => (
            <div className="card" key={product.category + product.lookFor}>
              <div className="label">{product.category}</div>
              <p>
                <strong>Look for:</strong> {product.lookFor}
              </p>
              <p>
                <strong>Examples:</strong> {product.example}
              </p>
              <p className="sub">{product.howToUse}</p>
            </div>
          ))}
          <div className="note" style={{ marginBottom: "1rem" }}>
            {state.advice.encouragement}
          </div>
          <p className="sub" style={{ marginBottom: "1rem" }}>
            Written by an AI model from your scan results. Only the numbers above
            were shared, never your photo. Product suggestions are ideas to
            research, not medical advice.
          </p>
        </>
      )}
    </section>
  );
}

type Stage =
  | { kind: "idle" }
  | { kind: "analyzing" }
  | { kind: "done"; report: Report; image: HTMLImageElement }
  | { kind: "error"; message: string };

function capitalize(word: string): string {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

export default function Home() {
  const [tab, setTab] = useState<"upload" | "camera">("upload");
  const [stage, setStage] = useState<Stage>({ kind: "idle" });
  const [dragging, setDragging] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const annotatedRef = useRef<HTMLCanvasElement>(null);

  // start downloading the models in the background on first visit
  useEffect(() => {
    warmUp();
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOn(false);
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const runAnalysis = useCallback(async (image: HTMLImageElement) => {
    setStage({ kind: "analyzing" });
    try {
      const report = await analyze(image);
      setStage({ kind: "done", report, image });
    } catch (error) {
      console.error(error);
      setStage({
        kind: "error",
        message:
          "Something went wrong while analyzing the photo. Check your connection " +
          "(the models download on first visit) and try again.",
      });
    }
  }, []);

  const handleBlob = useCallback(
    (blob: Blob) => {
      const url = URL.createObjectURL(blob);
      const image = new Image();
      image.onload = () => {
        URL.revokeObjectURL(url);
        void runAnalysis(image);
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        setStage({
          kind: "error",
          message: "That file does not look like a photo we can read. Try a jpg or png.",
        });
      };
      image.src = url;
    },
    [runAnalysis],
  );

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
      });
      streamRef.current = stream;
      setCameraOn(true);
      // the video element renders on the next tick
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          void videoRef.current.play();
        }
      });
    } catch {
      setStage({
        kind: "error",
        message: "We could not open the camera. You can upload a photo instead.",
      });
    }
  }, []);

  const capturePhoto = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    stopCamera();
    canvas.toBlob((blob) => blob && handleBlob(blob), "image/jpeg", 0.95);
  }, [handleBlob, stopCamera]);

  // draw the analyzed photo with lesion boxes
  useEffect(() => {
    if (stage.kind !== "done") return;
    const canvas = annotatedRef.current;
    if (!canvas) return;
    const { image, report } = stage;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(image, 0, 0);
    const stroke = Math.max(2, Math.round(image.naturalWidth / 320));
    ctx.lineWidth = stroke;
    ctx.strokeStyle = "#e2574c";
    for (const box of report.boxes) {
      ctx.strokeRect(box.x, box.y, box.width, box.height);
    }
  }, [stage]);

  const report = stage.kind === "done" ? stage.report : null;

  return (
    <main>
      <div className="hero">
        <h1>Acno</h1>
        <p>Scan smarter. Skin clearer.</p>
      </div>

      <div className="note">
        Take or upload a clear photo of your face in good light. Acno runs entirely in
        your browser: the photo is analyzed on your own device and never uploaded
        anywhere. When you close this page, it is gone.
      </div>

      <div className="tabs" role="tablist">
        <button
          role="tab"
          aria-selected={tab === "upload"}
          className={tab === "upload" ? "active" : ""}
          onClick={() => {
            stopCamera();
            setTab("upload");
          }}
        >
          Upload a photo
        </button>
        <button
          role="tab"
          aria-selected={tab === "camera"}
          className={tab === "camera" ? "active" : ""}
          onClick={() => setTab("camera")}
        >
          Use your camera
        </button>
      </div>

      {tab === "upload" && (
        <div
          className={`dropzone${dragging ? " drag" : ""}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const file = event.dataTransfer.files[0];
            if (file) handleBlob(file);
          }}
        >
          <p>Drop a photo here, or click to choose one.</p>
          <p className="sub">jpg, png or webp</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) handleBlob(file);
              event.target.value = "";
            }}
          />
        </div>
      )}

      {tab === "camera" && (
        <div>
          {cameraOn ? (
            <div>
              <video ref={videoRef} playsInline muted />
              <p style={{ textAlign: "center", marginTop: "0.8rem" }}>
                <button className="button" onClick={capturePhoto}>
                  Take the photo
                </button>
              </p>
            </div>
          ) : (
            <div className="dropzone" onClick={() => void startCamera()}>
              <p>Turn on the camera.</p>
              <p className="sub">The video stays on your device, nothing is recorded.</p>
            </div>
          )}
        </div>
      )}

      {stage.kind === "analyzing" && (
        <div className="status">
          <div className="spinner" aria-hidden />
          <p>Taking a close look at your skin...</p>
          <p className="sub">First visit? The models are downloading, give it a moment.</p>
        </div>
      )}

      {stage.kind === "error" && (
        <div className="derm" role="alert" style={{ marginTop: "1rem" }}>
          {stage.message}
        </div>
      )}

      {report && (
        <section aria-label="Your skin report">
          <h2 className="section-title">Your skin report</h2>

          <div className="row">
            <div className="card">
              <div className="label">Skin type</div>
              <div className="value">{capitalize(report.skinType)}</div>
              <div className="sub">{Math.round(report.skinTypeConfidence * 100)}% confident</div>
            </div>
            <div className="card">
              <div className="label">Severity</div>
              <div className="value">{capitalize(report.severity)}</div>
              <div className="sub">{report.lesionCount} spots found</div>
            </div>
          </div>

          <p className="sub" style={{ marginBottom: "1rem" }}>
            {SEVERITY_TEXT[report.severity]}
          </p>

          <div className="card">
            <div className="label">Main acne type we see</div>
            <h3>{report.acneInfo.plainName}</h3>
            <p>{report.acneInfo.explanation}</p>
            <p>
              <strong>What helps:</strong> {report.acneInfo.careTip}
            </p>
          </div>

          <h3 className="section-title">Where we looked</h3>
          <canvas ref={annotatedRef} className="annotated" />
          <p className="sub" style={{ margin: "0.5rem 0 1rem" }}>
            Each box is a spot the model found.
          </p>

          <h3 className="section-title">A simple routine for you</h3>
          <div className="card">
            <h3>Morning</h3>
            <p>{report.routine.morning}</p>
          </div>
          <div className="card">
            <h3>Evening</h3>
            <p>{report.routine.evening}</p>
          </div>
          <div className="card">
            <h3>Things to avoid</h3>
            <p>{report.routine.avoid}</p>
          </div>

          <AdviceSection report={report} />

          {report.seeDermatologist && (
            <div className="derm">
              <strong>Please talk to a dermatologist.</strong> What we detected is the
              kind of acne that home products alone usually cannot fix, and a
              professional can. Seeing one early is the best way to prevent scarring.
              You deserve that care.
            </div>
          )}

          <footer>
            <div className="note">{report.disclaimer}</div>
          </footer>
        </section>
      )}

      {stage.kind === "idle" && (
        <p className="status">Your report will appear here after you add a photo.</p>
      )}
    </main>
  );
}
