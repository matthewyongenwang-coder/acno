"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { analyze, warmUp, type Report } from "@/lib/analyze";
import { SEVERITY_TEXT } from "@/lib/guide";
import type { AdviceResponse, Profile } from "@/app/api/advice/route";

type AdviceState =
  | { kind: "gate" }
  | { kind: "form" }
  | { kind: "loading" }
  | { kind: "ready"; advice: AdviceResponse }
  | { kind: "unavailable" };

const AGE_RANGES = [
  "Under 13",
  "13 to 15",
  "16 to 18",
  "19 to 24",
  "25 or older",
];

const BACKGROUNDS = [
  "East Asian",
  "South Asian",
  "Southeast Asian",
  "Black or African",
  "Middle Eastern or North African",
  "Hispanic or Latino",
  "White or European",
  "Indigenous",
  "Mixed background",
  "Prefer not to say",
];

const ACTIVITIES = [
  "Swimming",
  "Sports with a helmet or pads",
  "Gym or weights",
  "Running or outdoor sports",
  "Dance",
  "Mostly indoors right now",
];

const ROUTINE = [
  "Nothing right now",
  "Just water",
  "Cleanser",
  "Moisturizer",
  "Sunscreen",
  "Benzoyl peroxide",
  "Salicylic acid",
  "Adapalene or a retinoid",
  "Makeup most days",
  "Something a doctor prescribed",
];

// A group of options you can pick more than one of, with room to write in
// anything the options did not cover.
function ChipGroup({
  legend,
  help,
  options,
  selected,
  onToggle,
  extra,
  onExtra,
  extraLabel,
}: {
  legend: string;
  help?: string;
  options: string[];
  selected: string[];
  onToggle: (value: string) => void;
  extra: string;
  onExtra: (value: string) => void;
  extraLabel: string;
}) {
  return (
    <fieldset className="field">
      <legend className="field-label">{legend}</legend>
      {help ? <p className="field-help">{help}</p> : null}
      <div className="chips">
        {options.map((option) => (
          <button
            type="button"
            key={option}
            className={`chip ${selected.includes(option) ? "on" : ""}`}
            aria-pressed={selected.includes(option)}
            onClick={() => onToggle(option)}
          >
            {option}
          </button>
        ))}
      </div>
      <input
        className="text-input"
        type="text"
        value={extra}
        aria-label={`${legend}: ${extraLabel}`}
        placeholder={extraLabel}
        maxLength={200}
        onChange={(event) => onExtra(event.target.value)}
      />
    </fieldset>
  );
}

// Pick exactly one.
function RadioGroup({
  legend,
  help,
  options,
  value,
  onChange,
}: {
  legend: string;
  help?: string;
  options: string[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <fieldset className="field">
      <legend className="field-label">{legend}</legend>
      {help ? <p className="field-help">{help}</p> : null}
      <div className="chips">
        {options.map((option) => (
          <button
            type="button"
            key={option}
            className={`chip ${value === option ? "on" : ""}`}
            aria-pressed={value === option}
            onClick={() => onChange(value === option ? "" : option)}
          >
            {option}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

function IntakeForm({
  onSubmit,
  onSkip,
}: {
  onSubmit: (profile: Profile) => void;
  onSkip: () => void;
}) {
  const [ageRange, setAgeRange] = useState("");
  const [background, setBackground] = useState("");
  const [backgroundExtra, setBackgroundExtra] = useState("");
  const [activities, setActivities] = useState<string[]>([]);
  const [activitiesExtra, setActivitiesExtra] = useState("");
  const [routine, setRoutine] = useState<string[]>([]);
  const [routineExtra, setRoutineExtra] = useState("");
  const [notes, setNotes] = useState("");

  // Updater form, not the captured array: two quick taps in the same group
  // would otherwise both read the pre-click state and the second would undo
  // the first.
  const toggle = (value: string) => (list: string[]) =>
    list.includes(value) ? list.filter((v) => v !== value) : [...list, value];

  const submit = () => {
    const withExtra = (list: string[], extra: string) => {
      const cleaned = extra.trim();
      return cleaned ? [...list, cleaned] : list;
    };
    onSubmit({
      ageRange: ageRange || undefined,
      background: [background, backgroundExtra.trim()]
        .filter(Boolean)
        .join(", ") || undefined,
      activities: withExtra(activities, activitiesExtra),
      routine: withExtra(routine, routineExtra),
      notes: notes.trim() || undefined,
    });
  };

  return (
    <div className="intake">
      <p className="intake-intro">
        The photo can only show what your skin looks like, not why. A few
        optional answers let the guide explain the why. Every question can be
        left blank.
      </p>

      <RadioGroup
        legend="How old are you?"
        help="Skin changes a lot through puberty, and some ingredients are not meant for younger skin."
        options={AGE_RANGES}
        value={ageRange}
        onChange={setAgeRange}
      />

      <RadioGroup
        legend="Your background"
        help="Deeper skin tones are more likely to be left with dark marks after a spot heals, which changes what is worth suggesting."
        options={BACKGROUNDS}
        value={background}
        onChange={setBackground}
      />
      <input
        className="text-input"
        type="text"
        value={backgroundExtra}
        aria-label="Describe your background in your own words"
        placeholder="Or describe it yourself"
        maxLength={200}
        onChange={(event) => setBackgroundExtra(event.target.value)}
      />

      <ChipGroup
        legend="Sports and activities"
        help="Sweat, friction, helmet straps, and chlorine all show up on skin in specific places."
        options={ACTIVITIES}
        selected={activities}
        onToggle={(value) => setActivities(toggle(value))}
        extra={activitiesExtra}
        onExtra={setActivitiesExtra}
        extraLabel="Anything else you do regularly"
      />

      <ChipGroup
        legend="What you already put on your skin"
        help="Knowing this avoids suggesting something you are doing already, or something that clashes with it."
        options={ROUTINE}
        selected={routine}
        onToggle={(value) => setRoutine(toggle(value))}
        extra={routineExtra}
        onExtra={setRoutineExtra}
        extraLabel="Specific products or anything else you use"
      />

      <fieldset className="field">
        <legend className="field-label">Anything else worth knowing</legend>
        <p className="field-help">
          Stress, sleep, a diet change, periods or hormones, medication, family
          history, or what seems to make it worse.
        </p>
        <textarea
          className="textarea"
          rows={4}
          value={notes}
          maxLength={400}
          placeholder="In your own words"
          onChange={(event) => setNotes(event.target.value)}
        />
      </fieldset>

      <div className="note">
        Your answers are sent as text to the AI along with your scan numbers, so
        it can reason about causes. Your photo is never sent, and nothing here is
        saved. Leave anything blank that you would rather not share.
      </div>

      <div className="intake-actions">
        <button type="button" className="button" onClick={submit}>
          Write my report
        </button>
        <button type="button" className="button quiet" onClick={onSkip}>
          Skip, use my scan only
        </button>
      </div>
    </div>
  );
}

function AdviceSection({ report }: { report: Report }) {
  const [state, setState] = useState<AdviceState>({ kind: "gate" });
  const dialogRef = useRef<HTMLDialogElement>(null);

  // A native modal dialog, so focus trapping, Escape, and inertness of the page
  // behind it are the browser's job rather than ours.
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (state.kind === "form" && !dialog.open) dialog.showModal();
    if (state.kind !== "form" && dialog.open) dialog.close();
  }, [state.kind]);

  // showModal blocks interaction behind it but not scrolling, so hold the page.
  useEffect(() => {
    if (state.kind !== "form") return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [state.kind]);

  const run = useCallback(
    (profile?: Profile) => {
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
          profile,
        }),
      })
        .then(async (response) => {
          if (!response.ok) throw new Error(String(response.status));
          return (await response.json()) as AdviceResponse;
        })
        .then((advice) => setState({ kind: "ready", advice }))
        .catch(() => setState({ kind: "unavailable" }));
    },
    [report],
  );

  // A new scan means the old report no longer applies.
  useEffect(() => {
    setState({ kind: "gate" });
  }, [report]);

  if (state.kind === "unavailable") {
    return (
      <section aria-label="AI skin guide">
        <div className="note">
          The AI guide is not available right now. Everything above still stands,
          and the routine below is written from your scan.
        </div>
      </section>
    );
  }

  // The form floats above the page, so the gate card stays where it was.
  const modal = (
    <dialog
      className="modal"
      ref={dialogRef}
      aria-label="A little about you"
      onCancel={() => setState({ kind: "gate" })}
      onClose={() => {
        setState((current) => (current.kind === "form" ? { kind: "gate" } : current));
      }}
    >
      {/* data-lenis-prevent keeps smooth scrolling off this inner panel */}
      <div className="modal-panel" data-lenis-prevent>
        <div className="modal-head">
          <h3>A little about you</h3>
          <button
            type="button"
            className="modal-close"
            aria-label="Close"
            onClick={() => setState({ kind: "gate" })}
          >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
              <path d="M6 6 L18 18 M18 6 L6 18" />
            </svg>
          </button>
        </div>
        {state.kind === "form" ? (
          <IntakeForm onSubmit={(profile) => run(profile)} onSkip={() => run()} />
        ) : null}
      </div>
    </dialog>
  );

  if (state.kind === "gate" || state.kind === "form") {
    return (
      <section aria-label="AI skin guide">
        <div className="gate">
          <h3>Want to know why?</h3>
          <p>
            The scan above says what your skin is doing. Answer a few optional
            questions about yourself and the AI will explain why it is doing it,
            then build a routine around your answers.
          </p>
          <button
            type="button"
            className="button"
            onClick={() => setState({ kind: "form" })}
          >
            See AI analysis and recommendations
          </button>
        </div>
        {modal}
      </section>
    );
  }

  return (
    <section aria-label="AI skin guide">
      <h3 className="section-title">What our AI makes of it</h3>
      {state.kind === "loading" ? (
        <div className="status">
          <div className="spinner" aria-hidden />
          <p className="sub">Reading your answers and writing your report...</p>
        </div>
      ) : (
        <>
          <div className="card">
            <p>{state.advice.analysis}</p>
          </div>

          {state.advice.causes?.length ? (
            <>
              <h3 className="section-title">Why your skin is like this</h3>
              <ol className="causes">
                {state.advice.causes.map((cause, index) => (
                  <li className="cause" key={cause.factor}>
                    <span className="cause-index">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <h4>{cause.factor}</h4>
                      <p>{cause.why}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </>
          ) : null}
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

// Tell the privacy-safe counter that one scan completed. Sends nothing but the
// increment itself, and quietly does nothing if the counter is not configured.
function recordScan() {
  fetch("/api/visits", { method: "POST" }).catch(() => {});
}

export default function Scan() {
  const [tab, setTab] = useState<"upload" | "camera">("upload");
  const [stage, setStage] = useState<Stage>({ kind: "idle" });
  const [dragging, setDragging] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [flashing, setFlashing] = useState(false);
  const [flashOn, setFlashOn] = useState(true);
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
      recordScan();
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
    const grab = () => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d")?.drawImage(video, 0, 0);
      stopCamera();
      setFlashing(false);
      canvas.toBlob((blob) => blob && handleBlob(blob), "image/jpeg", 0.95);
    };
    if (!flashOn) {
      grab();
      return;
    }
    // Flash on: paint the whole screen pure white first. The front camera has no
    // hardware torch, so it reflects the screen's own light to brighten the face.
    // Wait a beat so the white actually renders and the camera's auto-exposure
    // settles, then grab the (now better lit) frame.
    setFlashing(true);
    window.setTimeout(grab, 350);
  }, [flashOn, handleBlob, stopCamera]);

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
    <main className="tool">
      <div className="hero">
        <h1>Scan your skin</h1>
        <p>A private first answer, in your browser. Nothing is uploaded.</p>
      </div>

      <div className="note">
        Take or upload a clear photo of your face in good light. acno runs entirely in
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
          <div className="note" style={{ marginBottom: "1rem" }}>
            Good lighting matters. In a dim room the analysis can be inaccurate, so
            face a window or a lamp. Fit your whole face inside the outline so it
            fills the frame for the best detection.
          </div>
          {cameraOn ? (
            <div>
              <div className="camera-frame">
                <video ref={videoRef} playsInline muted />
                <div className="face-guide" aria-hidden>
                  <svg viewBox="0 0 200 260" preserveAspectRatio="xMidYMid meet">
                    <ellipse className="guide-stroke" cx="100" cy="96" rx="58" ry="74" />
                    <path className="guide-stroke" d="M20 260 Q100 176 180 260" />
                    <g className="guide-bracket">
                      <path d="M18 40 v-12 a10 10 0 0 1 10 -10 h12" />
                      <path d="M160 18 h12 a10 10 0 0 1 10 10 v12" />
                      <path d="M18 220 v12 a10 10 0 0 0 10 10 h12" />
                      <path d="M160 242 h12 a10 10 0 0 0 10 -10 v-12" />
                    </g>
                  </svg>
                </div>
              </div>
              <div className="camera-controls">
                <button className="button" onClick={capturePhoto}>
                  Take the photo
                </button>
                <button
                  type="button"
                  className={`flash-toggle${flashOn ? " on" : ""}`}
                  aria-pressed={flashOn}
                  onClick={() => setFlashOn((on) => !on)}
                >
                  Flash {flashOn ? "on" : "off"}
                </button>
              </div>
              <p className="flash-hint">
                {flashOn
                  ? "The screen flashes white as the photo is taken to light your face."
                  : "No flash. Best in a bright room or facing a window."}
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

      {flashing && <div className="screen-flash" aria-hidden />}

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
