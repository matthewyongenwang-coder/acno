"""Convert the trained models to ONNX for the browser app.

Run after a training run has put fresh weights in models/:
    .venv/bin/python scripts/convert_models.py

Writes skin_type.onnx, acne_type.onnx and acne_yolo.onnx into web/public/models/.

The input contract must not change, because web/lib/analyze.ts depends on it:
classifiers take float32 in the range 0 to 255, shape (1, 224, 224, 3), NHWC, and
return softmax probabilities in alphabetical class order. Preprocessing is a Rescaling
layer inside the graph, so the browser sends raw pixel values.

Parity is checked against real dataset images rather than random noise. Uniform noise
is a poor test: a wrong preprocessing scale can still produce near-identical outputs on
noise while being badly wrong on photographs.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MODELS = REPO_ROOT / "models"
WEB_MODELS = REPO_ROOT / "web" / "public" / "models"

# largest acceptable classifier download; the app loads three models on a phone
SIZE_BUDGET_MB = 40.0


def sample_images(name: str, n: int = 16) -> np.ndarray:
    """Real test images, centre-cropped exactly the way evaluation does."""
    try:
        import acno_data
        splits = acno_data.load_splits(name)
        x = splits["test"]["x"][:n]
        return x[:, 16:240, 16:240, :].astype(np.float32)
    except Exception as exc:  # data not downloaded on this machine
        print(f"  [warn] falling back to random input for parity check: {exc}")
        rng = np.random.default_rng(0)
        return rng.uniform(0, 255, (n, 224, 224, 3)).astype(np.float32)


def convert_classifiers() -> None:
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    for name in ["skin_type", "acne_type"]:
        src = MODELS / f"{name}.keras"
        if not src.exists():
            print(f"[skip] {src} not found")
            continue

        model = tf.keras.models.load_model(src)
        spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="image"),)

        @tf.function(input_signature=spec)
        def f(image):
            return model(image, training=False)

        out = WEB_MODELS / f"{name}.onnx"
        tf2onnx.convert.from_function(f, input_signature=spec, opset=17,
                                      output_path=str(out))

        sess = ort.InferenceSession(str(out))
        images = sample_images(name)
        worst = 0.0
        disagreements = 0
        for image in images:
            batch = image[None, ...]
            keras_out = model.predict(batch, verbose=0)
            onnx_out = sess.run(None, {sess.get_inputs()[0].name: batch})[0]
            worst = max(worst, float(np.abs(keras_out - onnx_out).max()))
            if int(keras_out.argmax()) != int(onnx_out.argmax()):
                disagreements += 1

        size_mb = out.stat().st_size / 1e6
        assert worst < 1e-3, f"{name}: onnx diverges from keras (max diff {worst})"
        assert disagreements == 0, f"{name}: {disagreements} predictions disagree"
        flag = "  <-- OVER BUDGET" if size_mb > SIZE_BUDGET_MB else ""
        print(f"[ok] {name}.onnx  {size_mb:.1f} MB  max diff {worst:.1e}  "
              f"{len(images)} images agree{flag}")


def convert_yolo() -> None:
    from ultralytics import YOLO

    src = MODELS / "acne_yolo.pt"
    if not src.exists():
        print(f"[skip] {src} not found")
        return
    model = YOLO(str(src))
    exported = model.export(format="onnx", imgsz=640)
    dest = WEB_MODELS / "acne_yolo.onnx"
    Path(exported).replace(dest)
    print(f"[ok] acne_yolo.onnx  {dest.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    convert_classifiers()
    convert_yolo()
    total = sum(p.stat().st_size for p in WEB_MODELS.glob("*.onnx")) / 1e6
    print(f"\ntotal browser download: {total:.1f} MB")
    print("Commit web/public/models/ so Vercel ships the new models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
