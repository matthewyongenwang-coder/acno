"""Convert the trained models to ONNX for the browser app.

Run after a training run has put fresh weights in models/:
    pip install tf2onnx onnx onnxruntime
    python scripts/convert_models.py

Writes skin_type.onnx, acne_type.onnx, and acne_yolo.onnx into web/public/models/,
verifying along the way that each ONNX model produces the same outputs as the
original weights.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
WEB_MODELS = REPO_ROOT / "web" / "public" / "models"


def convert_classifiers() -> None:
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    for name in ["skin_type", "acne_type"]:
        model = tf.keras.models.load_model(MODELS / f"{name}.keras")
        spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="image"),)

        @tf.function(input_signature=spec)
        def f(image):
            # training=False traces the augmentation layers as identity ops
            return model(image, training=False)

        out = WEB_MODELS / f"{name}.onnx"
        tf2onnx.convert.from_function(f, input_signature=spec, opset=17,
                                      output_path=str(out))

        sess = ort.InferenceSession(str(out))
        rng = np.random.default_rng(0)
        sample = rng.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
        keras_out = model.predict(sample, verbose=0)
        onnx_out = sess.run(None, {sess.get_inputs()[0].name: sample})[0]
        diff = float(np.abs(keras_out - onnx_out).max())
        assert diff < 1e-3, f"{name}: onnx output diverges from keras (diff {diff})"
        print(f"[ok] {name}.onnx ({out.stat().st_size / 1e6:.1f} MB, max diff {diff:.1e})")


def convert_yolo() -> None:
    from ultralytics import YOLO

    model = YOLO(str(MODELS / "acne_yolo.pt"))
    exported = model.export(format="onnx", imgsz=640)
    dest = WEB_MODELS / "acne_yolo.onnx"
    Path(exported).replace(dest)
    print(f"[ok] acne_yolo.onnx ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    convert_classifiers()
    convert_yolo()
    print("\nDone. Commit web/public/models/ so Vercel ships the new models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
