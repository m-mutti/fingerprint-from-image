# Hand models: provenance and conversion

The runtime hand stage (`fingerprint_from_image.hand`) uses Google's MediaPipe
**Hand Landmarker** models converted from TFLite to ONNX. No model binaries
are committed; `convert.py` reproduces them byte-for-byte.

## Source

| Item | Value |
|---|---|
| Upstream bundle | `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` (identical to `…/float16/latest/…`; upstream `Last-Modified: 2023-04-26`) |
| Bundle SHA-256 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` (7 819 105 bytes, zip) |
| Licence | **Apache-2.0**: page 2 of the [MediaPipe Hands (Lite/Full) model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf) says "Licensed under Apache License, Version 2.0". The card is linked from the [Hand Landmarker docs](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models). It also specifies RGB input in [0.0, 1.0], a 192×192 detector with 2016×18 outputs, and a 224×224 tracker. Redistributions of the converted files must keep the Apache-2.0 notice. |
| Members | `hand_detector.tflite` (2 339 878 B, SHA-256 `945f713b…7a7d`, labels `palm`), `hand_landmarks_detector.tflite` (5 478 949 B, SHA-256 `6acda74a…b6b9`, `handedness.txt` = `Left\nRight`) |

## Outputs (registered in `fingerprint_from_image/hand/runtime.py`)

| File | Bytes | SHA-256 | Max \|TFLite − ONNX\| (random inputs) |
|---|---:|---|---|
| `mediapipe_hand_detector.onnx` | 4 545 248 | `615009f34258d557312a4eea5cab235c4fbeb3ada2a329a9bac036ead4545a94` | boxes 1.8e-4 (range ±127), logits 2.8e-5 |
| `mediapipe_hand_landmarks_detector.onnx` | 10 873 556 | `27953c218051f32d3b9e95eb55bc0e6f860947eaa25a637499bb8e181c632b25` | landmarks 1.1e-4 (px), presence 5e-8, handedness 2.3e-6, world 3e-7 |

The fp16 weights are de-quantised to fp32 by tf2onnx, so each ONNX file is
about twice the size of its TFLite source. Input normalisation is RGB in
`[0, 1]`, NHWC (with `[-1, 1]` input the palm model saturates every anchor).

## Reproduce

Conversion uses its own environment; the runtime never imports TensorFlow.

```bash
uv venv .tools-venv -p 3.11
uv pip install --python .tools-venv "tensorflow-cpu==2.15.1" "tf2onnx==1.16.1" \
    "onnx==1.16.2" "onnxruntime==1.19.2" "numpy<2" "protobuf==3.20.3"
.tools-venv/Scripts/python tools/hand_models/convert.py --out-dir runs/hand_models
# -> mediapipe_hand_detector.onnx, mediapipe_hand_landmarks_detector.onnx, manifest.json
```

`convert.py` downloads the pinned bundle (or `--task <local copy>`), checks
its SHA-256, converts both members with `tf2onnx.convert.from_tflite`
(opset 17), and canonicalises node, tensor and initializer names. tf2onnx
numbers generated names in hash-seed order, so without this step two runs
give different bytes. It then runs `onnx.checker`, measures TFLite-vs-ONNX
parity, and writes `manifest.json`. Two consecutive runs produced identical
SHA-256s.

Install for the runtime:

```bash
export FINGERPRINT_FROM_IMAGE_MODELS=$PWD/runs/hand_models   # or copy the two files to ~/.cache/fingerprint-from-image
python -m pytest tests/test_hand_integration.py              # skipped when the files are absent
```

## Cross-check against official MediaPipe (tools-only)

In a separate venv with `mediapipe==0.10.14` (never a runtime dependency),
`HandLandmarker` was run on the test photos.

* `hands_two_palms.jpg` and `hand_palm_open.jpg`: mean landmark distance to
  this ONNX/numpy pipeline is 5–6 px, with palms 250–450 px long. MediaPipe's
  handedness category and score (`Right` 0.98, `Left` 0.95) match our side
  and probability.
* MediaPipe needs `min_hand_detection_confidence ≈ 0.05` to report the two
  portrait test photos, which our letterboxed palm stage scores at 0.91–0.98.
  Its low-confidence ROI on `hand_dorsal_left.jpg` gives visibly worse
  landmarks than ours.

## Demo

```bash
python tools/hand_models/demo.py --model-dir runs/hand_models --out-dir runs/hand_demo tests/data/*.jpg
```
