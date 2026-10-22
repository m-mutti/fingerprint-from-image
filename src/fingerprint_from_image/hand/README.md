# `fingerprint_from_image.hand` — learned fingertip localisation

From an RGB photograph of one or more hands, find each fingertip and return
upright, contrast-normalised distal-phalanx crops with anatomical labels
(`right_index`, `left_thumb`, …). Runtime dependencies: **numpy, Pillow,
onnxruntime** — no OpenCV, no `mediapipe` package.

```text
RGB ─► letterbox 192² ─► BlazePalm (ONNX) ─► SSD decode + weighted NMS ─► palm box + 7 keypoints
    ─► rotated hand ROI (wrist→middle-MCP up, ×2.6) ─► affine warp 224² ─► hand landmarks (ONNX)
    ─► 21 landmarks + handedness + presence ─► ROI re-fit on landmarks, 2nd pass (refine_iterations)
    ─► per finger: DIP→tip oriented box ─► bicubic sample from the ORIGINAL image (native resolution)
    ─► illumination flattening + CLAHE ─► ridge-visibility quality ─► FingerCrop(image, label, quality, polygon)
```

| Module | Content |
|---|---|
| `anchors.py` | SSD anchors (2016 for BlazePalm), box/keypoint decoding, MediaPipe weighted NMS |
| `geometry.py` | letterbox mapping, `RotatedRect`, palm→ROI and landmarks→ROI, Pillow affine warps |
| `runtime.py` | model registry entries (SHA-256 verified) and ONNX Runtime wrappers |
| `topology.py` | landmark indices, handedness → label mapping, palm/back-of-hand test |
| `crops.py` | distal-phalanx boxes and upright crop sampling |
| `enhance.py` | numpy blur/CLAHE, illumination flattening, `ridge_quality` |
| `detector.py` | `HandFingerDetector`, `DetectorConfig`, `HandResult`, `HandNotFoundError` |
| `visualize.py` | `draw_hands` (landmarks + crop polygons) and `crop_mosaic` |

## Models

Google's MediaPipe *Hand Landmarker* (`hand_landmarker.task`, float16 v1,
Apache-2.0) converted to ONNX by `tools/hand_models/convert.py`. Weights are
not shipped or downloaded automatically. Run the converter once, then put the
two files in the model cache (`$FINGERPRINT_FROM_IMAGE_MODELS`, default
`~/.cache/fingerprint-from-image`) or pass `model_dir`.

| Registry name | Input | Outputs | SHA-256 |
|---|---|---|---|
| `mediapipe_hand_detector.onnx` | `(1,192,192,3)` RGB in [0,1] | 2016×18 regressors, 2016 logits | `615009f3…5a94` |
| `mediapipe_hand_landmarks_detector.onnx` | `(1,224,224,3)` RGB in [0,1] | 21×3 landmarks, presence, P(right), 21×3 world | `27953c21…2b25` |

## Usage

```python
from fingerprint_from_image.io import load_image
from fingerprint_from_image.hand import DetectorConfig, HandFingerDetector, HandNotFoundError
from fingerprint_from_image.hand import crop_mosaic, draw_hands

detector = HandFingerDetector(model_dir="runs/hand_models",
                              config=DetectorConfig(max_hands=2, mirrored=False))
rgb = load_image("hand.jpg", mode="RGB")          # EXIF orientation applied
try:
    crops = detector(rgb, source="hand.jpg")      # list[FingerCrop], 5 per hand
except HandNotFoundError:
    crops = []
for crop in crops:
    print(crop.label, crop.quality, crop.image.shape, crop.polygon)

hands = detector.detect_hands(rgb)                # landmarks, side, palm_facing, ROI
draw_hands(rgb, hands, crops).save("annotated.jpg")
crop_mosaic(crops).save("crops.png")
```

`python tools/hand_models/demo.py --model-dir <dir> --out-dir <dir> images…`
writes annotated images, crops, and a JSON report with timings.

## Labels, handedness and mirrored images

* The landmark model outputs `p = P(right hand)` judged on the image **as
  given**; `side = "right" if p > 0.5`. Measured on the test photos:
  right palms `p = 0.98`, a left palm `p = 0.12`, back of a left hand
  `p = 0.01`. It equals the score MediaPipe Tasks reports as `"Right"`
  (cross-checked with `mediapipe==0.10.14`).
* A **mirrored** image (front-camera preview, flipped scan) shows the other
  hand's chirality. Pass `DetectorConfig(mirrored=True)` to swap the labels.
  Flipping a test image horizontally flips every label and leaves
  `palm_facing` unchanged (covered by the integration tests).
* `HandResult.palm_facing` combines the model's side with the 2-D chirality
  of wrist / index-MCP / little-MCP. `False` means the back of the hand faces
  the camera, so the crops show nails rather than pads.
  `DetectorConfig(require_palm_side=True)` drops such hands. `None` means the
  hand is seen edge-on.
* Labels are `{left,right}_{thumb,index,middle,ring,little}`, matching
  `fingerprint_from_image.types.FINGER_LABELS`.

## Crop geometry

For finger *f* the box axis runs from the DIP joint (IP for the thumb) through
the tip landmark, and the tip end is at the top of the crop. Defaults
(`CropConfig`):

* length = |DIP→tip| × (1 + `tip_margin` 0.35 + `base_margin` 0.05); the tip
  landmark lies inside the fingertip outline, hence the larger tip margin;
* finger width = 0.85 × mean spacing of the neighbouring MCP knuckles. For the
  thumb it is 1.0 × the index–middle spacing. Knuckle spacing does not change
  when fingers spread. Width is clamped to [0.55, 1.6] × segment length, and
  `side_margin` adds 8 % on each side;
* if the finger points at the camera (short 2-D segment), the length is kept
  at ≥ 0.9 × width;
* pixels are sampled bicubically from the **original** photo at native
  resolution (`output_width` resamples to a fixed finger width);
  `FingerCrop.polygon` holds the corners in source pixels, clockwise from the
  upright crop's top-left. Parts outside the photo are filled with the
  crop median and reduce `quality` proportionally.

## Normalisation and quality

* `normalize_fingertip`: divide by a wide Gaussian (σ = 2.5 ridge periods) to
  remove shading, standardise local contrast, then CLAHE with tiles about
  8 periods wide (numpy implementation with bilinear LUT interpolation).
* `ridge_quality` ∈ [0, 1] = resolution × √(band energy × coherence), measured on
  the central 60 % of the crop:
  * *resolution* ramps from 0 at 2.5 px per ridge period to 1 at 5 px;
  * *band energy* is the share of non-illumination spectral energy within
    ×/÷1.6 of the expected ridge frequency, rescaled so white noise scores 0;
  * *coherence* is the structure-tensor coherence of the band-passed signal.
    Ridges are locally parallel; noise and skin texture are not.

  The expected period comes from size priors: a 16 mm distal width and a
  0.45 mm ridge period, i.e. period ≈ finger width in px / 35.6.
  Measured values:

  | Input | Quality |
  |---|---|
  | synthetic ridges with sensor noise | 0.88 |
  | same, blurred σ = 3 px (ridge contrast below noise) | 0.00 |
  | white noise / smooth skin texture / flat | 0.00 |
  | real fingertip macro (`tests/data/fingertip_macro_pad.jpg`, ≈15 px period) | 0.61 |
  | same, blurred σ = 4 px | 0.00 |
  | same, downsampled ×4 (≈2.9 px period) | 0.00 |
  | whole-hand test photos (finger 60–120 px wide, period ≈ 2–3 px) | 0.00 |

  The score is a *visibility* measure, not a matcher-calibrated quality (cf.
  NFIQ 2); calibrate thresholds on your own capture set before relying on it.

## Requirements and limitations

* **Ridges need resolution.** Adult ridge period is ≈ 0.4–0.5 mm. The score
  reaches 1 at ≥ 5 px per period, i.e. **≥ ~180 px across a 16 mm fingertip**
  (≈ 280 ppi). Below ~90 px across (2.5 px per period) ridges are not
  resolved and quality is 0. For matching against 500 ppi contact prints, aim
  for **≥ 300 px across each fingertip**. A whole-hand photo needs the hand to
  fill a 12 MP frame to reach that; close-ups of 2–4 fingers are better.
* Ridges also need **focus** (fingers lie at different depths, and depth of
  field is shallow at close range) and **diffuse, even light**: glare
  saturates ridges, and hard side light gives shading that CLAHE only partly
  removes. Motion blur and JPEG compression remove ridge frequencies first.
* The palm detector misses strongly blurred, very large or heavily cropped
  hands (the left hand in `tests/data/hands_two_palms.jpg` is not detected).
  Official MediaPipe misses it too.
* Landmarks are model estimates (mean error ≈ 5 px against official
  MediaPipe on the test photos). The tip landmark is not the pad centre, so
  crops include some background at the corners and beyond the tip; no
  finger-skin segmentation is applied.
* Thumbs are often photographed side-on; their crops then show the side of the
  pad. Strongly bent or foreshortened fingers give shorter boxes.
* Handedness can be wrong for ambiguous poses (edge-on hands, fists, gloves),
  and it depends on declaring mirrored images correctly.
* Contactless photos are the mirror image of contact impressions (live-scan /
  ink). Flip crops horizontally (`np.fliplr`) before matching them against
  contact-based templates. The crops here are *not* flipped.
* No presentation-attack detection: a photo of a hand gives crops just as a
  real hand does.
