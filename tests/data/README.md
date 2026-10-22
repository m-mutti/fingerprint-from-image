# Test images

All images come from Wikimedia Commons under a CC0 dedication or a public-domain
release. They were downscaled and re-encoded as JPEG (no EXIF kept) to stay
under 300 KB. No biometric enrolment data is derived from them; they only
exercise hand localisation and ridge-visibility scoring.

| File | Source (Commons page) | Author | Licence | Processing | Content / ground truth |
|------|-----------------------|--------|---------|------------|------------------------|
| `hand_palm_open.jpg` | [File:Boys palm 手掌 手板心.png](https://commons.wikimedia.org/wiki/File:Boys_palm_%E6%89%8B%E6%8E%8C_%E6%89%8B%E6%9D%BF%E5%BF%83.png) | Jiajitan | CC0 1.0 | 1280 px thumbnail, alpha composited on black, 855×960, JPEG q88 | One open palm facing the camera, thumb on the image left → **left** hand (non-mirrored) |
| `hand_dorsal_left.jpg` | [File:Human hand.jpg](https://commons.wikimedia.org/wiki/File:Human_hand.jpg) | FrognerSvart | CC0 1.0 | 1280 px thumbnail, 720×960, JPEG q88 | Back of a **left** hand on a white wall (nails visible, no pads) |
| `hands_two_palms.jpg` | [File:Hand comparison.jpg](https://commons.wikimedia.org/wiki/File:Hand_comparison.jpg) | Tomwsulcer | CC0 1.0 | original 867×567, JPEG q88 | Two **right** palms; the left one is blurred and is *not* detected by the palm model (documents a limitation) |
| `fingertip_macro_pad.jpg` | [File:Smd led fingertip macro still.jpg](https://commons.wikimedia.org/wiki/File:Smd_led_fingertip_macro_still.jpg) | Fressmotron | Public domain (released by the author) | crop (380,170)–(840,540), rotated 90° so the tip points up, 370×460, JPEG q90 | Close-up fingertip with clearly visible ridges (≈15 px period); used for the quality-score test |
