"""Ridge-region estimation, quality gates, and RootSIFT template creation."""

from __future__ import annotations

import math
from pathlib import Path

import cv2 as cv
import numpy as np

from .config import MatcherConfig
from .errors import FingerprintMatchingError, FingerprintQualityError
from .templates import FingerprintTemplate


def _as_gray_uint8(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        path = Path(image)
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
        except OSError as exc:
            raise FingerprintQualityError(f"Could not read image: {path}") from exc
        data = cv.imdecode(encoded, cv.IMREAD_UNCHANGED) if encoded.size else None
        if data is None:
            raise FingerprintQualityError(f"Could not decode image: {path}")
    else:
        data = np.asarray(image)

    if data.size == 0:
        raise FingerprintQualityError("Fingerprint image is empty")
    if data.dtype != np.uint8:
        finite = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)
        data = cv.normalize(finite, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if data.ndim == 2:
        gray = data
    elif data.ndim == 3 and data.shape[2] == 3:
        gray = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
    elif data.ndim == 3 and data.shape[2] == 4:
        gray = cv.cvtColor(data, cv.COLOR_BGRA2GRAY)
    else:
        raise FingerprintQualityError(
            f"Expected a grayscale or color image, got shape {data.shape}"
        )
    return np.ascontiguousarray(gray)


def _resize_if_needed(gray: np.ndarray, maximum: int) -> np.ndarray:
    height, width = gray.shape
    largest = max(height, width)
    if largest <= maximum:
        return gray
    scale = maximum / float(largest)
    return cv.resize(
        gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv.INTER_AREA,
    )


def _fingerprint_roi(gray: np.ndarray) -> np.ndarray:
    """Estimate a filled ridge region while excluding crop/background edges."""

    height, width = gray.shape
    if min(height, width) < 12:
        return np.full_like(gray, 255)

    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    _threshold, dark = cv.threshold(
        blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    )
    # Link ridge fragments into a region, then fill the largest exterior hulls.
    radius = max(1, round(min(height, width) * 0.025))
    radius = min(radius, 15)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    connected = cv.morphologyEx(dark, cv.MORPH_CLOSE, kernel)
    connected = cv.dilate(connected, kernel, iterations=1)
    contours, _ = cv.findContours(connected, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    roi = np.zeros_like(gray)
    minimum_area = height * width * 0.01
    contours = [item for item in contours if cv.contourArea(item) >= minimum_area]
    if contours:
        contours.sort(key=cv.contourArea, reverse=True)
        kept: list[np.ndarray] = []
        area = 0.0
        for contour in contours[:4]:
            kept.append(cv.convexHull(contour))
            area += cv.contourArea(contour)
            if area >= height * width * 0.70:
                break
        cv.drawContours(roi, kept, -1, 255, cv.FILLED)

    fraction = float(np.count_nonzero(roi) / roi.size)
    if fraction > 0.98:
        # On tightly cropped sensor images the fingerprint can legitimately
        # fill the frame. Retaining a narrow margin avoids border keypoints.
        # A tiny or empty detected region must stay tiny so the caller's
        # minimum-ROI gate can reject a distant/partial fingerprint.
        roi.fill(255)
        margin = max(1, round(min(height, width) * 0.02))
        roi[:margin, :] = 0
        roi[-margin:, :] = 0
        roi[:, :margin] = 0
        roi[:, -margin:] = 0
    elif fraction > 0.0:
        erosion = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        roi = cv.erode(roi, erosion, iterations=1)
    return roi


def _image_quality(gray: np.ndarray, roi: np.ndarray) -> float:
    pixels = gray[roi != 0]
    if pixels.size < 16:
        return 0.0
    low, high = np.percentile(pixels, (5, 95))
    contrast = float(np.clip((high - low) / 160.0, 0.0, 1.0))
    laplacian_variance = float(cv.Laplacian(gray, cv.CV_32F).var())
    sharpness = laplacian_variance / (laplacian_variance + 350.0)
    otsu_value, _ = cv.threshold(
        pixels.reshape(-1, 1), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
    )
    dark_fraction = float(np.mean(pixels < otsu_value))
    ridge_balance = float(np.clip(1.0 - abs(dark_fraction - 0.5) / 0.5, 0.0, 1.0))
    return float(np.clip(0.42 * contrast + 0.38 * sharpness + 0.20 * ridge_balance, 0, 1))


def _orientation_coherence(gray: np.ndarray, roi: np.ndarray) -> float:
    """Measure whether gradients form a locally consistent fingerprint ridge flow."""

    gradient_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    sigma = float(np.clip(min(gray.shape) / 64.0, 3.0, 12.0))
    xx = cv.GaussianBlur(gradient_x * gradient_x, (0, 0), sigma)
    yy = cv.GaussianBlur(gradient_y * gradient_y, (0, 0), sigma)
    xy = cv.GaussianBlur(gradient_x * gradient_y, (0, 0), sigma)
    coherence = np.sqrt((xx - yy) ** 2 + 4.0 * xy * xy) / (xx + yy + 1e-6)
    selected = roi != 0
    values = coherence[selected]
    if values.size < 16:
        return 0.0
    energy = (xx + yy)[selected]
    energy_sum = float(energy.sum())
    if energy_sum <= 1e-6:
        return 0.0
    # Energy weighting focuses on actual ridge transitions rather than the
    # pepper-like low-energy regions visible in phone-camera crops.
    return float(np.sum(values * (energy / energy_sum)))


def create_template(
    image: str | Path | np.ndarray,
    config: MatcherConfig | None = None,
    *,
    quality_hint: float | None = None,
    source: str | None = None,
) -> FingerprintTemplate:
    """Create a RootSIFT template from one cropped fingerprint image."""

    config = config or MatcherConfig()
    gray = _resize_if_needed(_as_gray_uint8(image), config.max_image_dimension)
    roi = _fingerprint_roi(gray)
    roi_fraction = float(np.count_nonzero(roi) / roi.size)
    if roi_fraction < config.min_roi_fraction:
        raise FingerprintQualityError(
            f"Fingerprint ridge region is too small ({roi_fraction:.3f} < "
            f"{config.min_roi_fraction:.3f})"
        )
    base_quality = _image_quality(gray, roi)
    orientation_coherence = _orientation_coherence(gray, roi)
    if orientation_coherence < config.min_orientation_coherence:
        raise FingerprintQualityError(
            "Image does not contain sufficiently coherent fingerprint-like "
            f"ridge flow ({orientation_coherence:.3f} < "
            f"{config.min_orientation_coherence:.3f})"
        )

    if not hasattr(cv, "SIFT_create"):
        raise FingerprintMatchingError(
            "This OpenCV build does not provide SIFT; install a current "
            "opencv-python or opencv-contrib-python package"
        )
    sift = cv.SIFT_create(
        nfeatures=config.max_features,
        contrastThreshold=config.sift_contrast_threshold,
        edgeThreshold=config.sift_edge_threshold,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, roi)
    count = len(keypoints) if keypoints is not None else 0
    if descriptors is None or count < config.min_keypoints:
        raise FingerprintQualityError(
            f"Only {count} usable keypoints were found; at least "
            f"{config.min_keypoints} are required"
        )

    descriptors = np.asarray(descriptors, dtype=np.float32)
    # RootSIFT: L1-normalize the non-negative SIFT histogram and apply the
    # Hellinger square root.  Euclidean distance then approximates Hellinger.
    descriptors /= np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12)
    descriptors = np.sqrt(descriptors, out=descriptors)
    points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)

    feature_quality = min(1.0, count / max(32.0, config.min_keypoints * 3.0))
    quality = 0.78 * base_quality + 0.22 * feature_quality
    if quality_hint is not None:
        if not 0.0 <= quality_hint <= 1.0:
            raise ValueError("quality_hint must be in [0, 1]")
        quality = math.sqrt(max(0.0, quality * quality_hint))
    quality = float(np.clip(quality, 0.0, 1.0))
    if quality < config.min_quality:
        raise FingerprintQualityError(
            f"Fingerprint quality {quality:.3f} is below the configured minimum "
            f"of {config.min_quality:.3f}"
        )

    return FingerprintTemplate(
        keypoints=np.ascontiguousarray(points, dtype="<f4"),
        descriptors=np.ascontiguousarray(descriptors, dtype="<f4"),
        image_shape=(int(gray.shape[0]), int(gray.shape[1])),
        quality=quality,
        roi_fraction=roi_fraction,
        source=source,
    )
