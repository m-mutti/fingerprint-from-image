"""End-to-end enrollment, identification, and verification service."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import MatcherConfig
from .digests import _hand_crop_digest, _source_description, _source_digest
from .errors import FingerprintMatchingError, FingerprintQualityError
from .extraction import (
    ExtractionConfig,
    FingerprintCrop,
    FingerprintExtractionError,
    extract_fingerprints,
)
from .features import create_template
from .pairwise import match_templates
from .store import FingerprintStore
from .templates import (
    EnrollmentReport,
    FingerAssignment,
    FingerprintTemplate,
    IdentityMatch,
    MatchResult,
)

IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jp2",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _image_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    iterator: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


class FingerprintMatcher:
    """End-to-end enrollment and matching service backed by SQLite."""

    def __init__(
        self,
        database: str | Path | FingerprintStore = "fingerprints.db",
        config: MatcherConfig | None = None,
        extraction_config: ExtractionConfig | None = None,
    ) -> None:
        self.config = config or MatcherConfig()
        self.extraction_config = extraction_config or ExtractionConfig()
        self._owns_store = not isinstance(database, FingerprintStore)
        self.store = (
            database if isinstance(database, FingerprintStore) else FingerprintStore(database)
        )
        try:
            self.store.bind_feature_config(self.config)
            self.store.bind_extraction_config(self.extraction_config)
        except Exception:
            if self._owns_store:
                self.store.close()
            raise

    def __enter__(self) -> FingerprintMatcher:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_store:
            self.store.close()

    def clear(self) -> None:
        self.store.clear()

    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    def template(
        self,
        image: str | Path | np.ndarray,
        *,
        quality_hint: float | None = None,
    ) -> FingerprintTemplate:
        source = str(Path(image).resolve()) if isinstance(image, (str, Path)) else None
        return create_template(
            image,
            self.config,
            quality_hint=quality_hint,
            source=source,
        )

    def match(
        self,
        query: FingerprintTemplate | str | Path | np.ndarray,
        reference: FingerprintTemplate | str | Path | np.ndarray,
    ) -> MatchResult:
        """Low-level one-fingerprint match convenience method."""

        query_template = query if isinstance(query, FingerprintTemplate) else self.template(query)
        reference_template = (
            reference
            if isinstance(reference, FingerprintTemplate)
            else self.template(reference)
        )
        return match_templates(query_template, reference_template, self.config)

    def enroll_cropped(
        self,
        identity: str,
        image: str | Path | np.ndarray,
        *,
        finger: str = "unknown",
        source_digest: str | None = None,
    ) -> bool:
        """Incrementally enroll one cropped print; return false if duplicated."""

        template = self.template(image)
        digest = source_digest or _source_digest(image)
        source = str(Path(image).resolve()) if isinstance(image, (str, Path)) else None
        template_id = self.store.add_template(
            identity,
            finger,
            template,
            source_digest=digest,
            source_path=source,
        )
        return template_id is not None

    def _enroll_hand_crops(
        self,
        identity: str,
        image: str | Path | np.ndarray,
        crops: Sequence[FingerprintCrop],
    ) -> EnrollmentReport:
        report = EnrollmentReport()
        source_label = _source_description(image)
        if len(crops) < self.config.min_hand_fingers:
            report.add_error(
                source_label,
                f"hand extraction produced {len(crops)} crop(s); at least "
                f"{self.config.min_hand_fingers} are required",
            )
            return report
        source = str(Path(image).resolve()) if isinstance(image, (str, Path)) else None
        prepared: list[tuple[str, FingerprintTemplate]] = []
        for index, crop in enumerate(crops, start=1):
            finger = f"finger_{index}"
            try:
                prepared.append(
                    (
                        finger,
                        create_template(
                            crop.image,
                            self.config,
                            quality_hint=crop.quality,
                            source=source,
                        ),
                    )
                )
            except (FingerprintMatchingError, ValueError) as exc:
                report.add_error(f"{source_label}#{finger}", exc)
        if report.failed:
            # One hand capture is a single enrollment unit. Do not create a
            # sparse identity when any expected crop is unusable.
            return report

        base_digest = _source_digest(image)
        added: list[bool] = []
        try:
            with self.store.connection:
                self.store.connection.execute("BEGIN IMMEDIATE")
                for index, (finger, template) in enumerate(prepared, start=1):
                    template_id = self.store.add_template(
                        identity,
                        finger,
                        template,
                        source_digest=_hand_crop_digest(
                            base_digest, index, self.extraction_config
                        ),
                        source_path=source,
                        _within_transaction=True,
                    )
                    added.append(template_id is not None)
        except (FingerprintMatchingError, ValueError, sqlite3.Error) as exc:
            report.add_error(source_label, f"hand enrollment rolled back: {exc}")
            return report
        report.added = sum(added)
        report.skipped = len(added) - report.added
        return report

    def enroll_hand(
        self, identity: str, image: str | Path | np.ndarray
    ) -> EnrollmentReport:
        """Atomically extract and enroll every usable crop in a hand image."""

        try:
            crops = extract_fingerprints(image, self.extraction_config)
        except (FingerprintExtractionError, ValueError) as exc:
            report = EnrollmentReport()
            report.add_error(_source_description(image), exc)
            return report
        if not crops:
            report = EnrollmentReport()
            report.add_error(
                _source_description(image),
                "no fingerprints were extracted from the hand image",
            )
            return report
        return self._enroll_hand_crops(identity, image, crops)

    def enroll(
        self,
        identity: str,
        images: str | Path | np.ndarray | Sequence[str | Path | np.ndarray],
        *,
        finger: str = "unknown",
        kind: str = "cropped",
    ) -> EnrollmentReport:
        """Enroll one or more cropped prints or hand captures.

        ``kind`` may be ``"cropped"``, ``"hand"``, or ``"auto"``.  Auto mode
        treats an image as a hand only when extraction finds at least two crops.
        """

        if kind not in {"auto", "cropped", "hand"}:
            raise ValueError("kind must be 'auto', 'cropped', or 'hand'")
        if isinstance(images, (str, Path, np.ndarray)):
            sources: Sequence[str | Path | np.ndarray] = [images]
        else:
            sources = images
        report = EnrollmentReport()
        for image in sources:
            actual_kind = kind
            crops: list[FingerprintCrop] | None = None
            if kind == "auto":
                try:
                    crops = extract_fingerprints(image, self.extraction_config)
                except (FingerprintExtractionError, ValueError):
                    crops = []
                actual_kind = "hand" if len(crops) >= 2 else "cropped"
            if actual_kind == "hand":
                # Avoid doing extraction twice in the common auto case by using
                # the dedicated path only when no crops were already obtained.
                if crops is None:
                    report.merge(self.enroll_hand(identity, image))
                    continue
                if not crops:
                    report.add_error(
                        _source_description(image), "no fingerprints were extracted"
                    )
                    continue
                report.merge(self._enroll_hand_crops(identity, image, crops))
            else:
                try:
                    added = self.enroll_cropped(identity, image, finger=finger)
                    report.added += added
                    report.skipped += not added
                except (FingerprintMatchingError, ValueError) as exc:
                    report.add_error(_source_description(image), exc)
        return report

    def enroll_directory(self, dataset: str | Path) -> EnrollmentReport:
        """Enroll ``identity/finger/sample`` crops and ``identity/capture`` hands."""

        root = Path(dataset)
        if not root.is_dir():
            raise FingerprintMatchingError(f"Dataset directory does not exist: {root}")
        identities = sorted(
            (item for item in root.iterdir() if item.is_dir()),
            key=lambda path: path.name.casefold(),
        )
        if not identities:
            raise FingerprintMatchingError(
                f"Dataset must contain at least one identity directory: {root}"
            )
        report = EnrollmentReport()
        sources_seen = 0
        for identity_directory in identities:
            identity = identity_directory.name
            # Images directly below an identity are complete hand captures.
            captures = _image_files(identity_directory)
            sources_seen += len(captures)
            for capture in captures:
                report.merge(self.enroll_hand(identity, capture))
            # Images below a named child directory are samples of that finger.
            finger_directories = sorted(
                (item for item in identity_directory.iterdir() if item.is_dir()),
                key=lambda path: path.name.casefold(),
            )
            for finger_directory in finger_directories:
                samples = _image_files(finger_directory, recursive=True)
                sources_seen += len(samples)
                for sample in samples:
                    try:
                        added = self.enroll_cropped(
                            identity, sample, finger=finger_directory.name
                        )
                        report.added += added
                        report.skipped += not added
                    except (FingerprintMatchingError, ValueError) as exc:
                        report.add_error(sample, exc)
        if sources_seen == 0:
            raise FingerprintMatchingError(
                f"Dataset contains no supported fingerprint or hand images: {root}"
            )
        return report

    def _prepare_query(
        self,
        image: str | Path | np.ndarray,
        *,
        kind: str = "cropped",
    ) -> tuple[list[FingerprintTemplate], int]:
        """Return usable templates and the original number of query slots."""

        if kind not in {"auto", "cropped", "hand"}:
            raise ValueError("kind must be 'auto', 'cropped', or 'hand'")
        crops: list[FingerprintCrop] = []
        if kind != "cropped":
            try:
                crops = extract_fingerprints(image, self.extraction_config)
            except (FingerprintExtractionError, ValueError) as exc:
                if kind == "hand":
                    raise FingerprintMatchingError(str(exc)) from exc
            if kind == "hand" and len(crops) < self.config.min_hand_fingers:
                raise FingerprintQualityError(
                    f"Hand query produced {len(crops)} crop(s); at least "
                    f"{self.config.min_hand_fingers} are required"
                )
        if kind == "cropped" or (kind == "auto" and len(crops) < 2):
            return [self.template(image)], 1

        extracted_count = len(crops)
        templates: list[FingerprintTemplate] = []
        errors: list[str] = []
        source = str(Path(image).resolve()) if isinstance(image, (str, Path)) else None
        for index, crop in enumerate(crops, start=1):
            try:
                templates.append(
                    create_template(
                        crop.image,
                        self.config,
                        quality_hint=crop.quality,
                        source=source,
                    )
                )
            except FingerprintQualityError as exc:
                errors.append(f"crop {index}: {exc}")
        if not templates:
            detail = "; ".join(errors) if errors else "no usable crops"
            raise FingerprintQualityError(f"Hand query produced no templates ({detail})")
        if len(templates) < self.config.min_hand_fingers:
            detail = "; ".join(errors) if errors else "insufficient usable crops"
            raise FingerprintQualityError(
                f"Hand query produced only {len(templates)}/{extracted_count} usable "
                f"templates; at least {self.config.min_hand_fingers} are required "
                f"({detail})"
            )
        return templates, extracted_count

    def query_templates(
        self,
        image: str | Path | np.ndarray,
        *,
        kind: str = "cropped",
    ) -> list[FingerprintTemplate]:
        """Convert a query crop or hand image into usable templates."""

        templates, _expected_count = self._prepare_query(image, kind=kind)
        return templates

    def _match_identity(
        self,
        identity: str,
        query: Sequence[FingerprintTemplate],
        references: Sequence[FingerprintTemplate],
        *,
        expected_query_fingers: int | None = None,
    ) -> IdentityMatch:
        query_slots = (
            len(query)
            if expected_query_fingers is None
            else max(len(query), expected_query_fingers)
        )
        grouped: dict[str, list[FingerprintTemplate]] = {}
        for template in references:
            grouped.setdefault(template.finger or "unknown", []).append(template)
        fingers = sorted(grouped, key=str.casefold)
        if not query or not fingers:
            return IdentityMatch(
                identity,
                0.0,
                False,
                0,
                query_slots,
                usable_query_fingers=len(query),
            )

        best: list[list[MatchResult]] = []
        for query_template in query:
            row: list[MatchResult] = []
            for finger in fingers:
                results = [
                    match_templates(query_template, reference, self.config)
                    for reference in grouped[finger]
                ]
                row.append(max(results, key=lambda result: result.score))
            best.append(row)

        assignment = _maximum_assignment(best)
        assignments = tuple(
            FingerAssignment(query_index, fingers[finger_index], best[query_index][finger_index])
            for query_index, finger_index in assignment
        )
        accepted_assignments = [
            item
            for item in assignments
            if item.result.score >= self.config.finger_match_threshold
        ]
        # Weak, below-threshold similarities are not cumulative evidence. The
        # denominator remains the query count so identities with sparse
        # enrollment do not outrank identities that explain the full hand.
        total = sum(item.result.score for item in accepted_assignments)
        score = float(np.clip(total / max(1, query_slots), 0.0, 1.0))
        matched = len(accepted_assignments)
        required_matches = max(
            1, math.ceil(query_slots * self.config.min_matched_finger_fraction)
        )
        accepted = (
            score >= self.config.identity_accept_threshold
            and matched >= required_matches
        )
        return IdentityMatch(
            identity=identity,
            score=score,
            accepted=accepted,
            matched_fingers=matched,
            query_fingers=query_slots,
            assignments=assignments,
            usable_query_fingers=len(query),
        )

    def identify(
        self,
        image: str | Path | np.ndarray | Sequence[FingerprintTemplate],
        *,
        top_k: int = 5,
        kind: str = "cropped",
    ) -> list[IdentityMatch]:
        """Return best identities, sorted by an order-independent hand score."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if isinstance(image, Sequence) and not isinstance(image, (str, bytes, Path, np.ndarray)):
            query = list(image)
            if not all(isinstance(item, FingerprintTemplate) for item in query):
                raise TypeError("A query sequence must contain FingerprintTemplate values")
            expected_query_fingers = len(query)
        else:
            query, expected_query_fingers = self._prepare_query(
                image, kind=kind  # type: ignore[arg-type]
            )
        identities = self.store.identities()
        if not identities:
            raise FingerprintMatchingError("The fingerprint database is empty")
        matches = [
            self._match_identity(
                identity,
                query,
                list(self.store.templates(identity)),
                expected_query_fingers=expected_query_fingers,
            )
            for identity in identities
        ]
        matches.sort(key=lambda item: (-item.score, item.identity.casefold()))
        if (
            len(matches) > 1
            and matches[0].accepted
            and matches[0].score - matches[1].score < self.config.ambiguity_margin
        ):
            best_score = matches[0].score
            matches = [
                replace(item, accepted=False, ambiguous=True)
                if best_score - item.score < self.config.ambiguity_margin
                else item
                for item in matches
            ]
        return matches[:top_k]

    def verify(
        self,
        identity: str,
        image: str | Path | np.ndarray | Sequence[FingerprintTemplate],
        *,
        kind: str = "cropped",
    ) -> IdentityMatch:
        """Verify a claimed identity against all of its enrolled samples."""

        if not self.store.has_identity(identity):
            raise FingerprintMatchingError(f"Identity is not enrolled: {identity}")
        if isinstance(image, Sequence) and not isinstance(image, (str, bytes, Path, np.ndarray)):
            query = list(image)
            if not all(isinstance(item, FingerprintTemplate) for item in query):
                raise TypeError("A query sequence must contain FingerprintTemplate values")
            expected_query_fingers = len(query)
        else:
            query, expected_query_fingers = self._prepare_query(
                image, kind=kind  # type: ignore[arg-type]
            )
        return self._match_identity(
            identity,
            query,
            list(self.store.templates(identity)),
            expected_query_fingers=expected_query_fingers,
        )


def _maximum_assignment(matrix: Sequence[Sequence[MatchResult]]) -> list[tuple[int, int]]:
    """Maximize score with each query and reference label used at most once."""

    rows = len(matrix)
    columns = len(matrix[0]) if rows else 0
    if rows == 0 or columns == 0:
        return []
    if any(len(row) != columns for row in matrix):
        raise ValueError("Assignment matrix must be rectangular")

    # Finger sets are naturally small.  A bitmask DP gives an exact assignment
    # without adding scipy as a runtime dependency.
    if columns > 20:
        candidates = sorted(
            (
                (matrix[row][column].score, row, column)
                for row in range(rows)
                for column in range(columns)
            ),
            reverse=True,
        )
        used_rows: set[int] = set()
        used_columns: set[int] = set()
        result: list[tuple[int, int]] = []
        for score, row, column in candidates:
            if score <= 0 or row in used_rows or column in used_columns:
                continue
            used_rows.add(row)
            used_columns.add(column)
            result.append((row, column))
        return sorted(result)

    @cache
    def solve(row: int, used: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if row == rows:
            return 0.0, ()
        best_score, best_pairs = solve(row + 1, used)  # leave a weak crop unmatched
        for column in range(columns):
            bit = 1 << column
            if used & bit:
                continue
            tail_score, tail_pairs = solve(row + 1, used | bit)
            candidate_score = matrix[row][column].score + tail_score
            if candidate_score > best_score + 1e-12:
                best_score = candidate_score
                best_pairs = ((row, column),) + tail_pairs
        return best_score, best_pairs

    return list(solve(0, 0)[1])
