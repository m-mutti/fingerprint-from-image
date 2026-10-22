"""Exception hierarchy for the matching framework."""

from __future__ import annotations


class FingerprintMatchingError(RuntimeError):
    """Base class for errors raised by the matching framework."""


class FingerprintQualityError(FingerprintMatchingError):
    """Raised when an image cannot produce a useful feature template."""


class FingerprintDatabaseError(FingerprintMatchingError):
    """Raised for an incompatible or corrupt template database."""
