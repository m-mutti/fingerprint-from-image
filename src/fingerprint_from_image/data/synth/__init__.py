"""Procedural synthetic fingerprint generator (SFinGe-style, numpy + Pillow)."""

from .master import CLASS_PRIORS, MasterPrint, make_master
from .render import render_impression

__all__ = ["CLASS_PRIORS", "MasterPrint", "make_master", "render_impression"]
