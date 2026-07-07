"""Shared helpers for the example scripts (paths, style, colors)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless: render straight to files, no display needed
import matplotlib.pyplot as plt  # noqa: E402

# A small, colorblind-safe palette reused across every figure.
INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2f6fed"  # PageRank / "important via good links"
WARN = "#e8710a"  # in-degree / "popular by volume"
GRID = "#dfe3e8"

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def figure_path(name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 11,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
