"""Shared IEEE-oriented Matplotlib style for the matched 900k figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl


SINGLE_COLUMN_WIDTH = 3.45
DOUBLE_COLUMN_WIDTH = 7.16

# Okabe-Ito-inspired, colorblind-safe colors retained consistently everywhere.
AGENT_COLORS = {
    "PPO": "#0072B2",
    "QRL": "#CC79A7",
    "PPO-tiny": "#009E73",
    "DQN": "#D55E00",
}
AGENT_MARKERS = {
    "PPO": "o",
    "QRL": "D",
    "PPO-tiny": "^",
    "DQN": "s",
}
AGENT_ORDER = ("PPO", "QRL", "PPO-tiny", "DQN")


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": False,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "lines.linewidth": 1.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: Any, horizontal_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if horizontal_grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    else:
        ax.grid(False)


def save_figure_png_pdf(fig: Any, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf
