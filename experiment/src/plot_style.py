"""Publication-ready plotting style, shared by every experiment figure.

One place defines the look so that bar charts, scatter panels and line charts
stay visually consistent across the paper: Times New Roman throughout, bold
oversized tick/axis labels, no left/top/right spines, dashed y-grid, alternating
grey bands and the small dots on the y axis.

The method palette is the single source of truth for colour: the same hue means
the same method everywhere, including the t-SNE panels, so a reader can carry
Q2Q's colour from one figure to the next.

Two axis looks live here. ``style_axes`` is the full-width bar-chart look;
``style_line_panel`` is the narrow multi-curve look used by the single-column
line panels. See the comment above the latter for when each applies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Method identity: colour, bar hatch and line marker, keyed by method id.
METHOD_COLORS = {
    "q2q": "#53589A",  # deep indigo -- the proposed method
    "q2p": "#E07B54",  # warm orange
    "q2n": "#5BA585",  # teal green
    "q2r": "#C75B7A",  # rose
    "q2c": "#D9A441",  # amber
}

METHOD_HATCHES = {
    "q2q": "/",
    "q2p": "x",
    "q2n": "\\",
    "q2r": "+",
    "q2c": "o",
}

METHOD_MARKERS = {
    "q2q": "o",
    "q2p": "D",
    "q2n": "^",
    "q2r": "v",
    "q2c": "s",
}

METHOD_LABELS = {
    "q2q": "Q2Q",
    "q2p": "Q2P",
    "q2n": "Q2N",
    "q2r": "Q2R",
    "q2c": "Q2C",
}

METHOD_ORDER = ["q2q", "q2p", "q2n", "q2r", "q2c"]

# Embedding-type palette for the t-SNE panels. Independent of the method
# palette on purpose: the panels are read as a point cloud rather than as a
# method comparison, and the original high-saturation hues separate six
# overlapping types far better than the muted bar/line colours would.
EMBED_COLORS = {
    "true_query": "#E53935",
    "anchor_query": "#2196F3",
    "content": "#FF9800",
    "proposition": "#9C27B0",
    "note": "#E91E63",
    "reflection": "#009688",
}

BAND_COLOR = "#CCCCCC"
GRID_COLOR = "#999999"

# --------------------------------------------------------------------------- #
# Muted line-panel look
# --------------------------------------------------------------------------- #
# A second visual language, for the narrow multi-curve line panels that go into
# a single paper column. ``style_axes`` above is built for full-width bar charts
# -- its alternating bands and heavy ticks crowd a panel this small -- so these
# panels instead get an open L frame, a light grid in both directions and long
# rounded dashes. Colour is deliberately *not* fixed here: each figure brings
# its own palette so two such panels sitting on the same page stay telling
# apart, while the frame, grid, dash pattern and type scale keep them a family.

LINE_GRID_COLOR = "#E4E8E8"
LINE_AXIS_COLOR = "#A9B7B5"
# Titles, axis labels and ticks are plain black Times New Roman at regular
# weight, no tinting, matching the exp4/exp5 line charts.
LINE_LABEL_COLOR = "black"
LINE_DASH = (0, (5.5, 2.2))

# Type scale. The title and the x label sit a step below the y label: they name
# a constant each (the dataset, the swept knob) while the y label names the
# quantity being read off.
PANEL_TITLE_SIZE = 14.0
PANEL_XLABEL_SIZE = 13.0
PANEL_YLABEL_SIZE = 14.5
PANEL_TICK_SIZE = 12.5
PANEL_XTICK_SIZE = 13.0
PANEL_LEGEND_SIZE = 12.0


def style_line_panel(
    ax,
    *,
    tick_labelsize: float = PANEL_TICK_SIZE,
    xtick_labelsize: float = PANEL_XTICK_SIZE,
) -> None:
    """Open L frame: the two value axes are drawn, the opposite pair is not.

    Tick *positions* stay the caller's business -- this only sets the frame,
    the grid and the tick cosmetics, so a categorical and a numeric panel can
    share the look without sharing a scale.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(LINE_AXIS_COLOR)
        ax.spines[side].set_linewidth(1.1)

    ax.grid(True, color=LINE_GRID_COLOR, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=3.5, width=1.1, color=LINE_AXIS_COLOR,
                   labelcolor=LINE_LABEL_COLOR, labelsize=tick_labelsize, pad=4)
    ax.tick_params(axis="x", labelsize=xtick_labelsize)


def line_panel_handles(series: Sequence[dict]) -> list:
    """Proxy artists for a legend drawn in the muted line-panel look.

    Each entry needs ``color``, ``marker`` and ``label``; ``highlight`` thickens
    the line and enlarges the marker, as it does on the panel itself.
    """
    return [
        Line2D(
            [0], [0], color=s["color"],
            linewidth=2.6 if s.get("highlight") else 2.1,
            linestyle=LINE_DASH, dash_capstyle="round", marker=s["marker"],
            markersize=8.0 if s.get("highlight") else 6.4,
            markerfacecolor=s["color"], markeredgecolor="white",
            markeredgewidth=0.9, label=s["label"],
        )
        for s in series
    ]


def use_paper_style() -> None:
    """Install the global rcParams every figure in the paper shares."""
    matplotlib.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.size"] = 16
    matplotlib.rcParams["hatch.linewidth"] = 0.6
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def style_axes(
    ax,
    yticks: Sequence[float],
    *,
    bands: bool = True,
    band_alpha: float = 0.08,
    dots: bool = True,
    tick_labelsize: int = 24,
    xtick_labelsize: int = 24,
) -> None:
    """Apply the shared axis decoration: spines, grid, bands and tick dots.

    ``yticks`` drives both the alternating grey bands (every other interval)
    and the small black dots drawn at the tick positions on the y axis.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)

    ax.tick_params(axis="y", labelsize=tick_labelsize, length=6, width=1.2, direction="out")
    ax.tick_params(axis="x", labelsize=xtick_labelsize, pad=6)

    ticks = list(yticks)
    if bands:
        for lo, hi in zip(ticks[1::2], ticks[2::2]):
            ax.axhspan(lo, hi, facecolor=BAND_COLOR, alpha=band_alpha, zorder=0)
    if dots:
        for yt in ticks[1:]:
            ax.plot(
                0, yt, "o", color="black", markersize=3.5, clip_on=False,
                transform=ax.get_yaxis_transform(), zorder=10,
            )


def method_legend_handles(
    methods: Sequence[str], *, kind: str = "bar"
) -> list:
    """Proxy artists so a figure-level legend can be built without the axes."""
    handles = []
    for m in methods:
        if kind == "bar":
            handles.append(
                matplotlib.patches.Patch(
                    facecolor=METHOD_COLORS[m], edgecolor="white",
                    hatch=METHOD_HATCHES[m], label=METHOD_LABELS[m],
                )
            )
        else:
            handles.append(
                Line2D(
                    [0], [0], color=METHOD_COLORS[m], linewidth=2.8,
                    marker=METHOD_MARKERS[m], markersize=10,
                    markerfacecolor="white", markeredgewidth=2.2,
                    markeredgecolor=METHOD_COLORS[m], label=METHOD_LABELS[m],
                )
            )
    return handles


def save_paper_figure(fig, png_path: Path) -> Path:
    """Write the PNG at the requested path plus a vector PDF beside it."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path


def auto_ticks(vmin: float, vmax: float, *, target: int = 5) -> np.ndarray:
    """Round tick positions covering ``[vmin, vmax]``, roughly ``target`` of them.

    Steps are restricted to 1/2/2.5/5 times a power of ten so the labels stay
    readable; the range is expanded outwards to land on whole steps.
    """
    span = max(vmax - vmin, 1e-9)
    raw = span / max(target, 1)
    mag = 10.0 ** np.floor(np.log10(raw))
    step = 10.0 * mag
    for mult in (1.0, 2.0, 2.5, 5.0):
        if raw <= mult * mag:
            step = mult * mag
            break
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    return np.arange(lo, hi + step / 2, step)
