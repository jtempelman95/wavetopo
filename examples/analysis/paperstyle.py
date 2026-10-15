r"""Shared publication style for all NLTopt paper figures.
LaTeX-typeset (pdflatex via the pgf backend -> vector PDF, true Computer Modern to match
the article), restrained palette, no decorative titles. Import FIRST, before pyplot use.

    import paperstyle as ps
    fig = ps.figure(width=ps.FULL, height=2.2)   # inches; width defaults to \linewidth
    ...
    ps.panel(ax, "a"); ps.despine(ax)
    ps.save(fig, "figs/foo")                      # writes foo.pdf
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_NLTOPT = _os.environ.get("NLTOPT_DIR", _os.path.join(_os.path.dirname(_ROOT), "NLTopt"))
import matplotlib
matplotlib.use("pgf")
# Lock the backend: many example modules call matplotlib.use("Agg") at import time,
# which would clobber pgf (and force the dvipng path that isn't available here).
matplotlib.use = lambda *a, **k: None
import matplotlib.pyplot as plt
from matplotlib import rcParams

FULL = 6.5      # \linewidth for article, margin=1in on US Letter
HALF = 3.15

# --- restrained, print-safe palette ---
INK   = "#1a1a1a"   # near-black, primary ink
GREY  = "#5f6b76"   # baseline / uniform series
STEEL = "#2f6f8f"   # cool accent (secondary)
GREEN = "#1c7a54"   # steered / designed series
RED   = "#a4322a"   # instability / warning (used sparingly)
GRID  = "#e7eaee"   # hairline grid
FILL_WARN = "#f2e0de"
FILL_GOOD = "#dcece4"
FILL_MUTE = "#eceef1"

rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.3,
    "patch.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "axes.edgecolor": "#3a3a3a",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#3a3a3a",
    "ytick.color": "#3a3a3a",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "savefig.dpi": 300,           # rasterized field maps embedded at print resolution
})


def use_palatino():
    """Typeset figure text in Palatino (mathpazo), for manuscripts set in Palatino."""
    rcParams["pgf.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{mathpazo}\usepackage{bm}"


def figure(width=FULL, height=2.2):
    return plt.figure(figsize=(width, height))


def despine(ax, keep=("left", "bottom")):
    for k in ("top", "right", "left", "bottom"):
        ax.spines[k].set_visible(k in keep)
    ax.tick_params(direction="out")


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, letter):
    """Plain journal panel label, e.g. (a), at the top-left."""
    ax.set_title(rf"(\textbf{{{letter}}})", loc="left", pad=4)


PAPER_FIGS = (_ROOT + "/docs/paper/figs")


def save(fig, stem):
    import os, shutil
    if stem.endswith(".pdf"):
        stem = stem[:-4]
    fig.savefig(stem + ".pdf")
    dst = os.path.join(PAPER_FIGS, os.path.basename(stem) + ".pdf")
    if os.path.abspath(stem + ".pdf") != os.path.abspath(dst) and os.path.isdir(PAPER_FIGS):
        shutil.copyfile(stem + ".pdf", dst)
    print("wrote " + stem + ".pdf  (+ paper/figs)")
