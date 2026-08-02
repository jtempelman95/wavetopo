"""Shared publication figure style for the papers.
 - LaTeX-like font: STIX mathtext + serif text (Times-like, matches the elsarticle
   body font; usetex is not available here). Same family as wavetopo.figlib.journal().
 - Internal ticks on ALL FOUR boundaries of every data axis (call internal_ticks(ax), or
   rely on the rcParams for axes that keep all spines).
 - Vector PDF output with embedded editable fonts.
Keep DESCRIPTION in the LaTeX caption; titles on plots stay short (panel letters + a few
words at most)."""
import matplotlib as mpl


def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",   # Times-like, matches the elsarticle body font
        "axes.linewidth": 0.8,
        "axes.titlesize": 10,
        "font.size": 10,
        # internal ticks on all four sides, by default
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 4.0, "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def internal_ticks(ax, minor=True):
    """Internal ticks on all four boundaries of a data axis (keep all four spines)."""
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   bottom=True, left=True)
    if minor:
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True,
                       bottom=True, left=True)
