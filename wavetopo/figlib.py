"""
Shared plotting primitives for the paper figures.

Deliberately backend-agnostic: this module never calls ``matplotlib.use``, so it
works both under the Agg batch driver (examples/make_figures.py) and inline in the
VS Code interactive window (examples/figures_interactive.py).  Importing a module
that forces Agg is what would otherwise stop figures displaying in a notebook.

Style lives in the ``STYLE`` dict rather than in module constants so an
interactive session can mutate it and re-run a cell:

    from wavetopo import figlib
    figlib.STYLE["cmap_field"] = "inferno"
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .dolfinx_viz import plot_toolpaths_phase, plot_director_field

STYLE = dict(
    cmap_field="magma",     # wave amplitude panels
    cmap_curv="viridis",    # fiber-curvature panels
    director_n=44,          # director segments across the domain
    director_lw=2.0,
    tow_pitch=0.13,         # phase-field tow spacing
    tow_n=420,              # tow raster resolution
    field_pctl=99.0,        # colour saturation percentile
    field_mode="abs",       # abs | re | im | phase | t   (see field_values)
    field_comp="x",         # component used by re/im/phase/t
    phase_wt=0.0,           # omega*t for field_mode="t"
    cmap_signed="RdBu_r",   # diverging, for signed Re/Im/t
    cmap_phase="twilight",  # cyclic, for phase
    cbar_size="4%",         # colorbar width, as a fraction of the axes
    cbar_pad=0.08,
    dpi=145,
)

JOURNAL = dict(
    # Typography.  The paper sets Times text (newtxtext) with Computer Modern
    # math; STIX is a metric-compatible Times clone that matplotlib bundles, and
    # it covers text AND math, so figure labels sit next to the body text
    # without a visible font change.  DejaVu (the matplotlib default) does not.
    base=9.0,          # tick labels
    title=10.0,        # panel titles
    label=9.5,         # axis labels
    legend=8.5,        # legend entries
    suptitle=12.0,
    panel=10.5,        # (a) (b) (c) tags
    lw_axes=0.7,       # spines and ticks
    lw_line=1.6,       # data lines
    raster_dpi=400,    # resolution of the rasterized field layers inside a PDF
)


def journal(on=True):
    """Apply publication typography to the CURRENT matplotlib rcParams.

    Called by the batch driver and safe to call from a notebook.  Kept separate
    from STYLE because these are global rcParams, not per-figure choices.
    """
    if not on:
        plt.rcParams.update(plt.rcParamsDefault)
        return
    J = JOURNAL
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": J["base"],
        "axes.titlesize": J["title"],
        "axes.labelsize": J["label"],
        "xtick.labelsize": J["base"], "ytick.labelsize": J["base"],
        "legend.fontsize": J["legend"],
        "figure.titlesize": J["suptitle"],
        "axes.linewidth": J["lw_axes"],
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": J["lw_axes"], "ytick.major.width": J["lw_axes"],
        "xtick.minor.width": 0.5*J["lw_axes"],
        "ytick.minor.width": 0.5*J["lw_axes"],
        "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.top": True, "ytick.right": True,
        "lines.linewidth": J["lw_line"],
        "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "0.7", "legend.borderpad": 0.35,
        "axes.grid": False,
        "grid.linewidth": 0.5, "grid.alpha": 0.30,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        # Type 42 (TrueType) rather than Type 3: required by most publishers,
        # and it keeps figure text searchable in the compiled PDF.
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 110,
    })


def panels_of(fig):
    """Data axes of `fig`, in creation (reading) order, colorbars excluded.

    attach_cbar() appends a real Axes for every colorbar, so fig.axes contains
    roughly twice as many entries as there are panels; tagging those would put
    "(b)" on a colour scale.  The tag set in attach_cbar() is what distinguishes
    them -- an inset or a twin axes is deliberately still a panel.
    """
    return [a for a in fig.axes if not getattr(a, "_wavetopo_cbar", False)]


def panel_labels(target, labels=None, x=-0.015, y=1.02, weight="bold",
                 skip=(), fmt="({})"):
    """Stamp (a), (b), (c) ... at the top-left of each data axes.

    Accepts a Figure (panels are found automatically) or an explicit axes list.
    Placed in axes coordinates just OUTSIDE the frame so a tag never lands on
    data.  `skip` takes indices to leave unlabelled.  Idempotent: re-running a
    notebook cell will not double-stamp.
    """
    import string
    if hasattr(target, "axes") and not hasattr(target, "transAxes"):
        axes = panels_of(target)
    else:
        axes = [a for a in np.atleast_1d(np.asarray(target, dtype=object)).ravel()
                if a is not None]
    axes = [a for i, a in enumerate(axes) if i not in set(skip)]
    labels = labels or list(string.ascii_lowercase)
    for a, ch in zip(axes, labels):
        tag = fmt.format(ch)
        if any(getattr(t, "_wavetopo_panel", False) for t in a.texts):
            for t in a.texts:                       # refresh, do not stack
                if getattr(t, "_wavetopo_panel", False):
                    t.set_text(tag)
            continue
        t = a.text(x, y, tag, transform=a.transAxes,
                   fontsize=JOURNAL["panel"], fontweight=weight,
                   va="bottom", ha="right" if x < 0 else "left")
        t._wavetopo_panel = True
    return axes


def tidy(ax, minor=True, grid=False):
    """Line-plot housekeeping: minor ticks, light grid, no top/right spine clutter."""
    if minor:
        ax.minorticks_on()
    if grid:
        ax.grid(True, which="major", alpha=0.30, lw=0.5)
        ax.grid(True, which="minor", alpha=0.15, lw=0.4)
    return ax


FIGDIR = "docs/paper/figs"
RESDIR = "results"


MARGINS = dict(left=0.055, right=0.965, top=0.90, bottom=0.06,
               wspace=0.16, hspace=0.22)


def layout(fig, suptitle=None, y=0.975, fontsize=13, **over):
    """Deterministic figure layout: explicit margins, NO tight_layout.

    matplotlib documents tight_layout as incompatible with the axes_grid1
    divider used by attach_cbar(): it recomputes axes positions without
    accounting for the appended colorbar axes, so a grid can come out aligned
    under one backend/DPI and skewed under another.  subplots_adjust is
    deterministic, which is what a published figure needs.
    """
    if suptitle:
        fig.suptitle(suptitle, y=y, fontsize=fontsize)
    fig.subplots_adjust(**{**MARGINS, **over})


def attach_cbar(ax, mappable=None, ticks=None, ticklabels=None,
                size=None, pad=None):
    """Colorbar in a FIXED-WIDTH axes appended to `ax`.

    plt.colorbar(ax=...) steals space from the axes it is attached to, so a panel
    with a colorbar ends up narrower than one without -- which misaligns a grid
    where only some panels have one (the toolpath panel would hang left of the
    field panel above it).  Appending a fixed-size cax keeps every axes the same
    size; pass mappable=None to reserve the space and hide the bar.
    """
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size=size or STYLE["cbar_size"],
                          pad=pad or STYLE["cbar_pad"])
    cax._wavetopo_cbar = True      # so panel_labels() does not tag a colorbar
    if mappable is None:
        cax.axis("off")
        return None
    cb = ax.figure.colorbar(mappable, cax=cax, ticks=ticks)
    if ticklabels is not None:
        cb.ax.set_yticklabels(ticklabels)
    return cb


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    """Return (data, triangulation, xlim, ylim) from a driver's *_data.npz.

    Extents come from the triangulation itself rather than being hard-coded, so
    the same helper serves the 4x3 lens, the 8x5 cloak and the 8x4 guide.
    """
    d = np.load(path)
    tri = mtri.Triangulation(d["trix"], d["triy"], d["tris"])
    return d, tri, (0.0, float(d["trix"].max())), (0.0, float(d["triy"].max()))


def field_values(d, state, envelope=None, pctl=None):
    """Resolve STYLE["field_mode"] into (values, cmap, imshow-kwargs, label).

    |u| is only the ENVELOPE of a time-harmonic field; the oscillating
    wavefronts live in Re/Im and the propagation direction in the phase.  The
    components are present only if examples/resolve_full_fields.py has been run;
    if they are absent this falls back to |u| and says which file to fix.

    `envelope` overrides the stored |u| key for files that do not use m0/m1.
    """
    mode = STYLE["field_mode"]; comp = STYLE["field_comp"]
    pctl = STYLE["field_pctl"] if pctl is None else pctl
    has = f"ur_{comp}{state}" in d.files
    if mode != "abs" and not has:
        print(f"   [figlib] no Re/Im components saved -> using |u|. "
              f"Run: python examples/resolve_full_fields.py")
        mode = "abs"
    if mode == "abs":
        m = d[envelope] if envelope else d[f"m{state}"]
        return m, STYLE["cmap_field"], dict(vmax=np.percentile(m, pctl)), "$|u|$"
    ur, ui = d[f"ur_{comp}{state}"], d[f"ui_{comp}{state}"]
    if mode == "phase":
        return (np.arctan2(ui, ur), STYLE["cmap_phase"],
                dict(vmin=-np.pi, vmax=np.pi), f"phase $u_{comp}$")
    lim = np.percentile(np.abs(np.concatenate([ur, ui])), pctl)
    kw = dict(vmin=-lim, vmax=lim)
    if mode == "re":
        return ur, STYLE["cmap_signed"], kw, f"Re $u_{comp}$"
    if mode == "im":
        return ui, STYLE["cmap_signed"], kw, f"Im $u_{comp}$"
    if mode == "t":
        wt = STYLE["phase_wt"]
        return (ur*np.cos(wt) + ui*np.sin(wt), STYLE["cmap_signed"], kw,
                f"$u_{comp}(\\omega t={wt:.2f})$")
    raise ValueError(f"field_mode must be abs|re|im|phase|t, got {mode!r}")


def field_panel(ax, tri, m, xlim, ylim, title, vmax=None, marks=None,
                cmap=None, **kw):
    if not kw:
        kw = dict(vmax=vmax if vmax is not None
                  else np.percentile(m, STYLE["field_pctl"]))
    tp = ax.tripcolor(tri, m, cmap=cmap or STYLE["cmap_field"],
                      shading="gouraud", rasterized=True, **kw)
    if marks:
        marks(ax)
    ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    attach_cbar(ax, tp)
    return tp


def tow_panel(ax, cent, th, xlim, ylim, title, holes=None):
    plot_toolpaths_phase(ax, cent, th, xlim, ylim, holes=holes,
                         spacing=STYLE["tow_pitch"], n=STYLE["tow_n"])
    ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    attach_cbar(ax)            # reserve the same width so the grid stays aligned


def dir_panel(ax, cent, th, xlim, ylim, title, holes=None, n=None):
    lc = plot_director_field(ax, cent, th, xlim, ylim, holes=holes,
                             n=n or STYLE["director_n"], lw=STYLE["director_lw"])
    ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    attach_cbar(ax, lc, ticks=[0, np.pi/2, np.pi],
                ticklabels=["0", r"$\pi/2$", r"$\pi$"])
    return lc


def curv_panel(ax, tri, tris, zc, xlim, ylim, title, vmax, marks=None):
    """zeta is DG0 (one value per cell) saved in triangle order."""
    if len(zc) != len(tris):
        raise ValueError(f"curvature has {len(zc)} values for {len(tris)} "
                         "triangles -- cannot map to cells")
    tp = ax.tripcolor(tri, facecolors=zc, cmap=STYLE["cmap_curv"], vmax=vmax,
                      rasterized=True)
    if marks:
        marks(ax)
    ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    attach_cbar(ax, tp)
    return tp


def circles(specs, ec="w", ls="-", lw=1.1):
    tc = np.linspace(0, 2*np.pi, 220)

    def f(ax):
        for cx, cy, r in specs:
            ax.plot(cx + r*np.cos(tc), cy + r*np.sin(tc), color=ec, ls=ls, lw=lw)
    return f


def focus_marker(fx, fy, fr):
    def f(ax):
        ax.add_patch(plt.Circle((fx, fy), fr, ec="#bfefff", fc="none",
                                lw=1.3, ls="--"))
    return f


def holes_of(d):
    """Recover hole geometry from a data file, or None."""
    if "holes" in d:
        return [tuple(map(float, h)) for h in np.atleast_2d(d["holes"])]
    if "RJ" in d:
        return [(float(d["JX"]), float(d["JY"]), float(d["RJ"]))]
    if "RV" in d:
        if "CX" not in d or "CY" not in d:
            raise KeyError("file has RV but no CX/CY -- cannot place the void")
        return [(float(d["CX"]), float(d["CY"]), float(d["RV"]))]
    return None


def publish(fig, results_name, published_name=None, to_paper=True, root=None,
            vector=True):
    r"""Save to results/ (PNG for quick viewing) and to the paper (PNG + PDF).

    The paper gets a PDF as well because the field layers are rasterized while
    axes, ticks, labels and annotations stay vector: text is sharp at any zoom
    and stays searchable, without a 40 MB figure.  \includegraphics{figs/name}
    with no extension makes pdflatex prefer the PDF; the PNG remains for anyone
    reading the figure outside LaTeX.
    """
    root = root or repo_root()
    rp = os.path.join(root, RESDIR, results_name)
    fig.savefig(rp, dpi=STYLE["dpi"], bbox_inches="tight")
    out = [rp]
    if to_paper and published_name:
        pp = os.path.join(root, FIGDIR, published_name)
        fig.savefig(pp, dpi=STYLE["dpi"], bbox_inches="tight")
        out.append(pp)
        if vector:
            vp = os.path.splitext(pp)[0] + ".pdf"
            fig.savefig(vp, dpi=JOURNAL["raster_dpi"], bbox_inches="tight")
            out.append(vp)
    return out


def save_pair(fig, results_path, paper_path=None, label=True, dpi=None,
              vector=True, root=None):
    r"""One call for the standalone figure scripts: label the panels, then write
    results/<name>.png, docs/paper/figs/<name>.png and the vector .pdf.

    The scripts each grew their own pair of savefig lines with slightly
    different dpi, which is how figures in the same paper ended up at 140 and
    145 dpi with different text sizes.  Routing them through here makes the
    published set uniform, and gives every figure the PDF the paper prefers.
    """
    root = root or repo_root()
    if label:
        panel_labels(fig)
    dpi = dpi or STYLE["dpi"]

    def _abs(q):
        return q if os.path.isabs(q) else os.path.join(root, q)

    out = []
    if results_path:
        fig.savefig(_abs(results_path), dpi=dpi, bbox_inches="tight")
        out.append(_abs(results_path))
    if paper_path:
        pp = _abs(paper_path)
        fig.savefig(pp, dpi=dpi, bbox_inches="tight")
        out.append(pp)
        if vector:
            vp = os.path.splitext(pp)[0] + ".pdf"
            fig.savefig(vp, dpi=JOURNAL["raster_dpi"], bbox_inches="tight")
            out.append(vp)
    for q in out:
        print("  wrote", os.path.relpath(q, root))
    return out
