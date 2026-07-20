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
                      shading="gouraud", **kw)
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
    tp = ax.tripcolor(tri, facecolors=zc, cmap=STYLE["cmap_curv"], vmax=vmax)
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


def publish(fig, results_name, published_name=None, to_paper=True, root=None):
    """Save to results/, and optionally to the paper's figure directory."""
    root = root or repo_root()
    rp = os.path.join(root, RESDIR, results_name)
    fig.savefig(rp, dpi=STYLE["dpi"], bbox_inches="tight")
    out = [rp]
    if to_paper and published_name:
        pp = os.path.join(root, FIGDIR, published_name)
        fig.savefig(pp, dpi=STYLE["dpi"], bbox_inches="tight")
        out.append(pp)
    return out
