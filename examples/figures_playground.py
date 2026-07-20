# %% [markdown]
# # Figure playground — every knob, nothing hidden
#
# All data is loaded once in the setup cell. After that **each figure is a single
# self-contained cell**: a knobs block you edit, then the full plotting code
# inline. Nothing is delegated to a builder, so you can move a panel, retitle an
# axis, or change a marker without leaving the cell.
#
# * **Interpreter:** base conda env (`/home/jrt/miniforge3/bin/python3`) — no dolfinx.
# * `SAVE = False` while you play; flip it to write files.
# * Batch equivalent (fixed styling): `python examples/make_figures.py --all`
#
# The only imported plotting code is `plot_toolpaths_phase` / `plot_director_field`
# — those are real algorithms (phase-field tow delineation, director LineCollection),
# not styling. Everything else is here in the open.

# %%
# ============================ SETUP: load everything =======================
import os
import sys
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(
    globals().get("__file__", os.path.join(os.getcwd(), "examples", "_")))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from wavetopo.dolfinx_viz import plot_toolpaths_phase, plot_director_field  # noqa

SAVE = False          # True -> write results/ and docs/paper/figs/
DPI = 145

SOURCES = {
    "lens":       "results/dolfinx_lens_data.npz",
    "lens_curl":  "results/dolfinx_lens_curl_data.npz",
    "lens_multi": "results/dolfinx_lens_multi_data.npz",
    "lens_hole":  "results/dolfinx_lens_hole_data.npz",
    "lens_asym":  "results/dolfinx_lens_hole_asym_data.npz",
    "cloak":      "results/dolfinx_cloak_conforming_data.npz",
    "cloak_curl": "results/dolfinx_cloak_curl_data.npz",
    "guide":      "results/dolfinx_guide_joint_data.npz",
}

D, T, X, Y = {}, {}, {}, {}       # data, triangulation, xlim, ylim
for k, p in SOURCES.items():
    if not os.path.exists(p):
        print(f"  MISSING {k:11} {p}"); continue
    d = np.load(p)
    D[k] = d
    T[k] = mtri.Triangulation(d["trix"], d["triy"], d["tris"])
    X[k] = (0.0, float(d["trix"].max()))
    Y[k] = (0.0, float(d["triy"].max()))
    print(f"  {k:11} {len(d['tris']):>7,} cells   domain "
          f"{X[k][1]:.1f} x {Y[k][1]:.1f}   keys: {len(d.files)}")


from mpl_toolkits.axes_grid1 import make_axes_locatable   # noqa: E402

CBAR_SIZE, CBAR_PAD = "4%", 0.08

# Explicit figure margins, used INSTEAD of tight_layout.  matplotlib documents
# tight_layout as incompatible with the axes_grid1 divider used by cbar(): it
# recomputes axes positions without accounting for the appended colorbar axes, so
# the grid can come out aligned under one backend/DPI and skewed under another
# (Agg vs the VS Code inline renderer).  subplots_adjust is deterministic.
MARGINS = dict(left=0.055, right=0.965, top=0.90, bottom=0.06,
               wspace=0.16, hspace=0.22)


def layout(fig, suptitle=None, y=0.975, fontsize=13, **over):
    """Deterministic layout: explicit margins, no tight_layout."""
    if suptitle:
        fig.suptitle(suptitle, y=y, fontsize=fontsize)
    fig.subplots_adjust(**{**MARGINS, **over})


def cbar(ax, mappable=None, ticks=None, ticklabels=None):
    """Fixed-width colorbar appended to `ax`.

    plt.colorbar(ax=...) steals space FROM the axes, so panels with a colorbar
    come out narrower than panels without -- which is why a toolpath panel hangs
    left of the field panel above it.  Appending a fixed-size cax keeps every
    axes identical; call with mappable=None to reserve the width and hide the bar
    (do this on EVERY panel that has no colorbar, or the grid misaligns again).
    """
    cax = make_axes_locatable(ax).append_axes("right", size=CBAR_SIZE,
                                              pad=CBAR_PAD)
    if mappable is None:
        cax.axis("off")
        return None
    cb = ax.figure.colorbar(mappable, cax=cax, ticks=ticks)
    if ticklabels is not None:
        cb.ax.set_yticklabels(ticklabels)
    return cb


def save_fig(fig, results_name, paper_name=None):
    """Write only when SAVE is True."""
    if not SAVE:
        print("   (SAVE=False, not written)"); return
    fig.savefig(f"results/{results_name}", dpi=DPI, bbox_inches="tight")
    if paper_name:
        fig.savefig(f"docs/paper/figs/{paper_name}", dpi=DPI, bbox_inches="tight")
    print(f"   wrote {results_name}" + (f" + {paper_name}" if paper_name else ""))


def keys(k):
    """What is in a data file (handy when adding your own panel)."""
    return sorted(D[k].files)


print("keys('lens') ->", keys("lens"))

# ---------------------------------------------------------------------------
# FIELD REPRESENTATION
# Every figure cell below has a FIELD_MODE knob.  |u| is only the ENVELOPE; the
# oscillating wavefronts live in Re/Im, and the propagation direction in phase.
#   "abs"   sqrt(ur_x^2+ur_y^2+ui_x^2+ui_y^2)   sequential cmap
#   "re"    Re u_comp                            diverging, symmetric limits
#   "im"    Im u_comp                            diverging, symmetric limits
#   "phase" atan2(ui_comp, ur_comp)              cyclic cmap
#   "t"     ur cos(wt) + ui sin(wt) at PHASE_WT  diverging (a time snapshot)
# Re/Im/phase/t need the components, i.e. examples/resolve_full_fields.py must
# have been run; if they are absent this falls back to "abs" and says so.
CMAP_ABS, CMAP_SIGNED, CMAP_PHASE = "magma", "RdBu_r", "twilight"
PHASE_WT = 0.0                      # omega*t for FIELD_MODE = "t"


def field_of(key, state, mode="abs", comp="x", pctl=99.5, envelope=None):
    """-> (values, cmap, imshow-kwargs, label). `envelope` overrides the stored
    |u| key for files that do not use m0/m1 (the conforming cloak)."""
    d = D[key]
    has = f"ur_{comp}{state}" in d.files
    if mode != "abs" and not has:
        print(f"   [{key}] no components saved -> falling back to |u|.  "
              f"Run: python examples/resolve_full_fields.py {key}")
        mode = "abs"
    if mode == "abs":
        m = d[envelope] if envelope else d[f"m{state}"]
        return m, CMAP_ABS, dict(vmax=np.percentile(m, pctl)), "$|u|$"
    ur, ui = d[f"ur_{comp}{state}"], d[f"ui_{comp}{state}"]
    if mode == "phase":
        return (np.arctan2(ui, ur), CMAP_PHASE,
                dict(vmin=-np.pi, vmax=np.pi), f"phase $u_{comp}$")
    lim = np.percentile(np.abs(np.concatenate([ur, ui])), pctl)
    if mode == "re":
        return ur, CMAP_SIGNED, dict(vmin=-lim, vmax=lim), f"Re $u_{comp}$"
    if mode == "im":
        return ui, CMAP_SIGNED, dict(vmin=-lim, vmax=lim), f"Im $u_{comp}$"
    if mode == "t":
        v = ur*np.cos(PHASE_WT) + ui*np.sin(PHASE_WT)
        return (v, CMAP_SIGNED, dict(vmin=-lim, vmax=lim),
                f"$u_{comp}(\\omega t={PHASE_WT:.2f})$")
    raise ValueError(f"mode must be abs|re|im|phase|t, got {mode!r}")


print("\nfield_of(): modes abs | re | im | phase | t")


# %% [markdown]
# ## 1 — Wave lens

# %%
# ------------------------------- KNOBS -------------------------------------
FIGSIZE      = (14, 9)
FIELD_MODE   = "abs"          # abs | re | im | phase | t   <-- switch me
FIELD_COMP   = "x"            # component for re/im/phase/t
VMAX_PCTL    = 99.5
SHADING      = "gouraud"      # "gouraud" | "flat"
CBAR_FRAC    = 0.030
FOCUS_EC, FOCUS_LW, FOCUS_LS = "#bfefff", 1.3, "--"
FOCUS_XYR    = (0.78*X["lens"][1], Y["lens"][1]/2, 0.18)   # (x, y, r)
TOW_PITCH, TOW_N = 0.16, 420
DIR_N, DIR_LW = 40, 2.0
TITLE_BASE   = "straight fibers (baseline)"
TITLE_OPT    = "optimized fiber lens — {gain:.0f}$\\times$ focus gain"
TITLE_TOW    = "fiber toolpaths"
TITLE_DIR    = "anisotropy director field"
SUPTITLE     = "Curvilinear-fiber elastic wave lens"
SUP_Y, SUP_FS = 0.97, 13
HPAD         = 2.5
OUT, OUT_PAPER = "dolfinx_lens.png", "dfx_lens.png"
# ---------------------------------------------------------------------------
d, tri, xl, yl = D["lens"], T["lens"], X["lens"], Y["lens"]
gain = float(d["gain"])
fig, ax = plt.subplots(2, 2, figsize=FIGSIZE)


def focus(a):
    fx, fy, fr = FOCUS_XYR
    a.add_patch(plt.Circle((fx, fy), fr, ec=FOCUS_EC, fc="none",
                           lw=FOCUS_LW, ls=FOCUS_LS))


for a, st, t in [(ax[0, 0], 0, TITLE_BASE),
                 (ax[0, 1], 1, TITLE_OPT.format(gain=gain))]:
    v, cm, kw, lab = field_of("lens", st, FIELD_MODE, FIELD_COMP, VMAX_PCTL)
    tp = a.tripcolor(tri, v, cmap=cm, shading=SHADING, **kw)
    t = f"{t}   [{lab}]"
    focus(a)
    a.set_aspect("equal"); a.set_xlim(*xl); a.set_ylim(*yl)
    a.set_title(t, fontsize=10)
    cbar(a, tp)

plot_toolpaths_phase(ax[1, 0], d["cent"], d["thopt"], xl, yl,
                     spacing=TOW_PITCH, n=TOW_N)
focus(ax[1, 0])
ax[1, 0].set_aspect("equal"); ax[1, 0].set_xlim(*xl); ax[1, 0].set_ylim(*yl)
ax[1, 0].set_title(TITLE_TOW, fontsize=10)
cbar(ax[1, 0])          # reserve width -> stays aligned with the panel above

lc = plot_director_field(ax[1, 1], d["cent"], d["thopt"], xl, yl,
                         n=DIR_N, lw=DIR_LW)
ax[1, 1].set_aspect("equal"); ax[1, 1].set_xlim(*xl); ax[1, 1].set_ylim(*yl)
ax[1, 1].set_title(TITLE_DIR, fontsize=10)
cbar(ax[1, 1], lc, ticks=[0, np.pi/2, np.pi],
     ticklabels=["0", r"$\pi/2$", r"$\pi$"])

fig.suptitle(SUPTITLE, y=SUP_Y, fontsize=SUP_FS)
save_fig(fig, OUT, OUT_PAPER)
plt.show()

# %% [markdown]
# ## 2 — Curvature-constrained lens

# %%
# ------------------------------- KNOBS -------------------------------------
FIGSIZE      = (19, 9)
FIELD_MODE, FIELD_COMP = "abs", "x"     # abs | re | im | phase | t
CMAP_CURV    = "viridis"
VMAX_PCTL    = 99.5
TOW_PITCH, TOW_N = 0.16, 420
CBAR_FRAC    = 0.030
ROW_TAGS     = ["unconstrained", "$|\\zeta|\\leq${zall:.2f}"]
TITLE_FIELD  = "{tag}: field  ({gain:.0f}$\\times$)"
TITLE_TOW    = "{tag}: toolpaths  (max$|\\zeta|$={zmax:.2f})"
TITLE_CURV   = "{tag}: fiber curvature $|\\zeta|$"
SUPTITLE     = "Curvature-constrained wave lens (manufacturable toolpaths)"
SUP_Y, HPAD  = 0.98, 2.0
OUT, OUT_PAPER = "dolfinx_lens_curl.png", "dfx_lens_curl.png"
# ---------------------------------------------------------------------------
d, tri, xl, yl = D["lens_curl"], T["lens_curl"], X["lens_curl"], Y["lens_curl"]
zall = float(d["zeta_all"])
CVMAX = max(np.abs(d["zc0"]).max(), np.abs(d["zc1"]).max())

fig, ax = plt.subplots(2, 3, figsize=FIGSIZE)
for r in (0, 1):
    tag = ROW_TAGS[r].format(zall=zall)
    v, cm, kw, lab = field_of("lens_curl", r, FIELD_MODE, FIELD_COMP, VMAX_PCTL)
    tp = ax[r, 0].tripcolor(tri, v, cmap=cm, shading="gouraud", **kw)
    ax[r, 0].set_aspect("equal"); ax[r, 0].set_xlim(*xl); ax[r, 0].set_ylim(*yl)
    ax[r, 0].set_title(TITLE_FIELD.format(tag=tag, gain=float(d[f"gain{r}"]))
                       + f"  [{lab}]", fontsize=10)
    cbar(ax[r, 0], tp)

    plot_toolpaths_phase(ax[r, 1], d["cent"], d[f"th{r}"], xl, yl,
                         spacing=TOW_PITCH, n=TOW_N)
    ax[r, 1].set_aspect("equal"); ax[r, 1].set_xlim(*xl); ax[r, 1].set_ylim(*yl)
    ax[r, 1].set_title(TITLE_TOW.format(tag=tag, zmax=float(d[f"zmax{r}"])),
                       fontsize=10)
    cbar(ax[r, 1])

    cp = ax[r, 2].tripcolor(tri, facecolors=d[f"zc{r}"], cmap=CMAP_CURV,
                            vmax=CVMAX)
    ax[r, 2].set_aspect("equal"); ax[r, 2].set_xlim(*xl); ax[r, 2].set_ylim(*yl)
    ax[r, 2].set_title(TITLE_CURV.format(tag=tag), fontsize=10)
    cbar(ax[r, 2], cp)

layout(fig, SUPTITLE, y=SUP_Y, fontsize=13)
save_fig(fig, OUT, OUT_PAPER)
plt.show()

# %% [markdown]
# ## 3 — Multi-target beam shaping (two foci + a null)

# %%
# ------------------------------- KNOBS -------------------------------------
FIGSIZE   = (14, 9)
FIELD_MODE, FIELD_COMP = "abs", "x"     # abs | re | im | phase | t
VMAX_PCTL = 99.5
# target/null markers: (x, y, r, colour)
SPOTS     = [(0.62, 0.30, 0.16, "cyan"), (0.62, 0.70, 0.16, "cyan"),
             (0.86, 0.50, 0.16, "red")]        # x,y as FRACTIONS of the domain
MARK_LW, MARK_LS = 1.3, "--"
TOW_PITCH, TOW_N = 0.16, 420
DIR_N, DIR_LW = 40, 2.0
TITLE_BASE = "baseline: target/null contrast {c0:.1f}"
TITLE_OPT  = "two foci + a null: contrast {c1:.1f}  ({r:.1f}$\\times$)"
SUPTITLE   = "Multi-target beam shaping: two foci (cyan), one null (red)"
OUT, OUT_PAPER = "dolfinx_lens_multi.png", "dfx_lens_multi.png"
# ---------------------------------------------------------------------------
d, tri, xl, yl = D["lens_multi"], T["lens_multi"], X["lens_multi"], Y["lens_multi"]
c0 = float(d["tA0"])/float(d["tN0"]); c1 = float(d["tA1"])/float(d["tN1"])


def spots(a):
    for fx, fy, r, c in SPOTS:
        a.add_patch(plt.Circle((fx*xl[1], fy*yl[1]), r, ec=c, fc="none",
                               lw=MARK_LW, ls=MARK_LS))


fig, ax = plt.subplots(2, 2, figsize=FIGSIZE)
for a, st, t in [(ax[0, 0], 0, TITLE_BASE.format(c0=c0)),
                 (ax[0, 1], 1, TITLE_OPT.format(c1=c1, r=c1/c0))]:
    v, cm, kw, lab = field_of("lens_multi", st, FIELD_MODE, FIELD_COMP, VMAX_PCTL)
    tp = a.tripcolor(tri, v, cmap=cm, shading="gouraud", **kw)
    t = f"{t}   [{lab}]"
    spots(a)
    a.set_aspect("equal"); a.set_xlim(*xl); a.set_ylim(*yl)
    a.set_title(t, fontsize=10); cbar(a, tp)

plot_toolpaths_phase(ax[1, 0], d["cent"], d["thopt"], xl, yl,
                     spacing=TOW_PITCH, n=TOW_N)
ax[1, 0].set_aspect("equal"); ax[1, 0].set_xlim(*xl); ax[1, 0].set_ylim(*yl)
ax[1, 0].set_title("fiber toolpaths", fontsize=10)
cbar(ax[1, 0])
lc = plot_director_field(ax[1, 1], d["cent"], d["thopt"], xl, yl,
                         n=DIR_N, lw=DIR_LW)
ax[1, 1].set_aspect("equal"); ax[1, 1].set_xlim(*xl); ax[1, 1].set_ylim(*yl)
ax[1, 1].set_title("anisotropy director field", fontsize=10)
cbar(ax[1, 1], lc, ticks=[0, np.pi/2, np.pi],
     ticklabels=["0", r"$\pi/2$", r"$\pi$"])
layout(fig, SUPTITLE, y=0.97, fontsize=13)
save_fig(fig, OUT, OUT_PAPER)
plt.show()

# %% [markdown]
# ## 4 — Lens with two prescribed through-holes

# %%
# ------------------------------- KNOBS -------------------------------------
FIGSIZE   = (19, 9)
FIELD_MODE, FIELD_COMP = "abs", "x"     # abs | re | im | phase | t
CMAP_CURV, VMAX_PCTL = "viridis", 99.5
HOLE_EC, HOLE_LW = "w", 1.2
TOW_PITCH, TOW_N = 0.16, 420
ROW_TAGS  = ["unconstrained", "$|\\zeta|\\leq 1/R$={zall:.2f}"]
SUPTITLE  = "Wave lens with two prescribed circular through-holes"
OUT, OUT_PAPER = "dolfinx_lens_hole.png", "dfx_lens_hole.png"
# ---------------------------------------------------------------------------
d, tri, xl, yl = D["lens_hole"], T["lens_hole"], X["lens_hole"], Y["lens_hole"]
HOLES = [tuple(map(float, h)) for h in np.atleast_2d(d["holes"])]
zall = float(d["zeta_all"])
CVMAX = max(np.abs(d["zc0"]).max(), np.abs(d["zc1"]).max())
tc = np.linspace(0, 2*np.pi, 240)


def rings(a):
    for cx, cy, r in HOLES:
        a.plot(cx+r*np.cos(tc), cy+r*np.sin(tc), color=HOLE_EC, lw=HOLE_LW)


fig, ax = plt.subplots(2, 3, figsize=FIGSIZE)
for r in (0, 1):
    tag = ROW_TAGS[r].format(zall=zall)
    v, cm, kw, lab = field_of("lens_hole", r, FIELD_MODE, FIELD_COMP, VMAX_PCTL)
    tp = ax[r, 0].tripcolor(tri, v, cmap=cm, shading="gouraud", **kw)
    rings(ax[r, 0])
    ax[r, 0].set_aspect("equal"); ax[r, 0].set_xlim(*xl); ax[r, 0].set_ylim(*yl)
    ax[r, 0].set_title(f"{tag}: {lab} "
                       f"({float(d[f'gain{r}']):.0f}$\\times$)", fontsize=10)
    cbar(ax[r, 0], tp)

    plot_toolpaths_phase(ax[r, 1], d["cent"], d[f"th{r}"], xl, yl, holes=HOLES,
                         spacing=TOW_PITCH, n=TOW_N)
    rings(ax[r, 1])
    ax[r, 1].set_aspect("equal"); ax[r, 1].set_xlim(*xl); ax[r, 1].set_ylim(*yl)
    ax[r, 1].set_title(f"{tag}: toolpaths (max$|\\zeta|$="
                       f"{float(d[f'zmax{r}']):.2f})", fontsize=10)
    cbar(ax[r, 1])

    cp = ax[r, 2].tripcolor(tri, facecolors=d[f"zc{r}"], cmap=CMAP_CURV,
                            vmax=CVMAX)
    rings(ax[r, 2])
    ax[r, 2].set_aspect("equal"); ax[r, 2].set_xlim(*xl); ax[r, 2].set_ylim(*yl)
    ax[r, 2].set_title(f"{tag}: fiber curvature", fontsize=10)
    cbar(ax[r, 2], cp)

layout(fig, SUPTITLE, y=0.98, fontsize=13)
save_fig(fig, OUT, OUT_PAPER)
plt.show()

# %% [markdown]
# ## 5 — Asymmetric through-holes

# %%
# ------------------------------- KNOBS -------------------------------------
FIGSIZE   = (14, 9)
FIELD_MODE, FIELD_COMP = "abs", "x"     # abs | re | im | phase | t
VMAX_PCTL = 99.5
HOLE_EC, HOLE_LW = "w", 1.2
TOW_PITCH, TOW_N, DIR_N, DIR_LW = 0.16, 420, 40, 2.0
SUPTITLE  = "Wave lens with two ASYMMETRIC through-holes"
OUT, OUT_PAPER = "dolfinx_lens_hole_asym.png", "dfx_lens_hole_asym.png"
# ---------------------------------------------------------------------------
d, tri, xl, yl = D["lens_asym"], T["lens_asym"], X["lens_asym"], Y["lens_asym"]
HOLES = [tuple(map(float, h)) for h in np.atleast_2d(d["holes"])]
tc = np.linspace(0, 2*np.pi, 240)


def rings(a):
    for cx, cy, r in HOLES:
        a.plot(cx+r*np.cos(tc), cy+r*np.sin(tc), color=HOLE_EC, lw=HOLE_LW)


fig, ax = plt.subplots(2, 2, figsize=FIGSIZE)
for a, st, t in [(ax[0, 0], 0, "straight fibers (baseline)"),
                 (ax[0, 1], 1,
                  f"re-optimized: {float(d['gain']):.0f}$\\times$")]:
    v, cm, kw, lab = field_of("lens_asym", st, FIELD_MODE, FIELD_COMP, VMAX_PCTL)
    tp = a.tripcolor(tri, v, cmap=cm, shading="gouraud", **kw)
    t = f"{t}   [{lab}]"
    rings(a); a.set_aspect("equal"); a.set_xlim(*xl); a.set_ylim(*yl)
    a.set_title(t, fontsize=10); cbar(a, tp)
plot_toolpaths_phase(ax[1, 0], d["cent"], d["th1"], xl, yl, holes=HOLES,
                     spacing=TOW_PITCH, n=TOW_N)
rings(ax[1, 0])
ax[1, 0].set_aspect("equal"); ax[1, 0].set_xlim(*xl); ax[1, 0].set_ylim(*yl)
ax[1, 0].set_title(f"toolpaths (max$|\\zeta|$={float(d['zmax']):.2f}, limit "
                   f"{float(d['zeta_all']):.2f})", fontsize=10)
cbar(ax[1, 0])
lc = plot_director_field(ax[1, 1], d["cent"], d["th1"], xl, yl, holes=HOLES,
                         n=DIR_N, lw=DIR_LW)
ax[1, 1].set_aspect("equal"); ax[1, 1].set_xlim(*xl); ax[1, 1].set_ylim(*yl)
ax[1, 1].set_title("anisotropy director field", fontsize=10)
cbar(ax[1, 1], lc, ticks=[0, np.pi/2, np.pi],
     ticklabels=["0", r"$\pi/2$", r"$\pi$"])
layout(fig, SUPTITLE, y=0.97, fontsize=13)
save_fig(fig, OUT, OUT_PAPER)
plt.show()

# %% [markdown]
# ## 6 — Cloak: the void as the prescribed through-hole (soft-void variant)
#
# NOTE this data file stores **no geometry** — only `solc`, the per-triangle
# solid mask. `VOID` and `SHELL` below are the driver's constants
# (`examples/dolfinx_cloak_soft.py`); if you change them there, change them here.

# %%
# ------------------------------- KNOBS -------------------------------------
FIGSIZE   = (19, 9)
FIELD_MODE, FIELD_COMP = "abs", "x"     # abs | re | im | phase | t
CMAP_CURV, VMAX_PCTL = "viridis", 99.0
VOID_R, SHELL_R = 0.55, 1.4            # driver constants (RV, RC)
VOID_EC, VOID_LW = "c", 1.3
SHELL_EC, SHELL_LS, SHELL_LW = "w", ":", 1.0
TOW_PITCH, TOW_N = 0.16, 420
BLANK_VOID_CURV = True                  # curvature is meaningless with no material
SUPTITLE  = ("Cloak with the void as the prescribed through-hole "
             "($\\zeta_{\\rm all}=1/R_V$; soft-void variant)")
OUT, OUT_PAPER = "dolfinx_cloak_curl.png", "dfx_cloak_curl.png"
# ---------------------------------------------------------------------------
d, tri, xl, yl = D["cloak_curl"], T["cloak_curl"], X["cloak_curl"], Y["cloak_curl"]
CX, CY = xl[1]/2, yl[1]/2
solc = d["solc"].astype(bool)
CVMAX = float(np.nanmax([np.nanmax(d["zc0"]), np.nanmax(d["zc1"])]))
tc = np.linspace(0, 2*np.pi, 240)


def rings(a):
    a.plot(CX+VOID_R*np.cos(tc), CY+VOID_R*np.sin(tc), color=VOID_EC, lw=VOID_LW)
    a.plot(CX+SHELL_R*np.cos(tc), CY+SHELL_R*np.sin(tc), color=SHELL_EC,
           ls=SHELL_LS, lw=SHELL_LW)


fig, ax = plt.subplots(2, 3, figsize=FIGSIZE)
for r in (0, 1):
    tag = "unconstrained" if r == 0 else \
        f"$|\\zeta|\\leq${float(d['zeta_all']):.2f}"
    v, cm, kw, lab = field_of("cloak_curl", r, FIELD_MODE, FIELD_COMP, VMAX_PCTL)
    tp = ax[r, 0].tripcolor(tri, v, cmap=cm, shading="gouraud", **kw)
    rings(ax[r, 0])
    ax[r, 0].set_aspect("equal"); ax[r, 0].set_xlim(*xl); ax[r, 0].set_ylim(*yl)
    ax[r, 0].set_title(f"{tag}: {lab} ({float(d[f'red{r}']):.0f}$\\times$ "
                       "scatter reduction)", fontsize=10)
    cbar(ax[r, 0], tp)

    # holes=... is what keeps tows off the scatterer
    plot_toolpaths_phase(ax[r, 1], d["cent"], d[f"th{r}"], xl, yl,
                         holes=(CX, CY, VOID_R), spacing=TOW_PITCH, n=TOW_N)
    rings(ax[r, 1])
    ax[r, 1].set_aspect("equal"); ax[r, 1].set_xlim(*xl); ax[r, 1].set_ylim(*yl)
    ax[r, 1].set_title(f"{tag}: toolpaths (max$|\\zeta|$="
                       f"{float(d[f'zmax{r}']):.2f})", fontsize=10)
    cbar(ax[r, 1])                      # reserve -> column widths stay equal

    zc = np.array(d[f"zc{r}"], float).copy()
    if BLANK_VOID_CURV:
        zc[~solc] = np.nan
    cp = ax[r, 2].tripcolor(tri, facecolors=zc, cmap=CMAP_CURV, vmax=CVMAX)
    rings(ax[r, 2])
    ax[r, 2].set_aspect("equal"); ax[r, 2].set_xlim(*xl); ax[r, 2].set_ylim(*yl)
    ax[r, 2].set_title(f"{tag}: fiber curvature (void blanked)", fontsize=10)
    cbar(ax[r, 2], cp)

layout(fig, SUPTITLE, y=0.98, fontsize=13)
save_fig(fig, OUT, OUT_PAPER)
plt.show()

# %% [markdown]
# ## 7 — Elastic cloak of a conforming void
#
# NOTE on alignment: this grid deliberately mixes panel *shapes* — full-plate
# fields (8x5), a square zoom on the shell, and a line plot. With
# `set_aspect("equal")` a panel's drawn box follows its DATA aspect, so equal
# widths and equal heights cannot both hold. Panels align by row; the widths
# differ because the domains do. Set `ZOOM = None` to make the bottom row
# full-plate and the widths match.
# `ZOOM` controls the bottom-row window. Set `ZOOM = None` for the whole plate
# (mostly uniform, since orientation is designed only inside the shell).

# %%
# ------------------------------- KNOBS -------------------------------------
FIGSIZE   = (17.5, 9.6)
FIELD_MODE, FIELD_COMP = "abs", "x"     # abs | re | im | phase | t
# NB the reference panel lives on a SEPARATE hole-free mesh whose components were
# never saved, so it always shows |u| regardless of FIELD_MODE.
VMAX_PCTL = 99.5
ZOOM      = 1.85               # half-width of the bottom-row window, or None
VOID_EC, DES_EC, OBS_EC = "c", "r", "w"
DES_LS, OBS_LS = "--", ":"
TOW_PITCH, TOW_N = 0.10, 420
DIR_N, DIR_LW = 42, 2.0
CONV_COLOR, CONV_LW = "b", 1.7
SUPTITLE  = ("Elastic cloak of a CONFORMING traction-free void: orientation "
             "designed only inside the shell")
OUT, OUT_PAPER = "dolfinx_cloak_conforming_vec.png", "dfx_cloak_conf.png"
# ---------------------------------------------------------------------------
d, tri, xl, yl = D["cloak"], T["cloak"], X["cloak"], Y["cloak"]
trf = mtri.Triangulation(d["trfx"], d["trfy"], d["trfs"])
CX, CY = float(d["CX"]), float(d["CY"])
RV, RC, R_DES, red = (float(d["RV"]), float(d["RC"]), float(d["R_DES"]),
                      float(d["red"]))
zx = (CX-ZOOM, CX+ZOOM) if ZOOM else xl
zy = (CY-ZOOM, CY+ZOOM) if ZOOM else yl
VMAX = np.percentile(d["mref"], VMAX_PCTL)
tc = np.linspace(0, 2*np.pi, 240)


def rings(a, obs=True):
    a.plot(CX+RV*np.cos(tc), CY+RV*np.sin(tc), color=VOID_EC, lw=1.2)
    a.plot(CX+R_DES*np.cos(tc), CY+R_DES*np.sin(tc), color=DES_EC, ls=DES_LS,
           lw=1.5)
    if obs:
        a.plot(CX+RC*np.cos(tc), CY+RC*np.sin(tc), color=OBS_EC, ls=OBS_LS,
               lw=0.9)


fig = plt.figure(figsize=FIGSIZE)
gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.22)
panels = [(trf, None, None, "reference (hole-free plate)"),
          (tri, 0, "munc", f"uncloaked void  $J_0$={float(d['J0']):.2e}"),
          (tri, 1, "mcl", f"orientation-cloaked  ({red:.0f}$\\times$ less)")]
for k, (t_, st, envk, ttl) in enumerate(panels):
    a = fig.add_subplot(gs[0, k])
    if st is None:                       # reference: only |u| exists
        v, cm, kw, lab = d["mref"], CMAP_ABS, dict(vmax=VMAX), "$|u|$"
    else:
        v, cm, kw, lab = field_of("cloak", st, FIELD_MODE, FIELD_COMP,
                                  VMAX_PCTL, envelope=envk)
    tp = a.tripcolor(t_, v, cmap=cm, shading="gouraud", **kw)
    rings(a); a.set_aspect("equal"); a.set_xlim(*xl); a.set_ylim(*yl)
    a.set_title(f"{ttl}   [{lab}]", fontsize=10)
    cbar(a, tp)

a = fig.add_subplot(gs[1, 0])
plot_toolpaths_phase(a, d["cent"], d["thopt"], zx, zy, holes=(CX, CY, RV),
                     spacing=TOW_PITCH, n=TOW_N)
rings(a, obs=False); a.set_aspect("equal"); a.set_xlim(*zx); a.set_ylim(*zy)
a.set_title("fiber toolpaths (straight outside the red shell)", fontsize=10)
cbar(a)

a = fig.add_subplot(gs[1, 1])
lc = plot_director_field(a, d["cent"], d["thopt"], zx, zy, holes=(CX, CY, RV),
                         n=DIR_N, lw=DIR_LW)
rings(a, obs=False); a.set_aspect("equal"); a.set_xlim(*zx); a.set_ylim(*zy)
a.set_title("director field, zoomed to the design shell", fontsize=10)
cbar(a, lc, ticks=[0, np.pi/2, np.pi],
     ticklabels=["0", r"$\pi/2$", r"$\pi$"])

a = fig.add_subplot(gs[1, 2])
a.plot(np.arange(1, len(d["hist"])+1), d["hist"], CONV_COLOR, lw=CONV_LW)
a.set_xlabel("MMA iteration"); a.set_ylabel(r"scatter reduction $J_0/J$")
a.set_title(f"convergence (final {red:.0f}$\\times$)", fontsize=10)
a.grid(alpha=0.3)

layout(fig, SUPTITLE, y=0.98, fontsize=13)
save_fig(fig, OUT, OUT_PAPER)
plt.show()

# %% [markdown]
# ## 8 — Energy channelled around a joint

# %%
# ------------------------------- KNOBS -------------------------------------
FIGSIZE   = (17, 10.2)
FIELD_MODE, FIELD_COMP = "abs", "x"     # abs | re | im | phase | t
VMAX_PCTL = 99.0
CORR_EC, CORR_LS, CORR_LW = "c", "--", 1.0     # corridor edges
GUARD_EC, GUARD_LS, GUARD_LW = "r", ":", 1.4   # fatigue annulus
EXIT_EC, EXIT_LW = "#bfefff", 1.4
TOW_PITCH, TOW_N = 0.11, 460
DIR_N, DIR_LW = 52, 1.9
CORR_AMP, CORR_SIG, CORR_HALFW = 1.05, 1.60, 0.32   # driver's corridor shape
SUPTITLE  = ("Channelling elastic energy along a curved corridor around a "
             "through-hole joint")
OUT, OUT_PAPER = "dolfinx_guide_joint.png", "dfx_guide_joint.png"
# ---------------------------------------------------------------------------
d, tri, xl, yl = D["guide"], T["guide"], X["guide"], Y["guide"]
JX, JY, RJ = float(d["JX"]), float(d["JY"]), float(d["RJ"])
GUARD = float(d["GUARD"]); EXIT = d["EXIT"]
xs = np.linspace(0, xl[1], 900)
cy = JY + CORR_AMP*np.exp(-((xs - JX)/CORR_SIG)**2)
tc = np.linspace(0, 2*np.pi, 240)


def marks(a, corridor=True):
    if corridor:
        a.plot(xs, cy+CORR_HALFW, color=CORR_EC, ls=CORR_LS, lw=CORR_LW, alpha=.9)
        a.plot(xs, cy-CORR_HALFW, color=CORR_EC, ls=CORR_LS, lw=CORR_LW, alpha=.9)
    a.plot(JX+RJ*np.cos(tc), JY+RJ*np.sin(tc), "w-", lw=1.1)
    a.plot(JX+(RJ+GUARD)*np.cos(tc), JY+(RJ+GUARD)*np.sin(tc),
           color=GUARD_EC, ls=GUARD_LS, lw=GUARD_LW)
    ex, ey, er = EXIT
    a.add_patch(plt.Rectangle((ex-er, ey-er), 2*er, 2*er, ec=EXIT_EC, fc="none",
                              lw=EXIT_LW, ls="--"))
    a.set_aspect("equal"); a.set_xlim(*xl); a.set_ylim(*yl)


fig, ax = plt.subplots(2, 2, figsize=FIGSIZE)
for a, st, t in [
        (ax[0, 0], 0, f"straight fibers: downstream confinement "
                      f"{100*float(d['confd0']):.0f}%"),
        (ax[0, 1], 1, f"channelled: exit {float(d['gain']):.0f}$\\times$, "
                      f"guard {float(d['guard']):.2f}$\\times$, "
                      f"confinement {100*float(d['confd1']):.0f}%")]:
    v, cm, kw, lab = field_of("guide", st, FIELD_MODE, FIELD_COMP, VMAX_PCTL)
    tp = a.tripcolor(tri, v, cmap=cm, shading="gouraud", **kw)
    marks(a); a.set_title(f"{t}   [{lab}]", fontsize=11)
    cbar(a, tp)

plot_toolpaths_phase(ax[1, 0], d["cent"], d["thopt"], xl, yl, holes=(JX, JY, RJ),
                     spacing=TOW_PITCH, n=TOW_N)
ax[1, 0].plot(xs, cy, color=CORR_EC, ls=CORR_LS, lw=1.1, alpha=0.85)
ax[1, 0].plot(JX+(RJ+GUARD)*np.cos(tc), JY+(RJ+GUARD)*np.sin(tc),
              color=GUARD_EC, ls=GUARD_LS, lw=GUARD_LW)
ax[1, 0].set_aspect("equal"); ax[1, 0].set_xlim(*xl); ax[1, 0].set_ylim(*yl)
ax[1, 0].set_title(f"fiber toolpaths (max$|\\zeta|$={float(d['zmax']):.2f} "
                   f"$\\leq$ {float(d['zeta_all']):.2f})", fontsize=11)
cbar(ax[1, 0])

lc = plot_director_field(ax[1, 1], d["cent"], d["thopt"], xl, yl,
                         holes=(JX, JY, RJ), n=DIR_N, lw=DIR_LW)
ax[1, 1].plot(xs, cy, color=CORR_EC, ls=CORR_LS, lw=1.1, alpha=0.85)
ax[1, 1].set_aspect("equal"); ax[1, 1].set_xlim(*xl); ax[1, 1].set_ylim(*yl)
ax[1, 1].set_title("anisotropy director field", fontsize=11)
cbar(ax[1, 1], lc, ticks=[0, np.pi/2, np.pi],
     ticklabels=["0", r"$\pi/2$", r"$\pi$"])

fig.suptitle(SUPTITLE, y=0.97, fontsize=13)
save_fig(fig, OUT, OUT_PAPER)
plt.show()

# %% [markdown]
# ## 9 — Anisotropy director maps (all designs on one sheet)
#
# NOTE on alignment: the designs live on different domains (4x3, 4x3, a square
# zoom, 8x4). Under `set_aspect("equal")` their drawn widths necessarily differ.
# Drop a panel from `PANELS`, or give them all the same `zoom`, if you want a
# uniform grid.
# Edit `PANELS` to choose which designs appear, their order, and per-panel zoom.

# %%
# ------------------------------- KNOBS -------------------------------------
# (data key, title, zoom half-width or None)
PANELS = [("lens",       "wave lens",                          None),
          ("lens_hole",  "lens with two through-holes",        None),
          ("cloak",      "elastic cloak (zoomed to the shell)", 1.85),
          ("guide",      "energy guided around a joint",       None)]
NCOL      = 2
PANEL_W, PANEL_H = 8.4, 5.6
DIR_N, DIR_LW = 44, 2.0
SHOW_PROVENANCE = True        # stamp the source file + mtime on each panel
SUPTITLE  = ("Anisotropy orientation (fiber director) maps — the raw design "
             "field, before toolpath delineation")
OUT, OUT_PAPER = "dfx_orient.png", "dfx_orient.png"
# ---------------------------------------------------------------------------
import time  # noqa: E402

avail = [p for p in PANELS if p[0] in D]
nrow = int(np.ceil(len(avail)/NCOL))
fig, axes = plt.subplots(nrow, NCOL, figsize=(PANEL_W*NCOL, PANEL_H*nrow))
axs = np.atleast_1d(axes).ravel()
for a in axs[len(avail):]:
    a.axis("off")

for a, (key, title, zoom) in zip(axs, avail):
    dd, xlim, ylim = D[key], X[key], Y[key]
    th = dd["thopt"] if "thopt" in dd.files else dd["th1"]
    hol = None
    if "holes" in dd.files:
        hol = [tuple(map(float, h)) for h in np.atleast_2d(dd["holes"])]
    elif "RJ" in dd.files:
        hol = [(float(dd["JX"]), float(dd["JY"]), float(dd["RJ"]))]
    elif "RV" in dd.files and "CX" in dd.files:
        hol = [(float(dd["CX"]), float(dd["CY"]), float(dd["RV"]))]
    if zoom and hol:
        hx, hy = hol[0][0], hol[0][1]
        xlim, ylim = (hx-zoom, hx+zoom), (hy-zoom, hy+zoom)
    lc = plot_director_field(a, dd["cent"], th, xlim, ylim, holes=hol,
                             n=DIR_N, lw=DIR_LW)
    a.set_aspect("equal"); a.set_xlim(*xlim); a.set_ylim(*ylim)
    sub = ""
    if SHOW_PROVENANCE:
        f = SOURCES[key]
        sub = "\n" + os.path.basename(f) + "  (" + time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f))) + ")"
    a.set_title(title + sub, fontsize=9)

cb = fig.colorbar(lc, ax=list(axs), fraction=0.02, pad=0.01,
                  ticks=[0, np.pi/2, np.pi])
cb.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])
fig.suptitle(SUPTITLE, y=1.02, fontsize=13)
save_fig(fig, OUT, OUT_PAPER)
plt.show()

# %% [markdown]
# ## 10 — Flat-band metamaterial / design-freedom sweep
# These genuinely **re-simulate** band structures (the as-manufactured check
# needs solving), so they stay as scripts. Minutes, not seconds.

# %%
subprocess.run([sys.executable, "examples/flatband_figure.py"],
               check=True, env=dict(os.environ, PYTHONPATH=ROOT))

# %%
subprocess.run([sys.executable, "examples/flatband_sweep_figure.py"],
               check=True, env=dict(os.environ, PYTHONPATH=ROOT))

# %% [markdown]
# ---
# ## 11 — Real / imaginary parts, phase, wavefronts
#
# `results/*_data.npz` stores only `|u|` unless you have run
# `examples/resolve_full_fields.py`, which re-solves each saved design (~4 s each
# with MUMPS) and adds the four nodal components per state:
#
# ```
# ur_x0, ur_y0, ui_x0, ui_y0     baseline
# ur_x1, ur_y1, ui_x1, ui_y1     optimized
# ```
#
# With those, per node:
# `u(t) = ur*cos(wt) + ui*sin(wt)`, `|u| = sqrt(ur_x^2+ur_y^2+ui_x^2+ui_y^2)`,
# `phase = atan2(ui_x, ur_x)`. `|u|` is a smooth envelope; `Re(u)` is what
# actually looks like a wave.

# %%
# ------------------------------- KNOBS -------------------------------------
KEY        = "lens"          # any key in D that has been re-solved
STATE      = 1               # 0 = baseline, 1 = optimized
COMP       = "x"             # "x" | "y"
CMAP_SIGNED = "RdBu_r"       # diverging, for signed Re/Im
CMAP_PHASE  = "twilight"     # cyclic, for phase
CMAP_ENV    = "magma"
SYM_PCTL    = 99.0           # symmetric colour limit percentile for Re/Im
FIGSIZE     = (17, 4.6)
# ---------------------------------------------------------------------------
d, tri, xl, yl = D[KEY], T[KEY], X[KEY], Y[KEY]
need = f"ur_{COMP}{STATE}"
if need not in d.files:
    print(f"{KEY} has no full field yet -- run:\n"
          f"  python examples/resolve_full_fields.py {KEY}\n"
          f"(keys present: {sorted(k for k in d.files if k.startswith(('ur_','ui_')))})")
else:
    ur = d[f"ur_{COMP}{STATE}"]
    ui = d[f"ui_{COMP}{STATE}"]
    env = np.sqrt(d[f"ur_x{STATE}"]**2 + d[f"ur_y{STATE}"]**2
                  + d[f"ui_x{STATE}"]**2 + d[f"ui_y{STATE}"]**2)
    lim = np.percentile(np.abs(np.concatenate([ur, ui])), SYM_PCTL)

    fig, ax = plt.subplots(1, 4, figsize=FIGSIZE)
    for a, v, ttl, cm, kw in [
            (ax[0], ur, f"Re $u_{COMP}$", CMAP_SIGNED, dict(vmin=-lim, vmax=lim)),
            (ax[1], ui, f"Im $u_{COMP}$", CMAP_SIGNED, dict(vmin=-lim, vmax=lim)),
            (ax[2], np.arctan2(ui, ur), f"phase $u_{COMP}$", CMAP_PHASE,
             dict(vmin=-np.pi, vmax=np.pi)),
            (ax[3], env, "$|u|$ (envelope)", CMAP_ENV,
             dict(vmax=np.percentile(env, 99.5)))]:
        tp = a.tripcolor(tri, v, cmap=cm, shading="gouraud", **kw)
        a.set_aspect("equal"); a.set_xlim(*xl); a.set_ylim(*yl)
        a.set_title(ttl, fontsize=10)
        cbar(a, tp)
    fig.suptitle(f"{KEY}, state {STATE}: the envelope hides the wave structure "
                 "that Re/Im and phase show", y=1.02, fontsize=12)
    layout(fig)
    save_fig(fig, f"field_reim_{KEY}_{STATE}.png")
    plt.show()

# %% [markdown]
# ## 12 — Time snapshots of the propagating wave
# `u(t) = ur cos(wt) + ui sin(wt)` over one period. Set `MAKE_GIF = True` to
# write an animation instead of a strip of snapshots.

# %%
# ------------------------------- KNOBS -------------------------------------
KEY, STATE, COMP = "lens", 1, "x"
NPHASE     = 5                      # snapshots across one period
CMAP_SIGNED = "RdBu_r"
SYM_PCTL   = 99.0
FIGSIZE    = (18, 3.6)
MAKE_GIF   = False
GIF_FRAMES, GIF_FPS = 24, 12
# ---------------------------------------------------------------------------
d, tri, xl, yl = D[KEY], T[KEY], X[KEY], Y[KEY]
if f"ur_{COMP}{STATE}" not in d.files:
    print(f"run: python examples/resolve_full_fields.py {KEY}")
else:
    ur, ui = d[f"ur_{COMP}{STATE}"], d[f"ui_{COMP}{STATE}"]
    lim = np.percentile(np.abs(np.concatenate([ur, ui])), SYM_PCTL)

    if not MAKE_GIF:
        fig, ax = plt.subplots(1, NPHASE, figsize=FIGSIZE)
        for k, a in enumerate(np.atleast_1d(ax)):
            wt = 2*np.pi*k/NPHASE
            a.tripcolor(tri, ur*np.cos(wt) + ui*np.sin(wt), cmap=CMAP_SIGNED,
                        vmin=-lim, vmax=lim, shading="gouraud")
            a.set_aspect("equal"); a.set_xlim(*xl); a.set_ylim(*yl)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"$\\omega t = {k}/{NPHASE}\\cdot 2\\pi$", fontsize=9)
        fig.suptitle(f"{KEY}: one period of $u_{COMP}(t)$", y=1.04, fontsize=12)
        layout(fig)
        save_fig(fig, f"field_period_{KEY}_{STATE}.png")
        plt.show()
    else:
        from matplotlib.animation import FuncAnimation   # noqa: E402
        fig, a = plt.subplots(figsize=(7, 5))
        tp = a.tripcolor(tri, ur, cmap=CMAP_SIGNED, vmin=-lim, vmax=lim,
                         shading="gouraud")
        a.set_aspect("equal"); a.set_xlim(*xl); a.set_ylim(*yl)

        def frame(i):
            wt = 2*np.pi*i/GIF_FRAMES
            tp.set_array(ur*np.cos(wt) + ui*np.sin(wt))
            return (tp,)

        anim = FuncAnimation(fig, frame, frames=GIF_FRAMES, blit=True)
        out = f"results/field_{KEY}_{STATE}.gif"
        anim.save(out, writer="pillow", fps=GIF_FPS)
        print("wrote", out)
        plt.close(fig)
