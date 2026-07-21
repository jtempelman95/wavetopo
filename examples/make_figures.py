#!/usr/bin/env python3
"""
Rebuild every paper figure from SAVED DATA -- no re-solving, no dolfinx.

Every driver writes a results/*_data.npz holding the triangulation, the fields and
the design, so all figures can be regenerated in seconds in the base
numpy/scipy env.  This is the one script to play with when you want to restyle:
change a colormap, a director density, a layout, and re-run.

    PYTHONPATH=/home/jrt/wavetopo /home/jrt/miniforge3/bin/python3 \
        examples/make_figures.py --list          # what exists / what is stale
        examples/make_figures.py --all           # rebuild everything
        examples/make_figures.py lens cloak      # rebuild just these
        examples/make_figures.py --all --no-publish   # leave docs/paper/figs alone

Staleness: a figure is STALE when its source data is newer than the published
figure.  That is exactly the failure that produced a paper whose Fig 6 showed a
design that no longer appeared anywhere else, so it is checked explicitly.

--- STYLE KNOBS (edit these) ------------------------------------------------ """
CMAP_FIELD = "magma"        # wave amplitude panels
CMAP_CURV = "viridis"       # fiber-curvature panels
DIRECTOR_N = 44             # director segments across the domain
DIRECTOR_LW = 2.0
TOW_PITCH = 0.13            # phase-field tow spacing
TOW_N = 420                 # tow raster resolution
FIELD_PCTL = 99.0           # colour saturation percentile
DPI = 145
# ----------------------------------------------------------------------------

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib
# NB: the Agg backend is selected in main(), NOT at import.  Forcing it here
# would silently disable inline rendering for anyone importing these builders
# from the VS Code interactive window (examples/figures_interactive.py).
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from wavetopo.dolfinx_viz import plot_toolpaths_phase, plot_director_field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
FIGDIR = "docs/paper/figs"


# Plotting primitives live in wavetopo/figlib.py so this batch driver and the
# interactive cell script (examples/figures_interactive.py) cannot drift apart.
from wavetopo import figlib as F                                      # noqa: E402

F.STYLE.update(cmap_field=CMAP_FIELD, cmap_curv=CMAP_CURV,
               director_n=DIRECTOR_N, director_lw=DIRECTOR_LW,
               tow_pitch=TOW_PITCH, tow_n=TOW_N,
               field_pctl=FIELD_PCTL, dpi=DPI)

load = F.load
field_panel = F.field_panel
tow_panel = F.tow_panel
dir_panel = F.dir_panel
circles = F.circles
focus_marker = F.focus_marker


MODE_SUFFIX = ""


def _suffix(name):
    return name if not MODE_SUFFIX else name.replace(".png", MODE_SUFFIX + ".png")


def save(fig, results_name, published_name, publish, label_panels=True):
    """Write the figure and RETURN it.  Closing is the caller's job: the CLI
    closes to bound memory, an interactive session must not, or the cell renders
    nothing."""
    if label_panels:
        F.panel_labels(fig)        # (a) (b) (c) ... skipping colorbar axes
    # a non-|u| rendering is a variant, never a replacement for the paper figure
    F.publish(fig, _suffix(results_name),
              published_name if not MODE_SUFFIX else None,
              publish and not MODE_SUFFIX, root=ROOT)
    return fig


# ------------------------------------------------------------- builders -----
def build_lens(publish):
    d, tri, xl, yl = load("results/dolfinx_lens_data.npz")
    mk = focus_marker(0.78*xl[1], yl[1]/2, 0.18)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    vmax = np.percentile(d["m1"], FIELD_PCTL)
    for A, st, ttl in [(ax[0, 0], 0, "straight fibers (baseline)"),
                       (ax[0, 1], 1, f"orientation-optimized lens: "
                                     f"{float(d['gain']):.0f}$\\times$ focus gain")]:
        v, cm, kw, lab = F.field_values(d, st)
        field_panel(A, tri, v, xl, yl, f"{ttl}   [{lab}]", marks=mk, cmap=cm, **kw)
    tow_panel(ax[1, 0], d["cent"], d["thopt"], xl, yl, "fiber toolpaths")
    dir_panel(ax[1, 1], d["cent"], d["thopt"], xl, yl, "anisotropy director field")
    F.layout(fig, "Curvilinear-fiber elastic wave lens", y=0.97)
    return save(fig, "dolfinx_lens.png", "dfx_lens.png", publish)


def _two_row_curl(path, res_name, pub_name, suptitle, publish, holes=None):
    """Layout shared by the curvature-constrained studies: rows = unconstrained
    vs constrained, columns = field, toolpaths, curvature."""
    d, tri, xl, yl = load(path)
    hol = holes
    if hol is None and "holes" in d:
        hol = [tuple(map(float, h)) for h in np.atleast_2d(d["holes"])]
    zmax = max(float(np.abs(d["zc0"]).max()), float(np.abs(d["zc1"]).max()))
    vmax = np.percentile(d["m0"], FIELD_PCTL)
    mk = circles(hol, ec="w") if hol else None
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    for r, (m, th, zc, tag) in enumerate([
            (d["m0"], d["th0"], d["zc0"], "unconstrained"),
            (d["m1"], d["th1"], d["zc1"], f"$|\\zeta|\\leq${float(d['zeta_all']):.2f}")]):
        g = d.get(f"gain{r}", d.get("gain"))
        field_panel(ax[r, 0], tri, m, xl, yl,
                    f"{tag}: field" + (f"  ({float(g):.0f}$\\times$)" if g is not None else ""),
                    vmax, mk)
        tow_panel(ax[r, 1], d["cent"], th, xl, yl,
                  f"{tag}: toolpaths  (max$|\\zeta|$="
                  f"{float(d[f'zmax{r}']):.2f})", holes=hol)
        F.curv_panel(ax[r, 2], tri, d["tris"], zc, xl, yl,
                     f"{tag}: fiber curvature $|\\zeta|$", zmax, mk)
    F.layout(fig, suptitle, y=0.98)
    return save(fig, res_name, pub_name, publish)


def build_lens_curl(publish):
    _two_row_curl("results/dolfinx_lens_curl_data.npz",
                  "dolfinx_lens_curl.png", "dfx_lens_curl.png",
                  "Curvature-constrained wave lens (manufacturable toolpaths)",
                  publish)


def build_lens_hole(publish):
    _two_row_curl("results/dolfinx_lens_hole_data.npz",
                  "dolfinx_lens_hole.png", "dfx_lens_hole.png",
                  "Wave lens with two prescribed circular through-holes", publish)


def build_lens_hole_asym(publish):
    d, tri, xl, yl = load("results/dolfinx_lens_hole_asym_data.npz")
    hol = [tuple(map(float, h)) for h in np.atleast_2d(d["holes"])]
    mk = circles(hol, ec="w")
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    vmax = np.percentile(d["m1"], FIELD_PCTL)
    field_panel(ax[0, 0], tri, d["m0"], xl, yl, "straight fibers (baseline)", vmax, mk)
    field_panel(ax[0, 1], tri, d["m1"], xl, yl,
                f"re-optimized: {float(d['gain']):.0f}$\\times$ focus gain", vmax, mk)
    tow_panel(ax[1, 0], d["cent"], d["th1"], xl, yl,
              f"fiber toolpaths (max$|\\zeta|$={float(d['zmax']):.2f}, limit "
              f"{float(d['zeta_all']):.2f})", holes=hol)
    dir_panel(ax[1, 1], d["cent"], d["th1"], xl, yl, "anisotropy director field",
              holes=hol)
    fig.suptitle("Wave lens with two ASYMMETRIC through-holes: the design is "
                 "re-optimized per geometry", y=0.97, fontsize=13)
    F.layout(fig, y=0.97)
    return save(fig, "dolfinx_lens_hole_asym.png", "dfx_lens_hole_asym.png", publish)


def build_lens_multi(publish):
    d, tri, xl, yl = load("results/dolfinx_lens_multi_data.npz")
    FA, FB, NR = (0.62*xl[1], 0.30*yl[1], 0.16), (0.62*xl[1], 0.70*yl[1], 0.16), \
                 (0.86*xl[1], 0.50*yl[1], 0.16)

    def mk(a_):
        for (x, y, r), c in [(FA, "cyan"), (FB, "cyan"), (NR, "red")]:
            a_.add_patch(plt.Circle((x, y), r, ec=c, fc="none", lw=1.3, ls="--"))
    c0 = float(d["tA0"])/float(d["tN0"]); c1 = float(d["tA1"])/float(d["tN1"])
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    vmax = np.percentile(d["m1"], FIELD_PCTL)
    field_panel(ax[0, 0], tri, d["m0"], xl, yl,
                f"baseline: target/null contrast {c0:.1f}", vmax, mk)
    field_panel(ax[0, 1], tri, d["m1"], xl, yl,
                f"two foci + a null: contrast {c1:.1f} ({c1/c0:.1f}$\\times$)",
                vmax, mk)
    tow_panel(ax[1, 0], d["cent"], d["thopt"], xl, yl, "fiber toolpaths")
    dir_panel(ax[1, 1], d["cent"], d["thopt"], xl, yl, "anisotropy director field")
    fig.suptitle("Multi-target beam shaping: two foci (cyan) and a null (red)",
                 y=0.97, fontsize=13)
    F.layout(fig, y=0.97)
    return save(fig, "dolfinx_lens_multi.png", "dfx_lens_multi.png", publish)


def build_cloak_curl(publish):
    """Cloak curvature study (SOFT-void variant, dolfinx_cloak_soft.py curl).

    Needs a dedicated builder, not the generic two-row layout: this data file
    predates the convention of storing geometry, so it carries NO holes/RV/CX
    keys -- only ``solc``, the per-triangle solid mask.  Feeding it to the generic
    builder drew no void at all, ran the toolpaths straight through the
    scatterer, and left the curvature un-blanked inside it: a picture of a
    cloaking problem with nothing to cloak.  The void is a SIMP soft void, so the
    mesh DOES have cells inside it and masking must use ``solc``.
    """
    d, tri, xl, yl = load("results/dolfinx_cloak_curl_data.npz")
    # geometry constants of examples/dolfinx_cloak_soft.py (absent from the file)
    CX, CY, RV, RC = xl[1]/2, yl[1]/2, 0.55, 1.4
    solc = d["solc"].astype(bool)
    zmax = float(np.nanmax([np.nanmax(d["zc0"]), np.nanmax(d["zc1"])]))
    vmax = np.percentile(d["m0"], FIELD_PCTL)
    tc = np.linspace(0, 2*np.pi, 240)

    def rings(a_):
        a_.plot(CX+RV*np.cos(tc), CY+RV*np.sin(tc), "c-", lw=1.3)
        a_.plot(CX+RC*np.cos(tc), CY+RC*np.sin(tc), "w:", lw=1.0)

    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    for r in (0, 1):
        tag = "unconstrained" if r == 0 else \
            f"$|\\zeta|\\leq${float(d['zeta_all']):.2f}"
        red = float(d[f"red{r}"])
        field_panel(ax[r, 0], tri, d[f"m{r}"], xl, yl,
                    f"{tag}: field ({red:.0f}$\\times$ scatter reduction)",
                    vmax, rings)
        tow_panel(ax[r, 1], d["cent"], d[f"th{r}"], xl, yl,
                  f"{tag}: toolpaths (max$|\\zeta|$={float(d[f'zmax{r}']):.2f})",
                  holes=(CX, CY, RV))
        rings(ax[r, 1])
        F.attach_cbar(ax[r, 1])          # keep column widths equal
        zc = np.array(d[f"zc{r}"], float).copy()
        zc[~solc] = np.nan          # curvature is meaningless with no material
        F.curv_panel(ax[r, 2], tri, d["tris"], zc, xl, yl,
                     f"{tag}: fiber curvature (void blanked)", zmax, rings)
    fig.suptitle("Cloak with the void as the prescribed through-hole "
                 "($\\zeta_{\\rm all}=1/R_V$; soft-void variant)",
                 y=0.98, fontsize=13)
    F.layout(fig, y=0.98)
    return save(fig, "dolfinx_cloak_curl.png", "dfx_cloak_curl.png", publish)


def _delegate(script):
    """Figures whose rebuild genuinely re-simulates run as a subprocess; they
    write files rather than returning a live figure."""
    def f(publish):
        env = dict(os.environ, PYTHONPATH=ROOT,
                   MPLCONFIGDIR=os.environ.get("MPLCONFIGDIR", "/tmp/claude-1000/mpl"))
        subprocess.run([sys.executable, f"examples/{script}"], check=True, env=env)
    return f


# name -> (source data files, builder)
FIGS = {
    "lens":        (["results/dolfinx_lens_data.npz"], build_lens),
    "lens_curl":   (["results/dolfinx_lens_curl_data.npz"], build_lens_curl),
    "lens_multi":  (["results/dolfinx_lens_multi_data.npz"], build_lens_multi),
    "lens_hole":   (["results/dolfinx_lens_hole_data.npz"], build_lens_hole),
    "lens_asym":   (["results/dolfinx_lens_hole_asym_data.npz"], build_lens_hole_asym),
    "cloak":       (["results/dolfinx_cloak_conforming_data.npz"],
                    _delegate("dfx_cloak_figure.py")),
    "cloak_curl":  (["results/dolfinx_cloak_curl_data.npz"], build_cloak_curl),
    "guide":       (["results/dolfinx_guide_joint_data.npz"],
                    _delegate("dfx_guide_figure.py")),
    "orient":      (["results/dolfinx_lens_data.npz",
                     "results/dolfinx_lens_hole_data.npz",
                     "results/dolfinx_cloak_conforming_data.npz",
                     "results/dolfinx_guide_joint_data.npz"],
                    _delegate("dfx_orient_figure.py")),
    "flatband":    (["results/flatband_data.npz"], _delegate("flatband_figure.py")),
    "sweep":       ([f"results/flatband_C{t}.npz" for t in ("1", "2", "3", "3d")],
                    _delegate("flatband_sweep_figure.py")),
}
PUBLISHED = {
    "lens": "dfx_lens.png", "lens_curl": "dfx_lens_curl.png",
    "lens_multi": "dfx_lens_multi.png", "lens_hole": "dfx_lens_hole.png",
    "lens_asym": "dfx_lens_hole_asym.png", "cloak": "dfx_cloak_conf.png",
    "cloak_curl": "dfx_cloak_curl.png", "guide": "dfx_guide_joint.png",
    "orient": "dfx_orient.png", "flatband": "flatband.png",
    "sweep": "flatband_sweep.png",
}


def status(name):
    srcs, _ = FIGS[name]
    missing = [s for s in srcs if not os.path.exists(s)]
    if missing:
        return "NO DATA", missing
    pub = f"{FIGDIR}/{PUBLISHED[name]}"
    if not os.path.exists(pub):
        return "MISSING", []
    tsrc = max(os.path.getmtime(s) for s in srcs)
    return ("STALE" if tsrc > os.path.getmtime(pub) else "ok"), []


def main():
    matplotlib.use("Agg")          # batch driver: no display, bounded memory
    F.journal()                    # publication typography, applied before any figure
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"any of: {', '.join(FIGS)}")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stale", action="store_true", help="rebuild only stale figures")
    ap.add_argument("--mode", default="abs",
                    choices=["abs", "re", "im", "phase", "t"],
                    help="field representation: |u| (default), Re, Im, phase, "
                         "or a time snapshot. Non-abs modes need "
                         "examples/resolve_full_fields.py to have been run, and "
                         "are written with a _<mode> suffix so they never "
                         "overwrite the published |u| figures.")
    ap.add_argument("--comp", default="x", choices=["x", "y"],
                    help="displacement component used by re/im/phase/t")
    ap.add_argument("--wt", type=float, default=0.0,
                    help="omega*t for --mode t")
    ap.add_argument("--no-publish", action="store_true",
                    help="write results/ only, leave docs/paper/figs untouched")
    a = ap.parse_args()

    if a.list or not (a.names or a.all or a.stale):
        print(f"{'figure':12} {'status':8} {'published':26} source data")
        for n in FIGS:
            st, miss = status(n)
            src = ", ".join(os.path.basename(s) for s in FIGS[n][0])
            print(f"{n:12} {st:8} {PUBLISHED[n]:26} {src}")
            for m in miss:
                print(f"{'':22}   missing: {m}")
        if not (a.names or a.all or a.stale):
            print("\n(nothing rebuilt: pass --all, --stale, or figure names)")
        return

    F.STYLE.update(field_mode=a.mode, field_comp=a.comp, phase_wt=a.wt)
    if a.mode != "abs":
        global MODE_SUFFIX
        MODE_SUFFIX = f"_{a.mode}{a.comp}"
        print(f"field mode: {a.mode} (u_{a.comp}) -> writing *{MODE_SUFFIX}.png; "
              f"the published |u| figures are left untouched")
    todo = list(FIGS) if (a.all or a.stale) else a.names
    if a.stale:
        todo = [n for n in todo if status(n)[0] in ("STALE", "MISSING")]
    bad = [n for n in todo if n not in FIGS]
    if bad:
        raise SystemExit(f"unknown figure(s): {bad}\navailable: {list(FIGS)}")

    t0 = time.time()
    ok, skipped, failed = [], [], []
    for n in todo:
        st, miss = status(n)
        if st == "NO DATA":
            print(f"  {n:12} SKIP  (missing {miss[0]})", flush=True)
            skipped.append(n); continue
        try:
            print(f"  {n:12} building...", flush=True)
            FIGS[n][1](not a.no_publish)
            plt.close("all")
            ok.append(n)
        except Exception as e:
            print(f"  {n:12} FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(n)
    print(f"\n{len(ok)} rebuilt, {len(skipped)} skipped, {len(failed)} failed "
          f"in {time.time()-t0:.1f}s")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
