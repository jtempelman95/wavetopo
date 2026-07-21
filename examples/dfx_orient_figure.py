"""Rebuild the anisotropy (fiber director) map figure from the CURRENT design
data files, so it cannot drift out of step with the result figures it refers to.

The previous version of this figure had no generator -- it was produced ad hoc --
and silently went stale: its cloak panel still showed the old SOFT-VOID design on
the old domain while the paper's cloak figure had been replaced by the conforming
8x5 result.  Panels here are read straight from the *_data.npz written by the
drivers, and each panel prints the file and mtime it came from.

    PYTHONPATH=/home/jrt/wavetopo /home/jrt/miniforge3/bin/python3 \
        examples/dfx_orient_figure.py
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
from wavetopo import figlib as _F; _F.journal()  # publication typography
import matplotlib.pyplot as plt

from wavetopo import figlib as F

from wavetopo.dolfinx_viz import plot_director_field

# (npz, title, xlim, ylim, holes, zoom) -- holes is (cx,cy,r) or a list; zoom is
# a half-width about the void, used where the design is confined to a small shell
# and a full-plate view would be mostly uniform.
PANELS = [
    ("results/dolfinx_lens_data.npz",
     "wave lens", (0, 4.0), (0, 3.0), None, None),
    ("results/dolfinx_lens_hole_data.npz",
     "lens with two through-holes", (0, 4.0), (0, 3.0),
     [(1.8, 1.5-0.62, 0.28), (1.8, 1.5+0.62, 0.28)], None),
    ("results/dolfinx_cloak_conforming_data.npz",
     "elastic cloak (zoomed to the design shell)", None, None, "auto", 1.85),
    ("results/dolfinx_guide_joint_data.npz",
     "energy guided around a joint", None, None, "auto", None),
]


def theta_of(d):
    """The per-cell design field, whatever the driver happened to call it."""
    for k in ("thopt", "th1", "theta", "th"):
        if k in d:
            return np.asarray(d[k]).ravel()
    raise KeyError(f"no orientation field in {list(d.keys())}")


avail = [p for p in PANELS if os.path.exists(p[0])]
if not avail:
    raise SystemExit("no design data files found -- run the drivers first")

# 2 x 2 rather than 1 x N: the panels have different aspect ratios (4x3, 4x3,
# 8x5, 8x4) and a single row squeezes them all into equal-width slots.
ncol = 2 if len(avail) > 2 else len(avail)
nrow = int(np.ceil(len(avail)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(8.4*ncol, 5.6*nrow))
ax = np.atleast_1d(axes).ravel()
for a_ in ax[len(avail):]:
    a_.axis("off")

for a_, (path, title, xlim, ylim, holes, zoom) in zip(ax, avail):
    d = np.load(path)
    cent = d["cent"]; th = theta_of(d)
    if xlim is None:
        xlim = (0.0, float(d["Lx"]) if "Lx" in d else
                float(np.ceil(cent[:, 0].max())))
    if ylim is None:
        ylim = (0.0, float(d["Ly"]) if "Ly" in d else
                float(np.ceil(cent[:, 1].max())))
    if holes == "auto":                       # recover the void from the file
        # The cloak stores its centre as CX/CY and the joint study as JX/JY.  An
        # earlier version defaulted the missing key to 0 and silently drew the
        # cloak void at the ORIGIN; require the keys instead of guessing.
        if "RJ" in d:
            holes = (float(d["JX"]), float(d["JY"]), float(d["RJ"]))
        elif "RV" in d:
            if "CX" not in d or "CY" not in d:
                raise KeyError(f"{path}: has RV but no CX/CY -- cannot place the "
                               "void; re-run the driver so it saves them")
            holes = (float(d["CX"]), float(d["CY"]), float(d["RV"]))
        else:
            holes = None
    if zoom is not None and holes is not None:
        hx, hy = holes[0], holes[1]
        xlim = (hx-zoom, hx+zoom); ylim = (hy-zoom, hy+zoom)
    lc = plot_director_field(a_, cent, th, xlim, ylim, holes=holes,
                             n=44, lw=2.0)
    a_.set_xlim(*xlim); a_.set_ylim(*ylim)
    mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
    a_.set_title(f"{title}\n{os.path.basename(path)}  ({mt})", fontsize=9)
    a_.set_aspect("equal")
    print(f"  {title:38s} <- {path}  ({mt})", flush=True)

cb = fig.colorbar(lc, ax=list(ax), fraction=0.02, pad=0.01,
                  ticks=[0, np.pi/2, np.pi])
cb.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])
fig.suptitle("Anisotropy orientation (fiber director) maps -- the raw design "
             "field, before toolpath delineation", y=1.02, fontsize=13)
F.save_pair(fig, "results/dfx_orient.png", "docs/paper/figs/dfx_orient.png")
print("saved results/dfx_orient.png and docs/paper/figs/dfx_orient.png")
