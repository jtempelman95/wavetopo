"""Anisotropy (fiber director) map figure, read straight from the CURRENT design
*_data.npz files so it cannot drift out of step with the result figures.  Each panel
prints the file and mtime it came from.

Importable: ``build()`` returns a live Figure; ``main()`` is the batch CLI.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from wavetopo import figlib as F
from wavetopo.dolfinx_viz import plot_director_tows

# (npz, title, xlim, ylim, holes, zoom)
PANELS = [
    ("results/dolfinx_lens_data.npz", "wave lens", (0, 4.0), (0, 3.0), None, None),
    ("results/dolfinx_lens_hole_data.npz", "lens with two through-holes", (0, 4.0), (0, 3.0),
     [(1.8, 1.5 - 0.62, 0.28), (1.8, 1.5 + 0.62, 0.28)], None),
    ("results/dolfinx_cloak_conforming_data.npz",
     "elastic cloak (zoomed to the design shell)", None, None, "auto", 1.85),
    ("results/dolfinx_guide_joint_data.npz",
     "energy guided around a joint", None, None, "auto", None),
]


def theta_of(d):
    for k in ("thopt", "th1", "theta", "th"):
        if k in d:
            return np.asarray(d[k]).ravel()
    raise KeyError(f"no orientation field in {list(d.keys())}")


def build():
    F.journal()
    avail = [p for p in PANELS if os.path.exists(p[0])]
    if not avail:
        raise SystemExit("no design data files found -- run the drivers first")
    ncol = 2 if len(avail) > 2 else len(avail)
    nrow = int(np.ceil(len(avail) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.4 * ncol, 5.6 * nrow))
    ax = np.atleast_1d(axes).ravel()
    for a_ in ax[len(avail):]:
        a_.axis("off")
    lc = None
    for a_, (path, title, xlim, ylim, holes, zoom) in zip(ax, avail):
        d = np.load(path)
        cent = d["cent"]; th = theta_of(d)
        if xlim is None:
            xlim = (0.0, float(d["Lx"]) if "Lx" in d else float(np.ceil(cent[:, 0].max())))
        if ylim is None:
            ylim = (0.0, float(d["Ly"]) if "Ly" in d else float(np.ceil(cent[:, 1].max())))
        if holes == "auto":
            if "RJ" in d:
                holes = (float(d["JX"]), float(d["JY"]), float(d["RJ"]))
            elif "RV" in d:
                if "CX" not in d or "CY" not in d:
                    raise KeyError(f"{path}: has RV but no CX/CY -- re-run the driver")
                holes = (float(d["CX"]), float(d["CY"]), float(d["RV"]))
            else:
                holes = None
        if zoom is not None and holes is not None:
            hx, hy = holes[0], holes[1]
            xlim = (hx - zoom, hx + zoom); ylim = (hy - zoom, hy + zoom)
        lc = plot_director_tows(a_, cent, th, xlim, ylim, holes=holes, spacing=0.13, lw=1.2)
        a_.set_xlim(*xlim); a_.set_ylim(*ylim)
        mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        a_.set_title(f"{title}\n{os.path.basename(path)}  ({mt})", fontsize=9)
        a_.set_aspect("equal")
    cb = fig.colorbar(lc, ax=list(ax), fraction=0.02, pad=0.01, ticks=[0, np.pi / 2, np.pi])
    cb.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])
    fig.suptitle("Anisotropy orientation (fiber director) maps -- the raw design "
                 "field, before toolpath delineation", y=1.02, fontsize=13)
    return fig


def main():
    plt.switch_backend("Agg")
    fig = build()
    F.save_pair(fig, "results/dfx_orient.png", "docs/paper/figs/dfx_orient.png")
    print("saved dfx_orient.png")


if __name__ == "__main__":
    main()
