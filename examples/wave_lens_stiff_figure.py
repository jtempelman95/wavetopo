"""
Figure: one fiber layout that focuses an elastic wave AND stiffens the panel.

Top row     wave field magnitude for straight / focus-only / co-design fibers.
Bottom row  the fiber director for focus-only and co-design, and the focus-vs-
            stiffness Pareto (co-design at a range of compliance limits) with
            the three headline designs marked.

    PYTHONPATH=. python examples/wave_lens_stiff_figure.py
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.wave_lens_stiff import build

C_STR, C_FOC, C_CO = "#52514e", "#eb6834", "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def field_mag(hl, mesh, th):
    u = hl.solve(th)
    m = np.sqrt(np.abs(u[0::2]) ** 2 + np.abs(u[1::2]) ** 2)
    return m.reshape(mesh.nely + 1, mesh.nelx + 1)


def director(ax, hl, mesh, th_hat, step=7):
    th = hl.rbf.theta(th_hat).reshape(mesh.nely, mesh.nelx)
    ys, xs = np.mgrid[0:mesh.nely:step, 0:mesh.nelx:step]
    cx = (xs + 0.5) * mesh.dx
    cy = (ys + 0.5) * mesh.dy
    t = th[::step, ::step]
    ax.quiver(cx, cy, np.cos(t), np.sin(t), color="k", pivot="mid",
              headwidth=0, headlength=0, headaxislength=0, scale=32, width=0.005)


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/wave_lens_stiff.npz")
    ap.add_argument("--pareto", default="results/wave_lens_stiff_pareto.json")
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--omega", type=float, default=40.0)
    ap.add_argument("--out", default="results/wave_lens_stiff.png")
    args = ap.parse_args()

    d = np.load(args.data)
    Lx, Ly = float(d["Lx"]), float(d["Ly"])
    ny = int(round(args.nx * Ly / Lx))
    mesh, hl, focus = build(args.nx, ny, Lx, Ly, args.omega, 0.03, 0.33, 0.7)
    hl.M_ = hl.rbf.Phi.shape[1]
    fx, fy, fr = [float(v) for v in d["focus"]]

    thetas = {"straight": np.zeros(hl.M_), "focus": d["th_focus"],
              "co": d["th_co"]}
    G = {"straight": 1.0, "focus": float(d["G_focus"]), "co": float(d["G_co"])}
    S = {"straight": 1.0,
         "focus": float(d["c_straight"]) / float(d["c_focus"]),
         "co": float(d["c_straight"]) / float(d["c_co"])}

    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.0))

    # ---- top: wave fields ------------------------------------------- #
    mags = {k: field_mag(hl, mesh, thetas[k]) for k in thetas}
    vmax = np.percentile(mags["co"], 99.5)
    titles = {"straight": "straight fibers", "focus": "focus-only lens",
              "co": "co-design (focus + stiff)"}
    cols = {"straight": C_STR, "focus": C_FOC, "co": C_CO}
    for j, k in enumerate(("straight", "focus", "co")):
        a = ax[0, j]
        a.imshow(mags[k], origin="lower", extent=[0, Lx, 0, Ly], cmap="magma",
                 vmax=vmax, aspect="equal")
        a.add_patch(plt.Rectangle((fx - fr, fy - fr), 2 * fr, 2 * fr, ec="c",
                                  fc="none", lw=1.4, ls="--"))
        a.set_title(f"{titles[k]}\nfocus {G[k]:.0f}$\\times$,  "
                    f"stiffness {S[k]:.2f}$\\times$", fontsize=10.5,
                    color=cols[k], loc="left")
        a.set_xticks([]); a.set_yticks([])

    # ---- bottom left/mid: fiber directors --------------------------- #
    for j, k in (("focus", 1), ("co", 2))[:0]:
        pass
    for col, k in ((0, "focus"), (1, "co")):
        a = ax[1, col]
        director(a, hl, mesh, thetas[k])
        a.set_xlim(0, Lx); a.set_ylim(0, Ly); a.set_aspect("equal")
        a.set_title(f"{titles[k]} — fibers", fontsize=10.5, color=cols[k],
                    loc="left")
        a.set_xticks([]); a.set_yticks([])

    # ---- bottom right: focus-vs-stiffness Pareto -------------------- #
    a = ax[1, 2]
    if os.path.exists(args.pareto):
        rows = json.load(open(args.pareto))
        co = [r for r in rows if r["label"].startswith("co ")]
        a.plot([r["stiff"] for r in co], [r["G"] for r in co], "-o",
                color=C_CO, lw=2.0, ms=5, label="co-design front")
        for r in rows:
            if r["label"] == "focus-only":
                a.plot(r["stiff"], r["G"], "s", color=C_FOC, ms=9,
                       label="focus-only")
            if r["label"] == "straight":
                a.plot(r["stiff"], r["G"], "^", color=C_STR, ms=9,
                       label="straight")
    else:
        for k in ("straight", "focus", "co"):
            a.plot(S[k], G[k], "o", color=cols[k], ms=10, label=titles[k])
    a.axvline(1.0, color=INK2, lw=1.0, ls=":")
    a.text(1.0, a.get_ylim()[1], " as stiff as\n straight", fontsize=8,
           color=INK2, va="top")
    a.set_xlabel(r"stiffness / straight-fiber stiffness", color=INK, fontsize=10)
    a.set_ylabel(r"wave focus gain", color=INK, fontsize=10)
    a.set_title("focus vs stiffness trade-off", fontsize=10.5, loc="left",
                color=INK)
    a.legend(fontsize=8.5, frameon=False, labelcolor=INK2)
    tidy(a)

    fig.suptitle("One fiber layout, two functions: a curvilinear-fiber lens that "
                 f"focuses a wave {G['co']:.0f}$\\times$ while staying as stiff as "
                 "a straight-fiber panel", fontsize=12.5, y=0.99, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=185, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
