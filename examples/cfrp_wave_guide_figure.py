"""
Figure for examples/cfrp_wave_guide.py: co-designed topology + toolpath routing a
wave along a corridor between two through-holes, while stiffening the panel.

Reports a baseline-independent CONFINEMENT = corridor energy / interior energy,
not the E/E0 ratio (which explodes when the straight-fiber baseline sits near the
signed-weight zero crossing).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from examples.cfrp_wave_guide import setup, corridor_coords, TUBE_HALF, CORRIDOR, HOLES

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def confinement(mesh, P, u, tube):
    e = (np.abs(u[0::2]) ** 2 + np.abs(u[1::2]) ** 2)
    xs, ys = mesh.npx[:, 0], mesh.npx[:, 1]
    interior = (xs > 0.4) & (xs < mesh.Lx - 0.4) & (ys > 0.3) & (ys < mesh.Ly - 0.3)
    tot = e[interior].sum()
    return float(e[interior & tube].sum() / tot) if tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/cfrp_wave_guide.npz")
    ap.add_argument("--h", type=float, default=0.07)
    ap.add_argument("--omega", type=float, default=9.0)
    ap.add_argument("--out", default="results/cfrp_wave_guide.png")
    args = ap.parse_args()

    d = np.load(args.data)
    mesh, mat, P, wsig, interior = setup(args.h, args.omega, 0.006, 0.5)
    tri = mtri.Triangulation(mesh.npx[:, 0], mesh.npx[:, 1], mesh.tris)
    xs, ys = mesh.npx[:, 0], mesh.npx[:, 1]
    tube = corridor_coords(mesh.npx, CORRIDOR)[0] < TUBE_HALF

    y, th = d["y"], d["th"]
    u_str = P.dynamic(np.full(mesh.tris.shape[0], d["vol"]),
                      np.zeros(mesh.tris.shape[0]), False)[1]
    u_opt = P.dynamic(y, th, False)[1]
    C_str = confinement(mesh, P, u_str, tube)
    C_opt = confinement(mesh, P, u_opt, tube)
    print(f"corridor confinement: straight {C_str:.3f} -> optimized {C_opt:.3f}")

    def field(u):
        return np.sqrt(np.abs(u[0::2]) ** 2 + np.abs(u[1::2]) ** 2)

    fig, ax = plt.subplots(2, 3, figsize=(17, 6.6))

    def frame(a, holes=True, corr=True):
        for cx, cy, r in HOLES:
            a.add_patch(plt.Circle((cx, cy), r, fc="white", ec="k", lw=1.2,
                                   zorder=5))
        if corr:
            a.plot(CORRIDOR[:, 0], CORRIDOR[:, 1], "--", color="#1baf7a", lw=1.6,
                   alpha=0.8, zorder=6)
        a.set_xlim(0, mesh.Lx); a.set_ylim(0, mesh.Ly); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])

    # (a) topology
    a = ax[0, 0]
    a.tripcolor(tri, facecolors=y, cmap="bone_r", vmin=0, vmax=1)
    frame(a, corr=False)
    a.set_title(f"(a) topology $z$   vol={float(d['vol']):.2f}", fontsize=10,
                loc="left", color=INK)

    # (b) fiber toolpath (director)
    a = ax[0, 1]
    step = max(mesh.tris.shape[0] // 900, 1)
    idx = np.arange(0, mesh.tris.shape[0], step)
    solid = y[idx] > 0.5
    a.quiver(mesh.cx[idx][solid], mesh.cy[idx][solid],
             np.cos(th[idx][solid]), np.sin(th[idx][solid]), color="k",
             pivot="mid", headwidth=0, headlength=0, headaxislength=0,
             scale=40, width=0.004)
    frame(a)
    a.set_title("(b) fiber toolpath $\\theta$ (solid only)", fontsize=10,
                loc="left", color=INK)

    # (c) convergence of compliance
    a = ax[0, 2]
    a.plot(d["hist_c"], color="#2a78d6", lw=2)
    a.axhline(1.0, color=INK2, ls=":", lw=1)
    a.set_xlabel("iteration", fontsize=9); a.set_ylabel("$c/c_0$", fontsize=9)
    a.set_title(f"(c) compliance -> {float(d['c_ratio']):.2f}$\\times$ "
                "(stiffer)", fontsize=10, loc="left", color=INK)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(alpha=0.3)

    # (d) straight-fiber field
    ms = field(u_str); mo = field(u_opt)
    vmax = np.percentile(mo, 99.3)
    a = ax[1, 0]
    a.tripcolor(tri, ms, cmap="magma", vmax=vmax, shading="gouraud")
    frame(a)
    a.set_title(f"(d) straight fibers: confinement {C_str:.2f}", fontsize=10,
                loc="left", color=INK)

    # (e) optimized field
    a = ax[1, 1]
    a.tripcolor(tri, mo, cmap="magma", vmax=vmax, shading="gouraud")
    frame(a)
    a.set_title(f"(e) co-design: energy follows corridor, confinement {C_opt:.2f}",
                fontsize=10, loc="left", color=INK)

    # (f) signed target weight (what was asked for)
    a = ax[1, 2]
    a.tripcolor(tri, wsig, cmap="RdBu_r", vmin=-1, vmax=1, shading="gouraud")
    frame(a)
    a.set_title("(f) target: +1 corridor, -leak elsewhere", fontsize=10,
                loc="left", color=INK)

    fig.suptitle("Co-designed topology + toolpath on a conforming CFRP mesh: route "
                 "wave energy through a corridor between two through-holes "
                 f"(confinement {C_str:.2f}$\\to${C_opt:.2f}) while stiffening the "
                 f"panel ({float(d['c_ratio']):.2f}$\\times$ compliance)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
