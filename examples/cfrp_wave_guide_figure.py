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
from wavetopo.cfrp_wave import curl_operators, director_curl
from wavetopo.dolfinx_viz import plot_toolpaths_phase

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
    ap.add_argument("--h", type=float, default=0.06)
    ap.add_argument("--out", default="results/cfrp_wave_guide.png")
    args = ap.parse_args()

    d = np.load(args.data)
    # render the field at the SAME omega/eta the design was optimized at, so the
    # reported confinement is the as-optimized value, not a lower-damping proxy.
    omega = float(d["omega"]); eta = float(d["eta"])
    mesh, mat, P, wsig, interior = setup(args.h, omega, eta, 0.5)
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

    cents = np.column_stack([mesh.cx, mesh.cy])

    # (a) as-manufactured fiber TOOLPATHS: the reference-paper wave projection
    # chi = 1/2 + 1/2 cos(2 pi psi / d) drawn only where there is fiber (dens>0.5),
    # so black tows on a white matrix/void -- the actual printable tow layout, not
    # a density map.  The soft-matrix wave channel appears as the tow-free band.
    a = ax[0, 0]
    plot_toolpaths_phase(a, cents, th, (0, mesh.Lx), (0, mesh.Ly), holes=HOLES,
                         spacing=0.14, dens=y, dens_thresh=0.5, lw=0.6)
    a.plot(CORRIDOR[:, 0], CORRIDOR[:, 1], "--", color="#1baf7a", lw=1.6,
           alpha=0.9, zorder=6)
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"(a) fiber toolpaths (black tows / matrix / hole)   "
                f"vol={float(d['vol']):.2f}", fontsize=10, loc="left", color=INK)

    # (b) fiber-curvature map |zeta| on the fiber region (printability), as in the
    # reference-paper curl contour plots.
    a = ax[0, 1]
    Px, Py = curl_operators(mesh)
    zeta = director_curl(th, Px, Py)
    zall = 1.0 / min(r for _, _, r in HOLES)
    zshow = np.where(y > 0.5, np.minimum(zeta, 2 * zall), np.nan)
    tc = a.tripcolor(tri, facecolors=zshow, cmap="viridis", vmin=0, vmax=2 * zall)
    cb = fig.colorbar(tc, ax=a, fraction=0.03, pad=0.02)
    cb.set_label("$|\\zeta|$  (turning curvature)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(zall, color="r", lw=1.2)          # the 1/R printable bound
    over = (mesh.area[y > 0.5][zeta[y > 0.5] > zall].sum()
            / mesh.area[y > 0.5].sum())
    frame(a, corr=False)
    a.set_title(f"(b) fiber curvature $|\\zeta|$ (max {zeta[y>0.5].max():.1f}, "
                f"{100*over:.0f}% over $1/R$={zall:.1f})",
                fontsize=9, loc="left", color=INK)

    # (c) convergence of compliance AND mass (the channel + holes shed mass)
    a = ax[0, 2]
    a.plot(d["hist_c"], color="#2a78d6", lw=2, label="compliance $c/c_0$")
    if "hist_mass" in d.files:
        a.plot(d["hist_mass"], color="#d1662a", lw=2, label="mass / solid")
    a.axhline(1.0, color=INK2, ls=":", lw=1)
    a.set_xlabel("iteration", fontsize=9); a.set_ylim(0.4, 1.15)
    a.legend(fontsize=8, loc="upper right")
    mass = float(d["mass"]) if "mass" in d.files else np.nan
    a.set_title(f"(c) stiffer ($c/c_0$={float(d['c_ratio']):.2f}) and lighter "
                f"({mass:.2f}$\\times$)", fontsize=10, loc="left", color=INK)
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

    mass = float(d["mass"]) if "mass" in d.files else np.nan
    fig.suptitle("Co-designed topology + toolpath on a conforming CFRP mesh: a light "
                 "resin channel guides the wave between two through-holes to an "
                 f"interior pocket (confinement {C_str:.2f}$\\to${C_opt:.2f}); the "
                 f"channel + holes cut mass to {mass:.2f}$\\times$ solid at "
                 f"{float(d['c_ratio']):.2f}$\\times$ compliance",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
