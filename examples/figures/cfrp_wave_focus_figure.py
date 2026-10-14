"""Figure for examples/cfrp_wave_focus.py: a mass-reduced, topology-optimized CFRP
cantilever whose fiber toolpath focuses an injected wave at the loaded tip.  The
wave lives IN the fiber structure, so the topology (top) and the field (bottom) are
one physical object -- the structure outline is overlaid on the field panels."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from examples.drivers.cfrp_wave_focus import setup, FOCUS, HOLES, LX, LY
from wavetopo.cfrp_wave import curl_operators, director_curl
from wavetopo.dolfinx_viz import plot_toolpaths_phase

INK, INK2 = "#0b0b0b", "#52514e"


def focus_frac(mesh, u):
    """Baseline-independent focus metric: energy in the focus disk / interior."""
    e = np.abs(u[0::2]) ** 2 + np.abs(u[1::2]) ** 2
    xs, ys = mesh.npx[:, 0], mesh.npx[:, 1]
    inter = (xs > 0.3) & (xs < LX - 0.15) & (ys > 0.3) & (ys < LY - 0.3)
    fx, fy, fr = FOCUS
    infoc = np.hypot(xs - fx, ys - fy) < fr
    tot = e[inter].sum()
    return float(e[inter & infoc].sum() / tot) if tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/cfrp_wave_focus.npz")
    ap.add_argument("--h", type=float, default=0.06)
    ap.add_argument("--out", default="results/cfrp_wave_focus.png")
    args = ap.parse_known_args()[0]

    d = np.load(args.data)
    omega, eta = float(d["omega"]), float(d["eta"])
    mesh, mat, P, wfoc, interior = setup(args.h, omega, eta)
    tri = mtri.Triangulation(mesh.npx[:, 0], mesh.npx[:, 1], mesh.tris)
    cents = np.column_stack([mesh.cx, mesh.cy])
    y, th = d["y"], d["th"]

    u_str = P.dynamic(np.full(mesh.tris.shape[0], float(d["vol"])),
                      np.zeros(mesh.tris.shape[0]), False)[1]
    u_opt = P.dynamic(y, th, False)[1]
    Fs, Fo = focus_frac(mesh, u_str), focus_frac(mesh, u_opt)
    print(f"focus fraction: straight {Fs:.3f} -> optimized {Fo:.3f}")

    def field(u):
        return np.sqrt(np.abs(u[0::2]) ** 2 + np.abs(u[1::2]) ** 2)

    fig, ax = plt.subplots(2, 3, figsize=(17, 6.6))
    fx, fy, fr = FOCUS

    def holes_and_focus(a, focus=True):
        for cx, cy, r in HOLES:
            a.add_patch(plt.Circle((cx, cy), r, fc="white", ec="k", lw=1.1, zorder=5))
        if focus:
            a.add_patch(plt.Circle((fx, fy), fr, ec="#1baf7a", fc="none", lw=1.8,
                                   ls="--", zorder=6))
        a.set_xlim(0, LX); a.set_ylim(0, LY); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])

    # (a) fiber toolpaths (reference wave projection, tows only in the fiber)
    a = ax[0, 0]
    plot_toolpaths_phase(a, cents, th, (0, LX), (0, LY), holes=HOLES,
                         spacing=0.14, dens=y, dens_thresh=0.5, lw=0.6)
    a.add_patch(plt.Circle((fx, fy), fr, ec="#1baf7a", fc="none", lw=1.8, ls="--",
                           zorder=6))
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"(a) fiber toolpaths (black tows / matrix / hole), "
                f"mass {float(d['mass']):.2f}$\\times$ solid",
                fontsize=10, loc="left", color=INK)

    # (b) fiber curvature |zeta| on the structure (printability)
    a = ax[0, 1]
    Px, Py = curl_operators(mesh)
    zeta = director_curl(th, Px, Py)
    zall = 1.0 / min(r for _, _, r in HOLES)
    zshow = np.where(y > 0.5, np.minimum(zeta, 2 * zall), np.nan)
    tc = a.tripcolor(tri, facecolors=zshow, cmap="viridis", vmin=0, vmax=2 * zall)
    cb = fig.colorbar(tc, ax=a, fraction=0.03, pad=0.02)
    cb.set_label("$|\\zeta|$", fontsize=8); cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(zall, color="r", lw=1.2)
    over = (mesh.area[y > 0.5][zeta[y > 0.5] > zall].sum() / mesh.area[y > 0.5].sum())
    holes_and_focus(a, focus=False)
    a.set_title(f"(b) fiber curvature $|\\zeta|$ (max {zeta[y>0.5].max():.1f}, "
                f"{100*over:.0f}% over $1/R$={zall:.1f})",
                fontsize=9, loc="left", color=INK)

    # (c) convergence: compliance and mass
    a = ax[0, 2]
    a.plot(d["hist_c"], color="#2a78d6", lw=2, label="compliance $c/c_0$")
    a.plot(d["hist_mass"], color="#d1662a", lw=2, label="mass / solid")
    a.axhline(1.0, color=INK2, ls=":", lw=1)
    a.set_xlabel("iteration", fontsize=9); a.set_ylim(0.3, 1.45)
    a.legend(fontsize=8, loc="upper right")
    a.set_title(f"(c) stiffer ($c/c_0$={float(d['c_ratio']):.2f}) and lighter "
                f"({float(d['mass']):.2f}$\\times$)", fontsize=10, loc="left", color=INK)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(alpha=0.3)

    # structure outline (fiber boundary) to overlay on the field panels: element
    # density -> nodal (area-weighted) so tricontour can draw the y=0.5 boundary.
    ms = field(u_str); mo = field(u_opt); vmax = np.percentile(mo, 99.3)
    nn = mesh.npx.shape[0]
    ynode = np.zeros(nn); wnode = np.zeros(nn)
    for e, trian in enumerate(mesh.tris):
        ynode[trian] += mesh.area[e] * y[e]; wnode[trian] += mesh.area[e]
    ynode /= np.maximum(wnode, 1e-12)

    def overlay_structure(a):
        a.tricontour(tri, ynode, levels=[0.5], colors="w", linewidths=0.7, alpha=0.7)

    # (d) straight-fiber field
    a = ax[1, 0]
    a.tripcolor(tri, ms, cmap="magma", vmax=vmax, shading="gouraud")
    overlay_structure(a); holes_and_focus(a)
    a.set_title(f"(d) straight fibers: focus fraction {Fs:.2f}",
                fontsize=10, loc="left", color=INK)

    # (e) optimized field: energy focused at the tip, inside the structure
    a = ax[1, 1]
    a.tripcolor(tri, mo, cmap="magma", vmax=vmax, shading="gouraud")
    overlay_structure(a); holes_and_focus(a)
    a.set_title(f"(e) co-design: wave focuses at the loaded tip, "
                f"focus fraction {Fo:.2f}", fontsize=10, loc="left", color=INK)

    # (f) density (topology) as fiber/matrix, with the load/BC annotation
    a = ax[1, 2]
    from matplotlib.colors import LinearSegmentedColormap
    cmap2 = LinearSegmentedColormap.from_list("fm", ["#e8e2d0", "#12263a"])
    a.tripcolor(tri, facecolors=y, cmap=cmap2, vmin=0, vmax=1)
    a.plot([0, 0], [0.1, LY - 0.1], color="#c02020", lw=4, zorder=6)   # clamp
    a.annotate("", xy=(LX - 0.05, 1.0), xytext=(LX - 0.05, 1.6),
               arrowprops=dict(arrowstyle="-|>", color="#c02020", lw=2), zorder=7)
    holes_and_focus(a)
    a.set_title("(f) topology $z$ (clamp | tip load); fiber dark / matrix light",
                fontsize=9, loc="left", color=INK)

    fig.suptitle("Mass-reduced topology + toolpath co-design on a conforming CFRP "
                 f"cantilever: the fiber layout focuses the wave at the loaded tip "
                 f"(focus fraction {Fs:.2f}$\\to${Fo:.2f}) while cutting mass to "
                 f"{float(d['mass']):.2f}$\\times$ at {float(d['c_ratio']):.2f}"
                 f"$\\times$ compliance", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
