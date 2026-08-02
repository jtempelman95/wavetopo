"""Figure for examples/cfrp_wave_shield.py: a broadband, multi-SOURCE, multi-target
fiber-orientation wave shield on a solid CFRP plate with TRUE holes.  Vibration
enters at both bottom corners; both top corners are kept quiet across a frequency
band.  Because the holes are real (no cells inside), the wave reflects off their rims
and there is zero wave energy in any void -- every bright region is solid material.
The plotted field at each frequency is the COMBINED energy over the source load
cases, sqrt(sum_s |u_s|^2)."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from examples.cfrp_wave_shield import setup, SOURCES, PROTECTS, HOLES, LX, LY
from wavetopo.cfrp_wave import curl_operators, director_curl
from wavetopo.dolfinx_viz import plot_toolpaths_phase
from _paperfig import set_style, internal_ticks

set_style()
INK, INK2 = "#111111", "#6b6b6b"


def node_energy(u):
    return np.abs(u[0::2]) ** 2 + np.abs(u[1::2]) ** 2


def protected_frac(mesh, e):
    """Share of interior (combined) energy in the union of the protected disks."""
    xs, ys = mesh.npx[:, 0], mesh.npx[:, 1]
    inter = (xs > 0.3) & (xs < LX - 0.3) & (ys > 0.3) & (ys < LY - 0.3)
    inpro = np.zeros(xs.shape[0], bool)
    for px, py, pr in PROTECTS:
        inpro |= np.hypot(xs - px, ys - py) < pr
    tot = e[inter].sum()
    return float(e[inter & inpro].sum() / tot) if tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/cfrp_wave_shield.npz")
    ap.add_argument("--h", type=float, default=0.07)
    ap.add_argument("--out", default="results/cfrp_wave_shield.pdf")
    args = ap.parse_args()

    d = np.load(args.data)
    omegas = [float(o) for o in d["omegas"]]; eta = float(d["eta"])
    mesh, mat, plates, drives, wpro, interior = setup(args.h, omegas, eta)
    tri = mtri.Triangulation(mesh.npx[:, 0], mesh.npx[:, 1], mesh.tris)
    cents = np.column_stack([mesh.cx, mesh.cy])
    th = d["th"]; z = np.ones(mesh.tris.shape[0])

    es, eo, Fs, Fo = [], [], [], []
    for P in plates:
        uss = P.dynamic_multi(z, np.zeros(mesh.tris.shape[0]), drives, False)[1]
        uso = P.dynamic_multi(z, th, drives, False)[1]
        e_s = sum(node_energy(u) for u in uss)     # combined energy over sources
        e_o = sum(node_energy(u) for u in uso)
        es.append(e_s); eo.append(e_o)
        Fs.append(protected_frac(mesh, e_s)); Fo.append(protected_frac(mesh, e_o))
    print("omega:", omegas)
    print("straight  protected fraction:", [f"{x:.3f}" for x in Fs])
    print("optimized protected fraction:", [f"{x:.3f}" for x in Fo])

    fig, ax = plt.subplots(2, 3, figsize=(8.4, 2.92))

    def holes_rings(a, fill="white"):
        for hx, hy, hr in HOLES:
            a.add_patch(plt.Circle((hx, hy), hr, fc=fill, ec="k", lw=1.0, zorder=6))

    def marks(a, src=True):
        if src:
            for sx, sy, sr in SOURCES:
                a.add_patch(plt.Circle((sx, sy), sr, ec="#f2c14e", fc="none", lw=1.9,
                                       zorder=7))
        for px, py, pr in PROTECTS:
            a.add_patch(plt.Circle((px, py), pr, ec="#1baf7a", fc="none", lw=2.0,
                                   ls="--", zorder=7))
        a.set_xlim(0, LX); a.set_ylim(0, LY); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#cccccc")

    # (a) fiber toolpaths on the solid, holes cut out
    a = ax[0, 0]
    plot_toolpaths_phase(a, cents, th, (0, LX), (0, LY), holes=HOLES,
                         spacing=0.16, lw=0.6)
    holes_rings(a, fill="none"); marks(a)
    a.set_title("(a) fiber tows", loc="left", color=INK)

    # (b) fiber curvature
    a = ax[0, 1]
    Px, Py = curl_operators(mesh)
    zeta = director_curl(th, Px, Py)
    zref = 1.0 / min(hr for _, _, hr in HOLES)
    tc = a.tripcolor(tri, facecolors=np.minimum(zeta, 2 * zref), cmap="viridis",
                     vmin=0, vmax=2 * zref)
    cb = fig.colorbar(tc, ax=a, fraction=0.03, pad=0.02)
    cb.set_label("$|\\zeta|$", fontsize=8); cb.ax.tick_params(labelsize=7)
    over = mesh.area[zeta > zref].sum() / mesh.area.sum()
    holes_rings(a); marks(a, src=False)
    a.set_title(r"(b) fiber curvature $|\zeta|$", loc="left", color=INK)

    # (c) broadband: protected fraction at each frequency, straight vs optimized
    a = ax[0, 2]
    x = np.arange(len(omegas)); w = 0.38
    a.bar(x - w / 2, Fs, w, color="#c9b7a0", label="straight fibers")
    a.bar(x + w / 2, Fo, w, color="#1baf7a", label="optimized")
    a.set_yscale("log"); a.set_xticks(x)
    a.set_xticklabels([f"{o:.0f}" for o in omegas])
    a.set_xlabel("frequency $\\omega$", fontsize=9)
    a.set_ylabel("protected fraction", fontsize=9)
    a.legend(fontsize=8, loc="upper right")
    a.set_title("(c) broadband shield", loc="left", color=INK)
    a.grid(alpha=0.3, axis="y", which="both")
    internal_ticks(a, minor=False)

    # (d-f) combined optimized energy field at each frequency
    vmax = np.percentile(np.concatenate([np.sqrt(e) for e in es]), 99.3)
    labels = "def"
    for i, (om, e) in enumerate(zip(omegas, eo)):
        a = ax[1, i]
        a.tripcolor(tri, np.sqrt(e), cmap="magma", vmax=vmax, shading="gouraud")
        holes_rings(a); marks(a)
        a.set_title(f"({labels[i]}) $\\omega={om:.0f}$", loc="left", color=INK)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    import os
    _pv = os.environ.get("FIGPREVIEW")
    if _pv:
        fig.savefig(_pv, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
