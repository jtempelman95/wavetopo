"""
Figure for examples/multifunc_finite.py: a co-designed cell, and the finite
truncated panel built from it actually stopping a wave.

Top row     the designed cell -- density, fiber orientation, Bloch bands with
            the complete gap shaded.
Bottom row  the finite panel -- response driven inside the gap (attenuated) and
            outside it (transmitted), and the measured transmission spectrum
            against the gap predicted from the infinite-crystal analysis.

    PYTHONPATH=. python examples/multifunc_finite_figure.py
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.cfrp import Material, QuadMesh
from wavetopo.harmonic import HarmonicValley
from examples.multifunc_finite import MAT, tile_panel, transmission_sweep

C_CO, C_GE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.set_axisbelow(True)


def field_at(d, n, Nx, Ny, nbuf, om, eta):
    mesh, z, th = tile_panel(d['co_y'], d['co_theta_elem'], n, Nx, Ny, nbuf)
    hv = HarmonicValley(mesh, Material(**MAT), omega=om, eta=eta)
    hv.set_design(z, th)
    marg = 0.55 * nbuf / n
    dd = np.maximum((marg - mesh.cx) / marg, (mesh.cx - (mesh.Lx - marg)) / marg)
    hv.set_sponge(np.maximum(0.0, dd) ** 2 * 30.0)
    xs, _ = mesh.node_xy()
    src = np.flatnonzero(np.abs(xs - 0.35 * nbuf / n) < 0.6 * mesh.dx)
    F = np.zeros(mesh.ndof, complex); F[2 * src] = 1.0
    hv.set_load(F)
    u = hv.solve()
    mag = np.sqrt(np.abs(u[0::2]) ** 2 + np.abs(u[1::2]) ** 2)
    return mesh, mag.reshape(mesh.nely + 1, mesh.nelx + 1), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/multifunc_finite.npz")
    ap.add_argument("--design", default="results/multifunc_finite_design.npz")
    ap.add_argument("--Nx", type=int, default=6)
    ap.add_argument("--Ny", type=int, default=4)
    ap.add_argument("--nbuf", type=int, default=28)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--out", default="results/multifunc_finite.png")
    args = ap.parse_args()

    r = np.load(args.res, allow_pickle=True)
    d = dict(np.load(args.design, allow_pickle=True))
    n = int(d['co_n'])
    lo, hi = float(r['gap_lo']), float(r['gap_hi'])
    w, Tco, Tge = r['omega'], r['co_T'], r['ge_T']

    # representative in-gap and pass-band drive frequencies
    w_gap = 0.5 * (lo + hi)
    out_mask = (w < lo) | (w > hi)
    w_pass = float(w[out_mask][np.argmax(Tco[out_mask])])

    fig = plt.figure(figsize=(15.0, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.32,
                          wspace=0.26)

    # ---- (a) cell density ------------------------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.asarray(d['co_y']).reshape(n, n), origin="lower",
              cmap="bone_r", vmin=0, vmax=1)
    ax.set_title(f"(a)  co-designed cell   $C_{{\\rm bulk}}$="
                 f"{float(d['co_C_bulk']):.2f}", fontsize=10.5, loc="left",
                 color=INK)
    ax.set_xticks([]); ax.set_yticks([])

    # ---- (b) fiber orientation -------------------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    th = np.asarray(d['co_theta_elem']).reshape(n, n)
    ax.imshow(np.rad2deg(th), origin="lower", cmap="twilight",
              vmin=-90, vmax=90)
    s = max(n // 14, 1)
    yy, xx = np.mgrid[0:n:s, 0:n:s]
    ax.quiver(xx, yy, np.cos(th[::s, ::s]), np.sin(th[::s, ::s]),
              color="k", pivot="mid", headwidth=0, headlength=0,
              headaxislength=0, scale=26, width=0.006)
    ax.set_title("(b)  fiber orientation $\\theta$", fontsize=10.5,
                 loc="left", color=INK)
    ax.set_xticks([]); ax.set_yticks([])

    # ---- (c) bands --------------------------------------------------- #
    ax = fig.add_subplot(gs[0, 2])
    B = np.asarray(d['co_bands'])
    ax.axhspan(lo, hi, color="#1baf7a", alpha=0.18, lw=0)
    for j in range(B.shape[1]):
        ax.plot(B[:, j], color=C_CO, lw=1.5)
    ax.set_xlim(0, B.shape[0] - 1)
    ax.set_ylabel(r"$\omega$", color=INK, fontsize=10)
    ax.set_title(f"(c)  Bloch bands, complete gap {hi-lo:+.3f}",
                 fontsize=10.5, loc="left", color=INK)
    try:
        ax.set_xticks(list(np.asarray(d['co_ticks'])))
        ax.set_xticklabels([str(s) for s in np.asarray(d['co_labels'])])
    except Exception:
        ax.set_xticks([])
    tidy(ax)

    # ---- (d,e) finite-panel fields ----------------------------------- #
    for col, (om, lab) in enumerate([(w_gap, "inside the gap"),
                                     (w_pass, "outside the gap")]):
        mesh, mag, zp = field_at(d, n, args.Nx, args.Ny, args.nbuf, om,
                                 args.eta)
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(mag, origin="lower", cmap="magma",
                  vmax=np.percentile(mag, 99.5),
                  extent=[0, mesh.Lx, 0, mesh.Ly], aspect="equal")
        x0 = args.nbuf / n
        for xv in (x0, x0 + args.Nx):
            ax.axvline(xv, color="c", ls="--", lw=1.0, alpha=0.75)
        ax.set_title(f"({'de'[col]})  panel driven {lab}:  "
                     f"$\\omega$={om:.3f}", fontsize=10.5, loc="left",
                     color=INK)
        ax.set_xticks([]); ax.set_yticks([])

    # ---- (f) transmission spectrum ----------------------------------- #
    ax = fig.add_subplot(gs[1, 2])
    ax.axvspan(lo, hi, color="#1baf7a", alpha=0.18, lw=0)
    ax.semilogy(w, np.maximum(Tco, 1e-14), color=C_CO, lw=2.2, marker="o",
                ms=3.4, label="co-designed ($z$ and $\\theta$)")
    ax.semilogy(w, np.maximum(Tge, 1e-14), color=C_GE, lw=1.8, ls="--",
                marker="s", ms=3.2, label="geometry only ($z$)")
    ax.set_xlabel(r"frequency  $\omega$", color=INK, fontsize=10)
    ax.set_ylabel(r"transmitted energy  $T$", color=INK, fontsize=10)
    ax.set_title("(f)  finite-panel transmission vs predicted gap",
                 fontsize=10.5, loc="left", color=INK)
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="lower left")
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    tidy(ax)

    ing = (w > lo) & (w < hi)
    att = Tco[~ing].mean() / max(Tco[ing].mean(), 1e-300)
    fig.suptitle("Co-designed stiff + band-gap metamaterial, verified on a "
                 f"finite {args.Nx}$\\times${args.Ny}-cell truncated panel:  "
                 f"static $k_{{\\rm eff}}$={float(r['co_k_eff']):.3g},  "
                 f"in-gap attenuation {att:.0f}$\\times$",
                 fontsize=12, y=0.985, color=INK)
    fig.savefig(args.out, dpi=190, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
