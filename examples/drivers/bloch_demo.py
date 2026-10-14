"""
Demonstrate that fiber orientation makes a phononic band gap DIRECTIONAL.

Builds two-phase unit cells (stiff anisotropic fiber composite + soft polymer
matrix) and compares the dispersion for waves travelling along x (Gamma-X) vs
along y (Gamma-Y).  A lamellar cell with horizontal fibers opens a gap for
vertical propagation that is absent (or much smaller) for horizontal
propagation -- the signature of a directional gap.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.cfrp import Material, QuadMesh, grid_support_points, CSRBFMapping
from wavetopo.bloch import (BlochProblem, ibz_path_square, directional_path,
                           gap_between_bands)


def design_layers(mesh, frac=0.5):
    """Horizontal stiff layer (lamellar) occupying middle `frac` of the cell."""
    z = np.zeros(mesh.N)
    yc = mesh.cy
    z[np.abs(yc - mesh.Ly / 2) < frac * mesh.Ly / 2] = 1.0
    return z


def design_block(mesh, frac=0.6):
    """Centered stiff square inclusion."""
    z = np.zeros(mesh.N)
    inside = ((np.abs(mesh.cx - mesh.Lx / 2) < frac * mesh.Lx / 2) &
              (np.abs(mesh.cy - mesh.Ly / 2) < frac * mesh.Ly / 2))
    z[inside] = 1.0
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--design", choices=["layers", "block"], default="layers")
    ap.add_argument("--angle", type=float, default=0.0, help="fiber angle deg")
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--out", default="results/bloch_demo.png")
    args = ap.parse_args()

    a = 1.0
    mesh = QuadMesh(args.n, args.n, a, a)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    sx, sy = grid_support_points(mesh, a / 4, a / 4)
    rbf = CSRBFMapping(mesh, (sx, sy), r_s=a / 2)
    bp = BlochProblem(mesh, mat, rbf, penal=3.0)

    z = design_layers(mesh) if args.design == "layers" else design_block(mesh)
    th = np.full(rbf.M, np.deg2rad(args.angle))
    bp.assemble(z, th)

    # full IBZ path
    path, ticks, labels = ibz_path_square(a, a, n=30)
    bands = bp.band_structure(path, n_bands=args.bands)

    # directional segments Gamma->X (along x) and Gamma->Y (along y)
    pX = directional_path(a, a, theta_dir=0.0, n=40)
    pY = directional_path(a, a, theta_dir=np.pi / 2, n=40)
    bX = bp.band_structure(pX, n_bands=args.bands)
    bY = bp.band_structure(pY, n_bands=args.bands)

    # report gap between each consecutive band pair, per direction
    print(f"design={args.design} fiber angle={args.angle} deg")
    print(f"{'pair':>6} {'gapX (along fiber x)':>22} {'gapY (transverse)':>20}")
    for nlow in range(args.bands - 1):
        gX, _, _ = gap_between_bands(bX, nlow)
        gY, _, _ = gap_between_bands(bY, nlow)
        flag = "  <-- directional" if (max(gX, gY) > 0 and
                                       min(gX, gY) < 0.3 * max(gX, gY)) else ""
        print(f"{nlow}-{nlow+1:>2} {gX:>22.3f} {gY:>20.3f}{flag}")

    # ---- plot ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    xx = np.arange(len(path))
    ax[0].plot(xx, bands, "k-", lw=1)
    ax[0].set_xticks(ticks); ax[0].set_xticklabels(labels)
    for t in ticks:
        ax[0].axvline(t, color="0.8", lw=0.7)
    ax[0].set_title(f"full IBZ  ({args.design}, fibers {args.angle} deg)")
    ax[0].set_ylabel(r"$\omega$")

    for a_, b_, ttl in [(ax[1], bX, r"$\Gamma\to$X  (along fiber, x)"),
                        (ax[2], bY, r"$\Gamma\to$Y  (transverse, y)")]:
        kk = np.linspace(0, 1, b_.shape[0])
        a_.plot(kk, b_, "b-", lw=1.2)
        a_.set_title(ttl); a_.set_xlabel(r"$|k|/(\pi/a)$")
        # shade largest gap
        best = max(range(args.bands - 1),
                   key=lambda n: gap_between_bands(b_, n)[0])
        g, lo, hi = gap_between_bands(b_, best)
        if g > 0:
            a_.axhspan(lo, hi, color="orange", alpha=0.3)
            a_.text(0.05, (lo + hi) / 2, f"gap={g:.2f}", fontsize=9)
    fig.suptitle("Directional phononic gap via fiber anisotropy", y=1.02)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
