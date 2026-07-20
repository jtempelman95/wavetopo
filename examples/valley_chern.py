"""
Compute the VALLEY CHERN number of the fiber-mass honeycomb: the Berry
curvature of the lower Dirac band, integrated over each half-BZ (one valley
each).  A genuine valley-Hall phase has C_valley = +/-1/2 with opposite sign at
K and K', flipping when the fiber mass (A/B tilt) reverses.

Outputs results/valley_berry.png: Berry-curvature maps for +phi and -phi.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.bloch import berry_curvature
from examples.valley_mass import build
from examples.valley_viz import sublattice_support_angles


def valley_split(kxs, kys, F, ksplit):
    """Sum Berry flux over the two half-BZs split at ky=ksplit -> (C_K, C_Kp)."""
    # F[i,j] lives on plaquette with lower-left (kxs[i], kys[j])
    lower = 0.0; upper = 0.0
    for j, ky in enumerate(kys):
        if ky < ksplit:
            lower += F[:, j].sum()
        else:
            upper += F[:, j].sum()
    return lower / (2 * np.pi), upper / (2 * np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=18)
    ap.add_argument("--radius", type=float, default=0.33)
    ap.add_argument("--phi", type=float, default=30.0)
    ap.add_argument("--band", type=int, default=3, help="lower Dirac band index")
    ap.add_argument("--nk", type=int, default=24)
    args = ap.parse_args()

    mesh, bp, rbf, z, supp, period = build(args.nx, args.radius)
    Lx, Ly = period
    ksplit = np.pi / Ly      # K at ky<this, K' at ky>this

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    results = {}
    for col, sgn in enumerate([+1, -1]):
        th = sublattice_support_angles(*supp, period,
                                       np.deg2rad(sgn * args.phi))
        bp.assemble(z, th)
        kxs, kys, F, chern = berry_curvature(bp, args.band, nk=args.nk)
        cK, cKp = valley_split(kxs, kys, F, ksplit)
        results[sgn] = (cK, cKp, chern)
        print(f"phi={sgn*args.phi:+.0f}:  C_K={cK:+.3f}  C_K'={cKp:+.3f}  "
              f"total Chern={chern:+.3f}")
        # plot Berry curvature (flux per plaquette)
        im = ax[col].imshow(F.T, origin="lower", cmap="RdBu_r",
                            extent=[0, kxs[-1], 0, kys[-1]], aspect="auto",
                            vmin=-np.abs(F).max(), vmax=np.abs(F).max())
        ax[col].axhline(ksplit, color="k", lw=0.7, ls="--")
        ax[col].set_title(f"φ={sgn*args.phi:+.0f}°   "
                          f"C$_K$={cK:+.2f},  C$_{{K'}}$={cKp:+.2f}")
        ax[col].set_xlabel("kx"); ax[col].set_ylabel("ky")
        plt.colorbar(im, ax=ax[col], fraction=0.046, label="Berry curvature")
    fig.suptitle("Valley Chern from fiber mass: Berry curvature peaks at the "
                 "valleys, flips sign with φ", y=1.0)
    plt.tight_layout()
    fig.savefig("results/valley_berry.png", dpi=140, bbox_inches="tight")
    print("saved results/valley_berry.png")
    cK_p = results[+1][0]; cK_m = results[-1][0]
    print(f"\nvalley Chern flips: C_K(+φ)={cK_p:+.2f}  C_K(-φ)={cK_m:+.2f}  "
          f"(ideal ±0.5)")


if __name__ == "__main__":
    main()
