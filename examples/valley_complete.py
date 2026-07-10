"""
Co-optimize honeycomb geometry + fiber for a COMPLETE bulk gap (bands 3-4 over
the whole BZ) -- required for confined valley-Hall edge states.  Starts from the
honeycomb + sublattice fiber mass.  Saves the cell for the edge-transport demo.
"""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.valley_mass import build, line_bands
from examples.valley_viz import sublattice_support_angles
from topoopt.valley_opt import ValleyOptimizer
from topoopt.bloch import k_segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--iters", type=int, default=70)
    ap.add_argument("--phi0", type=float, default=25.0)
    ap.add_argument("--out", default="results/valley_complete.png")
    args = ap.parse_args()

    mesh, bp, rbf, z0, supp, period = build(args.nx, 0.33)
    Lx, Ly = period
    opt = ValleyOptimizer(bp, R=0.12, period=period, vtarget=float(z0.mean()),
                          w_vol=60.0)
    # BZ k-sample (mirror symmetry in kx -> sample kx>=0 half)
    kxs = np.linspace(0, np.pi / Lx, 3)
    kys = np.linspace(0.08, 2 * np.pi / Ly - 0.08, 6)
    ksample = [(kx, ky) for kx in kxs for ky in kys]
    opt.set_ksample(ksample, p3=3, rho_ks=25.0)

    th0 = sublattice_support_angles(*supp, period, np.deg2rad(args.phi0))

    # quick FD check of the complete-gap gradient
    J, dz, dth, info = opt.objective_complete(z0, th0, True)
    e = 300; zp = z0.copy(); zp[e] += 1e-6; zm = z0.copy(); zm[e] -= 1e-6
    fd = (opt.objective_complete(zp, th0, False)[0]
          - opt.objective_complete(zm, th0, False)[0]) / 2e-6
    print(f"FD check dJ/dz[{e}]: an={dz[e]:.3e} fd={fd:.3e}")
    print(f"initial complete gap={info['gap']:+.4f}")

    t0 = time.time()
    res = opt.run_complete(z0, th0, max_iter=args.iters, move=0.08)
    print(f"optimized in {time.time()-t0:.0f}s  best complete gap={res['gap']:+.4f}")
    z, th = res['z'], res['th']
    np.savez("results/valley_complete.npz", z=z, th=th,
             nelx=mesh.nelx, nely=mesh.nely, Lx=Lx, Ly=Ly,
             omega=res['omega'], sigma=res['omega']**2)

    # band structure Gamma-X-M-Y-Gamma showing the complete gap
    y = opt.P @ z; bp.assemble(y, th)
    G = np.array([0, 0]); X = np.array([np.pi / Lx, 0])
    Mp = np.array([np.pi / Lx, np.pi / Ly]); Y = np.array([0, np.pi / Ly])
    path = (k_segment(G, X, 25) + k_segment(X, Mp, 25)[1:]
            + k_segment(Mp, Y, 25)[1:] + k_segment(Y, G, 25)[1:])
    bands = bp.band_structure(path, 7)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(bands, "k-", lw=1)
    ax[0].axhspan(bands[:, 3].max(), bands[:, 4].min(), color="orange",
                  alpha=0.35)
    ax[0].set_xticks([0, 24, 48, 72, 96])
    ax[0].set_xticklabels([r"$\Gamma$", "X", "M", "Y", r"$\Gamma$"])
    ax[0].set_ylabel(r"$\omega$")
    ax[0].set_title(f"complete bulk gap = {res['gap']:.3f}  (ω≈{res['omega']:.2f})")
    g = [h['gap'] for h in res['hist']]
    ax[1].plot(g, "darkred"); ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("complete bulk gap")
    ax[1].set_title("convergence"); ax[1].grid(alpha=0.3)
    fig.suptitle("Optimizing a COMPLETE valley-Hall gap (geometry + fiber)", y=1.0)
    plt.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
