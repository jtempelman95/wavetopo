"""
Co-optimize honeycomb geometry (z) + fiber orientation (theta) for a WIDE
COMPLETE bulk gap of the SCALAR (antiplane-shear) honeycomb -- the clean
setting for valley-Hall edge transport.  Reuses ValleyOptimizer with ScalarBloch.

Validates the gap on a fine grid (no coarse-sample self-deception) and saves the
cell for the edge-transport demo.
"""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topoopt.cfrp import QuadMesh, CSRBFMapping
from topoopt.scalar import MaterialSH, ScalarBloch
from topoopt.valley_opt import ValleyOptimizer
from topoopt.bloch import k_segment
from examples.valley_hall import A_FRAC, B_FRAC
from examples.valley_viz import sublattice_support_angles


def honey(mesh, rA, rB, period):
    Lx, Ly = period
    z = np.zeros(mesh.N)
    for fr, r in [(A_FRAC, rA), (B_FRAC, rB)]:
        for fx, fy in fr:
            cx, cy = fx * Lx, fy * Ly
            dx = mesh.cx - cx; dy = mesh.cy - cy
            dx -= Lx * np.round(dx / Lx); dy -= Ly * np.round(dy / Ly)
            z[np.hypot(dx, dy) < r] = 1.0
    return z


def true_complete_gap(sb, y, th, p, nk=29):
    sb.assemble(y, th)
    Lx, Ly = sb.mesh.Lx, sb.mesh.Ly
    kxs = np.linspace(0, 2 * np.pi / Lx, nk); kys = np.linspace(0, 2 * np.pi / Ly, nk)
    lo, hi = np.inf, -np.inf
    for kx in kxs:
        for ky in kys:
            w = sb.bands_at_k(kx, ky, p + 3)
            lo = min(lo, w[p + 1]); hi = max(hi, w[p])
    return lo - hi, hi, lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--out", default="results/scalar_complete.png")
    args = ap.parse_args()

    a = 1.0
    Lx, Ly = np.sqrt(3) * a, 3 * a
    ny = int(round(args.nx * Ly / Lx))
    mesh = QuadMesh(args.nx, ny, Lx, Ly)
    mat = MaterialSH(mu_L=6.0, mu_T=2.0, mu_m=1.0, rho_f=1.6, rho_m=1.2)
    sx = np.linspace(0, Lx, 6, endpoint=False); sy = np.linspace(0, Ly, 10, endpoint=False)
    SX, SY = np.meshgrid(sx, sy)
    rbf = CSRBFMapping(mesh, (SX.ravel(), SY.ravel()), r_s=0.55, period=(Lx, Ly))
    sb = ScalarBloch(mesh, mat, rbf, penal=3.0)

    z0 = honey(mesh, 0.40, 0.24, (Lx, Ly))           # seed A/B asymmetry
    th0 = sublattice_support_angles(SX.ravel(), SY.ravel(), (Lx, Ly),
                                    np.deg2rad(30))

    opt = ValleyOptimizer(sb, R=0.11, period=(Lx, Ly), vtarget=float(z0.mean()),
                          w_vol=80.0)
    kxs = np.linspace(0, np.pi / Lx, 4)
    kys = np.linspace(0.05, 2 * np.pi / Ly - 0.05, 9)
    opt.set_ksample([(kx, ky) for kx in kxs for ky in kys], p3=1, rho_ks=30.0)

    g0, _, _ = true_complete_gap(sb, opt.P @ z0, th0, 1)
    print(f"mesh {mesh.nelx}x{mesh.nely}, ndof_red={sb.n_red}, "
          f"initial TRUE complete gap={g0:+.4f}")

    t0 = time.time()
    res = opt.run_complete(z0, th0, max_iter=args.iters, move=0.07)
    z, th = res['z'], res['th']
    gt, hi, lo = true_complete_gap(sb, opt.P @ z, th, 1, nk=33)
    omega = 0.5 * (hi + lo)
    print(f"optimized in {time.time()-t0:.0f}s  "
          f"sampled gap={res['gap']:+.3f}  TRUE complete gap={gt:+.4f} "
          f"window=[{hi:.3f},{lo:.3f}] omega={omega:.3f}")
    np.savez("results/scalar_complete.npz", z=z, th=th, nelx=mesh.nelx,
             nely=mesh.nely, Lx=Lx, Ly=Ly, omega=omega, sigma=omega**2, p=1)

    # band structure
    y = opt.P @ z; sb.assemble(y, th)
    G = np.array([0, 0]); X = np.array([np.pi / Lx, 0])
    Mp = np.array([np.pi / Lx, np.pi / Ly]); Y = np.array([0, np.pi / Ly])
    path = (k_segment(G, X, 25) + k_segment(X, Mp, 25)[1:]
            + k_segment(Mp, Y, 25)[1:] + k_segment(Y, G, 25)[1:])
    bands = sb.band_structure(path, 5)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(bands, "k-", lw=1)
    if gt > 0:
        ax[0].axhspan(hi, lo, color="orange", alpha=0.35)
    ax[0].set_xticks([0, 24, 48, 72, 96])
    ax[0].set_xticklabels([r"$\Gamma$", "X", "M", "Y", r"$\Gamma$"])
    ax[0].set_title(f"scalar bands: complete gap={gt:.3f} (ω≈{omega:.2f})")
    ax[0].set_ylabel(r"$\omega$")
    ax[1].plot([h['gap'] for h in res['hist']], "darkred")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].grid(alpha=0.3)
    ax[1].set_xlabel("iteration"); ax[1].set_title("convergence (sampled gap)")
    fig.suptitle("Scalar honeycomb: co-optimizing geometry + fiber for a "
                 "complete valley gap", y=1.0)
    plt.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
