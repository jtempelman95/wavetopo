"""
Co-optimize honeycomb GEOMETRY (z) + FIBER orientation (theta) to maximize an
ISOLATED valley-Hall gap (robustness).  Visualizes the result: band structure
(before/after), the co-designed atoms+toolpaths, and convergence.
"""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.valley_mass import build, line_bands
from examples.valley_viz import sublattice_support_angles
from wavetopo.valley_opt import ValleyOptimizer, _straddle
from wavetopo.cfrp_viz import orientation_streamlines


def bands_through_K(bp, y, th, kyD, sigma, nev, sp=0.45, n=31):
    ks, by = line_bands(bp, y, th, (0, kyD), (0, 1), sp, n, nev)
    _, bx = line_bands(bp, y, th, (0, kyD), (1, 0), sp, n, nev)
    return ks, by, bx


def plot_bands(ax, by, bx, il, iu, color, title):
    n = by.shape[0]
    x1 = np.linspace(-1, 0, n); x2 = np.linspace(0, 1, n)
    ax.plot(x1, by[:, il:iu + 1], color=color, lw=1.7)
    ax.plot(x2, bx[:, il:iu + 1], color=color, lw=1.7)
    ax.plot(x1, by[:, max(0, il - 1):il], color="0.7", lw=0.8)
    ax.plot(x2, bx[:, max(0, il - 1):il], color="0.7", lw=0.8)
    if iu + 1 < by.shape[1]:
        ax.plot(x1, by[:, iu + 1:iu + 2], color="0.7", lw=0.8)
        ax.plot(x2, bx[:, iu + 1:iu + 2], color="0.7", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.6, ls="--")
    ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(["Y", "K", "X"])
    ax.set_title(title); ax.set_ylabel(r"$\omega$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--iters", type=int, default=70)
    ap.add_argument("--phi0", type=float, default=15.0)
    ap.add_argument("--out", default="results/valley_cooptimized.png")
    args = ap.parse_args()

    mesh, bp, rbf, z0, supp, period = build(args.nx, 0.33)
    Lx, Ly = period
    nev = 6
    opt = ValleyOptimizer(bp, R=0.12, period=period, vtarget=float(z0.mean()),
                          w_iso=0.8, iso_margin=0.5, w_vol=60.0)

    th0 = sublattice_support_angles(*supp, period, np.deg2rad(args.phi0))
    sigma0 = 2.67**2

    # initial bands
    y0 = opt.P @ z0
    kyD0, sig0 = opt.track_dirac(y0, th0, sigma0)
    w0 = bp.bands_at_k(0.0, kyD0, nev, sigma=sig0)
    il0, iu0 = _straddle(w0**2, sig0)
    ks, by0, bx0 = bands_through_K(bp, y0, th0, kyD0, sig0, nev)
    g_init = w0[iu0] - w0[il0]
    print(f"initial gap={g_init:.4f} at kyD={kyD0:.3f}")

    t0 = time.time()
    res = opt.run(z0, th0, sigma0, max_iter=args.iters, move=0.07)
    print(f"optimized in {time.time()-t0:.0f}s  best gap={res['gap']:.4f}")
    z, th = res['z'], res['th']

    # final bands
    y = opt.P @ z
    kyD, sig = opt.track_dirac(y, th, res['sigma'])
    wf = bp.bands_at_k(0.0, kyD, nev, sigma=sig)
    il, iu = _straddle(wf**2, sig)
    ksf, byf, bxf = bands_through_K(bp, y, th, kyD, sig, nev)
    g_fin = wf[iu] - wf[il]
    iso = min(wf[il] - wf[il - 1], wf[iu + 1] - wf[iu])
    print(f"final gap={g_fin:.4f}  isolation={iso:.3f}")

    # ---- figure ----
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    plot_bands(ax1, by0, bx0, il0, iu0, "tab:blue",
               f"initial: gap={g_init:.3f}")
    ax2 = fig.add_subplot(2, 2, 2)
    plot_bands(ax2, byf, bxf, il, iu, "tab:red",
               f"co-optimized: gap={g_fin:.3f}, isolation={iso:.2f}")
    ax3 = fig.add_subplot(2, 2, 3)
    yg = y.reshape(mesh.nely, mesh.nelx)
    ax3.imshow(yg, origin="lower", cmap="Greys", vmin=0, vmax=1.4,
               extent=[0, Lx, 0, Ly], aspect="equal", alpha=0.9)
    xs = (np.arange(mesh.nelx) + 0.5) * mesh.dx
    ys = (np.arange(mesh.nely) + 0.5) * mesh.dy
    tg = bp.rbf.theta(th).reshape(mesh.nely, mesh.nelx)
    strm = ax3.streamplot(xs, ys, np.cos(tg), np.sin(tg), density=2.2,
                          color=np.rad2deg(tg), cmap="coolwarm",
                          linewidth=1.0, arrowsize=1e-6)
    plt.colorbar(strm.lines, ax=ax3, fraction=0.046, label="fiber angle (°)")
    ax3.set_title("co-optimized cell: geometry + fiber toolpaths")
    ax3.set_xlim(0, Lx); ax3.set_ylim(0, Ly)
    ax4 = fig.add_subplot(2, 2, 4)
    gaps = [h['gap'] for h in res['hist']]
    ax4.plot(gaps, "darkred")
    ax4.set_xlabel("iteration"); ax4.set_ylabel("valley gap at K")
    ax4.set_title("convergence"); ax4.grid(alpha=0.3)
    fig.suptitle("Co-optimizing honeycomb geometry + fiber toolpath for a "
                 "robust (isolated) valley-Hall gap", y=1.0, fontsize=14)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    np.savez("results/valley_cooptimized.npz", z=z, th=th, kyD=kyD, sigma=sig,
             nelx=mesh.nelx, nely=mesh.nely, Lx=Lx, Ly=Ly)
    print("saved", args.out)


if __name__ == "__main__":
    main()
