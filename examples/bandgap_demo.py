"""
Optimize a DIRECTIONAL phononic band gap in a fiber-composite unit cell.

Starting from a lamellar cell (which already has a small transverse gap), the
optimizer adjusts density + fiber orientation to widen the gap for waves
travelling along y (transverse) while suppressing it along x (the fiber
direction).  Demonstrates using the CFRP design freedom for wave control.
"""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topoopt.cfrp import Material, QuadMesh, grid_support_points, CSRBFMapping
from topoopt.bloch import BlochProblem, directional_path, gap_between_bands
from topoopt.bandgap_opt import BandGapOptimizer
from topoopt.cfrp_viz import orientation_streamlines


def directional_bands(bp, a, n_bands, n=40):
    pY = directional_path(a, a, np.pi / 2, n=n)
    pX = directional_path(a, a, 0.0, n=n)
    return bp.band_structure(pX, n_bands), bp.band_structure(pY, n_bands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--nlow", type=int, default=2)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--out", default="results/bandgap_demo.png")
    args = ap.parse_args()

    a = 1.0
    mesh = QuadMesh(args.n, args.n, a, a)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    # PERIODIC design fields so the optimized cell tiles seamlessly:
    #   - support points on a grid with NO endpoint duplication
    #   - CS-RBF + density filter use toroidal (wrap-around) distances
    ks = 4
    xs = np.linspace(0, a, ks, endpoint=False)
    SX, SY = np.meshgrid(xs, xs)
    rbf = CSRBFMapping(mesh, (SX.ravel(), SY.ravel()), r_s=a / 2,
                       period=(a, a))
    bp = BlochProblem(mesh, mat, rbf, penal=3.0)
    nb = max(args.nlow + 3, 6)

    # initial design: horizontal stiff lamella + horizontal fibers
    z0 = np.zeros(mesh.N)
    z0[np.abs(mesh.cy - a / 2) < 0.25 * a] = 1.0
    z0 = 0.5 * z0 + 0.25            # soften so filter/optim has room
    th0 = np.zeros(rbf.M)           # fibers along x

    opt = BandGapOptimizer(bp, nlow=args.nlow, n_bands=nb, R=0.12,
                           target_dir=np.pi / 2, orth_dir=0.0,
                           n_k=9, rho_ks=15.0, v_target=0.5, c_vol=80.0,
                           w_orth=0.4, period=(a, a))

    # baseline bands
    y0 = opt.P @ z0
    bp.assemble(y0, th0)
    bX0, bY0 = directional_bands(bp, a, nb)
    gY0 = gap_between_bands(bY0, args.nlow)[0]
    gX0 = gap_between_bands(bX0, args.nlow)[0]
    print(f"initial: gapY={gY0:+.4f}  gapX={gX0:+.4f}")

    t0 = time.time()
    res = opt.run(z0, th0, max_iter=args.iters)
    print(f"optimized in {time.time()-t0:.0f}s")

    z, th = res['z'], res['th']
    yopt = opt.P @ z
    bp.assemble(yopt, th)
    bX, bY = directional_bands(bp, a, nb)
    gY = gap_between_bands(bY, args.nlow)
    gX = gap_between_bands(bX, args.nlow)
    print(f"final:   gapY={gY[0]:+.4f}  gapX={gX[0]:+.4f}   "
          f"(directional ratio gapY/|gapX| favorable)")

    # ---- figure ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    kk = np.linspace(0, 1, bY0.shape[0])
    for col, (bb, ttl) in enumerate([
            (bX0, r"init $\Gamma\to$X (along fiber)"),
            (bY0, r"init $\Gamma\to$Y (transverse)")]):
        ax[0, col].plot(kk, bb, "0.6", lw=1)
    ax[0, 0].plot(kk, bX, "b", lw=1.2); ax[0, 0].set_title(r"$\Gamma\to$X (along fiber)")
    ax[0, 1].plot(kk, bY, "b", lw=1.2); ax[0, 1].set_title(r"$\Gamma\to$Y (transverse)")
    if gY[0] > 0:
        ax[0, 1].axhspan(gY[1], gY[2], color="orange", alpha=0.35)
        ax[0, 1].text(0.05, 0.5*(gY[1]+gY[2]), f"gap={gY[0]:.2f}", fontsize=10)
    if gX[0] > 0:
        ax[0, 0].axhspan(gX[1], gX[2], color="orange", alpha=0.35)
    for a_ in ax[0, :2]:
        a_.set_xlabel(r"$|k|/(\pi/a)$"); a_.set_ylabel(r"$\omega$")

    # convergence
    gtar = [h['gap_target'] for h in res['hist']]
    gorth = [h.get('gap_orth', np.nan) for h in res['hist']]
    ax[0, 2].plot(gtar, label="gap target (Y)")
    ax[0, 2].plot(gorth, label="gap orth (X)")
    ax[0, 2].axhline(0, color="k", lw=0.6)
    ax[0, 2].set_title("convergence"); ax[0, 2].legend(); ax[0, 2].set_xlabel("iter")

    # periodicity check + tiled designs (prove the cell is seamlessly periodic)
    dg = yopt.reshape(mesh.nely, mesh.nelx)
    lr = np.max(np.abs(dg[:, 0] - dg[:, -1]))
    tb = np.max(np.abs(dg[0, :] - dg[-1, :]))
    print(f"periodicity (filtered density) max edge mismatch: "
          f"L-R={lr:.2e}  T-B={tb:.2e}")
    tile = np.tile(dg, (2, 2))
    im = ax[1, 0].imshow(tile, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                         extent=[0, 2 * a, 0, 2 * a], aspect="equal")
    for c in (a,):
        ax[1, 0].axhline(c, color="r", lw=0.6); ax[1, 0].axvline(c, color="r", lw=0.6)
    ax[1, 0].set_title("optimized density, 2x2 tiling (seamless)")
    plt.colorbar(im, ax=ax[1, 0], fraction=0.046)
    # tiled orientation streamlines
    theta_t = np.tile(bp._theta.reshape(mesh.nely, mesh.nelx), (2, 2)).ravel()
    mesh_t = QuadMesh(2 * mesh.nelx, 2 * mesh.nely, 2 * a, 2 * a)
    orientation_streamlines(ax[1, 1], mesh_t, theta_t,
                            np.tile(dg, (2, 2)).ravel(), dens_thresh=0.3)
    ax[1, 1].axhline(a, color="r", lw=0.6); ax[1, 1].axvline(a, color="r", lw=0.6)
    ax[1, 1].set_title("optimized fiber orientation, 2x2 tiling")
    ax[1, 2].axis("off")
    ax[1, 2].text(0.0, 0.7,
                  f"band pair {args.nlow}-{args.nlow+1}\n\n"
                  f"initial  gapY={gY0:+.3f}  gapX={gX0:+.3f}\n"
                  f"final    gapY={gY[0]:+.3f}  gapX={gX[0]:+.3f}\n\n"
                  f"-> wider transverse gap, suppressed\n   along-fiber gap "
                  f"= directional band gap",
                  fontsize=12, family="monospace", va="top")
    fig.suptitle("Directional phononic band-gap optimization "
                 "(fiber-composite cell)", y=1.0, fontsize=14)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
