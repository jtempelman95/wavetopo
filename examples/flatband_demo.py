"""
Flat-band + band-gap metamaterial: co-optimize topology (density) and toolpath
(fiber orientation) so a target band becomes FLAT (a slow-wave / high-DOS mode)
and is isolated by gaps to its neighbours.

    /home/jrt/miniforge3/bin/python3 examples/flatband_demo.py --band 3 --iters 90
"""
import argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.cfrp import Material, QuadMesh, CSRBFMapping
from wavetopo.bloch import BlochProblem, ibz_path_square
from wavetopo.flatband_opt import FlatBandOptimizer
from wavetopo.cfrp_viz import orientation_streamlines


def bandwidth(bands, b):
    return bands[:, b].max() - bands[:, b].min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--band", type=int, default=3)
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--nk", type=int, default=8)
    ap.add_argument("--ks", type=int, default=4,
                    help="CS-RBF support grid (ks x ks): orientation flexibility")
    ap.add_argument("--R", type=float, default=0.16, help="density filter radius")
    ap.add_argument("--rs", type=float, default=None,
                    help="CS-RBF support radius; default 2*h_s = 2a/ks keeps the "
                         "support OVERLAP RATIO fixed, so more supports genuinely "
                         "buy a finer-turning fiber field (not just more DOF)")
    ap.add_argument("--out", default="results/flatband_demo.png")
    ap.add_argument("--data", default="results/flatband_data.npz")
    args = ap.parse_args()
    a = 1.0
    b = args.band
    mesh = QuadMesh(args.n, args.n, a, a)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    ks = args.ks
    xs = np.linspace(0, a, ks, endpoint=False)
    SX, SY = np.meshgrid(xs, xs)
    r_s = args.rs if args.rs is not None else 2.0*a/ks
    rbf = CSRBFMapping(mesh, (SX.ravel(), SY.ravel()), r_s=r_s, period=(a, a))
    bp = BlochProblem(mesh, mat, rbf, penal=3.0)
    nb = b + 3

    # PASSIVE SOLID FRAME on the cell walls -> connected, periodic, manufacturable
    # cell with mass on all four walls (locally-resonant frame + interior resonator)
    dx, dy = a/mesh.nelx, a/mesh.nely
    bw = max(2, round(args.n/12))          # keep the frame's PHYSICAL width fixed
    frame = ((mesh.cx < bw*dx) | (mesh.cx > a - bw*dx) |
             (mesh.cy < bw*dy) | (mesh.cy > a - bw*dy))
    z0 = np.full(mesh.N, 0.25)
    z0[frame] = 1.0
    r2 = (mesh.cx - a/2)**2 + (mesh.cy - a/2)**2
    z0[r2 < (0.18*a)**2] = 1.0                     # central resonator seed
    th0 = np.zeros(rbf.M)

    # optimization k-set (coarse IBZ path) vs plotting path (fine)
    kset, _, _ = ibz_path_square(a, a, n=args.nk)
    ppath, ticks, labels = ibz_path_square(a, a, n=26)

    opt = FlatBandOptimizer(bp, band=b, n_bands=nb, R=args.R, kset=kset,
                            rho_ks=15.0, v_target=0.5, c_vol=80.0,
                            w_flat=1.0, w_up=1.5, w_dn=1.5, period=(a, a),
                            passive=frame)

    P = opt.P
    y0 = P @ z0; y0[frame] = 1.0
    bp.assemble(y0, th0)
    bands0 = bp.band_structure(ppath, nb)
    print(f"initial band {b}: width={bandwidth(bands0, b):.3f}")

    t0 = time.time()
    res = opt.run(z0, th0, max_iter=args.iters)
    print(f"optimized in {time.time()-t0:.0f}s")
    z, th = res['z'], res['th']
    y = P @ z; y[frame] = 1.0                       # apply the solid frame
    bp.assemble(y, th)
    bands1 = bp.band_structure(ppath, nb)
    info = res['info']
    w0, w1 = bandwidth(bands0, b), bandwidth(bands1, b)
    # true gaps over the plotting path
    gap_up = bands1[:, b+1].min() - bands1[:, b].max()
    gap_dn = bands1[:, b].min() - bands1[:, b-1].max()
    print(f"final band {b}: width {w0:.3f}->{w1:.3f}  gap_up={gap_up:+.3f} "
          f"gap_dn={gap_dn:+.3f}")
    np.savez(args.data,
             bands0=bands0, bands1=bands1, z=y, theta=bp._theta.copy(),
             hist_w=[h['width'] for h in res['hist']],
             hist_gu=[h['gap_up'] for h in res['hist']],
             hist_gd=[h['gap_dn'] for h in res['hist']],
             band=b, w0=w0, w1=w1, gap_up=gap_up, gap_dn=gap_dn,
             nelx=mesh.nelx, nely=mesh.nely, a=a,
             ks=ks, R=args.R, bw=bw, M=rbf.M, r_s=r_s, nk=args.nk,
             iters=args.iters)

    # ---------------- figure ----------------
    kk = np.arange(bands0.shape[0])
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for a_, bands, ttl in [(ax[0, 0], bands0, f"baseline (band {b} width {w0:.2f})"),
                           (ax[0, 1], bands1, f"optimized (band {b} width {w1:.2f})")]:
        a_.plot(kk, bands, "0.7", lw=1.0)
        a_.plot(kk, bands[:, b], "b", lw=2.2)           # the target band
        a_.set_xticks(ticks); a_.set_xticklabels(labels)
        a_.set_ylabel(r"$\omega$"); a_.set_title(ttl); a_.grid(alpha=0.3)
    # shade isolating gaps on the optimized panel
    if gap_up > 0:
        ax[0, 1].axhspan(bands1[:, b].max(), bands1[:, b+1].min(),
                         color="orange", alpha=0.35)
    if gap_dn > 0:
        ax[0, 1].axhspan(bands1[:, b-1].max(), bands1[:, b].min(),
                         color="orange", alpha=0.35)

    ax[0, 2].plot([h['width'] for h in res['hist']], label="band width")
    ax[0, 2].plot([h['gap_up'] for h in res['hist']], label="gap above")
    ax[0, 2].plot([h['gap_dn'] for h in res['hist']], label="gap below")
    ax[0, 2].axhline(0, color="k", lw=0.6)
    ax[0, 2].set_title("convergence"); ax[0, 2].set_xlabel("iter"); ax[0, 2].legend()
    ax[0, 2].grid(alpha=0.3)

    dg = y.reshape(mesh.nely, mesh.nelx)
    tile = np.tile(dg, (2, 2))
    im = ax[1, 0].imshow(tile, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                         extent=[0, 2*a, 0, 2*a], aspect="equal")
    ax[1, 0].axhline(a, color="r", lw=0.6); ax[1, 0].axvline(a, color="r", lw=0.6)
    ax[1, 0].set_title("optimized density (topology), 2$\\times$2 tiling")
    plt.colorbar(im, ax=ax[1, 0], fraction=0.046)

    theta_t = np.tile(bp._theta.reshape(mesh.nely, mesh.nelx), (2, 2)).ravel()
    mesh_t = QuadMesh(2*mesh.nelx, 2*mesh.nely, 2*a, 2*a)
    orientation_streamlines(ax[1, 1], mesh_t, theta_t, np.tile(dg, (2, 2)).ravel(),
                            dens_thresh=0.3)
    ax[1, 1].axhline(a, color="r", lw=0.6); ax[1, 1].axvline(a, color="r", lw=0.6)
    ax[1, 1].set_title("optimized fiber toolpaths, 2$\\times$2 tiling")

    ax[1, 2].axis("off")
    ax[1, 2].text(0.0, 0.75,
                  f"target band: {b}\n\n"
                  f"band width  {w0:.3f} -> {w1:.3f}\n"
                  f"  ({w0/max(w1,1e-6):.1f}x flatter)\n\n"
                  f"gap above   {gap_up:+.3f}\n"
                  f"gap below   {gap_dn:+.3f}\n\n"
                  f"co-designed: density (topology)\n"
                  f"           + fiber orientation (toolpath)",
                  fontsize=12, family="monospace", va="top")
    fig.suptitle("Flat-band + band-gap metamaterial: co-optimized topology and "
                 "fiber toolpath (in-plane orthotropic unit cell)", y=1.0, fontsize=14)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
