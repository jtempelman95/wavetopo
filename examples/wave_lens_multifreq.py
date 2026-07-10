"""
Broadband (multi-frequency) toolpath wave lens.

A single-material fiber-composite plate, one fixed fiber-orientation design,
that focuses SEVERAL excitation frequencies at the same target using the
robust max-min (worst-case) objective so every frequency benefits.  Compares
per-frequency focus gain against the straight-fiber baseline, and against a
lens optimized for the centre frequency only (which is narrow-band).

    python examples/wave_lens_multifreq.py --nx 120 --iters 60
"""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topoopt.cfrp import Material, QuadMesh, grid_support_points, CSRBFMapping
from topoopt.wave_control import WaveFocus, ramp_sponge
from topoopt.cfrp_optimizer import MMA
from topoopt.cfrp_viz import orientation_streamlines


def nodal_mag(u, mesh):
    ux = u[0::2]; uy = u[1::2]
    return np.sqrt(np.abs(ux)**2 + np.abs(uy)**2).reshape(mesh.nely + 1,
                                                          mesh.nelx + 1)


def build(nx, omegas):
    Lx, Ly = 4.0, 3.0
    ny = int(round(nx * Ly / Lx))
    mesh = QuadMesh(nx, ny, Lx, Ly)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    sx, sy = grid_support_points(mesh, 0.28, 0.28)
    rbf = CSRBFMapping(mesh, (sx, sy), r_s=0.6)
    wf = WaveFocus(mesh, mat, rbf, omegas, eta=0.03)
    F = np.zeros(mesh.ndof, dtype=complex)
    for iy in range(ny + 1):
        n = wf.node_id(0, iy); F[2 * n] = 1.0
    wf.set_load(F)
    wf.set_sponge(ramp_sponge(mesh, 0.6, 80.0))
    focus = (0.78 * Lx, Ly / 2)
    wf.set_targets([wf.focus_weight(focus, 0.18) for _ in omegas])
    return mesh, wf, focus, (Lx, Ly)


def optimize(wf, rbf_M, iters, mode, base, rho_ks=25.0, move=0.1):
    mma = MMA(np.full(rbf_M, -np.pi / 2), np.full(rbf_M, np.pi / 2), move=move)
    th = np.zeros(rbf_M)
    wf.set_baseline(base)
    best = (-np.inf, th.copy())
    hist = []
    for it in range(iters):
        J = wf.objective(th, mode=mode, rho_ks=rho_ks)
        g = wf.grad()
        gains = wf.energies() / base
        worst = gains.min()
        hist.append(worst)
        if worst > best[0]:
            best = (worst, th.copy())
        th = mma.update(th, g)
    return best[1], hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--omegas", type=float, nargs="+",
                    default=[28.0, 38.0, 48.0])
    ap.add_argument("--out", default="results/wave_lens_multifreq.png")
    args = ap.parse_args()

    omegas = args.omegas
    mesh, wf, focus, (Lx, Ly) = build(args.nx, omegas)
    print(f"mesh {mesh.nelx}x{mesh.nely}, {len(omegas)} freqs {omegas}, "
          f"M={wf.M_}")

    # baseline: straight fibers
    base = wf.energies(np.zeros(wf.M_))
    print("straight-fiber baseline focus energies:", np.round(base, 5))

    # (a) broadband optimize (max-min)
    t0 = time.time()
    th_bb, hist_bb = optimize(wf, wf.M_, args.iters, 'minmax', base)
    gains_bb = wf.energies(th_bb) / base
    print(f"broadband (max-min) gains per freq: {np.round(gains_bb,2)}  "
          f"({time.time()-t0:.0f}s)")

    # (b) centre-frequency-only lens, evaluated across the band
    ic = len(omegas) // 2
    mesh2, wfc, _, _ = build(args.nx, [omegas[ic]])
    base_c1 = wfc.energies(np.zeros(wfc.M_))
    th_c, _ = optimize(wfc, wfc.M_, args.iters, 'sum', base_c1)
    # evaluate the centre-only design across the full band
    gains_c = wf.energies(th_c) / base
    print(f"centre-only lens gains across band: {np.round(gains_c,2)}")

    # ---- figure ----
    nfr = len(omegas)
    fig, ax = plt.subplots(2, nfr, figsize=(4.6 * nfr, 8))
    wf.solve(th_bb)
    us = wf._state['us']
    vmax = np.percentile(nodal_mag(us[ic], mesh), 99.5)
    for j, (u, w) in enumerate(zip(us, omegas)):
        m = nodal_mag(u, mesh)
        im = ax[0, j].imshow(m, origin="lower", cmap="magma", vmax=vmax,
                             extent=[0, Lx, 0, Ly], aspect="equal")
        ax[0, j].add_patch(plt.Rectangle((focus[0]-0.18, focus[1]-0.18), 0.36,
                                         0.36, ec="cyan", fc="none", lw=1.5))
        ax[0, j].set_title(f"omega={w:.0f}   gain {gains_bb[j]:.1f}x")
        plt.colorbar(im, ax=ax[0, j], fraction=0.035)
    # bottom row: toolpaths, gains bar, convergence
    orientation_streamlines(ax[1, 0], mesh, wf.rbf.theta(th_bb),
                            np.ones(mesh.N), dens_thresh=-1, density_bg=False,
                            color="k", linewidth=0.7)
    ax[1, 0].add_patch(plt.Rectangle((focus[0]-0.18, focus[1]-0.18), 0.36, 0.36,
                                     ec="cyan", fc="none", lw=1.5))
    ax[1, 0].set_title("broadband fiber toolpaths")
    x = np.arange(nfr)
    ax[1, 1].bar(x - 0.2, gains_bb, 0.4, label="broadband (max-min)")
    ax[1, 1].bar(x + 0.2, gains_c, 0.4, label="centre-only lens")
    ax[1, 1].axhline(1, color="k", lw=0.6)
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels([f"{w:.0f}" for w in omegas])
    ax[1, 1].set_xlabel("omega"); ax[1, 1].set_ylabel("focus gain vs straight")
    ax[1, 1].set_title("per-frequency gain"); ax[1, 1].legend(fontsize=8)
    if nfr > 2:
        ax[1, 2].plot(hist_bb, "b-")
        ax[1, 2].set_xlabel("iteration"); ax[1, 2].set_ylabel("worst-case gain")
        ax[1, 2].set_title("max-min convergence"); ax[1, 2].grid(alpha=0.3)
    fig.suptitle("Broadband toolpath wave lens: one fiber design focuses a "
                 "band of frequencies", y=1.0, fontsize=13)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)
    np.savez("results/wave_lens_multifreq.npz", th_bb=th_bb, th_c=th_c,
             omegas=omegas, gains_bb=gains_bb, gains_c=gains_c, base=base)


if __name__ == "__main__":
    main()
