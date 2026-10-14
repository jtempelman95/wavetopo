"""
Manufacturable (curvature-limited) toolpath wave lens.

The single-frequency fiber lens, now with the *practical manufacturing
constraint* of the quasi-static CFRP problem: local fiber-curvature (curl)
limits |zeta_l| <= zeta_all enforced by the augmented-Lagrangian + MMA driver.
Sweeps the curvature allowable and maps the printability/performance Pareto
front: how much focusing survives when the toolpaths must respect a minimum
turning radius (r_min = 1/zeta_all)?

    python examples/wave_lens_mfg.py --nx 120 --omega 40
"""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.cfrp import Material, QuadMesh, grid_support_points, CSRBFMapping
from wavetopo.wave_mfg import HarmonicPlate, optimize_orientation, ramp_sponge
from wavetopo.cfrp_viz import orientation_streamlines


def nodal_mag(u, mesh):
    ux = u[0::2]; uy = u[1::2]
    return np.sqrt(np.abs(ux)**2 + np.abs(uy)**2).reshape(mesh.nely + 1,
                                                          mesh.nelx + 1)


def curl_grid(hp, th, mesh):
    return np.abs(hp.curl(th)).reshape(mesh.nely, mesh.nelx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--omega", type=float, default=40.0)
    ap.add_argument("--iters", type=int, default=110)
    ap.add_argument("--outer", type=int, default=40)
    ap.add_argument("--zetas", type=float, nargs="+", default=[2.0, 1.0, 0.5])
    ap.add_argument("--out", default="results/wave_lens_mfg.png")
    args = ap.parse_args()

    Lx, Ly = 4.0, 3.0
    ny = int(round(args.nx * Ly / Lx))
    mesh = QuadMesh(args.nx, ny, Lx, Ly)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    sx, sy = grid_support_points(mesh, 0.25, 0.25)
    rbf = CSRBFMapping(mesh, (sx, sy), r_s=0.55)
    hp = HarmonicPlate(mesh, mat, rbf, [args.omega], eta=0.03)
    hp.set_load(hp.plane_load("left"))
    hp.set_sponge(ramp_sponge(mesh, 0.6, 80.0))
    focus = (0.78 * Lx, Ly / 2)
    hp.set_focus(hp.box_weight(focus, 0.18))
    base = hp.raw_values(np.zeros(rbf.M))
    hp.set_baseline(base)
    print(f"mesh {mesh.nelx}x{mesh.nely}  M={rbf.M}  omega={args.omega}  "
          f"baseline E0={base[0]:.4e}")

    designs = {}          # label -> dict(th, gain, curl_max, hist)
    t0 = time.time()

    # (a) unconstrained
    res = optimize_orientation(hp, np.zeros(rbf.M), iters=args.iters,
                               verbose=False)
    designs["unconstrained"] = dict(
        th=res['theta_hat'], gain=-res['J'], curl_max=res['curl_max'],
        hist=[-j for j in res['hist']['J']])
    print(f"unconstrained: gain={-res['J']:.1f}x  |zeta|max="
          f"{res['curl_max']:.2f}  ({time.time()-t0:.0f}s)")

    # (b) curvature-limited designs
    for za in args.zetas:
        t1 = time.time()
        res = optimize_orientation(hp, np.zeros(rbf.M), zeta_all=za,
                                   mu0=50.0, xi=1.5, max_outer=args.outer,
                                   mma_iter=5, verbose=False)
        designs[f"zeta<={za:g}"] = dict(
            th=res['theta_hat'], gain=-res['J'], curl_max=res['curl_max'],
            hist=[-j for j in res['hist']['J']], zeta_all=za)
        print(f"zeta_all={za:g}: gain={-res['J']:.1f}x  |zeta|max="
              f"{res['curl_max']:.3f}  ({time.time()-t1:.0f}s)")

    # ---- figure: fields / toolpaths / curl maps per design + Pareto ---- #
    labels = list(designs.keys())
    ncol = len(labels)
    fig = plt.figure(figsize=(4.4 * ncol, 13.5))
    gs = fig.add_gridspec(4, ncol, height_ratios=[1, 1, 1, 0.85],
                          hspace=0.28, wspace=0.22)
    fx, fy = focus
    # common field scale from the unconstrained design
    u_ref = hp.solve(designs["unconstrained"]['th'])[0]
    vmax = np.percentile(nodal_mag(u_ref, mesh), 99.5)
    zmax = max(d.get('zeta_all', 0) for d in designs.values()) * 1.6

    for j, lab in enumerate(labels):
        d = designs[lab]
        u = hp.solve(d['th'])[0]
        theta = rbf.theta(d['th'])
        # row 0: field
        a = fig.add_subplot(gs[0, j])
        im = a.imshow(nodal_mag(u, mesh), origin="lower", cmap="magma",
                      vmax=vmax, extent=[0, Lx, 0, Ly], aspect="equal")
        a.add_patch(plt.Rectangle((fx - 0.18, fy - 0.18), 0.36, 0.36,
                                  ec="cyan", fc="none", lw=1.5))
        a.set_title(f"{lab}\n|u|, gain {d['gain']:.1f}x", fontsize=11)
        plt.colorbar(im, ax=a, fraction=0.035)
        # row 1: toolpaths
        a = fig.add_subplot(gs[1, j])
        orientation_streamlines(a, mesh, theta, np.ones(mesh.N),
                                dens_thresh=-1, density_bg=False, color="k",
                                linewidth=0.7)
        a.add_patch(plt.Rectangle((fx - 0.18, fy - 0.18), 0.36, 0.36,
                                  ec="cyan", fc="none", lw=1.5))
        a.set_title("fiber toolpaths", fontsize=11)
        # row 2: curl map
        a = fig.add_subplot(gs[2, j])
        im = a.imshow(curl_grid(hp, d['th'], mesh), origin="lower",
                      cmap="magma", vmin=0, vmax=zmax,
                      extent=[0, Lx, 0, Ly], aspect="equal")
        ttl = f"|zeta| (max {d['curl_max']:.2f}"
        ttl += f", limit {d['zeta_all']:g})" if 'zeta_all' in d else ")"
        a.set_title(ttl, fontsize=11)
        plt.colorbar(im, ax=a, fraction=0.035)

    # row 3: Pareto + convergence
    a = fig.add_subplot(gs[3, :ncol // 2])
    zs = [designs[l]['curl_max'] for l in labels]
    gains = [designs[l]['gain'] for l in labels]
    a.plot(zs[1:], gains[1:], "o-", color="#1f77b4", lw=1.8, ms=7,
           label="curvature-limited")
    a.plot([zs[0]], [gains[0]], "s", color="#d62728", ms=8,
           label="unconstrained")
    for zli, gi, lab in zip(zs, gains, labels):
        a.annotate(f" {lab}", (zli, gi), fontsize=8, va="bottom")
    a.set_xlabel("achieved max fiber curvature |zeta|max [1/m]")
    a.set_ylabel("focus gain vs straight fibers")
    a.set_title("printability-performance Pareto front")
    a.grid(alpha=0.25); a.legend(fontsize=9)
    a2 = fig.add_subplot(gs[3, ncol // 2:])
    for lab, color in zip(labels, ["#d62728", "#1f77b4", "#2ca02c",
                                   "#9467bd"]):
        a2.plot(designs[lab]['hist'], color=color, lw=1.5, label=lab)
    a2.set_xlabel("(outer) iteration"); a2.set_ylabel("focus gain")
    a2.set_title("convergence"); a2.grid(alpha=0.25); a2.legend(fontsize=9)

    fig.suptitle("Curvature-limited toolpath wave lens: printable fiber paths "
                 f"(omega={args.omega:g})", y=0.995, fontsize=14)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)
    np.savez(args.out.replace(".png", ".npz"),
             **{f"th_{i}": designs[l]['th'] for i, l in enumerate(labels)},
             labels=labels, gains=gains, curls=zs, base=base)


if __name__ == "__main__":
    main()
