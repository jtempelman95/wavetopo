"""
Toolpath wave lens in a plate with through-holes.

A fiber-composite plate carries three drilled through-holes (true voids --
e.g. bolt/rivet holes) standing between the source edge and the focus
target.  Straight fibers scatter off the holes and the focus starves; the
optimized fiber orientation routes the wave *around* the holes and restores
the focus, while a local curvature (curl) limit keeps the toolpaths
printable.  Fibers are only constrained where they are actually deposited
(outside the holes).

    python examples/wave_lens_holes.py --nx 120 --omega 40
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--omega", type=float, default=40.0)
    ap.add_argument("--iters", type=int, default=110)
    ap.add_argument("--outer", type=int, default=40)
    ap.add_argument("--zeta", type=float, default=1.5)
    ap.add_argument("--rh", type=float, default=0.28, help="hole radius")
    ap.add_argument("--out", default="results/wave_lens_holes.png")
    args = ap.parse_args()

    Lx, Ly = 4.0, 3.0
    ny = int(round(args.nx * Ly / Lx))
    mesh = QuadMesh(args.nx, ny, Lx, Ly)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    sx, sy = grid_support_points(mesh, 0.25, 0.25)
    rbf = CSRBFMapping(mesh, (sx, sy), r_s=0.55)

    # three through-holes forming a barrier between source and focus
    holes = [(1.5, 0.85), (1.35, 1.5), (1.5, 2.15)]
    z = np.ones(mesh.N)
    for hx, hy in holes:
        z[np.hypot(mesh.cx - hx, mesh.cy - hy) < args.rh] = 0.0

    hp = HarmonicPlate(mesh, mat, rbf, [args.omega], eta=0.03, z=z)
    hp.set_load(hp.plane_load("left"))
    hp.set_sponge(ramp_sponge(mesh, 0.6, 80.0))
    focus = (0.78 * Lx, Ly / 2)
    hp.set_focus(hp.box_weight(focus, 0.18))

    # baselines: solid straight, holes straight
    hp_solid = HarmonicPlate(mesh, mat, rbf, [args.omega], eta=0.03)
    hp_solid.set_load(hp.plane_load("left"))
    hp_solid.set_sponge(ramp_sponge(mesh, 0.6, 80.0))
    hp_solid.set_focus(hp.box_weight(focus, 0.18))
    E_solid = hp_solid.raw_values(np.zeros(rbf.M))[0]
    E_holes0 = hp.raw_values(np.zeros(rbf.M))[0]
    u_base = hp._state['us'][0]
    hp.set_baseline([E_holes0])
    print(f"mesh {mesh.nelx}x{mesh.nely}  M={rbf.M}  "
          f"straight-fiber focus energy: solid={E_solid:.3e}  "
          f"with holes={E_holes0:.3e} ({E_holes0/E_solid:.2f}x of solid)")

    # optimize with curvature limit (curl constrained only where solid)
    t0 = time.time()
    res = optimize_orientation(hp, np.zeros(rbf.M), zeta_all=args.zeta,
                               mu0=50.0, xi=1.5, max_outer=args.outer,
                               mma_iter=5, verbose=True)
    th = res['theta_hat']
    gain = -res['J']
    u_opt = hp.solve(th)[0]
    E_opt = hp.raw_values()[0]
    print(f"optimized in {time.time()-t0:.0f}s: gain={gain:.1f}x over "
          f"holes-baseline ({E_opt/E_solid:.1f}x of solid-straight), "
          f"|zeta|max={res['curl_max']:.3f} (limit {args.zeta})")

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 9))
    fx, fy = focus
    vmax = np.percentile(nodal_mag(u_opt, mesh), 99.5)
    circ = np.linspace(0, 2 * np.pi, 80)
    for a_, u, ttl in [
            (ax[0, 0], u_base,
             f"straight fibers + holes   E={E_holes0:.2e}"),
            (ax[0, 1], u_opt,
             f"optimized (curl-limited)   E={E_opt:.2e}   gain {gain:.1f}x")]:
        im = a_.imshow(nodal_mag(u, mesh), origin="lower", cmap="magma",
                       vmax=vmax, extent=[0, Lx, 0, Ly], aspect="equal")
        a_.add_patch(plt.Rectangle((fx - 0.18, fy - 0.18), 0.36, 0.36,
                                   ec="cyan", fc="none", lw=1.5))
        for hx, hy in holes:
            a_.plot(hx + args.rh * np.cos(circ), hy + args.rh * np.sin(circ),
                    "w-", lw=1.2)
        a_.set_title(ttl, fontsize=11)
        plt.colorbar(im, ax=a_, fraction=0.035)

    theta = rbf.theta(th)
    orientation_streamlines(ax[1, 0], mesh, theta, z, dens_thresh=0.5,
                            density_bg=True, color="tab:red",
                            seed_density=2.6, linewidth=0.7)
    for hx, hy in holes:
        ax[1, 0].plot(hx + args.rh * np.cos(circ),
                      hy + args.rh * np.sin(circ), "c-", lw=1.4)
    ax[1, 0].add_patch(plt.Rectangle((fx - 0.18, fy - 0.18), 0.36, 0.36,
                                     ec="cyan", fc="none", lw=1.5))
    ax[1, 0].set_title("fiber toolpaths route around the through-holes",
                       fontsize=11)

    zeta_map = np.abs(hp.curl(th))
    zeta_map[z < 0.5] = np.nan
    im = ax[1, 1].imshow(zeta_map.reshape(mesh.nely, mesh.nelx),
                         origin="lower", cmap="magma", vmin=0,
                         vmax=1.3 * args.zeta, extent=[0, Lx, 0, Ly],
                         aspect="equal")
    ax[1, 1].set_title(f"|zeta| where fiber is deposited "
                       f"(max {res['curl_max']:.2f}, limit {args.zeta:g})",
                       fontsize=11)
    plt.colorbar(im, ax=ax[1, 1], fraction=0.035)

    fig.suptitle("Wave lens with drilled through-holes: printable toolpaths "
                 f"recover the focus (omega={args.omega:g}, "
                 f"zeta_all={args.zeta:g})", y=0.995, fontsize=13)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)
    np.savez(args.out.replace(".png", ".npz"), th=th, z=z, gain=gain,
             E_solid=E_solid, E_holes0=E_holes0, E_opt=E_opt,
             curl_max=res['curl_max'], holes=holes, rh=args.rh)


if __name__ == "__main__":
    main()
