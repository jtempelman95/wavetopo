"""
Broadband, printable cloak of a true through-hole.

A drilled through-hole (true void, not a soft inclusion) scatters an incident
plane wave.  The fiber orientation in an annular shell is optimized so the
field outside the shell matches the hole-free reference *simultaneously at
several frequencies* (worst-case / KS-softmax objective), under a local
fiber-curvature limit so the cloak toolpaths respect the printer's minimum
turning radius.

    python examples/wave_cloak_mfg.py --nx 110
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
    ap.add_argument("--nx", type=int, default=110)
    ap.add_argument("--omegas", type=float, nargs="+",
                    default=[30.0, 36.0, 42.0])
    ap.add_argument("--outer", type=int, default=35)
    ap.add_argument("--zeta", type=float, default=1.5)
    ap.add_argument("--rv", type=float, default=0.55)
    ap.add_argument("--rc", type=float, default=1.4)
    ap.add_argument("--out", default="results/wave_cloak_mfg.png")
    args = ap.parse_args()

    Lx, Ly = 6.0, 4.0
    ny = int(round(args.nx * Ly / Lx))
    mesh = QuadMesh(args.nx, ny, Lx, Ly)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    sx, sy = grid_support_points(mesh, 0.28, 0.28)
    rbf = CSRBFMapping(mesh, (sx, sy), r_s=0.6)
    cx, cy = Lx / 2, Ly / 2
    omegas = args.omegas

    z = np.ones(mesh.N)
    z[np.hypot(mesh.cx - cx, mesh.cy - cy) < args.rv] = 0.0

    hp = HarmonicPlate(mesh, mat, rbf, omegas, eta=0.03)
    hp.set_load(hp.plane_load("left"))
    marg = 0.55
    hp.set_sponge(ramp_sponge(mesh, marg, 70.0))

    # reference: hole-free plate, straight fibers, all frequencies
    u_ref = hp.solve(np.zeros(rbf.M))

    # observation region: outside the cloak shell, inside the sponge-free box
    xs, ys = mesh.node_xy()
    rr = np.hypot(xs - cx, ys - cy)
    obs = (rr > args.rc) & (xs > marg) & (xs < Lx - marg) \
        & (ys > marg) & (ys < Ly - marg)
    w = np.zeros(mesh.ndof)
    w[0::2][obs] = 1.0; w[1::2][obs] = 1.0

    hp.set_density(z)
    hp.set_match(u_ref, w)
    base = hp.raw_values(np.zeros(rbf.M))      # uncloaked scatter per freq
    hp.set_baseline(base)
    u_unc = hp._state['us']
    print(f"mesh {mesh.nelx}x{mesh.nely}  M={rbf.M}  freqs={omegas}")
    print("uncloaked scatter per freq:", [f"{b:.3e}" for b in base])

    # design only the shell; curl constrained only in the shell (where the
    # cloak toolpaths are actually laid)
    sdist = np.hypot(sx - cx, sy - cy)
    active = sdist < args.rc + 0.3
    shell = (np.hypot(mesh.cx - cx, mesh.cy - cy) < args.rc + 0.3) & (z > 0.5)
    print(f"{int(active.sum())} active support points")

    t0 = time.time()
    res = optimize_orientation(hp, np.zeros(rbf.M), zeta_all=args.zeta,
                               curl_mask=shell, active=active, mode='minmax',
                               mu0=50.0, xi=1.5, max_outer=args.outer,
                               mma_iter=5, move=0.15, verbose=True)
    th = res['theta_hat']
    u_cl = hp.solve(th)
    scat = hp.raw_values()
    red = base / scat
    print(f"optimized in {time.time()-t0:.0f}s  per-freq scatter reduction: "
          + "  ".join(f"{r:.1f}x" for r in red)
          + f"  |zeta|max={res['curl_max']:.3f} (limit {args.zeta})")

    # ---- figure: rows = frequencies, cols = ref/uncloaked/cloaked ----- #
    nfr = len(omegas)
    fig = plt.figure(figsize=(16.5, 3.6 * nfr + 4.0))
    gs = fig.add_gridspec(nfr + 1, 3, hspace=0.3, wspace=0.15,
                          height_ratios=[1] * nfr + [1.1])
    circ = np.linspace(0, 2 * np.pi, 100)
    for i, om in enumerate(omegas):
        vmax = np.percentile(nodal_mag(u_ref[i], mesh), 99.5)
        for j, (u, ttl) in enumerate([
                (u_ref[i], "reference (no hole)"),
                (u_unc[i], f"uncloaked hole"),
                (u_cl[i], f"cloaked  ({red[i]:.1f}x less scatter)")]):
            a = fig.add_subplot(gs[i, j])
            im = a.imshow(nodal_mag(u, mesh), origin="lower", cmap="magma",
                          vmax=vmax, extent=[0, Lx, 0, Ly], aspect="equal")
            a.plot(cx + args.rv * np.cos(circ), cy + args.rv * np.sin(circ),
                   "c-", lw=1)
            if j == 0:
                a.set_ylabel(f"omega = {om:g}", fontsize=11)
            if i == 0:
                a.set_title(ttl, fontsize=11)
            a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.03)

    # bottom row: toolpaths, per-frequency reduction, curl map
    a = fig.add_subplot(gs[nfr, 0])
    orientation_streamlines(a, mesh, rbf.theta(th), z, dens_thresh=0.5,
                            density_bg=True, color="tab:red",
                            seed_density=3.0, linewidth=0.7)
    a.plot(cx + args.rv * np.cos(circ), cy + args.rv * np.sin(circ), "c-",
           lw=1.5)
    a.plot(cx + args.rc * np.cos(circ), cy + args.rc * np.sin(circ), "y--",
           lw=1, alpha=0.7)
    a.set_title("cloak toolpaths (curvature-limited)", fontsize=11)

    a = fig.add_subplot(gs[nfr, 1])
    x = np.arange(nfr)
    a.bar(x, red, 0.55, color="#1f77b4")
    for xi_, ri in zip(x, red):
        a.annotate(f"{ri:.1f}x", (xi_, ri), ha="center", va="bottom",
                   fontsize=10)
    a.axhline(1, color="k", lw=0.6)
    a.set_xticks(x); a.set_xticklabels([f"{om:g}" for om in omegas])
    a.set_xlabel("omega"); a.set_ylabel("scatter reduction")
    a.set_title("broadband (worst-case) cloaking", fontsize=11)
    a.grid(alpha=0.25, axis="y")

    a = fig.add_subplot(gs[nfr, 2])
    zeta_map = np.abs(hp.curl(th))
    zeta_map[~shell] = np.nan
    im = a.imshow(zeta_map.reshape(mesh.nely, mesh.nelx), origin="lower",
                  cmap="magma", vmin=0, vmax=1.3 * args.zeta,
                  extent=[0, Lx, 0, Ly], aspect="equal")
    a.set_title(f"|zeta| in the shell (max {res['curl_max']:.2f}, "
                f"limit {args.zeta:g})", fontsize=11)
    plt.colorbar(im, ax=a, fraction=0.03)

    fig.suptitle("Broadband printable cloak of a through-hole "
                 f"(worst-case over omega={omegas}, zeta_all={args.zeta:g})",
                 y=0.995, fontsize=14)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)
    np.savez(args.out.replace(".png", ".npz"), th=th, z=z, omegas=omegas,
             base=base, scat=scat, red=red, curl_max=res['curl_max'])


if __name__ == "__main__":
    main()
