"""
Elastic cloak via toolpath anisotropy.

A fiber-composite plate contains a void that scatters an incident plane wave
(casting a shadow + reflections).  We design the fiber ORIENTATION in a shell
around the void so the field outside the shell matches the void-free reference
-- the void becomes ~invisible.  Single material; only the fiber toolpaths
(which curve around the void) do the work.

    python examples/wave_cloak.py --nx 130 --omega 36 --iters 60
"""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topoopt.cfrp import Material, QuadMesh, grid_support_points, CSRBFMapping
from topoopt.harmonic import HarmonicCloak
from topoopt.cfrp_optimizer import MMA
from topoopt.cfrp_viz import orientation_streamlines


def nodal_mag(u, mesh):
    ux = u[0::2]; uy = u[1::2]
    return np.sqrt(np.abs(ux)**2 + np.abs(uy)**2).reshape(mesh.nely + 1,
                                                          mesh.nelx + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=130)
    ap.add_argument("--omega", type=float, default=36.0)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--rv", type=float, default=0.55)
    ap.add_argument("--rc", type=float, default=1.4)
    ap.add_argument("--out", default="results/wave_cloak.png")
    args = ap.parse_args()

    Lx, Ly = 6.0, 4.0
    ny = int(round(args.nx * Ly / Lx))
    mesh = QuadMesh(args.nx, ny, Lx, Ly)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    sx, sy = grid_support_points(mesh, 0.28, 0.28)
    rbf = CSRBFMapping(mesh, (sx, sy), r_s=0.6)
    cl = HarmonicCloak(mesh, mat, rbf, omega=args.omega, eta=0.03)
    cx, cy = Lx / 2, Ly / 2

    # void density
    z = np.ones(mesh.N)
    z[np.hypot(mesh.cx - cx, mesh.cy - cy) < args.rv] = 0.0

    # plane pressure wave from the left
    F = np.zeros(mesh.ndof, dtype=complex)
    for iy in range(ny + 1):
        n = cl.node_id(0, iy); F[2 * n] = 1.0
    cl.set_load(F)
    # absorbing rim (open domain), leave left source edge
    marg = 0.55
    d = np.maximum.reduce([(mesh.cx - (Lx - marg)) / marg,
                           (marg - mesh.cy) / marg,
                           (mesh.cy - (Ly - marg)) / marg])
    cl.set_sponge(np.maximum(0, d)**2 * 70.0)

    # reference field: NO void
    cl.set_density(np.ones(mesh.N))
    u_ref = cl.solve(np.zeros(rbf.M))

    # observation weight: everywhere outside the cloak shell (and inside the
    # non-sponge region) -> match the void-free reference there
    xs, ys = mesh.node_xy()
    rr = np.hypot(xs - cx, ys - cy)
    obs = (rr > args.rc) & (xs > marg) & (xs < Lx - marg) \
        & (ys > marg) & (ys < Ly - marg)
    w = np.zeros(mesh.ndof)
    w[0::2][obs] = 1.0; w[1::2][obs] = 1.0

    cl.set_density(z)
    cl.set_target(u_ref, w)

    # uncloaked field (void, uniform fiber)
    u_unc = cl.solve(np.zeros(rbf.M))
    J_unc = cl.objective()

    # design only support points within the cloak shell
    sdist = np.hypot(sx - cx, sy - cy)
    active = sdist < args.rc + 0.3
    xmin = np.where(active, -np.pi / 2, 0.0)
    xmax = np.where(active, np.pi / 2, 0.0)
    mma = MMA(xmin, xmax, move=0.15)
    th = np.zeros(rbf.M)
    print(f"mesh {mesh.nelx}x{mesh.nely}, {int(active.sum())} active support pts, "
          f"uncloaked scatter J={J_unc:.4e}")
    t0 = time.time()
    best = (J_unc, th.copy())
    for it in range(args.iters):
        J = cl.objective(th)
        g = cl.grad()
        if J < best[0]:
            best = (J, th.copy())
        if it % 5 == 0 or it == args.iters - 1:
            print(f"[{it:3d}] scatter J={J:.4e}  reduction={J_unc/J:.2f}x")
        th = mma.update(th, g)
    th = best[1]
    u_cloak = cl.solve(th)
    J_cl = cl.objective()
    print(f"optimized in {time.time()-t0:.0f}s  cloaked J={J_cl:.4e}  "
          f"scatter reduced {J_unc/J_cl:.1f}x")

    # ---- figure ----
    mref = nodal_mag(u_ref, mesh)
    munc = nodal_mag(u_unc, mesh)
    mcl = nodal_mag(u_cloak, mesh)
    vmax = np.percentile(mref, 99.5)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))
    th_circle = np.linspace(0, 2 * np.pi, 100)
    for a_, m, ttl in [(ax[0, 0], mref, "reference: no void"),
                       (ax[0, 1], munc, f"uncloaked void  (scatter {J_unc:.1e})"),
                       (ax[1, 0], mcl, f"CLOAKED  (scatter {J_cl:.1e}, "
                                       f"{J_unc/J_cl:.0f}x less)")]:
        im = a_.imshow(m, origin="lower", cmap="magma", vmax=vmax,
                       extent=[0, Lx, 0, Ly], aspect="equal")
        a_.plot(cx + args.rv * np.cos(th_circle), cy + args.rv * np.sin(th_circle),
                "c-", lw=1)
        a_.set_title(ttl); plt.colorbar(im, ax=a_, fraction=0.03)
    # cloak toolpaths
    theta = rbf.theta(th)
    orientation_streamlines(ax[1, 1], mesh, theta, z, dens_thresh=0.5,
                            density_bg=True, color="tab:red", seed_density=3.0)
    ax[1, 1].plot(cx + args.rv * np.cos(th_circle),
                  cy + args.rv * np.sin(th_circle), "c-", lw=1.5)
    ax[1, 1].plot(cx + args.rc * np.cos(th_circle),
                  cy + args.rc * np.sin(th_circle), "y--", lw=1, alpha=0.7)
    ax[1, 1].set_title("cloak fiber toolpaths (wrap the void)")
    fig.suptitle("Elastic cloak via toolpath anisotropy "
                 f"(single material, orientation-only, omega={args.omega})",
                 y=1.0, fontsize=13)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
