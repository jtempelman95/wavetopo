"""
Curvilinear-fiber elastic-wave lens.

A single-material composite plate, driven by a time-harmonic pressure wave from
the left edge.  Only the FIBER ORIENTATION field is optimized so the wave
focuses to a target point on the right -- a manufacturable graded-index lens
made of continuous curved fibers.  Optionally limits fiber curvature (curl) so
the toolpaths stay printable.

    python examples/wave_lens.py --nx 120 --omega 40 --iters 70
"""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topoopt.cfrp import Material, QuadMesh, grid_support_points, CSRBFMapping
from topoopt.harmonic import HarmonicLens
from topoopt.cfrp_optimizer import MMA
from topoopt.cfrp_viz import orientation_streamlines


def build(nx, ny, Lx, Ly, omega, eta, support_spc, r_s):
    mesh = QuadMesh(nx, ny, Lx, Ly)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    sx, sy = grid_support_points(mesh, support_spc, support_spc)
    rbf = CSRBFMapping(mesh, (sx, sy), r_s=r_s)
    hl = HarmonicLens(mesh, mat, rbf, omega=omega, eta=eta)

    # plane pressure wave: x-force on the left edge
    F = np.zeros(mesh.ndof, dtype=complex)
    for iy in range(ny + 1):
        n = hl.node_id(0, iy)
        F[2 * n] = 1.0
    hl.set_load(F)

    # absorbing sponge near right / top / bottom (not the left source edge)
    marg = 0.6
    c = np.zeros(mesh.N)
    d = np.maximum.reduce([
        (mesh.cx - (Lx - marg)) / marg,
        (marg - mesh.cy) / marg,
        (mesh.cy - (Ly - marg)) / marg])
    c = np.maximum(0.0, d) ** 2 * 80.0
    hl.set_sponge(c)

    # focus target: small box at (0.78 Lx, Ly/2)
    fx, fy, fr = 0.78 * Lx, Ly / 2, 0.18
    w = np.zeros(mesh.ndof)
    xs, ys = mesh.node_xy()
    for n in range(mesh.nnode):
        if abs(xs[n] - fx) < fr and abs(ys[n] - fy) < fr:
            w[2 * n] = 1.0
            w[2 * n + 1] = 1.0
    hl.set_focus(w)
    return mesh, hl, (fx, fy, fr)


def nodal_mag(u, mesh):
    ux = u[0::2]; uy = u[1::2]
    mag = np.sqrt(np.abs(ux) ** 2 + np.abs(uy) ** 2)
    return mag.reshape(mesh.nely + 1, mesh.nelx + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--omega", type=float, default=40.0)
    ap.add_argument("--iters", type=int, default=70)
    ap.add_argument("--out", default="results/wave_lens.png")
    args = ap.parse_args()

    Lx, Ly = 4.0, 3.0
    ny = int(round(args.nx * Ly / Lx))
    mesh, hl, focus = build(args.nx, ny, Lx, Ly, args.omega, eta=0.03,
                            support_spc=0.33, r_s=0.7)
    print(f"mesh {mesh.nelx}x{mesh.nely}={mesh.N}  M={hl.M_} support pts  "
          f"omega={args.omega}")

    # baseline: uniform horizontal fibers (no lens)
    th = np.zeros(hl.M_)
    u0 = hl.solve(th)
    J0 = hl.focus_energy()
    print(f"baseline focus energy J0={J0:.4e}")

    # optimize orientation to maximize focus energy
    xmin = np.full(hl.M_, -np.pi / 2)
    xmax = np.full(hl.M_, np.pi / 2)
    mma = MMA(xmin, xmax, move=0.1)
    hist = []
    t0 = time.time()
    best = (-np.inf, th.copy())
    for it in range(args.iters):
        J = hl.focus_energy(th)
        g = hl.focus_grad()
        hist.append(J)
        if J > best[0]:
            best = (J, th.copy())
        if it % 5 == 0 or it == args.iters - 1:
            print(f"[{it:3d}] focus energy J={J:.4e}  gain={J/J0:.2f}x")
        th = mma.update(th, -g)        # minimize -J
    th = best[1]
    print(f"optimized in {time.time()-t0:.0f}s  best gain={best[0]/J0:.2f}x")

    uopt = hl.solve(th)
    Jopt = hl.focus_energy()

    # ---- figure ----
    m0 = nodal_mag(u0, mesh)
    m1 = nodal_mag(uopt, mesh)
    vmax = np.percentile(m1, 99.5)
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fx, fy, fr = focus
    for a_, m, ttl in [(ax[0, 0], m0, f"baseline |u| (straight fibers)  J={J0:.2e}"),
                       (ax[0, 1], m1, f"optimized |u| (fiber lens)  J={Jopt:.2e}  "
                                      f"gain={Jopt/J0:.1f}x")]:
        im = a_.imshow(m, origin="lower", cmap="magma", vmax=vmax,
                       extent=[0, Lx, 0, Ly], aspect="equal")
        a_.add_patch(plt.Rectangle((fx - fr, fy - fr), 2 * fr, 2 * fr,
                                   ec="cyan", fc="none", lw=1.5))
        a_.set_title(ttl); plt.colorbar(im, ax=a_, fraction=0.035)

    orientation_streamlines(ax[1, 0], mesh, hl._state['theta'],
                            np.ones(mesh.N), dens_thresh=-1, density_bg=False,
                            color="k", linewidth=0.8)
    ax[1, 0].add_patch(plt.Rectangle((fx - fr, fy - fr), 2 * fr, 2 * fr,
                                     ec="cyan", fc="none", lw=1.5))
    ax[1, 0].set_title("optimized fiber toolpaths (the lens)")

    ax[1, 1].plot(np.array(hist) / J0, "b-")
    ax[1, 1].set_xlabel("iteration"); ax[1, 1].set_ylabel("focus gain  J/J0")
    ax[1, 1].set_title("convergence"); ax[1, 1].grid(alpha=0.3)
    fig.suptitle("Curvilinear-fiber elastic-wave lens "
                 f"(single material, orientation-only, omega={args.omega})",
                 y=1.0, fontsize=14)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
