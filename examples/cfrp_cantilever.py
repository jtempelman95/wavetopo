"""
Toolpath-integrated CFRP cantilever (Section 8.1 of Wong et al. 2026).

Reproduces the no-void cantilever: compliance and max-curl with and without
the curl constraint zeta_all = 2 /m (Table 3 trend: f 0.42 -> 0.49,
|zeta|max 8.04 -> 2.0).

Material moduli are entered as bare GPa numbers (131, 9, 5, 2.6), load 0.5 N,
lengths in metres, thickness 1 -- the unit convention that makes the paper's
O(0.1-1) compliance values consistent.

Usage:
    python examples/cfrp_cantilever.py --nely 50 --curl none
    python examples/cfrp_cantilever.py --nely 150 --curl 2.0 --save fig.png
"""
import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topoopt.cfrp import Material, QuadMesh, grid_support_points
from topoopt.cfrp_problem import CFRPProblem
from topoopt.cfrp_optimizer import optimize_cfrp
from topoopt.cfrp_viz import plot_design


def save_npz(path, mesh, res):
    np.savez(path, z=res['z'], theta=res['theta'], theta_hat=res['theta_hat'],
             chi_hat=res['chi_hat'], curl=res['curl'], f=res['f'],
             curl_max=res['curl_max'], nelx=mesh.nelx, nely=mesh.nely,
             Lx=mesh.Lx, Ly=mesh.Ly)
    print("saved", path)


def build_cantilever(nely, d, beta, vf_all=0.25):
    L, H = 2.4, 1.5
    nelx = int(round(nely * L / H))
    mesh = QuadMesh(nelx, nely, L, H)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3)
    sx, sy = grid_support_points(mesh, 0.16, 0.167)   # ~160 support points
    prob = CFRPProblem(mesh, mat, (sx, sy), R=0.075, r_s=0.33, d=d,
                       beta=beta, eta=0.5, eta_alpha=0.5,
                       v_all=0.5, vf_all=vf_all)
    # clamp left edge
    fixed = []
    for iy in range(nely + 1):
        n = prob.node_id(0, iy)
        fixed += [2 * n, 2 * n + 1]
    prob.set_fixed_dofs(fixed)
    # downward point load at middle of right edge
    F = np.zeros(mesh.ndof)
    n = prob.node_id(nelx, nely // 2)
    F[2 * n + 1] = -0.5
    prob.set_load(F)
    return mesh, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nely", type=int, default=50)
    ap.add_argument("--d", type=float, default=0.06)
    ap.add_argument("--beta", type=float, default=100.0)
    ap.add_argument("--curl", default="none",
                    help="'none' or a float curl limit, e.g. 2.0")
    ap.add_argument("--max-outer", type=int, default=120)
    ap.add_argument("--vf", type=float, default=0.25)
    ap.add_argument("--save", default=None)
    ap.add_argument("--save-npz", default=None)
    args = ap.parse_args()

    zeta_all = None if args.curl == "none" else float(args.curl)
    mesh, prob = build_cantilever(args.nely, args.d, args.beta, vf_all=args.vf)
    print(f"mesh {mesh.nelx}x{mesh.nely} = {mesh.N} elems, "
          f"M={prob.M} support pts, Fourier fit err={prob.fs.fit_err:.1e}, "
          f"dx={mesh.dx:.4f} wavelength d={args.d} ({args.d/mesh.dx:.1f} elems)")

    z0 = np.full(mesh.N, 0.5)
    th0 = np.zeros(prob.M)

    t0 = time.time()
    res = optimize_cfrp(prob, z0, th0, zeta_all=zeta_all,
                        max_outer=args.max_outer, mma_iter=5)
    dt = time.time() - t0
    print(f"\nDONE in {dt:.1f}s  ({res['outer_iters']} outer, "
          f"{res['inner_iters']} inner)")
    print(f"  f = {res['f']:.4f}   |zeta|max = {res['curl_max']:.3f} /m   "
          f"converged={res['converged']}")

    if args.save_npz:
        save_npz(args.save_npz, mesh, res)
    if args.save:
        ttl = (f"CFRP cantilever {mesh.nelx}x{mesh.nely}  "
               + ("no curl constraint" if zeta_all is None
                  else f"curl<={zeta_all}/m"))
        plot_design(mesh, res, args.save, ttl)


if __name__ == "__main__":
    main()
