r"""
Stiffness-gap trade-off by CONSTRAINED gradient optimization with continuation.

    maximize   C_bulk(z, theta)
    subject to gap(z, theta) >= g_target
               vol = v_target,   0 <= z <= 1,  |theta| <= pi/2

and trace the front by tightening g_target, warm-starting each solve from the
previous one.

Why this rather than a weighted sweep
-------------------------------------
Both are gradient-based -- MMA on analytic sensitivities either way.  What
changes is the parameterization, and it matters for three reasons.

1. `w_gap = 1.5` is not a quantity anyone wants.  `gap >= 0.4` is.  The
   constraint level is the engineering variable, so the front comes out indexed
   by something interpretable.

2. Scalarizing needs normalizers, and those are exactly what failed before:
   `gap_ref` inferred from a seed whose gap is badly CLOSED (-3.36) made the gap
   term go numerically deaf once the gap opened, converging to a useless
   +0.012.  A constraint has no normalizer to get wrong.

3. Independent runs from one seed at different weights are not a front.  They
   are N local optima that can land in different basins, giving a ragged or
   even non-monotone curve.  Continuation warm-starts each point from its
   neighbour, so the sequence follows one branch and is monotone by
   construction -- and it is cheaper, since each solve starts near its answer.

The augmented Lagrangian is the standard one (Section 14): with
g = g_target - gap and h = max(g, -lambda/mu),

    L = -C/C_scale + lambda*h + (mu/2)h^2 + c_vol (vol - v*)^2

lambda <- max(lambda + mu*h, 0) after each outer subproblem, and mu grows only
while the constraint is violated (inflating it after feasibility just
ill-conditions the subproblem).

    PYTHONPATH=. python examples/multifunc_front.py --mode co
    PYTHONPATH=. python examples/multifunc_front.py --mode geom
"""
import argparse
import json
import os
import time

import numpy as np

from wavetopo.bloch import ibz_path_square
from wavetopo.cfrp_optimizer import MMA
from wavetopo.multifunc_opt import MultiFunctionalOptimizer
from examples.multifunc_finite import build_cell


def solve_at_target(opt, z, th, g_target, *, geom_only, n_outer, n_inner,
                    C_scale, v_target, c_vol, lam0, mu0, xi, mu_max, move,
                    beta_of):
    """One constrained solve at a fixed gap target, warm-started from (z, th)."""
    N, M = opt.bp.mesh.N, opt.bp.M
    if geom_only:
        mma = MMA(np.zeros(N), np.ones(N), move=move, adapt=True)
    else:
        mma = MMA(np.concatenate([np.zeros(N), np.full(M, -np.pi / 2)]),
                  np.concatenate([np.ones(N), np.full(M, np.pi / 2)]),
                  move=move, adapt=True)
    lam, mu = lam0, mu0
    best = dict(C=-np.inf)
    it = 0
    for outer in range(n_outer):
        for _ in range(n_inner):
            opt.beta = beta_of(it); it += 1
            c = opt.components(z, th)
            g = g_target - c['gap']                     # <= 0 is feasible
            h = max(g, -lam / mu)
            dv = c['vol'] - v_target
            L = (-c['C'] / C_scale + lam * h + 0.5 * mu * h * h
                 + c_vol * dv * dv)
            act = 1.0 if g > -lam / mu else 0.0
            coef = (lam + mu * h) * act
            dz = (-c['dC_dz'] / C_scale - coef * c['dgap_dz']
                  + 2.0 * c_vol * dv * c['dvol_dz'])
            dt = (-c['dC_dth'] / C_scale - coef * c['dgap_dth'])
            feas = (c['gap'] >= g_target - 5e-3) and (abs(dv) < 0.03)
            if feas and c['C'] > best['C']:
                best = dict(C=c['C'], gap=c['gap'], vol=c['vol'],
                            z=z.copy(), th=th.copy(), y=c['y'].copy())
            if geom_only:
                z = mma.update(z, dz, f=L)
            else:
                W = mma.update(np.concatenate([z, th]),
                               np.concatenate([dz, dt]), f=L)
                z, th = W[:N], W[N:]
        lam = max(lam + mu * h, 0.0)
        if g > 0:                       # grow mu ONLY while violated
            mu = min(xi * mu, mu_max)
    return best, z, th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["co", "geom"], required=True)
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.0, 0.15, 0.30, 0.45, 0.60, 0.75])
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--band", type=int, default=3)
    ap.add_argument("--nk", type=int, default=5)
    ap.add_argument("--R", type=float, default=0.13)
    ap.add_argument("--vol", type=float, default=0.5)
    ap.add_argument("--c-vol", dest="c_vol", type=float, default=1500.0)
    ap.add_argument("--C-scale", dest="C_scale", type=float, default=7.0)
    ap.add_argument("--n-outer", dest="n_outer", type=int, default=6)
    ap.add_argument("--n-inner", dest="n_inner", type=int, default=8)
    ap.add_argument("--lam0", type=float, default=0.0)
    ap.add_argument("--mu0", type=float, default=20.0)
    ap.add_argument("--xi", type=float, default=1.4)
    ap.add_argument("--mu-max", dest="mu_max", type=float, default=1e5)
    ap.add_argument("--move", type=float, default=0.08)
    ap.add_argument("--beta-max", dest="beta_max", type=float, default=16.0)
    ap.add_argument("--out", default="results/front")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    a = 1.0
    mesh, mat, rbf, bp = build_cell(args.n, args.ks, a)
    kset, _, _ = ibz_path_square(a, a, n=args.nk)
    nb = args.band + 3
    opt = MultiFunctionalOptimizer(bp, band=args.band, n_bands=nb, R=args.R,
                                   kset=kset, v_target=args.vol,
                                   c_vol=args.c_vol, period=(a, a),
                                   C_ref=args.C_scale, gap_ref=1.0)

    rng = np.random.default_rng(0)
    z = np.clip(args.vol + 0.10 * rng.standard_normal(mesh.N), 0.15, 0.95)
    r2 = (mesh.cx - a / 2) ** 2 + (mesh.cy - a / 2) ** 2
    z[r2 < (0.17 * a) ** 2] = 0.95
    th = 0.05 * rng.standard_normal(rbf.M)

    # Beta (projection sharpness) must be IDENTICAL at every point of the
    # front.  Ramping it across the continuation -- as a first version did --
    # runs late targets at crisper projection than early ones, and the front
    # then RISES with the constraint: C_bulk 4.68 -> 7.25 as the gap target
    # tightened 0 -> 0.60.  That is impossible for a genuine trade-off curve
    # (tightening a constraint cannot raise the optimum), and is the tell that
    # the curve was tracking beta rather than the physics.  So: ramp beta during
    # a warm-up at the loosest target, then hold it fixed for every front point.
    warm = args.n_outer * args.n_inner
    done = [0]

    def beta_of(_):
        i = done[0]; done[0] += 1
        if i >= warm:
            return args.beta_max
        f = i / max(warm - 1, 1)
        return 1.0 if f < 0.25 else min(args.beta_max,
                                        args.beta_max ** ((f - 0.25) / 0.75))

    # warm-up: ramp beta to beta_max at the loosest target, discard the result,
    # then trace the front with beta pinned at beta_max throughout
    print(f"warm-up at gap>={args.targets[0]:.2f} (beta 1 -> {args.beta_max:g})",
          flush=True)
    _, z, th = solve_at_target(
        opt, z, th, args.targets[0], geom_only=(args.mode == "geom"),
        n_outer=args.n_outer, n_inner=args.n_inner, C_scale=args.C_scale,
        v_target=args.vol, c_vol=args.c_vol, lam0=args.lam0, mu0=args.mu0,
        xi=args.xi, mu_max=args.mu_max, move=args.move, beta_of=beta_of)

    rows = []
    for gt in args.targets:
        t0 = time.time()
        best, z, th = solve_at_target(
            opt, z, th, gt, geom_only=(args.mode == "geom"),
            n_outer=args.n_outer, n_inner=args.n_inner, C_scale=args.C_scale,
            v_target=args.vol, c_vol=args.c_vol, lam0=args.lam0, mu0=args.mu0,
            xi=args.xi, mu_max=args.mu_max, move=args.move, beta_of=beta_of)
        if not np.isfinite(best['C']):
            print(f"  target gap>={gt:.2f}: INFEASIBLE (no feasible iterate)",
                  flush=True)
            continue
        rows.append(dict(mode=args.mode, target=gt, C=best['C'],
                         gap=best['gap'], vol=best['vol'],
                         secs=round(time.time() - t0, 1)))
        print(f"  target gap>={gt:.2f}:  C_bulk={best['C']:.3f}  "
              f"gap={best['gap']:+.4f}  vol={best['vol']:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        np.savez(os.path.join(args.out, f"{args.mode}_g{gt:.2f}.npz"),
                 y=best['y'], z=best['z'], th=best['th'],
                 C=best['C'], gap=best['gap'], vol=best['vol'], target=gt)
        # warm start the next target from this solution
        z, th = best['z'].copy(), best['th'].copy()

    with open(os.path.join(args.out, f"{args.mode}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"saved {args.out}/{args.mode}.json", flush=True)


if __name__ == "__main__":
    main()
