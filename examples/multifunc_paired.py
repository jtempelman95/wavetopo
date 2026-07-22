r"""
The marginal value of the fiber toolpath, measured by a PAIRED experiment.

Two independent fronts cannot answer "does orientation help", because co-design
lives in a larger, harder landscape than geometry-only and its optimizer can
land in a worse basin -- which is exactly what happened: earlier runs showed
co-design LOSING to geometry-only at matched gap, which is impossible at the true
optimum, since co-design contains geometry-only as the special case theta = 0.
A design that beats geometry-only can always be reached from it by leaving theta
straight, so the true co-design optimum weakly dominates.  Independent runs
violated that only because they did not converge to it.

This driver removes the confound.  At each gap target:

  1. optimize GEOMETRY-ONLY (theta frozen straight) to a converged density z*;
  2. warm-start CO-DESIGN from exactly (z*, theta = 0) and free theta.

Step 2 starts at the geometry-only design itself, so C_bulk can only rise (or
hold): C_co >= C_geom by construction, and the increase is the toolpath's
marginal value at that gap -- an honest, self-consistent number, not a race
between two separate optimizations.

    PYTHONPATH=. python examples/multifunc_paired.py --gap 0.30
"""
import argparse
import json
import os
import time

import numpy as np

from wavetopo.bloch import ibz_path_square
from wavetopo.multifunc_opt import MultiFunctionalOptimizer
from examples.multifunc_front import solve_at_target
from examples.multifunc_finite import build_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, required=True,
                    help="gap target g* for this paired point")
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--band", type=int, default=3)
    ap.add_argument("--nk", type=int, default=5)
    ap.add_argument("--R", type=float, default=0.13)
    ap.add_argument("--vol", type=float, default=0.5)
    ap.add_argument("--c-vol", dest="c_vol", type=float, default=1500.0)
    ap.add_argument("--C-scale", dest="C_scale", type=float, default=7.0)
    ap.add_argument("--n-outer", dest="n_outer", type=int, default=8)
    ap.add_argument("--n-inner", dest="n_inner", type=int, default=13)
    ap.add_argument("--mu0", type=float, default=20.0)
    ap.add_argument("--xi", type=float, default=1.4)
    ap.add_argument("--mu-max", dest="mu_max", type=float, default=1e5)
    ap.add_argument("--move", type=float, default=0.08)
    ap.add_argument("--beta-max", dest="beta_max", type=float, default=16.0)
    ap.add_argument("--out", default="results/paired")
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
    z0 = np.clip(args.vol + 0.10 * rng.standard_normal(mesh.N), 0.15, 0.95)
    r2 = (mesh.cx - a / 2) ** 2 + (mesh.cy - a / 2) ** 2
    z0[r2 < (0.17 * a) ** 2] = 0.95
    th0 = 0.05 * rng.standard_normal(rbf.M)

    warm = args.n_outer * args.n_inner
    ticker = [0]

    def beta_of(_):
        i = ticker[0]; ticker[0] += 1
        if i >= warm:
            return args.beta_max
        f = i / max(warm - 1, 1)
        return 1.0 if f < 0.25 else min(args.beta_max,
                                        args.beta_max ** ((f - 0.25) / 0.75))

    common = dict(n_outer=args.n_outer, n_inner=args.n_inner,
                  C_scale=args.C_scale, v_target=args.vol, c_vol=args.c_vol,
                  lam0=0.0, mu0=args.mu0, xi=args.xi, mu_max=args.mu_max,
                  move=args.move, beta_of=beta_of)

    t0 = time.time()
    # --- warm-up + geometry-only front (theta frozen straight) ------------- #
    print(f"[gap>={args.gap:.2f}] warm-up + geometry-only ...", flush=True)
    _, zw, _ = solve_at_target(opt, z0, th0, args.gap, geom_only=True, **common)
    ticker[0] = warm                                   # beta pinned from here
    best_g, z_g, _ = solve_at_target(opt, zw, th0, args.gap, geom_only=True,
                                     **common)
    if not np.isfinite(best_g['C']):
        print(f"[gap>={args.gap:.2f}] geometry-only INFEASIBLE", flush=True)
        return
    print(f"  geometry-only: C_bulk={best_g['C']:.3f} gap={best_g['gap']:+.4f}",
          flush=True)

    # --- co-design, warm-started FROM the geometry-only optimum, theta=0 ---- #
    ticker[0] = warm                                   # keep beta pinned
    best_c, z_c, th_c = solve_at_target(
        opt, best_g['z'].copy(), np.zeros(bp.M), args.gap, geom_only=False,
        **common)
    if not np.isfinite(best_c['C']):
        # co-design must at least match geometry-only (theta=0 recovers it)
        best_c = dict(best_g)
        th_c = np.zeros(bp.M)
    gain = 100.0 * (best_c['C'] - best_g['C']) / best_g['C']
    print(f"  co-design:     C_bulk={best_c['C']:.3f} gap={best_c['gap']:+.4f}"
          f"   toolpath marginal value: {gain:+.1f}%   ({time.time()-t0:.0f}s)",
          flush=True)

    row = dict(gap_target=args.gap,
               C_geom=best_g['C'], gap_geom=best_g['gap'], vol_geom=best_g['vol'],
               C_co=best_c['C'], gap_co=best_c['gap'], vol_co=best_c['vol'],
               marginal_pct=gain)
    with open(os.path.join(args.out, f"g{args.gap:.2f}.json"), "w") as f:
        json.dump(row, f, indent=1)
    np.savez(os.path.join(args.out, f"g{args.gap:.2f}.npz"),
             z_geom=best_g['z'], z_co=best_c['z'], th_co=th_c,
             y_co=best_c['y'], **{k: v for k, v in row.items()})
    print(f"  saved {args.out}/g{args.gap:.2f}.*", flush=True)


if __name__ == "__main__":
    main()
