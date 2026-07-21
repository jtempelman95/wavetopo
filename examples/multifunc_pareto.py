r"""
Does co-design buy stiffness AND gap, or just trade one for the other?

A single weighted run cannot answer this.  Ours converged to C_bulk = 4.80 with
a gap of +0.63 from a seed at C_bulk = 6.97 with NO gap -- i.e. it bought the gap
by spending 31% of the stiffness.  That is a point on a trade-off curve, not
evidence of joint improvement, and quoting it as "stiff AND gapped" is wrong.

The claim only means something as a comparison BETWEEN design freedoms at
matched volume:

    at the SAME gap, does adding the orientation field buy more stiffness
    than geometry alone can reach?

So sweep the stiffness/gap weighting for both design freedoms and trace the two
Pareto fronts in the (gap, C_bulk) plane.  If the co-design front lies outside
the geometry-only front, orientation is buying something geometry cannot.  If the
fronts coincide, it is not, and we say so.

Each invocation runs ONE (mode, w_gap) point so several can run in parallel:

    PYTHONPATH=. python examples/multifunc_pareto.py --mode co   --wgap 1.0
    PYTHONPATH=. python examples/multifunc_pareto.py --mode geom --wgap 1.0
"""
import argparse
import json
import os
import time

import numpy as np

from wavetopo.cfrp import Material
from examples.multifunc_finite import (MAT, codesign, tile_panel,
                                       static_stiffness)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["co", "geom"], required=True)
    ap.add_argument("--wgap", type=float, required=True)
    ap.add_argument("--wstiff", type=float, default=1.0)
    ap.add_argument("--gap-ref", dest="gap_ref", type=float, default=0.30)
    ap.add_argument("--C-ref", dest="C_ref", type=float, default=7.0)
    ap.add_argument("--c-vol", dest="c_vol", type=float, default=1500.0)
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--band", type=int, default=3)
    ap.add_argument("--nk", type=int, default=5)
    ap.add_argument("--R", type=float, default=0.13)
    ap.add_argument("--vol", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=70)
    ap.add_argument("--Nx", type=int, default=6)
    ap.add_argument("--Ny", type=int, default=4)
    ap.add_argument("--nbuf", type=int, default=28)
    ap.add_argument("--outdir", default="results/pareto")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    res = codesign(args, geom_only=(args.mode == "geom"))
    n = args.n

    # Static stiffness of the truncated panel, and of a SOLID panel of identical
    # size -- without that reference k_eff is a bare number with no scale.
    mat = Material(**MAT)
    mesh, z, th = tile_panel(res['y'], res['theta_elem'], n,
                             args.Nx, args.Ny, args.nbuf)
    k_panel = static_stiffness(mesh, mat, z, th)['k_eff']
    k_solid = static_stiffness(mesh, mat, np.ones_like(z),
                               np.zeros_like(th))['k_eff']

    row = dict(mode=args.mode, wgap=args.wgap, wstiff=args.wstiff,
               C_bulk=float(res['C_bulk']), gap=float(res['gap']),
               gap_lo=float(res['gap_lo']), gap_hi=float(res['gap_hi']),
               vol=float(res['vol']), k_panel=float(k_panel),
               k_solid=float(k_solid), k_frac=float(k_panel / k_solid),
               iters=args.iters, secs=round(time.time() - t0, 1))
    tag = f"{args.mode}_wg{args.wgap:g}"
    with open(os.path.join(args.outdir, tag + ".json"), "w") as f:
        json.dump(row, f, indent=1)
    np.savez(os.path.join(args.outdir, tag + ".npz"),
             y=res['y'], theta_elem=res['theta_elem'], bands=res['bands'],
             **{k: v for k, v in row.items() if not isinstance(v, str)})
    print(f"[{tag}] C_bulk={row['C_bulk']:.3f} gap={row['gap']:+.4f} "
          f"vol={row['vol']:.3f} k_panel={k_panel:.3f} "
          f"({100*row['k_frac']:.1f}% of solid) {row['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
