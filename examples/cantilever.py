"""
Classic cantilever beam topology optimization.

Problem setup
-------------
- Rectangular domain: 2 × 1 (length × height), discretized into 2nx × nx quads.
- Left edge: fully clamped (zero displacement).
- Right edge midpoint: concentrated downward load (total F = 1).
- Objective: minimize compliance C = F · u  (maximize stiffness).
- Constraint: volume fraction = vf (default 0.4).

Method
------
SIMP (Solid Isotropic Material with Penalization) with:
  - Cone (weighted-average) density filter — prevents checkerboard and
    disconnected elements.  Filter radius r_min must satisfy r_min >= 2*h
    where h = element size.
  - Optimality Criteria (OC) density update.

Usage
-----
    conda run -n dolfinx_complex python examples/cantilever.py

or inside the dolfinx_complex environment:

    python examples/cantilever.py [--nx 60] [--vf 0.4] [--r-min 0.06] [--max-iter 100]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from topoopt import ElasticityProblem, ConeFilter, OptimizationRecorder
from topoopt.optimizer import OptimizationResult


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cantilever beam topology optimization")
    p.add_argument("--nx", type=int, default=60,
                   help="Number of elements in y-direction (x gets 2×nx)")
    p.add_argument("--vf", type=float, default=0.4, help="Target volume fraction")
    p.add_argument("--penal", type=float, default=3.0, help="SIMP penalization exponent")
    p.add_argument("--r-min", type=float, default=None,
                   help="Filter radius (default: 1.5 × element size)")
    p.add_argument("--max-iter", type=int, default=100, help="Max OC iterations")
    p.add_argument("--tol", type=float, default=1e-3, help="Convergence tolerance")
    p.add_argument("--load", choices=["midpoint", "distributed"], default="midpoint",
                   help="Load type: midpoint (classic) or distributed")
    p.add_argument("--no-plot", action="store_true", help="Skip matplotlib output")
    p.add_argument("--save-vtk", action="store_true",
                   help="Save final density to XDMF/VTK")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    nx = args.nx          # elements in y
    ny = 2 * nx           # elements in x  (2:1 aspect ratio)
    lx, ly = 2.0, 1.0

    # Default r_min = 1.5 × element height (standard Andreassen 2011 convention)
    h = ly / nx
    r_min = args.r_min if args.r_min is not None else 1.5 * h

    print("=" * 60)
    print("Cantilever Beam Topology Optimization")
    print("=" * 60)
    print(f"  Mesh        : {ny} × {nx}  ({ny*nx} elements, h = {h:.4f})")
    print(f"  Load type   : {args.load}")
    print(f"  Volume frac : {args.vf}")
    print(f"  SIMP penal  : {args.penal}")
    print(f"  Filter r_min: {r_min:.4f}  (= {r_min/h:.1f}×h)")
    print(f"  Max iters   : {args.max_iter}")
    print("=" * 60)

    # --- Build FE problem ---------------------------------------------------
    fem_prob = ElasticityProblem(
        nx=ny, ny=nx,
        lx=lx, ly=ly,
        penal=args.penal,
        load_type=args.load,
    )

    # --- Build cone filter --------------------------------------------------
    filt = ConeFilter(fem_prob.DG0, r_min=r_min)

    # --- Optimization loop (top88 algorithm, Andreassen 2011) ---------------
    #
    # Design variables x (= rho in the paper) are the RAW (unfiltered) densities.
    # Physical densities xPhys = M @ x are passed to the FE solver.
    # The OC update formula uses xPhys (not x) as the scaling base:
    #
    #   x_new = clip( xPhys * sqrt(|dc_filt| / λ),
    #                 xPhys - move,  xPhys + move )
    #
    # Then xPhys_new = M @ x_new.
    #
    # This is critical: a void element (xPhys ≈ 0) stays void regardless of
    # its sensitivity because the scaling factor is xPhys itself.  Without this,
    # isolated void elements near structural members can flip to solid and create
    # the "speckled" scatter pattern.
    #
    # Sensitivity filter (weighted, Sigmund 2001):
    #   dc_filt_e = Σ_f H_ef * xPhys_f * dc_f  /  (Hs_e * max(ε, xPhys_e))
    # In matrix form: dc_filt = M @ (xPhys * dc) / max(ε, xPhys)

    n_elem = ny * nx
    vf = args.vf
    move = 0.2
    tol = args.tol

    x = np.full(n_elem, vf)                   # design variable (unfiltered)
    xPhys = filt.apply(x)                      # physical density

    result = OptimizationResult(densities=x.copy())
    t0 = time.perf_counter()
    recorder = OptimizationRecorder(
        nx=ny, ny=nx, out_dir="results",
        centroids=filt.centroids, lx=lx, ly=ly,
    )

    for it in range(1, args.max_iter + 1):
        x_old = x.copy()

        # FE solve with physical density
        xPhys_clipped = np.clip(xPhys, 1e-3, 1.0)
        fem_prob.update_density(xPhys_clipped)
        u, compliance = fem_prob.solve()

        # Raw sensitivity w.r.t. physical density
        dc_raw = fem_prob.compliance_sensitivity(u)

        # Weighted sensitivity filter (void elements contribute zero)
        dc = filt.apply(xPhys_clipped * dc_raw) / np.maximum(1e-3, xPhys_clipped)

        vol = xPhys.mean()
        result.compliance_history.append(compliance)
        result.volume_history.append(vol)
        recorder.callback(it, compliance, vol, xPhys)

        # OC bisection (Lagrange multiplier for volume constraint)
        dc_abs = np.maximum(-dc, 1e-30)
        l1, l2 = 1e-9, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-6:
            lmid = 0.5 * (l1 + l2)
            # Note: scale by xPhys, move limits relative to xPhys
            x_new = np.clip(
                xPhys * (dc_abs / lmid) ** 0.5,
                np.maximum(0.001, xPhys - move),
                np.minimum(1.0,   xPhys + move),
            )
            if x_new.mean() > vf:
                l1 = lmid
            else:
                l2 = lmid

        x = x_new
        xPhys = filt.apply(x)   # re-filter to get new physical density

        change = np.max(np.abs(x - x_old))
        if change < tol and it > 5:
            result.converged = True
            break

    result.densities = xPhys.copy()   # report physical densities
    result.n_iterations = it
    elapsed = time.perf_counter() - t0

    print("=" * 60)
    status = "CONVERGED" if result.converged else "MAX ITER"
    print(f"  Status      : {status}")
    print(f"  Iterations  : {result.n_iterations}")
    print(f"  Final C     : {result.compliance_history[-1]:.6f}")
    print(f"  Final vol   : {result.volume_history[-1]:.4f}")
    print(f"  Wall time   : {elapsed:.1f} s")
    print("=" * 60)

    if not args.no_plot:
        recorder.save_all(
            final_rho=result.densities,
            final_C=result.compliance_history[-1],
        )

    if args.save_vtk:
        _save_vtk(fem_prob, result.densities)


def _save_vtk(fem_prob: ElasticityProblem, rho: np.ndarray) -> None:
    try:
        from dolfinx.io import XDMFFile
        from mpi4py import MPI
        from dolfinx import fem

        rho_fn = fem.Function(fem_prob.DG0, name="density")
        rho_fn.x.array[:] = rho
        with XDMFFile(MPI.COMM_WORLD, "cantilever_density.xdmf", "w") as f:
            f.write_mesh(fem_prob.domain)
            f.write_function(rho_fn)
        print("  VTK saved   : cantilever_density.xdmf")
    except Exception as exc:
        print(f"  VTK export failed: {exc}")


if __name__ == "__main__":
    main()
