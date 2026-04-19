"""
Classic cantilever beam topology optimization.

Problem setup
-------------
- Rectangular domain: 2 × 1 (length × height), discretized into 2nx × nx quads.
- Left edge: fully clamped (zero displacement).
- Right edge: distributed downward load (total F = 1).
- Objective: minimize compliance (= maximize stiffness).
- Constraint: volume fraction vf = 0.4.

Method
------
SIMP with Helmholtz PDE filter and Optimality Criteria (OC) update.

Usage
-----
    conda run -n dolfinx_complex python examples/cantilever.py

or inside the dolfinx_complex environment:

    python examples/cantilever.py [--nx 60] [--vf 0.4] [--r-min 0.04] [--max-iter 100]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make the package importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from topoopt import ElasticityProblem, HelmholtzFilter, SIMPOptimizer, OptimizationRecorder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cantilever beam topology optimization")
    p.add_argument("--nx", type=int, default=60,
                   help="Number of elements in y-direction (x gets 2×nx)")
    p.add_argument("--vf", type=float, default=0.4, help="Target volume fraction")
    p.add_argument("--penal", type=float, default=3.0, help="SIMP penalization exponent")
    p.add_argument("--r-min", type=float, default=0.04,
                   help="Filter radius (fraction of beam height)")
    p.add_argument("--max-iter", type=int, default=100, help="Max OC iterations")
    p.add_argument("--tol", type=float, default=1e-3, help="Convergence tolerance")
    p.add_argument("--no-plot", action="store_true", help="Skip matplotlib output")
    p.add_argument("--save-vtk", action="store_true",
                   help="Save final density to VTK (requires pyvista or meshio)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    nx = args.nx          # elements in y
    ny = 2 * nx           # elements in x  (2:1 aspect ratio)
    lx, ly = 2.0, 1.0    # domain dimensions

    print("=" * 60)
    print("Cantilever Beam Topology Optimization")
    print("=" * 60)
    print(f"  Mesh        : {ny} × {nx}  ({ny*nx} elements)")
    print(f"  Volume frac : {args.vf}")
    print(f"  SIMP penal  : {args.penal}")
    print(f"  Filter r_min: {args.r_min}")
    print(f"  Max iters   : {args.max_iter}")
    print("=" * 60)

    # --- Build FE problem ---------------------------------------------------
    fem_prob = ElasticityProblem(
        nx=ny, ny=nx,          # dolfinx: nx = cols, ny = rows
        lx=lx, ly=ly,
        penal=args.penal,
    )

    # --- Build filter -------------------------------------------------------
    filt = HelmholtzFilter(fem_prob.DG0, r_min=args.r_min)

    # --- Define the fem_solve callable expected by SIMPOptimizer ------------
    def fem_solve(rho: np.ndarray) -> tuple[float, np.ndarray]:
        """Run FE solve + sensitivity, with filter chain rule."""
        # Filter density before updating FE
        rho_tilde = filt.apply(rho)
        np.clip(rho_tilde, 1e-3, 1.0, out=rho_tilde)

        fem_prob.update_density(rho_tilde)
        u, compliance = fem_prob.solve()

        # Raw element sensitivity dC/d(rho_tilde)
        dc_tilde = fem_prob.compliance_sensitivity(u)

        # Chain rule: dC/drho = dC/d(rho_tilde) * d(rho_tilde)/drho
        dc = filt.sensitivity_chain(dc_tilde, rho)

        return compliance, dc

    # --- Run optimizer ------------------------------------------------------
    opt = SIMPOptimizer(
        n_elem=ny * nx,
        vf=args.vf,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    t0 = time.perf_counter()

    recorder = OptimizationRecorder(nx=ny, ny=nx, out_dir="results")
    result = opt.optimize(fem_solve, callback=recorder.callback)

    elapsed = time.perf_counter() - t0

    print("=" * 60)
    status = "CONVERGED" if result.converged else "MAX ITER"
    print(f"  Status      : {status}")
    print(f"  Iterations  : {result.n_iterations}")
    print(f"  Final C     : {result.compliance_history[-1]:.6f}")
    print(f"  Final vol   : {result.volume_history[-1]:.4f}")
    print(f"  Wall time   : {elapsed:.1f} s")
    print("=" * 60)

    # --- Save PNGs ----------------------------------------------------------
    if not args.no_plot:
        recorder.save_all(
            final_rho=result.densities,
            final_C=result.compliance_history[-1],
        )

    if args.save_vtk:
        _save_vtk(fem_prob, result.densities)


def _save_vtk(fem_prob: ElasticityProblem, rho: np.ndarray) -> None:
    """Write final density to a VTK file via dolfinx XDMF."""
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
