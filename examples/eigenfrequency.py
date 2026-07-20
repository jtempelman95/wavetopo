"""
Eigenfrequency maximization topology optimization.

Problem setup
-------------
- Rectangular domain: 2 × 1 (length × height), discretized into 2*nx × nx quads.
- Left edge: fully clamped (zero displacement).
- No external load.
- Objective: maximize the sum of the first n_eigs squared natural frequencies
  (eigenvalues lambda_j = omega_j^2).
- Constraint: volume fraction = vf (default 0.4).

Method
------
SIMP with:
  - Cone (weighted-average) density filter.
  - Optimality Criteria (OC) density update adapted for maximization.
  - The sensitivity df/drho is POSITIVE (adding material increases eigenvalues),
    so the OC bisection direction is the same as compliance minimization.

Usage
-----
    conda run -n dolfinx_complex python examples/eigenfrequency.py

or inside the dolfinx_complex environment:

    python examples/eigenfrequency.py [--nx 40] [--vf 0.4] [--n-eigs 5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from wavetopo import ConeFilter
from wavetopo.eigenfrequency import EigenfrequencyProblem
from wavetopo.visualize import density_snapshot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eigenfrequency maximization topology optimization"
    )
    p.add_argument(
        "--nx", type=int, default=40,
        help="Number of elements in y-direction (x gets 2×nx, default 40)",
    )
    p.add_argument("--vf", type=float, default=0.4, help="Target volume fraction")
    p.add_argument("--penal", type=float, default=3.0, help="SIMP penalization exponent")
    p.add_argument(
        "--r-min", type=float, default=None,
        help="Filter radius (default: 1.5 × element size)",
    )
    p.add_argument("--n-eigs", type=int, default=5, help="Number of eigenvalues to maximize")
    p.add_argument("--max-iter", type=int, default=200, help="Max OC iterations")
    p.add_argument("--tol", type=float, default=1e-3, help="Convergence tolerance on design change")
    return p.parse_args()


def save_convergence_plot(
    obj_history: list[float],
    volume_history: list[float],
    out_path: str | Path,
) -> None:
    """Save a two-panel convergence plot: sum-of-eigenvalues + volume fraction."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iters = list(range(1, len(obj_history) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.plot(iters, obj_history, "b-o", markersize=3, linewidth=1.2)
    ax1.set_ylabel(r"$\sum_j \lambda_j$  (sum of eigenvalues)", fontsize=11)
    ax1.set_title("Eigenfrequency Optimization Convergence", fontsize=12)
    ax1.grid(True, alpha=0.35)
    ax1.tick_params(labelsize=9)

    ax2.plot(iters, [v * 100 for v in volume_history], "r-s", markersize=3, linewidth=1.2)
    ax2.axhline(
        volume_history[-1] * 100, color="k", linestyle="--", linewidth=0.8,
        label=f"target {volume_history[-1]*100:.0f}%",
    )
    ax2.set_ylabel("Volume fraction (%)", fontsize=11)
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    args = parse_args()

    ny = args.nx          # elements in y (height)
    nx = 2 * args.nx     # elements in x (length — 2:1 aspect ratio)
    lx, ly = 2.0, 1.0
    vf = args.vf
    n_eigs = args.n_eigs

    # Default r_min = 1.5 × element height
    h = ly / ny
    r_min = args.r_min if args.r_min is not None else 1.5 * h

    print("=" * 60)
    print("Eigenfrequency Maximization Topology Optimization")
    print("=" * 60)
    print(f"  Mesh        : {nx} x {ny}  ({nx*ny} elements, h = {h:.4f})")
    print(f"  Volume frac : {vf}")
    print(f"  SIMP penal  : {args.penal}")
    print(f"  Filter r_min: {r_min:.4f}  (= {r_min/h:.1f}xh)")
    print(f"  Eigenvalues : {n_eigs}")
    print(f"  Max iters   : {args.max_iter}")
    print("=" * 60)

    # --- Build FE eigenfrequency problem ------------------------------------
    fem_prob = EigenfrequencyProblem(
        nx=nx, ny=ny,
        lx=lx, ly=ly,
        penal=args.penal,
        n_eigs=n_eigs,
    )

    # --- Build cone filter --------------------------------------------------
    filt = ConeFilter(fem_prob.DG0, r_min=r_min)

    # --- Optimization setup -------------------------------------------------
    n_elem = nx * ny
    move = 0.2

    x = np.full(n_elem, vf)         # design variables (unfiltered)
    xPhys = filt.apply(x)           # physical (filtered) densities

    # --- Evaluate initial (uniform) design ----------------------------------
    print("\n  Evaluating initial uniform design ...")
    fem_prob.update_density(np.clip(xPhys, 1e-3, 1.0))
    lams0, _ = fem_prob.solve()
    freqs0 = np.sqrt(np.maximum(lams0, 0.0)) / (2.0 * np.pi)
    print(f"  Initial eigenvalues (lambda): {lams0}")
    print(f"  Initial frequencies (Hz):     {freqs0}")
    print()

    # --- OC optimization loop -----------------------------------------------
    obj_history: list[float] = []
    vol_history: list[float] = []
    t0 = time.perf_counter()

    converged = False
    it = 0

    for it in range(1, args.max_iter + 1):
        x_old = x.copy()

        # FE solve with clipped physical density (avoid exact zero)
        xPhys_clipped = np.clip(xPhys, 1e-3, 1.0)
        fem_prob.update_density(xPhys_clipped)
        eigenvalues, eigenvectors = fem_prob.solve()

        n_conv = len(eigenvalues)
        obj = float(eigenvalues.sum())   # maximize sum of eigenvalues
        vol = float(xPhys.mean())

        obj_history.append(obj)
        vol_history.append(vol)

        # Sensitivity: df/d(rho_tilde)_e  — positive = adding material helps
        dc_raw = fem_prob.eigenvalue_sensitivity(eigenvalues, eigenvectors)

        # Apply the cone filter adjoint to the raw sensitivity.
        # We weight by xPhys before filtering (same Sigmund 2001 convention
        # used in cantilever.py) so void elements contribute less sensitivity.
        dc = filt.apply(xPhys_clipped * dc_raw) / np.maximum(1e-3, xPhys_clipped)

        # Print iteration info
        freqs = np.sqrt(np.maximum(eigenvalues, 0.0)) / (2.0 * np.pi)
        freq_str = "  ".join(f"{f:.4f}" for f in freqs)
        print(
            f"  Iter {it:4d} | obj = {obj:12.6f} | vol = {vol:.4f} | "
            f"freqs (Hz): [{freq_str}]"
        )

        # --- OC bisection (Lagrange multiplier for volume constraint) --------
        # Sensitivity is positive → elements with large dc get MORE material.
        # Same bisection logic as compliance (sensitivity already positive):
        #   x_new = xPhys * sqrt(dc / lmid)
        # If vol > vf: raise l1 (increase lmid → reduce x_new → reduce vol).
        dc_pos = np.maximum(dc, 1e-30)   # guard against numerical negatives
        l1, l2 = 1e-9, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-6:
            lmid = 0.5 * (l1 + l2)
            x_new = np.clip(
                xPhys * (dc_pos / lmid) ** 0.5,
                np.maximum(0.001, xPhys - move),
                np.minimum(1.0,   xPhys + move),
            )
            if x_new.mean() > vf:
                l1 = lmid
            else:
                l2 = lmid

        x = x_new
        xPhys = filt.apply(x)

        change = float(np.max(np.abs(x - x_old)))
        if change < args.tol and it > 5:
            converged = True
            break

    elapsed = time.perf_counter() - t0

    # --- Final solve for summary --------------------------------------------
    xPhys_clipped = np.clip(xPhys, 1e-3, 1.0)
    fem_prob.update_density(xPhys_clipped)
    lams_final, _ = fem_prob.solve()
    freqs_final = np.sqrt(np.maximum(lams_final, 0.0)) / (2.0 * np.pi)

    print()
    print("=" * 60)
    status = "CONVERGED" if converged else "MAX ITER"
    print(f"  Status      : {status}")
    print(f"  Iterations  : {it}")
    print(f"  Final obj   : {float(lams_final.sum()):.6f}  (sum of eigenvalues)")
    print(f"  Final vol   : {float(xPhys.mean()):.4f}")
    print(f"  Wall time   : {elapsed:.1f} s")
    print()
    print("  Eigenfrequency summary:")
    print(f"  {'Mode':>6}  {'lambda (initial)':>18}  {'freq Hz (init)':>16}"
          f"  {'lambda (final)':>16}  {'freq Hz (final)':>16}")
    for j in range(min(len(lams0), len(lams_final))):
        print(
            f"  {j+1:>6}  {lams0[j]:>18.6f}  {freqs0[j]:>16.6f}"
            f"  {lams_final[j]:>16.6f}  {freqs_final[j]:>16.6f}"
        )
    print("=" * 60)

    # --- Save output --------------------------------------------------------
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    density_snapshot(
        xPhys, nx, ny,
        out_dir / "eigenfreq_density.png",
        title=f"Eigenfrequency opt — sum(lambda) = {float(lams_final.sum()):.4f}",
        centroids=filt.centroids,
        lx=lx, ly=ly,
    )
    print(f"  Saved: {out_dir / 'eigenfreq_density.png'}")

    save_convergence_plot(
        obj_history, vol_history,
        out_dir / "eigenfreq_convergence.png",
    )


if __name__ == "__main__":
    main()
