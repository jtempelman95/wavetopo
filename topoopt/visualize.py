"""
Visualization utilities for topology optimization results.

Provides:
- density_snapshot: save a single density field as a PNG
- OptimizationRecorder: accumulate iteration snapshots, dump them all at end
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def density_snapshot(
    rho: np.ndarray,
    nx: int,
    ny: int,
    out_path: str | Path,
    title: str = "",
    cmap: str = "gray_r",
) -> None:
    """
    Save a single density field as a PNG.

    Parameters
    ----------
    rho :
        Flat DG0 density array, length = nx*ny.
        dolfinx quad mesh numbers cells column-major (x-major for rows,
        i.e. index = ix*ny + iy), so reshape to (nx, ny) then transpose.
    nx, ny :
        Number of elements in x and y directions.
    out_path :
        Output file path (PNG).
    title :
        Optional figure title.
    cmap :
        Matplotlib colormap (default 'gray_r': black=solid, white=void).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # dolfinx quad mesh: cell index = ix * ny + iy  (x varies fastest in col)
    rho_grid = rho.reshape((nx, ny)).T  # shape (ny, nx) for imshow

    # Two panels: continuous density + thresholded (binary) view
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, data, subtitle in [
        (axes[0], rho_grid, "Density ρ (continuous)"),
        (axes[1], (rho_grid >= 0.5).astype(float), "Thresholded (ρ ≥ 0.5)"),
    ]:
        im = ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, label="ρ", fraction=0.046, pad=0.04)
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel("x elements")
        ax.set_ylabel("y elements")
        ax.tick_params(labelsize=9)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_convergence_plot(
    compliance_history: Sequence[float],
    volume_history: Sequence[float],
    out_path: str | Path,
) -> None:
    """
    Save a two-panel convergence plot (compliance + volume fraction vs iteration).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iters = list(range(1, len(compliance_history) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.semilogy(iters, compliance_history, "b-o", markersize=3, linewidth=1.2)
    ax1.set_ylabel("Compliance C", fontsize=11)
    ax1.set_title("Optimization Convergence", fontsize=12)
    ax1.grid(True, which="both", alpha=0.35)
    ax1.tick_params(labelsize=9)

    ax2.plot(iters, [v * 100 for v in volume_history], "r-s", markersize=3, linewidth=1.2)
    ax2.axhline(volume_history[-1] * 100, color="k", linestyle="--", linewidth=0.8,
                label=f"target {volume_history[-1]*100:.0f}%")
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


def save_optimization_strip(
    snapshots: list[tuple[int, float, np.ndarray]],
    nx: int,
    ny: int,
    out_path: str | Path,
    cmap: str = "gray_r",
) -> None:
    """
    Save a horizontal strip of density snapshots at selected iterations.

    Parameters
    ----------
    snapshots :
        List of (iteration, compliance, rho) tuples.
    nx, ny :
        Number of elements in x and y directions.
    out_path :
        Output PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, (it, C, rho) in zip(axes, snapshots):
        rho_grid = rho.reshape((nx, ny)).T
        # Show thresholded design in the strip for clarity
        ax.imshow((rho_grid >= 0.5).astype(float),
                  origin="lower", cmap=cmap, vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"Iter {it}\nC = {C:.1f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Topology Optimization — Cantilever Beam", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


class OptimizationRecorder:
    """
    Collects per-iteration data and writes PNGs at the end.

    Usage
    -----
    recorder = OptimizationRecorder(nx, ny, out_dir="results")
    # pass recorder.callback to SIMPOptimizer.optimize(...)
    result = opt.optimize(fem_solve, callback=recorder.callback)
    recorder.save_all()
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        out_dir: str | Path = "results",
        snapshot_iters: list[int] | None = None,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_iters = snapshot_iters  # None = auto-select
        self._compliances: list[float] = []
        self._volumes: list[float] = []
        self._snapshots: list[tuple[int, float, np.ndarray]] = []

    def callback(self, it: int, C: float, vol: float, rho: np.ndarray) -> None:
        self._compliances.append(C)
        self._volumes.append(vol)

        should_snap = (
            self.snapshot_iters is None  # auto: store every iter, thin later
            or it in self.snapshot_iters
        )
        if should_snap:
            self._snapshots.append((it, C, rho.copy()))

        print(f"  Iter {it:4d} | C = {C:10.4f} | vol = {vol:.4f}")

    def save_all(self, final_rho: np.ndarray, final_C: float) -> None:
        """Write all output PNGs."""
        nx, ny = self.nx, self.ny

        # 1. Final density
        density_snapshot(
            final_rho, nx, ny,
            self.out_dir / "density_final.png",
            title=f"Final density  C = {final_C:.2f}",
        )
        print(f"  Saved: {self.out_dir / 'density_final.png'}")

        # 2. Convergence plot
        save_convergence_plot(
            self._compliances,
            self._volumes,
            self.out_dir / "convergence.png",
        )

        # 3. Iteration strip: auto-select ~6 evenly-spaced snapshots
        if self._snapshots:
            raw = self._snapshots
            if self.snapshot_iters is None:
                # thin to at most 6 evenly spaced snapshots
                idx = np.round(np.linspace(0, len(raw) - 1, min(6, len(raw)))).astype(int)
                raw = [raw[i] for i in idx]
            save_optimization_strip(
                raw, nx, ny,
                self.out_dir / "optimization_strip.png",
            )
