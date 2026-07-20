"""
Optimality Criteria (OC) update for SIMP topology optimization.

The OC method finds the optimal density update satisfying:

    minimize   C(rho)  = f^T u
    subject to V(rho)  = vf * V_total
               0 ≤ rho_e ≤ 1

The heuristic update rule (Bendsøe & Sigmund 2004, §1.2):

    rho_e^{new} = clip( rho_e * (−dC/drho_e / λ)^η , rho_min, 1 )

where λ is a Lagrange multiplier for the volume constraint found via
bisection, and η = 0.5 is a numerical damping factor.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class OptimizationResult:
    """Container returned after an optimization run."""
    densities: np.ndarray           # final density field
    compliance_history: list[float] = field(default_factory=list)
    volume_history: list[float] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False


class SIMPOptimizer:
    """
    Optimality Criteria optimizer for SIMP compliance minimization.

    Parameters
    ----------
    n_elem :
        Total number of finite elements.
    vf :
        Target volume fraction (0 < vf ≤ 1).
    move :
        Maximum allowed density change per iteration (default 0.2).
    eta :
        OC damping exponent (default 0.5).
    tol :
        Convergence tolerance on max density change (default 1e-3).
    max_iter :
        Maximum number of iterations.
    """

    def __init__(
        self,
        n_elem: int,
        vf: float,
        move: float = 0.2,
        eta: float = 0.5,
        tol: float = 1e-3,
        max_iter: int = 100,
    ) -> None:
        self.n_elem = n_elem
        self.vf = vf
        self.move = move
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------

    def optimize(
        self,
        fem_solve: Callable[[np.ndarray], tuple[float, np.ndarray]],
        rho0: np.ndarray | None = None,
        callback: Callable[[int, float, float, np.ndarray], None] | None = None,
    ) -> OptimizationResult:
        """
        Run the OC optimization loop.

        Parameters
        ----------
        fem_solve :
            A callable ``(rho) -> (compliance, sensitivity)`` that runs the FE
            solve and returns the scalar compliance and the element-wise
            sensitivity array dC/drho_e (already chain-ruled through any filter).
        rho0 :
            Initial density field.  Defaults to uniform vf.
        callback :
            Optional function called each iteration as
            ``callback(iter, compliance, volume_frac, rho)``.

        Returns
        -------
        OptimizationResult
        """
        rho = np.full(self.n_elem, self.vf) if rho0 is None else rho0.copy()

        result = OptimizationResult(densities=rho.copy())

        for it in range(1, self.max_iter + 1):
            rho_old = rho.copy()

            compliance, dc = fem_solve(rho)

            # Volume fraction
            vol = rho.mean()

            result.compliance_history.append(compliance)
            result.volume_history.append(vol)

            if callback is not None:
                callback(it, compliance, vol, rho)

            # OC density update
            rho = self._oc_update(rho, dc)

            # Convergence check
            change = np.max(np.abs(rho - rho_old))
            if change < self.tol and it > 5:
                result.converged = True
                break

        result.densities = rho.copy()
        result.n_iterations = it
        return result

    # ------------------------------------------------------------------

    def _oc_update(self, rho: np.ndarray, dc: np.ndarray) -> np.ndarray:
        """
        Perform the OC density update with bisection for λ.

        The sensitivity dc = dC/drho_e is expected to be negative (compliance
        decreases when density increases in solid regions).  We pass |dc| into
        the update rule so the formula matches standard references.
        """
        # Use absolute value: OC expects positive sensitivity measure
        dc_abs = np.maximum(-dc, 1e-30)

        move = self.move
        eta = self.eta
        vf = self.vf

        # Bisection for Lagrange multiplier
        l1, l2 = 1e-9, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-6:
            lmid = 0.5 * (l1 + l2)
            rho_new = np.clip(
                rho * (dc_abs / lmid) ** eta,
                np.maximum(0.001, rho - move),
                np.minimum(1.0, rho + move),
            )
            if rho_new.mean() > vf:
                l1 = lmid
            else:
                l2 = lmid

        return rho_new
