"""
Density filters for topology optimization.

Filtering regularises the problem, prevents checkerboard patterns, and
introduces a minimum length-scale.

**ConeFilter** (default)
    Classical weighted-average filter from Sigmund (2001) and Andreassen (2011).
    For each element e:

        rho_tilde_e = (sum_f w_ef * rho_f) / (sum_f w_ef)
        w_ef = max(0, r_min - dist(centroid_e, centroid_f))

    Implemented as sparse matrix-vector products; the adjoint M^T is EXACT
    (not an approximation), which avoids gradient errors near boundaries.

    Requires r_min >= ~2 * h_element for effective regularization.

**ProjectionFilter**
    Smooth Heaviside projection applied on top of a ConeFilter
    (Wang, Lazarov & Sigmund 2011). β controls sharpness; doubling β
    periodically drives the design toward crisp 0/1.
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem
import ufl

# scipy is available in dolfinx_complex environment
import scipy.sparse as sp
from scipy.spatial import cKDTree


class ConeFilter:
    """
    Weighted-average (cone kernel) density filter.

    Parameters
    ----------
    DG0 :
        DG0 function space already built on the mesh.
    r_min :
        Filter radius in the same physical units as the mesh.
        Rule of thumb: r_min >= 2 * h_element for effective regularization.
    """

    def __init__(self, DG0: fem.FunctionSpaceBase, r_min: float) -> None:
        self.r_min = r_min
        self.DG0 = DG0

        centroids = self._centroids(DG0)
        self.centroids = centroids   # (n, 2) physical coords in dolfinx DOF order
        n = len(centroids)

        # Build sparse filter matrix H (cone kernel, symmetric in distance)
        tree = cKDTree(centroids)
        pairs = tree.query_pairs(r_min, output_type="ndarray")  # upper-triangle pairs

        rows, cols, vals = [], [], []
        # Self-entries (every element is within r_min of itself)
        for i in range(n):
            rows.append(i); cols.append(i); vals.append(r_min)

        # Cross-entries (symmetric: H_ij = H_ji)
        for i, j in pairs:
            d = np.linalg.norm(centroids[i] - centroids[j])
            w = r_min - d
            rows.extend([i, j]); cols.extend([j, i]); vals.extend([w, w])

        H = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)

        # Row sums Hs_i = sum_j H_ij  (larger for interior elements)
        Hs = np.asarray(H.sum(axis=1)).ravel()

        # Forward filter: M = diag(1/Hs) @ H  →  rho_tilde = M @ rho
        self._M = sp.diags(1.0 / Hs) @ H  # shape (n, n)

    # ------------------------------------------------------------------

    @staticmethod
    def _centroids(DG0: fem.FunctionSpaceBase) -> np.ndarray:
        """Return element centroids as (n, 2) array using DG0 interpolation."""
        domain = DG0.mesh
        x = ufl.SpatialCoordinate(domain)
        pts = DG0.element.interpolation_points()

        cx = fem.Function(DG0)
        cy = fem.Function(DG0)
        cx.interpolate(fem.Expression(x[0], pts))
        cy.interpolate(fem.Expression(x[1], pts))
        return np.column_stack([cx.x.array, cy.x.array])

    # ------------------------------------------------------------------

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """
        Filter density field rho → rho_tilde.

        Parameters
        ----------
        rho : np.ndarray
            Unfiltered element densities (length = number of cells).

        Returns
        -------
        rho_tilde : np.ndarray
            Filtered densities.
        """
        return self._M @ rho

    def sensitivity_chain(
        self, drho_tilde: np.ndarray, rho: np.ndarray
    ) -> np.ndarray:
        """
        Back-propagate sensitivity through the filter (exact adjoint).

        dC/drho = M^T @ dC/drho_tilde

        M is NOT symmetric (Hs_i varies near boundaries), so M^T ≠ M.
        Using M^T is exact; using M (forward) would bias boundary gradients.

        Parameters
        ----------
        drho_tilde : np.ndarray
            dC/d(rho_tilde) — sensitivity w.r.t. filtered density.
        rho : np.ndarray
            Current unfiltered density (unused; kept for API consistency).

        Returns
        -------
        drho : np.ndarray
            dC/drho — sensitivity w.r.t. unfiltered density.
        """
        return self._M.T @ drho_tilde


class ProjectionFilter:
    """
    Smooth Heaviside projection applied after a ConeFilter.

    rho_bar = [tanh(β η) + tanh(β (rho_tilde − η))]
              / [tanh(β η) + tanh(β (1 − η))]

    Parameters
    ----------
    cone :
        A configured ConeFilter instance.
    beta :
        Sharpness parameter.  Start at 1 and double every ~50 iterations.
    eta :
        Threshold (typically 0.5).
    """

    def __init__(
        self,
        cone: ConeFilter,
        beta: float = 1.0,
        eta: float = 0.5,
    ) -> None:
        self.cone = cone
        self.beta = beta
        self.eta = eta

    def apply(self, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter then project.  Returns (rho_bar, rho_tilde)."""
        rho_tilde = self.cone.apply(rho)
        rho_bar = self._project(rho_tilde)
        return rho_bar, rho_tilde

    def _project(self, rho_tilde: np.ndarray) -> np.ndarray:
        b, n = self.beta, self.eta
        numer = np.tanh(b * n) + np.tanh(b * (rho_tilde - n))
        denom = np.tanh(b * n) + np.tanh(b * (1.0 - n))
        return numer / denom

    def _dproject(self, rho_tilde: np.ndarray) -> np.ndarray:
        b, n = self.beta, self.eta
        denom = np.tanh(b * n) + np.tanh(b * (1.0 - n))
        return b * (1.0 - np.tanh(b * (rho_tilde - n)) ** 2) / denom

    def sensitivity_chain(
        self, drho_bar: np.ndarray, rho_tilde: np.ndarray, rho: np.ndarray
    ) -> np.ndarray:
        """Chain rule through projection and cone filter."""
        drho_tilde = drho_bar * self._dproject(rho_tilde)
        return self.cone.sensitivity_chain(drho_tilde, rho)


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------
HelmholtzFilter = ConeFilter  # old name kept for existing test imports
