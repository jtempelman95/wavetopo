"""
Density filters for topology optimization.

Filtering regularises the problem, prevents checkerboard patterns, and
introduces a minimum length-scale.

Two approaches are provided:

1. **HelmholtzFilter** — PDE-based (Lazarov & Sigmund 2016).
   Solves:  -r² Δρ̃ + ρ̃ = ρ   with homogeneous Neumann BC.
   Uses a CG1 space internally (DG0 has zero element-interior gradient).
   The filter radius r is related to the physical radius R by r = R / (2√3).

2. **ProjectionFilter** — smooth Heaviside projection applied on top of the
   Helmholtz filter (Wang, Lazarov & Sigmund 2011).
   β controls sharpness; β→∞ recovers crisp 0/1 designs.
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    create_vector,
    LinearProblem,
)
import ufl
from petsc4py import PETSc
from mpi4py import MPI


class HelmholtzFilter:
    """
    PDE-based density filter.

    The filter PDE is solved on a CG1 space (continuous piecewise-linear)
    so that the Laplacian couples neighbouring elements.  The DG0 input is
    used directly as the source term; the CG1 filtered field is then
    projected back to DG0.

    The stiffness matrix is assembled once and factorised (MUMPS LU); only
    the RHS changes each iteration, so the solve is cheap.

    Parameters
    ----------
    DG0 :
        DG0 function space already built on the mesh.
    r_min :
        Physical filter radius R.  Converted to PDE radius r = R / (2√3).
    """

    def __init__(self, DG0: fem.FunctionSpaceBase, r_min: float) -> None:
        self.DG0 = DG0
        self.r = r_min / (2.0 * np.sqrt(3.0))
        domain = DG0.mesh

        # CG1 scalar space for the filtered field
        self._CG1 = functionspace(domain, ("Lagrange", 1))

        phi = ufl.TestFunction(self._CG1)
        psi = ufl.TrialFunction(self._CG1)
        r = self.r

        # Bilinear form: -r² Δρ̃ + ρ̃ = ρ  →  a(ρ̃, φ) = L(φ)
        a_ufl = (r**2 * ufl.dot(ufl.grad(psi), ufl.grad(phi)) + psi * phi) * ufl.dx

        # Source function lives in DG0; used in the RHS each call
        self._rho_in = fem.Function(DG0, name="rho_unfiltered")
        L_ufl = self._rho_in * phi * ufl.dx

        # Pre-compile and assemble stiffness matrix (constant)
        self._a_form = fem.form(a_ufl)
        self._A = assemble_matrix(self._a_form)
        self._A.assemble()

        self._L_form = fem.form(L_ufl)
        self._b = create_vector(self._L_form)

        # Solution vector (CG1)
        self._x_cg1 = self._A.createVecRight()

        # Factorised KSP — reused across all iterations
        self._ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        self._ksp.setOperators(self._A)
        self._ksp.setType("preonly")
        pc = self._ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        self._ksp.setUp()

        # Pre-built L2 projection from CG1 → DG0
        self._cg1_fn = fem.Function(self._CG1, name="rho_filtered_cg1")
        self._dg0_out = fem.Function(DG0, name="rho_filtered")
        phi0 = ufl.TestFunction(DG0)
        psi0 = ufl.TrialFunction(DG0)
        # DG0 mass matrix (diagonal)
        self._proj_a = fem.form(psi0 * phi0 * ufl.dx)
        self._proj_L = fem.form(self._cg1_fn * phi0 * ufl.dx)
        self._proj_A = assemble_matrix(self._proj_a)
        self._proj_A.assemble()
        self._proj_b = create_vector(self._proj_L)
        self._proj_x = self._proj_A.createVecRight()

        self._proj_ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        self._proj_ksp.setOperators(self._proj_A)
        self._proj_ksp.setType("preonly")
        self._proj_ksp.getPC().setType("jacobi")  # diagonal mass matrix
        self._proj_ksp.setUp()

    # ------------------------------------------------------------------

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """
        Filter DG0 density field rho → DG0 filtered field rho_tilde.

        Parameters
        ----------
        rho : np.ndarray
            Unfiltered element densities.

        Returns
        -------
        rho_tilde : np.ndarray
            Filtered densities (DG0).
        """
        # 1. Load source
        self._rho_in.x.array[:] = rho
        self._rho_in.x.scatter_forward()

        # 2. Assemble and solve Helmholtz PDE on CG1
        with self._b.localForm() as loc:
            loc.set(0.0)
        assemble_vector(self._b, self._L_form)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self._ksp.solve(self._b, self._x_cg1)

        # 3. L2-project CG1 solution → DG0
        self._cg1_fn.x.array[:] = self._x_cg1.array
        self._cg1_fn.x.scatter_forward()

        with self._proj_b.localForm() as loc:
            loc.set(0.0)
        assemble_vector(self._proj_b, self._proj_L)
        self._proj_b.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        self._proj_ksp.solve(self._proj_b, self._proj_x)

        return self._proj_x.array.copy()

    # ------------------------------------------------------------------

    def sensitivity_chain(
        self, drho_tilde: np.ndarray, rho: np.ndarray
    ) -> np.ndarray:
        """
        Back-propagate sensitivity through the filter via the adjoint.

        The combined filter F = P_{DG0} ∘ F_{CG1} is self-adjoint in the
        sense that the adjoint solve uses the same system matrices.

        Parameters
        ----------
        drho_tilde : np.ndarray
            dC/d(rho_tilde) in DG0.
        rho : np.ndarray
            Current unfiltered density (unused; kept for API consistency).

        Returns
        -------
        drho : np.ndarray
            dC/drho in DG0.
        """
        # Adjoint of (DG0-mass)^{-1} ∘ (CG1→DG0 coupling) ∘ (CG1-Helmholtz)^{-1}
        # ∘ (DG0→CG1 coupling) ∘ (DG0-mass)^{-1}  ∘ drho_tilde
        # Since all operators are symmetric this reduces to apply().
        return self.apply(drho_tilde)


class ProjectionFilter:
    """
    Smooth Heaviside projection applied after Helmholtz filtering.

    rho_bar = [tanh(β η) + tanh(β (rho_tilde − η))]
              / [tanh(β η) + tanh(β (1 − η))]

    Parameters
    ----------
    helmholtz :
        A configured HelmholtzFilter instance.
    beta :
        Sharpness parameter.  Start at 1–2 and double periodically.
    eta :
        Threshold (typically 0.5).
    """

    def __init__(
        self,
        helmholtz: HelmholtzFilter,
        beta: float = 1.0,
        eta: float = 0.5,
    ) -> None:
        self.helmholtz = helmholtz
        self.beta = beta
        self.eta = eta

    def apply(self, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter then project.  Returns (rho_bar, rho_tilde)."""
        rho_tilde = self.helmholtz.apply(rho)
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
        """Chain rule through projection and Helmholtz filter."""
        drho_tilde = drho_bar * self._dproject(rho_tilde)
        return self.helmholtz.sensitivity_chain(drho_tilde, rho)
