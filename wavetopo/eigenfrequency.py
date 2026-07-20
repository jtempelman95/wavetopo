"""
Eigenfrequency topology optimization FE problem using FEniCSx (dolfinx 0.7).

Solves the generalized eigenvalue problem:

    K phi = lambda M phi

where K is the SIMP-penalized stiffness matrix and M is the linearized mass
matrix.  The objective is to maximize the sum of the first n_eigs eigenvalues
(squared natural frequencies), which is equivalent to maximizing structural
stiffness against free vibration.

SIMP stiffness:
    E(rho_e) = E_min + rho_e^p * (E0 - E_min)

Linear mass interpolation (avoids localized void modes):
    rho_m(rho_e) = rho_min_m + rho_e * (1 - rho_min_m)

The left edge is fully clamped.  Constrained DOFs are given negligible mass
(1e-30 diagonal entry in M) so that the associated "rigid-body" modes are
pushed to very high frequency and never appear in the first n_eigs eigenpairs.

Requires SLEPc (via slepc4py) for the eigensolver.
"""

from __future__ import annotations

import numpy as np
from math import sqrt

from dolfinx import fem, mesh
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
import ufl
from mpi4py import MPI


class EigenfrequencyProblem:
    """
    Plane-stress free-vibration eigenvalue problem on a rectangular domain.

    Solves K phi_j = lambda_j M phi_j for the first n_eigs eigenpairs.
    Both K and M are assembled as PETSc matrices and handed to SLEPc's EPS.

    Parameters
    ----------
    nx, ny :
        Number of elements in x and y directions.
    lx, ly :
        Physical domain dimensions.
    E0 :
        Young's modulus of the solid phase.
    nu :
        Poisson's ratio.
    E_min :
        Young's modulus of the void phase (avoids singular K).
    penal :
        SIMP penalization exponent p.
    rho0 :
        Material mass density (physical units).
    rho_min_m :
        Minimum mass density for void elements (avoids localized void modes).
    n_eigs :
        Number of eigenfrequencies to compute.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        lx: float = 2.0,
        ly: float = 1.0,
        E0: float = 1.0,
        nu: float = 0.3,
        E_min: float = 1e-9,
        penal: float = 3.0,
        rho0: float = 1.0,
        rho_min_m: float = 1e-6,
        n_eigs: int = 5,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.E0 = E0
        self.nu = nu
        self.E_min = E_min
        self.penal = penal
        self.rho0 = rho0
        self.rho_min_m = rho_min_m
        self.n_eigs = n_eigs

        # --- Mesh and function spaces (same pattern as ElasticityProblem) ----
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([lx, ly])],
            [nx, ny],
            mesh.CellType.quadrilateral,
        )
        self.V = functionspace(self.domain, ("Lagrange", 1, (2,)))
        self.DG0 = functionspace(self.domain, ("DG", 0))

        self.rho = fem.Function(self.DG0, name="density")
        self.rho.x.array[:] = 1.0

        self._setup_boundary_conditions()
        self._build_variational_forms()

    # ------------------------------------------------------------------
    # Boundary conditions — clamp left edge (same as ElasticityProblem)
    # ------------------------------------------------------------------

    def _setup_boundary_conditions(self) -> None:
        left_dofs = fem.locate_dofs_geometrical(
            self.V, lambda x: np.isclose(x[0], 0.0)
        )
        zero = fem.Constant(self.domain, np.zeros(2, dtype=np.float64))
        self.bc = [fem.dirichletbc(zero, left_dofs, self.V)]

    # ------------------------------------------------------------------
    # Variational forms for K and M
    # ------------------------------------------------------------------

    def _build_variational_forms(self) -> None:
        rho = self.rho
        E0, E_min, nu, p = self.E0, self.E_min, self.nu, self.penal
        rho0, rho_min_m = self.rho0, self.rho_min_m

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # SIMP stiffness interpolation
        E_eff = E_min + rho**p * (E0 - E_min)
        lmbda_eff = E_eff * nu / ((1 + nu) * (1 - nu))   # plane-stress lambda*
        mu_eff = E_eff / (2 * (1 + nu))

        def eps(w):
            return ufl.sym(ufl.grad(w))

        def sigma(w, lmbda, mu):
            return lmbda * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu * eps(w)

        # Bilinear form for stiffness K
        a_K = ufl.inner(sigma(u, lmbda_eff, mu_eff), eps(v)) * ufl.dx

        # Linear mass interpolation — avoids localized modes in void regions
        rho_m = rho_min_m + rho * (1.0 - rho_min_m)

        # Bilinear form for mass M
        a_M = rho_m * rho0 * ufl.inner(u, v) * ufl.dx

        self.a_K_form = fem.form(a_K)
        self.a_M_form = fem.form(a_M)

    # ------------------------------------------------------------------
    # Density update
    # ------------------------------------------------------------------

    def update_density(self, rho_array: np.ndarray) -> None:
        """Write a new density array into the FE density function."""
        self.rho.x.array[:] = rho_array
        self.rho.x.scatter_forward()

    # ------------------------------------------------------------------
    # Eigensolver
    # ------------------------------------------------------------------

    def solve(self) -> tuple[np.ndarray, list[fem.Function]]:
        """
        Assemble K and M, then solve the generalized eigenproblem K phi = lam M phi.

        Constrained DOFs are zeroed out in M (replaced by 1e-30) so they
        appear at very high frequency and do not pollute the lowest modes.

        Returns
        -------
        eigenvalues : np.ndarray, shape (n_conv,)
            Converged eigenvalues (lambda_j = omega_j^2).  n_conv <= n_eigs.
        eigenvectors : list of fem.Function, length n_conv
            Mass-normalised mode shapes in the displacement space V.
        """
        from slepc4py import SLEPc

        # --- Assemble K with Dirichlet BCs ---
        K = assemble_matrix(self.a_K_form, bcs=self.bc)
        K.assemble()

        # --- Assemble M (no BCs — we handle them manually below) ---
        M = assemble_matrix(self.a_M_form)
        M.assemble()

        # Apply BCs to M: zero the constrained rows/columns, put tiny diagonal
        # so those DOFs are effectively decoupled and pushed to high frequency.
        dofs = self.bc[0].dof_indices()[0]
        if len(dofs) > 0:
            M.zeroRowsColumns(dofs, 1e-30)
        M.assemble()

        # --- SLEPc EPS setup ---
        eps = SLEPc.EPS().create(MPI.COMM_WORLD)
        eps.setOperators(K, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        eps.setDimensions(self.n_eigs, max(2 * self.n_eigs, self.n_eigs + 10))
        eps.setTolerances(tol=1e-8, max_it=300)
        eps.solve()

        n_conv = eps.getConverged()
        n_extract = min(n_conv, self.n_eigs)

        eigenvalues = np.zeros(n_extract)
        eigenvectors: list[fem.Function] = []

        Mv = K.createVecRight()  # work vector for M @ vr

        for i in range(n_extract):
            vr, vi = K.createVecs()
            lam = eps.getEigenpair(i, vr, vi)

            # Mass-normalise: vr <- vr / sqrt(vr^T M vr)
            M.mult(vr, Mv)
            norm2 = vr.dot(Mv)
            if norm2 > 0.0:
                vr.scale(1.0 / sqrt(norm2))

            # Wrap in a fem.Function
            phi = fem.Function(self.V, name=f"mode_{i}")
            phi.x.array[:] = vr.getArray()
            phi.x.scatter_forward()

            eigenvalues[i] = lam.real
            eigenvectors.append(phi)

        return eigenvalues, eigenvectors

    # ------------------------------------------------------------------
    # Sensitivity  d(sum lambda_j) / d(rho_e)
    # ------------------------------------------------------------------

    def eigenvalue_sensitivity(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: list[fem.Function],
    ) -> np.ndarray:
        """
        Element-wise sensitivity of the sum of eigenvalues w.r.t. physical density.

        For objective f = sum_j lambda_j, the analytic sensitivity is:

            df/drho_e = sum_j [
                p * rho_e^(p-1) * (E0 - E_min) * mse_j_e
                - lambda_j * (1 - rho_min_m) * rho0 * mke_j_e
            ]

        where mse_j_e is the modal strain energy density (using unit E0 stress)
        and mke_j_e is the modal kinetic energy density, both projected to DG0.

        The sensitivity is POSITIVE: adding material tends to increase eigenvalues.

        Parameters
        ----------
        eigenvalues :
            Array of eigenvalues returned by solve().
        eigenvectors :
            List of mass-normalised mode shapes returned by solve().

        Returns
        -------
        sensitivity : np.ndarray, shape (n_elem,)
            df/d(rho_tilde)_e — positive for solid-improving elements.
        """
        rho = self.rho
        E0, E_min, nu, p = self.E0, self.E_min, self.nu, self.penal
        rho0, rho_min_m = self.rho0, self.rho_min_m

        # Unit-stiffness helpers (same as compliance_sensitivity in fem.py)
        lmbda0 = E0 * nu / ((1 + nu) * (1 - nu))
        mu0 = E0 / (2 * (1 + nu))

        def eps(w):
            return ufl.sym(ufl.grad(w))

        def sigma0(w):
            return lmbda0 * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu0 * eps(w)

        # DG0 projection helpers
        phi_dg = ufl.TestFunction(self.DG0)
        psi_dg = ufl.TrialFunction(self.DG0)
        # DG0 mass matrix (diagonal — one entry per element)
        a_proj = psi_dg * phi_dg * ufl.dx
        petsc_opts = {"ksp_type": "preonly", "pc_type": "jacobi"}

        # Derivative of SIMP stiffness w.r.t. rho
        dE_drho = p * rho ** (p - 1) * (E0 - E_min)

        n_elem = len(self.rho.x.array)
        sensitivity = np.zeros(n_elem)

        for lam, phi in zip(eigenvalues, eigenvectors):
            # --- Modal strain energy density (dK contribution) ---
            # se_density = (dE/drho / E0) * sigma0(phi) : eps(phi)
            # Factor dE_drho/E0 projects the unit-E0 strain energy to the
            # actual sensitivity derivative.
            se_density = (dE_drho / E0) * ufl.inner(sigma0(phi), eps(phi))

            proj_se = LinearProblem(
                a_proj,
                se_density * phi_dg * ufl.dx,
                petsc_options=petsc_opts,
            )
            mse_j = proj_se.solve().x.array.copy()

            # --- Modal kinetic energy density (dM contribution) ---
            ke_density = ufl.inner(phi, phi)

            proj_ke = LinearProblem(
                a_proj,
                ke_density * phi_dg * ufl.dx,
                petsc_options=petsc_opts,
            )
            mke_j = proj_ke.solve().x.array.copy()

            # Accumulate: stiffness term raises eigenvalue, mass term lowers it
            sensitivity += mse_j - lam * (1.0 - rho_min_m) * rho0 * mke_j

        return sensitivity
