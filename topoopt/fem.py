"""
Linear elasticity FE problem using FEniCSx (dolfinx 0.7).

The weak form solved is:

    a(u, v; rho) = L(v)

where the stiffness is penalized element-wise via SIMP:

    E(rho_e) = E_min + rho_e^p * (E_0 - E_min)
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI


class ElasticityProblem:
    """
    Plane-stress linear elasticity on a rectangular domain.

    Parameters
    ----------
    nx, ny:
        Number of elements in x and y directions.
    lx, ly:
        Physical dimensions of the domain.
    E0:
        Young's modulus of the solid material.
    nu:
        Poisson's ratio.
    E_min:
        Young's modulus of the void phase (avoids singularity).
    penal:
        SIMP penalization exponent p.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        lx: float = 1.0,
        ly: float = 0.5,
        E0: float = 1.0,
        nu: float = 0.3,
        E_min: float = 1e-9,
        penal: float = 3.0,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.E0 = E0
        self.nu = nu
        self.E_min = E_min
        self.penal = penal

        # Build mesh and function spaces
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([lx, ly])],
            [nx, ny],
            mesh.CellType.quadrilateral,
        )
        self.V = functionspace(self.domain, ("Lagrange", 1, (2,)))
        self.DG0 = functionspace(self.domain, ("DG", 0))

        # Density function — updated each iteration via update_density()
        self.rho = fem.Function(self.DG0, name="density")
        self.rho.x.array[:] = 1.0

        self._setup_boundary_conditions()
        self._build_variational_form()

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _setup_boundary_conditions(self) -> None:
        """Clamp left edge (x = 0)."""
        left_dofs = fem.locate_dofs_geometrical(
            self.V,
            lambda x: np.isclose(x[0], 0.0),
        )
        zero = fem.Constant(self.domain, np.zeros(2, dtype=np.float64))
        self.bc = [fem.dirichletbc(zero, left_dofs, self.V)]

    # ------------------------------------------------------------------
    # Variational formulation (UFL)
    # ------------------------------------------------------------------

    def _build_variational_form(self) -> None:
        """Build UFL forms with SIMP-penalized stiffness."""
        domain = self.domain
        V = self.V
        rho = self.rho
        E0, E_min, nu, p = self.E0, self.E_min, self.nu, self.penal

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # SIMP: E(rho) = E_min + rho^p * (E0 - E_min)
        E_eff = E_min + rho**p * (E0 - E_min)

        # Plane-stress Lamé parameters
        lmbda = E_eff * nu / ((1 + nu) * (1 - nu))
        mu = E_eff / (2 * (1 + nu))

        def eps(w):
            return ufl.sym(ufl.grad(w))

        def sigma(w):
            return lmbda * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu * eps(w)

        # Bilinear form (UFL — NOT pre-compiled)
        self.a_ufl = ufl.inner(sigma(u), eps(v)) * ufl.dx

        # Right edge: uniform downward traction, total load = 1
        facet_dim = domain.topology.dim - 1
        right_facets = mesh.locate_entities_boundary(
            domain, facet_dim, lambda x: np.isclose(x[0], self.lx)
        )
        facet_tag = mesh.meshtags(
            domain,
            facet_dim,
            right_facets,
            np.full(len(right_facets), 1, dtype=np.int32),
        )
        ds_right = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
        t = fem.Constant(domain, np.array([0.0, -1.0 / self.ly]))

        # Linear form (UFL — NOT pre-compiled)
        self.L_ufl = ufl.dot(t, v) * ds_right(1)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self) -> tuple[fem.Function, float]:
        """
        Solve the elastic problem for the current density field.

        Returns
        -------
        u : fem.Function
            Displacement solution.
        compliance : float
            Compliance C = f^T u (scalar measure of structural flexibility).
        """
        problem = LinearProblem(
            self.a_ufl,
            self.L_ufl,
            bcs=self.bc,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        u = problem.solve()

        # Compliance = L(u) = ∫ t·u ds  (= a(u,u) for linear elasticity)
        compliance = fem.assemble_scalar(fem.form(ufl.action(self.L_ufl, u)))
        return u, float(compliance)

    # ------------------------------------------------------------------
    # Sensitivity dC/drho_e
    # ------------------------------------------------------------------

    def compliance_sensitivity(self, u: fem.Function) -> np.ndarray:
        """
        Element-wise sensitivity of compliance w.r.t. density.

        dC/drho_e = -p * rho_e^(p-1) * (E0 - E_min)
                    * integral_e [ sigma_0(u) : eps(u) ]

        where sigma_0 uses the unit (E=E0) stiffness tensor, so the
        penalization gradient factor dE/drho factors out cleanly.

        Result is projected onto DG0 via L2 projection.
        """
        rho = self.rho
        E0, E_min, nu, p = self.E0, self.E_min, self.nu, self.penal

        dE_drho = p * rho ** (p - 1) * (E0 - E_min)

        def eps(w):
            return ufl.sym(ufl.grad(w))

        # Unit stiffness (E0) stress for the sensitivity integrand
        lmbda0 = E0 * nu / ((1 + nu) * (1 - nu))
        mu0 = E0 / (2 * (1 + nu))

        def sigma0(w):
            return lmbda0 * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu0 * eps(w)

        # Sensitivity density (negative: adding material reduces compliance)
        se_density = -(dE_drho / E0) * ufl.inner(sigma0(u), eps(u))

        # L2 projection onto DG0
        phi = ufl.TestFunction(self.DG0)
        psi = ufl.TrialFunction(self.DG0)

        # NOTE: pass UFL forms to LinearProblem (not pre-compiled fem.form)
        a_proj_ufl = psi * phi * ufl.dx
        L_proj_ufl = se_density * phi * ufl.dx

        proj = LinearProblem(
            a_proj_ufl,
            L_proj_ufl,
            petsc_options={"ksp_type": "preonly", "pc_type": "jacobi"},
        )
        se_fn = proj.solve()
        return se_fn.x.array.copy()

    def update_density(self, rho_array: np.ndarray) -> None:
        """Write a new density array into the FE density function."""
        self.rho.x.array[:] = rho_array
        self.rho.x.scatter_forward()
