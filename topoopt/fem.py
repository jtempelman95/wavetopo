"""
Linear elasticity FE problem using FEniCSx (dolfinx 0.7).

The weak form solved is:

    a(u, v; rho) = L(v)

where the stiffness is penalized element-wise via SIMP:

    E(rho_e) = E_min + rho_e^p * (E_0 - E_min)

Load
----
A **point load** (concentrated traction on the two central facets of the
right edge, symmetric about y = ly/2) is used.  This is the standard
cantilever benchmark load.  Distributed loading (over the full right edge)
is available via load_type="distributed".

Why point load?  With a distributed traction the force is applied to every
right-edge element — including void ones.  Because void elements have
E_eff ≈ E_min → 0, they deform enormously under direct load, producing
artificially large strain-energy sensitivities that drive disconnected
boundary elements solid.  A point load avoids this.
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
    load_type:
        "midpoint"    — concentrated load at centre of right edge (default).
        "distributed" — uniform traction over the full right edge.
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
        load_type: str = "midpoint",
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.E0 = E0
        self.nu = nu
        self.E_min = E_min
        self.penal = penal
        self.load_type = load_type

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
        self._build_variational_form()

    # ------------------------------------------------------------------
    # Boundary conditions — clamp left edge
    # ------------------------------------------------------------------

    def _setup_boundary_conditions(self) -> None:
        left_dofs = fem.locate_dofs_geometrical(
            self.V, lambda x: np.isclose(x[0], 0.0)
        )
        zero = fem.Constant(self.domain, np.zeros(2, dtype=np.float64))
        self.bc = [fem.dirichletbc(zero, left_dofs, self.V)]

    # ------------------------------------------------------------------
    # Variational formulation
    # ------------------------------------------------------------------

    def _build_variational_form(self) -> None:
        domain = self.domain
        V = self.V
        rho = self.rho
        E0, E_min, nu, p = self.E0, self.E_min, self.nu, self.penal

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # SIMP stiffness
        E_eff = E_min + rho**p * (E0 - E_min)
        lmbda = E_eff * nu / ((1 + nu) * (1 - nu))   # plane-stress λ*
        mu = E_eff / (2 * (1 + nu))

        def eps(w):
            return ufl.sym(ufl.grad(w))

        def sigma(w):
            return lmbda * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu * eps(w)

        self.a_ufl = ufl.inner(sigma(u), eps(v)) * ufl.dx

        # ------ Load -------------------------------------------------------
        facet_dim = domain.topology.dim - 1

        if self.load_type == "midpoint":
            # Concentrated load: two facets straddling the midpoint y = ly/2.
            # Condition: facet has a vertex strictly closer than h_y to ly/2.
            # This selects exactly the two facets immediately above and below
            # the midpoint vertex (for any even ny), giving exactly 2 facets.
            h_y = self.ly / self.ny
            load_facets = mesh.locate_entities_boundary(
                domain,
                facet_dim,
                lambda x: (
                    np.isclose(x[0], self.lx)
                    & (np.abs(x[1] - self.ly / 2) <= h_y + 1e-10)
                ),
            )
            n_facets = max(len(load_facets), 1)  # guard against empty
            # Total force = 1; distributed over n_facets * h_y length
            t_mag = -1.0 / (n_facets * h_y)
        else:
            # Distributed: uniform traction over the entire right edge
            load_facets = mesh.locate_entities_boundary(
                domain, facet_dim, lambda x: np.isclose(x[0], self.lx)
            )
            t_mag = -1.0 / self.ly

        load_tag = mesh.meshtags(
            domain,
            facet_dim,
            load_facets,
            np.full(len(load_facets), 1, dtype=np.int32),
        )
        ds_load = ufl.Measure("ds", domain=domain, subdomain_data=load_tag)
        t = fem.Constant(domain, np.array([0.0, t_mag]))
        self.L_ufl = ufl.dot(t, v) * ds_load(1)

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
            C = L(u)  (= a(u,u) for self-adjoint linear elasticity).
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
        compliance = fem.assemble_scalar(fem.form(ufl.action(self.L_ufl, u)))
        return u, float(compliance)

    # ------------------------------------------------------------------
    # Sensitivity  dC/d(rho_tilde)_e
    # ------------------------------------------------------------------

    def compliance_sensitivity(self, u: fem.Function) -> np.ndarray:
        """
        Element-wise sensitivity of compliance w.r.t. the FILTERED density.

        dC/drho_e = -(dE/drho_e / E0) * integral_e [ sigma0(u) : eps(u) ]

        where sigma0 uses E = E0 (the unit-stiffness stress), so the
        penalization derivative dE/drho factors out cleanly.

        The result is projected to DG0 via an L2 projection (the DG0 mass
        matrix is diagonal, so each element gets its local average).
        """
        rho = self.rho
        E0, E_min, nu, p = self.E0, self.E_min, self.nu, self.penal

        dE_drho = p * rho ** (p - 1) * (E0 - E_min)

        def eps(w):
            return ufl.sym(ufl.grad(w))

        lmbda0 = E0 * nu / ((1 + nu) * (1 - nu))
        mu0 = E0 / (2 * (1 + nu))

        def sigma0(w):
            return lmbda0 * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu0 * eps(w)

        # Sensitivity density (negative: adding material reduces compliance)
        se_density = -(dE_drho / E0) * ufl.inner(sigma0(u), eps(u))

        # L2 projection onto DG0
        phi = ufl.TestFunction(self.DG0)
        psi = ufl.TrialFunction(self.DG0)
        proj = LinearProblem(
            psi * phi * ufl.dx,
            se_density * phi * ufl.dx,
            petsc_options={"ksp_type": "preonly", "pc_type": "jacobi"},
        )
        return proj.solve().x.array.copy()

    # ------------------------------------------------------------------

    def update_density(self, rho_array: np.ndarray) -> None:
        """Write a new density array into the FE density function."""
        self.rho.x.array[:] = rho_array
        self.rho.x.scatter_forward()
