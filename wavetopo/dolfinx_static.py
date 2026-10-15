r"""
dolfinx STATIC in-plane orthotropic elasticity -- the stiffness/mass companion to
the time-harmonic :class:`~wavetopo.dolfinx_elastic.ElasticWave`, sharing the SAME
rotated fiber constitutive law  Df(theta) = T Qf T^T  (engineering-strain Voigt)
and the SAME SIMP density zfac, so a single design (theta, z) can be co-designed
for wave control AND static stiffness/mass on one mesh.

    K(theta, z) u = f ,   K = int eps(v)^T ( zp(z) Df(theta) ) eps(u) dx ,
    zp(z) = zeps + (1-zeps) z^penal ,   compliance c = f . u ,
    mass  M = int ( z rho + (1-z) rho_m ) dx .

Bolt holes are clamped (u = 0 on their rims); a service load f is applied.
Self-adjoint compliance sensitivities (u frozen):

    dc/dtheta = - eps(u)^T (dDf/dtheta) eps(u) ,
    dc/dz     = - penal (1-zeps) z^(penal-1) eps(u)^T Df eps(u) ,
    dM/dz     = (rho - rho_m) .

All FD-verified (see the __main__ self-test).
"""
from __future__ import annotations

import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector,
                               apply_lifting, set_bc)

from wavetopo.dolfinx_elastic import Qf_matrix, _lu, _SUM


class StaticElastic:
    def __init__(self, domain, *, Ef1=131.0, Ef2=9.0, G12=5.0, nu12=0.27,
                 rho=1.6, rho_m=1.2, penal=3.0, zeps=1e-4):
        self.domain = domain
        self.rho = rho; self.rho_m = rho_m; self.penal = penal; self._zeps = zeps
        self.Qf = Qf_matrix(Ef1, Ef2, G12, nu12)
        self.V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
        self.S = fem.functionspace(domain, ("DG", 0))
        self.theta = fem.Function(self.S)
        self.zfac = fem.Function(self.S); self.zfac.x.array[:] = 1.0
        self.f = fem.Function(self.V)
        self.u = fem.Function(self.V)
        self._bc = []
        self._forms()

    def _Dmat(self, theta):
        c, s = ufl.cos(theta), ufl.sin(theta)
        T = ufl.as_matrix([[c*c, s*s, -2*s*c],
                           [s*s, c*c,  2*s*c],
                           [s*c, -s*c, c*c - s*s]])
        Q = ufl.as_matrix([[float(self.Qf[0, 0]), float(self.Qf[0, 1]), 0.0],
                           [float(self.Qf[0, 1]), float(self.Qf[1, 1]), 0.0],
                           [0.0, 0.0, float(self.Qf[2, 2])]])
        return T * Q * T.T

    @staticmethod
    def _evoigt(u):
        return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)])

    def _zp(self):
        return self._zeps + (1 - self._zeps) * self.zfac**self.penal

    def _forms(self):
        D = self._zp() * self._Dmat(self.theta)
        u = ufl.TrialFunction(self.V); v = ufl.TestFunction(self.V)
        self.a_form = fem.form(ufl.dot(self._evoigt(v), D * self._evoigt(u))*ufl.dx)
        self.L_form = fem.form(ufl.dot(self.f, v) * ufl.dx)

    # ---- setters ------------------------------------------------------ #
    def set_theta(self, arr):
        self.theta.x.array[:] = np.asarray(arr, float)

    def set_zfac(self, fn):
        self.zfac.interpolate(lambda x: fn(x[0], x[1]))

    def set_load(self, fx, fy):
        self.f.interpolate(lambda x: np.vstack([fx(x[0], x[1]), fy(x[0], x[1])]))

    def clamp(self, holes, tol):
        """Fully clamp (u = 0) the rims of the given circular holes."""
        def on_holes(x):
            m = np.zeros(x.shape[1], dtype=bool)
            for cx, cy, r in holes:
                m |= np.hypot(x[0]-cx, x[1]-cy) <= r + tol
            return m
        dofs = fem.locate_dofs_geometrical(self.V, on_holes)
        zero = fem.Constant(self.domain,
                            (default_scalar_type(0.0), default_scalar_type(0.0)))
        self._bc = [fem.dirichletbc(zero, dofs, self.V)]
        return dofs

    # ---- solve / objectives ------------------------------------------ #
    def solve(self):
        A = assemble_matrix(self.a_form, bcs=self._bc); A.assemble()
        b = assemble_vector(self.L_form)
        apply_lifting(b, [self.a_form], bcs=[self._bc])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self._bc)
        self._A = A
        _lu(A, self.domain.comm).solve(b, self.u.vector)
        self.u.x.scatter_forward()
        return self.u

    def compliance(self):
        c = fem.assemble_scalar(fem.form(ufl.dot(self.f, self.u) * ufl.dx))
        return self.domain.comm.allreduce(c, op=_SUM())

    def mass(self):
        rho = self.zfac*self.rho + (1 - self.zfac)*self.rho_m
        return self.domain.comm.allreduce(
            fem.assemble_scalar(fem.form(rho * ufl.dx)), op=_SUM())

    # ---- self-adjoint sensitivities (u frozen) ----------------------- #
    def compliance_grad_theta(self):
        D = self._zp() * self._Dmat(self.theta)
        E = ufl.dot(self._evoigt(self.u), D * self._evoigt(self.u)) * ufl.dx
        dE = fem.form(ufl.derivative(E, self.theta, ufl.TestFunction(self.S)))
        return -assemble_vector(dE).array.copy()

    def compliance_grad_z(self):
        D = self._zp() * self._Dmat(self.theta)
        E = ufl.dot(self._evoigt(self.u), D * self._evoigt(self.u)) * ufl.dx
        dE = fem.form(ufl.derivative(E, self.zfac, ufl.TestFunction(self.S)))
        return -assemble_vector(dE).array.copy()

    def mass_grad_z(self):
        one = ufl.TestFunction(self.S)
        rho = self.zfac*self.rho + (1 - self.zfac)*self.rho_m
        return assemble_vector(fem.form(
            ufl.derivative(rho*ufl.dx, self.zfac, one))).array.copy()


if __name__ == "__main__":
    # --- finite-difference verification of the orientation gradient --------
    import sys
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
    from wavetopo.dolfinx_mesh import rect_mesh

    Lx, Ly = 4.0, 2.5
    holes = [(0.9, 0.6, 0.18), (0.9, 1.9, 0.18)]
    dom = rect_mesh(Lx, Ly, 0.06, holes=holes, h_hole=0.03)
    se = StaticElastic(dom)
    se.clamp(holes, tol=0.03)
    # downward load over a central "payload" box
    se.set_load(lambda x, y: np.zeros_like(x),
                lambda x, y: -1.0*((np.abs(x-2.3) < 0.4) & (np.abs(y-Ly/2) < 0.4)))
    rng = np.random.default_rng(0)
    n = se.theta.x.array.size
    th = 0.3*rng.standard_normal(n)
    se.set_theta(th); se.solve()
    c0 = se.compliance()
    g = se.compliance_grad_theta()
    print(f"cells={n}  compliance={c0:.6e}  ||g||={np.linalg.norm(g):.3e}")
    eps = 1e-6
    idx = rng.choice(n, 6, replace=False)
    print("  dof     analytic        FD           rel.err")
    for i in idx:
        tp = th.copy(); tp[i] += eps; se.set_theta(tp); se.solve(); cp = se.compliance()
        tm = th.copy(); tm[i] -= eps; se.set_theta(tm); se.solve(); cm = se.compliance()
        fd = (cp - cm)/(2*eps)
        re = abs(g[i]-fd)/(abs(fd)+1e-30)
        print(f"  {i:5d}  {g[i]:+.6e}  {fd:+.6e}  {re:.2e}")
