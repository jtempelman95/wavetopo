r"""
dolfinx (FEniCSx) IN-PLANE (vector) orthotropic time-harmonic elastodynamics --
a faithful port of wavetopo.harmonic.HarmonicLens to dolfinx.  This is the SAME
physics as the original numpy pipeline (Q4 -> P1 triangles): full 2-D elasticity
with the rotated fiber plane-stress stiffness  Df(theta) = T(theta) Qf T(theta)^T
(engineering-strain Voigt), NOT the weaker scalar antiplane-shear model.

Complex-symmetric dynamic operator, solved in a real PETSc build via the real
2-field split u = u_r + i u_i (each a 2-vector), U = (uxr, uyr, uxi, uyi):

    D(theta) u = f,   D = (1+i eta) K(theta) - w^2 M + i w C ,
    [ A -B ; B A ] (u_r; u_i) = (f_r; f_i),  A = K - w^2 M,  B = eta K + w C .

K(theta) = int  eps(v) : Df(theta) : eps(u) ,  eps in engineering Voigt
[exx, eyy, gxy=ux,y+uy,x].  M consistent mass (rho), C mass-proportional sponge.

Design = per-cell (DG0) fiber orientation theta (optionally via a smooth CS-RBF
support map).  Discrete adjoint by UFL differentiation of the residual, exactly
as in wavetopo.dolfinx_wave.  FD-verified.
"""
from __future__ import annotations

import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector


def Qf_matrix(Ef1, Ef2, G12, nu12):
    """Fiber plane-stress stiffness in material axes (as in cfrp.Material.Qf)."""
    nu21 = nu12 * Ef2 / Ef1
    den = 1.0 - nu12 * nu21
    return np.array([[Ef1/den, nu12*Ef2/den, 0.0],
                     [nu12*Ef2/den, Ef2/den, 0.0],
                     [0.0, 0.0, G12]])


def _lu(A, comm):
    """Sparse direct solve.  PETSc's BUILT-IN LU (the default when no
    factorSolverType is set) is a sequential, comparatively slow factorization;
    on an 83k x 83k wave operator with 2.3M nonzeros it takes 2.97 s against
    0.68 s for MUMPS -- a 4.4x difference, measured by examples/bench_lu.py.

    The gain is the ALGORITHM, not parallelism: MUMPS is ~4.9x faster even at
    OMP_NUM_THREADS=1, and setting OMP_NUM_THREADS higher changes nothing here
    because a sparse LU barely touches the dense BLAS kernels that threading
    accelerates.  Genuine multi-core scaling for these solves needs MPI
    (mpirun -n N), which requires the drivers to be rank-safe.

    Falls back to the built-in factorization if MUMPS is not in the build."""
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A); ksp.setType("preonly")
    pc = ksp.getPC(); pc.setType("lu")
    try:
        pc.setFactorSolverType("mumps")
        ksp.setUp()
    except PETSc.Error:
        pc.setFactorSolverType("")          # built-in
    return ksp


class ElasticWave:
    def __init__(self, domain, *, omega, Ef1=131.0, Ef2=9.0, G12=5.0,
                 nu12=0.27, rho=1.6, rho_m=1.2, eta=0.02, penal=3.0):
        self.domain = domain; self.omega = omega
        self.rho = rho; self.rho_m = rho_m; self.eta = eta
        self.penal = penal; self._zeps = 1e-4
        self.Qf = Qf_matrix(Ef1, Ef2, G12, nu12)
        # 4-component P1 field U = (uxr, uyr, uxi, uyi)
        self.W = fem.functionspace(domain, ("Lagrange", 1, (4,)))
        self.S = fem.functionspace(domain, ("DG", 0))
        self.theta = fem.Function(self.S)
        self.zfac = fem.Function(self.S); self.zfac.x.array[:] = 1.0  # SIMP density
        self.cspg = fem.Function(self.S)
        self.fr = fem.Function(self.W)
        self.U = fem.Function(self.W)
        self.Lam = fem.Function(self.W)
        self.wsel = fem.Function(self.S)
        self.Uref = fem.Function(self.W)
        self._forms()

    # ---- constitutive: rotated fiber stiffness Df(theta) -------------- #
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
    def _evoigt(u):                       # engineering Voigt strain of a 2-vector
        return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)])

    def _stiff(self, a, b, D):
        return ufl.dot(self._evoigt(a), D * self._evoigt(b)) * ufl.dx

    def _zp(self):
        return self._zeps + (1 - self._zeps) * self.zfac**self.penal

    def _rho_eff(self):
        return self.zfac * self.rho + (1 - self.zfac) * self.rho_m

    def _forms(self):
        w = self.omega; et = self.eta
        D = self._zp() * self._Dmat(self.theta)          # SIMP-weighted stiffness
        rho = self._rho_eff()
        U = ufl.TrialFunction(self.W); P = ufl.TestFunction(self.W)
        ur = ufl.as_vector([U[0], U[1]]); ui = ufl.as_vector([U[2], U[3]])
        pr = ufl.as_vector([P[0], P[1]]); pi = ufl.as_vector([P[2], P[3]])
        kk = lambda a, b: self._stiff(a, b, D)
        ms = lambda a, b: rho * ufl.dot(a, b) * ufl.dx
        dp = lambda a, b: self.cspg * ufl.dot(a, b) * ufl.dx
        # [A -B; B A],  A=K-w^2 M,  B=eta K + w C
        aE = (kk(ur, pr) - w**2*ms(ur, pr) - et*kk(ui, pr) - w*dp(ui, pr)
              + kk(ui, pi) - w**2*ms(ui, pi) + et*kk(ur, pi) + w*dp(ur, pi))
        self.a_form = fem.form(aE)
        self.L_form = fem.form(ufl.inner(self.fr, P) * ufl.dx)

    # ---- setters ------------------------------------------------------ #
    def set_sponge(self, fn):
        self.cspg.interpolate(lambda x: fn(x[0], x[1]))

    def set_source(self, fx, fy=None):
        fy = fy or (lambda x, y: np.zeros_like(x))
        self.fr.interpolate(lambda x: np.vstack([
            fx(x[0], x[1]), fy(x[0], x[1]),
            np.zeros(x.shape[1]), np.zeros(x.shape[1])]))

    def set_region(self, fn):
        self.wsel.interpolate(lambda x: fn(x[0], x[1]))

    def set_theta(self, arr):
        self.theta.x.array[:] = np.asarray(arr, float)

    def set_zfac(self, fn):
        self.zfac.interpolate(lambda x: fn(x[0], x[1]))

    def store_reference(self):
        self.Uref.x.array[:] = self.U.x.array[:]

    # ---- solve -------------------------------------------------------- #
    def solve(self):
        A = assemble_matrix(self.a_form, bcs=[]); A.assemble()
        b = assemble_vector(self.L_form)
        self._A = A
        _lu(A, self.domain.comm).solve(b, self.U.vector)
        self.U.x.scatter_forward()
        return self.U

    # ---- objectives --------------------------------------------------- #
    def focus_energy(self):
        u2 = sum(self.U[i]*self.U[i] for i in range(4))
        return self.domain.comm.allreduce(
            fem.assemble_scalar(fem.form(self.wsel * u2 * ufl.dx)), op=_SUM())

    def total_energy(self):
        u2 = sum(self.U[i]*self.U[i] for i in range(4))
        return self.domain.comm.allreduce(
            fem.assemble_scalar(fem.form(u2 * ufl.dx)), op=_SUM())

    def cloak_mismatch(self):
        d = self.U - self.Uref
        j = sum(d[i]*d[i] for i in range(4))
        return self.domain.comm.allreduce(
            fem.assemble_scalar(fem.form(self.wsel * j * ufl.dx)), op=_SUM())

    # ---- discrete-adjoint sensitivity dJ/dtheta ----------------------- #
    def _sensitivity(self, Jexpr):
        P = ufl.TestFunction(self.W)
        Jform = self.wsel * Jexpr(self.U) * ufl.dx
        g = assemble_vector(fem.form(ufl.derivative(Jform, self.U, P)))
        ksp = _lu(self._A.transpose(), self.domain.comm)
        neg = g.copy(); neg.scale(-1.0)
        ksp.solve(neg, self.Lam.vector); self.Lam.x.scatter_forward()
        w = self.omega; et = self.eta
        D = self._zp() * self._Dmat(self.theta)
        rho = self._rho_eff()
        Uu, Lm = self.U, self.Lam
        ur = ufl.as_vector([Uu[0], Uu[1]]); ui = ufl.as_vector([Uu[2], Uu[3]])
        lr = ufl.as_vector([Lm[0], Lm[1]]); li = ufl.as_vector([Lm[2], Lm[3]])
        kk = lambda a, b: self._stiff(a, b, D)
        ms = lambda a, b: rho * ufl.dot(a, b) * ufl.dx
        dp = lambda a, b: self.cspg * ufl.dot(a, b) * ufl.dx
        R = (kk(ur, lr) - w**2*ms(ur, lr) - et*kk(ui, lr) - w*dp(ui, lr)
             + kk(ui, li) - w**2*ms(ui, li) + et*kk(ur, li) + w*dp(ur, li)
             - ufl.inner(self.fr, Lm)*ufl.dx)
        dR = fem.form(ufl.derivative(R, self.theta, ufl.TestFunction(self.S)))
        return assemble_vector(dR).array.copy()

    def focus_grad(self):
        return self._sensitivity(lambda U: sum(U[i]*U[i] for i in range(4)))

    def cloak_grad(self):
        Ur = self.Uref
        return self._sensitivity(
            lambda U: sum((U[i]-Ur[i])*(U[i]-Ur[i]) for i in range(4)))


def _SUM():
    from mpi4py import MPI
    return MPI.SUM
