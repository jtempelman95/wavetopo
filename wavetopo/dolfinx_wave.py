r"""
dolfinx (FEniCSx) finite-element wave-control on continuous-fiber composites.

Scalar (antiplane-shear) time-harmonic elastodynamics with a spatially varying,
anisotropic fiber shear tensor whose ORIENTATION is the design field.  The PETSc
build here is real, so the complex harmonic problem

    [ K(theta) - w^2 M + i w C ] u = f,     u = u_r + i u_i

is solved as the equivalent real 2x2 block system on U=(u_r,u_i):

    [ A  -B ] [u_r]   [f_r]
    [ B   A ] [u_i] = [f_i],   A = K(theta)-w^2 M,   B = w C ,

with  K(theta) = int mu(theta) grad . grad,  M = int rho (.)(.), and C the
mass-proportional sponge (absorbing layer) emulating an open domain.  The
anisotropic shear tensor

    mu(theta) = a I + b [[cos2t, sin2t],[sin2t,-cos2t]],
    a=(muL+muT)/2, b=(muL-muT)/2 ,

is theta-independent when muL=muT (isotropic): anisotropy (b!=0) is exactly what
lets the fiber toolpath steer the wave.

Objectives:
    localization : maximize  E = int_focus |u|^2
    cloak        : minimize  J = int_obs |u - u_ref|^2

Sensitivities w.r.t. the DG0 orientation field use the discrete adjoint via UFL
differentiation of the residual (finite-difference verified).
"""
from __future__ import annotations

import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector


def mu_tensor(theta, muL, muT):
    a = 0.5 * (muL + muT); b = 0.5 * (muL - muT)
    c2, s2 = ufl.cos(2 * theta), ufl.sin(2 * theta)
    return ufl.as_matrix([[a + b * c2, b * s2], [b * s2, a - b * c2]])


def cell_filter(cents, R):
    """Linear (cone-weight) neighborhood-average filter over cell centroids,
    used to regularize the per-cell orientation field so the resulting fiber
    toolpaths are smooth and manufacturable.  Returns a row-stochastic sparse
    operator P (physical = P @ design; chain rule uses P.T)."""
    from scipy.spatial import cKDTree
    import scipy.sparse as sp
    tree = cKDTree(cents); pairs = tree.query_pairs(R, output_type="ndarray")
    n = cents.shape[0]
    rows = list(range(n)); cols = list(range(n)); w = [R]*n
    for i, j in pairs:
        d = np.hypot(*(cents[i]-cents[j])); wij = R-d
        rows += [i, j]; cols += [j, i]; w += [wij, wij]
    W = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    return sp.diags(1.0/np.asarray(W.sum(1)).ravel()) @ W


def support_map(cents, xlim, ylim, spacing, r, supp=None):
    """Smooth low-dimensional orientation parametrization: interpolate a coarse
    grid of support-point angles to the cells with a compactly-supported
    Wendland C2 radial basis (phi(q)=(1-q)^4(4q+1), q=d/r).  Returns (B, supp)
    with physical per-cell theta = B @ x over the M support values x; the chain
    rule uses B.T.  This is the dolfinx analogue of the CS-RBF orientation map
    and gives smooth, manufacturable fiber toolpaths by construction.

    An explicit support-point array ``supp`` (M,2) may be passed (e.g. a grid
    made mirror-symmetric about a symmetry axis)."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    if supp is None:
        sx = np.arange(xlim[0], xlim[1] + 1e-9, spacing)
        sy = np.arange(ylim[0], ylim[1] + 1e-9, spacing)
        SX, SY = np.meshgrid(sx, sy)
        supp = np.column_stack([SX.ravel(), SY.ravel()])
    tree = cKDTree(supp)
    rows, cols, vals = [], [], []
    for c, p in enumerate(cents):
        idx = tree.query_ball_point(p, r)
        if not idx:
            _, i = tree.query(p); idx = [int(i)]
        for s in idx:
            q = np.hypot(*(p - supp[s])) / r
            wv = (1-q)**4 * (4*q + 1) if q < 1 else 0.0
            rows.append(c); cols.append(s); vals.append(wv + 1e-12)
    B = sp.csr_matrix((vals, (rows, cols)), shape=(cents.shape[0], supp.shape[0]))
    B = sp.diags(1.0/np.asarray(B.sum(1)).ravel()) @ B
    return B, supp


def csrbf_operators(cents, supp, r):
    """CS-RBF interpolation B and its analytic spatial-derivative operators
    Bx, By (cells x M), consistently differentiating the row normalization
    (partition of unity), so theta=B@x, d theta/dx = Bx@x, d theta/dy = By@x.
    Wendland C2 phi(q)=(1-q)^4(4q+1); phi'(q)=-20 q (1-q)^3, giving
    d phi/d(coord) = -20 (1-q)^3 (coord_c - coord_s)/r^2.  Used for the fiber
    curvature (curl) constraint zeta = cos(theta) theta_x + sin(theta) theta_y."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    tree = cKDTree(supp)
    n, M = cents.shape[0], supp.shape[0]
    rows, cols, wv, wxv, wyv = [], [], [], [], []
    for c, p in enumerate(cents):
        idx = tree.query_ball_point(p, r)
        if not idx:
            _, i = tree.query(p); idx = [int(i)]
        for s in idx:
            dx = p[0]-supp[s, 0]; dy = p[1]-supp[s, 1]
            q = np.hypot(dx, dy)/r
            if q >= 1.0:
                continue
            fac = -20.0*(1-q)**3/r**2
            rows.append(c); cols.append(s)
            wv.append((1-q)**4*(4*q+1) + 1e-12); wxv.append(fac*dx); wyv.append(fac*dy)
    shp = (n, M)
    W = sp.csr_matrix((wv, (rows, cols)), shape=shp)
    Wx = sp.csr_matrix((wxv, (rows, cols)), shape=shp)
    Wy = sp.csr_matrix((wyv, (rows, cols)), shape=shp)
    Sinv = sp.diags(1.0/np.asarray(W.sum(1)).ravel())
    B = Sinv @ W
    Bxr = Sinv @ Wx; Byr = Sinv @ Wy
    Bx = Bxr - sp.diags(np.asarray(Bxr.sum(1)).ravel()) @ B
    By = Byr - sp.diags(np.asarray(Byr.sum(1)).ravel()) @ B
    return B, Bx, By


def curl_of(x, B, Bx, By):
    """Fiber-path curvature (curl) field zeta = cos(theta) theta_x +
    sin(theta) theta_y = (n . grad) theta, with theta=B@x, theta_x=Bx@x,
    theta_y=By@x.  Returns (zeta, theta, theta_x, theta_y)."""
    th = B @ x; a = Bx @ x; b = By @ x
    return np.cos(th)*a + np.sin(th)*b, th, a, b


def curl_penalty(x, B, Bx, By, zeta_all, mask=None):
    """Mean squared-violation penalty  P = <max(zeta^2 - zeta_all^2, 0)>  and its
    gradient dP/dx.  zeta_all sets the manufacturable fiber-curvature limit; for a
    prescribed circular through-hole of radius R the paper uses zeta_all = 1/R.
    An optional boolean ``mask`` (per cell) restricts the penalty to the solid
    material -- curvature inside a void has no fibers and must not be penalized."""
    zeta, th, a, b = curl_of(x, B, Bx, By)
    m = np.ones_like(zeta, dtype=bool) if mask is None else np.asarray(mask, bool)
    viol = np.where(m, np.maximum(zeta**2 - zeta_all**2, 0.0), 0.0)
    n = max(int(m.sum()), 1)
    v = (2.0/n) * zeta * ((zeta**2 > zeta_all**2) & m)
    coef = -np.sin(th)*a + np.cos(th)*b
    dP = B.T @ (coef*v) + Bx.T @ (np.cos(th)*v) + By.T @ (np.sin(th)*v)
    return viol.sum()/n, dP


def _lu(A, comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A); ksp.setType("preonly")
    pc = ksp.getPC(); pc.setType("lu")
    return ksp


class WaveControl:
    def __init__(self, domain, *, omega, muL=6.0, muT=2.0, rho=1.6, eta=0.03):
        self.domain = domain; self.omega = omega
        self.muL, self.muT = muL, muT
        self.eta = eta                     # structural (hysteretic) damping
        self.W = fem.functionspace(domain, ("Lagrange", 1, (2,)))
        self.S = fem.functionspace(domain, ("DG", 0))
        self.theta = fem.Function(self.S)
        self.cspg = fem.Function(self.S)
        self.zfac = fem.Function(self.S); self.zfac.x.array[:] = 1.0   # density
        self.rho_f = fem.Function(self.S); self.rho_f.x.array[:] = rho
        self.fr = fem.Function(self.W)
        self.U = fem.Function(self.W)
        self.Lam = fem.Function(self.W)
        self.wsel = fem.Function(self.S)          # region indicator (focus/obs)
        self.Uref = fem.Function(self.W)          # reference (cloak)
        self._forms()

    def _forms(self):
        w = self.omega; zeps = 1e-4
        zf = zeps + (1 - zeps) * self.zfac                 # SIMP-like density
        mu = zf * mu_tensor(self.theta, self.muL, self.muT)
        rho, c = self.rho_f, self.cspg
        U = ufl.TrialFunction(self.W); P = ufl.TestFunction(self.W)
        ur, ui, pr, pi = U[0], U[1], P[0], P[1]
        st = lambda a, b: ufl.inner(mu * ufl.grad(a), ufl.grad(b)) * ufl.dx
        ms = lambda a, b: rho * a * b * ufl.dx
        dp = lambda a, b: c * a * b * ufl.dx
        et = self.eta
        # D = (1+i eta) K - w^2 M + i w C ;  block [A -B; B A],
        # A = K - w^2 M,  B = eta K + w C
        aE = (st(ur, pr) - w**2*ms(ur, pr) - et*st(ui, pr) - w*dp(ui, pr)
              + st(ui, pi) - w**2*ms(ui, pi) + et*st(ur, pi) + w*dp(ur, pi))
        self.a_form = fem.form(aE)
        self.L_form = fem.form(ufl.inner(self.fr, P) * ufl.dx)

    # ---- coefficient setters (interpolate callables) ------------------ #
    def set_sponge(self, fn):
        self.cspg.interpolate(lambda x: fn(x[0], x[1]))

    def set_source(self, fx):
        self.fr.interpolate(lambda x: np.vstack([fx(x[0], x[1]),
                                                 np.zeros(x.shape[1])]))

    def set_region(self, fn):
        self.wsel.interpolate(lambda x: fn(x[0], x[1]))

    def set_theta(self, arr):
        self.theta.x.array[:] = np.asarray(arr, float)

    def set_zfac(self, fn):
        self.zfac.interpolate(lambda x: fn(x[0], x[1]))

    def store_reference(self):
        self.Uref.x.array[:] = self.U.x.array[:]

    # ---- forward solve ------------------------------------------------ #
    def solve(self):
        A = assemble_matrix(self.a_form, bcs=[]); A.assemble()
        b = assemble_vector(self.L_form)
        self._ksp = _lu(A, self.domain.comm); self._A = A
        self._ksp.solve(b, self.U.vector); self.U.x.scatter_forward()
        return self.U

    # ---- objectives --------------------------------------------------- #
    def focus_energy(self):
        ur, ui = self.U[0], self.U[1]
        form = fem.form(self.wsel * (ur*ur + ui*ui) * ufl.dx)
        return self.domain.comm.allreduce(fem.assemble_scalar(form), op=_SUM())

    def total_energy(self):
        ur, ui = self.U[0], self.U[1]
        form = fem.form((ur*ur + ui*ui) * ufl.dx)
        return self.domain.comm.allreduce(fem.assemble_scalar(form), op=_SUM())

    def cloak_mismatch(self):
        d = self.U - self.Uref
        form = fem.form(self.wsel * (d[0]*d[0] + d[1]*d[1]) * ufl.dx)
        return self.domain.comm.allreduce(fem.assemble_scalar(form), op=_SUM())

    # ---- adjoint sensitivity dJ/dtheta (DG0) -------------------------- #
    def _sensitivity(self, Jexpr_of_U):
        """Return dJ/dtheta as a numpy array over DG0 cells, for objective
        J = int wsel * Jexpr(U) dx.  Uses the discrete adjoint."""
        P = ufl.TestFunction(self.W)
        # dJ/dU as a linear form in the test function P
        Jform = self.wsel * Jexpr_of_U(self.U) * ufl.dx
        dJdU = fem.form(ufl.derivative(Jform, self.U, P))
        g = assemble_vector(dJdU)                 # vector
        # adjoint:  A^T Lam = -g
        AT = self._A.transpose()
        ksp = _lu(AT, self.domain.comm)
        neg = g.copy(); neg.scale(-1.0)
        ksp.solve(neg, self.Lam.vector); self.Lam.x.scatter_forward()
        # residual with U (solution) and Lam (adjoint) as test:  R = a(U,Lam)-L(Lam)
        zeps = 1e-4
        mu = (zeps + (1-zeps)*self.zfac) * mu_tensor(self.theta, self.muL, self.muT)
        w = self.omega; rho, c = self.rho_f, self.cspg
        Uu, Lm = self.U, self.Lam
        ur, ui, lr, li = Uu[0], Uu[1], Lm[0], Lm[1]
        st = lambda a, b: ufl.inner(mu*ufl.grad(a), ufl.grad(b))*ufl.dx
        ms = lambda a, b: rho*a*b*ufl.dx
        dp = lambda a, b: c*a*b*ufl.dx
        et = self.eta
        R = (st(ur, lr) - w**2*ms(ur, lr) - et*st(ui, lr) - w*dp(ui, lr)
             + st(ui, li) - w**2*ms(ui, li) + et*st(ur, li) + w*dp(ur, li)
             - ufl.inner(self.fr, Lm)*ufl.dx)
        dRdth = fem.form(ufl.derivative(R, self.theta, ufl.TestFunction(self.S)))
        return assemble_vector(dRdth).array.copy()

    def focus_grad(self):
        # dE/dtheta with E = int wsel |u|^2 (gradient of the maximized energy)
        return self._sensitivity(lambda U: U[0]*U[0] + U[1]*U[1])

    def cloak_grad(self):
        Ur = self.Uref
        return self._sensitivity(lambda U: (U[0]-Ur[0])**2 + (U[1]-Ur[1])**2)


def _SUM():
    from mpi4py import MPI
    return MPI.SUM
