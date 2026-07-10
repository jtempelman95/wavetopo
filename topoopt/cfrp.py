"""
Toolpath-integrated topology optimization of continuous fiber-reinforced
polymer (CFRP) structures.

Reimplementation of the method of:

    J. Wong, E. D. Sanders, D. W. Rosen,
    "Toolpath-integrated topology optimization for design of additively
    manufactured fiber-reinforced structures considering limits on fiber
    curvature", Composite Structures 378 (2026) 119897.

Pure numpy/scipy, structured bilinear-quad mesh (top88-style), so it mirrors
the paper's MATLAB implementation and is easy to gradient-check.

Design fields
-------------
z      : density design variables, one per element            (length N)
theta_hat : fiber orientation at CS-RBF support points        (length M)

Pipeline (forward model)
------------------------
z --filter(P)--> y --SIMP--> E                       (composite stiffness scale)
theta_hat --CS-RBF(Phi)--> theta                     (continuous orientation)
(z, theta) --wave projection--> chi_hat              (fiber=1 / matrix=0 state)
K = sum_l E_l [ chi_hat_l kf(theta_l) + (1-chi_hat_l) km ]
f = F^T u,   K u = F

Equation numbers in comments refer to the paper.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ====================================================================== #
#  Materials                                                             #
# ====================================================================== #
class Material:
    """Plane-stress fiber (orthotropic) + matrix (isotropic) constitutive data."""

    def __init__(self, Ef1=131e9, Ef2=9e9, G12=5e9, nu12=0.27,
                 Em=2.6e9, nu_m=0.3, rho_f=1.6, rho_m=1.2):
        self.Ef1, self.Ef2, self.G12, self.nu12 = Ef1, Ef2, G12, nu12
        self.Em, self.nu_m = Em, nu_m
        # mass densities (used for dynamic / band-structure problems)
        self.rho_f, self.rho_m = rho_f, rho_m

    # --- fiber stiffness in material axes, eq (7)-(8) ------------------ #
    @property
    def Qf(self):
        nu12, Ef1, Ef2, G12 = self.nu12, self.Ef1, self.Ef2, self.G12
        nu21 = nu12 * Ef2 / Ef1
        den = 1.0 - nu12 * nu21
        Q11 = Ef1 / den
        Q12 = nu12 * Ef2 / den
        Q22 = Ef2 / den
        Q66 = G12
        return np.array([[Q11, Q12, 0.0],
                         [Q12, Q22, 0.0],
                         [0.0, 0.0, Q66]])

    # --- isotropic matrix plane-stress stiffness ----------------------- #
    @property
    def Dm(self):
        Em, nu = self.Em, self.nu_m
        return Em / (1 - nu**2) * np.array([[1.0, nu, 0.0],
                                            [nu, 1.0, 0.0],
                                            [0.0, 0.0, (1 - nu) / 2]])


def T_matrix(theta):
    """Voigt stiffness transformation matrix T(theta), eq (6)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c*c,      s*s,     -2*s*c],
                     [s*s,      c*c,      2*s*c],
                     [s*c,     -s*c,      c*c - s*s]])


def Df_of_theta(theta, Qf):
    """Rotated fiber stiffness Df(theta) = T Qf T^T, eq (6)."""
    T = T_matrix(theta)
    return T @ Qf @ T.T


# ====================================================================== #
#  Structured bilinear-quad mesh + element stiffness                     #
# ====================================================================== #
class QuadMesh:
    """Regular grid of bilinear quads on [0,Lx] x [0,Ly]."""

    def __init__(self, nelx, nely, Lx, Ly):
        self.nelx, self.nely = nelx, nely
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx / nelx, Ly / nely
        self.N = nelx * nely
        self.nnode = (nelx + 1) * (nely + 1)
        self.ndof = 2 * self.nnode

        # element centroids (design points), element index e = ey*nelx + ex
        ex = np.arange(nelx)
        ey = np.arange(nely)
        EX, EY = np.meshgrid(ex, ey)            # shape (nely, nelx)
        self.cx = ((EX + 0.5) * self.dx).ravel()
        self.cy = ((EY + 0.5) * self.dy).ravel()
        self.area = self.dx * self.dy
        self.A = np.full(self.N, self.area)     # element volumes A_l

        # connectivity / edof (8 dof per element)
        self.edof = np.zeros((self.N, 8), dtype=np.int64)
        for e in range(self.N):
            exi, eyi = e % nelx, e // nelx
            n1 = eyi * (nelx + 1) + exi
            n2 = n1 + 1
            n3 = n2 + (nelx + 1)
            n4 = n1 + (nelx + 1)
            nodes = [n1, n2, n3, n4]
            self.edof[e] = np.array([[2*n, 2*n+1] for n in nodes]).ravel()

        # precompute sparse assembly index pattern
        iK = np.kron(self.edof, np.ones((8, 1), dtype=np.int64)).ravel()
        jK = np.kron(self.edof, np.ones((1, 8), dtype=np.int64)).ravel()
        self.iK, self.jK = iK, jK

    def node_xy(self):
        xs = np.linspace(0, self.Lx, self.nelx + 1)
        ys = np.linspace(0, self.Ly, self.nely + 1)
        X, Y = np.meshgrid(xs, ys)
        return X.ravel(), Y.ravel()


def element_stiffness(D, dx, dy):
    """8x8 stiffness of one bilinear quad with constitutive matrix D (Voigt).

    Linear in D, so used both for the matrix km and the Fourier coefficient
    matrices [C]_s.
    """
    g = 1.0 / np.sqrt(3.0)
    gp = [(-g, -g), (g, -g), (g, g), (-g, g)]   # 2x2 Gauss points
    Ke = np.zeros((8, 8))
    # shape-function natural derivatives at (xi,eta)
    for xi, eta in gp:
        dN_dxi = 0.25 * np.array([-(1 - eta),  (1 - eta),
                                  (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi),
                                   (1 + xi),  (1 - xi)])
        # Jacobian for axis-aligned rectangle: diag(dx/2, dy/2)
        J = np.array([[dx / 2, 0.0], [0.0, dy / 2]])
        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)
        dN_dx = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
        dN_dy = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta
        B = np.zeros((3, 8))
        B[0, 0::2] = dN_dx
        B[1, 1::2] = dN_dy
        B[2, 0::2] = dN_dy
        B[2, 1::2] = dN_dx
        Ke += (B.T @ D @ B) * detJ
    return Ke


def element_mass(dx, dy, rho=1.0):
    """8x8 consistent mass matrix of one bilinear quad (unit thickness)."""
    g = 1.0 / np.sqrt(3.0)
    gp = [(-g, -g), (g, -g), (g, g), (-g, g)]
    Me = np.zeros((8, 8))
    for xi, eta in gp:
        N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        Nmat = np.zeros((2, 8))
        Nmat[0, 0::2] = N
        Nmat[1, 1::2] = N
        detJ = (dx / 2) * (dy / 2)
        Me += rho * (Nmat.T @ Nmat) * detJ
    return Me


# ====================================================================== #
#  Fourier-series precomputation of element stiffness, eq (9)-(10)        #
# ====================================================================== #
class FourierStiffness:
    """Precompute element matrices so kf(theta) = sum_s kf_s * c_s(theta).

    basis  c(theta)   = [1, cos(2 theta), sin(2 theta), cos(4 theta), sin(4 theta)]
    (period T = pi, h = 2pi/T = 2).  The fit of each component of
    Df(theta) = T Qf T^T onto this basis is exact.
    """

    def __init__(self, material: Material, mesh: QuadMesh, nsamp=720):
        self.mat = material
        self.mesh = mesh
        Qf = material.Qf

        thetas = np.linspace(-np.pi, np.pi, nsamp, endpoint=False)
        Bmat = self._basis(thetas)                         # (nsamp, 5)
        # stack the 6 independent components of Df over samples
        comps = np.zeros((nsamp, 3, 3))
        for i, th in enumerate(thetas):
            comps[i] = Df_of_theta(th, Qf)
        # least squares fit per component -> coefficient matrices C_s (3x3)
        C = np.zeros((5, 3, 3))
        coef, *_ = np.linalg.lstsq(Bmat, comps.reshape(nsamp, 9), rcond=None)
        C = coef.reshape(5, 3, 3)
        self.C = C
        # verify fit quality
        recon = (Bmat @ coef).reshape(nsamp, 3, 3)
        self.fit_err = np.max(np.abs(recon - comps)) / np.max(np.abs(comps))

        # precompute element matrices for each basis term and the matrix
        self.kf_s = np.array([element_stiffness(C[s], mesh.dx, mesh.dy)
                              for s in range(5)])           # (5,8,8)
        self.km = element_stiffness(material.Dm, mesh.dx, mesh.dy)

    @staticmethod
    def _basis(theta):
        theta = np.atleast_1d(theta)
        return np.stack([np.ones_like(theta),
                         np.cos(2 * theta), np.sin(2 * theta),
                         np.cos(4 * theta), np.sin(4 * theta)], axis=-1)

    @staticmethod
    def _dbasis(theta):
        theta = np.atleast_1d(theta)
        z = np.zeros_like(theta)
        return np.stack([z,
                         -2 * np.sin(2 * theta), 2 * np.cos(2 * theta),
                         -4 * np.sin(4 * theta), 4 * np.cos(4 * theta)],
                        axis=-1)

    def kf(self, theta):
        """Fiber element matrix at orientation theta -> (..,8,8)."""
        c = self._basis(theta)                              # (...,5)
        return np.einsum('...s,sij->...ij', c, self.kf_s)

    def dkf_dtheta(self, theta):
        c = self._dbasis(theta)
        return np.einsum('...s,sij->...ij', c, self.kf_s)


# ====================================================================== #
#  CS-RBF orientation mapping, eq (3)-(4), (20)-(21)                      #
# ====================================================================== #
class CSRBFMapping:
    """Map orientation at M support points to N design points via Wendland's
    compactly-supported RBF, with analytic spatial-derivative matrices used
    for the curl constraints.

    Builds sparse N x M matrices:
        Phi   (theta = Phi @ theta_hat)
        Phi_x = d Phi / dx
        Phi_y = d Phi / dy
    """

    def __init__(self, mesh: QuadMesh, support_xy, r_s, rho=1e-6, period=None):
        from scipy.spatial import cKDTree

        self.mesh = mesh
        self.r_s = r_s
        self.rho = rho
        self.period = period            # (Lx, Ly) -> toroidal (tileable) cell
        sx, sy = support_xy
        self.sx, self.sy = np.asarray(sx), np.asarray(sy)
        self.M = self.sx.size
        N = mesh.N

        dpts = np.column_stack([mesh.cx, mesh.cy])
        spts = np.column_stack([self.sx, self.sy])
        if period is not None:
            tree = cKDTree(spts, boxsize=list(period))
        else:
            tree = cKDTree(spts)
        # design points within support radius of each support point
        neigh = tree.query_ball_point(dpts, r_s)

        rows, cols = [], []
        phi_v, phix_v, phiy_v = [], [], []
        for l, qs in enumerate(neigh):
            for q in qs:
                dxq = mesh.cx[l] - self.sx[q]
                dyq = mesh.cy[l] - self.sy[q]
                if period is not None:
                    # minimum-image displacement for a periodic cell
                    dxq -= period[0] * np.round(dxq / period[0])
                    dyq -= period[1] * np.round(dyq / period[1])
                r = np.sqrt(dxq**2 + dyq**2 + rho**2) / r_s
                if r >= 1.0:
                    continue
                base = max(0.0, 1.0 - r)
                phi = (4 * r + 1) * base**4
                # eq (20)-(21): phi_q,x = -20 (x_l-x_q)/r_s^2 (1-r)^3
                phi_x = -20.0 * dxq / r_s**2 * base**3
                phi_y = -20.0 * dyq / r_s**2 * base**3
                rows.append(l); cols.append(q)
                phi_v.append(phi); phix_v.append(phi_x); phiy_v.append(phi_y)

        shape = (N, self.M)
        Praw = sp.csr_matrix((phi_v, (rows, cols)), shape=shape)
        Pxraw = sp.csr_matrix((phix_v, (rows, cols)), shape=shape)
        Pyraw = sp.csr_matrix((phiy_v, (rows, cols)), shape=shape)

        s = np.asarray(Praw.sum(axis=1)).ravel()            # sum_s phi_s(x_l)
        sx_ = np.asarray(Pxraw.sum(axis=1)).ravel()          # sum_s phi_s,x
        sy_ = np.asarray(Pyraw.sum(axis=1)).ravel()
        s = np.where(s <= 0, 1.0, s)                         # guard empty rows

        inv_s = sp.diags(1.0 / s)
        # normalized Phi, eq (3)
        self.Phi = inv_s @ Praw
        # quotient rule, eq (20)-(21):
        #   Phi_x = phi_x/s - phi*(s_x)/s^2
        inv_s2 = sp.diags(1.0 / s**2)
        self.Phi_x = inv_s @ Pxraw - sp.diags(sx_) @ inv_s2 @ Praw
        self.Phi_y = inv_s @ Pyraw - sp.diags(sy_) @ inv_s2 @ Praw

    def theta(self, theta_hat):
        return self.Phi @ theta_hat


def grid_support_points(mesh: QuadMesh, dx_s, dy_s, pad=True):
    """Evenly spaced grid of CS-RBF support points over the domain."""
    nx = max(2, int(round(mesh.Lx / dx_s)) + 1)
    ny = max(2, int(round(mesh.Ly / dy_s)) + 1)
    xs = np.linspace(0, mesh.Lx, nx)
    ys = np.linspace(0, mesh.Ly, ny)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


# ====================================================================== #
#  Heaviside projection helpers, eq (14)/(17)                            #
# ====================================================================== #
def heaviside(x, eta, beta):
    """Smooth Heaviside projection, eq (14)/(17)."""
    num = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / den


def dheaviside(x, eta, beta):
    """d/dx of heaviside(), eq (35)/(39)."""
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta * (1 - np.tanh(beta * (x - eta))**2) / den


# ====================================================================== #
#  Wave projection: orientation -> fiber/matrix material state           #
#  Sections 3, eq (11)-(17); adjoint uses eq (35)-(38),(45)              #
# ====================================================================== #
class WaveProjection:
    """Turn (density z, orientation theta) into a discrete fiber/matrix state
    chi_hat in [0,1]^N via the wave-projection analogy (Rumpf & Pazos, Ren).

    Forward:  alpha(z) -> grad_psi(alpha,theta) -> psi (LS solve)
              -> chi (cosine wave) -> chi_hat (Heaviside).
    """

    def __init__(self, mesh: QuadMesh, d, beta=100.0, eta=0.5,
                 eta_alpha=0.5, anchor=None):
        self.mesh = mesh
        self.d = d
        self.beta = beta
        self.eta = eta
        self.eta_alpha = eta_alpha
        # anchor = top-left element (max y, min x)
        self.anchor = anchor if anchor is not None else (mesh.nely - 1) * mesh.nelx

        self.Ax, self.Ay = self._build_fd_matrices()
        # constant normal-equations operator G = Ax^T Ax + Ay^T Ay
        G = (self.Ax.T @ self.Ax + self.Ay.T @ self.Ay).tocsc()
        N = mesh.N
        free = np.setdiff1d(np.arange(N), [self.anchor])
        self.free = free
        self.G = G
        self.G_ff = G[free][:, free].tocsc()
        self.lu = spla.splu(self.G_ff)
        self._cache = {}

    # ---- centered finite-difference matrices on the structured grid --- #
    def _build_fd_matrices(self):
        m = self.mesh
        nelx, nely, N = m.nelx, m.nely, m.N
        dx, dy = m.dx, m.dy

        def idx(ex, ey):
            return ey * nelx + ex

        rx, cx, vx = [], [], []
        ry, cy, vy = [], [], []
        for ey in range(nely):
            for ex in range(nelx):
                e = idx(ex, ey)
                # x derivative
                if 0 < ex < nelx - 1:
                    rx += [e, e]; cx += [idx(ex+1, ey), idx(ex-1, ey)]
                    vx += [1/(2*dx), -1/(2*dx)]
                elif ex == 0:
                    rx += [e, e]; cx += [idx(ex+1, ey), e]
                    vx += [1/dx, -1/dx]
                else:
                    rx += [e, e]; cx += [e, idx(ex-1, ey)]
                    vx += [1/dx, -1/dx]
                # y derivative
                if 0 < ey < nely - 1:
                    ry += [e, e]; cy += [idx(ex, ey+1), idx(ex, ey-1)]
                    vy += [1/(2*dy), -1/(2*dy)]
                elif ey == 0:
                    ry += [e, e]; cy += [idx(ex, ey+1), e]
                    vy += [1/dy, -1/dy]
                else:
                    ry += [e, e]; cy += [e, idx(ex, ey-1)]
                    vy += [1/dy, -1/dy]
        Ax = sp.csr_matrix((vx, (rx, cx)), shape=(N, N))
        Ay = sp.csr_matrix((vy, (ry, cy)), shape=(N, N))
        return Ax, Ay

    def _solve_pinned(self, rhs):
        """Solve G_ff x_f = rhs_f, x_anchor = 0."""
        x = np.zeros(self.mesh.N)
        x[self.free] = self.lu.solve(rhs[self.free])
        return x

    # ---- forward ------------------------------------------------------ #
    def forward(self, z, theta):
        beta = self.beta
        # alpha, eq (14)
        alpha = heaviside(z, self.eta_alpha, beta)
        # grad of phase, eq (13)
        gx = -alpha * np.sin(theta)
        gy = alpha * np.cos(theta)
        # least-squares phase, eq (15): G psi = A^T grad_psi
        rhs = self.Ax.T @ gx + self.Ay.T @ gy
        psi = self._solve_pinned(rhs)
        psi0 = psi[self.anchor]                  # = 0 by pinning
        # wave, eq (16)
        phase = (2 * np.pi / self.d) * (psi - psi0)
        chi = 0.5 + 0.5 * np.cos(phase)
        # material state, eq (17)
        chi_hat = heaviside(chi, self.eta, beta)

        self._cache = dict(z=z, theta=theta, alpha=alpha, gx=gx, gy=gy,
                           psi=psi, psi0=psi0, chi=chi, chi_hat=chi_hat)
        return chi_hat

    # ---- adjoint: given df/dchi_hat, return df/dz and df/dtheta -------- #
    def adjoint(self, df_dchihat):
        c = self._cache
        beta = self.beta
        theta = c['theta']
        # eq (35): d chi_hat / d chi
        dchihat_dchi = dheaviside(c['chi'], self.eta, beta)
        # eq (36): d chi / d psi
        dchi_dpsi = -(np.pi / self.d) * np.sin(
            (2 * np.pi / self.d) * (c['psi'] - c['psi0']))
        lam = df_dchihat * dchihat_dchi * dchi_dpsi          # = df/dpsi
        # adjoint of LS phase solve, eq (37): mu = (G_ff)^-1 lam
        mu = self._solve_pinned(lam)
        # df/d grad_psi = A mu
        adj_gx = self.Ax @ mu
        adj_gy = self.Ay @ mu
        # via alpha, eq (38): d grad_psi / d alpha = [-sin th, cos th]
        df_dalpha = -adj_gx * np.sin(theta) + adj_gy * np.cos(theta)
        dalpha_dz = dheaviside(c['z'], self.eta_alpha, beta)  # eq (39)
        df_dz = df_dalpha * dalpha_dz
        # via theta, eq (45): d grad_psi / d theta = [-alpha cos th, -alpha sin th]
        df_dtheta = -c['alpha'] * (adj_gx * np.cos(theta)
                                   + adj_gy * np.sin(theta))
        return df_dz, df_dtheta


if __name__ == "__main__":
    # quick self-test of the forward constitutive model
    mat = Material()
    print("Qf (GPa):\n", mat.Qf / 1e9)
    print("Dm (GPa):\n", mat.Dm / 1e9)

    # fiber at 0deg: Df should equal Qf; D11 >> D22 (stiff along x)
    D0 = Df_of_theta(0.0, mat.Qf)
    print("\nDf(0) == Qf ?", np.allclose(D0, mat.Qf))
    # fiber at 90deg: D11 and D22 swap
    D90 = Df_of_theta(np.pi / 2, mat.Qf)
    print("Df(90) D11≈Qf22 ?", np.isclose(D90[0, 0], mat.Qf[1, 1]))

    mesh = QuadMesh(4, 3, 2.4, 1.5)
    fs = FourierStiffness(mat, mesh)
    print("\nFourier fit relative error:", fs.fit_err)
    # check reconstruction of kf at a random angle vs direct integration
    th = 0.37
    kf_direct = element_stiffness(Df_of_theta(th, mat.Qf), mesh.dx, mesh.dy)
    kf_fourier = fs.kf(th)
    print("kf Fourier vs direct max rel err:",
          np.max(np.abs(kf_direct - kf_fourier)) / np.max(np.abs(kf_direct)))
    # derivative check via finite difference
    h = 1e-6
    dk_fd = (fs.kf(th + h) - fs.kf(th - h)) / (2 * h)
    dk_an = fs.dkf_dtheta(th)
    print("dkf/dtheta max rel err:",
          np.max(np.abs(dk_fd - dk_an)) / np.max(np.abs(dk_an)))

    # ---- CS-RBF mapping + spatial derivative verification ------------- #
    print("\n--- CS-RBF mapping ---")
    meshC = QuadMesh(40, 25, 2.4, 1.5)
    sx, sy = grid_support_points(meshC, 0.16, 0.167)
    rbf = CSRBFMapping(meshC, (sx, sy), r_s=0.33)
    print("M support points:", rbf.M, " N design points:", meshC.N)
    print("Phi row sums (should be 1):",
          np.allclose(np.asarray(rbf.Phi.sum(1)).ravel(), 1.0))
    # finite-difference check of Phi_x, Phi_y by perturbing design-point coords
    # build Phi at shifted centroids using a fresh mapping
    eps = 1e-6

    def phi_dense_at(cx, cy):
        mtmp = QuadMesh(meshC.nelx, meshC.nely, meshC.Lx, meshC.Ly)
        mtmp.cx, mtmp.cy = cx, cy
        return CSRBFMapping(mtmp, (sx, sy), r_s=0.33).Phi

    Pxp = phi_dense_at(meshC.cx + eps, meshC.cy)
    Pxm = phi_dense_at(meshC.cx - eps, meshC.cy)
    Phix_fd = (Pxp - Pxm) / (2 * eps)
    err_x = np.max(np.abs((Phix_fd - rbf.Phi_x).toarray()))
    Pyp = phi_dense_at(meshC.cx, meshC.cy + eps)
    Pym = phi_dense_at(meshC.cx, meshC.cy - eps)
    Phiy_fd = (Pyp - Pym) / (2 * eps)
    err_y = np.max(np.abs((Phiy_fd - rbf.Phi_y).toarray()))
    print("Phi_x max abs err vs FD:", err_x)
    print("Phi_y max abs err vs FD:", err_y)

    # ---- Wave projection forward + adjoint verification --------------- #
    print("\n--- Wave projection ---")
    rng = np.random.default_rng(0)
    meshW = QuadMesh(24, 15, 2.4, 1.5)
    sxW, syW = grid_support_points(meshW, 0.24, 0.25)
    rbfW = CSRBFMapping(meshW, (sxW, syW), r_s=0.5)
    wp = WaveProjection(meshW, d=0.12, beta=20.0)   # softer beta for smooth FD
    z = 0.5 + 0.3 * rng.standard_normal(meshW.N)
    z = np.clip(z, 0.05, 1.0)
    theta_hat = 0.3 * rng.standard_normal(rbfW.M)
    theta = rbfW.theta(theta_hat)
    chi_hat = wp.forward(z, theta)
    print("chi_hat range:", chi_hat.min(), chi_hat.max(),
          " fiber frac:", chi_hat.mean())
    # scalar objective g = w . chi_hat
    w = rng.standard_normal(meshW.N)
    g0 = w @ chi_hat
    dz_an, dth_an = wp.adjoint(w)
    # FD wrt z
    epsf = 1e-6
    k = 7
    zp = z.copy(); zp[k] += epsf
    zm = z.copy(); zm[k] -= epsf
    gzp = w @ wp.forward(zp, theta)
    gzm = w @ wp.forward(zm, theta)
    dz_fd = (gzp - gzm) / (2 * epsf)
    print(f"df/dz[{k}]  analytic={dz_an[k]:.6e}  FD={dz_fd:.6e}")
    # FD wrt theta (a design-point orientation)
    j = 50
    thp = theta.copy(); thp[j] += epsf
    thm = theta.copy(); thm[j] -= epsf
    gtp = w @ wp.forward(z, thp)
    gtm = w @ wp.forward(z, thm)
    dth_fd = (gtp - gtm) / (2 * epsf)
    print(f"df/dth[{j}] analytic={dth_an[j]:.6e}  FD={dth_fd:.6e}")
    # restore cache to consistent state
    wp.forward(z, theta)
