"""
Toolpath-integrated CFRP topology optimization problem.

Ties together the primitives in ``cfrp.py`` (anisotropic Fourier-precomputed
FE, CS-RBF orientation mapping, wave projection) into a single object that
exposes, for design variables (z, theta_hat):

    objective f (compliance) and df/dz, df/dtheta_hat
    constraints  g = [vg, vfg, curl_1..curl_N]  and their gradients

All equation numbers refer to Wong, Sanders & Rosen, Compos. Struct. 378 (2026)
119897.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from .cfrp import (Material, QuadMesh, FourierStiffness, CSRBFMapping,
                   WaveProjection, grid_support_points)


# ---------------------------------------------------------------------- #
#  Linear ("hat") density filter, eq (2)                                 #
# ---------------------------------------------------------------------- #
def density_filter(mesh: QuadMesh, R, period=None):
    """Sparse regularization map P with P_ij = w_ij A_j / sum_k w_ik A_k.

    If ``period=(Lx, Ly)`` the filter wraps around the cell (toroidal
    convolution), so the physical density is periodic / seamlessly tileable.
    """
    cents = np.column_stack([mesh.cx, mesh.cy])
    if period is not None:
        tree = cKDTree(cents, boxsize=list(period))
    else:
        tree = cKDTree(cents)
    pairs = tree.query_pairs(R, output_type="ndarray")
    rows = list(range(mesh.N)); cols = list(range(mesh.N)); w = [R] * mesh.N
    for i, j in pairs:
        delta = np.abs(cents[i] - cents[j])
        if period is not None:
            delta = np.minimum(delta, np.array(period) - delta)
        d = np.linalg.norm(delta)
        wij = R - d
        rows += [i, j]; cols += [j, i]; w += [wij, wij]
    W = sp.csr_matrix((w, (rows, cols)), shape=(mesh.N, mesh.N))
    WA = W @ sp.diags(mesh.A)
    rsum = np.asarray(WA.sum(1)).ravel()
    P = sp.diags(1.0 / rsum) @ WA
    return P.tocsr()


class CFRPProblem:
    def __init__(self, mesh, material, support_xy, *, R, r_s, d,
                 beta=100.0, eta=0.5, eta_alpha=0.5, penal=3.0, eps=1e-4,
                 v_all=0.5, vf_all=None, use_alpha=True, nsamp=720):
        self.mesh = mesh
        self.mat = material
        self.fs = FourierStiffness(material, mesh, nsamp=nsamp)
        self.rbf = CSRBFMapping(mesh, support_xy, r_s=r_s)
        self.wp = WaveProjection(mesh, d=d, beta=beta, eta=eta,
                                 eta_alpha=eta_alpha)
        self.P = density_filter(mesh, R)
        self.penal = penal
        self.eps = eps
        self.v_all = v_all
        self.vf_all = vf_all
        self.use_alpha = use_alpha

        self.M = self.rbf.M
        self.N = mesh.N
        self.km = self.fs.km

        # BCs / loads / passive (set via helpers)
        self.fixed = np.array([], dtype=int)
        self.F = np.zeros(mesh.ndof)
        self.passive = np.zeros(mesh.N, dtype=bool)   # void elements (z=0)
        self._free_dofs = None

    # ----- boundary conditions / loads ------------------------------- #
    def set_fixed_dofs(self, dofs):
        self.fixed = np.unique(np.asarray(dofs, dtype=int))
        self._free_dofs = np.setdiff1d(np.arange(self.mesh.ndof), self.fixed)

    def set_load(self, F):
        self.F = np.asarray(F, dtype=float)

    def set_passive(self, mask):
        self.passive = np.asarray(mask, dtype=bool)

    def node_id(self, ix, iy):
        return iy * (self.mesh.nelx + 1) + ix

    # ----- forward model --------------------------------------------- #
    def forward(self, z, theta_hat):
        m = self.mesh
        z = np.array(z, dtype=float)
        z[self.passive] = 0.0
        y = self.P @ z                                  # filtered density
        E = self.eps + (1 - self.eps) * y**self.penal   # SIMP, eq Sec 2
        theta = self.rbf.theta(theta_hat)               # eq (3)
        zwave = z if self.use_alpha else np.ones_like(z)
        chi_hat = self.wp.forward(zwave, theta)         # eq (11)-(17)

        kf = self.fs.kf(theta)                          # (N,8,8)
        km = self.km
        k0 = chi_hat[:, None, None] * kf + (1 - chi_hat)[:, None, None] * km
        kel = E[:, None, None] * k0                     # element matrices

        K = self._assemble(kel)
        u = self._solve(K)

        f = float(self.F @ u)
        self._state = dict(z=z, y=y, E=E, theta=theta, theta_hat=theta_hat,
                           chi_hat=chi_hat, kf=kf, k0=k0, u=u, f=f)
        return f

    def _assemble(self, kel):
        m = self.mesh
        K = sp.csr_matrix((kel.ravel(), (m.iK, m.jK)),
                          shape=(m.ndof, m.ndof))
        return K

    def _solve(self, K):
        free = self._free_dofs
        Kff = K[free][:, free].tocsc()
        uf = spla.spsolve(Kff, self.F[free])
        u = np.zeros(self.mesh.ndof)
        u[free] = uf
        return u

    # ----- element displacement energies ----------------------------- #
    def _element_energy(self, M_all):
        """u_e^T M_e u_e for each element, given per-element matrices (N,8,8)."""
        u = self._state['u']
        ue = u[self.mesh.edof]                           # (N,8)
        return np.einsum('ei,eij,ej->e', ue, M_all, ue)

    # ----- objective sensitivities ----------------------------------- #
    def objective_grad(self):
        s = self._state
        E, chi_hat, theta = s['E'], s['chi_hat'], s['theta']
        kf, k0 = s['kf'], s['k0']
        p, eps = self.penal, self.eps

        # df/dE_l = -u_l^T k0_l u_l    (compliance self-adjoint)
        e_k0 = self._element_energy(k0)
        dfdE = -e_k0
        dfdy = dfdE * p * (1 - eps) * s['y']**(p - 1)
        dfdz_filter = self.P.T @ dfdy

        # df/dchi_hat_l = -E_l u_l^T (kf-km) u_l
        e_diff = self._element_energy(kf - self.km[None])
        dfdchihat = -E * e_diff
        if self.use_alpha:
            dfdz_chi, dfdtheta_chi = self.wp.adjoint(dfdchihat)
        else:
            # wave depends on theta only (alpha frozen to 1)
            _, dfdtheta_chi = self.wp.adjoint(dfdchihat)
            dfdz_chi = np.zeros(self.N)

        dfdz = dfdz_filter + dfdz_chi
        dfdz[self.passive] = 0.0

        # direct orientation dependence through kf(theta), eq (43) 2nd term
        dkf = self.fs.dkf_dtheta(theta)                  # (N,8,8)
        e_dkf = self._element_energy(dkf)
        dfdtheta_direct = -E * chi_hat * e_dkf
        dfdtheta = dfdtheta_direct + dfdtheta_chi
        dfdtheta_hat = self.rbf.Phi.T @ dfdtheta          # eq (40)
        return dfdz, dfdtheta_hat

    # ----- volume constraints ---------------------------------------- #
    def volume_constraints(self):
        s = self._state
        A, sumA = self.mesh.A, self.mesh.A.sum()
        y, chi_hat = s['y'], s['chi_hat']
        vg = (A @ y) / sumA - self.v_all                  # eq (1)
        out = {'vg': vg}
        if self.vf_all is not None:
            vfg = (A @ (y * chi_hat)) / sumA - self.vf_all
            out['vfg'] = vfg
        return out

    def volume_grads(self):
        s = self._state
        A, sumA = self.mesh.A, self.mesh.A.sum()
        # vg, eq (47)
        dvg_dy = A / sumA
        dvg_dz = self.P.T @ dvg_dy
        dvg_dz[self.passive] = 0.0
        dvg_dth = np.zeros(self.M)
        grads = {'vg': (dvg_dz, dvg_dth)}
        if self.vf_all is not None:
            y, chi_hat = s['y'], s['chi_hat']
            # eq (48): through y (filter) and chi_hat (wave)
            dvfg_dy = A * chi_hat / sumA
            dvfg_dchihat = A * y / sumA
            dz_c, dth_c = self.wp.adjoint(dvfg_dchihat)
            dvfg_dz = self.P.T @ dvfg_dy + dz_c
            dvfg_dz[self.passive] = 0.0
            dvfg_dth = self.rbf.Phi.T @ dth_c             # eq (49)-(50)
            grads['vfg'] = (dvfg_dz, dvfg_dth)
        return grads

    # ----- curl constraint, eq (18)-(19), (51)-(52) ------------------- #
    def curl(self, theta_hat=None):
        if theta_hat is None:
            theta_hat = self._state['theta_hat']
        theta = self.rbf.theta(theta_hat)
        a = self.rbf.Phi_x @ theta_hat
        b = self.rbf.Phi_y @ theta_hat
        zeta = np.cos(theta) * a + np.sin(theta) * b      # eq (19)
        return zeta

    def curl_jac(self, theta_hat=None):
        """Jacobian d zeta / d theta_hat  (N x M), eq (52)."""
        if theta_hat is None:
            theta_hat = self._state['theta_hat']
        theta = self.rbf.theta(theta_hat)
        a = self.rbf.Phi_x @ theta_hat
        b = self.rbf.Phi_y @ theta_hat
        coef = -np.sin(theta) * a + np.cos(theta) * b
        J = (sp.diags(coef) @ self.rbf.Phi
             + sp.diags(np.cos(theta)) @ self.rbf.Phi_x
             + sp.diags(np.sin(theta)) @ self.rbf.Phi_y)
        return J.tocsr()
