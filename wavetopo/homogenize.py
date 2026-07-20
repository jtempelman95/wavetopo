"""
Numerical homogenization of the periodic fiber-composite cell.

Gives the effective plane-stress stiffness C_H (3x3, Voigt) of the SAME two-phase
cell that bloch.py analyses dynamically, so a design can be scored on STATIC
stiffness and on BAND behaviour at once -- the ingredient a multi-functional
metamaterial needs.

Method (standard cell-problem homogenization):
  for each unit macroscopic strain  eps^i in {(1,0,0),(0,1,0),(0,0,1)}
      solve  K_per chi^i = f^i,     f^i = sum_e ke_e chi0_e^i
      C_H_ij = (1/|Y|) sum_e (chi0^i - chi^i)_e^T ke_e (chi0^j - chi^j)_e

Periodicity reuses BlochProblem.T(0,0) -- the k=0 reduction is exactly the real
periodic constraint -- so the static and dynamic problems cannot disagree about
what "the same cell" means.

Sensitivities are self-adjoint (no extra solves):
    dC_H_ij/dz_e     = (1/|Y|) chi_d^i,T (dke/dz)  chi_d^j,   chi_d = chi0 - chi
    dC_H_ij/dtheta_e = (1/|Y|) chi_d^i,T (dke/dth) chi_d^j
with the same SIMP interpolation bloch.py uses:
    ke = z^p kf(theta) + (1 - z^p) km
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class Homogenizer:
    def __init__(self, bp):
        """bp: an assembled BlochProblem (shares mesh, material, SIMP, CS-RBF)."""
        self.bp = bp
        m = bp.mesh
        self.m = m
        self.T0 = bp.T(0.0, 0.0).real.tocsr()        # k=0 -> real periodicity
        self.area = m.Lx * m.Ly
        self.chi0 = self._macro_fields()

    # ---- nodal displacement fields that realise each unit strain ------ #
    def _macro_fields(self):
        """chi0[i] (ndof_full,) : displacement giving uniform strain eps^i.
        eps1=(1,0,0) -> u=(x,0);  eps2=(0,1,0) -> u=(0,y);  eps3=(0,0,1) -> u=(y,0)
        (engineering shear: gamma_xy = du_x/dy + du_y/dx = 1)."""
        m = self.m
        nx, ny = m.nelx, m.nely
        ix = np.arange(nx + 1) * m.dx
        iy = np.arange(ny + 1) * m.dy
        X, Y = np.meshgrid(ix, iy)                    # (ny+1, nx+1), row-major
        x = X.ravel(); y = Y.ravel()
        chi0 = np.zeros((3, m.ndof))
        chi0[0, 0::2] = x                             # eps_xx = 1
        chi0[1, 1::2] = y                             # eps_yy = 1
        chi0[2, 0::2] = y                             # gamma_xy = 1
        return chi0

    # ---- element stiffness and its derivatives ------------------------ #
    def _kel(self):
        bp = self.bp
        z, theta, p = bp._z, bp._theta, bp.penal
        kf = bp.fs.kf(theta)                          # (N,8,8)
        km = bp.fs.km[None]
        w = z**p
        ke = w[:, None, None]*kf + (1 - w)[:, None, None]*km
        dke_dz = (p*z**(p-1))[:, None, None]*(kf - km)
        dke_dth = w[:, None, None]*bp.fs.dkf_dtheta(theta)
        return ke, dke_dz, dke_dth

    # ---- the three cell problems -------------------------------------- #
    def solve(self):
        """Return (C_H, chi_d) with chi_d[i] the fluctuation-corrected field."""
        m, T0 = self.m, self.T0
        ke, _, _ = self._kel()
        K = sp.csr_matrix((ke.ravel(), (m.iK, m.jK)), shape=(m.ndof, m.ndof))
        Kr = (T0.T @ K @ T0).tocsc()
        # the periodic cell floats: pin one node (rigid translation only)
        keep = np.ones(Kr.shape[0], bool); keep[0] = keep[1] = False
        idx = np.flatnonzero(keep)
        Krr = Kr[idx][:, idx]
        lu = spla.factorized(Krr)

        chi_d = np.zeros((3, m.ndof))
        C = np.zeros((3, 3))
        edof = m.edof
        for i in range(3):
            f_full = K @ self.chi0[i]
            fr = (T0.T @ f_full)[idx]
            xr = np.zeros(Kr.shape[0])
            xr[idx] = lu(fr)
            chi_d[i] = self.chi0[i] - (T0 @ xr)
        for i in range(3):
            di = chi_d[i][edof]                        # (N,8)
            for j in range(i, 3):
                dj = chi_d[j][edof]
                C[i, j] = C[j, i] = np.einsum('ei,eij,ej->', di, ke, dj)/self.area
        self._chi_d = chi_d
        self._C = C
        return C, chi_d

    # ---- sensitivity of a scalar functional of C_H --------------------- #
    def dC_d(self, weights):
        """d(sum_ij weights_ij C_H_ij) / d(z, theta_hat).

        weights (3,3) picks the functional, e.g. np.diag([1,1,0]) for
        C11+C22 (biaxial stiffness) or a single entry for a directional one.
        """
        _, dke_dz, dke_dth = self._kel()
        edof = self.m.edof
        gz = np.zeros(self.m.N)
        gth_e = np.zeros(self.m.N)
        for i in range(3):
            di = self._chi_d[i][edof]
            for j in range(3):
                w = weights[i, j]
                if w == 0.0:
                    continue
                dj = self._chi_d[j][edof]
                gz += w*np.einsum('ei,eij,ej->e', di, dke_dz, dj)/self.area
                gth_e += w*np.einsum('ei,eij,ej->e', di, dke_dth, dj)/self.area
        return gz, self.bp.rbf.Phi.T @ gth_e

    # ---- convenience --------------------------------------------------- #
    @staticmethod
    def bulk(C):
        """Plane-stress bulk-like modulus (resists area change)."""
        return (C[0, 0] + C[1, 1] + 2*C[0, 1])/4.0

    @staticmethod
    def bulk_weights():
        w = np.zeros((3, 3))
        w[0, 0] = w[1, 1] = 0.25
        w[0, 1] = w[1, 0] = 0.25
        return w
