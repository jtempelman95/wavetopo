"""
Out-of-plane (antiplane-shear / SH) scalar elastodynamics on the honeycomb
fiber-composite lattice -- the clean setting for valley-Hall edge transport.

Single scalar field u_z(x,y) per node:
    div( mu(theta) grad u_z ) + rho omega^2 u_z = 0
The fiber's out-of-plane shear is anisotropic,
    mu(theta) = R(theta) diag(mu_L, mu_T) R(theta)^T          (2x2 tensor)
and the tensor rotation by theta is the tunable inversion-breaking Dirac mass.

mu(theta) is a 3-term exact Fourier series in {1, cos2theta, sin2theta}, so
element matrices are precomputed once (as in cfrp.py, but scalar):
    kf(theta) = sum_s kf_s c_s(theta),  km = isotropic matrix shear.

ScalarBloch mirrors bloch.BlochProblem (T(k) with 1 dof/node) so it works with
the existing FHS berry_curvature and the KS/MMA optimization machinery.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

from .cfrp import QuadMesh, CSRBFMapping


class MaterialSH:
    def __init__(self, mu_L=5.0, mu_T=1.5, mu_m=1.0, rho_f=1.6, rho_m=1.2):
        self.mu_L, self.mu_T, self.mu_m = mu_L, mu_T, mu_m
        self.rho_f, self.rho_m = rho_f, rho_m

    def mu(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return R @ np.diag([self.mu_L, self.mu_T]) @ R.T


def elem_K_scalar(mu, dx, dy):
    """4x4 element matrix  int B^T mu B  for a bilinear quad (scalar field)."""
    g = 1 / np.sqrt(3.0)
    gp = [(-g, -g), (g, -g), (g, g), (-g, g)]
    Ke = np.zeros((4, 4))
    J = np.array([[dx / 2, 0], [0, dy / 2]])
    detJ = np.linalg.det(J)
    Jinv = np.linalg.inv(J)
    for xi, eta in gp:
        dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dN_de = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        B = np.vstack([Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_de,
                       Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_de])  # (2,4)
        Ke += (B.T @ mu @ B) * detJ
    return Ke


def elem_M_scalar(dx, dy, rho=1.0):
    g = 1 / np.sqrt(3.0)
    gp = [(-g, -g), (g, -g), (g, g), (-g, g)]
    Me = np.zeros((4, 4))
    detJ = (dx / 2) * (dy / 2)
    for xi, eta in gp:
        N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        Me += rho * np.outer(N, N) * detJ
    return Me


class FourierMu:
    """Precompute scalar element matrices: kf(theta)=sum kf_s c_s(theta)."""

    def __init__(self, material: MaterialSH, mesh: QuadMesh):
        muL, muT = material.mu_L, material.mu_T
        a, b = (muL + muT) / 2, (muL - muT) / 2
        C = [np.array([[a, 0], [0, a]]),
             np.array([[b, 0], [0, -b]]),
             np.array([[0, b], [b, 0]])]
        self.kf_s = np.array([elem_K_scalar(Cs, mesh.dx, mesh.dy) for Cs in C])
        self.km = elem_K_scalar(material.mu_m * np.eye(2), mesh.dx, mesh.dy)

    @staticmethod
    def _basis(theta):
        theta = np.atleast_1d(theta)
        return np.stack([np.ones_like(theta), np.cos(2 * theta),
                         np.sin(2 * theta)], -1)

    @staticmethod
    def _dbasis(theta):
        theta = np.atleast_1d(theta)
        z = np.zeros_like(theta)
        return np.stack([z, -2 * np.sin(2 * theta), 2 * np.cos(2 * theta)], -1)

    def kf(self, theta):
        return np.einsum('...s,sij->...ij', self._basis(theta), self.kf_s)

    def dkf(self, theta):
        return np.einsum('...s,sij->...ij', self._dbasis(theta), self.kf_s)


class ScalarBloch:
    """Bloch eigenproblem for the scalar (antiplane) honeycomb crystal."""

    def __init__(self, mesh: QuadMesh, material: MaterialSH, rbf: CSRBFMapping,
                 *, penal=3.0, iso_mu=None):
        self.mesh = mesh
        self.mat = material
        self.rbf = rbf
        self.M = rbf.M
        self.fs = FourierMu(material, mesh)
        self.penal = penal
        self.Me = elem_M_scalar(mesh.dx, mesh.dy, 1.0)
        self.k_iso = (elem_K_scalar(iso_mu * np.eye(2), mesh.dx, mesh.dy)
                      if iso_mu else None)
        self._scalar_conn()
        self._build_bloch_index()

    # scalar connectivity: 4 node ids per element (1 dof/node)
    def _scalar_conn(self):
        m = self.mesh
        edof = np.zeros((m.N, 4), dtype=np.int64)
        for e in range(m.N):
            ex, ey = e % m.nelx, e // m.nelx
            n1 = ey * (m.nelx + 1) + ex
            edof[e] = [n1, n1 + 1, n1 + 1 + (m.nelx + 1), n1 + (m.nelx + 1)]
        self.sedof = edof
        self.iK = np.kron(edof, np.ones((4, 1), np.int64)).ravel()
        self.jK = np.kron(edof, np.ones((1, 4), np.int64)).ravel()
        self.nnode = m.nnode

    def _build_bloch_index(self):
        m = self.mesh
        nelx, nely = m.nelx, m.nely
        self.n_red = nelx * nely
        master = np.zeros(m.nnode, np.int64)
        sx = np.zeros(m.nnode); sy = np.zeros(m.nnode)
        for iy in range(nely + 1):
            for ix in range(nelx + 1):
                n = iy * (nelx + 1) + ix
                master[n] = (iy % nely) * nelx + (ix % nelx)
                sx[n] = 1.0 if ix == nelx else 0.0
                sy[n] = 1.0 if iy == nely else 0.0
        self._master, self._sx, self._sy = master, sx, sy

    def T(self, kx, ky):
        m = self.mesh
        ph = np.exp(1j * (kx * m.Lx * self._sx + ky * m.Ly * self._sy))
        rows = np.arange(m.nnode)
        return sp.csr_matrix((ph, (rows, self._master)),
                             shape=(m.nnode, self.n_red))

    def assemble(self, z, theta_hat):
        m = self.mesh
        z = np.asarray(z, float)
        theta = self.rbf.theta(theta_hat)
        w = z**self.penal
        kf = self.fs.kf(theta)
        km = self.fs.km[None]
        kel = w[:, None, None] * kf + (1 - w)[:, None, None] * km
        K = sp.csr_matrix((kel.ravel(), (self.iK, self.jK)),
                          shape=(self.nnode, self.nnode))
        rho = z * self.mat.rho_f + (1 - z) * self.mat.rho_m
        mel = rho[:, None, None] * self.Me[None]
        Mm = sp.csr_matrix((mel.ravel(), (self.iK, self.jK)),
                           shape=(self.nnode, self.nnode))
        self._K, self._M = K, Mm
        self._theta, self._z = theta, z
        return K, Mm

    def bands_at_k(self, kx, ky, n_bands=8, return_vec=False, sigma=None):
        T = self.T(kx, ky)
        Kr = (T.conj().T @ self._K @ T)
        Mr = (T.conj().T @ self._M @ T)
        if sigma is None:
            w2, V = sla.eigh(Kr.toarray(), Mr.toarray(),
                             subset_by_index=[0, n_bands - 1])
        else:
            w2, V = spla.eigsh(Kr.tocsc(), k=n_bands, M=Mr.tocsc(),
                               sigma=sigma, which='LM')
            o = np.argsort(w2.real); w2, V = w2[o].real, V[:, o]
            for c in range(V.shape[1]):
                V[:, c] /= np.sqrt(np.real(np.vdot(V[:, c], Mr @ V[:, c])))
        w2 = np.clip(w2, 0, None); w = np.sqrt(w2)
        if return_vec:
            return w, w2, V, T
        return w

    def band_structure(self, kpath, n_bands=8):
        return np.array([self.bands_at_k(kx, ky, n_bands) for kx, ky in kpath])

    def eigen_sensitivity(self, kx, ky, n_bands=8, sigma=None):
        w, w2, V, T = self.bands_at_k(kx, ky, n_bands, True, sigma)
        theta, z, p = self._theta, self._z, self.penal
        kf = self.fs.kf(theta); dkf = self.fs.dkf(theta); km = self.fs.km
        drho = self.mat.rho_f - self.mat.rho_m
        edof = self.sedof
        dz = np.zeros((n_bands, self.mesh.N))
        dth = np.zeros((n_bands, self.M))
        for b in range(n_bands):
            phi = (T @ V[:, b])[edof]            # (N,4) complex
            cE = np.conj(phi)
            e_kfmkm = np.einsum('ei,eij,ej->e', cE, kf - km[None], phi).real
            e_M = np.einsum('ei,ij,ej->e', cE, self.Me, phi).real
            e_dkf = np.einsum('ei,eij,ej->e', cE, dkf, phi).real
            lam = w2[b]
            dlam_dz = p * z**(p - 1) * e_kfmkm - lam * drho * e_M
            sc = 0.5 / max(w[b], 1e-12)
            dz[b] = sc * dlam_dz
            dth[b] = sc * (self.rbf.Phi.T @ (z**p * e_dkf))
        return w, dz, dth


class ScalarHarmonic:
    """Forced harmonic response of the scalar (antiplane) domain:
    (1+i eta) K - omega^2 M + i omega C_sponge) u = F, 1 dof/node, with an
    explicit density z and per-element fiber field theta (isotropic-control flag).
    """

    def __init__(self, mesh: QuadMesh, material: MaterialSH, *, omega,
                 eta=0.03, penal=3.0, iso_mu=3.0):
        self.mesh = mesh
        self.mat = material
        self.omega = omega
        self.eta = eta
        self.penal = penal
        self.fs = FourierMu(material, mesh)
        self.Me = elem_M_scalar(mesh.dx, mesh.dy, 1.0)
        self.k_iso = elem_K_scalar(iso_mu * np.eye(2), mesh.dx, mesh.dy)
        # scalar connectivity
        m = mesh
        edof = np.zeros((m.N, 4), np.int64)
        for e in range(m.N):
            ex, ey = e % m.nelx, e // m.nelx
            n1 = ey * (m.nelx + 1) + ex
            edof[e] = [n1, n1 + 1, n1 + 1 + (m.nelx + 1), n1 + (m.nelx + 1)]
        self.sedof = edof
        self.iK = np.kron(edof, np.ones((4, 1), np.int64)).ravel()
        self.jK = np.kron(edof, np.ones((1, 4), np.int64)).ravel()
        self.nnode = m.nnode
        self.F = np.zeros(self.nnode, dtype=complex)
        self.C = sp.csr_matrix((self.nnode, self.nnode))

    def _asm(self, mats):
        return sp.csr_matrix((mats.ravel(), (self.iK, self.jK)),
                             shape=(self.nnode, self.nnode))

    def set_design(self, z, theta, isotropic=False):
        z = np.asarray(z, float)
        w = z**self.penal
        km = self.fs.km[None]
        if isotropic:
            kel = w[:, None, None] * self.k_iso[None] + (1 - w)[:, None, None] * km
        else:
            kel = w[:, None, None] * self.fs.kf(theta) + (1 - w)[:, None, None] * km
        self.K = self._asm(kel)
        rho = z * self.mat.rho_f + (1 - z) * self.mat.rho_m
        self.M = self._asm(rho[:, None, None] * self.Me[None])

    def set_sponge(self, c_elem):
        self.C = self._asm(c_elem[:, None, None] * self.Me[None])

    def set_load(self, F):
        self.F = np.asarray(F, dtype=complex)

    def node_id(self, ix, iy):
        return iy * (self.mesh.nelx + 1) + ix

    def solve(self):
        D = (1 + 1j * self.eta) * self.K - self.omega**2 * self.M \
            + 1j * self.omega * self.C
        self.u = spla.spsolve(D.tocsc(), self.F)
        return self.u

    def energy_in_box(self, x0, x1, y0, y1):
        xs, ys = self.mesh.node_xy()
        sel = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        return float(np.sum(np.abs(self.u[sel])**2))
