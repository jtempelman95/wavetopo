"""
Bloch band-structure solver for fiber-reinforced phononic / elastic
metamaterials, built on the CFRP assembly primitives.

A periodic unit cell carries two design fields (exactly as in the CFRP
problem):

    z         density -> where the stiff fiber-composite phase sits
    theta_hat fiber orientation (via CS-RBF) -> makes the cell ANISOTROPIC,
              which is what makes a band gap DIRECTION-DEPENDENT.

For a Bloch wavevector k the displacement obeys u(x+a) = e^{i k.a} u(x).
We assemble K, M on the open cell, then impose periodicity with a complex
reduction matrix T(k) and solve the Hermitian generalized eigenproblem

    K_r(k) phi = omega^2 M_r(k) phi,   K_r = T^H K T,  M_r = T^H M T.

Two-phase interpolation (stiff anisotropic fiber composite vs soft matrix):
    k_e = (eps + (1-eps) z^p) * kf(theta)         (stiff phase fades to ~void)
    m_e = (rho_min + z (1-rho_min)) * rho_f * Me   (linear mass, avoids
                                                    spurious void modes)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

from .cfrp import (Material, QuadMesh, FourierStiffness, CSRBFMapping,
                   element_mass)


class BlochProblem:
    def __init__(self, mesh: QuadMesh, material: Material, rbf: CSRBFMapping,
                 *, penal=3.0, eps=1e-3, rho_min=1e-3, nsamp=720):
        self.mesh = mesh
        self.mat = material
        self.rbf = rbf
        self.M = rbf.M
        self.fs = FourierStiffness(material, mesh, nsamp=nsamp)
        self.penal = penal
        self.eps = eps
        self.rho_min = rho_min
        self.Me = element_mass(mesh.dx, mesh.dy, rho=1.0)
        self._build_bloch_index()

    # ---- periodic node bookkeeping ---------------------------------- #
    def _build_bloch_index(self):
        m = self.mesh
        nelx, nely = m.nelx, m.nely
        self.n_red_nodes = nelx * nely
        self.ndof_red = 2 * self.n_red_nodes
        # for each FULL node -> (master independent node, shift_x, shift_y)
        master = np.zeros(m.nnode, dtype=np.int64)
        sx = np.zeros(m.nnode); sy = np.zeros(m.nnode)
        for iy in range(nely + 1):
            for ix in range(nelx + 1):
                n = iy * (nelx + 1) + ix
                mx, my = ix % nelx, iy % nely
                master[n] = my * nelx + mx
                sx[n] = 1.0 if ix == nelx else 0.0
                sy[n] = 1.0 if iy == nely else 0.0
        self._master = master
        self._sx, self._sy = sx, sy

    def T(self, kx, ky):
        """Complex reduction matrix (ndof_full x ndof_red) for wavevector k."""
        m = self.mesh
        phase = np.exp(1j * (kx * m.Lx * self._sx + ky * m.Ly * self._sy))
        rows = np.empty(m.ndof, dtype=np.int64)
        cols = np.empty(m.ndof, dtype=np.int64)
        vals = np.empty(m.ndof, dtype=complex)
        for n in range(m.nnode):
            mn = self._master[n]
            ph = phase[n]
            rows[2*n] = 2*n; cols[2*n] = 2*mn; vals[2*n] = ph
            rows[2*n+1] = 2*n+1; cols[2*n+1] = 2*mn+1; vals[2*n+1] = ph
        return sp.csr_matrix((vals, (rows, cols)),
                             shape=(m.ndof, self.ndof_red))

    # ---- assemble full-cell K, M for a design ----------------------- #
    def assemble(self, z, theta_hat):
        """Two-phase cell: z=1 -> stiff anisotropic fiber composite,
        z=0 -> soft isotropic polymer matrix (SIMP-interpolated)."""
        m = self.mesh
        z = np.asarray(z, float)
        theta = self.rbf.theta(theta_hat)
        w = z**self.penal                                       # SIMP weight
        kf = self.fs.kf(theta)                                  # (N,8,8)
        km = self.fs.km[None]                                   # (1,8,8)
        kel = w[:, None, None] * kf + (1 - w)[:, None, None] * km
        K = sp.csr_matrix((kel.ravel(), (m.iK, m.jK)),
                          shape=(m.ndof, m.ndof))
        # linear mass interpolation between fiber and matrix density
        rho = z * self.mat.rho_f + (1 - z) * self.mat.rho_m
        mel = rho[:, None, None] * self.Me[None]
        M = sp.csr_matrix((mel.ravel(), (m.iK, m.jK)),
                          shape=(m.ndof, m.ndof))
        self._K, self._M = K, M
        self._theta = theta
        self._z = z
        return K, M

    # ---- reduced matrices at a single k ----------------------------- #
    def _reduced(self, kx, ky):
        T = self.T(kx, ky)
        Kr = (T.conj().T @ self._K @ T)
        Mr = (T.conj().T @ self._M @ T)
        return Kr, Mr, T

    # ---- bands at a single k ---------------------------------------- #
    def bands_at_k(self, kx, ky, n_bands=10, return_vec=False, sigma=None):
        """Lowest n_bands (sigma=None, dense) or n_bands nearest a target
        frequency**2 sigma (sparse shift-invert -- ~20x faster for a few
        interior bands, used for the valley/Dirac work)."""
        Kr, Mr, T = self._reduced(kx, ky)
        if sigma is None:
            w2, V = sla.eigh(Kr.toarray(), Mr.toarray(),
                             subset_by_index=[0, n_bands - 1])
        else:
            w2, V = spla.eigsh(Kr.tocsc(), k=n_bands, M=Mr.tocsc(),
                               sigma=sigma, which='LM')
            order = np.argsort(w2.real)
            w2, V = w2[order].real, V[:, order]
            # M_r-normalize (eigsh normalizes to V^H V = I, not V^H Mr V = I)
            for c in range(V.shape[1]):
                nrm = np.sqrt(np.real(np.vdot(V[:, c], Mr @ V[:, c])))
                V[:, c] /= nrm
        w2 = np.clip(w2, 0.0, None)
        w = np.sqrt(w2)
        if return_vec:
            return w, w2, V, T          # V is M_r-orthonormal: V^H Mr V = I
        return w

    # ---- eigenvalue sensitivities at a single k --------------------- #
    def eigen_sensitivity(self, kx, ky, n_bands=10, sigma=None):
        """Return omega (n_bands,) and gradients d omega / d z (n_bands,N),
        d omega / d theta_hat (n_bands,M) for the lowest n_bands (or n_bands
        nearest sigma if given).

        Simple-eigenvalue perturbation for K phi = lambda M phi, lambda=omega^2:
            d lambda = phi^H (dK - lambda dM) phi   (M-orthonormal phi)
        dK, dM are element-local, so this reduces to element energies.
        """
        w, w2, V, T = self.bands_at_k(kx, ky, n_bands, return_vec=True,
                                      sigma=sigma)
        m = self.mesh
        theta = self._theta
        z = self._z
        p = self.penal
        kf = self.fs.kf(theta)                       # (N,8,8)
        dkf = self.fs.dkf_dtheta(theta)              # (N,8,8)
        km = self.fs.km                              # (8,8)
        drho = self.mat.rho_f - self.mat.rho_m
        edof = m.edof

        dz = np.zeros((n_bands, m.N))
        dth = np.zeros((n_bands, self.M))
        for b in range(n_bands):
            phi_full = T @ V[:, b]                    # full complex eigvec
            phiE = phi_full[edof]                     # (N,8) complex
            cE = np.conj(phiE)
            e_kfmkm = np.einsum('ei,eij,ej->e', cE, kf - km[None], phiE).real
            e_M = np.einsum('ei,ij,ej->e', cE, self.Me, phiE).real
            e_dkf = np.einsum('ei,eij,ej->e', cE, dkf, phiE).real
            lam = w2[b]
            # d lambda / d z
            dlam_dz = p * z**(p - 1) * e_kfmkm - lam * drho * e_M
            # d lambda / d theta -> theta_hat via CS-RBF
            dlam_dth_dp = z**p * e_dkf
            dlam_dth = self.rbf.Phi.T @ dlam_dth_dp
            # convert to d omega (omega = sqrt(lambda))
            scale = 0.5 / max(w[b], 1e-12)
            dz[b] = scale * dlam_dz
            dth[b] = scale * dlam_dth
        return w, dz, dth

    # ---- band structure along a k-path ------------------------------ #
    def band_structure(self, kpath, n_bands=10):
        out = np.zeros((len(kpath), n_bands))
        for i, (kx, ky) in enumerate(kpath):
            out[i] = self.bands_at_k(kx, ky, n_bands)
        return out


# ---------------------------------------------------------------------- #
#  Berry curvature via Fukui-Hatsugai-Suzuki (gauge-invariant)           #
# ---------------------------------------------------------------------- #
def berry_curvature(bp: BlochProblem, band, nk=24, n_bands=None):
    """Berry curvature F(k) of an isolated band over the full BZ.

    Uses M-weighted full-space Bloch states psi(k) = T(k) V(k) (the full mass
    matrix M is k-independent, so the inner product is well defined) and the
    Fukui-Hatsugai-Suzuki plaquette construction, which is gauge invariant:

        U_mu(k) = <psi(k)|M|psi(k+e_mu)> / |.|
        F(k)    = Im ln( U_x(k) U_y(k+x) U_x(k+y)^* U_y(k)^* )

    The reduction matrix T(k) is exactly BZ-periodic (shift vectors are lattice
    vectors), so the discrete loop closes without an extra boundary unitary.

    Returns (kxs, kys, F[nk,nk], chern). For a time-reversal-symmetric medium
    chern == 0 and F(-k) == -F(k); F is identically ~0 if inversion symmetry
    also holds, and becomes non-zero once inversion is broken (e.g. by a chiral
    fiber-orientation field).
    """
    Lx, Ly = bp.mesh.Lx, bp.mesh.Ly
    nbnd = (band + 2) if n_bands is None else n_bands
    kxs = 2 * np.pi / Lx * np.arange(nk) / nk
    kys = 2 * np.pi / Ly * np.arange(nk) / nk
    ndof = bp._M.shape[0]          # works for vector (2/node) and scalar (1/node)
    psi = np.empty((nk, nk, ndof), dtype=complex)
    for i, kx in enumerate(kxs):
        for j, ky in enumerate(kys):
            _, _, V, T = bp.bands_at_k(kx, ky, nbnd, return_vec=True)
            psi[i, j] = T @ V[:, band]
    M = bp._M

    def ov(a, b):
        return np.vdot(a, M @ b)            # a^H M b

    F = np.zeros((nk, nk))
    for i in range(nk):
        for j in range(nk):
            a = psi[i, j]
            b = psi[(i + 1) % nk, j]
            c = psi[(i + 1) % nk, (j + 1) % nk]
            d = psi[i, (j + 1) % nk]
            U = ov(a, b) * ov(b, c) * ov(c, d) * ov(d, a)
            F[i, j] = np.angle(U)
    chern = F.sum() / (2 * np.pi)
    return kxs, kys, F, chern


# ---------------------------------------------------------------------- #
#  Brillouin-zone path helpers (square lattice)                          #
# ---------------------------------------------------------------------- #
def k_segment(k0, k1, n):
    return [tuple(k0 + (k1 - k0) * t) for t in np.linspace(0, 1, n)]


def ibz_path_square(a, b, n=30):
    """Gamma-X-M-Gamma for a rectangular cell of size a x b."""
    G = np.array([0.0, 0.0])
    X = np.array([np.pi / a, 0.0])
    M = np.array([np.pi / a, np.pi / b])
    path = (k_segment(G, X, n) + k_segment(X, M, n)[1:]
            + k_segment(M, G, n)[1:])
    ticks = [0, n - 1, 2 * n - 2, 3 * n - 3]
    labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
    return path, ticks, labels


def directional_path(a, b, theta_dir, kmax_frac=1.0, n=40):
    """Straight k-path from Gamma along direction theta_dir (radians).

    Used to probe how the dispersion (and any gap) depends on propagation
    direction -- the signature of a *directional* band gap.
    """
    kmax = kmax_frac * np.pi / a
    kx = kmax * np.cos(theta_dir)
    ky = kmax * np.sin(theta_dir)
    return k_segment(np.array([0.0, 0.0]), np.array([kx, ky]), n)


def gap_between_bands(bands, lower):
    """Gap between band `lower` (0-based) and band `lower+1` across the path."""
    top_of_lower = bands[:, lower].max()
    bot_of_upper = bands[:, lower + 1].min()
    return bot_of_upper - top_of_lower, top_of_lower, bot_of_upper
