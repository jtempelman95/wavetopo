"""
Toolpath-guided wave-energy localization (single- and multi-frequency).

A single-material continuous-fiber composite plate whose fiber ORIENTATION
field theta(x) = Phi(x) theta_hat is the design variable.  Time-harmonic
excitation at one or more frequencies is steered by the spatially-varying
anisotropy so that elastic energy concentrates in prescribed focus regions.

Physics (in-plane elastodynamics, complex-symmetric):
    D_i(theta) u_i = F,
    D_i = (1 + i eta) K(theta) - omega_i^2 M + i omega_i C_sponge
with K(theta) assembled from the exact Fourier expansion of the rotated fiber
stiffness (see cfrp.FourierStiffness) and C_sponge a mass-proportional
absorbing layer emulating an open domain.

Focus energy at frequency i:  E_i(theta) = u_i^H L_i u_i   (L_i diagonal, real).
Objectives (all MINIMIZED here; we negate gains we want to maximize):
    mode='sum'    : J = -sum_i  E_i / base_i
    mode='minmax' : J = -softmin_i ( E_i / base_i )   (broadband / worst-case)
where base_i normalizes each frequency (e.g. the straight-fiber baseline).

Adjoint (per frequency, complex-symmetric):
    lambda_i = conj(D_i)^{-1} (L_i u_i),
    dE_i/dtheta_e = -2 Re( lambda_{i,e}^H (1+i eta) dkf(theta_e) u_{i,e} ).

The multi-frequency gradient is the (softmin- or sum-)weighted combination.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .cfrp import Material, QuadMesh, FourierStiffness, CSRBFMapping, element_mass


def _softmin(v, rho):
    """KS soft-min and its per-element weights: smin ~ min(v), sum(w)=1."""
    vmin = np.min(v)
    e = np.exp(-rho * (v - vmin))
    s = e.sum()
    smin = vmin - np.log(s) / rho
    return smin, e / s


class WaveFocus:
    """Multi-frequency energy-localization optimizer over fiber orientation."""

    def __init__(self, mesh: QuadMesh, material: Material, rbf: CSRBFMapping,
                 omegas, *, eta=0.03, nsamp=720):
        self.mesh = mesh
        self.mat = material
        self.rbf = rbf
        self.M_ = rbf.M
        self.omegas = np.atleast_1d(np.asarray(omegas, float))
        self.nf = self.omegas.size
        self.eta = eta
        self.fs = FourierStiffness(material, mesh, nsamp=nsamp)
        Me = element_mass(mesh.dx, mesh.dy, rho=material.rho_f)
        self.M = self._asm(Me[None].repeat(mesh.N, 0))
        self.F = np.zeros(mesh.ndof, dtype=complex)
        self.C = sp.csr_matrix((mesh.ndof, mesh.ndof))
        self.L = [sp.csr_matrix((mesh.ndof, mesh.ndof)) for _ in range(self.nf)]
        self.base = np.ones(self.nf)

    # ---- assembly helpers ------------------------------------------- #
    def _asm(self, mats):
        m = self.mesh
        return sp.csr_matrix((mats.ravel(), (m.iK, m.jK)),
                             shape=(m.ndof, m.ndof))

    def node_id(self, ix, iy):
        return iy * (self.mesh.nelx + 1) + ix

    def set_load(self, F):
        self.F = np.asarray(F, dtype=complex)

    def set_sponge(self, c_elem):
        self.C = self._asm(c_elem[:, None, None]
                           * element_mass(self.mesh.dx, self.mesh.dy, 1.0)[None])

    def set_targets(self, weight_list):
        """One diagonal focus weight vector (length ndof) per frequency."""
        assert len(weight_list) == self.nf
        self.L = [sp.diags(np.asarray(w, float)).tocsr() for w in weight_list]

    def focus_weight(self, center, half):
        """Build a nodal focus weight: 1 on both dofs inside a box, else 0."""
        cx, cy = center
        xs, ys = self.mesh.node_xy()
        sel = (np.abs(xs - cx) < half) & (np.abs(ys - cy) < half)
        w = np.zeros(self.mesh.ndof)
        w[0::2][sel] = 1.0
        w[1::2][sel] = 1.0
        return w

    # ---- forward solves --------------------------------------------- #
    def _K(self, theta):
        return self._asm(self.fs.kf(theta))

    def solve(self, theta_hat):
        theta = self.rbf.theta(theta_hat)
        K = self._K(theta)
        us, Ds = [], []
        for i, w in enumerate(self.omegas):
            D = (1 + 1j * self.eta) * K - w**2 * self.M + 1j * w * self.C
            Ds.append(D.tocsc())
            us.append(spla.spsolve(Ds[-1], self.F))
        self._state = dict(theta=theta, us=us, Ds=Ds)
        return us

    def energies(self, theta_hat=None):
        if theta_hat is not None:
            self.solve(theta_hat)
        us = self._state['us']
        return np.array([float(np.real(np.vdot(u, self.L[i] @ u)))
                         for i, u in enumerate(us)])

    def set_baseline(self, base):
        self.base = np.asarray(base, float)

    # ---- objective + gradient --------------------------------------- #
    def objective(self, theta_hat=None, mode='minmax', rho_ks=20.0):
        E = self.energies(theta_hat)
        g = E / self.base
        self._obj_cache = dict(E=E, g=g, mode=mode, rho_ks=rho_ks)
        if mode == 'sum':
            return -float(np.sum(g))
        smin, w = _softmin(g, rho_ks)
        self._obj_cache['w'] = w
        return -float(smin)

    def grad(self):
        c = self._obj_cache
        s = self._state
        us, Ds = s['us'], s['Ds']
        theta = s['theta']
        dkf = self.fs.dkf_dtheta(theta)              # (N,8,8)
        edof = self.mesh.edof
        fac = (1 + 1j * self.eta)
        if c['mode'] == 'sum':
            wts = np.ones(self.nf)
        else:
            wts = c['w']
        dJ_dtheta = np.zeros(self.mesh.N)
        for i in range(self.nf):
            u = us[i]
            lam = spla.spsolve(Ds[i].conj().tocsc(), self.L[i] @ u)
            uE = u[edof]; lamE = lam[edof]
            term = np.einsum('ei,eij,ej->e', np.conj(lamE), dkf, uE) * fac
            dE_dtheta = -2.0 * term.real
            # d(-g_i)/dtheta = -(1/base_i) dE_i ; weighted by wts[i]
            dJ_dtheta += -(wts[i] / self.base[i]) * dE_dtheta
        return self.rbf.Phi.T @ dJ_dtheta


def ramp_sponge(mesh: QuadMesh, margin, strength, sides=("right", "top", "bottom")):
    """Quadratic mass-proportional absorbing ramp on chosen domain sides."""
    Lx, Ly = mesh.Lx, mesh.Ly
    cx, cy = mesh.cx, mesh.cy
    d = np.zeros(mesh.N)
    if "right" in sides:
        d = np.maximum(d, (cx - (Lx - margin)) / margin)
    if "left" in sides:
        d = np.maximum(d, (margin - cx) / margin)
    if "top" in sides:
        d = np.maximum(d, (cy - (Ly - margin)) / margin)
    if "bottom" in sides:
        d = np.maximum(d, (margin - cy) / margin)
    return np.maximum(0.0, d)**2 * strength
