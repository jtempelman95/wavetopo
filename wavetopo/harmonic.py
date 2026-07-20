"""
Steady-state harmonic elastodynamics for curvilinear-fiber wave optics.

A finite, fully-solid composite plate whose *fiber orientation field* is the
design variable.  Driven by a time-harmonic excitation at frequency omega, the
spatially-varying anisotropy bends/focuses the elastic wave -- a manufacturable
graded-index lens made of continuous (curved) fibers.

Dynamic system (complex symmetric):
    D(theta) u = F,   D = (1 + i*eta) K(theta) - omega^2 M + i*omega*C_sponge

* eta            : small structural damping (numerical stability)
* C_sponge       : mass-proportional absorbing layer near the open boundaries
                   so outgoing waves do not reflect (mimics an open domain)
* design         : fiber orientation theta = Phi @ theta_hat  (density fixed = 1)

Objective (maximize):  J = u^H L u  = elastic energy at the focus region.
Complex-symmetric adjoint:  lambda = conj(D)^{-1} (L u),
    dJ/dtheta_l = -2 Re( lambda_e^H (1+i*eta) dkf(theta_l) u_e ).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .cfrp import (Material, QuadMesh, FourierStiffness, CSRBFMapping,
                   element_mass, element_stiffness)


def _isotropic_D(E, nu=0.3):
    return E / (1 - nu**2) * np.array([[1, nu, 0],
                                       [nu, 1, 0],
                                       [0, 0, (1 - nu) / 2]])


class HarmonicLens:
    def __init__(self, mesh: QuadMesh, material: Material, rbf: CSRBFMapping,
                 *, omega, eta=0.02, nsamp=720):
        self.mesh = mesh
        self.mat = material
        self.rbf = rbf
        self.M_ = rbf.M
        self.fs = FourierStiffness(material, mesh, nsamp=nsamp)
        self.omega = omega
        self.eta = eta

        # mass matrix (fully solid fiber composite, density rho_f)
        Me = element_mass(mesh.dx, mesh.dy, rho=material.rho_f)
        self.Me = Me
        self.M = self._assemble_const(Me[None].repeat(mesh.N, 0))

        # placeholders set by helpers
        self.F = np.zeros(mesh.ndof, dtype=complex)
        self.C = sp.csr_matrix((mesh.ndof, mesh.ndof))
        self.Lsel = sp.csr_matrix((mesh.ndof, mesh.ndof))
        self.fixed = np.array([], dtype=int)
        self._free = np.arange(mesh.ndof)

    # ---- assembly helpers ------------------------------------------- #
    def _assemble_const(self, mats):
        m = self.mesh
        return sp.csr_matrix((mats.ravel(), (m.iK, m.jK)),
                             shape=(m.ndof, m.ndof))

    def assemble_K(self, theta):
        kf = self.fs.kf(theta)                       # (N,8,8)
        return self._assemble_const(kf)

    # ---- problem set-up --------------------------------------------- #
    def set_fixed_dofs(self, dofs):
        self.fixed = np.unique(np.asarray(dofs, int))
        self._free = np.setdiff1d(np.arange(self.mesh.ndof), self.fixed)

    def set_load(self, F):
        self.F = np.asarray(F, dtype=complex)

    def set_sponge(self, c_elem):
        """Mass-proportional absorbing layer: C = sum_e c_e * Me."""
        mats = c_elem[:, None, None] * self.Me[None]
        self.C = self._assemble_const(mats)

    def set_focus(self, weight_dof):
        """Diagonal selection L (real) picking out the focus DOFs/weights."""
        self.Lsel = sp.diags(weight_dof).tocsr()

    def node_id(self, ix, iy):
        return iy * (self.mesh.nelx + 1) + ix

    # ---- dynamic solve ---------------------------------------------- #
    def _dynamic_matrix(self, theta):
        K = self.assemble_K(theta)
        D = (1 + 1j * self.eta) * K - self.omega**2 * self.M \
            + 1j * self.omega * self.C
        return D, K

    def solve(self, theta_hat):
        theta = self.rbf.theta(theta_hat)
        D, K = self._dynamic_matrix(theta)
        free = self._free
        u = np.zeros(self.mesh.ndof, dtype=complex)
        Dff = D[free][:, free].tocsc()
        u[free] = spla.spsolve(Dff, self.F[free])
        self._state = dict(theta=theta, theta_hat=theta_hat, u=u, D=D, K=K)
        return u

    # ---- focus energy + gradient ------------------------------------ #
    def focus_energy(self, theta_hat=None):
        if theta_hat is not None:
            self.solve(theta_hat)
        u = self._state['u']
        Lu = self.Lsel @ u
        return float(np.real(np.vdot(u, Lu)))         # u^H L u

    def focus_grad(self):
        s = self._state
        u, D = s['u'], s['D']
        theta = s['theta']
        free = self._free
        # adjoint: conj(D) lambda = L u
        lam = np.zeros(self.mesh.ndof, dtype=complex)
        rhs = self.Lsel @ u
        Dcff = D.conj()[free][:, free].tocsc()
        lam[free] = spla.spsolve(Dcff, rhs[free])
        # element gradient: dJ/dtheta_l = -2 Re( lam_e^H (1+i eta) dkf u_e )
        dkf = self.fs.dkf_dtheta(theta)               # (N,8,8)
        edof = self.mesh.edof
        uE = u[edof]                                   # (N,8) complex
        lamE = lam[edof]
        fac = (1 + 1j * self.eta)
        term = np.einsum('ei,eij,ej->e', np.conj(lamE), dkf, uE) * fac
        dJ_dtheta = -2.0 * term.real
        dJ_dthhat = self.rbf.Phi.T @ dJ_dtheta
        return dJ_dthhat


# ====================================================================== #
#  Two-phase harmonic solver for valley-Hall edge transport              #
# ====================================================================== #
class HarmonicValley:
    """Steady-state harmonic response of a finite, tiled honeycomb domain with
    an explicit density field z (stiff fiber composite vs soft matrix) and an
    explicit per-element fiber orientation field theta.  Used to drive a
    topological valley domain wall and compare against an isotropic control.

        D u = F,  D = (1+i*eta) K - omega^2 M + i*omega*C_sponge
        K_e = z^p [ kf(theta_e) | k_iso ]  +  (1 - z^p) km     (two-phase)
        M_e = (z rho_f + (1-z) rho_m) Me
    """

    def __init__(self, mesh, material, *, omega, eta=0.03, penal=3.0,
                 iso_E=70.0, nsamp=360):
        self.mesh = mesh
        self.mat = material
        self.omega = omega
        self.eta = eta
        self.penal = penal
        self.fs = FourierStiffness(material, mesh, nsamp=nsamp)
        self.Me = element_mass(mesh.dx, mesh.dy, rho=1.0)
        self.k_iso = element_stiffness(_isotropic_D(iso_E), mesh.dx, mesh.dy)
        self.F = np.zeros(mesh.ndof, dtype=complex)
        self.C = sp.csr_matrix((mesh.ndof, mesh.ndof))
        self.fixed = np.array([], dtype=int)
        self._free = np.arange(mesh.ndof)

    def _assemble(self, mats):
        m = self.mesh
        return sp.csr_matrix((mats.ravel(), (m.iK, m.jK)),
                             shape=(m.ndof, m.ndof))

    def set_design(self, z, theta, isotropic=False):
        m = self.mesh
        z = np.asarray(z, float)
        w = z**self.penal
        km = self.fs.km[None]
        if isotropic:
            kstiff = self.k_iso[None]
            kel = w[:, None, None] * kstiff + (1 - w)[:, None, None] * km
        else:
            kf = self.fs.kf(theta)
            kel = w[:, None, None] * kf + (1 - w)[:, None, None] * km
        self.K = self._assemble(kel)
        rho = z * self.mat.rho_f + (1 - z) * self.mat.rho_m
        self.M = self._assemble(rho[:, None, None] * self.Me[None])

    def set_sponge(self, c_elem):
        self.C = self._assemble(c_elem[:, None, None] * self.Me[None])

    def set_load(self, F):
        self.F = np.asarray(F, dtype=complex)

    def node_id(self, ix, iy):
        return iy * (self.mesh.nelx + 1) + ix

    def solve(self):
        D = (1 + 1j * self.eta) * self.K - self.omega**2 * self.M \
            + 1j * self.omega * self.C
        free = self._free
        u = np.zeros(self.mesh.ndof, dtype=complex)
        u[free] = spla.spsolve(D[free][:, free].tocsc(), self.F[free])
        self.u = u
        return u

    def energy_in_box(self, x0, x1, y0, y1):
        """Sum |u|^2 over nodes in a box (a transmission probe)."""
        xs, ys = self.mesh.node_xy()
        sel = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        ux = self.u[0::2][sel]; uy = self.u[1::2][sel]
        return float(np.sum(np.abs(ux)**2 + np.abs(uy)**2))


# ====================================================================== #
#  Elastic cloak: minimize a void's scattering via fiber orientation     #
# ====================================================================== #
class HarmonicCloak:
    """In-plane harmonic response of a fiber-composite plate containing a fixed
    void, with the fiber orientation field as the design.  Objective matches the
    field to a void-free reference in an observation region (cloaking):

        J = (u - u_ref)^H W (u - u_ref),   minimized over theta.

    Two-phase stiffness K_e = z^p kf(theta_e) + (1 - z^p) km (z=0 in the void).
    Complex-symmetric adjoint, same structure as HarmonicLens.
    """

    def __init__(self, mesh, material, rbf, *, omega, eta=0.02, penal=3.0,
                 nsamp=720):
        self.mesh = mesh
        self.mat = material
        self.rbf = rbf
        self.M_ = rbf.M
        self.fs = FourierStiffness(material, mesh, nsamp=nsamp)
        self.omega = omega
        self.eta = eta
        self.penal = penal
        self.Me = element_mass(mesh.dx, mesh.dy, rho=1.0)
        self.z = np.ones(mesh.N)
        self.F = np.zeros(mesh.ndof, dtype=complex)
        self.C = sp.csr_matrix((mesh.ndof, mesh.ndof))
        self.W = None
        self.u_ref = None
        self._free = np.arange(mesh.ndof)

    def _asm(self, mats):
        m = self.mesh
        return sp.csr_matrix((mats.ravel(), (m.iK, m.jK)),
                             shape=(m.ndof, m.ndof))

    def set_density(self, z):
        self.z = np.asarray(z, float)
        w = self.z**self.penal
        self._w = w
        rho = self.z * self.mat.rho_f + (1 - self.z) * self.mat.rho_m
        self.M = self._asm(rho[:, None, None] * self.Me[None])

    def set_load(self, F):
        self.F = np.asarray(F, dtype=complex)

    def set_sponge(self, c):
        self.C = self._asm(c[:, None, None] * self.Me[None])

    def set_target(self, u_ref, weight_dof):
        self.u_ref = np.asarray(u_ref, complex)
        self.W = sp.diags(weight_dof).tocsr()

    def node_id(self, ix, iy):
        return iy * (self.mesh.nelx + 1) + ix

    def _dynamic(self, theta):
        w = self._w
        km = self.fs.km[None]
        kel = w[:, None, None] * self.fs.kf(theta) + (1 - w)[:, None, None] * km
        K = self._asm(kel)
        D = (1 + 1j * self.eta) * K - self.omega**2 * self.M \
            + 1j * self.omega * self.C
        return D

    def solve(self, theta_hat):
        theta = self.rbf.theta(theta_hat)
        D = self._dynamic(theta)
        u = spla.spsolve(D.tocsc(), self.F)
        self._state = dict(theta=theta, u=u, D=D)
        return u

    def objective(self, theta_hat=None):
        if theta_hat is not None:
            self.solve(theta_hat)
        r = self._state['u'] - self.u_ref
        return float(np.real(np.vdot(r, self.W @ r)))

    def grad(self):
        s = self._state
        u, D, theta = s['u'], s['D'], s['theta']
        r = u - self.u_ref
        lam = spla.spsolve(D.conj().tocsc(), self.W @ r)
        dkf = self.fs.dkf_dtheta(theta)
        edof = self.mesh.edof
        uE = u[edof]; lamE = lam[edof]
        fac = (1 + 1j * self.eta) * self._w
        term = np.einsum('ei,eij,ej->e', np.conj(lamE), dkf, uE) * fac
        dJ_dtheta = -2.0 * term.real
        return self.rbf.Phi.T @ dJ_dtheta
