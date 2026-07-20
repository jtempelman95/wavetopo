"""
Manufacturability-constrained toolpath wave control.

Unifies the finite-domain harmonic wave devices (lens / broadband lens /
cloak) into one two-phase problem class and adds the *practical
manufacturing constraints* of the quasi-static CFRP problem to the wave
setting:

* **through-holes** -- a fixed density field z with true (Ersatz) voids, so a
  plate with drilled/bolt holes is modeled honestly: hole stiffness
  ``s = eps_v + (1-eps_v) z^p`` scaling the *fiber* element stiffness, hole
  mass ``rho = z rho_f`` (single fiber-composite material + air, not a soft
  polymer inclusion);
* **fiber-curvature (curl) limits** -- the local constraints
  ``zeta_l^2 <= zeta_all^2`` on the out-of-plane curl of the unit orientation
  field (identically the fiber-path curvature, zeta = v . grad(theta) with
  v = (cos theta, sin theta)), enforced with the same augmented-Lagrangian +
  MMA machinery used for the quasi-static problem, restricted to elements
  where fiber is actually deposited (solid, non-hole).

Physics per frequency (complex-symmetric):
    D_i(theta) u_i = F,
    D_i = (1 + i eta) K(theta) - omega_i^2 M + i omega_i C_sponge,
    K   = sum_l s_l kf(theta_l),   s_l = eps_v + (1 - eps_v) z_l^p.

Objectives (both MINIMIZED):
    focus : J = -softmin_i(E_i / base_i),  E_i = u_i^H L u_i     (lens)
    match : J = softmax_i(J_i / base_i),   J_i = r_i^H W r_i,
            r_i = u_i - u_ref_i                                   (cloak)
with Kreisselmeier--Steinhauser soft-min/max for robust (worst-case)
broadband behavior; 'sum' mode replaces the KS weights by 1/nf.

Adjoint per frequency (D symmetric => D^H = conj(D)):
    conj(D_i) lambda_i = L u_i          (focus)
    conj(D_i) lambda_i = W r_i          (match)
    dJ_i/dtheta_l = -2 Re( lambda_{i,l}^H (1+i eta) s_l dkf(theta_l) u_{i,l} ).

All gradients are finite-difference verified in tests/test_wave_mfg.py.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .cfrp import Material, QuadMesh, CSRBFMapping, FourierStiffness, element_mass


# ---------------------------------------------------------------------- #
#  Fiber-curvature (curl) of an orientation-only design                   #
# ---------------------------------------------------------------------- #
def curl_field(rbf: CSRBFMapping, theta_hat):
    """zeta_l = cos(th) (Phi_x th^) + sin(th) (Phi_y th^)  = fiber curvature."""
    theta = rbf.Phi @ theta_hat
    a = rbf.Phi_x @ theta_hat
    b = rbf.Phi_y @ theta_hat
    return np.cos(theta) * a + np.sin(theta) * b


def curl_jacobian(rbf: CSRBFMapping, theta_hat):
    """Sparse Jacobian d zeta / d theta_hat, shape (N, M)."""
    theta = rbf.Phi @ theta_hat
    a = rbf.Phi_x @ theta_hat
    b = rbf.Phi_y @ theta_hat
    coef = -np.sin(theta) * a + np.cos(theta) * b
    J = (sp.diags(coef) @ rbf.Phi
         + sp.diags(np.cos(theta)) @ rbf.Phi_x
         + sp.diags(np.sin(theta)) @ rbf.Phi_y)
    return J.tocsr()


def _ks_softmin(v, rho):
    """KS soft-min with weights (sum w = 1)."""
    vmin = np.min(v)
    e = np.exp(-rho * (v - vmin))
    return vmin - np.log(e.sum()) / rho, e / e.sum()


def _ks_softmax(v, rho):
    """KS soft-max with weights (sum w = 1)."""
    vmax = np.max(v)
    e = np.exp(rho * (v - vmax))
    return vmax + np.log(e.sum()) / rho, e / e.sum()


# ---------------------------------------------------------------------- #
#  Two-phase harmonic plate, orientation-only design                      #
# ---------------------------------------------------------------------- #
class HarmonicPlate:
    """Fiber-composite plate with fixed density z (true Ersatz voids for
    through-holes), multi-frequency harmonic response, orientation design."""

    def __init__(self, mesh: QuadMesh, material: Material, rbf: CSRBFMapping,
                 omegas, *, eta=0.03, z=None, penal=3.0, eps_void=1e-4,
                 nsamp=720):
        self.mesh = mesh
        self.mat = material
        self.rbf = rbf
        self.M_ = rbf.M
        self.omegas = np.atleast_1d(np.asarray(omegas, float))
        self.nf = self.omegas.size
        self.eta = eta
        self.penal = penal
        self.eps_void = eps_void
        self.fs = FourierStiffness(material, mesh, nsamp=nsamp)
        self.Me = element_mass(mesh.dx, mesh.dy, rho=1.0)
        self.F = np.zeros(mesh.ndof, dtype=complex)
        self.C = sp.csr_matrix((mesh.ndof, mesh.ndof))
        self.set_density(np.ones(mesh.N) if z is None else z)
        # objective config
        self.obj_type = None          # 'focus' | 'match'
        self.L = None                 # list of diagonal weights (focus)
        self.W = None                 # diagonal weight (match)
        self.u_ref = None             # list of reference fields (match)
        self.base = np.ones(self.nf)

    # ---- assembly ---------------------------------------------------- #
    def _asm(self, mats):
        m = self.mesh
        return sp.csr_matrix((mats.ravel(), (m.iK, m.jK)),
                             shape=(m.ndof, m.ndof))

    def set_density(self, z):
        """Fixed density field; z=0 elements are true (Ersatz) voids."""
        self.z = np.asarray(z, float)
        self.s = self.eps_void + (1 - self.eps_void) * self.z**self.penal
        rho = self.z * self.mat.rho_f
        self.M = self._asm(rho[:, None, None] * self.Me[None])

    def set_load(self, F):
        self.F = np.asarray(F, dtype=complex)

    def set_sponge(self, c_elem):
        self.C = self._asm(np.asarray(c_elem)[:, None, None] * self.Me[None])

    def node_id(self, ix, iy):
        return iy * (self.mesh.nelx + 1) + ix

    # ---- convenience builders ---------------------------------------- #
    def plane_load(self, side="left", component=0):
        """Unit harmonic traction on one edge (plane pressure wave)."""
        m, F = self.mesh, np.zeros(self.mesh.ndof, dtype=complex)
        if side == "left":
            ids = [self.node_id(0, iy) for iy in range(m.nely + 1)]
        elif side == "right":
            ids = [self.node_id(m.nelx, iy) for iy in range(m.nely + 1)]
        else:
            raise ValueError(side)
        for n in ids:
            F[2 * n + component] = 1.0
        return F

    def box_weight(self, center, half):
        """Nodal weight = 1 on both dofs inside a square box."""
        cx, cy = center
        xs, ys = self.mesh.node_xy()
        sel = (np.abs(xs - cx) < half) & (np.abs(ys - cy) < half)
        w = np.zeros(self.mesh.ndof)
        w[0::2][sel] = 1.0
        w[1::2][sel] = 1.0
        return w

    # ---- objectives --------------------------------------------------- #
    def set_focus(self, weights):
        """Focus (lens) objective. ``weights``: one ndof vector, or one per
        frequency."""
        if isinstance(weights, np.ndarray) and weights.ndim == 1:
            weights = [weights] * self.nf
        assert len(weights) == self.nf
        self.L = [sp.diags(np.asarray(w, float)).tocsr() for w in weights]
        self.obj_type = 'focus'

    def set_match(self, u_ref, weight_dof):
        """Field-matching (cloak) objective. ``u_ref``: one reference field,
        or one per frequency."""
        if isinstance(u_ref, np.ndarray) and u_ref.ndim == 1:
            u_ref = [u_ref] * self.nf
        assert len(u_ref) == self.nf
        self.u_ref = [np.asarray(u, complex) for u in u_ref]
        self.W = sp.diags(np.asarray(weight_dof, float)).tocsr()
        self.obj_type = 'match'

    def set_baseline(self, base):
        self.base = np.atleast_1d(np.asarray(base, float))

    # ---- forward ------------------------------------------------------ #
    def solve(self, theta_hat):
        theta = self.rbf.theta(theta_hat)
        kel = self.s[:, None, None] * self.fs.kf(theta)
        K = self._asm(kel)
        us, Ds = [], []
        for w in self.omegas:
            D = ((1 + 1j * self.eta) * K - w**2 * self.M
                 + 1j * w * self.C).tocsc()
            Ds.append(D)
            us.append(spla.spsolve(D, self.F))
        self._state = dict(theta=theta, us=us, Ds=Ds)
        return us

    def raw_values(self, theta_hat=None):
        """Per-frequency raw objective values (focus energy / mismatch)."""
        if theta_hat is not None:
            self.solve(theta_hat)
        us = self._state['us']
        if self.obj_type == 'focus':
            return np.array([float(np.real(np.vdot(u, self.L[i] @ u)))
                             for i, u in enumerate(us)])
        r = [u - self.u_ref[i] for i, u in enumerate(us)]
        self._state['r'] = r
        return np.array([float(np.real(np.vdot(ri, self.W @ ri)))
                         for ri in r])

    def objective(self, theta_hat=None, mode='minmax', rho_ks=25.0):
        v = self.raw_values(theta_hat) / self.base
        self._obj = dict(v=v, mode=mode, rho_ks=rho_ks)
        if self.obj_type == 'focus':
            if mode == 'sum':
                self._obj['w'] = np.full(self.nf, 1.0 / self.nf)
                return -float(np.mean(v))
            smin, w = _ks_softmin(v, rho_ks)
            self._obj['w'] = w
            return -float(smin)
        # match: minimize the (worst-case) normalized mismatch
        if mode == 'sum':
            self._obj['w'] = np.full(self.nf, 1.0 / self.nf)
            return float(np.mean(v))
        smax, w = _ks_softmax(v, rho_ks)
        self._obj['w'] = w
        return float(smax)

    # ---- adjoint gradient ---------------------------------------------- #
    def grad(self):
        s_ = self._state
        us, Ds, theta = s_['us'], s_['Ds'], s_['theta']
        wts = self._obj['w']
        dkf = self.fs.dkf_dtheta(theta)                     # (N,8,8)
        edof = self.mesh.edof
        fac = (1 + 1j * self.eta) * self.s                  # per-element
        sign = -1.0 if self.obj_type == 'focus' else +1.0   # J = -+ softKS
        dJ_dtheta = np.zeros(self.mesh.N)
        for i in range(self.nf):
            u = us[i]
            if self.obj_type == 'focus':
                rhs = self.L[i] @ u
            else:
                rhs = self.W @ (u - self.u_ref[i])
            lam = spla.spsolve(Ds[i].conj(), rhs)
            uE, lamE = u[edof], lam[edof]
            term = np.einsum('ei,eij,ej->e', np.conj(lamE), dkf, uE) * fac
            dV_dtheta = -2.0 * term.real                    # d(raw_i)/dtheta
            dJ_dtheta += sign * (wts[i] / self.base[i]) * dV_dtheta
        return self.rbf.Phi.T @ dJ_dtheta

    # ---- curvature ----------------------------------------------------- #
    def curl(self, theta_hat):
        return curl_field(self.rbf, theta_hat)

    def curl_jac(self, theta_hat):
        return curl_jacobian(self.rbf, theta_hat)


# ---------------------------------------------------------------------- #
#  Orientation-only optimizer: MMA (+ augmented Lagrangian for curl)      #
# ---------------------------------------------------------------------- #
def optimize_orientation(prob: HarmonicPlate, theta_hat0, *,
                         zeta_all=None, curl_mask=None, active=None,
                         mode='minmax', rho_ks=25.0, move=0.1,
                         iters=90, mma_iter=5, max_outer=40,
                         lam0=0.0, mu0=20.0, xi=1.3, mu_max=1e6,
                         tolC=5e-3, verbose=True, callback=None):
    """Maximize the wave objective over theta_hat with optional local
    fiber-curvature constraints  zeta_l^2 <= zeta_all^2.

    * ``zeta_all=None``  : plain bound-constrained MMA (``iters`` steps).
    * ``zeta_all=value`` : augmented-Lagrangian outer loop (``max_outer``
      subproblems of ``mma_iter`` MMA steps), local constraints only on the
      elements in ``curl_mask`` (default: solid elements, z > 0.5 -- fiber is
      only deposited there).
    * ``active``         : boolean mask of designable support points; the rest
      stay frozen at their initial value.

    Returns dict with the best(-feasible) design and history.
    """
    from .cfrp_optimizer import MMA

    M = prob.M_
    th = np.array(theta_hat0, float)
    lo = np.where(active, -np.pi / 2, th) if active is not None \
        else np.full(M, -np.pi / 2)
    hi = np.where(active, np.pi / 2, th) if active is not None \
        else np.full(M, np.pi / 2)
    mma = MMA(lo, hi, move=move)
    hist = {'J': [], 'curl_max': []}

    # -------- plain MMA (no curvature limit) --------------------------- #
    if zeta_all is None:
        best = (np.inf, th.copy())
        for it in range(iters):
            J = prob.objective(th, mode=mode, rho_ks=rho_ks)
            g = prob.grad()
            cmax = float(np.max(np.abs(prob.curl(th))))
            hist['J'].append(J)
            hist['curl_max'].append(cmax)
            if J < best[0]:
                best = (J, th.copy())
            if verbose and (it % 10 == 0 or it == iters - 1):
                print(f"  [{it:3d}] J={J:.4f}  |zeta|max={cmax:.2f}")
            if callback:
                callback(it, th, J)
            th = mma.update(th, g)
        th = best[1]
        J = prob.objective(th, mode=mode, rho_ks=rho_ks)
        return dict(theta_hat=th, J=J, hist=hist,
                    curl_max=float(np.max(np.abs(prob.curl(th)))))

    # -------- augmented Lagrangian for local curl constraints ---------- #
    # constraints are normalized dimensionless:  g_l = zeta_l^2/zeta_all^2 - 1
    if curl_mask is None:
        curl_mask = prob.z > 0.5
    idx = np.flatnonzero(curl_mask)
    Nc = idx.size
    lam = np.full(Nc, lam0)
    mu = mu0
    best = {'J': np.inf, 'th': th.copy(), 'curl_max': np.inf}

    def _feasible_scale(thv):
        """Largest kappa in (0,1] with max|zeta(kappa*thv)| <= zeta_all.
        Straight fibers (kappa=0) have zeta=0, so bisection always succeeds;
        curl evaluation is a cheap sparse matvec."""
        cmax = np.max(np.abs(curl_field(prob.rbf, thv)[idx]))
        if cmax <= zeta_all:
            return 1.0
        lo_k, hi_k = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo_k + hi_k)
            c = np.max(np.abs(curl_field(prob.rbf, mid * thv)[idx]))
            if c <= zeta_all:
                lo_k = mid
            else:
                hi_k = mid
        return lo_k

    for k in range(max_outer):
        for _ in range(mma_iter):
            J = prob.objective(th, mode=mode, rho_ks=rho_ks)
            gJ = prob.grad()
            zeta = prob.curl(th)[idx]
            g = zeta**2 / zeta_all**2 - 1.0
            h = np.maximum(g, -lam / mu)
            fac = (lam + mu * h) * (g > -lam / mu)
            Jc = prob.curl_jac(th)[idx]                    # (Nc, M)
            gAL = gJ + (1.0 / Nc) * (Jc.T @ (2.0 * zeta * fac)) / zeta_all**2
            th = mma.update(th, gAL)

        J = prob.objective(th, mode=mode, rho_ks=rho_ks)
        zeta_full = prob.curl(th)
        cmax = float(np.max(np.abs(zeta_full[idx])))
        hist['J'].append(J)
        hist['curl_max'].append(cmax)
        feasible = cmax / zeta_all < 1.0 + tolC
        if feasible and J < best['J'] - 1e-9:
            best = {'J': J, 'th': th.copy(), 'curl_max': cmax}
        elif not feasible and cmax / zeta_all < 1.25:
            # near-feasible iterate: scale toward straight fibers until the
            # curl limit holds exactly, then score the *feasible* design
            kap = _feasible_scale(th)
            th_f = kap * th
            Jf = prob.objective(th_f, mode=mode, rho_ks=rho_ks)
            if Jf < best['J'] - 1e-9:
                cf = float(np.max(np.abs(prob.curl(th_f)[idx])))
                best = {'J': Jf, 'th': th_f, 'curl_max': cf}
        if verbose:
            print(f"  [outer {k:3d}] J={J:.4f}  |zeta|max={cmax:.3f}"
                  f"/{zeta_all}  mu={mu:.1e}  best={best['J']:.4f}")
        if callback:
            callback(k, th, J)
        # multiplier / penalty update (mu grows only while infeasible)
        zeta = zeta_full[idx]
        g = zeta**2 / zeta_all**2 - 1.0
        h = np.maximum(g, -lam / mu)
        lam = np.maximum(lam + mu * h, 0.0)
        if not feasible:
            mu = min(xi * mu, mu_max)

    th = best['th'] if np.isfinite(best['J']) else th
    J = prob.objective(th, mode=mode, rho_ks=rho_ks)
    return dict(theta_hat=th, J=J, hist=hist,
                curl_max=float(np.max(np.abs(prob.curl(th)[idx]))))


def ramp_sponge(mesh: QuadMesh, margin, strength,
                sides=("right", "top", "bottom")):
    """Quadratic mass-proportional absorbing ramp on chosen domain sides."""
    d = np.zeros(mesh.N)
    if "right" in sides:
        d = np.maximum(d, (mesh.cx - (mesh.Lx - margin)) / margin)
    if "left" in sides:
        d = np.maximum(d, (margin - mesh.cx) / margin)
    if "top" in sides:
        d = np.maximum(d, (mesh.cy - (mesh.Ly - margin)) / margin)
    if "bottom" in sides:
        d = np.maximum(d, (margin - mesh.cy) / margin)
    return np.maximum(0.0, d)**2 * strength
