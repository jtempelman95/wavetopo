"""
Directional phononic band-gap optimization on the fiber-composite unit cell.

Maximize the gap between bands (nlow, nlow+1) for waves travelling in a TARGET
direction, optionally while suppressing it in the orthogonal direction, using
the fiber orientation field to make the gap directional.

Objective (minimized):
    J = [ softmax_k(omega_low) - softmin_k(omega_high) ]            (= -gap_target)
      + w_orth * relu( gap_orth )                                   (kill orthogonal gap)
      + c_vol * (vol - v_target)^2                                  (volume penalty)

Design variables: density z (filtered) and CS-RBF orientation theta_hat.
Gradients use the FD-verified eigenvalue sensitivities in bloch.py + KS
aggregation; updates via the bound-constrained MMA from cfrp_optimizer.
"""
from __future__ import annotations

import numpy as np

from .bloch import BlochProblem, directional_path
from .cfrp_problem import density_filter
from .cfrp_optimizer import MMA


def _softmax(v, rho):
    """KS soft-max and per-element weights (stable)."""
    vmax = np.max(v)
    e = np.exp(rho * (v - vmax))
    s = e.sum()
    sm = vmax + np.log(s) / rho
    return sm, e / s


def _softmin(v, rho):
    sm, w = _softmax(-v, rho)
    return -sm, w


class BandGapOptimizer:
    def __init__(self, bp: BlochProblem, *, nlow, n_bands, R,
                 target_dir=np.pi / 2, orth_dir=0.0, n_k=9, kfrac=1.0,
                 rho_ks=12.0, v_target=0.5, c_vol=50.0, w_orth=0.0,
                 period=None):
        self.bp = bp
        self.nlow = nlow
        self.n_bands = n_bands
        self.P = density_filter(bp.mesh, R, period=period)
        self.rho_ks = rho_ks
        self.v_target = v_target
        self.c_vol = c_vol
        self.w_orth = w_orth
        a = bp.mesh.Lx
        # drop the Gamma point (omega=0, degenerate) from the aggregation set
        self.K_tar = directional_path(a, a, target_dir, kfrac, n_k)[1:]
        self.K_orth = directional_path(a, a, orth_dir, kfrac, n_k)[1:]

    # ---- gap + gradient along a set of k-points --------------------- #
    def _gap_and_grad(self, kset, want_grad=True):
        bp, nl = self.bp, self.nlow
        low, high = [], []
        dlow_y, dhigh_y, dlow_t, dhigh_t = [], [], [], []
        for (kx, ky) in kset:
            if want_grad:
                w, dz, dt = bp.eigen_sensitivity(kx, ky, self.n_bands)
                dlow_y.append(dz[nl]); dhigh_y.append(dz[nl + 1])
                dlow_t.append(dt[nl]); dhigh_t.append(dt[nl + 1])
            else:
                w = bp.bands_at_k(kx, ky, self.n_bands)
            low.append(w[nl]); high.append(w[nl + 1])
        low = np.array(low); high = np.array(high)
        smax_low, wlo = _softmax(low, self.rho_ks)
        smin_high, whi = _softmin(high, self.rho_ks)
        gap = smin_high - smax_low
        out = dict(gap=gap, top_low=low.max(), bot_high=high.min())
        if want_grad:
            dlow_y = np.array(dlow_y); dhigh_y = np.array(dhigh_y)
            dlow_t = np.array(dlow_t); dhigh_t = np.array(dhigh_t)
            # d gap / d* = sum_k whi_k dhigh_k - sum_k wlo_k dlow_k
            out['dgap_y'] = whi @ dhigh_y - wlo @ dlow_y
            out['dgap_t'] = whi @ dhigh_t - wlo @ dlow_t
        return out

    # ---- objective + gradient --------------------------------------- #
    def objective(self, z, th, want_grad=True):
        mesh = self.bp.mesh
        y = self.P @ z
        self.bp.assemble(y, th)
        tar = self._gap_and_grad(self.K_tar, want_grad)
        J = -tar['gap']
        info = dict(gap_target=tar['gap'])

        if want_grad:
            dJ_dy = -tar['dgap_y']
            dJ_dt = -tar['dgap_t']

        # suppress orthogonal-direction gap (keep it directional)
        if self.w_orth > 0:
            orth = self._gap_and_grad(self.K_orth, want_grad)
            info['gap_orth'] = orth['gap']
            if orth['gap'] > 0:
                J += self.w_orth * orth['gap']
                if want_grad:
                    dJ_dy += self.w_orth * orth['dgap_y']
                    dJ_dt += self.w_orth * orth['dgap_t']
        elif not want_grad:
            info['gap_orth'] = self._gap_and_grad(self.K_orth, False)['gap']

        # volume penalty on the filtered density
        A, sumA = mesh.A, mesh.A.sum()
        vol = (A @ y) / sumA
        info['vol'] = vol
        J += self.c_vol * (vol - self.v_target)**2
        if want_grad:
            dvol_dy = A / sumA
            dJ_dy += self.c_vol * 2 * (vol - self.v_target) * dvol_dy
            dJ_dz = self.P.T @ dJ_dy
            return J, dJ_dz, dJ_dt, info
        return J, info

    # ---- run -------------------------------------------------------- #
    def run(self, z0, th0, *, max_iter=120, verbose=True):
        N, M = self.bp.mesh.N, self.bp.M
        xmin = np.concatenate([np.zeros(N), np.full(M, -np.pi / 2)])
        xmax = np.concatenate([np.ones(N), np.full(M, np.pi / 2)])
        mma = MMA(xmin, xmax, move=0.1)
        z, th = np.array(z0, float), np.array(th0, float)
        best = dict(gap=-np.inf)
        hist = []
        for it in range(max_iter):
            J, dz, dt, info = self.objective(z, th, want_grad=True)
            hist.append(info)
            if info['gap_target'] > best['gap'] and abs(info['vol'] - self.v_target) < 0.03:
                best = dict(gap=info['gap_target'], z=z.copy(), th=th.copy(),
                            info=dict(info))
            if verbose and (it % 5 == 0 or it == max_iter - 1):
                go = info.get('gap_orth', float('nan'))
                print(f"[{it:3d}] gap_tar={info['gap_target']:+.4f} "
                      f"gap_orth={go:+.4f} vol={info['vol']:.3f} J={J:.4f}")
            W = np.concatenate([z, th])
            Wn = mma.update(W, np.concatenate([dz, dt]))
            z, th = Wn[:N], Wn[N:]
        result = best if np.isfinite(best['gap']) else dict(
            gap=info['gap_target'], z=z, th=th, info=info)
        result['hist'] = hist
        return result
