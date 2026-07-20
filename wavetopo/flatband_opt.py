"""
Flat-band + band-gap co-optimization on the fiber-composite unit cell.

Design an isolated FLAT band: minimize the bandwidth of a target band `band`
over the Brillouin zone (a slow-wave / high-density-of-states mode) while opening
gaps to the neighbouring bands so it sits isolated.  Both the density (topology)
and the fiber orientation (toolpath) are designed, exactly as in the directional
band-gap problem.

Objective (minimized), with KS soft aggregation over a k-set covering the BZ:
    J = w_flat * ( softmax_k w_b - softmin_k w_b )              (flatten band b)
      - w_up   * ( softmin_k w_{b+1} - softmax_k w_b )          (open gap above)
      - w_dn   * ( softmin_k w_b     - softmax_k w_{b-1} )      (open gap below)
      + c_vol  * (vol - v_target)^2                             (volume target)

Design variables: filtered density z and CS-RBF orientation theta_hat.  Gradients
use the FD-verified eigenvalue sensitivities in bloch.py + KS aggregation, updated
by the bound-constrained MMA.
"""
from __future__ import annotations

import numpy as np

from .bloch import BlochProblem
from .bandgap_opt import _softmax, _softmin
from .cfrp_problem import density_filter
from .cfrp_optimizer import MMA


class FlatBandOptimizer:
    def __init__(self, bp: BlochProblem, *, band, n_bands, R, kset,
                 rho_ks=15.0, v_target=0.5, c_vol=80.0,
                 w_flat=1.0, w_up=1.0, w_dn=1.0, period=None, passive=None):
        self.bp = bp
        self.band = band
        self.n_bands = n_bands
        self.P = density_filter(bp.mesh, R, period=period)
        self.rho_ks = rho_ks
        self.v_target = v_target
        self.c_vol = c_vol
        self.w_flat, self.w_up, self.w_dn = w_flat, w_up, w_dn
        self.kset = list(kset)
        # passive SOLID elements (e.g. a frame on the cell walls): filtered density
        # is forced to 1 there and their gradient is zeroed -> guarantees mass on
        # all four walls and a connected, periodic, manufacturable cell.
        self.passive = None if passive is None else np.asarray(passive, bool)

    # ---- band values (and grads) for b-1, b, b+1 over the k-set -------- #
    def _collect(self, want_grad):
        b = self.band
        wb1, wb, wb2 = [], [], []                 # omega_{b-1}, omega_b, omega_{b+1}
        gz = {b-1: [], b: [], b+1: []}
        gt = {b-1: [], b: [], b+1: []}
        for (kx, ky) in self.kset:
            if want_grad:
                w, dz, dt = self.bp.eigen_sensitivity(kx, ky, self.n_bands)
                for j in (b-1, b, b+1):
                    gz[j].append(dz[j]); gt[j].append(dt[j])
            else:
                w = self.bp.bands_at_k(kx, ky, self.n_bands)
            wb1.append(w[b-1]); wb.append(w[b]); wb2.append(w[b+1])
        out = dict(wb1=np.array(wb1), wb=np.array(wb), wb2=np.array(wb2))
        if want_grad:
            out['gz'] = {j: np.array(v) for j, v in gz.items()}
            out['gt'] = {j: np.array(v) for j, v in gt.items()}
        return out

    # ---- objective + gradient ---------------------------------------- #
    def objective(self, z, th, want_grad=True):
        b = self.band
        mesh = self.bp.mesh
        y = self.P @ z
        if self.passive is not None:
            y = y.copy(); y[self.passive] = 1.0        # solid frame on the walls
        self.bp.assemble(y, th)
        c = self._collect(want_grad)
        rho = self.rho_ks
        smax_b, w_maxb = _softmax(c['wb'], rho)       # top of band b
        smin_b, w_minb = _softmin(c['wb'], rho)       # bottom of band b
        smin_up, w_minup = _softmin(c['wb2'], rho)    # bottom of band b+1
        smax_dn, w_maxdn = _softmax(c['wb1'], rho)    # top of band b-1

        width = smax_b - smin_b
        gap_up = smin_up - smax_b
        gap_dn = smin_b - smax_dn
        A, sumA = mesh.A, mesh.A.sum()
        vol = (A @ y) / sumA
        J = (self.w_flat*width - self.w_up*gap_up - self.w_dn*gap_dn
             + self.c_vol*(vol - self.v_target)**2)
        info = dict(width=width, gap_up=gap_up, gap_dn=gap_dn, vol=vol,
                    wb_lo=c['wb'].min(), wb_hi=c['wb'].max())
        if not want_grad:
            return J, info

        gz, gt = c['gz'], c['gt']
        # d(softmax_b)/d* = sum_k w_maxb[k] dwb[k]/d* ; etc.
        dsmax_b_z = w_maxb @ gz[b];   dsmax_b_t = w_maxb @ gt[b]
        dsmin_b_z = w_minb @ gz[b];   dsmin_b_t = w_minb @ gt[b]
        dsmin_up_z = w_minup @ gz[b+1]; dsmin_up_t = w_minup @ gt[b+1]
        dsmax_dn_z = w_maxdn @ gz[b-1]; dsmax_dn_t = w_maxdn @ gt[b-1]

        dwidth_z = dsmax_b_z - dsmin_b_z; dwidth_t = dsmax_b_t - dsmin_b_t
        dgapup_z = dsmin_up_z - dsmax_b_z; dgapup_t = dsmin_up_t - dsmax_b_t
        dgapdn_z = dsmin_b_z - dsmax_dn_z; dgapdn_t = dsmin_b_t - dsmax_dn_t

        dJ_dy = self.w_flat*dwidth_z - self.w_up*dgapup_z - self.w_dn*dgapdn_z
        dJ_dt = self.w_flat*dwidth_t - self.w_up*dgapup_t - self.w_dn*dgapdn_t
        dJ_dy += self.c_vol*2*(vol - self.v_target)*(A/sumA)
        if self.passive is not None:
            dJ_dy = dJ_dy.copy(); dJ_dy[self.passive] = 0.0   # frame is fixed
        dJ_dz = self.P.T @ dJ_dy
        return J, dJ_dz, dJ_dt, info

    # ---- run --------------------------------------------------------- #
    def run(self, z0, th0, *, max_iter=120, verbose=True):
        N, M = self.bp.mesh.N, self.bp.M
        xmin = np.concatenate([np.zeros(N), np.full(M, -np.pi/2)])
        xmax = np.concatenate([np.ones(N), np.full(M, np.pi/2)])
        mma = MMA(xmin, xmax, move=0.1)
        z, th = np.array(z0, float), np.array(th0, float)
        best = dict(score=np.inf); hist = []
        for it in range(max_iter):
            J, dz, dt, info = self.objective(z, th, want_grad=True)
            hist.append(info)
            # track best FEASIBLE (near volume target) by a flat+isolated score
            score = info['width'] - min(info['gap_up'], 0) - min(info['gap_dn'], 0)
            if score < best['score'] and abs(info['vol'] - self.v_target) < 0.03:
                best = dict(score=score, z=z.copy(), th=th.copy(), info=dict(info))
            if verbose and (it % 5 == 0 or it == max_iter-1):
                print(f"[{it:3d}] width={info['width']:.4f} "
                      f"gap_up={info['gap_up']:+.4f} gap_dn={info['gap_dn']:+.4f} "
                      f"vol={info['vol']:.3f} J={J:.4f}")
            W = np.concatenate([z, th])
            Wn = mma.update(W, np.concatenate([dz, dt]))
            z, th = Wn[:N], Wn[N:]
        result = best if np.isfinite(best['score']) else dict(
            z=z, th=th, info=info)
        result['hist'] = hist
        return result
