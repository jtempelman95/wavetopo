"""
Multi-functional metamaterial: STIFF *and* wave-functional, in one design.

A load-bearing panel that also controls vibration has to satisfy two objectives
that pull in different directions:

  * STATIC   maximise the homogenised stiffness C_H (homogenize.py)
  * DYNAMIC  open a complete band gap, and set the POLARITY of the gap-edge
             modes -- which of the two competing mode symmetries sits lower

Both are functions of the SAME two design fields, density z (geometry) and fiber
orientation theta (toolpath), on the SAME cell.  That is the point: geometry
alone must trade stiffness against gap, whereas orientation adds a second,
partly independent lever -- it changes the stiffness tensor's ANISOTROPY without
removing material, so it can retune the bands at nearly fixed density.

Band polarity
-------------
At Gamma the two acoustic branches sit at zero.  The optical modes above them
carry definite parity under inversion r -> -r (the cell is inversion symmetric
when the design is).  We classify each mode by

    P = <phi, I phi> / <phi, phi>   in [-1, 1]

with I the inversion permutation acting on the periodic node set; P ~ +1 is an
even (monopole/quadrupole-like) mode, P ~ -1 an odd (dipole-like) mode.  Driving
the even mode below the odd one -- a BAND INVERSION -- flips the polarity of the
gap without changing its width, and is the phononic analogue of the band
inversion that underlies topological edge states.  It is a genuinely different
objective from "make the gap wide", and orientation turns out to be an effective
handle on it.

Objective (minimised)
---------------------
    J = -w_K * C_bulk/C_ref                       (stiffness, maximised)
        -w_g * gap/gap_ref                        (complete gap, maximised)
        +w_p * polarity_penalty                   (optional: force an inversion)
        +c_vol*(vol - v_target)^2
"""
from __future__ import annotations

import numpy as np

from .bloch import BlochProblem
from .bandgap_opt import _softmax, _softmin
from .cfrp_problem import density_filter
from .cfrp_optimizer import MMA
from .homogenize import Homogenizer


class MultiFunctionalOptimizer:
    def __init__(self, bp: BlochProblem, *, band, n_bands, R, kset,
                 w_stiff=1.0, w_gap=1.0, w_pol=0.0, pol_target=None,
                 rho_ks=15.0, v_target=0.5, c_vol=80.0, period=None,
                 passive=None, C_ref=None, gap_ref=None):
        self.bp = bp
        self.band = band                 # gap sits between `band` and `band+1`
        self.n_bands = n_bands
        self.P = density_filter(bp.mesh, R, period=period)
        self.kset = list(kset)
        self.rho_ks = rho_ks
        self.w_stiff, self.w_gap, self.w_pol = w_stiff, w_gap, w_pol
        self.pol_target = pol_target     # +1 -> even below odd, -1 -> odd below
        self.v_target, self.c_vol = v_target, c_vol
        self.passive = None if passive is None else np.asarray(passive, bool)
        self.C_ref = C_ref               # normalisers, set on first evaluation
        self.gap_ref = gap_ref
        self.Wb = Homogenizer.bulk_weights()
        self._inv = self._inversion_map()

    # ---- inversion permutation on the reduced (periodic) node set ----- #
    def _inversion_map(self):
        m = self.bp.mesh
        nx, ny = m.nelx, m.nely
        idx = np.arange(nx*ny).reshape(ny, nx)
        inv = idx[::-1, ::-1]                    # r -> -r on the periodic torus
        inv = np.roll(np.roll(inv, 1, axis=0), 1, axis=1)
        perm = np.empty(2*nx*ny, dtype=int)
        perm[0::2] = 2*inv.ravel()
        perm[1::2] = 2*inv.ravel() + 1
        return perm

    def parity(self, vec):
        """<phi, I phi>/<phi,phi> in [-1,1]; the displacement flips sign under
        inversion, hence the leading minus."""
        v = np.asarray(vec).ravel()
        num = -np.vdot(v, v[self._inv]).real
        den = np.vdot(v, v).real
        return float(num/max(den, 1e-30))

    def _apply(self, z):
        y = self.P @ z
        if self.passive is not None:
            y = y.copy(); y[self.passive] = 1.0
        return y

    # ---- one full evaluation ------------------------------------------ #
    def evaluate(self, z, th, want_grad=True):
        b, bp = self.band, self.bp
        y = self._apply(z)
        bp.assemble(y, th)

        # ---------- static: homogenised stiffness ---------------------- #
        H = Homogenizer(bp)
        C, _ = H.solve()
        Cb = Homogenizer.bulk(C)
        if self.C_ref is None:
            self.C_ref = max(Cb, 1e-12)
        gz_C, gth_C = H.dC_d(self.Wb) if want_grad else (0.0, 0.0)

        # ---------- dynamic: complete gap over the k-set --------------- #
        wl, wu, gzl, gzu, gthl, gthu = [], [], [], [], [], []
        for (kx, ky) in self.kset:
            if want_grad:
                w, dz_, dt_ = bp.eigen_sensitivity(kx, ky, self.n_bands)
                gzl.append(dz_[b]); gzu.append(dz_[b+1])
                gthl.append(dt_[b]); gthu.append(dt_[b+1])
            else:
                w = bp.bands_at_k(kx, ky, self.n_bands)
            wl.append(w[b]); wu.append(w[b+1])
        wl, wu = np.array(wl), np.array(wu)
        top_l, w_top = _softmax(wl, self.rho_ks)      # top of band b
        bot_u, w_bot = _softmin(wu, self.rho_ks)      # bottom of band b+1
        gap = bot_u - top_l
        if self.gap_ref is None:
            self.gap_ref = max(abs(gap), 1e-3)

        # ---------- polarity of the gap-edge modes at Gamma ------------ #
        wg, _, Vg, Tg = bp.bands_at_k(0.0, 0.0, self.n_bands, return_vec=True)
        par = [self.parity(Vg[:, j]) for j in range(self.n_bands)]

        A, sumA = bp.mesh.A, bp.mesh.A.sum()
        vol = (A @ y)/sumA
        J = (-self.w_stiff*Cb/self.C_ref
             - self.w_gap*gap/self.gap_ref
             + self.c_vol*(vol - self.v_target)**2)

        info = dict(C_bulk=Cb, C=C, gap=gap, vol=vol,
                    parity=par, w_gamma=wg,
                    band_lo=wl.min(), band_hi=wu.max())
        if not want_grad:
            return J, info

        dJ_dy = -self.w_stiff*gz_C/self.C_ref
        dJ_dt = -self.w_stiff*gth_C/self.C_ref
        dgap_z = (w_bot @ np.array(gzu)) - (w_top @ np.array(gzl))
        dgap_t = (w_bot @ np.array(gthu)) - (w_top @ np.array(gthl))
        dJ_dy = dJ_dy - self.w_gap*dgap_z/self.gap_ref
        dJ_dt = dJ_dt - self.w_gap*dgap_t/self.gap_ref
        dJ_dy = dJ_dy + self.c_vol*2*(vol - self.v_target)*(A/sumA)
        if self.passive is not None:
            dJ_dy = dJ_dy.copy(); dJ_dy[self.passive] = 0.0
        return J, self.P.T @ dJ_dy, dJ_dt, info

    # ---- driver -------------------------------------------------------- #
    def run(self, z0, th0, *, max_iter=100, move=0.08, verbose=True):
        N, M = self.bp.mesh.N, self.bp.M
        xmin = np.concatenate([np.zeros(N), np.full(M, -np.pi/2)])
        xmax = np.concatenate([np.ones(N), np.full(M, np.pi/2)])
        mma = MMA(xmin, xmax, move=move, adapt=True)
        z, th = np.array(z0, float), np.array(th0, float)
        best = dict(score=np.inf); hist = []
        for it in range(max_iter):
            J, dz, dt, info = self.evaluate(z, th)
            hist.append(dict(C=info['C_bulk'], gap=info['gap'], vol=info['vol'], J=J))
            feasible = abs(info['vol'] - self.v_target) < 0.03
            if feasible and J < best['score']:
                best = dict(score=J, z=z.copy(), th=th.copy(), info=dict(info))
            if verbose and (it % 10 == 0 or it == max_iter-1):
                print(f"[{it:3d}] C_bulk={info['C_bulk']:8.3f} gap={info['gap']:+.4f} "
                      f"vol={info['vol']:.3f} J={J:+.4f} move={mma.move:.3f}",
                      flush=True)
            W = np.concatenate([z, th])
            Wn = mma.update(W, np.concatenate([dz, dt]), f=J)
            z, th = Wn[:N], Wn[N:]
        out = best if np.isfinite(best['score']) else dict(z=z, th=th, info=info)
        out['hist'] = hist
        return out
