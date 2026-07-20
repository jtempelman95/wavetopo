"""
Co-optimize unit-cell GEOMETRY (density z) and FIBER orientation (theta) to
maximize an ISOLATED valley gap of the elastic honeycomb -- i.e. a robust,
cleanly-quantized valley-Hall band gap, manufacturable (curl-limited fibers).

Objective (maximized):
    J = gap_34(K)                                   valley gap at the Dirac K
        - w_iso * [ relu(m - (w3-w2)) + relu(m - (w5-w4)) ]   isolation
        - w_vol * (vol - vtarget)^2                  keep the honeycomb fill
        - w_curl * mean(relu(|zeta| - zeta_all)^2)   printable fiber curvature

Design fields are periodic (toroidal filter + periodic CS-RBF) so the optimized
cell tiles.  The Dirac sits on the kx=0 line and is re-tracked each iteration
(it can drift as the geometry changes).
"""
from __future__ import annotations

import numpy as np

from .bloch import BlochProblem
from .cfrp_problem import density_filter
from .cfrp_optimizer import MMA
from .bandgap_opt import _softmax, _softmin


def _straddle(w2, sigma):
    """Indices of the two bands straddling sigma (the Dirac pair)."""
    below = np.where(w2 <= sigma)[0]
    above = np.where(w2 > sigma)[0]
    if len(below) == 0 or len(above) == 0:
        return None
    return below[-1], above[0]


class ValleyOptimizer:
    def __init__(self, bp: BlochProblem, *, R, period, vtarget,
                 w_iso=1.0, iso_margin=0.4, w_vol=40.0, w_curl=0.0,
                 zeta_all=None, nev=6):
        self.bp = bp
        self.P = density_filter(bp.mesh, R, period=period)
        self.period = period
        self.vtarget = vtarget
        self.w_iso, self.iso_margin = w_iso, iso_margin
        self.w_vol = w_vol
        self.w_curl, self.zeta_all = w_curl, zeta_all
        self.nev = nev
        self.Ly = period[1]

    # ---- track the Dirac on the kx=0 line --------------------------- #
    def track_dirac(self, y, th, sigma0, nky=41):
        bp = self.bp
        kys = np.linspace(0.05, 2 * np.pi / self.Ly - 0.05, nky)
        bp.assemble(y, th)
        best = (np.inf, None, None)
        for ky in kys:
            w = bp.bands_at_k(0.0, ky, self.nev, sigma=sigma0)
            st = _straddle(w**2, sigma0)
            if st is None:
                continue
            il, iu = st
            g = w[iu] - w[il]
            if g < best[0]:
                best = (g, ky, 0.5 * (w[il] + w[iu]))
        # refine
        _, ky0, wc = best
        if ky0 is None:
            return None
        kf = np.linspace(ky0 - 0.08, ky0 + 0.08, 17)
        bestf = (np.inf, ky0, wc)
        for ky in kf:
            w = bp.bands_at_k(0.0, ky, self.nev, sigma=sigma0)
            st = _straddle(w**2, sigma0)
            if st is None:
                continue
            il, iu = st
            g = w[iu] - w[il]
            if g < bestf[0]:
                bestf = (g, ky, 0.5 * (w[il] + w[iu]))
        return bestf[1], bestf[2]**2     # kyD, sigma=wc^2

    # ---- objective + gradient --------------------------------------- #
    def objective(self, z, th, kyD, sigma, want_grad=True):
        bp = self.bp
        y = self.P @ z
        bp.assemble(y, th)
        if want_grad:
            w, dz, dth = bp.eigen_sensitivity(0.0, kyD, self.nev, sigma=sigma)
        else:
            w = bp.bands_at_k(0.0, kyD, self.nev, sigma=sigma)
        st = _straddle(w**2, sigma)
        il, iu = st
        gap = w[iu] - w[il]
        info = dict(gap=gap, omega=0.5 * (w[il] + w[iu]))

        J = gap
        if want_grad:
            dJ_dy = dz[iu] - dz[il]
            dJ_dth = dth[iu] - dth[il]

        # isolation: keep band below (il-1) and above (iu+1) away
        m = self.iso_margin
        for (lo, hi, sgn) in [(il - 1, il, +1), (iu, iu + 1, +1)]:
            if lo < 0 or hi >= len(w):
                continue
            sep = w[hi] - w[lo]
            if sep < m:
                J -= self.w_iso * (m - sep)
                if want_grad:
                    dJ_dy -= self.w_iso * (-(dz[hi] - dz[lo]))
                    dJ_dth -= self.w_iso * (-(dth[hi] - dth[lo]))
        info['iso_below'] = w[il] - w[il - 1] if il >= 1 else np.nan
        info['iso_above'] = w[iu + 1] - w[iu] if iu + 1 < len(w) else np.nan

        # volume penalty on filtered density
        A, sumA = bp.mesh.A, bp.mesh.A.sum()
        vol = (A @ y) / sumA
        info['vol'] = vol
        J -= self.w_vol * (vol - self.vtarget)**2
        if want_grad:
            dJ_dy -= self.w_vol * 2 * (vol - self.vtarget) * (A / sumA)

        # curl penalty (manufacturable fiber curvature)
        if self.w_curl > 0 and self.zeta_all is not None:
            zeta = bp.rbf.theta(th)  # placeholder; curl uses Phi_x/Phi_y
            # curl = cos(theta) (Phi_x th) + sin(theta)(Phi_y th)
            thel = bp.rbf.theta(th)
            a = bp.rbf.Phi_x @ th
            b = bp.rbf.Phi_y @ th
            zeta = np.cos(thel) * a + np.sin(thel) * b
            viol = np.maximum(np.abs(zeta) - self.zeta_all, 0.0)
            info['curl_max'] = float(np.max(np.abs(zeta)))
            J -= self.w_curl * np.mean(viol**2)
            # (curl gradient omitted in v1; w_curl=0 by default)

        if want_grad:
            dJ_dz = self.P.T @ dJ_dy
            return J, dJ_dz, dJ_dth, info
        return J, info

    # ---- COMPLETE bulk gap (bands p3..p4 over a BZ k-sample) --------- #
    def set_ksample(self, ksample, p3=3, rho_ks=20.0):
        self.ksample = ksample
        self.p3, self.p4 = p3, p3 + 1
        self.rho_ks = rho_ks

    def objective_complete(self, z, th, want_grad=True):
        bp = self.bp
        y = self.P @ z
        bp.assemble(y, th)
        w3, w4 = [], []
        d3z, d4z, d3t, d4t = [], [], [], []
        for (kx, ky) in self.ksample:
            if want_grad:
                w, dz, dt = bp.eigen_sensitivity(kx, ky, self.p4 + 2)
                d3z.append(dz[self.p3]); d4z.append(dz[self.p4])
                d3t.append(dt[self.p3]); d4t.append(dt[self.p4])
            else:
                w = bp.bands_at_k(kx, ky, self.p4 + 2)
            w3.append(w[self.p3]); w4.append(w[self.p4])
        w3 = np.array(w3); w4 = np.array(w4)
        smax3, wmax = _softmax(w3, self.rho_ks)
        smin4, wmin = _softmin(w4, self.rho_ks)
        gap = smin4 - smax3                       # complete bulk gap
        info = dict(gap=gap, top3=w3.max(), bot4=w4.min(),
                    omega=0.5 * (w3.max() + w4.min()))
        J = gap
        A, sumA = bp.mesh.A, bp.mesh.A.sum()
        vol = (A @ y) / sumA
        info['vol'] = vol
        J -= self.w_vol * (vol - self.vtarget)**2
        if want_grad:
            d4z = np.array(d4z); d3z = np.array(d3z)
            d4t = np.array(d4t); d3t = np.array(d3t)
            dJ_dy = wmin @ d4z - wmax @ d3z
            dJ_dy -= self.w_vol * 2 * (vol - self.vtarget) * (A / sumA)
            dJ_dth = wmin @ d4t - wmax @ d3t
            dJ_dz = self.P.T @ dJ_dy
            return J, dJ_dz, dJ_dth, info
        return J, info

    def run_complete(self, z0, th0, *, max_iter=80, move=0.08, verbose=True):
        N, M = self.bp.mesh.N, self.bp.M
        xmin = np.concatenate([np.zeros(N), np.full(M, -np.pi / 2)])
        xmax = np.concatenate([np.ones(N), np.full(M, np.pi / 2)])
        mma = MMA(xmin, xmax, move=move)
        z, th = np.array(z0, float), np.array(th0, float)
        hist = []
        best = (-np.inf, z.copy(), th.copy())
        for it in range(max_iter):
            J, dz, dth, info = self.objective_complete(z, th, want_grad=True)
            hist.append(info)
            if info['gap'] > best[0] and abs(info['vol'] - self.vtarget) < 0.04:
                best = (info['gap'], z.copy(), th.copy())
            if verbose and (it % 5 == 0 or it == max_iter - 1):
                print(f"[{it:3d}] complete gap={info['gap']:+.4f} "
                      f"window=[{info['top3']:.3f},{info['bot4']:.3f}] "
                      f"vol={info['vol']:.3f}")
            W = np.concatenate([z, th])
            Wn = mma.update(W, -np.concatenate([dz, dth]))
            z, th = Wn[:N], Wn[N:]
        return dict(z=best[1], th=best[2], gap=best[0], hist=hist,
                    omega=0.5 * (hist[-1]['top3'] + hist[-1]['bot4']))

    # ---- run -------------------------------------------------------- #
    def run(self, z0, th0, sigma0, *, max_iter=80, move=0.08, verbose=True):
        N, M = self.bp.mesh.N, self.bp.M
        xmin = np.concatenate([np.zeros(N), np.full(M, -np.pi / 2)])
        xmax = np.concatenate([np.ones(N), np.full(M, np.pi / 2)])
        mma = MMA(xmin, xmax, move=move)
        z, th = np.array(z0, float), np.array(th0, float)
        sigma = sigma0
        hist = []
        best = (-np.inf, z.copy(), th.copy())
        for it in range(max_iter):
            y = self.P @ z
            tr = self.track_dirac(y, th, sigma)
            if tr is None:
                print("lost the Dirac; stopping"); break
            kyD, sigma = tr
            J, dz, dth, info = self.objective(z, th, kyD, sigma, want_grad=True)
            hist.append(info)
            if info['gap'] > best[0] and abs(info['vol'] - self.vtarget) < 0.03:
                best = (info['gap'], z.copy(), th.copy())
            if verbose and (it % 5 == 0 or it == max_iter - 1):
                print(f"[{it:3d}] gap={info['gap']:.4f} "
                      f"iso(below/above)={info['iso_below']:.3f}/"
                      f"{info['iso_above']:.3f} vol={info['vol']:.3f} "
                      f"omega={info['omega']:.3f} kyD={kyD:.3f}")
            W = np.concatenate([z, th])
            Wn = mma.update(W, -np.concatenate([dz, dth]))   # minimize -J
            z, th = Wn[:N], Wn[N:]
        return dict(z=best[1], th=best[2], gap=best[0], hist=hist, sigma=sigma)
