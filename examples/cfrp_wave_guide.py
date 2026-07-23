r"""
Co-designed topology + toolpath: route elastic-wave energy along a corridor that
threads BETWEEN two asymmetric through-holes, DELIVER it to the far end, keep it
out of the rest of the panel, and keep the panel stiff -- on a conforming CFRP
mesh.

Full CFRP methodology on a gmsh mesh that CONFORMS to the holes (traction-free
rims), co-designing

  z_hat  -> filtered -> tanh-projected  z   : TOPOLOGY (crisp), volume-constrained
  theta_hat -> CS-RBF                 theta : TOOLPATH (smooth, manufacturable tows)

against a coupled objective:

  DYNAMIC  maximize DELIVERED path energy  E = u^H W u,  W = +ramp(arclength) in
           the corridor tube (rewarding energy that reaches the far end),
           -w_leak in the rest of the interior
  STATIC   minimize cantilever compliance c

  J = -w_dyn (E/E0)  +  w_stat (c/c0)  +  volume penalty          (minimized)

Both objectives share one K(z,theta); both adjoints FD-verified (cfrp_wave.py).

    PYTHONPATH=. python examples/cfrp_wave_guide.py
"""
import argparse
import time

import numpy as np

from wavetopo.cfrp_wave import (conforming_plate, CFRPWavePlate, cone_filter,
                                Material)
from wavetopo.cfrp import CSRBFMapping
from wavetopo.cfrp_optimizer import MMA


LX, LY = 6.0, 3.0
HOLES = [(2.0, 2.05, 0.35), (4.1, 0.95, 0.45)]
CORRIDOR = np.array([(0.3, 1.5), (2.0, 1.15), (4.1, 1.80), (5.7, 1.5)])
TUBE_HALF = 0.38


class _Adapter:
    """Expose a PlateMesh's element centroids as the (cx, cy, N) a CSRBFMapping
    needs, so the smooth-orientation map works on the unstructured mesh."""
    def __init__(self, mesh):
        self.cx, self.cy, self.N = mesh.cx, mesh.cy, mesh.tris.shape[0]


def corridor_coords(pts, poly):
    """Return (perp distance, fractional arc-length s in [0,1]) of each point
    projected onto the corridor polyline."""
    seglen = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seglen)]); total = cum[-1]
    best_d = np.full(pts.shape[0], np.inf); best_s = np.zeros(pts.shape[0])
    for k in range(len(poly) - 1):
        p0, seg = poly[k], poly[k + 1] - poly[k]
        t = np.clip((pts - p0) @ seg / (seg @ seg), 0, 1)
        proj = p0 + t[:, None] * seg
        d = np.linalg.norm(pts - proj, axis=1)
        m = d < best_d
        best_d[m] = d[m]; best_s[m] = (cum[k] + t[m] * seglen[k]) / total
    return best_d, best_s


def project(y, beta, eta=0.5):
    num = np.tanh(beta * eta) + np.tanh(beta * (y - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / den


def dproject(y, beta, eta=0.5):
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta * (1 - np.tanh(beta * (y - eta)) ** 2) / den


def setup(h, omega, eta, w_leak):
    mesh = conforming_plate(LX, LY, h, holes=HOLES)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    P = CFRPWavePlate(mesh, mat, omega=omega, eta=eta)
    xs, ys = mesh.npx[:, 0], mesh.npx[:, 1]

    F = np.zeros(mesh.ndof, complex)
    mouth = (xs < 1e-6) & (np.abs(ys - 1.5) < 0.45)
    F[2 * np.flatnonzero(mouth)] = 1.0
    P.set_drive(F)

    marg = 0.5
    d = np.maximum.reduce([(mesh.cx - (LX - marg)) / marg, (marg - mesh.cy) / marg,
                           (mesh.cy - (LY - marg)) / marg])
    P.set_sponge(np.maximum(0.0, d) ** 2 * 50.0)

    # signed weight with a DELIVERY RAMP: reward corridor energy in proportion to
    # how far along the corridor it is (0.15 -> 1 from mouth to far end), so the
    # optimizer is paid to carry energy to the output, not pile it at the source.
    dcor, scor = corridor_coords(mesh.npx, CORRIDOR)
    interior = (xs > 0.4) & (xs < LX - 0.4) & (ys > 0.3) & (ys < LY - 0.3)
    ramp = 0.15 + 0.85 * scor
    w = np.where(dcor < TUBE_HALF, ramp, -w_leak) * interior
    P.set_weight(w)

    left = np.flatnonzero(xs < 1e-6)
    right = np.flatnonzero(xs > LX - 1e-6)
    Fs = np.zeros(mesh.ndof); Fs[2 * right + 1] = -1.0 / len(right)
    P.set_static(Fs, np.concatenate([2 * left, 2 * left + 1]))
    return mesh, mat, P, w, interior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.07)
    ap.add_argument("--omega", type=float, default=9.0)
    ap.add_argument("--eta", type=float, default=0.006)
    ap.add_argument("--iters", type=int, default=140)
    ap.add_argument("--vol", type=float, default=0.55)
    ap.add_argument("--w-dyn", dest="w_dyn", type=float, default=1.0)
    ap.add_argument("--w-stat", dest="w_stat", type=float, default=0.6)
    ap.add_argument("--w-leak", dest="w_leak", type=float, default=0.5)
    ap.add_argument("--rmin", type=float, default=0.11)
    ap.add_argument("--rbf-spc", dest="rbf_spc", type=float, default=0.30)
    ap.add_argument("--cvol", type=float, default=200.0)
    ap.add_argument("--betamax", type=float, default=8.0)
    ap.add_argument("--out", default="results/cfrp_wave_guide")
    a = ap.parse_args()

    mesh, mat, P, wsigned, interior = setup(a.h, a.omega, a.eta, a.w_leak)
    nt = mesh.tris.shape[0]
    Filt = cone_filter(mesh, a.rmin)

    # CS-RBF smooth-orientation map on the triangle centroids
    sx = np.arange(a.rbf_spc / 2, LX, a.rbf_spc)
    sy = np.arange(a.rbf_spc / 2, LY, a.rbf_spc)
    SX, SY = np.meshgrid(sx, sy)
    rbf = CSRBFMapping(_Adapter(mesh), (SX.ravel(), SY.ravel()),
                       r_s=1.6 * a.rbf_spc)
    Mth = rbf.M
    print(f"conforming plate: {mesh.nnode} nodes, {nt} tris, 2 holes; "
          f"{Mth} orientation supports; omega={a.omega}", flush=True)

    z = np.full(nt, a.vol); thh = np.zeros(Mth)
    A = mesh.area; Asum = A.sum()

    def phys(zv, thhv, beta):
        y = project(Filt @ zv, beta)
        th = rbf.Phi @ thhv
        return y, th

    y0, th0 = phys(z, thh, 1.0)
    u0 = P.dynamic(y0, th0, False)[1]
    Enorm = float((np.abs(u0[0::2])**2 + np.abs(u0[1::2])**2)[interior].sum())
    Enorm = Enorm if Enorm > 1e-12 else 1.0
    E0 = Enorm                                   # stable positive normalizer
    c0 = P.static(y0, th0, False)[0]
    print(f"baseline: interior energy norm={Enorm:.3e}  compliance={c0:.3e}",
          flush=True)

    mma = MMA(np.concatenate([np.zeros(nt), np.full(Mth, -np.pi / 2)]),
              np.concatenate([np.ones(nt), np.full(Mth, np.pi / 2)]),
              move=0.06, adapt=True)
    best = dict(J=np.inf); hist = []
    t0 = time.time()
    for it in range(a.iters):
        beta = 1.0 if it < a.iters * 0.2 else min(
            a.betamax, a.betamax ** ((it - a.iters * 0.2) / (a.iters * 0.8)))
        yf = Filt @ z
        y = project(yf, beta); dpr = dproject(yf, beta)
        th = rbf.Phi @ thh
        E, uE, dEz, dEth = P.dynamic(y, th)
        c, uc, dcz, dcth = P.static(y, th)
        vol = float((A @ y) / Asum)
        J = -a.w_dyn * (E / E0) + a.w_stat * (c / c0) + a.cvol * (vol - a.vol)**2
        gy = (-a.w_dyn * dEz / E0 + a.w_stat * dcz / c0
              + a.cvol * 2 * (vol - a.vol) * (A / Asum))
        dz = Filt.T @ (dpr * gy)
        dth = rbf.Phi.T @ (-a.w_dyn * dEth / E0 + a.w_stat * dcth / c0)
        hist.append(dict(E=E / E0, c=c / c0, vol=vol, J=J))
        if abs(vol - a.vol) < 0.03 and beta > 2.0 and J < best['J']:
            best = dict(J=J, z=z.copy(), thh=thh.copy(), y=y.copy(), th=th.copy(),
                        E=E / E0, c=c / c0, vol=vol)
        if it % 15 == 0 or it == a.iters - 1:
            print(f"[{it:3d}] beta={beta:4.1f} E/E0={E/E0:7.2f} c/c0={c/c0:6.3f} "
                  f"vol={vol:.3f} J={J:+.3f}", flush=True)
        W = mma.update(np.concatenate([z, thh]), np.concatenate([dz, dth]), f=J)
        z, thh = W[:nt], W[nt:]
    r = best if np.isfinite(best['J']) else dict(
        z=z, y=project(Filt @ z, a.betamax), th=rbf.Phi @ thh,
        E=E / E0, c=c / c0, vol=vol)
    print(f"optimized in {time.time()-t0:.0f}s: E/E0={r['E']:.1f}x  "
          f"c/c0={r['c']:.3f}x  vol={r['vol']:.3f}", flush=True)

    np.savez(a.out + ".npz", trix=mesh.npx[:, 0], triy=mesh.npx[:, 1],
             tris=mesh.tris, cx=mesh.cx, cy=mesh.cy, z=r['z'], y=r['y'],
             th=r['th'], holes=np.array(HOLES), corridor=CORRIDOR,
             E_gain=r['E'], c_ratio=r['c'], vol=r['vol'],
             hist_E=[hh['E'] for hh in hist], hist_c=[hh['c'] for hh in hist],
             omega=a.omega, eta=a.eta, Lx=LX, Ly=LY, wsigned=wsigned)
    print("saved", a.out + ".npz", flush=True)


if __name__ == "__main__":
    main()
