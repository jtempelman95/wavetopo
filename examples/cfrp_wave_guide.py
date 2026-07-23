r"""
Co-designed topology + toolpath: route elastic-wave energy along a corridor that
threads BETWEEN two asymmetric through-holes, keep it out of the rest of the
panel, and keep the panel stiff -- all at once, on a conforming CFRP mesh.

This is the full CFRP methodology on a gmsh mesh that CONFORMS to the holes
(traction-free rims, not staircased soft voids), co-designing

  z      : element density (TOPOLOGY -- where material goes), volume-constrained
  theta  : element fiber orientation (TOOLPATH)

against a coupled objective:

  DYNAMIC  maximize path localization  E = u^H W u,  W = +1 in a corridor tube
           that weaves around the two holes, -w_leak in the rest of the interior
  STATIC   minimize cantilever compliance c (clamp left edge, transverse tip load)

  J = -w_dyn (E/E0)  +  w_stat (c/c0)  +  volume penalty      (minimized)

Both objectives share one K(z,theta); both adjoints are FD-verified
(wavetopo/cfrp_wave.py).  The holes are genuine geometry, so the corridor must
physically go around them.

    PYTHONPATH=. python examples/cfrp_wave_guide.py
"""
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from wavetopo.cfrp_wave import conforming_plate, CFRPWavePlate, cone_filter, Material
from wavetopo.cfrp_optimizer import MMA


# -- geometry (asymmetric holes; corridor threads between them) ------------- #
LX, LY = 6.0, 3.0
HOLES = [(2.0, 2.05, 0.35), (4.1, 0.95, 0.45)]
#  weave: enter mid-left, dip UNDER the high hole, rise OVER the low hole, exit
CORRIDOR = np.array([(0.3, 1.5), (2.0, 1.15), (4.1, 1.80), (5.7, 1.5)])
TUBE_HALF = 0.38


def dist_polyline(pts, poly):
    pts = np.atleast_2d(pts)
    d = np.full(pts.shape[0], np.inf)
    for k in range(len(poly) - 1):
        p0, seg = poly[k], poly[k + 1] - poly[k]
        t = np.clip((pts - p0) @ seg / (seg @ seg), 0, 1)
        d = np.minimum(d, np.linalg.norm(pts - (p0 + t[:, None] * seg), axis=1))
    return d


def setup(h, omega, eta, w_leak):
    mesh = conforming_plate(LX, LY, h, holes=HOLES)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    P = CFRPWavePlate(mesh, mat, omega=omega, eta=eta)
    xs, ys = mesh.npx[:, 0], mesh.npx[:, 1]

    # localized aperture drive at the corridor mouth (x~0, y~1.5)
    F = np.zeros(mesh.ndof, complex)
    mouth = (xs < 1e-6) & (np.abs(ys - 1.5) < 0.45)
    F[2 * np.flatnonzero(mouth)] = 1.0
    P.set_drive(F)

    # sponge on the three non-source edges
    marg = 0.5
    d = np.maximum.reduce([(mesh.cx - (LX - marg)) / marg,
                           (marg - mesh.cy) / marg,
                           (mesh.cy - (LY - marg)) / marg])
    P.set_sponge(np.maximum(0.0, d) ** 2 * 50.0)

    # signed weight: +1 in the corridor tube, -w_leak in the rest of the interior
    dcor = dist_polyline(mesh.npx, CORRIDOR)
    interior = (xs > 0.4) & (xs < LX - 0.4) & (ys > 0.3) & (ys < LY - 0.3)
    w = np.where(dcor < TUBE_HALF, 1.0, -w_leak) * interior
    P.set_weight(w)

    # static cantilever: clamp left edge, transverse load on right edge
    left = np.flatnonzero(xs < 1e-6)
    right = np.flatnonzero(xs > LX - 1e-6)
    Fs = np.zeros(mesh.ndof); Fs[2 * right + 1] = -1.0 / len(right)
    P.set_static(Fs, np.concatenate([2 * left, 2 * left + 1]))
    return mesh, mat, P, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.075)
    ap.add_argument("--omega", type=float, default=9.0)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--vol", type=float, default=0.55)
    ap.add_argument("--w-dyn", dest="w_dyn", type=float, default=1.0)
    ap.add_argument("--w-stat", dest="w_stat", type=float, default=1.0)
    ap.add_argument("--w-leak", dest="w_leak", type=float, default=0.4)
    ap.add_argument("--rmin", type=float, default=0.12)
    ap.add_argument("--cvol", type=float, default=40.0)
    ap.add_argument("--out", default="results/cfrp_wave_guide")
    a = ap.parse_args()

    mesh, mat, P, wsigned = setup(a.h, a.omega, a.eta, a.w_leak)
    nt = mesh.tris.shape[0]
    Filt = cone_filter(mesh, a.rmin)
    print(f"conforming plate: {mesh.nnode} nodes, {nt} tris, 2 holes; "
          f"omega={a.omega}", flush=True)

    # baselines: straight fibers, uniform density = vol target
    z = np.full(nt, a.vol); th = np.zeros(nt)
    E0 = P.dynamic(Filt @ z, th, False)[0]
    c0 = P.static(Filt @ z, th, False)[0]
    E0 = abs(E0) if abs(E0) > 1e-12 else 1.0
    print(f"baseline: E_path={E0:.3e}  compliance={c0:.3e}", flush=True)

    N = nt
    mma = MMA(np.concatenate([np.zeros(N), np.full(N, -np.pi / 2)]),
              np.concatenate([np.ones(N), np.full(N, np.pi / 2)]),
              move=0.08, adapt=True)
    A = mesh.area; Asum = A.sum()
    best = dict(J=np.inf); hist = []
    t0 = time.time()
    for it in range(a.iters):
        y = Filt @ z
        E, uE, dEz, dEth = P.dynamic(y, th)
        c, uc, dcz, dcth = P.static(y, th)
        vol = float((A @ y) / Asum)
        J = -a.w_dyn * (E / E0) + a.w_stat * (c / c0) + a.cvol * (vol - a.vol)**2
        # chain through the density filter for z-gradients
        dz = Filt.T @ (-a.w_dyn * dEz / E0 + a.w_stat * dcz / c0
                       + a.cvol * 2 * (vol - a.vol) * (A / Asum))
        dth = -a.w_dyn * dEth / E0 + a.w_stat * dcth / c0
        hist.append(dict(E=E / E0, c=c / c0, vol=vol, J=J))
        if abs(vol - a.vol) < 0.03 and J < best['J']:
            best = dict(J=J, z=z.copy(), th=th.copy(), y=y.copy(),
                        E=E / E0, c=c / c0, vol=vol)
        if it % 10 == 0 or it == a.iters - 1:
            print(f"[{it:3d}] E/E0={E/E0:7.2f}  c/c0={c/c0:6.3f}  "
                  f"vol={vol:.3f}  J={J:+.3f}", flush=True)
        W = mma.update(np.concatenate([z, th]), np.concatenate([dz, dth]), f=J)
        z, th = W[:N], W[N:]
    r = best if np.isfinite(best['J']) else dict(z=z, th=th, y=Filt @ z,
                                                 E=E / E0, c=c / c0, vol=vol)
    print(f"optimized in {time.time()-t0:.0f}s: E/E0={r['E']:.1f}x  "
          f"c/c0={r['c']:.3f}x  vol={r['vol']:.3f}", flush=True)

    np.savez(a.out + ".npz", trix=mesh.npx[:, 0], triy=mesh.npx[:, 1],
             tris=mesh.tris, cx=mesh.cx, cy=mesh.cy, z=r['z'], y=r['y'],
             th=r['th'], holes=np.array(HOLES), corridor=CORRIDOR,
             E_gain=r['E'], c_ratio=r['c'], vol=r['vol'],
             hist_E=[hh['E'] for hh in hist], hist_c=[hh['c'] for hh in hist],
             omega=a.omega, Lx=LX, Ly=LY, wsigned=wsigned)
    print("saved", a.out + ".npz", flush=True)


if __name__ == "__main__":
    main()
