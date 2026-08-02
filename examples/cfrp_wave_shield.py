r"""
Broadband, multi-source, multi-target wave shield by fiber orientation on a SOLID
CFRP plate with TRUE conforming holes -- modeled after the working elastic cloak
(Fig. 5), which wraps the fiber toolpath around a real traction-free void.

The plate is solid fiber with drilled through-holes (the mass reduction); the fiber
TOOLPATH is optimized so harmonic vibration entering at SEVERAL mounting points (the
two bottom corners) is diverted away from BOTH top-corner regions, across a BAND of
frequencies, while the panel stays stiff.  The sources are independent load cases:
the shield must quiet the protected corners against each source and each frequency at
once.  Because every void is a REAL hole cut from the mesh (no cells inside), the wave
reflects off its traction-free rim and there is provably ZERO wave energy in any void.

  theta_hat -> CS-RBF  theta : TOOLPATH (smooth, manufacturable tows)

  DYNAMIC  minimize  sum_s u_{s,k}^H W u_{s,k}  over sources s at each frequency
           omega_k (W = +1 in the union of the protected disks), averaged over the
           band -- a broadband, multi-source shield.  One LU factorization per
           frequency serves all source load cases (CFRPWavePlate.dynamic_multi).
  STATIC   minimize cantilever compliance c (frequency-independent; one solve)
           + fiber-curvature (curl) limit (printable turning radius)

    PYTHONPATH=. python examples/cfrp_wave_shield.py
"""
import argparse
import time

import numpy as np

from wavetopo.cfrp_wave import (conforming_plate, CFRPWavePlate, curl_operators,
                                curl_penalty, director_curl, Material)
from wavetopo.cfrp import CSRBFMapping
from wavetopo.cfrp_optimizer import MMA


LX, LY = 7.0, 4.0
# vibration ingress at BOTH bottom corners (independent load cases)
SOURCES = [(0.8, 1.0, 0.42), (6.2, 1.0, 0.42)]
# protected regions: BOTH top corners, kept quiet against every source
PROTECTS = [(0.95, 3.15, 0.65), (5.85, 3.15, 0.62)]
# solid plate with TRUE through-holes (symmetric), clear of sources and protects
HOLES = [(3.5, 2.0, 0.60), (2.0, 1.25, 0.50), (5.0, 1.25, 0.50),
         (2.3, 3.0, 0.48), (4.7, 3.0, 0.48)]
RHO_F = 1.6


class _Adapter:
    def __init__(self, mesh):
        self.cx, self.cy, self.N = mesh.cx, mesh.cy, mesh.tris.shape[0]


def _drive(mesh, src):
    sx, sy, sr = src
    xs, ys = mesh.npx[:, 0], mesh.npx[:, 1]
    F = np.zeros(mesh.ndof, complex)
    idx = np.flatnonzero(np.hypot(xs - sx, ys - sy) < sr)
    F[2 * idx] = 1.0; F[2 * idx + 1] = 1.0
    return F


def setup(h, omegas, eta):
    """Conforming holed mesh + one CFRPWavePlate per frequency (all sharing sponge,
    protected weight, static BC) and the list of source drives."""
    mesh = conforming_plate(LX, LY, h, holes=HOLES)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=RHO_F, rho_m=0.45)
    xs, ys = mesh.npx[:, 0], mesh.npx[:, 1]

    drives = [_drive(mesh, s) for s in SOURCES]

    # absorbing sponge on TOP/BOTTOM only -- both side edges reflect, so a wave from
    # either bottom corner sweeps up its side toward the protected top corners.
    m = 0.55
    d = np.maximum.reduce([(m - mesh.cy) / m, (mesh.cy - (LY - m)) / m])
    sponge = np.maximum(0.0, d) ** 2 * 45.0

    # protected weight: union of ALL protected disks
    wpro = np.zeros(xs.shape[0])
    for px, py, pr in PROTECTS:
        wpro = np.maximum(wpro, (np.hypot(xs - px, ys - py) < pr).astype(float))

    left = np.flatnonzero(xs < 1e-6); right = np.flatnonzero(xs > LX - 1e-6)
    Fs = np.zeros(mesh.ndof); Fs[2 * right + 1] = -1.0 / len(right)
    fixed = np.concatenate([2 * left, 2 * left + 1])

    plates = []
    for om in omegas:
        P = CFRPWavePlate(mesh, mat, omega=om, eta=eta)
        P.set_drive(drives[0]); P.set_sponge(sponge); P.set_weight(wpro)
        P.set_static(Fs, fixed)
        plates.append(P)
    interior = (xs > 0.3) & (xs < LX - 0.3) & (ys > 0.3) & (ys < LY - 0.3)
    return mesh, mat, plates, drives, wpro, interior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.07)
    ap.add_argument("--omegas", type=float, nargs="+", default=[9.0, 11.0, 13.0])
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--w-dyn", dest="w_dyn", type=float, default=1.5)
    ap.add_argument("--w-stat", dest="w_stat", type=float, default=0.6)
    ap.add_argument("--rbf-spc", dest="rbf_spc", type=float, default=0.30)
    ap.add_argument("--w-curl", dest="w_curl", type=float, default=0.15)
    ap.add_argument("--curl-ramp", dest="curl_ramp", type=float, default=1.5)
    ap.add_argument("--out", default="results/cfrp_wave_shield")
    a = ap.parse_args()

    mesh, mat, plates, drives, wpro, interior = setup(a.h, a.omegas, a.eta)
    nt = mesh.tris.shape[0]
    A = mesh.area; Asum = A.sum()
    Px, Py = curl_operators(mesh)
    nf = len(plates)

    sx = np.arange(a.rbf_spc / 2, LX, a.rbf_spc)
    sy = np.arange(a.rbf_spc / 2, LY, a.rbf_spc)
    SX, SY = np.meshgrid(sx, sy)
    rbf = CSRBFMapping(_Adapter(mesh), (SX.ravel(), SY.ravel()), r_s=1.6 * a.rbf_spc)
    Mth = rbf.M

    z = np.ones(nt)                            # SOLID plate; mass reduced by the holes
    m_solid = RHO_F * (LX * LY)
    mass = float((A @ (z * RHO_F)) / m_solid)
    E0s = []
    for P in plates:
        e0 = P.dynamic_multi(z, np.zeros(nt), drives, want_grad=False)[0]
        E0s.append(e0 if abs(e0) > 1e-12 else 1.0)
    c0 = plates[0].static(z, np.zeros(nt), False)[0]
    print(f"solid plate, {len(HOLES)} true holes, {len(SOURCES)} sources, "
          f"{len(PROTECTS)} protected corners; {mesh.nnode} nodes, {nt} tris; "
          f"{Mth} supports; omegas={a.omegas}; mass {mass:.3f}x solid", flush=True)

    thh = np.zeros(Mth)
    mma = MMA(np.full(Mth, -np.pi / 2), np.full(Mth, np.pi / 2), move=0.08, adapt=True)
    best = dict(J=np.inf); hist = []
    t0 = time.time()
    for it in range(a.iters):
        th = rbf.Phi @ thh
        frac = min(it / (0.65 * max(a.iters - 1, 1)), 1.0)
        wcurl = a.w_curl * (1.0 + a.curl_ramp * frac)

        Esum = 0.0; dEth_sum = np.zeros(nt); Eks = []
        for P, E0k in zip(plates, E0s):
            Ek, us, dEthk = P.dynamic_multi(z, th, drives)
            Esum += Ek / E0k; dEth_sum += dEthk / E0k; Eks.append(Ek / E0k)
        E_avg = Esum / nf
        c, uc, dcy, dcth = plates[0].static(z, th)
        Pi, dPi = curl_penalty(th, Px, Py, A); Pi /= Asum; dPi /= Asum
        J = a.w_dyn * E_avg + a.w_stat * (c / c0) + wcurl * Pi
        dth = rbf.Phi.T @ (a.w_dyn * dEth_sum / nf + a.w_stat * dcth / c0
                           + wcurl * dPi)

        c_r = c / c0
        hist.append(dict(E=E_avg, c=c_r, Eks=list(Eks), J=J))
        if it >= 0.4 * a.iters and J < best['J']:
            best = dict(J=J, th=th.copy(), E=E_avg, c=c_r, Eks=list(Eks))
        if it % 15 == 0 or it == a.iters - 1:
            zmax = float(director_curl(th, Px, Py).max())
            perf = " ".join(f"{e:.3f}" for e in Eks)
            print(f"[{it:3d}] protE/E0 avg={E_avg:.3f} [{perf}] "
                  f"({1/max(E_avg,1e-9):5.1f}x) c/c0={c_r:.3f} |z|max={zmax:5.2f}",
                  flush=True)
        thh = mma.update(thh, dth, f=J)
    r = best if np.isfinite(best['J']) else dict(th=rbf.Phi @ thh, E=E_avg, c=c_r,
                                                 Eks=list(Eks))
    perf = " ".join(f"{e:.3f}" for e in r['Eks'])
    print(f"optimized in {time.time()-t0:.0f}s: protected E/E0 avg={r['E']:.3f} "
          f"[{perf}] ({1/max(r['E'],1e-9):.1f}x quieter)  c/c0={r['c']:.3f}x  "
          f"mass={mass:.3f}x solid", flush=True)

    np.savez(a.out + ".npz", trix=mesh.npx[:, 0], triy=mesh.npx[:, 1],
             tris=mesh.tris, cx=mesh.cx, cy=mesh.cy, z=z, y=z, th=r['th'],
             sources=np.array(SOURCES), protects=np.array(PROTECTS),
             holes=np.array(HOLES), omegas=np.array(a.omegas),
             E0s=np.array(E0s), E_ratio=r['E'], Eks=np.array(r['Eks']),
             c_ratio=r['c'], vol=1.0, mass=mass,
             hist_E=[h['E'] for h in hist], hist_c=[h['c'] for h in hist],
             eta=a.eta, Lx=LX, Ly=LY, wpro=wpro)
    print("saved", a.out + ".npz", flush=True)


if __name__ == "__main__":
    main()
