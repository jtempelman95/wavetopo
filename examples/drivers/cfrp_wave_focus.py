r"""
Co-optimized topology + toolpath on a conforming CFRP mesh -- the PHYSICALLY
COHERENT version.  A cantilever bracket with two prescribed through-holes is

  * TOPOLOGY (density z): compliance-minimized at a volume budget -> a real
    optimized structure that is LIGHTER (mass reduction), holes cut from the mesh;
  * TOOLPATH (orientation theta): aligned with the load for stiffness AND steering
    an injected elastic wave to FOCUS at the loaded tip.

The wave propagates THROUGH the fiber structure and concentrates at the target, so
the material (topology) and the field (wave) occupy the SAME region -- unlike a
guided channel, where the wave must live in a soft channel disjoint from the stiff
frame.  Wave control here is FOCUSING (bend the rays), which needs no soft channel;
it is compatible with a solid load-bearing structure.

  z_hat  -> cone filter -> tanh project (robust erode/dilate)  z    : TOPOLOGY
  theta_hat -> CS-RBF                                          theta : TOOLPATH

  DYNAMIC  maximize focus energy   E = u^H W u,  W = +1 in the tip focus disk
  STATIC   minimize cantilever compliance c  (clamped left, tip load right)
  VOLUME   (A.y)/A = V*  (the mass budget)          + fiber-curvature (curl) limit

    PYTHONPATH=. python examples/cfrp_wave_focus.py
"""
import argparse
import time

import numpy as np

from wavetopo.cfrp_wave import (conforming_plate, CFRPWavePlate, cone_filter,
                                curl_operators, curl_penalty, director_curl,
                                Material)
from wavetopo.cfrp import CSRBFMapping
from wavetopo.cfrp_optimizer import MMA


LX, LY = 6.0, 3.0
HOLES = [(2.1, 2.00, 0.38), (3.9, 1.05, 0.42)]     # two asymmetric through-holes
FOCUS = (5.35, 1.5, 0.42)                           # tip focus disk (target)
RHO_F, RHO_M = 1.6, 0.45                            # fiber vs light-matrix density


class _Adapter:
    def __init__(self, mesh):
        self.cx, self.cy, self.N = mesh.cx, mesh.cy, mesh.tris.shape[0]


def project(y, beta, eta=0.5):
    num = np.tanh(beta * eta) + np.tanh(beta * (y - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / den


def dproject(y, beta, eta=0.5):
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta * (1 - np.tanh(beta * (y - eta)) ** 2) / den


def setup(h, omega, eta):
    mesh = conforming_plate(LX, LY, h, holes=HOLES)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=RHO_F, rho_m=RHO_M)
    P = CFRPWavePlate(mesh, mat, omega=omega, eta=eta)
    xs, ys = mesh.npx[:, 0], mesh.npx[:, 1]

    # localized aperture beam injected at the left, mid-height
    F = np.zeros(mesh.ndof, complex)
    mouth = (xs < 1e-6) & (np.abs(ys - LY / 2) < 0.5)
    F[2 * np.flatnonzero(mouth)] = 1.0
    P.set_drive(F)

    # absorbing sponge on the top/bottom/right margins (not the tip focus band)
    marg = 0.45
    d = np.maximum.reduce([(mesh.cx - (LX - marg)) / marg, (marg - mesh.cy) / marg,
                           (mesh.cy - (LY - marg)) / marg])
    P.set_sponge(np.maximum(0.0, d) ** 2 * 45.0)

    # wave weight: +1 in the tip focus disk (maximize focus energy), like the lens
    fx, fy, fr = FOCUS
    wfoc = (np.hypot(mesh.npx[:, 0] - fx, mesh.npx[:, 1] - fy) < fr).astype(float)
    P.set_weight(wfoc)

    # cantilever: clamp the left edge, transverse (downward) tip load on the right
    left = np.flatnonzero(xs < 1e-6)
    right = np.flatnonzero(xs > LX - 1e-6)
    Fs = np.zeros(mesh.ndof); Fs[2 * right + 1] = -1.0 / len(right)
    P.set_static(Fs, np.concatenate([2 * left, 2 * left + 1]))
    interior = (xs > 0.3) & (xs < LX - 0.2) & (ys > 0.3) & (ys < LY - 0.3)
    return mesh, mat, P, wfoc, interior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.06)
    ap.add_argument("--omega", type=float, default=12.0)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=220)
    ap.add_argument("--vol", type=float, default=0.5)
    ap.add_argument("--w-dyn", dest="w_dyn", type=float, default=1.5)
    ap.add_argument("--w-stat", dest="w_stat", type=float, default=1.0)
    ap.add_argument("--w-stat-theta", dest="w_stat_theta", type=float, default=0.6)
    ap.add_argument("--rmin", type=float, default=0.13)
    ap.add_argument("--rbf-spc", dest="rbf_spc", type=float, default=0.30)
    ap.add_argument("--cvol", type=float, default=400.0)
    ap.add_argument("--betamax", type=float, default=48.0)
    ap.add_argument("--deta", type=float, default=0.15)
    ap.add_argument("--w-grey", dest="w_grey", type=float, default=1.0)
    ap.add_argument("--w-curl", dest="w_curl", type=float, default=0.4)
    ap.add_argument("--out", default="results/cfrp_wave_focus")
    a = ap.parse_args()

    mesh, mat, P, wfoc, interior = setup(a.h, a.omega, a.eta)
    nt = mesh.tris.shape[0]
    A = mesh.area; Asum = A.sum()
    Filt = cone_filter(mesh, a.rmin)
    Px, Py = curl_operators(mesh)

    sx = np.arange(a.rbf_spc / 2, LX, a.rbf_spc)
    sy = np.arange(a.rbf_spc / 2, LY, a.rbf_spc)
    SX, SY = np.meshgrid(sx, sy)
    rbf = CSRBFMapping(_Adapter(mesh), (SX.ravel(), SY.ravel()), r_s=1.6 * a.rbf_spc)
    Mth = rbf.M
    print(f"conforming plate: {mesh.nnode} nodes, {nt} tris, 2 holes; "
          f"{Mth} orientation supports; omega={a.omega}; NO prescribed regions "
          f"(free topology)", flush=True)

    z = np.full(nt, a.vol); thh = np.zeros(Mth)

    # baseline: uniform density at the volume budget, straight fibers
    y0 = project(Filt @ z, 1.0)
    u0 = P.dynamic(y0, np.zeros(nt), False)[1]
    E0 = float(np.real(np.vdot(u0, P.W @ u0)))
    E0 = E0 if abs(E0) > 1e-12 else 1.0
    c0 = P.static(y0, np.zeros(nt), False)[0]
    # solid-fiber reference mass (full rectangle) for the mass-reduction number
    m_solid = RHO_F * (LX * LY)
    print(f"baseline: focus energy={E0:.3e}  compliance={c0:.3e}", flush=True)

    etas = [0.5 + a.deta, 0.5, 0.5 - a.deta]
    rho_ks = 12.0
    stage = max(a.iters // 8, 1)
    mma = MMA(np.concatenate([np.zeros(nt), np.full(Mth, -np.pi / 2)]),
              np.concatenate([np.ones(nt), np.full(Mth, np.pi / 2)]),
              move=0.06, adapt=True)
    best = dict(J=np.inf); hist = []
    t0 = time.time()
    for it in range(a.iters):
        beta = min(a.betamax, 2.0 ** (it // stage))
        yf = Filt @ z
        th = rbf.Phi @ thh

        # z <- COMPLIANCE, robust over erode/dilate (connected + crisp topology)
        cs, dcz_parts = [], []
        for e in etas:
            y = project(yf, beta, e); dpr = dproject(yf, beta, e)
            c, uc, dcy, dcth = P.static(y, th)
            cs.append(c / c0); dcz_parts.append(dpr * (dcy / c0))
        cs = np.array(cs)
        mM = cs.max(); wk = np.exp(rho_ks * (cs - mM)); wk /= wk.sum()
        Crob = float(mM + np.log(np.sum(np.exp(rho_ks * (cs - mM)))) / rho_ks)

        # intermediate field: focus (drives theta), compliance-on-theta, vol, curl
        y_i = project(yf, beta, 0.5); dpr_i = dproject(yf, beta, 0.5)
        E, uE, dEy, dEth = P.dynamic(y_i, th)
        c_i, uc_i, dcy_i, dcth_i = P.static(y_i, th)
        vol = float((A @ y_i) / Asum)
        grey = float((A @ (y_i * (1 - y_i))) / Asum)
        Mnd = float((A @ (4 * y_i * (1 - y_i))) / Asum)
        Pi_curl, dPi_curl = curl_penalty(th, Px, Py, A); Pi_curl /= Asum; dPi_curl /= Asum
        J = (a.w_stat * Crob - a.w_dyn * (E / E0)
             + a.cvol * (vol - a.vol) ** 2 + a.w_grey * grey
             + a.w_curl * Pi_curl + a.w_stat_theta * (c_i / c0))

        # z gradient: compliance + volume + grey (no wave -> connected topology)
        dz = Filt.T @ (a.w_stat * sum(wk[k] * dcz_parts[k] for k in range(3))
                       + dpr_i * (a.cvol * 2 * (vol - a.vol) * (A / Asum)
                                  + a.w_grey * (1 - 2 * y_i) * (A / Asum)))
        # theta gradient: FOCUS (wave) + stiffness + curl -- all compatible, since
        # the wave lives in the same fiber the load flows through.
        dth = rbf.Phi.T @ (-a.w_dyn * dEth / E0 + a.w_curl * dPi_curl
                           + a.w_stat_theta * dcth_i / c0)

        E_i, c_r = E / E0, c_i / c0
        mass = float((A @ (y_i * RHO_F + (1 - y_i) * RHO_M)) / m_solid)
        hist.append(dict(E=E_i, c=c_r, vol=vol, Mnd=Mnd, mass=mass, J=J))
        if abs(vol - a.vol) < 0.05 and beta >= a.betamax * 0.9 and Mnd < 0.1 \
                and J < best['J']:
            best = dict(J=J, z=z.copy(), y=y_i.copy(), th=th.copy(),
                        E=E_i, c=c_r, vol=vol, Mnd=Mnd, mass=mass)
        if it % 15 == 0 or it == a.iters - 1:
            zmax = float(director_curl(th, Px, Py)[y_i > 0.5].max())
            print(f"[{it:3d}] beta={beta:5.1f} focus={E_i:6.2f}x c/c0={c_r:6.3f} "
                  f"vol={vol:.3f} mass={mass:.3f} Mnd={Mnd:.3f} |z|max={zmax:5.2f}",
                  flush=True)
        W = mma.update(np.concatenate([z, thh]), np.concatenate([dz, dth]), f=J)
        z, thh = W[:nt], W[nt:]
    r = best if np.isfinite(best['J']) else dict(
        z=z, y=project(Filt @ z, a.betamax, 0.5), th=rbf.Phi @ thh,
        E=E_i, c=c_r, vol=vol, Mnd=Mnd, mass=mass)
    print(f"optimized in {time.time()-t0:.0f}s: focus={r['E']:.1f}x  "
          f"c/c0={r['c']:.3f}x  vol={r['vol']:.3f}  mass={r['mass']:.3f}x solid  "
          f"Mnd={r['Mnd']:.3f}", flush=True)

    np.savez(a.out + ".npz", trix=mesh.npx[:, 0], triy=mesh.npx[:, 1],
             tris=mesh.tris, cx=mesh.cx, cy=mesh.cy, z=r['z'], y=r['y'],
             th=r['th'], holes=np.array(HOLES), focus=np.array(FOCUS),
             E_gain=r['E'], c_ratio=r['c'], vol=r['vol'], mass=r['mass'],
             hist_E=[hh['E'] for hh in hist], hist_c=[hh['c'] for hh in hist],
             hist_mass=[hh['mass'] for hh in hist],
             omega=a.omega, eta=a.eta, Lx=LX, Ly=LY, wsigned=wfoc)
    print("saved", a.out + ".npz", flush=True)


if __name__ == "__main__":
    main()
