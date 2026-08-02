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
                                curl_operators, curl_penalty, director_curl,
                                Material)
from wavetopo.cfrp import CSRBFMapping
from wavetopo.cfrp_optimizer import MMA


LX, LY = 6.0, 3.0
# Two asymmetric through-holes, one high-left, one low-right, that the guiding
# channel threads BETWEEN.  The channel is a single soft-matrix lane that enters at
# the clamped left edge (both banks grounded, so the entry slit cannot disconnect
# the frame) and TERMINATES in an interior output pocket -- so the stiff-fiber frame
# closes around the channel end (x > OUTPUT_x stays solid) instead of being split
# into two disjoint top/bottom rows by a channel that reaches the free right edge.
HOLES = [(1.7, 2.25, 0.36), (3.7, 0.75, 0.44)]
CORRIDOR = np.array([(0.25, 1.5), (1.7, 1.35), (3.7, 1.70), (5.05, 1.5)])
OUTPUT = (5.05, 1.5, 0.40)                              # interior delivery pocket
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


def corridor_tangent(pts, poly):
    """Angle of the corridor's local tangent at each point (from the nearest
    polyline segment). theta = this + pi/2 lays fibers ACROSS the corridor, whose
    slow (E_f2) axis then lies along the propagation direction -> a slow guiding
    channel in solid fiber, no void on the wave path."""
    best_d = np.full(pts.shape[0], np.inf)
    best_ang = np.zeros(pts.shape[0])
    for k in range(len(poly) - 1):
        p0, seg = poly[k], poly[k + 1] - poly[k]
        t = np.clip((pts - p0) @ seg / (seg @ seg), 0, 1)
        proj = p0 + t[:, None] * seg
        d = np.linalg.norm(pts - proj, axis=1)
        m = d < best_d
        best_d[m] = d[m]; best_ang[m] = np.arctan2(seg[1], seg[0])
    return best_ang


def project(y, beta, eta=0.5):
    num = np.tanh(beta * eta) + np.tanh(beta * (y - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / den


def dproject(y, beta, eta=0.5):
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta * (1 - np.tanh(beta * (y - eta)) ** 2) / den


def setup(h, omega, eta, w_leak):
    mesh = conforming_plate(LX, LY, h, holes=HOLES)
    # Light matrix (rho_m << rho_f): the soft-matrix guiding channel is therefore a
    # genuine MASS SAVING, not just a wave feature -- the channel does double duty.
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=0.45)
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
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=160)
    ap.add_argument("--vol", type=float, default=0.6)
    ap.add_argument("--w-dyn", dest="w_dyn", type=float, default=1.0)
    ap.add_argument("--w-stat", dest="w_stat", type=float, default=0.6)
    ap.add_argument("--w-leak", dest="w_leak", type=float, default=0.5)
    ap.add_argument("--rmin", type=float, default=0.11)
    ap.add_argument("--rbf-spc", dest="rbf_spc", type=float, default=0.30)
    ap.add_argument("--cvol", type=float, default=200.0)
    ap.add_argument("--betamax", type=float, default=32.0)
    ap.add_argument("--deta", type=float, default=0.12,
                    help="robust erode/dilate threshold offset (0 disables)")
    ap.add_argument("--w-grey", dest="w_grey", type=float, default=0.5,
                    help="penalty on z(1-z) to push the topology binary")
    ap.add_argument("--w-curl", dest="w_curl", type=float, default=0.3,
                    help="fiber-curvature (Dirichlet) penalty on theta for a "
                         "printable turning radius")
    ap.add_argument("--curl-ramp", dest="curl_ramp", type=float, default=6.0,
                    help="continuation: w_curl grows to (1+curl_ramp)x by the last "
                         "iteration, driving |zeta|max under the print bound")
    ap.add_argument("--w-stat-theta", dest="w_stat_theta", type=float, default=0.5,
                    help="also orient theta for stiffness (co-design), so a "
                         "stronger curl penalty need not cost load alignment")
    ap.add_argument("--out", default="results/cfrp_wave_guide")
    a = ap.parse_args()

    mesh, mat, P, wsigned, interior = setup(a.h, a.omega, a.eta, a.w_leak)
    nt = mesh.tris.shape[0]
    Filt = cone_filter(mesh, a.rmin)
    Px, Py = curl_operators(mesh)          # director-curvature (curl) operators

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

    # STEP-INDEX waveguide via a MATERIAL index step (the manufacturable one).  An
    # elastic wave is guided by a sharp slow/fast contrast.  Building that contrast
    # from fiber ORIENTATION (across->along) needs a ~90 deg fiber turn over a print
    # cell -- unbuildable, and the printable smooth version guides only ~0.45 (below
    # the 0.52 straight-fiber baseline).  Building it from MATERIAL -- a soft-matrix
    # core (just: do not lay fiber there) inside a stiff-fiber cladding -- is trivial
    # to manufacture and guides at ~0.85.  So the guiding "channel" is soft matrix
    # (present material, E_m), not void.  A fiber RING also wraps each hole so tows
    # thread fully around the through-holes.  z is driven by COMPLIANCE alone
    # (connected by construction); theta carries the wave and the far-field stiffness.
    cxy = np.column_stack([mesh.cx, mesh.cy])
    dcor_e = corridor_coords(cxy, CORRIDOR)[0]
    dhole_e = np.full(nt, np.inf)
    for hx, hy, r in HOLES:
        dhole_e = np.minimum(dhole_e, np.abs(np.hypot(mesh.cx - hx, mesh.cy - hy) - r))
    core = dcor_e < TUBE_HALF * 0.65                       # soft-matrix guiding lane
    ring = dhole_e < 0.28                                  # fiber wrap around holes
    passive = core | ring
    pval = np.where(core, 0.0, 1.0)                        # core=matrix, ring=fiber
    # Only the thin matrix channel (the wave path, a physical necessity -- the wave
    # localizes in the SLOW phase) and a thin fiber wrap around each hole are
    # prescribed.  Everything else -- including the fast fiber cladding that confines
    # the wave -- is left to the COMPLIANCE topology optimization, which naturally
    # lays fiber around the channel and holes to carry the bending load.

    def apply_passive(y):
        yp = np.where(passive, pval, y)
        return yp, (~passive).astype(float)               # y_phys, active

    # baseline = uniform half-density, STRAIGHT fibers (theta=0), so E/E0 and c/c0
    # measure improvement over an unguided panel.
    y0 = apply_passive(project(Filt @ z, 1.0))[0]
    u0 = P.dynamic(y0, np.zeros(nt), False)[1]
    Enorm = float((np.abs(u0[0::2])**2 + np.abs(u0[1::2])**2)[interior].sum())
    E0 = Enorm if Enorm > 1e-12 else 1.0
    c0 = P.static(y0, np.zeros(nt), False)[0]

    print(f"baseline: interior energy norm={E0:.3e}  compliance={c0:.3e}; "
          f"prescribed: matrix channel {int(core.sum())} elts, hole rings "
          f"{int(ring.sum())} elts; free (topology-optimized): "
          f"{int((~passive).sum())} elts", flush=True)

    # Robust erode/dilate (Wang, Lazarov & Sigmund 2011) on the STRUCTURAL
    # (compliance) objective, which drives z: the worst case over eroded /
    # intermediate / dilated projections forces crisp, length-scale-controlled,
    # connected members.  The wave routes through theta on the intermediate field.
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

        # z <- COMPLIANCE, robust over erode/dilate (connected + crisp)
        cs, dcz_parts = [], []
        for eta in etas:
            y = project(yf, beta, eta); dpr = dproject(yf, beta, eta)
            yp, act = apply_passive(y)
            c, uc, dcy, dcth = P.static(yp, th)
            cs.append(c / c0)
            dcz_parts.append(dpr * act * (dcy / c0))
        cs = np.array(cs)
        mM = cs.max(); wk = np.exp(rho_ks * (cs - mM)); wk /= wk.sum()
        Crob = float(mM + np.log(np.sum(np.exp(rho_ks * (cs - mM)))) / rho_ks)

        # intermediate field: wave objective (drives theta), volume, grey, metrics
        y_i = project(yf, beta, 0.5); dpr_i = dproject(yf, beta, 0.5)
        yp_i, act_i = apply_passive(y_i)
        E, uE, dEy, dEth = P.dynamic(yp_i, th)
        c_i, uc_i, dcy_i, dcth_i = P.static(yp_i, th)
        vol = float((A @ yp_i) / Asum)
        grey = float((A @ (y_i * (1 - y_i))) / Asum)
        Mnd = float((A @ (4 * y_i * (1 - y_i))) / Asum)
        # mass vs a full solid-fiber rectangle: holes (no cells) + the light-matrix
        # channel + any matrix in the free region are ALL counted as savings.
        mass = float((A @ (yp_i * mat.rho_f + (1 - yp_i) * mat.rho_m))
                     / (mat.rho_f * LX * LY))
        Pi_curl, dPi_curl = curl_penalty(th, Px, Py, A)
        Pi_curl /= Asum; dPi_curl /= Asum
        # curvature continuation: ramp the curl weight up over the FIRST 65% of the
        # run so the fiber turning radius is driven UNDER the printable bound, then
        # HOLD it fixed so the topology can settle to a crisp binary at a stationary
        # objective (a ramp that never plateaus keeps the density oscillating).
        frac = min(it / (0.65 * max(a.iters - 1, 1)), 1.0)
        wcurl = a.w_curl * (1.0 + a.curl_ramp * frac)
        J = (a.w_stat * Crob - a.w_dyn * (E / E0)
             + a.cvol * (vol - a.vol) ** 2 + a.w_grey * grey
             + wcurl * Pi_curl + a.w_stat_theta * (c_i / c0))

        # z gradient: robust compliance + volume + grey (NO wave -> no islands)
        dz = Filt.T @ (a.w_stat * sum(wk[k] * dcz_parts[k] for k in range(3))
                       + dpr_i * (a.cvol * 2 * (vol - a.vol) * (A / Asum)
                                  + a.w_grey * (1 - 2 * y_i) * (A / Asum)))
        # theta gradient: WAVE routing only.  Each design field owns one
        # objective -- z<->compliance (connected topology), theta<->wave -- coupled
        # only through the shared K(z,theta).  Feeding compliance into theta orients
        # fibers for stiffness and actively de-confines the wave.
        dth = rbf.Phi.T @ (-a.w_dyn * dEth / E0 + wcurl * dPi_curl
                           + a.w_stat_theta * dcth_i / c0)

        E_i, c_i = E / E0, c_i / c0
        hist.append(dict(E=E_i, c=c_i, vol=vol, Mnd=Mnd, mass=mass, J=J))
        # only accept a near-binary, connected iterate at the volume budget as best
        if abs(vol - a.vol) < 0.05 and beta >= a.betamax * 0.9 and Mnd < 0.12 \
                and J < best['J']:
            best = dict(J=J, z=z.copy(), thh=thh.copy(), y=yp_i.copy(),
                        th=th.copy(), E=E_i, c=c_i, vol=vol, Mnd=Mnd, mass=mass)
        if it % 15 == 0 or it == a.iters - 1:
            zmax = float(director_curl(th, Px, Py)[yp_i > 0.5].max())
            print(f"[{it:3d}] beta={beta:5.1f} E/E0={E_i:7.2f} c/c0={c_i:6.3f} "
                  f"vol={vol:.3f} mass={mass:.3f} Mnd={Mnd:.3f} |z|max={zmax:5.2f} "
                  f"J={J:+.3f}", flush=True)
        W = mma.update(np.concatenate([z, thh]), np.concatenate([dz, dth]), f=J)
        z, thh = W[:nt], W[nt:]
    r = best if np.isfinite(best['J']) else dict(
        z=z, y=apply_passive(project(Filt @ z, a.betamax, 0.5))[0],
        th=rbf.Phi @ thh, E=E_i, c=c_i, vol=vol, Mnd=Mnd, mass=mass)
    print(f"optimized in {time.time()-t0:.0f}s: E/E0={r['E']:.1f}x  "
          f"c/c0={r['c']:.3f}x  vol={r['vol']:.3f}  mass={r['mass']:.3f}x solid  "
          f"Mnd={r['Mnd']:.3f} (0=binary)", flush=True)

    np.savez(a.out + ".npz", trix=mesh.npx[:, 0], triy=mesh.npx[:, 1],
             tris=mesh.tris, cx=mesh.cx, cy=mesh.cy, z=r['z'], y=r['y'],
             th=r['th'], holes=np.array(HOLES), corridor=CORRIDOR,
             output=np.array(OUTPUT), E_gain=r['E'], c_ratio=r['c'],
             vol=r['vol'], mass=r['mass'],
             hist_E=[hh['E'] for hh in hist], hist_c=[hh['c'] for hh in hist],
             hist_mass=[hh['mass'] for hh in hist],
             omega=a.omega, eta=a.eta, Lx=LX, Ly=LY, wsigned=wsigned)
    print("saved", a.out + ".npz", flush=True)


if __name__ == "__main__":
    main()
