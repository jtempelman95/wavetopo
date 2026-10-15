r"""Revision R1 analyses of the lens (reviewer points M1-M3, M5, clarity 3, Figs 3-4).

    power     exact time-averaged power balance and the POWER routed into the target
              (Pi = power dissipated in the target / injected power), straight vs lens,
              over the band; for both the paper lens and the lens re-optimized with the
              better absorber
    fluxmap   flux--fiber angle per cell with low-flux cells excluded (|P| < 5% of max),
              flux-weighted statistics, and the qP-like polarization content
              (kinetic energy polarized along the local fibers)
    fine      fine frequency sweep of the field growth around the design frequency
              (resonance diagnostic: half-power width vs the loss factor)
    absorber  sponge-thickness series (width 0.6, 1.2, 1.8, 2.4): reflection |R| and the
              band-averaged redirection of both lenses
    asbuilt   tow density of the lens field (streamline spreading) and the band
              redirection with a cut/add gap or overlap defect model (N tows per course)

Energy bookkeeping for the e^{+i w t} convention used by the solver:
  injected   P_in   = -(w/2) int f . u_i
  structural P_eta  = (w eta / 2) int [eps(u_r).D eps(u_r) + eps(u_i).D eps(u_i)]
  sponge     P_c    = (w^2 / 2) int c |u|^2
In steady state P_in = P_eta + P_c (checked).  The net time-averaged power flowing into
the target region equals the power dissipated inside it (no sponge there), so
Pi = P_eta,target / P_in is the fraction of the injected power delivered to the target.

    PYTHONPATH=. python examples/analysis/wt_r1_lens.py power|fluxmap|fine|absorber|asbuilt
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
import numpy as np
import ufl
from dolfinx import fem

sys.path.insert(0, _os.path.join(_ROOT, "examples", "dolfinx"))
sys.path.insert(0, _os.path.join(_ROOT, "examples", "analysis"))
import dolfinx_lens as DL
import wt_verify as V
from lens_flux import flux_cells
from wavetopo.dolfinx_elastic import _SUM

RES = _os.path.join(_ROOT, "results")
M_SP = 0.6
LENS = np.load(_os.path.join(RES, "dolfinx_lens_data.npz"))
ABS = np.load(_os.path.join(RES, "lens_absorber.npz"))
BAND = (35.0, 40.0, 45.0, 50.0)


def _int(ew, expr):
    return ew.domain.comm.allreduce(fem.assemble_scalar(fem.form(expr * ufl.dx)), op=_SUM())


def powers(ew):
    U = ew.U
    ur = ufl.as_vector([U[0], U[1]]); ui = ufl.as_vector([U[2], U[3]])
    D = ew.kfac * ew._Dmat(ew.theta)
    er, ei = ew._evoigt(ur), ew._evoigt(ui)
    w, eta = ew.omega, ew.eta
    f = ufl.as_vector([ew.fr[0], ew.fr[1]])
    dens = 0.5 * w * eta * (ufl.dot(er, D * er) + ufl.dot(ei, D * ei))
    Pin = -0.5 * w * _int(ew, ufl.dot(f, ui))
    Peta = _int(ew, dens)
    Pc = 0.5 * w ** 2 * _int(ew, ew.cspg * (ufl.dot(ur, ur) + ufl.dot(ui, ui)))
    Ptgt = _int(ew, ew.wsel * dens)
    return Pin, Peta, Pc, Ptgt


def lens_theta_for(ew, which):
    if which == "paper":
        return V.transfer(LENS["cent"], LENS["thopt"], DL.centroids(ew))
    return V.transfer(ABS["cent"], ABS["th_new"], DL.centroids(ew))


def study_power():
    rows = []
    for which, pad in (("paper", 0.0), ("absorber", 0.6)):
        for om in np.arange(30.0, 60.01, 2.5):
            ew = V.lens_make(omega=om, pad=pad)
            n = ew.theta.x.array.size
            out = [om, 0 if which == "paper" else 1]
            for th in (np.zeros(n), lens_theta_for(ew, which)):
                ew.set_theta(th); ew.solve()
                Pin, Peta, Pc, Pt = powers(ew)
                out += [Pin, Peta, Pc, Pt]
            rows.append(out)
            r = np.array(out)
            bal0 = abs(r[3] + r[4] - r[2]) / r[2]; bal1 = abs(r[7] + r[8] - r[6]) / r[6]
            print(f"[power] {which:8s} w={om:4.1f}: Pi straight {100*r[5]/r[2]:5.2f}%  lens {100*r[9]/r[6]:5.2f}%"
                  f"  ratio {r[9]/r[6]/(r[5]/r[2]):5.2f}  P_in ratio {r[6]/r[2]:5.2f}  balance err {max(bal0, bal1):.1e}",
                  flush=True)
    np.save(_os.path.join(RES, "wt_r1_power.npy"), np.array(rows))


def study_fluxmap():
    out = {}
    for om in BAND:
        ew = DL.make(nx=130, omega=om)
        cent = DL.centroids(ew)
        inner = ((cent[:, 0] > 0.3) & (cent[:, 0] < DL.Lx - M_SP) &
                 (cent[:, 1] > M_SP) & (cent[:, 1] < DL.Ly - M_SP))
        th = V.transfer(LENS["cent"], LENS["thopt"], cent)
        for name, t in (("straight", np.zeros(th.size)), ("lens", th)):
            ew.set_theta(t); ew.solve()
            P = flux_cells(ew)
            mag = np.linalg.norm(P, axis=1)
            dev = np.degrees(np.abs(np.angle(np.exp(2j * (np.arctan2(P[:, 1], P[:, 0]) - t)))) / 2)
            strong = inner & (mag > 0.05 * mag[inner].max())
            w = mag * strong
            # polarization: kinetic energy along the local fiber, per cell (DG0 projection of |u.t|^2, |u|^2)
            U = ew.U
            tf = ufl.as_vector([ufl.cos(ew.theta), ufl.sin(ew.theta)])
            ur = ufl.as_vector([U[0], U[1]]); ui = ufl.as_vector([U[2], U[3]])
            S = ew.S
            a = fem.Function(S); a.interpolate(fem.Expression(ufl.dot(ur, tf) ** 2 + ufl.dot(ui, tf) ** 2, S.element.interpolation_points()))
            b = fem.Function(S); b.interpolate(fem.Expression(ufl.dot(ur, ur) + ufl.dot(ui, ui), S.element.interpolation_points()))
            pol = a.x.array[inner].sum() / b.x.array[inner].sum()
            out[f"{name}_{int(om)}"] = dict(dev_w=float(np.sum(w * dev) / w.sum()),
                                           share20=float(np.sum(w * (dev < 20)) / w.sum()),
                                           p90=float(np.percentile(np.repeat(dev[strong], 1), 90)),
                                           frac_strong=float(strong.sum() / inner.sum()),
                                           dev_all=float(np.sum(mag * inner * dev) / np.sum(mag * inner)),
                                           pol=float(pol))
            print(f"[fluxmap] w={om:4.0f} {name:8s}: |P|-weighted dev {out[f'{name}_{int(om)}']['dev_w']:5.1f} deg "
                  f"(all cells {out[f'{name}_{int(om)}']['dev_all']:5.1f}), within 20 deg {100*out[f'{name}_{int(om)}']['share20']:4.1f}%, "
                  f"strong-flux cells {100*out[f'{name}_{int(om)}']['frac_strong']:4.1f}%, kinetic along fibers {100*pol:4.1f}%",
                  flush=True)
            if om == 40.0:
                np.save(_os.path.join(RES, f"wt_r1_fluxmap_{name}.npy"), np.column_stack([cent, P, t, dev, strong]))
    np.save(_os.path.join(RES, "wt_r1_fluxmap.npy"), out, allow_pickle=True)


def study_fine():
    rows = []
    for om in np.arange(36.0, 44.01, 0.25):
        ew = V.lens_make(omega=om)
        G, Gp, gr = V.lens_pair(ew, lens_theta_for(ew, "paper"))
        rows.append((om, G, Gp, gr))
        print(f"[fine] w={om:6.2f}: G={G:7.2f} G_phi={Gp:5.2f} growth={gr:6.2f}", flush=True)
    np.save(_os.path.join(RES, "wt_r1_fine.npy"), np.array(rows))


def study_absorber():
    rows = []
    for pad in (0.0, 0.6, 1.2, 1.8):
        for om in BAND:
            ew = V.lens_make(omega=om, pad=pad)
            ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve()
            R, res = V.reflection(ew, om, 0.02)
            r = [pad, om, R]
            for which in ("paper", "absorber"):
                G, Gp, gr = V.lens_pair(ew, lens_theta_for(ew, which))
                r += [G, Gp, gr]
            rows.append(r)
            print(f"[absorber] sponge {0.6+pad:.1f} w={om:4.0f}: |R|={R:.3f}  paper lens G_phi={r[4]:5.2f}  "
                  f"absorber lens G_phi={r[7]:5.2f} (growth {r[8]:5.2f})", flush=True)
    np.save(_os.path.join(RES, "wt_r1_absorber.npy"), np.array(rows))


def tow_density(cent, th, Lx, Ly, x0=0.0):
    """tau = exp(-int div n dl) back along the tow path to the inflow edge, on cell centroids."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator
    xs = np.linspace(x0, Lx, 241); ys = np.linspace(0, Ly, 181)
    GX, GY = np.meshgrid(xs, ys)
    v = np.c_[np.cos(2 * th), np.sin(2 * th)]
    lin = LinearNDInterpolator(cent, v)(GX, GY); near = NearestNDInterpolator(cent, v)(GX, GY)
    lin = np.where(np.isnan(lin), near, lin)
    TH = 0.5 * np.arctan2(lin[..., 1], lin[..., 0])
    nx, ny = np.cos(TH), np.sin(TH)
    dy, dx = ys[1] - ys[0], xs[1] - xs[0]
    DIV = np.gradient(nx, dx, axis=1) + np.gradient(ny, dy, axis=0)
    It = RegularGridInterpolator((ys, xs), TH); Id = RegularGridInterpolator((ys, xs), DIV)
    x, y = cent[:, 0].copy(), cent[:, 1].copy()
    acc = np.zeros(x.size); alive = np.ones(x.size, bool); ds = 0.01
    for _ in range(int(3 * Lx / ds)):
        if not alive.any():
            break
        t = It(np.c_[np.clip(y[alive], 0, Ly), np.clip(x[alive], x0, Lx)])
        acc[alive] += ds * Id(np.c_[np.clip(y[alive], 0, Ly), np.clip(x[alive], x0, Lx)])
        x[alive] -= ds * np.cos(t); y[alive] -= ds * np.sin(t)
        alive &= (x > x0) & (x < Lx) & (y > 0) & (y < Ly)
    return np.exp(-acc)


def study_asbuilt():
    ew = DL.make(nx=130, omega=40.0)
    cent = DL.centroids(ew)
    th = V.transfer(LENS["cent"], LENS["thopt"], cent)
    tau = tow_density(cent, th, DL.Lx, DL.Ly)
    inner = (cent[:, 0] < DL.Lx - M_SP) & (cent[:, 1] > M_SP) & (cent[:, 1] < DL.Ly - M_SP)
    print(f"[asbuilt] tau in the interior: {tau[inner].min():.2f}-{tau[inner].max():.2f}; "
          f"|ln tau|>0.1 on {100*np.mean(np.abs(np.log(tau[inner]))>0.1):.0f}% of the interior", flush=True)
    np.save(_os.path.join(RES, "wt_r1_tau.npy"), np.column_stack([cent, tau, th]))
    E_RES, RHO_RES = 3.5, 1.2
    rows = []
    for N in (4, 8, 16):
        fd = 0.5 / N * np.minimum(1.0, np.abs(np.log(tau)) / np.log(1 + 1.0 / N))
        models = {"gap": (1 - fd + fd * E_RES / 131.0, 1 - fd + fd * RHO_RES / 1.6), "ovl": (1 + fd, 1 + fd)}
        for mname, (kf, mf) in models.items():
            r = [N, 0 if mname == "gap" else 1]
            for om in BAND:
                e = DL.make(nx=130, omega=om)
                e.set_theta(np.zeros(th.size)); e.solve()
                E0, I0 = V.lens_metrics(e)
                e.kfac.x.array[:] = kf; e.mfac.x.array[:] = mf
                e.set_theta(th); e.solve(); E1, I1 = V.lens_metrics(e)
                r.append((E1 / I1) / (E0 / I0))
            rows.append(r)
            print(f"[asbuilt] N={N:2d} {mname}: G_phi at {BAND} = " + " ".join(f"{v:5.2f}" for v in r[2:]) +
                  f"  band mean {np.mean(r[2:]):.2f}", flush=True)
    np.save(_os.path.join(RES, "wt_r1_asbuilt.npy"), np.array(rows))


if __name__ == "__main__":
    for w in sys.argv[1:]:
        {"power": study_power, "fluxmap": study_fluxmap, "fine": study_fine,
         "absorber": study_absorber, "asbuilt": study_asbuilt}[w]()
