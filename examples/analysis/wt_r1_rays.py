r"""Ray (WKB) prediction of the redirection from the fiber field alone (Reviewer R1, M1).

In a slowly varying fiber field theta(x), qP energy travels along rays of the Hamiltonian
omega(x, k) = |k| c(psi - theta(x)), psi the wave-normal angle, c the qP phase speed of the
lamina at relative angle psi - theta (Christoffel problem, fibers along x).  Hamilton's
equations, with omega conserved, reduce to

    dx/dt   = c n + c' n_perp                      (group velocity, eq. skew of the paper)
    dpsi/dt = c'(psi - theta) (n_perp . grad theta)

The drive launches plane-wave rays (psi = 0) uniformly across the interior part of the left
edge (the drive outside it lies in the absorbing layer).  Rays that
enter the absorbing layer are absorbed (no reflections, single pass).  In steady state the
energy carried by a ray is proportional to the time it spends in a region, so the target's
share of the interior energy is the target's share of the ray dwell time.  The ray estimate
of the redirection is G_ray = share(lens) / share(straight fibers).  It contains no
interference, no reflections and no resonance, and does not depend on frequency.

    python examples/analysis/wt_r1_rays.py      -> results/wt_r1_rays.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator

sys.path.insert(0, _os.path.join(_ROOT, "examples", "analysis"))
from lamina import lamina_Q

RES = _os.path.join(_ROOT, "results")
LX, LY, M_SP = 4.0, 3.0, 0.6
FX, FY, FR = 0.78 * LX, LY / 2, 0.18
Q = lamina_Q() / 1e9; RHO = 1.6


def qp_speed(psi):
    c, s = np.cos(psi), np.sin(psi)
    G11 = Q[0, 0] * c ** 2 + Q[2, 2] * s ** 2
    G22 = Q[2, 2] * c ** 2 + Q[1, 1] * s ** 2
    G12 = (Q[0, 1] + Q[2, 2]) * c * s
    tr, det = G11 + G22, G11 * G22 - G12 ** 2
    return np.sqrt((0.5 * tr + np.sqrt(0.25 * tr ** 2 - det)) / RHO)


PS = np.linspace(-np.pi, np.pi, 4001)
CP = qp_speed(PS)
DCP = np.gradient(CP, PS)
c_of = lambda p: np.interp(np.angle(np.exp(1j * p)), PS, CP)
dc_of = lambda p: np.interp(np.angle(np.exp(1j * p)), PS, DCP)


def field_on_grid(cent, th):
    xs = np.linspace(0, LX, 401); ys = np.linspace(0, LY, 301)
    GX, GY = np.meshgrid(xs, ys)
    v = np.c_[np.cos(2 * th), np.sin(2 * th)]
    lin = LinearNDInterpolator(cent, v)(GX, GY); near = NearestNDInterpolator(cent, v)(GX, GY)
    lin = np.where(np.isnan(lin), near, lin)
    TH = 0.5 * np.arctan2(lin[..., 1], lin[..., 0])
    TX = np.gradient(TH, xs[1] - xs[0], axis=1); TY = np.gradient(TH, ys[1] - ys[0], axis=0)
    mk = lambda F: RegularGridInterpolator((ys, xs), F, bounds_error=False, fill_value=0.0)
    return mk(TH), mk(TX), mk(TY)


def trace(It, Ix, Iy, nray=2000, dt=2e-3, tmax=4.0, keep=0):
    y0 = np.linspace(M_SP + 0.002, LY - M_SP - 0.002, nray)
    x = np.full(nray, 0.1); y = y0.copy(); psi = np.zeros(nray)
    alive = np.ones(nray, bool)
    t_in = np.zeros(nray); t_tg = np.zeros(nray)
    paths = [[] for _ in range(keep)]
    sel = np.linspace(0, nray - 1, keep).astype(int) if keep else []
    for step in range(int(tmax / dt)):
        if not alive.any():
            break
        pts = np.c_[y, x]
        th, tx, ty = It(pts), Ix(pts), Iy(pts)
        rel = psi - th
        c, dc = c_of(rel), dc_of(rel)
        n = np.c_[np.cos(psi), np.sin(psi)]; npp = np.c_[-np.sin(psi), np.cos(psi)]
        vx = c * n[:, 0] + dc * npp[:, 0]; vy = c * n[:, 1] + dc * npp[:, 1]
        dpsi = dc * (npp[:, 0] * tx + npp[:, 1] * ty)
        x = np.where(alive, x + dt * vx, x); y = np.where(alive, y + dt * vy, y)
        psi = np.where(alive, psi + dt * dpsi, psi)
        interior = (x < LX - M_SP) & (y > M_SP) & (y < LY - M_SP) & (x > 0)
        t_in += dt * (alive & interior)
        t_tg += dt * (alive & (np.abs(x - FX) < FR) & (np.abs(y - FY) < FR))
        alive &= (x > 0) & (x < LX - M_SP) & (y > M_SP) & (y < LY - M_SP)     # absorbed in the sponge
        for j, k in enumerate(sel):
            if alive[k]:
                paths[j].append((x[k], y[k]))
    return t_tg.sum() / t_in.sum(), [np.array(p) for p in paths]


def main():
    out = {}
    L = np.load(_os.path.join(RES, "dolfinx_lens_data.npz"))
    A = np.load(_os.path.join(RES, "lens_absorber.npz"))
    straight = field_on_grid(L["cent"], np.zeros(L["thopt"].size))
    s0, p0 = trace(*straight, keep=24)
    for name, cent, th in (("paper", L["cent"], L["thopt"]), ("absorber", A["cent"], A["th_new"])):
        s1, p1 = trace(*field_on_grid(cent, th), keep=24)
        out[f"{name}_G_ray"] = s1 / s0
        out[f"{name}_share"] = s1
        out[f"{name}_paths"] = np.array(p1, dtype=object)
        print(f"{name} lens: ray target share {100*s1:.2f}% vs straight {100*s0:.2f}%  ->  G_ray = {s1/s0:.2f}", flush=True)
    out["straight_share"] = s0
    np.savez(_os.path.join(RES, "wt_r1_rays.npz"), **out, allow_pickle=True)


if __name__ == "__main__":
    main()
