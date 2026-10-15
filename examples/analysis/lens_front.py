r"""Manufacturability front of the lens on the SAME mesh and settings as the main lens
(nx = 130, omega = 40, symmetric CS-RBF h_s = 0.33, 70 MMA iterations), scored by
the robust redirection metric rather than the resonant focus gain.

For each tow-curvature limit zeta_all (inf = unconstrained) we optimize the focus
energy (continuation penalty on |zeta|, best feasible iterate, dolfinx_lens.
optimize_curl) and report G(40), G_phi(40) and the band-averaged G_phi over
omega in {35, 40, 45, 50}.  The minimum tow radius is R = 1/zeta_all; the quasi-P
wavelength along the fibre at omega = 40 is lambda = 1.42.

    PYTHONPATH=. .../dolfinx_complex/bin/python examples/analysis/lens_front.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
import time
import numpy as np

sys.path.insert(0, (_ROOT + "/examples/dolfinx"))
sys.path.insert(0, (_ROOT + "/examples/analysis"))
import dolfinx_lens as DL
from wavetopo.dolfinx_wave import csrbf_operators
from lens_mechanism import metrics


def evaluate(th):
    res = {}
    for om in (35.0, 40.0, 45.0, 50.0):
        ew = DL.make(nx=130, omega=om)
        ew.set_theta(np.zeros(th.size)); ew.solve(); E0, I0, _ = metrics(ew)
        ew.set_theta(th); ew.solve(); E1, I1, _ = metrics(ew)
        res[om] = (E1 / E0, (E1 / I1) / (E0 / I0))
    return res


def main():
    ew = DL.make(nx=130)
    cent = DL.centroids(ew)
    supp, mir = DL.symmetric_support(0.33)
    B, Bx, By = csrbf_operators(cent, supp, 0.7)
    sym = lambda v: 0.5 * (v - v[mir])
    rows = []
    for za in (None, 4.0, 2.5, 1.5, 1.0, 0.6):
        t0 = time.time()
        x, th, _, _, _ = DL.optimize_curl(ew, B, Bx, By, sym, 70, zeta_all=za, mu_max=8.0)
        zmax = np.abs(np.cos(B @ x) * (Bx @ x) + np.sin(B @ x) * (By @ x)).max()
        r = evaluate(th)
        band = np.mean([r[o][1] for o in r])
        rows.append((np.inf if za is None else za, zmax, r[40.0][0], r[40.0][1], band))
        print(f"zeta_all={'inf' if za is None else za:>4}: max|zeta|={zmax:5.2f}  G(40)={r[40.0][0]:7.1f}  "
              f"G_phi(40)={r[40.0][1]:5.2f}  <G_phi>_band={band:5.2f}  [{time.time()-t0:.0f}s]", flush=True)
        np.save(f"results/lens_front_theta_{'inf' if za is None else za}.npy", th)
    np.save("results/lens_front.npy", np.array(rows))


if __name__ == "__main__" and "tight" not in sys.argv:
    main()


def optimise_warm(ew, B, Bx, By, sym, x_unc, za, iters=70, mu=8.0):
    """Tight limits: start from the unconstrained design scaled until FEASIBLE
    (max|zeta| <= 0.95 zeta_all), then MMA on -log E + mu * penalty, tracking the
    best feasible iterate (so the result is never worse than the feasible start)."""
    from wavetopo.cfrp_optimizer import MMA
    zfun = lambda x: np.abs(np.cos(B @ x) * (Bx @ x) + np.sin(B @ x) * (By @ x)).max()
    lo, hi = 0.0, 1.0
    for _ in range(40):
        a = 0.5 * (lo + hi)
        lo, hi = (a, hi) if zfun(a * x_unc) <= 0.95 * za else (lo, a)
    x = lo * x_unc
    mma = MMA(np.full(x.size, -np.pi / 2), np.full(x.size, np.pi / 2), move=0.05)
    best = (-np.inf, x.copy())
    for it in range(iters):
        th = B @ x; ew.set_theta(th); ew.solve(); E = ew.focus_energy()
        if zfun(x) <= 1.0 * za and E > best[0]:
            best = (E, x.copy())
        g = -sym(B.T @ ew.focus_grad()) / max(E, 1e-30)
        _, dP = DL.curl_penalty(x, B, Bx, By, za)
        x = sym(mma.update(x, g + mu * sym(dP)))
    return best[1], lo


def main_tight():
    ew = DL.make(nx=130)
    cent = DL.centroids(ew)
    supp, mir = DL.symmetric_support(0.33)
    B, Bx, By = csrbf_operators(cent, supp, 0.7)
    sym = lambda v: 0.5 * (v - v[mir])
    th_unc = np.load("results/lens_front_theta_inf.npy")
    from scipy.sparse.linalg import lsqr
    x_unc = sym(lsqr(B, th_unc)[0])
    rows = list(np.load("results/lens_front.npy"))
    rows = [r for r in rows if r[0] > 1.6 or not np.isfinite(r[0])]
    for za in (1.5, 1.0, 0.6, 0.3):
        t0 = time.time()
        x, a = optimise_warm(ew, B, Bx, By, sym, x_unc, za)
        th = B @ x
        zmax = np.abs(np.cos(th) * (Bx @ x) + np.sin(th) * (By @ x)).max()
        r = evaluate(th)
        band = np.mean([r[o][1] for o in r])
        rows.append((za, zmax, r[40.0][0], r[40.0][1], band))
        print(f"[warm] zeta_all={za}: start scale {a:.2f}, max|zeta|={zmax:.2f}  G(40)={r[40.0][0]:.1f}  "
              f"G_phi(40)={r[40.0][1]:.2f}  <G_phi>_band={band:.2f}  [{time.time()-t0:.0f}s]", flush=True)
        np.save(f"results/lens_front_theta_{za}.npy", th)
    np.save("results/lens_front.npy", np.array(rows))


if __name__ == "__main__" and "tight" in sys.argv:
    main_tight()
