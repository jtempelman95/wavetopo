r"""Anisotropy dose-response of orientation-only wave control (prediction W4).

Material family Q(s) = Q_iso + s (Q_cfrp - Q_iso), Q_iso sharing the CFRP's
rotation invariants U1, U4, U5: s = 0 is isotropic (orientation is a dead variable),
s = 1 the CFRP lamina; the anisotropic Tsai-Pagano invariants U2, U3 scale as s.
For each s the lens (focus energy at omega = 40) is re-optimized exactly as in
dolfinx_lens.run (symmetric CS-RBF, MMA, 70 iterations) and we report
  G      focus-energy gain at omega = 40,
  G_phi  redirection (focus fraction ratio) at omega = 40,
  <G_phi> band-averaged redirection over omega in {35, 40, 45, 50}.

    PYTHONPATH=. .../dolfinx_complex/bin/python examples/analysis/lens_dose.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
import time
import numpy as np
import ufl
from dolfinx import fem

sys.path.insert(0, (_ROOT + "/examples/dolfinx"))
sys.path.insert(0, (_ROOT + "/examples/analysis"))
import dolfinx_lens as DL
from wavetopo.dolfinx_wave import support_map
from wavetopo.cfrp_optimizer import MMA
from lens_mechanism import metrics


def Q_family(s, Q):
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    U1 = (3 * Q11 + 3 * Q22 + 2 * Q12 + 4 * Q66) / 8.0
    U5 = (Q11 + Q22 - 2 * Q12 + 4 * Q66) / 8.0
    Qiso = np.array([[U1, U1 - 2 * U5, 0.0], [U1 - 2 * U5, U1, 0.0], [0.0, 0.0, U5]])
    return Qiso + s * (Q - Qiso)


def make_s(s, omega):
    ew = DL.make(nx=130, omega=omega)
    ew.Qf = Q_family(s, ew.Qf)
    ew._forms()
    return ew


def optimise(s, iters=70):
    ew = make_s(s, 40.0)
    cent = DL.centroids(ew)
    supp, mir = DL.symmetric_support(0.33)
    B, supp = support_map(cent, (0, DL.Lx), (0, DL.Ly), 0.33, 0.7, supp=supp)
    sym = lambda v: 0.5 * (v - v[mir])
    mma = MMA(np.full(B.shape[1], -np.pi / 2), np.full(B.shape[1], np.pi / 2), move=0.10)
    x = np.zeros(B.shape[1])
    ew.set_theta(np.zeros(B.shape[0])); ew.solve(); E0 = ew.focus_energy()
    best = (E0, np.zeros(B.shape[0]))
    for it in range(iters):
        th = B @ x; ew.set_theta(th); ew.solve(); J = ew.focus_energy()
        if J > best[0]:
            best = (J, th.copy())
        g = sym(B.T @ ew.focus_grad())
        x = sym(mma.update(x, -g))
    return best[1]


def evaluate(s, th):
    res = {}
    for om in (35.0, 40.0, 45.0, 50.0):
        ew = make_s(s, om)
        ew.set_theta(np.zeros(th.size)); ew.solve(); E0, I0, P0 = metrics(ew)
        ew.set_theta(th); ew.solve(); E1, I1, P1 = metrics(ew)
        res[om] = (E1 / E0, (E1 / I1) / (E0 / I0))
    return res


def main():
    rows = []
    for s in (0.0, 0.25, 0.5, 0.75, 1.0, 1.1):
        t0 = time.time()
        th = optimise(s)
        r = evaluate(s, th)
        Q = Q_family(s, DL.make(nx=20).Qf)
        ratio = Q[0, 0] / Q[1, 1]
        Gphi_band = np.mean([r[o][1] for o in r])
        rows.append((s, ratio, r[40.0][0], r[40.0][1], Gphi_band))
        print(f"s={s:4.2f} E1/E2~{ratio:5.1f}: G(40)={r[40.0][0]:7.1f}  G_phi(40)={r[40.0][1]:5.2f}  "
              f"<G_phi>_band={Gphi_band:5.2f}   [{time.time()-t0:.0f}s]")
        np.save(f"results/lens_dose_theta_s{s:.2f}.npy", th)
    np.save("results/lens_dose.npy", np.array(rows))
    print("wrote results/lens_dose.npy")


if __name__ == "__main__":
    main()
