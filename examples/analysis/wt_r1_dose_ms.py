r"""Multi-start anisotropy study (Reviewer R1, m2): how much of the W4 trend is optimizer noise?

For one lamina s of the family Q(s) (lens_dose.py), the lens is optimized from straight
fibers (the paper's run, reloaded) and from three random symmetric starts (support values
N(0, 0.3 rad), seeds fixed), exactly as in lens_dose.optimise otherwise.  Every design is
evaluated by its band-averaged redirection over omega in {35, 40, 45, 50}, with the paper's
absorbing layer and with the layer twice as wide (|R| <= 0.13).

    PYTHONPATH=. python examples/analysis/wt_r1_dose_ms.py <s>   -> results/wt_r1_dose_ms_<s>.npy
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
import numpy as np

sys.path.insert(0, _os.path.join(_ROOT, "examples", "dolfinx"))
sys.path.insert(0, _os.path.join(_ROOT, "examples", "analysis"))
import dolfinx_lens as DL
import lens_dose as LD
import wt_verify as V
from wavetopo.dolfinx_wave import support_map
from wavetopo.cfrp_optimizer import MMA

BAND = (35.0, 40.0, 45.0, 50.0)


def optimise_from(s, x0, iters=70):
    ew = LD.make_s(s, 40.0)
    cent = DL.centroids(ew)
    supp, mir = DL.symmetric_support(0.33)
    B, supp = support_map(cent, (0, DL.Lx), (0, DL.Ly), 0.33, 0.7, supp=supp)
    sym = lambda v: 0.5 * (v - v[mir])
    mma = MMA(np.full(B.shape[1], -np.pi / 2), np.full(B.shape[1], np.pi / 2), move=0.10)
    x = sym(x0(B.shape[1], mir))
    ew.set_theta(np.zeros(B.shape[0])); ew.solve(); E0 = ew.focus_energy()
    best = (E0, np.zeros(B.shape[0]))
    for it in range(iters):
        th = B @ x; ew.set_theta(th); ew.solve(); J = ew.focus_energy()
        if J > best[0]:
            best = (J, th.copy())
        g = sym(B.T @ ew.focus_grad())
        x = sym(mma.update(x, -g))
    return best[1], cent


def band(s, th, cent, pad):
    vals = []
    for om in BAND:
        ew = V.lens_make(omega=om, pad=pad)
        ew.Qf = LD.Q_family(s, ew.Qf); ew._forms()
        t = V.transfer(cent, th, DL.centroids(ew)) if pad else th
        _, Gp, _ = V.lens_pair(ew, t)
        vals.append(Gp)
    return float(np.mean(vals))


if __name__ == "__main__":
    s = float(sys.argv[1])
    rows = []
    th0 = np.load(_os.path.join(_ROOT, "results", f"lens_dose_theta_s{s:.2f}.npy"))
    cent0 = DL.centroids(LD.make_s(s, 40.0))
    starts = [("straight (paper run)", None)]
    for seed in (11, 12, 13):
        rng = np.random.default_rng(seed)
        starts.append((f"random {seed}", lambda n, mir, r=rng: 0.3 * r.standard_normal(n)))
    for lab, x0 in starts:
        if x0 is None:
            th, cent = th0, cent0
        else:
            th, cent = optimise_from(s, x0)
        b0, b1 = band(s, th, cent, 0.0), band(s, th, cent, 0.6)
        rows.append((s, b0, b1))
        print(f"s={s:.2f} {lab:22s}: band G_phi {b0:5.2f} (paper layer)  {b1:5.2f} (layer x2)", flush=True)
    np.save(_os.path.join(_ROOT, "results", f"wt_r1_dose_ms_{s:.2f}.npy"), np.array(rows))
