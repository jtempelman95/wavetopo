r"""Robustness of the lens curvature front (referee request): several independent starts per
tow-curvature limit, so that a collapse can be told apart from optimizer stalling.

For one zeta_all (argv[1]) we run the warm-start optimizer of lens_front.py (start scaled
until FEASIBLE, MMA on -log E + mu * curvature penalty, best feasible iterate kept) from:
  * every saved front design (unconstrained, 4.0, 2.5, 1.5, 1.0), each rescaled to feasibility,
  * three random symmetric starts (support values ~ N(0, 0.3 rad), seeds 1-3), rescaled likewise,
and score each result by G(40), G_phi(40) and the band-averaged G_phi over {35,40,45,50}.

    PYTHONPATH=. python  # dolfinx (real PETSc) environment examples/analysis/lens_front_robust.py 0.6
Writes results/lens_front_robust_<zeta>.npy (one row per start).
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_NLTOPT = _os.environ.get("NLTOPT_DIR", _os.path.join(_os.path.dirname(_ROOT), "NLTopt"))
import sys
import time
import numpy as np
from scipy.sparse.linalg import lsqr

sys.path.insert(0, (_ROOT + "/examples/dolfinx"))
sys.path.insert(0, (_ROOT + "/examples/analysis"))
import dolfinx_lens as DL
from wavetopo.dolfinx_wave import csrbf_operators
from lens_front import optimise_warm, evaluate

za = float(sys.argv[1])
ew = DL.make(nx=130)
cent = DL.centroids(ew)
supp, mir = DL.symmetric_support(0.33)
B, Bx, By = csrbf_operators(cent, supp, 0.7)
sym = lambda v: 0.5 * (v - v[mir])
starts = {}
for tag in ("inf", "4.0", "2.5", "1.5", "1.0"):
    th = np.load(f"results/lens_front_theta_{tag}.npy")
    starts[f"from_{tag}"] = sym(lsqr(B, th)[0])
for s in (1, 2, 3):
    starts[f"random{s}"] = sym(np.random.default_rng(s).normal(0, 0.3, B.shape[1]))
rows = []
for k, (name, x0) in enumerate(starts.items()):
    t0 = time.time()
    x, a = optimise_warm(ew, B, Bx, By, sym, x0, za)
    th = B @ x
    zmax = np.abs(np.cos(th) * (Bx @ x) + np.sin(th) * (By @ x)).max()
    r = evaluate(th)
    band = np.mean([r[o][1] for o in r])
    rows.append((za, k, zmax, r[40.0][0], r[40.0][1], band))
    print(f"zeta_all={za}: start {name:9s} (scale {a:.2f})  max|zeta|={zmax:.2f}  G(40)={r[40.0][0]:6.1f}  "
          f"G_phi(40)={r[40.0][1]:5.2f}  <G_phi>_band={band:5.2f}  [{time.time()-t0:.0f}s]", flush=True)
np.save(f"results/lens_front_robust_{za}.npy", np.array(rows))
