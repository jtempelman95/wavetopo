r"""The lens re-optimized with a better absorber (referee request).

The paper's sponge (width 0.6) reflects |R| ~ 0.32 of the along-fiber qP wave; doubled
outward (width 1.2, interior unchanged) it reflects |R| <= 0.13 (wt_verify.py C).  Here the
lens is RE-OPTIMIZED from straight fibers with the doubled sponge (same MMA, move limit,
iteration count, mirror-symmetric CS-RBF field and objective as dolfinx_lens.run), and both
it and the original design are swept over omega in [20, 60] with the doubled sponge.

    PYTHONPATH=. python  # dolfinx (real PETSc) environment examples/analysis/lens_absorber.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_NLTOPT = _os.environ.get("NLTOPT_DIR", _os.path.join(_os.path.dirname(_ROOT), "NLTopt"))
import sys
import numpy as np
sys.path.insert(0, (_ROOT + "/examples/dolfinx"))
sys.path.insert(0, (_ROOT + "/examples/analysis"))
import dolfinx_lens as DL
import wt_verify as V
from wavetopo.dolfinx_wave import support_map
from wavetopo.cfrp_optimizer import MMA

PAD = 0.6
ew = V.lens_make(omega=40.0, pad=PAD)
cent = DL.centroids(ew)
supp, mir = DL.symmetric_support(0.33)
B, supp = support_map(cent, (0, DL.Lx), (0, DL.Ly), 0.33, 0.7, supp=supp)
Md = B.shape[1]
sym = lambda v: 0.5 * (v - v[mir])
ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve(); E0 = ew.focus_energy()
mma = MMA(np.full(Md, -np.pi / 2), np.full(Md, np.pi / 2), move=0.10)
x = np.zeros(Md); best = (E0, np.zeros(ew.theta.x.array.size))
for it in range(70):
    th = B @ x; ew.set_theta(th); ew.solve()
    J = ew.focus_energy()
    if J > best[0]:
        best = (J, th.copy())
    g = sym(B.T @ ew.focus_grad())
    x = sym(mma.update(x, -g))
    if it % 10 == 0 or it == 69:
        print(f"[{it:3d}] focus gain {J/E0:.1f}x", flush=True)
th_new = best[1]
th_old = V.lens_theta(ew)

rows = []
for om in np.arange(20.0, 60.01, 2.5):
    e = V.lens_make(omega=om, pad=PAD)
    G1, P1, g1 = V.lens_pair(e, th_new)
    G0, P0, g0 = V.lens_pair(e, th_old)
    rows.append((om, G1, P1, g1, G0, P0, g0))
    print(f"omega={om:4.1f}  re-optimized: G={G1:6.1f} G_phi={P1:5.2f} growth={g1:5.2f} | "
          f"original design: G={G0:6.1f} G_phi={P0:5.2f}", flush=True)
np.savez("results/lens_absorber.npz", rows=np.array(rows), th_new=th_new, cent=cent,
         cols="omega,G_new,Gphi_new,growth_new,G_old,Gphi_old,growth_old")
print("wrote results/lens_absorber.npz")
