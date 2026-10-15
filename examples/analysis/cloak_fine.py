r"""Mesh-convergence of the optimized cloak: re-optimize (warm start) on h = 0.03.

The 36x design was optimized on h = 0.045; evaluated on h = 0.03 it gives 26x
(wt_verify.py D).  An interference cancellation tuned to one mesh's discrete
dispersion is mesh-sensitive, so the fair test is whether the same level is
reachable on the finer mesh.  Outputs go to *_h030* files (paper data untouched).

    PYTHONPATH=. python  # dolfinx (real PETSc) environment examples/analysis/cloak_fine.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_NLTOPT = _os.environ.get("NLTOPT_DIR", _os.path.join(_os.path.dirname(_ROOT), "NLTopt"))
import sys
import numpy as np
sys.path.insert(0, (_ROOT + "/examples/dolfinx"))
sys.path.insert(0, (_ROOT + "/examples/analysis"))
import dolfinx_cloak_conforming_vec as CV
from wt_verify import transfer

CV.HMESH = 0.030
_savez = np.savez
CV.np.savez = lambda p, **kw: _savez(p.replace("conforming_data", "conforming_h030_data"), **kw)
import matplotlib.figure as mf
_sf = mf.Figure.savefig
mf.Figure.savefig = lambda self, p, *a, **k: _sf(self, str(p).replace(".png", "_h030.png"), *a, **k)

ew, ref = CV.build()
cn = CV.centroids(ew)
old = np.load("results/dolfinx_cloak_conforming_data.npz")
np.savez("results/cloak_warm_h030.npz", thopt=transfer(old["cent"], old["thopt"], cn))
del ew, ref
CV.main(iters=80, move=0.05, warm="results/cloak_warm_h030.npz")
