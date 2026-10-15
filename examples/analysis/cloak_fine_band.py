r"""Frequency response of the cloak re-optimized on the finer mesh (h = 0.03), to check that
the narrowband conclusion (W5) does not rest on the mesh-tuned h = 0.045 design.

    PYTHONPATH=. python  # dolfinx (real PETSc) environment examples/analysis/cloak_fine_band.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_NLTOPT = _os.environ.get("NLTOPT_DIR", _os.path.join(_os.path.dirname(_ROOT), "NLTopt"))
import sys
import numpy as np
sys.path.insert(0, (_ROOT + "/examples/dolfinx"))
import dolfinx_cloak_conforming_vec as CV

CV.HMESH = 0.030
th = np.load("results/dolfinx_cloak_conforming_h030_data.npz")["thopt"]
rows = []
for om in (45.0, 47.5, 50.0, 52.5, 55.0):
    CV.OMEGA = om
    ew, ref = CV.build()
    n = ew.theta.x.array.size
    ew.set_theta(np.zeros(n)); ew.solve(); J0 = ew.cloak_mismatch()
    ew.set_theta(th); ew.solve(); r = J0 / ew.cloak_mismatch()
    rows.append((om, r)); print(f"omega={om:4.1f}  fine-mesh optimized cloak reduction={r:5.1f}x", flush=True)
np.save("results/cloak_fine_band.npy", np.array(rows))
