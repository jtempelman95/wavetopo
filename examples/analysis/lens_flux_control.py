r"""Control for W3: is energy channeling generic physics, or selected by the optimizer?

Same flux-vs-fiber measurement as lens_flux.py, on curved fiber fields that were NOT
optimized: three random smooth CS-RBF fields (same support grid and radius as the lens,
seeded, support values ~ N(0, 0.5 rad)) and one analytic field.  If the flux follows these
fibers as closely as it follows the optimized lens's, channeling is a property of the
lamina, not of the design.

    PYTHONPATH=. python  # dolfinx (real PETSc) environment examples/analysis/lens_flux_control.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_NLTOPT = _os.environ.get("NLTOPT_DIR", _os.path.join(_os.path.dirname(_ROOT), "NLTopt"))
import sys
import numpy as np
sys.path.insert(0, (_ROOT + "/examples/dolfinx"))
sys.path.insert(0, (_ROOT + "/examples/analysis"))
import dolfinx_lens as DL
from lens_flux import flux_cells, alignment, M_SP
from wavetopo.dolfinx_wave import support_map

rows = []
for om in (40.0, 50.0):
    ew = DL.make(nx=130, omega=om)
    cent = DL.centroids(ew)
    mask = ((cent[:, 0] > 0.3) & (cent[:, 0] < DL.Lx - M_SP) &
            (cent[:, 1] > M_SP) & (cent[:, 1] < DL.Ly - M_SP)).astype(float)
    B, supp = support_map(cent, (0, DL.Lx), (0, DL.Ly), 0.33, 0.7)
    fields = {f"random{s}": B @ np.random.default_rng(s).normal(0, 0.5, B.shape[1]) for s in (1, 2, 3)}
    fields["analytic"] = 0.6 * np.sin(np.pi * cent[:, 0] / 2) * np.sin(2 * np.pi * cent[:, 1] / 3)
    fields["lens"] = np.load("results/dolfinx_lens_data.npz")["thopt"]
    for name, t in fields.items():
        ew.set_theta(t); ew.solve(); P = flux_cells(ew)
        dfib, ffib = alignment(P, t, mask)
        dx, fx = alignment(P, np.zeros(t.size), mask)
        rms = np.degrees(np.sqrt(np.sum(mask * t ** 2) / mask.sum()))
        rows.append((om, name, dfib, 100 * ffib, dx, 100 * fx, rms))
        print(f"omega={om:4.0f} {name:9s} (RMS angle {rms:4.1f} deg): flux-vs-fiber {dfib:5.1f} deg "
              f"({100*ffib:4.1f}% within 20 deg) | flux-vs-x {dx:5.1f} deg", flush=True)
np.save("results/lens_flux_control.npy", np.array(rows, dtype=object), allow_pickle=True)
