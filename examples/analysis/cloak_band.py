r"""Is the optimized cloak a broadband or a single-frequency device?

Re-evaluates the saved conforming-mesh cloak designs (optimized at omega = 50) --
unconstrained (36x) and curvature-limited at zeta_all = 1/R_V (9.6x) -- and the
analytical transformation-direction field, across omega, on the same holed mesh
and hole-free reference (rebuilt at each omega).  Reduction = J_uncloaked / J_design
of the far-field mismatch outside R_C.

    PYTHONPATH=. .../dolfinx_complex/bin/python examples/analysis/cloak_band.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
import numpy as np

sys.path.insert(0, (_ROOT + "/examples/dolfinx"))
import dolfinx_cloak_conforming_vec as CV


def main():
    unc = np.load("results/dolfinx_cloak_conforming_data.npz")["thopt"]
    h2h = np.load("results/cloak_headtohead.npz")
    designs = {"optimized": unc, "curv-limited": h2h["th_opt"], "transformation": h2h["th_an"]}
    rows = []
    for om in (40.0, 45.0, 47.5, 50.0, 52.5, 55.0, 60.0):
        CV.OMEGA = om
        ew, ref = CV.build()
        n = ew.theta.x.array.size
        ew.set_theta(np.zeros(n)); ew.solve(); J0 = ew.cloak_mismatch()
        r = [om]
        for name, th in designs.items():
            ew.set_theta(th); ew.solve(); r.append(J0 / ew.cloak_mismatch())
        rows.append(r)
        print(f"omega={om:5.1f}  " + "  ".join(f"{k}={v:6.1f}x" for k, v in zip(designs, r[1:])),
              flush=True)
    np.save("results/cloak_band.npy", np.array(rows))


if __name__ == "__main__":
    main()
