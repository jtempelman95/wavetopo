r"""Panel across the band: redirection vs resonance (Reviewer R1, Fig. 7).

For the band-optimized and the single-tone panel designs, at each omega in [41, 59], the
region energies of the straight-fiber plate (0) and the design (1) are split exactly as for
the lens:  E_r1/E_r0 = [ (E_r1/E_I1) / (E_r0/E_I0) ] x [ E_I1/E_I0 ],  r in {sensor, payload,
rims}, E_I the interior energy.  The first factor is the redistribution (share), the second
the whole-field growth.

    PYTHONPATH=. python examples/analysis/wt_r1_panel.py   -> results/wt_r1_panel.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
import numpy as np

sys.path.insert(0, _os.path.join(_ROOT, "examples", "dolfinx"))
import dolfinx_panel as P

FREQS = np.arange(41.0, 59.01, 1.0)
DESIGNS = {"band": "results/dolfinx_panel_band_repro_data.npz",
           "tone": "results/dolfinx_panel_tone_repro_data.npz"}

out = {"freqs": FREQS}
for name, path in DESIGNS.items():
    th = np.load(_os.path.join(_ROOT, path))["thopt"]
    rows = []
    for w in FREQS:
        dom = P.rect_mesh(P.Lx, P.Ly, P.HMESH, holes=P.HOLES, h_hole=0.5 * P.HMESH)
        ew = P._attach(dom, float(w))
        res = []
        for t in (np.zeros(ew.theta.x.array.size), th):
            ew.set_theta(t); ew.solve()
            ES, EP, EF, _ = P.measure(ew)
            EI = P.energy_in(ew, P.w_interior)
            res.append((ES, EP, EF, EI))
        (S0, P0, F0, I0), (S1, P1, F1, I1) = res
        g = I1 / I0
        rows.append((w, S1 / S0, P1 / P0, F1 / F0, g, (S1 / I1) / (S0 / I0), (P1 / I1) / (P0 / I0), (F1 / I1) / (F0 / I0)))
        print(f"[{name}] w={w:4.1f}: sensor {S1/S0:5.2f} = share {rows[-1][5]:5.2f} x growth {g:4.2f}; "
              f"payload {P1/P0:4.2f} (share {rows[-1][6]:4.2f}); rims {F1/F0:4.2f} (share {rows[-1][7]:4.2f})", flush=True)
    out[name] = np.array(rows)
np.savez(_os.path.join(_ROOT, "results", "wt_r1_panel.npz"), **out,
         cols="omega,sensor,payload,rims,growth,sensor_share,payload_share,rims_share")
print("wrote results/wt_r1_panel.npz")
