r"""Mesh check of the multifunctional panel (referee request): the band design re-evaluated on a
finer conforming mesh (h = 0.04 instead of the design mesh 0.055), across the band.

The design field is resampled through (cos 2theta, sin 2theta) onto the finer mesh's cells.
    PYTHONPATH=. ~/miniforge3/envs/dolfinx_complex/bin/python examples/analysis/panel_mesh_check.py
"""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "examples", "dolfinx")); sys.path.insert(0, HERE)
import dolfinx_panel as P
from wt_verify import transfer

D = np.load(os.path.join(ROOT, "results", "dolfinx_panel_band_repro_data.npz"))
rows = []
for h in (0.055, 0.040):
    for w in (44.0, 47.0, 50.0, 53.0, 56.0):
        dom = P.rect_mesh(P.Lx, P.Ly, h, holes=P.HOLES, h_hole=0.5 * h)
        ew = P._attach(dom, w)
        n = ew.theta.x.array.size
        dom.topology.create_connectivity(2, 0); conn = dom.topology.connectivity(2, 0); xy = dom.geometry.x[:, :2]
        cent = np.zeros((n, 2))
        for e in range(dom.topology.index_map(2).size_local):
            cent[ew.S.dofmap.cell_dofs(e)[0]] = xy[conn.links(e)].mean(0)
        th = transfer(D["cent"], D["thopt"], cent)
        ew.set_theta(np.zeros(n)); ew.solve(); S0, P0, F0, _ = P.measure(ew)
        ew.set_theta(th); ew.solve(); S1, P1, F1, _ = P.measure(ew)
        rows.append((h, w, S1 / S0, P1 / P0, F1 / F0))
        print(f"h={h:.3f} omega={w:4.1f}  sensor={S1/S0:5.2f}x payload={P1/P0:5.2f}x rims={F1/F0:5.2f}x", flush=True)
np.save(os.path.join(ROOT, "results", "panel_mesh_check.npy"), np.array(rows))
