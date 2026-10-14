"""
Run EXACT scipy top88 loop but use dolfinx for the FE solve.
If this gives clean results: the bug is in cantilever.py's loop structure.
Uses rmin=1.5 element units = 0.0375 physical (standard Andreassen).
"""
import sys; sys.path.insert(0, '/home/jrt/wavetopo')
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wavetopo import ElasticityProblem
from wavetopo.filters import ConeFilter

nelx, nely = 80, 40
lx, ly = 2.0, 1.0
volfrac, penal = 0.40, 3.0
E0, Emin = 1.0, 1e-9
max_iter, move = 150, 0.2
rmin_elem = 2.5           # element units
h = ly / nely             # = 0.025
rmin_phys = rmin_elem * h  # = 0.0375

# ── dolfinx FE problem ───────────────────────────────────────────────────
fem_prob = ElasticityProblem(nx=nelx, ny=nely, lx=lx, ly=ly,
                              penal=penal, load_type="midpoint")

# ── scipy-style cone filter in element units ─────────────────────────────
# Use element-unit coordinates (identical to physical scaled by h)
ex_all = np.repeat(np.arange(nelx), nely)
ey_all = np.tile(np.arange(nely), nelx)
cx_elem = ex_all + 0.5
cy_elem = ey_all + 0.5
pairs = cKDTree(np.column_stack([cx_elem, cy_elem])).query_pairs(
    rmin_elem, output_type="ndarray")
rows, cols, vals = (list(range(nelx*nely)), list(range(nelx*nely)),
                     [rmin_elem]*(nelx*nely))
for i, j in pairs:
    d = np.hypot(cx_elem[i]-cx_elem[j], cy_elem[i]-cy_elem[j])
    rows += [i, j]; cols += [j, i]; vals += [rmin_elem-d, rmin_elem-d]
H_sp = sp.csr_matrix((vals,(rows,cols)), shape=(nelx*nely,nelx*nely))
Hs_sp = np.asarray(H_sp.sum(1)).ravel()

# ── scipy-style centroid ordering vs dolfinx cell ordering ───────────────
# scipy: cell e = ex*nely + ey  (y varies fastest)
# dolfinx: need to determine actual ordering via centroids
from dolfinx import fem as dfem
import ufl
xcoord = ufl.SpatialCoordinate(fem_prob.domain)
pts = fem_prob.DG0.element.interpolation_points()
cx_fn = dfem.Function(fem_prob.DG0); cy_fn = dfem.Function(fem_prob.DG0)
cx_fn.interpolate(dfem.Expression(xcoord[0], pts))
cy_fn.interpolate(dfem.Expression(xcoord[1], pts))
cx_dx = cx_fn.x.array; cy_dx = cy_fn.x.array

# Physical centroids of scipy cells
sp_cx_phys = (ex_all + 0.5) * (lx/nelx)
sp_cy_phys = (ey_all + 0.5) * (ly/nely)

# For each scipy cell e, find matching dolfinx cell
sp_to_dx = np.zeros(nelx*nely, dtype=int)
for e in range(nelx*nely):
    dists = (cx_dx - sp_cx_phys[e])**2 + (cy_dx - sp_cy_phys[e])**2
    sp_to_dx[e] = np.argmin(dists)
dx_to_sp = np.argsort(sp_to_dx)   # dolfinx index -> scipy index

def sp_filt(r_sp): return (H_sp @ r_sp) / Hs_sp   # filter in scipy order

def dolfinx_solve(xPhys_sp):
    """Solve FE and return compliance + sensitivity, both in scipy cell order."""
    # reorder to dolfinx
    xPhys_dx = xPhys_sp[dx_to_sp]
    xc_dx = np.maximum(xPhys_dx, 1e-3)
    fem_prob.update_density(xc_dx)
    u, compliance = fem_prob.solve()
    dc_dx = fem_prob.compliance_sensitivity(u)
    # reorder back to scipy
    dc_sp = dc_dx[sp_to_dx]
    xc_sp = np.maximum(xPhys_sp, 1e-3)
    return compliance, dc_sp, xc_sp

# ── top88 loop (pure scipy structure, dolfinx FE) ────────────────────────
x_sp = np.full(nelx*nely, volfrac)
xPhys_sp = sp_filt(x_sp)
C_hist = []

for it in range(1, max_iter+1):
    C, dc_raw_sp, xc_sp = dolfinx_solve(xPhys_sp)
    C_hist.append(C)

    # weighted sensitivity filter
    dc_f = (H_sp @ (xc_sp * dc_raw_sp)) / Hs_sp / np.maximum(1e-3, xc_sp)

    dc_abs = np.maximum(-dc_f, 1e-30)
    l1, l2 = 1e-9, 1e9
    while (l2-l1)/(l1+l2) > 1e-6:
        lm = 0.5*(l1+l2)
        xn = np.clip(xPhys_sp * np.sqrt(dc_abs/lm),
                     np.maximum(0.001, xPhys_sp - move),
                     np.minimum(1.0, xPhys_sp + move))
        if xn.mean() > volfrac: l1 = lm
        else: l2 = lm
    x_sp = xn
    xPhys_sp = sp_filt(x_sp)

    if it <= 5 or it % 20 == 0:
        print(f"  iter {it:4d}  C={C:.4f}  vol={xPhys_sp.mean():.4f}")

# ── Plot ─────────────────────────────────────────────────────────────────
rho_grid = xPhys_sp.reshape(nelx, nely).T
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, dat, title in [(axes[0], rho_grid, "continuous"),
                        (axes[1], rho_grid>=0.5, "thresholded")]:
    im = ax.imshow(dat.astype(float), origin="lower", cmap="gray_r",
                   vmin=0, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"scipy loop + dolfinx FE  rmin=1.5  C={C_hist[-1]:.2f} — {title}")
plt.tight_layout()
plt.savefig("/tmp/scipy_loop_dolfinx_fe.png", dpi=130, bbox_inches="tight")
print(f"\nFinal C={C_hist[-1]:.4f}")
print("Saved /tmp/scipy_loop_dolfinx_fe.png")
