"""
Convergence comparison: scipy SIMP vs dolfinx cantilever.

Runs both optimizers on the same 80×40 cantilever problem (rmin=1.5 elem,
vf=0.4, penal=3) for 150 iterations and plots compliance vs iteration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo import ElasticityProblem, ConeFilter

# ── Shared problem parameters ────────────────────────────────────────────────
nelx, nely   = 80, 40
lx, ly       = 2.0, 1.0
volfrac      = 0.40
penal        = 3.0
rmin_elem    = 1.5          # element units (Andreassen 2011 default)
h            = ly / nely    # 0.025
rmin_phys    = rmin_elem * h
max_iter     = 150
move         = 0.2
E0, Emin, nu = 1.0, 1e-9, 0.3


# ── 1. scipy SIMP ────────────────────────────────────────────────────────────
print("Running scipy SIMP ...")

def lk():
    k = [0.5-nu/6, 0.125+nu/8, -0.25-nu/12, -0.125+3*nu/8,
         -0.25+nu/12, -0.125-nu/8, nu/6, 0.125-3*nu/8]
    return E0/(1-nu**2) * np.array([
        [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
        [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
        [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
        [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
        [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])

KE = lk()
ndof = 2 * (nelx+1) * (nely+1)

def node(ex, ey): return ey + ex*(nely+1)

ex_all = np.repeat(np.arange(nelx), nely)
ey_all = np.tile(np.arange(nely), nelx)
n1 = node(ex_all, ey_all);   n2 = node(ex_all+1, ey_all)
n3 = node(ex_all+1, ey_all+1); n4 = node(ex_all, ey_all+1)
edofs = np.column_stack([2*n1, 2*n1+1, 2*n2, 2*n2+1,
                          2*n3, 2*n3+1, 2*n4, 2*n4+1])

fixed = np.array(sorted({d for ey in range(nely+1)
                          for n in [node(0, ey)] for d in [2*n, 2*n+1]}))
free = np.setdiff1d(np.arange(ndof), fixed)

# 1-node midpoint load
F_sp = np.zeros(ndof)
F_sp[2*node(nelx, nely//2) + 1] = -1.0

# vectorised K assembly indices
ii = np.concatenate([edofs[:, ei] for ei in range(8) for _ in range(8)])
jj = np.concatenate([edofs[:, ej] for _ in range(8) for ej in range(8)])

# cone filter (element units)
cx = ex_all + 0.5; cy = ey_all + 0.5
pairs = cKDTree(np.column_stack([cx, cy])).query_pairs(rmin_elem, output_type="ndarray")
rows, cols, vals = list(range(nelx*nely)), list(range(nelx*nely)), [rmin_elem]*(nelx*nely)
for i, j in pairs:
    d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
    rows += [i, j]; cols += [j, i]; vals += [rmin_elem-d, rmin_elem-d]
H_sp = sp.csr_matrix((vals, (rows, cols)), shape=(nelx*nely, nelx*nely))
Hs_sp = np.asarray(H_sp.sum(1)).ravel()
sp_filt = lambda r: (H_sp @ r) / Hs_sp

x_sp  = np.full(nelx*nely, volfrac)
xP_sp = sp_filt(x_sp)
C_hist_sp = []

for it in range(1, max_iter+1):
    xc = np.maximum(xP_sp, 1e-3)
    ks = Emin + xc**penal * (E0-Emin)
    kv = np.array([ks * KE[ei, ej] for ei in range(8) for ej in range(8)]).ravel()
    K  = sp.csr_matrix((kv, (ii, jj)), shape=(ndof, ndof))
    u  = np.zeros(ndof)
    u[free] = spla.spsolve(K[free, :][:, free].tocsc(), F_sp[free])
    C  = float(F_sp @ u)
    C_hist_sp.append(C)
    Ue = u[edofs]
    dc = -(penal * xc**(penal-1) * (E0-Emin)) * np.einsum('ni,ij,nj->n', Ue, KE, Ue)
    dc_f = (H_sp @ (xc * dc)) / Hs_sp / np.maximum(1e-3, xc)
    dc_abs = np.maximum(-dc_f, 1e-30)
    l1, l2 = 1e-9, 1e9
    while (l2-l1)/(l1+l2) > 1e-6:
        lm = 0.5*(l1+l2)
        xn = np.clip(xP_sp * np.sqrt(dc_abs/lm),
                     np.maximum(0.001, xP_sp-move), np.minimum(1.0, xP_sp+move))
        if xn.mean() > volfrac: l1 = lm
        else:                   l2 = lm
    x_sp  = xn
    xP_sp = sp_filt(x_sp)
    if it % 30 == 0:
        print(f"  scipy  iter {it:4d}  C={C:.4f}")

print(f"  scipy  DONE  C_final={C_hist_sp[-1]:.4f}")


# ── 2. dolfinx cantilever ────────────────────────────────────────────────────
print("Running dolfinx cantilever ...")

fem_prob = ElasticityProblem(nx=nelx, ny=nely, lx=lx, ly=ly,
                              penal=penal, load_type="midpoint")
filt_dx  = ConeFilter(fem_prob.DG0, r_min=rmin_phys)

x_dx  = np.full(nelx*nely, volfrac)
xP_dx = filt_dx.apply(x_dx)
C_hist_dx = []

for it in range(1, max_iter+1):
    xc = np.clip(xP_dx, 1e-3, 1.0)
    fem_prob.update_density(xc)
    u, C = fem_prob.solve()
    C_hist_dx.append(C)
    dc_raw = fem_prob.compliance_sensitivity(u)
    dc_f   = filt_dx.apply(xc * dc_raw) / np.maximum(1e-3, xc)
    dc_abs = np.maximum(-dc_f, 1e-30)
    l1, l2 = 1e-9, 1e9
    while (l2-l1)/(l1+l2) > 1e-6:
        lm = 0.5*(l1+l2)
        xn = np.clip(xP_dx * np.sqrt(dc_abs/lm),
                     np.maximum(0.001, xP_dx-move), np.minimum(1.0, xP_dx+move))
        if xn.mean() > volfrac: l1 = lm
        else:                   l2 = lm
    x_dx  = xn
    xP_dx = filt_dx.apply(x_dx)
    if it % 30 == 0:
        print(f"  dolfinx iter {it:4d}  C={C:.4f}")

print(f"  dolfinx DONE  C_final={C_hist_dx[-1]:.4f}")


# ── 3. Plot ──────────────────────────────────────────────────────────────────
iters_sp = np.arange(1, len(C_hist_sp)+1)
iters_dx = np.arange(1, len(C_hist_dx)+1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# --- Convergence curve ---
ax = axes[0]
ax.plot(iters_sp, C_hist_sp, label=f"scipy  (C_f={C_hist_sp[-1]:.2f})", lw=1.8)
ax.plot(iters_dx, C_hist_dx, label=f"dolfinx (C_f={C_hist_dx[-1]:.2f})", lw=1.8, ls="--")
ax.set_xlabel("Iteration")
ax.set_ylabel("Compliance  C")
ax.set_title("Compliance convergence")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Log-scale difference ---
ax2 = axes[1]
n_common = min(len(C_hist_sp), len(C_hist_dx))
diff = np.abs(np.array(C_hist_sp[:n_common]) - np.array(C_hist_dx[:n_common]))
ax2.semilogy(np.arange(1, n_common+1), diff, color="C2", lw=1.8)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("|C_scipy − C_dolfinx|")
ax2.set_title("Absolute compliance difference (log scale)")
ax2.grid(True, alpha=0.3, which="both")

fig.suptitle(
    f"scipy vs dolfinx — 80×40 cantilever, rmin={rmin_elem} elem, vf={volfrac}, penal={penal}",
    fontsize=11,
)
plt.tight_layout()
out = Path(__file__).parent / "convergence_comparison.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved {out}")
