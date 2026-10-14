"""
Standalone SIMP cantilever - pure scipy, no dolfinx.
Reimplements the Andreassen 88-line top88 algorithm directly.
If this gives scatter, the bug is in SIMP itself for this problem.
If this is clean, the bug is in our dolfinx formulation.
"""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Problem parameters ──────────────────────────────────────────────────
nelx, nely = 80, 40        # elements in x (width) and y (height)
volfrac    = 0.40
penal      = 3.0
rmin       = 2.5           # filter radius in element units
E0, Emin   = 1.0, 1e-9
nu         = 0.3
max_iter   = 150
move       = 0.2

# ── Element stiffness matrix (unit square, plane stress) ─────────────────
def lk(nu):
    k = np.array([0.5-nu/6, 0.125+nu/8, -0.25-nu/12, -0.125+3*nu/8,
                  -0.25+nu/12, -0.125-nu/8, nu/6, 0.125-3*nu/8])
    KE = E0 / (1-nu**2) * np.array([
        [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
        [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],  # fixed: was k[3] → k[5]
        [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
        [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
        [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]
    ])
    return KE

KE = lk(nu)

# ── DOF mapping ─────────────────────────────────────────────────────────
ndof = 2 * (nelx+1) * (nely+1)

def node(ex, ey):
    return ey + ex*(nely+1)

def edof(ex, ey):
    n1 = node(ex, ey); n2 = node(ex+1, ey)
    n3 = node(ex+1, ey+1); n4 = node(ex, ey+1)
    return np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1,
                     2*n3, 2*n3+1, 2*n4, 2*n4+1])

# ── Load: point load at midpoint of right edge (y = nely/2) ─────────────
F = np.zeros(ndof)
mid_node = node(nelx, nely//2)
F[2*mid_node + 1] = -1.0   # downward unit force

# ── Boundary conditions: clamp left edge (x=0) ──────────────────────────
fixed = []
for ey in range(nely+1):
    n = node(0, ey)
    fixed += [2*n, 2*n+1]
fixed = np.array(sorted(set(fixed)))
free  = np.setdiff1d(np.arange(ndof), fixed)

# ── Cone filter (element-unit distances) ────────────────────────────────
# centroids in element units: (ix+0.5, iy+0.5)
cx = np.array([(ex+0.5) for ex in range(nelx) for ey in range(nely)])
cy = np.array([(ey+0.5) for ex in range(nelx) for ey in range(nely)])
cents = np.column_stack([cx, cy])
tree  = cKDTree(cents)
pairs = tree.query_pairs(rmin, output_type="ndarray")

rows, cols, vals = list(range(len(cents))), list(range(len(cents))), [rmin]*len(cents)
for i,j in pairs:
    d = np.linalg.norm(cents[i]-cents[j])
    w = rmin - d
    rows += [i,j]; cols += [j,i]; vals += [w,w]

H  = sp.csr_matrix((vals,(rows,cols)), shape=(len(cents),len(cents)))
Hs = np.asarray(H.sum(1)).ravel()

def apply_filter(rho):
    return (H @ rho) / Hs

# ── Global stiffness assembly ────────────────────────────────────────────
def assemble_K(xPhys):
    # element stiffness scale: Emin + xPhys^p * (E0 - Emin)
    ke_scale = Emin + xPhys**penal * (E0 - Emin)
    
    rows_K, cols_K, vals_K = [], [], []
    for ex in range(nelx):
        for ey in range(nely):
            e   = ex*nely + ey
            dofs = edof(ex, ey)
            Ke   = ke_scale[e] * KE
            for i,di in enumerate(dofs):
                for j,dj in enumerate(dofs):
                    rows_K.append(di); cols_K.append(dj); vals_K.append(Ke[i,j])
    K = sp.csr_matrix((vals_K,(rows_K,cols_K)), shape=(ndof,ndof))
    return K

# ── Main optimization loop ───────────────────────────────────────────────
x     = np.full(nelx*nely, volfrac)
xPhys = apply_filter(x)

compliance_hist = []

for it in range(1, max_iter+1):
    xPhys_c = np.maximum(xPhys, 1e-3)
    K = assemble_K(xPhys_c)
    
    # Solve
    K_free = K[free,:][:,free].tocsc()
    u_free = spla.spsolve(K_free, F[free])
    U = np.zeros(ndof)
    U[free] = u_free
    C = float(F @ U)
    
    # Element sensitivities
    dc = np.zeros(nelx*nely)
    for ex in range(nelx):
        for ey in range(nely):
            e    = ex*nely + ey
            dofs = edof(ex, ey)
            Ue   = U[dofs]
            dc[e] = -penal * (xPhys_c[e]**(penal-1)) * (E0-Emin) * (Ue @ KE @ Ue)
    
    # Weighted sensitivity filter (top88 style)
    dc_filt = (H @ (xPhys_c * dc)) / Hs / np.maximum(1e-3, xPhys_c)
    
    compliance_hist.append(C)
    if it <= 10 or it % 20 == 0:
        print(f"  Iter {it:4d}  C={C:.4f}  vol={xPhys.mean():.4f}")
    
    # OC update (scaling by xPhys, top88 style)
    dc_abs = np.maximum(-dc_filt, 1e-30)
    l1, l2 = 1e-9, 1e9
    while (l2-l1)/(l1+l2) > 1e-6:
        lmid  = 0.5*(l1+l2)
        x_new = np.clip(xPhys * np.sqrt(dc_abs/lmid),
                        np.maximum(0.001, xPhys-move),
                        np.minimum(1.0, xPhys+move))
        if x_new.mean() > volfrac: l1 = lmid
        else:                      l2 = lmid
    x     = x_new
    xPhys = apply_filter(x)
    
    change = np.max(np.abs(x - xPhys))
    if change < 1e-3 and it > 5:
        print(f"  Converged at iter {it}")
        break

# ── Plot ────────────────────────────────────────────────────────────────
rho_grid = xPhys.reshape(nelx, nely).T   # (nely, nelx)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, data, title in [
    (axes[0], rho_grid, "Continuous density"),
    (axes[1], (rho_grid >= 0.5).astype(float), "Thresholded"),
]:
    im = ax.imshow(data, origin="lower", cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("x elements"); ax.set_ylabel("y elements")
fig.suptitle(f"Scipy SIMP cantilever  C={compliance_hist[-1]:.2f}", y=1.02)
plt.tight_layout()
out = Path(__file__).parent / "simp_scipy_result.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}  (C_final={compliance_hist[-1]:.4f})")
