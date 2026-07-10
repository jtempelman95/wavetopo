"""Compare 1-node vs 3-node load in scipy SIMP to isolate dolfinx scatter cause."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nelx, nely = 60, 30
volfrac, penal, rmin = 0.40, 3.0, 1.5
E0, Emin, nu = 1.0, 1e-9, 0.3
max_iter, move = 80, 0.2

def lk():
    k = [0.5-nu/6, 0.125+nu/8, -0.25-nu/12, -0.125+3*nu/8,
         -0.25+nu/12, -0.125-nu/8, nu/6, 0.125-3*nu/8]
    return E0/(1-nu**2)*np.array([
        [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
        [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
        [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
        [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
        [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
KE = lk()

ndof = 2*(nelx+1)*(nely+1)
def node(ex, ey): return ey + ex*(nely+1)

ex_all = np.repeat(np.arange(nelx), nely)
ey_all = np.tile(np.arange(nely), nelx)
n1 = node(ex_all, ey_all);   n2 = node(ex_all+1, ey_all)
n3 = node(ex_all+1, ey_all+1); n4 = node(ex_all, ey_all+1)
edofs = np.column_stack([2*n1, 2*n1+1, 2*n2, 2*n2+1,
                          2*n3, 2*n3+1, 2*n4, 2*n4+1])

fixed = np.array(sorted({d for ey in range(nely+1)
                          for n in [node(0,ey)] for d in [2*n, 2*n+1]}))
free = np.setdiff1d(np.arange(ndof), fixed)

ii = np.concatenate([edofs[:, ei] for ei in range(8) for _ in range(8)])
jj = np.concatenate([edofs[:, ej] for _ in range(8) for ej in range(8)])

def assemble_K(xc):
    ks = Emin + xc**penal * (E0 - Emin)
    vals = np.array([ks * KE[ei, ej]
                     for ei in range(8) for ej in range(8)]).ravel()
    return sp.csr_matrix((vals, (ii, jj)), shape=(ndof, ndof))

cx = ex_all + 0.5; cy = ey_all + 0.5
pairs = cKDTree(np.column_stack([cx, cy])).query_pairs(rmin, output_type="ndarray")
rows, cols, vals = list(range(nelx*nely)), list(range(nelx*nely)), [rmin]*(nelx*nely)
for i, j in pairs:
    d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
    rows += [i, j]; cols += [j, i]; vals += [rmin-d, rmin-d]
H = sp.csr_matrix((vals, (rows, cols)), shape=(nelx*nely, nelx*nely))
Hs = np.asarray(H.sum(1)).ravel()
def filt(r): return (H @ r) / Hs

def run(label, F):
    x = np.full(nelx*nely, volfrac)
    xPhys = filt(x)
    C_last = 0
    for it in range(1, max_iter+1):
        xc = np.maximum(xPhys, 1e-3)
        K = assemble_K(xc)
        u = np.zeros(ndof)
        u[free] = spla.spsolve(K[free,:][:,free].tocsc(), F[free])
        C = float(F @ u); C_last = C
        Ue = u[edofs]
        dc = -(penal * xc**(penal-1) * (E0-Emin)) * np.einsum('ni,ij,nj->n', Ue, KE, Ue)
        dc_f = (H @ (xc * dc)) / Hs / np.maximum(1e-3, xc)
        l1, l2 = 1e-9, 1e9
        while (l2-l1)/(l1+l2) > 1e-6:
            lm = 0.5*(l1+l2)
            xn = np.clip(xPhys * np.sqrt(np.maximum(-dc_f, 1e-30)/lm),
                         np.maximum(0.001, xPhys-move),
                         np.minimum(1.0, xPhys+move))
            if xn.mean() > volfrac: l1 = lm
            else: l2 = lm
        x = xn; xPhys = filt(x)
        if it % 20 == 0:
            print(f"  [{label}] iter {it:3d}  C={C:.4f}")
    return xPhys, C_last

mid = nely // 2
print("=== 1-node load ===")
F1 = np.zeros(ndof); F1[2*node(nelx, mid)+1] = -1.0
rho1, C1 = run("1-node", F1)

print("=== 3-node load ===")
F3 = np.zeros(ndof)
F3[2*node(nelx, mid-1)+1] = -0.25
F3[2*node(nelx, mid  )+1] = -0.50
F3[2*node(nelx, mid+1)+1] = -0.25
rho3, C3 = run("3-node", F3)

fig, axes = plt.subplots(2, 2, figsize=(14, 6))
for row, (rho, C, lb) in enumerate([(rho1,C1,"1-node"),(rho3,C3,"3-node (dolfinx style)")]):
    g = rho.reshape(nelx, nely).T
    for col, (dat, suf) in enumerate([(g,"continuous"),(g>=0.5,"thresholded")]):
        axes[row][col].imshow(dat.astype(float), origin="lower", cmap="gray_r",
                               vmin=0, vmax=1, aspect="equal")
        axes[row][col].set_title(f"{lb}  C={C:.2f} — {suf}", fontsize=9)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "load_compare.png", dpi=130, bbox_inches="tight")
print(f"\n1-node C={C1:.4f}   3-node C={C3:.4f}")
print(f"Saved {Path(__file__).parent / 'load_compare.png'}")
