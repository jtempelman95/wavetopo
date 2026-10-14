"""
Compare sensitivity computation between scipy (direct KE) and dolfinx (L2 projection).
Runs one FE solve at uniform density vf=0.4, computes dc both ways, compares.
"""
import sys; sys.path.insert(0, '/home/jrt/wavetopo')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ── dolfinx solve ────────────────────────────────────────────────────────
from wavetopo import ElasticityProblem

nx_arg, ny_arg = 20, 10   # small enough to debug
lx, ly = 2.0, 1.0
fem_prob = ElasticityProblem(nx=nx_arg*2, ny=nx_arg, lx=lx, ly=ly, penal=3.0)
nelx, nely = nx_arg*2, nx_arg   # 40 x-elems, 20 y-elems

rho_val = 0.4
rho_arr = np.full(nelx*nely, rho_val)
fem_prob.update_density(np.clip(rho_arr, 1e-3, 1.0))
u_dolfinx, C_dolfinx = fem_prob.solve()
dc_dolfinx = fem_prob.compliance_sensitivity(u_dolfinx)
print(f"dolfinx:  C={C_dolfinx:.6f}   dc min={dc_dolfinx.min():.4e}  max={dc_dolfinx.max():.4e}")

# ── dolfinx cell centroids to identify cell ordering ────────────────────
from dolfinx import fem as dfem
import ufl
DG0 = fem_prob.DG0
xcoord = ufl.SpatialCoordinate(fem_prob.domain)
pts = DG0.element.interpolation_points()
cx_fn = dfem.Function(DG0); cy_fn = dfem.Function(DG0)
cx_fn.interpolate(dfem.Expression(xcoord[0], pts))
cy_fn.interpolate(dfem.Expression(xcoord[1], pts))
cx_arr = cx_fn.x.array.copy()
cy_arr = cy_fn.x.array.copy()

# ── scipy solve ──────────────────────────────────────────────────────────
E0, Emin, nu, penal = 1.0, 1e-9, 0.3, 3.0

def lk():
    k=[0.5-nu/6,0.125+nu/8,-0.25-nu/12,-0.125+3*nu/8,-0.25+nu/12,-0.125-nu/8,nu/6,0.125-3*nu/8]
    return E0/(1-nu**2)*np.array([[k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
     [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],[k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
     [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],[k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
     [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],[k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
     [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
KE = lk()

ndof = 2*(nelx+1)*(nely+1)
def node(ex,ey): return ey+ex*(nely+1)

ex_all = np.repeat(np.arange(nelx), nely)
ey_all = np.tile(np.arange(nely), nelx)
n1=node(ex_all,ey_all); n2=node(ex_all+1,ey_all)
n3=node(ex_all+1,ey_all+1); n4=node(ex_all,ey_all+1)
edofs=np.column_stack([2*n1,2*n1+1,2*n2,2*n2+1,2*n3,2*n3+1,2*n4,2*n4+1])

fixed=np.array(sorted({d for ey in range(nely+1) for n in [node(0,ey)] for d in [2*n,2*n+1]}))
free=np.setdiff1d(np.arange(ndof),fixed)

# Load: 3-node style matching dolfinx
h_y = ly / nely
mid = nely//2
F=np.zeros(ndof)
F[2*node(nelx,mid-1)+1]=-0.25; F[2*node(nelx,mid)+1]=-0.50; F[2*node(nelx,mid+1)+1]=-0.25

xc=np.maximum(rho_arr,1e-3)
ii=np.concatenate([edofs[:,ei] for ei in range(8) for _ in range(8)])
jj=np.concatenate([edofs[:,ej] for _ in range(8) for ej in range(8)])
ks=Emin+xc**penal*(E0-Emin)
kvals=np.array([ks*KE[ei,ej] for ei in range(8) for ej in range(8)]).ravel()
K=sp.csr_matrix((kvals,(ii,jj)),shape=(ndof,ndof))
u_sp=np.zeros(ndof); u_sp[free]=spla.spsolve(K[free,:][:,free].tocsc(),F[free])
C_scipy = float(F@u_sp)
print(f"scipy:    C={C_scipy:.6f}")

# Scipy sensitivity
Ue=u_sp[edofs]  # (N,8)
dc_scipy_raw=-(penal*xc**(penal-1)*(E0-Emin))*np.einsum('ni,ij,nj->n',Ue,KE,Ue)
# scipy dc is integrated (not averaged): divide by element area to get density
h_x = lx/nelx
dc_scipy = dc_scipy_raw / (h_x * h_y)   # normalize by element area
print(f"scipy:    dc min={dc_scipy.min():.4e}  max={dc_scipy.max():.4e}")

# ── Reorder: dolfinx cells vs scipy cells ──────────────────────────────
# dolfinx cell i has centroid (cx_arr[i], cy_arr[i])
# scipy cell e=ex*nely+ey has centroid ((ex+0.5)*h_x, (ey+0.5)*h_y)
# Map scipy ordering to dolfinx ordering
def scipy_cell_centroid(e):
    ex, ey = e//nely, e%nely
    return (ex+0.5)*h_x, (ey+0.5)*h_y

# Build mapping: for each dolfinx cell, find matching scipy cell
scipy_to_dolfinx = np.zeros(nelx*nely, dtype=int)
for e in range(nelx*nely):
    sx, sy = scipy_cell_centroid(e)
    # Find dolfinx cell with matching centroid
    dists = (cx_arr - sx)**2 + (cy_arr - sy)**2
    scipy_to_dolfinx[e] = np.argmin(dists)

# Reorder scipy dc to match dolfinx order
dc_scipy_dof = np.zeros(nelx*nely)
for e in range(nelx*nely):
    dc_scipy_dof[scipy_to_dolfinx[e]] = dc_scipy[e]

print(f"\nElement-wise comparison (dolfinx vs reordered scipy dc):")
ratio = dc_dolfinx / np.where(np.abs(dc_scipy_dof) > 1e-30, dc_scipy_dof, 1e-30)
print(f"  ratio min={ratio.min():.4f}  max={ratio.max():.4f}  mean={ratio.mean():.4f}")
print(f"  max abs diff = {np.max(np.abs(dc_dolfinx - dc_scipy_dof)):.4e}")
print(f"  relative error = {np.max(np.abs(dc_dolfinx - dc_scipy_dof)/np.abs(dc_scipy_dof+1e-30)):.4e}")

# Check a few specific boundary elements
print("\nBoundary element check (left edge, x≈0):")
left_mask = cx_arr < h_x + 1e-6
print(f"  dolfinx dc at left: mean={dc_dolfinx[left_mask].mean():.4e}  min={dc_dolfinx[left_mask].min():.4e}")
print(f"  scipy dc at left:   mean={dc_scipy_dof[left_mask].mean():.4e}  min={dc_scipy_dof[left_mask].min():.4e}")
print(f"  scipy C = {C_scipy:.6f}, dolfinx C = {C_dolfinx:.6f}")
print(f"  C ratio = {C_dolfinx/C_scipy:.6f}")
