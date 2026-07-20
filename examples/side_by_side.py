"""
Run scipy SIMP and dolfinx cantilever side by side.
Compare xPhys arrays at each iteration to find where they diverge.
Uses a small 20x10 mesh for speed.
"""
import sys; sys.path.insert(0, '/home/jrt/wavetopo')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from wavetopo import ElasticityProblem
from wavetopo.filters import ConeFilter

nelx, nely = 20, 10
lx, ly = 2.0, 1.0
volfrac, penal = 0.40, 3.0
E0, Emin, nu = 1.0, 1e-9, 0.3
move = 0.2
rmin_elem = 1.5        # filter radius in element units
h = ly / nely
rmin_phys = rmin_elem * h

# ── Build scipy objects ──────────────────────────────────────────────────
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
ex_all=np.repeat(np.arange(nelx),nely); ey_all=np.tile(np.arange(nely),nelx)
n1=node(ex_all,ey_all); n2=node(ex_all+1,ey_all); n3=node(ex_all+1,ey_all+1); n4=node(ex_all,ey_all+1)
edofs=np.column_stack([2*n1,2*n1+1,2*n2,2*n2+1,2*n3,2*n3+1,2*n4,2*n4+1])
fixed=np.array(sorted({d for ey in range(nely+1) for n in [node(0,ey)] for d in [2*n,2*n+1]}))
free=np.setdiff1d(np.arange(ndof),fixed)
ii=np.concatenate([edofs[:,ei] for ei in range(8) for _ in range(8)])
jj=np.concatenate([edofs[:,ej] for _ in range(8) for ej in range(8)])

# scipy load: 3-node to match dolfinx
mid=nely//2
F_sp=np.zeros(ndof); F_sp[2*node(nelx,mid-1)+1]=-0.25; F_sp[2*node(nelx,mid)+1]=-0.50; F_sp[2*node(nelx,mid+1)+1]=-0.25

# scipy filter
cx=ex_all+0.5; cy=ey_all+0.5
pairs=cKDTree(np.column_stack([cx,cy])).query_pairs(rmin_elem,output_type="ndarray")
rows,cols,vals=list(range(nelx*nely)),list(range(nelx*nely)),[rmin_elem]*(nelx*nely)
for i,j in pairs:
    d=np.hypot(cx[i]-cx[j],cy[i]-cy[j]); rows+=[i,j];cols+=[j,i];vals+=[rmin_elem-d,rmin_elem-d]
H=sp.csr_matrix((vals,(rows,cols)),shape=(nelx*nely,nelx*nely)); Hs=np.asarray(H.sum(1)).ravel()
sp_filt=lambda r: (H@r)/Hs

# ── Build dolfinx objects ────────────────────────────────────────────────
fem_prob = ElasticityProblem(nx=nelx, ny=nely, lx=lx, ly=ly, penal=penal)
dx_filt  = ConeFilter(fem_prob.DG0, r_min=rmin_phys)

# ── Build centroid-index mapping scipy <-> dolfinx ───────────────────────
from dolfinx import fem as dfem
import ufl
xcoord=ufl.SpatialCoordinate(fem_prob.domain)
pts=fem_prob.DG0.element.interpolation_points()
cx_fn=dfem.Function(fem_prob.DG0); cy_fn=dfem.Function(fem_prob.DG0)
cx_fn.interpolate(dfem.Expression(xcoord[0],pts))
cy_fn.interpolate(dfem.Expression(xcoord[1],pts))
cx_arr=cx_fn.x.array; cy_arr=cy_fn.x.array

# scipy cell e centroid: ((e//nely+0.5)*h_x, (e%nely+0.5)*h_y)
h_x = lx/nelx; h_y = ly/nely
sp_cx = (ex_all+0.5)*h_x; sp_cy = (ey_all+0.5)*h_y

# mapping: scipy cell e -> dolfinx cell index
sp_to_dx = np.zeros(nelx*nely, dtype=int)
for e in range(nelx*nely):
    dists = (cx_arr-sp_cx[e])**2+(cy_arr-sp_cy[e])**2
    sp_to_dx[e] = np.argmin(dists)
dx_to_sp = np.argsort(sp_to_dx)  # dolfinx -> scipy

# ── OC step ──────────────────────────────────────────────────────────────
def oc_step(xPhys, dc_f, vf, move):
    dc_abs=np.maximum(-dc_f,1e-30)
    l1,l2=1e-9,1e9
    while (l2-l1)/(l1+l2)>1e-6:
        lm=0.5*(l1+l2)
        xn=np.clip(xPhys*np.sqrt(dc_abs/lm),np.maximum(0.001,xPhys-move),np.minimum(1.,xPhys+move))
        if xn.mean()>vf: l1=lm
        else: l2=lm
    return xn

# ── Run both ──────────────────────────────────────────────────────────────
sp_x=np.full(nelx*nely,volfrac); sp_xP=sp_filt(sp_x)
dx_x=np.full(nelx*nely,volfrac); dx_xP=dx_filt.apply(dx_x)

print(f"{'Iter':>4} | {'sp_C':>10} | {'dx_C':>10} | {'max_xP_diff':>12} | {'max_dc_diff':>12}")
print("-"*60)

for it in range(1, 11):
    # scipy step
    xc_sp=np.maximum(sp_xP,1e-3)
    ks=Emin+xc_sp**penal*(E0-Emin)
    kv=np.array([ks*KE[ei,ej] for ei in range(8) for ej in range(8)]).ravel()
    K=sp.csr_matrix((kv,(ii,jj)),shape=(ndof,ndof))
    u_sp=np.zeros(ndof); u_sp[free]=spla.spsolve(K[free,:][:,free].tocsc(),F_sp[free])
    C_sp=float(F_sp@u_sp)
    Ue_sp=u_sp[edofs]
    dc_sp=-(penal*xc_sp**(penal-1)*(E0-Emin))*np.einsum('ni,ij,nj->n',Ue_sp,KE,Ue_sp)
    dc_f_sp=(H@(xc_sp*dc_sp))/Hs/np.maximum(1e-3,xc_sp)
    sp_xn=oc_step(sp_xP, dc_f_sp, volfrac, move)
    sp_x=sp_xn; sp_xP=sp_filt(sp_x)

    # dolfinx step (reorder to dolfinx-space for FE, then reorder back)
    # xP_dx is already in dolfinx order
    xc_dx=np.maximum(dx_xP,1e-3)
    fem_prob.update_density(xc_dx)
    u_dx, C_dx = fem_prob.solve()
    dc_raw_dx=fem_prob.compliance_sensitivity(u_dx)
    dc_f_dx=dx_filt.apply(xc_dx*dc_raw_dx)/np.maximum(1e-3,xc_dx)
    dx_xn=oc_step(dx_xP, dc_f_dx, volfrac, move)
    dx_x=dx_xn; dx_xP=dx_filt.apply(dx_x)

    # Compare: reorder scipy arrays to dolfinx indexing for comparison
    sp_xP_in_dx = sp_xP[dx_to_sp]   # scipy xPhys reordered to dolfinx cell order
    sp_dc_in_dx = dc_f_sp[dx_to_sp]

    max_xP = np.max(np.abs(dx_xP - sp_xP_in_dx))
    max_dc = np.max(np.abs(dc_f_dx - sp_dc_in_dx))
    print(f"{it:4d} | {C_sp:10.4f} | {C_dx:10.4f} | {max_xP:12.6f} | {max_dc:12.4e}")

print("\nFinal xPhys check:")
print(f"  scipy xPhys: min={sp_xP.min():.4f} max={sp_xP.max():.4f} mean={sp_xP.mean():.4f}")
print(f"  dolfinx xPhys: min={dx_xP.min():.4f} max={dx_xP.max():.4f} mean={dx_xP.mean():.4f}")
print(f"  max diff (reordered): {max_xP:.6f}")
