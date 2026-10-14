"""Test whether rmin=2.5 causes scatter vs rmin=1.5 in scipy, both 80x40 mesh."""
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

nelx, nely = 80, 40
volfrac, penal = 0.40, 3.0
E0, Emin, nu = 1.0, 1e-9, 0.3
max_iter, move = 100, 0.2

def lk():
    k=[0.5-nu/6,0.125+nu/8,-0.25-nu/12,-0.125+3*nu/8,-0.25+nu/12,-0.125-nu/8,nu/6,0.125-3*nu/8]
    return E0/(1-nu**2)*np.array([[k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
     [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],[k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
     [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],[k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
     [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],[k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
     [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
KE = lk()
ndof=2*(nelx+1)*(nely+1)
def node(ex,ey): return ey+ex*(nely+1)
ex_all=np.repeat(np.arange(nelx),nely); ey_all=np.tile(np.arange(nely),nelx)
n1=node(ex_all,ey_all);n2=node(ex_all+1,ey_all);n3=node(ex_all+1,ey_all+1);n4=node(ex_all,ey_all+1)
edofs=np.column_stack([2*n1,2*n1+1,2*n2,2*n2+1,2*n3,2*n3+1,2*n4,2*n4+1])
fixed=np.array(sorted({d for ey in range(nely+1) for n in [node(0,ey)] for d in [2*n,2*n+1]}))
free=np.setdiff1d(np.arange(ndof),fixed)
ii=np.concatenate([edofs[:,ei] for ei in range(8) for _ in range(8)])
jj=np.concatenate([edofs[:,ej] for _ in range(8) for ej in range(8)])
mid=nely//2
F1=np.zeros(ndof); F1[2*node(nelx,mid)+1]=-1.0   # 1-node load

def run(rmin_val, label):
    cx=ex_all+0.5; cy=ey_all+0.5
    pairs=cKDTree(np.column_stack([cx,cy])).query_pairs(rmin_val,output_type="ndarray")
    rows,cols,vals=list(range(nelx*nely)),list(range(nelx*nely)),[rmin_val]*(nelx*nely)
    for i,j in pairs:
        d=np.hypot(cx[i]-cx[j],cy[i]-cy[j]); rows+=[i,j];cols+=[j,i];vals+=[rmin_val-d,rmin_val-d]
    H=sp.csr_matrix((vals,(rows,cols)),shape=(nelx*nely,nelx*nely)); Hs=np.asarray(H.sum(1)).ravel()
    filt=lambda r:(H@r)/Hs

    x=np.full(nelx*nely,volfrac); xP=filt(x); C_last=0
    for it in range(1,max_iter+1):
        xc=np.maximum(xP,1e-3)
        ks=Emin+xc**penal*(E0-Emin)
        kv=np.array([ks*KE[ei,ej] for ei in range(8) for ej in range(8)]).ravel()
        K=sp.csr_matrix((kv,(ii,jj)),shape=(ndof,ndof))
        u=np.zeros(ndof); u[free]=spla.spsolve(K[free,:][:,free].tocsc(),F1[free])
        C=float(F1@u); C_last=C
        Ue=u[edofs]
        dc=-(penal*xc**(penal-1)*(E0-Emin))*np.einsum('ni,ij,nj->n',Ue,KE,Ue)
        dc_f=(H@(xc*dc))/Hs/np.maximum(1e-3,xc)
        l1,l2=1e-9,1e9
        while(l2-l1)/(l1+l2)>1e-6:
            lm=0.5*(l1+l2)
            xn=np.clip(xP*np.sqrt(np.maximum(-dc_f,1e-30)/lm),np.maximum(0.001,xP-move),np.minimum(1.,xP+move))
            if xn.mean()>volfrac: l1=lm
            else: l2=lm
        x=xn; xP=filt(x)
        if it%20==0: print(f"  [{label}] iter {it:3d}  C={C:.4f}")
    return xP, C_last

print("=== rmin=1.5 (element units) ===")
rho15, C15 = run(1.5, "rmin=1.5")
print("=== rmin=2.5 (element units) ===")
rho25, C25 = run(2.5, "rmin=2.5")

fig,axes=plt.subplots(2,2,figsize=(16,7))
for row,(rho,C,rm) in enumerate([(rho15,C15,"1.5"),(rho25,C25,"2.5")]):
    g=rho.reshape(nelx,nely).T
    for col,(dat,suf) in enumerate([(g,"continuous"),(g>=0.5,"thresholded")]):
        axes[row][col].imshow(dat.astype(float),origin="lower",cmap="gray_r",vmin=0,vmax=1,aspect="equal")
        axes[row][col].set_title(f"rmin={rm} C={C:.2f} — {suf}",fontsize=9)
plt.tight_layout(); plt.savefig(str(Path(__file__).parent / "rmin_test.png"),dpi=130,bbox_inches="tight")
print(f"\nrmin=1.5: C={C15:.4f}   rmin=2.5: C={C25:.4f}")
print(f"Saved {Path(__file__).parent / 'rmin_test.png'}")
