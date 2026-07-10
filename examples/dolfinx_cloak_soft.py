"""
Elastic cloak of a soft (SIMP) void -- IN-PLANE (vector) orthotropic physics,
the faithful dolfinx port of examples/wave_cloak.py.  A fiber-composite plate
contains a void (density z=0) that scatters an incident plane wave.  Optimizing
the fiber orientation in a shell around the void makes the far field match the
void-free reference -- the void becomes ~invisible.  Single material; only the
curved fiber toolpaths do the work.

    .../dolfinx_complex/bin/python3 examples/dolfinx_cloak_soft.py
"""
import numpy as np
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from topoopt.dolfinx_elastic import ElasticWave
from topoopt.dolfinx_wave import support_map
from topoopt.dolfinx_viz import plot_toolpaths
from topoopt.cfrp_optimizer import MMA

Lx, Ly = 6.0, 4.0
CX, CY = Lx/2, Ly/2
RV, RC = 0.55, 1.4                 # void radius, cloak-shell / observation radius
OMEGA, ETA = 36.0, 0.03


def field_tri(ew):
    dom = ew.domain
    V = fem.functionspace(dom, ("Lagrange", 1))
    mag = fem.Function(V)
    u2 = sum(ew.U[i]*ew.U[i] for i in range(4))
    mag.interpolate(fem.Expression(ufl.sqrt(u2), V.element.interpolation_points()))
    coords = V.tabulate_dof_coordinates()[:, :2]
    dom.topology.create_connectivity(2, 0)
    n = dom.topology.index_map(2).size_local
    tris = np.array([V.dofmap.cell_dofs(e) for e in range(n)])
    return mtri.Triangulation(coords[:, 0], coords[:, 1], tris), mag.x.array


def centroids(ew):
    dom = ew.domain; dom.topology.create_connectivity(2, 0)
    xy = dom.geometry.x[:, :2]; conn = dom.topology.connectivity(2, 0)
    n = dom.topology.index_map(2).size_local
    c = np.zeros((ew.theta.x.array.size, 2))
    for e in range(n):
        c[ew.S.dofmap.cell_dofs(e)[0]] = xy[conn.links(e)].mean(0)
    return c


def main():
    dom = dmesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]],
                                 [150, 100], dmesh.CellType.triangle)
    ew = ElasticWave(dom, omega=OMEGA, Ef1=131., Ef2=9., G12=5., nu12=0.27,
                     rho=1.6, rho_m=1.2, eta=ETA, penal=3.0)
    m = 0.55
    ew.set_sponge(lambda x, y: np.maximum(0, np.maximum.reduce([
        (x-(Lx-m))/m, (m-y)/m, (y-(Ly-m))/m]))**2 * 70.0)
    ew.set_source(lambda x, y: 30.0*np.exp(-80*(x-0.1)**2)*np.ones_like(y))

    # reference field (no void) -> store
    ew.set_zfac(lambda x, y: np.ones_like(x))
    ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve(); ew.store_reference()
    trf, mref = field_tri(ew)

    # observation region: far field outside the shell, inside the non-sponge box
    ew.set_region(lambda x, y: ((np.hypot(x-CX, y-CY) > RC) & (x > m) &
                                (x < Lx-m) & (y > m) & (y < Ly-m)).astype(float))
    # soft void
    ew.set_zfac(lambda x, y: 1.0 - (np.hypot(x-CX, y-CY) < RV).astype(float))
    ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve()
    J0 = ew.cloak_mismatch(); tri, munc = field_tri(ew)

    # design orientation on CS-RBF support points within the cloak shell
    cent = centroids(ew)
    B, supp = support_map(cent, (0, Lx), (0, Ly), spacing=0.28, r=0.6)
    active = np.hypot(supp[:, 0]-CX, supp[:, 1]-CY) < RC + 0.3
    lo = np.where(active, -np.pi/2, 0.0); hi = np.where(active, np.pi/2, 0.0)
    print(f"{int(active.sum())}/{B.shape[1]} active support pts; uncloaked J0={J0:.3e}")

    mma = MMA(lo, hi, move=0.15)
    x = np.zeros(B.shape[1]); best = (J0, np.zeros(ew.theta.x.array.size))
    for it in range(60):
        th = B @ x; ew.set_theta(th); ew.solve()
        J = ew.cloak_mismatch()
        if J < best[0]:
            best = (J, th.copy())
        g = B.T @ ew.cloak_grad()
        x = mma.update(x, g)
        if it % 10 == 0 or it == 59:
            print(f"[{it:3d}] scatter J={J:.3e}  reduction={J0/J:.1f}x")
    thopt = best[1]
    ew.set_theta(thopt); ew.solve(); J1 = ew.cloak_mismatch()
    tri, mcl = field_tri(ew); red = J0/J1
    print(f"vector soft-void cloak: J0={J0:.3e} -> J1={J1:.3e}  reduction={red:.1f}x")

    vmax = np.percentile(mref, 99.5)
    fig, ax = plt.subplots(1, 4, figsize=(24, 5))
    tc = np.linspace(0, 2*np.pi, 100)
    for a_, t_, m_, ttl in [(ax[0], trf, mref, "reference (no void)"),
                            (ax[1], tri, munc, f"uncloaked void\nscatter $J_0={J0:.2e}$"),
                            (ax[2], tri, mcl, f"orientation-cloaked\n$J_1={J1:.2e}$  "
                             f"({red:.0f}$\\times$ less)")]:
        tp = a_.tripcolor(t_, m_, cmap="magma", vmax=vmax, shading="gouraud")
        a_.plot(CX+RV*np.cos(tc), CY+RV*np.sin(tc), "c-", lw=1.0)
        a_.plot(CX+RC*np.cos(tc), CY+RC*np.sin(tc), "w:", lw=0.8)
        a_.set_aspect("equal"); a_.set_xlim(0, Lx); a_.set_ylim(0, Ly)
        a_.set_title(ttl); plt.colorbar(tp, ax=a_, fraction=0.03)
    plot_toolpaths(ax[3], cent, thopt, (0, Lx), (0, Ly), hole=(CX, CY, RV),
                   nseed=34, lw=0.7)
    ax[3].add_patch(plt.Circle((CX, CY), RC, ec="tab:blue", fc="none",
                    lw=1.2, ls=":"))
    ax[3].set_title("optimized fiber toolpaths\n(curving around the void)")
    fig.suptitle("dolfinx in-plane (vector) elastic cloak of a soft void "
                 f"($\\omega$={OMEGA:.0f}): {red:.0f}$\\times$ scatter reduction, "
                 "far field restored", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig("results/dolfinx_cloak_soft.png", dpi=140, bbox_inches="tight")
    print("saved results/dolfinx_cloak_soft.png")


if __name__ == "__main__":
    main()
