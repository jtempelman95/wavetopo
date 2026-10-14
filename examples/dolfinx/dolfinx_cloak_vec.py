"""
Elastic cloak on a CONFORMING mesh, IN-PLANE (vector) orthotropic physics --
the faithful dolfinx port (same model as examples/dolfinx_lens.py).  The void is
a real circular hole cut from the mesh (traction-free boundary).  The undisturbed
reference is solved on a separate hole-free mesh and transferred by non-matching
interpolation.  We design the fiber orientation in an annular shell around the
hole to restore the downstream field, and draw the fiber toolpaths wrapping the
void.

    .../dolfinx_complex/bin/python3 examples/dolfinx_cloak_vec.py
"""
import numpy as np
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from dolfinx import fem

from wavetopo.dolfinx_mesh import rect_mesh
from wavetopo.dolfinx_elastic import ElasticWave
from wavetopo.dolfinx_wave import cell_filter
from wavetopo.dolfinx_viz import plot_toolpaths
from wavetopo.cfrp_optimizer import MMA

Lx, Ly = 4.0, 3.0
CX, CY, R = 2.0, 1.5, 0.42
OMEGA = 28.0
ETA = 0.04
R_SHELL = 1.5


def setup(ew):
    m = 0.6
    ew.set_sponge(lambda x, y: np.maximum(0, np.maximum.reduce([
        (x-(Lx-m))/m, (m-y)/m, (y-(Ly-m))/m]))**2 * 80.0)
    ew.set_source(lambda x, y: 30.0*np.exp(-80*(x-0.1)**2)*np.ones_like(y))


def centroids_dg0(ew):
    dom = ew.domain; dom.topology.create_connectivity(2, 0)
    xy = dom.geometry.x[:, :2]; conn = dom.topology.connectivity(2, 0)
    n = dom.topology.index_map(2).size_local
    c = np.zeros((ew.theta.x.array.size, 2))
    for e in range(n):
        c[ew.S.dofmap.cell_dofs(e)[0]] = xy[conn.links(e)].mean(0)
    return c


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


def main():
    mh = rect_mesh(Lx, Ly, 0.05, hole=(CX, CY, R))
    mf = rect_mesh(Lx, Ly, 0.05, hole=None)
    ew = ElasticWave(mh, omega=OMEGA, Ef1=131., Ef2=9., G12=5., nu12=0.27,
                     rho=1.6, eta=ETA); setup(ew)
    ref = ElasticWave(mf, omega=OMEGA, Ef1=131., Ef2=9., G12=5., nu12=0.27,
                      rho=1.6, eta=ETA); setup(ref)

    # reference (undisturbed) field on the hole-free mesh -> holed mesh
    ref.set_theta(np.zeros(ref.theta.x.array.size)); ref.solve()
    idata = fem.create_nonmatching_meshes_interpolation_data(
        ew.W.mesh._cpp_object, ew.W.element, ref.W.mesh._cpp_object, 1e-8)
    ew.Uref.interpolate(ref.U, nmm_interpolation_data=idata)
    ew.Uref.x.scatter_forward()

    ew.set_region(lambda x, y: ((x > CX+0.55) & (x < Lx-0.7) &
                                (np.abs(y-CY) < 0.9)).astype(float))
    Nt = ew.theta.x.array.size
    ew.set_theta(np.zeros(Nt)); ew.solve()
    J0 = ew.cloak_mismatch(); tri, munc = field_tri(ew)

    cent = centroids_dg0(ew)
    d = np.hypot(cent[:, 0]-CX, cent[:, 1]-CY)
    mask = (d > R) & (d < R_SHELL)
    P = cell_filter(cent[mask], R=0.12)
    print(f"holed {Nt} cells, {mask.sum()} shell design cells; J0={J0:.3e}")

    xm = np.zeros(int(mask.sum()))
    mma = MMA(np.full(xm.size, -np.pi/2), np.full(xm.size, np.pi/2), move=0.10)
    th = np.zeros(Nt); best = (J0, th.copy())
    for it in range(75):
        th[mask] = P @ xm; ew.set_theta(th); ew.solve()
        J = ew.cloak_mismatch()
        if J < best[0]:
            best = (J, th.copy())
        g = P.T @ (ew.cloak_grad()[mask] / max(J, 1e-30))
        xm = mma.update(xm, g)
        if it % 10 == 0:
            print(f"[{it:3d}] J={J:.3e}  ({J0/J:.1f}x)")
    thopt = best[1]
    ew.set_theta(thopt); ew.solve(); J1 = ew.cloak_mismatch()
    tri, mcl = field_tri(ew); red = J0/J1
    print(f"vector conforming cloak: J0={J0:.3e} -> J1={J1:.3e}  reduction={red:.1f}x")

    trf, mref = field_tri(ref)
    vmax = np.percentile(mref, 99.0)
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    panels = [(ax[0], trf, mref, "reference (hole-free plate)"),
              (ax[1], tri, munc, f"bare conforming void\nshadow $J_0={J0:.2e}$"),
              (ax[2], tri, mcl, f"orientation-cloaked\n$J_1={J1:.2e}$  ({red:.0f}$\\times$ less)")]
    for a_, t_, m_, ttl in panels:
        tp = a_.tripcolor(t_, m_, cmap="magma", vmax=vmax, shading="gouraud")
        thc = np.linspace(0, 2*np.pi, 80)
        if a_ is not ax[0]:
            a_.fill(CX+R*np.cos(thc), CY+R*np.sin(thc), color="white",
                    ec="0.4", lw=1.0, zorder=5)
        a_.add_patch(plt.Rectangle((CX+0.55, CY-0.9), (Lx-0.7)-(CX+0.55), 1.8,
                     ec="lime", fc="none", lw=1.2, ls="--"))
        a_.set_aspect("equal"); a_.set_xlim(0, Lx); a_.set_ylim(0, Ly)
        a_.set_title(ttl); plt.colorbar(tp, ax=a_, fraction=0.03)
    plot_toolpaths(ax[3], cent, thopt, (0, Lx), (0, Ly), hole=(CX, CY, R),
                   nseed=30, lw=0.8)
    ax[3].add_patch(plt.Circle((CX, CY), R_SHELL, ec="tab:blue", fc="none",
                    lw=1.2, ls=":"))
    ax[3].set_title("optimized fiber toolpaths\n(wrapping the void)")
    fig.suptitle("dolfinx in-plane (vector) elastic cloak on a conforming mesh "
                 f"(real traction-free hole, $\\omega$={OMEGA:.0f}): "
                 f"{red:.0f}$\\times$ shadow reduction", y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig("results/dolfinx_cloak_vec.png", dpi=140, bbox_inches="tight")
    print("saved results/dolfinx_cloak_vec.png")


if __name__ == "__main__":
    main()
