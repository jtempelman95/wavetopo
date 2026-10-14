"""
Elastic cloak on a CONFORMING mesh: the void is a real circular hole cut from the
mesh (traction-free boundary), not a soft SIMP density hole on a structured grid.

The undisturbed reference field is solved on a separate hole-free mesh of the same
plate and transferred to the holed mesh by non-matching interpolation.  We then
design the fiber orientation in an annular shell around the hole to restore the
downstream field (minimize the observation-window mismatch), and draw the
optimized fiber toolpaths wrapping the void.

    .../dolfinx_complex/bin/python3 examples/dolfinx_cloak_conforming.py
"""
import numpy as np
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from dolfinx import fem

from wavetopo.dolfinx_mesh import rect_mesh
from wavetopo.dolfinx_wave import WaveControl, cell_filter
from wavetopo.dolfinx_viz import plot_toolpaths
from wavetopo.cfrp_optimizer import MMA

Lx, Ly = 4.0, 3.0
CX, CY, R = 2.0, 1.5, 0.45
OMEGA, MUL, MUT = 24.0, 6.0, 2.0
R_SHELL = 1.15                       # design annulus outer radius


def setup(wc):
    m = 0.55
    wc.set_sponge(lambda x, y: np.maximum(0, np.maximum.reduce([
        (x-(Lx-m))/m, (m-y)/m, (y-(Ly-m))/m]))**2 * 70.0)
    wc.set_source(lambda x, y: np.exp(-40*(x-0.25)**2)*np.ones_like(y))


def centroids_dg0(wc):
    dom = wc.domain
    dom.topology.create_connectivity(2, 0)
    xy = dom.geometry.x[:, :2]
    conn = dom.topology.connectivity(2, 0)
    ncell = dom.topology.index_map(2).size_local
    cent = np.zeros((wc.theta.x.array.size, 2))
    for c in range(ncell):
        verts = conn.links(c)
        dof = wc.S.dofmap.cell_dofs(c)[0]
        cent[dof] = xy[verts].mean(0)
    return cent


def field_tri(wcobj):
    """A matplotlib Triangulation and |u| nodal values that are *consistently*
    ordered (via a scalar CG1 space), robust to gmsh's geometry/dof reordering."""
    dom = wcobj.domain
    V = fem.functionspace(dom, ("Lagrange", 1))
    ur, ui = wcobj.U[0], wcobj.U[1]
    mag = fem.Function(V)
    mag.interpolate(fem.Expression(ufl.sqrt(ur*ur + ui*ui),
                                   V.element.interpolation_points()))
    coords = V.tabulate_dof_coordinates()[:, :2]
    dom.topology.create_connectivity(2, 0)
    ncell = dom.topology.index_map(2).size_local
    tris = np.array([V.dofmap.cell_dofs(c) for c in range(ncell)])
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)
    return tri, mag.x.array


def main():
    # ---- meshes: conforming hole + hole-free reference ------------------ #
    mh = rect_mesh(Lx, Ly, 0.045, hole=(CX, CY, R))
    mf = rect_mesh(Lx, Ly, 0.045, hole=None)
    wc = WaveControl(mh, omega=OMEGA, muL=MUL, muT=MUT, rho=1.6); setup(wc)
    ref = WaveControl(mf, omega=OMEGA, muL=MUL, muT=MUT, rho=1.6); setup(ref)

    # ---- reference (undisturbed) field on the hole-free mesh ------------ #
    ref.set_theta(np.zeros(ref.theta.x.array.size)); ref.solve()
    idata = fem.create_nonmatching_meshes_interpolation_data(
        wc.W.mesh._cpp_object, wc.W.element, ref.W.mesh._cpp_object, 1e-8)
    wc.Uref.interpolate(ref.U, nmm_interpolation_data=idata)
    wc.Uref.x.scatter_forward()

    # observation window downstream of the hole
    wc.set_region(lambda x, y: ((x > CX+0.55) & (x < Lx-0.7) &
                                (np.abs(y-CY) < 0.9)).astype(float))

    # ---- bare void (straight fibers) ------------------------------------ #
    Nt = wc.theta.x.array.size
    wc.set_theta(np.zeros(Nt)); wc.solve()
    J0 = wc.cloak_mismatch(); tri, munc = field_tri(wc)

    # ---- design mask: annular shell around the hole --------------------- #
    cent = centroids_dg0(wc)
    d = np.hypot(cent[:, 0]-CX, cent[:, 1]-CY)
    mask = (d > R) & (d < R_SHELL)
    print(f"holed mesh: {Nt} cells, {mask.sum()} design cells in the shell; "
          f"bare-void J0={J0:.3e}")

    # ---- optimize orientation in the shell (log-objective MMA), with a
    #      neighborhood filter so the fiber toolpaths come out smooth --------- #
    P = cell_filter(cent[mask], R=0.11)
    xm = np.zeros(int(mask.sum()))
    mma = MMA(np.full(xm.size, -np.pi/2), np.full(xm.size, np.pi/2), move=0.10)
    th = np.zeros(Nt); best = (J0, th.copy())
    for it in range(70):
        th[mask] = P @ xm; wc.set_theta(th); wc.solve()
        J = wc.cloak_mismatch()
        if J < best[0]:
            best = (J, th.copy())
        g = P.T @ (wc.cloak_grad()[mask] / max(J, 1e-30))
        xm = mma.update(xm, g)
    thopt = best[1]
    wc.set_theta(thopt); wc.solve(); J1 = wc.cloak_mismatch()
    tri, mcl = field_tri(wc)
    red = J0/J1
    print(f"conforming cloak: J0={J0:.3e} -> J1={J1:.3e}  reduction={red:.1f}x")

    # ---- figure: fields + toolpaths ------------------------------------- #
    trf, mref = field_tri(ref)
    vmax = np.percentile(mref, 99.0)
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    obsrect = plt.Rectangle((CX+0.55, CY-0.9), (Lx-0.7)-(CX+0.55), 1.8,
                            ec="lime", fc="none", lw=1.4, ls="--")
    panels = [(ax[0], trf, mref, "reference (hole-free plate)"),
              (ax[1], tri, munc, f"bare conforming void\nshadow $J_0={J0:.2e}$"),
              (ax[2], tri, mcl, f"orientation-cloaked\n$J_1={J1:.2e}$  "
               f"({red:.0f}$\\times$ less)")]
    for a_, t_, m_, ttl in panels:
        tp = a_.tripcolor(t_, m_, cmap="magma", vmax=vmax, shading="gouraud")
        thc = np.linspace(0, 2*np.pi, 80)
        if a_ is not ax[0]:
            a_.fill(CX+R*np.cos(thc), CY+R*np.sin(thc), color="white",
                    ec="0.4", lw=1.0, zorder=5)
        a_.add_patch(plt.Rectangle((CX+0.55, CY-0.9), (Lx-0.7)-(CX+0.55), 1.8,
                     ec="lime", fc="none", lw=1.2, ls="--"))
        a_.set_aspect("equal"); a_.set_title(ttl)
        a_.set_xlim(0, Lx); a_.set_ylim(0, Ly)
        plt.colorbar(tp, ax=a_, fraction=0.03)
    # toolpaths of the cloak (shell only shown; rest is straight)
    plot_toolpaths(ax[3], cent, thopt, (0, Lx), (0, Ly),
                   hole=(CX, CY, R), nseed=30, lw=0.8)
    ax[3].add_patch(plt.Circle((CX, CY), R_SHELL, ec="tab:blue", fc="none",
                    lw=1.2, ls=":"))
    ax[3].set_title("optimized fiber toolpaths\n(wrapping the void)")
    fig.suptitle(f"Elastic cloak on a conforming mesh (real traction-free hole, "
                 f"$\\mu_L/\\mu_T={MUL/MUT:.0f}$, $\\omega={OMEGA:.0f}$): "
                 f"{red:.0f}$\\times$ shadow reduction", y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig("results/dolfinx_cloak_conforming.png", dpi=140, bbox_inches="tight")
    print("saved results/dolfinx_cloak_conforming.png")


if __name__ == "__main__":
    main()
