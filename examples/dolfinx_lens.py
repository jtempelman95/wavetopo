"""
Curvilinear-fiber elastic-wave lens in dolfinx -- IN-PLANE (vector) orthotropic
elastodynamics, the faithful dolfinx port of examples/wave_lens.py.  Same
material (Ef1=131, Ef2=9, G12=5), same omega, same CS-RBF smooth orientation,
same absorbing sponge and left-edge drive.  Only the FIBER ORIENTATION is
optimized to focus the wave at a target box on the right.

    .../dolfinx_complex/bin/python3 examples/dolfinx_lens.py           # optimize
    .../dolfinx_complex/bin/python3 examples/dolfinx_lens.py fdcheck   # verify grad
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
import ufl

from topoopt.dolfinx_elastic import ElasticWave
from topoopt.dolfinx_wave import support_map
from topoopt.dolfinx_viz import plot_toolpaths
from topoopt.cfrp_optimizer import MMA

Lx, Ly = 4.0, 3.0
FOCUS = (0.78 * Lx, Ly / 2, 0.18)
OMEGA = 40.0


def make(nx=130, omega=OMEGA, eta=0.02):
    dom = dmesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]],
                                 [nx, int(nx*Ly/Lx)], dmesh.CellType.triangle)
    ew = ElasticWave(dom, omega=omega, Ef1=131.0, Ef2=9.0, G12=5.0,
                     nu12=0.27, rho=1.6, eta=eta)
    m = 0.6
    ew.set_sponge(lambda x, y: np.maximum(0, np.maximum.reduce([
        (x-(Lx-m))/m, (m-y)/m, (y-(Ly-m))/m]))**2 * 80.0)
    # plane pressure wave from the left edge: x-force line source at x~0.1
    ew.set_source(lambda x, y: 30.0*np.exp(-80*(x-0.1)**2)*np.ones_like(y))
    fx, fy, fr = FOCUS
    ew.set_region(lambda x, y: ((np.abs(x-fx) < fr) & (np.abs(y-fy) < fr)).astype(float))
    return ew


def centroids(ew):
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


def fdcheck():
    ew = make(nx=50)
    Nt = ew.theta.x.array.size
    rng = np.random.default_rng(0); th = rng.uniform(-0.6, 0.6, Nt)
    ew.set_theta(th); ew.solve(); g = ew.focus_grad()
    h = 1e-6
    for c in rng.choice(Nt, 6, replace=False):
        tp = th.copy(); tp[c] += h; ew.set_theta(tp); ew.solve(); Ep = ew.focus_energy()
        tm = th.copy(); tm[c] -= h; ew.set_theta(tm); ew.solve(); Em = ew.focus_energy()
        fd = (Ep-Em)/(2*h)
        print(f"cell {c:5d}: adj={g[c]:+.5e} fd={fd:+.5e} "
              f"relerr={abs(fd-g[c])/(abs(fd)+1e-30):.2e}")


def run(save="results/dolfinx_lens.png", iters=70):
    ew = make()
    cent = centroids(ew)
    B, supp = support_map(cent, (0, Lx), (0, Ly), spacing=0.33, r=0.7)
    Md = B.shape[1]
    print(f"vector lens: {ew.W.dofmap.index_map.size_local*4} dofs, "
          f"{Md} CS-RBF support pts, omega={OMEGA}")
    ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve()
    E0, T0 = ew.focus_energy(), ew.total_energy()
    tri, m0 = field_tri(ew)

    mma = MMA(np.full(Md, -np.pi/2), np.full(Md, np.pi/2), move=0.10)
    x = np.zeros(Md); best = (E0, np.zeros(ew.theta.x.array.size)); hist = []
    for it in range(iters):
        th = B @ x; ew.set_theta(th); ew.solve()
        J = ew.focus_energy(); hist.append(J/E0)
        if J > best[0]:
            best = (J, th.copy())
        g = B.T @ ew.focus_grad()
        x = mma.update(x, -g)               # maximize -> minimize -J
        if it % 10 == 0 or it == iters-1:
            print(f"[{it:3d}] focus gain {J/E0:.1f}x")
    thopt = best[1]
    ew.set_theta(thopt); ew.solve()
    E1, T1 = ew.focus_energy(), ew.total_energy()
    _, m1 = field_tri(ew)
    gain, f0, f1 = E1/E0, E0/T0*100, E1/T1*100
    print(f"lens: E0={E0:.3e} E1={E1:.3e} gain={gain:.1f}x  "
          f"focus fraction {f0:.2f}% -> {f1:.2f}%")

    fx, fy, fr = FOCUS; vmax = np.percentile(m1, 99.5)
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    for a_, m, t in [(ax[0], m0, f"straight fibers (baseline)\nfocus fraction {f0:.2f}%"),
                     (ax[1], m1, f"optimized fiber lens\nfocus fraction {f1:.2f}%  "
                      f"gain {gain:.0f}$\\times$")]:
        tp = a_.tripcolor(tri, m, cmap="magma", vmax=vmax, shading="gouraud")
        a_.add_patch(plt.Rectangle((fx-fr, fy-fr), 2*fr, 2*fr, ec="cyan",
                                   fc="none", lw=1.5))
        a_.set_aspect("equal"); a_.set_xlim(0, Lx); a_.set_ylim(0, Ly)
        a_.set_title(t); plt.colorbar(tp, ax=a_, fraction=0.03)
    plot_toolpaths(ax[2], cent, thopt, (0, Lx), (0, Ly), nseed=20, lw=0.9)
    ax[2].add_patch(plt.Rectangle((fx-fr, fy-fr), 2*fr, 2*fr, ec="cyan",
                                  fc="none", lw=1.5, zorder=6))
    ax[2].set_title("optimized fiber toolpaths (the lens)")
    fig.suptitle("dolfinx in-plane (vector) orthotropic wave lens "
                 f"(Ef1/Ef2=14.6, orientation-only, $\\omega$={OMEGA:.0f})", y=1.02)
    plt.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight")
    print("saved", save)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "fdcheck": fdcheck}[cmd]()
