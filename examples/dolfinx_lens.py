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

from wavetopo.dolfinx_elastic import ElasticWave
from wavetopo.dolfinx_wave import support_map
from wavetopo.dolfinx_viz import (plot_toolpaths, plot_toolpaths_phase,
                                 plot_director_field)
from wavetopo.cfrp_optimizer import MMA

Lx, Ly = 4.0, 3.0
FOCUS = (0.78 * Lx, Ly / 2, 0.18)
OMEGA = 40.0


def focus_marker(ax, zorder=6):
    """Light, dashed, circular target indicator (replaces the bold cyan box)."""
    fx, fy, fr = FOCUS
    ax.add_patch(plt.Circle((fx, fy), fr, ec="#bfefff", fc="none",
                            lw=1.1, ls="--", alpha=0.9, zorder=zorder))


def _configure(dom, omega, eta):
    """Attach the material, sponge, drive and target region to a domain --
    shared by the structured mesh and the conforming (hole-cut) gmsh mesh."""
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


def make(nx=130, omega=OMEGA, eta=0.02):
    dom = dmesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]],
                                 [nx, int(nx*Ly/Lx)], dmesh.CellType.triangle)
    return _configure(dom, omega, eta)


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


def symmetric_support(spacing):
    """CS-RBF support grid made mirror-symmetric about y=Ly/2, and the index map
    of each support point to its mirror partner (x, Ly-y)."""
    ax = Ly / 2.0
    sx = np.arange(0.0, Lx + 1e-9, spacing)
    ny = int(np.ceil(ax / spacing))
    sy = ax + spacing * np.arange(-ny, ny + 1)
    SX, SY = np.meshgrid(sx, sy)
    supp = np.column_stack([SX.ravel(), SY.ravel()])
    key = {(round(px, 5), round(py, 5)): i for i, (px, py) in enumerate(supp)}
    mir = np.array([key[(round(px, 5), round(Ly - py, 5))] for px, py in supp])
    return supp, mir


def run(save="results/dolfinx_lens.png", iters=70):
    ew = make()
    cent = centroids(ew)
    # symmetric parametrization: a fiber field is mirror-symmetric about y=Ly/2
    # iff theta(x,Ly-y) = -theta(x,y); enforce it via a symmetric support grid
    # and the projector  sym(v) = (v - v[mirror]) / 2.
    supp, mir = symmetric_support(0.33)
    B, supp = support_map(cent, (0, Lx), (0, Ly), 0.33, 0.7, supp=supp)
    Md = B.shape[1]
    sym = lambda v: 0.5 * (v - v[mir])
    print(f"vector lens: {ew.W.dofmap.index_map.size_local*4} dofs, "
          f"{Md} symmetric CS-RBF support pts, omega={OMEGA}")
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
        g = sym(B.T @ ew.focus_grad())      # keep the search in the symmetric subspace
        x = sym(mma.update(x, -g))          # maximize -> minimize -J; project design
        if it % 10 == 0 or it == iters-1:
            print(f"[{it:3d}] focus gain {J/E0:.1f}x")
    thopt = best[1]
    ew.set_theta(thopt); ew.solve()
    E1, T1 = ew.focus_energy(), ew.total_energy()
    _, m1 = field_tri(ew)
    gain, f0, f1 = E1/E0, E0/T0*100, E1/T1*100
    print(f"lens: E0={E0:.3e} E1={E1:.3e} gain={gain:.1f}x  "
          f"focus fraction {f0:.2f}% -> {f1:.2f}%")

    np.savez("results/dolfinx_lens_hist.npz", hist=np.array(hist))
    np.savez("results/dolfinx_lens_data.npz", trix=tri.x, triy=tri.y,
             tris=tri.triangles, cent=cent, m0=m0, m1=m1, thopt=thopt,
             hist=np.array(hist), f0=f0, f1=f1, gain=gain)
    vmax = np.percentile(m1, 99.5)
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    for a_, m, t in [(ax[0, 0], m0, f"straight fibers (baseline)\nfocus fraction {f0:.2f}%"),
                     (ax[0, 1], m1, f"optimized fiber lens\nfocus fraction {f1:.2f}%  "
                      f"gain {gain:.0f}$\\times$")]:
        tp = a_.tripcolor(tri, m, cmap="magma", vmax=vmax, shading="gouraud")
        focus_marker(a_)
        a_.set_aspect("equal"); a_.set_xlim(0, Lx); a_.set_ylim(0, Ly)
        a_.set_title(t); plt.colorbar(tp, ax=a_, fraction=0.046)
    plot_toolpaths_phase(ax[1, 0], cent, thopt, (0, Lx), (0, Ly), spacing=0.16)
    focus_marker(ax[1, 0])
    ax[1, 0].set_title("optimized fiber toolpaths (the lens)")
    ax[1, 1].plot(np.arange(1, len(hist)+1), hist, "b-", lw=1.8)
    ax[1, 1].set_xlabel("MMA iteration"); ax[1, 1].set_ylabel(r"focus gain  $J/J_0$")
    ax[1, 1].set_title(f"convergence (final {gain:.0f}$\\times$)")
    ax[1, 1].grid(alpha=0.3); ax[1, 1].set_box_aspect(0.62)
    fig.suptitle("dolfinx in-plane (vector) orthotropic wave lens "
                 f"(Ef1/Ef2=14.6, orientation-only, $\\omega$={OMEGA:.0f})", y=1.0)
    plt.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight")
    print("saved", save)


# ----------------------------------------------------------------------- #
#  Fiber-curvature (curl) constraint:  zeta = cos(t) t_x + sin(t) t_y,
#  |zeta| <= zeta_all, enforced by a penalty on zeta^2 - zeta_all^2.
# ----------------------------------------------------------------------- #
def curl_of(x, B, Bx, By):
    th = B @ x; a = Bx @ x; b = By @ x
    return np.cos(th)*a + np.sin(th)*b, th, a, b


def curl_penalty(x, B, Bx, By, zeta_all):
    """Mean squared-violation penalty P and its gradient dP/dx."""
    zeta, th, a, b = curl_of(x, B, Bx, By)
    viol = np.maximum(zeta**2 - zeta_all**2, 0.0)
    n = len(zeta)
    v = (2.0/n) * zeta * (zeta**2 > zeta_all**2)
    coef = -np.sin(th)*a + np.cos(th)*b
    dP = B.T @ (coef*v) + Bx.T @ (np.cos(th)*v) + By.T @ (np.sin(th)*v)
    return viol.mean(), dP


def curlcheck():
    from wavetopo.dolfinx_wave import csrbf_operators
    ew = make(nx=50)
    cent = centroids(ew)
    supp, _ = symmetric_support(0.33)
    B, Bx, By = csrbf_operators(cent, supp, 0.7)
    rng = np.random.default_rng(1); x = rng.uniform(-0.5, 0.5, B.shape[1])
    za = 1.0
    P0, dP = curl_penalty(x, B, Bx, By, za)
    h = 1e-6
    for k in rng.choice(x.size, 6, replace=False):
        xp = x.copy(); xp[k] += h; Pp, _ = curl_penalty(xp, B, Bx, By, za)
        xm = x.copy(); xm[k] -= h; Pm, _ = curl_penalty(xm, B, Bx, By, za)
        fd = (Pp-Pm)/(2*h)
        print(f"supp {k:4d}: dP={dP[k]:+.5e} fd={fd:+.5e} "
              f"relerr={abs(fd-dP[k])/(abs(fd)+1e-30):.2e}")


def optimize_curl(ew, B, Bx, By, sym, iters, zeta_all=None, mu_max=8.0):
    """Return (x_best, theta_best, gain_hist, zmax_hist, E0).  Maximizes the
    focus energy (log-objective) with a continuation penalty on the fiber
    curvature |zeta| when zeta_all is set; tracks the best FEASIBLE design."""
    Md = B.shape[1]
    mma = MMA(np.full(Md, -np.pi/2), np.full(Md, np.pi/2), move=0.10)
    x = np.zeros(Md)
    ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve(); E0 = ew.focus_energy()
    best = (-np.inf, x.copy(), np.zeros(ew.theta.x.array.size))
    histG, histZ = [], []
    for it in range(iters):
        th = B @ x; ew.set_theta(th); ew.solve()
        E = ew.focus_energy()
        zeta = np.cos(th)*(Bx@x) + np.sin(th)*(By@x); zmax = np.abs(zeta).max()
        histG.append(E/E0); histZ.append(zmax)
        feasible = (zeta_all is None) or (zmax <= 1.02*zeta_all)
        if E > best[0] and feasible:
            best = (E, x.copy(), th.copy())
        g = -sym(B.T @ ew.focus_grad()) / max(E, 1e-30)     # minimize -log E
        if zeta_all is not None:
            mu = mu_max * min(1.0, (it+1)/(0.4*iters))       # continuation ramp
            _, dP = curl_penalty(x, B, Bx, By, zeta_all)
            g = g + mu*sym(dP)
        x = sym(mma.update(x, g))
        if it % 5 == 0 or it == iters-1:
            tag = "uncon" if zeta_all is None else "con"
            print(f"  [{tag} {it:3d}] gain {E/E0:.1f}x  max|zeta|={zmax:.2f}",
                  flush=True)
    if not np.isfinite(best[0]) or best[0] < 0:              # never feasible
        best = (E, x.copy(), th.copy())
    return best[1], best[2], histG, histZ, E0


def run_curl(save="results/dolfinx_lens_curl.png", iters=35):
    from wavetopo.dolfinx_wave import csrbf_operators
    ew = make(nx=80)
    cent = centroids(ew)
    supp, mir = symmetric_support(0.33)
    B, Bx, By = csrbf_operators(cent, supp, 0.7)
    sym = lambda v: 0.5*(v - v[mir])
    curlabs = lambda x: np.abs(np.cos(B@x)*(Bx@x) + np.sin(B@x)*(By@x))

    # --- unconstrained ---
    x0, th0, g0, z0, E0 = optimize_curl(ew, B, Bx, By, sym, iters, zeta_all=None)
    ew.set_theta(th0); ew.solve(); E0f = ew.focus_energy(); T0f = ew.total_energy()
    tri, m0 = field_tri(ew); zc0 = curlabs(x0)
    zeta_all = 0.45*zc0.max()                                # limit = 45% of peak
    print(f"unconstrained: gain {E0f/E0:.1f}x  focus {E0f/T0f*100:.2f}%  "
          f"max|zeta|={zc0.max():.2f};  set zeta_all={zeta_all:.2f}")
    # --- constrained ---
    x1, th1, g1, z1, _ = optimize_curl(ew, B, Bx, By, sym, iters,
                                       zeta_all=zeta_all, mu_max=8.0)
    ew.set_theta(th1); ew.solve(); E1f = ew.focus_energy(); T1f = ew.total_energy()
    _, m1 = field_tri(ew); zc1 = curlabs(x1)
    print(f"constrained:   gain {E1f/E0:.1f}x  focus {E1f/T1f*100:.2f}%  "
          f"max|zeta|={zc1.max():.2f}  (limit {zeta_all:.2f})")

    # curvature arrays are in DG0-dof order; map to cell order for facecolors
    ncell = ew.domain.topology.index_map(2).size_local
    dg0 = np.array([ew.S.dofmap.cell_dofs(c)[0] for c in range(ncell)])
    # persist everything needed to re-plot, BEFORE plotting (so a mathtext/plot
    # error never discards the expensive optimization)
    np.savez("results/dolfinx_lens_curl_data.npz",
             trix=tri.x, triy=tri.y, tris=tri.triangles, cent=cent,
             m0=m0, m1=m1, th0=th0, th1=th1, zc0=zc0[dg0], zc1=zc1[dg0],
             gain0=E0f/E0, gain1=E1f/E0, zmax0=zc0.max(), zmax1=zc1.max(),
             zeta_all=zeta_all)
    fx, fy, fr = FOCUS
    vmax = np.percentile(np.concatenate([m0, m1]), 99.5)
    zmax = max(zc0.max(), zc1.max())
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    rows = [("unconstrained", m0, th0, zc0, E0f/E0),
            (rf"curl-constrained ($|\zeta|\leq{zeta_all:.1f}$)", m1, th1, zc1, E1f/E0)]
    for i, (name, m, th, zc, gain) in enumerate(rows):
        tp = ax[i, 0].tripcolor(tri, m, cmap="magma", vmax=vmax, shading="gouraud")
        focus_marker(ax[i, 0])
        ax[i, 0].set_aspect("equal"); ax[i, 0].set_xlim(0, Lx); ax[i, 0].set_ylim(0, Ly)
        ax[i, 0].set_title(f"{name}\nfield  (gain {gain:.0f}$\\times$)")
        plt.colorbar(tp, ax=ax[i, 0], fraction=0.046)
        plot_toolpaths_phase(ax[i, 1], cent, th, (0, Lx), (0, Ly), spacing=0.16)
        focus_marker(ax[i, 1])
        ax[i, 1].set_title(rf"toolpaths  (max $|\zeta|$={zc.max():.2f})")
        cc = ax[i, 2].tripcolor(tri, facecolors=zc[dg0], cmap="viridis", vmax=zmax)
        ax[i, 2].set_aspect("equal"); ax[i, 2].set_xlim(0, Lx); ax[i, 2].set_ylim(0, Ly)
        ax[i, 2].set_title(r"fiber curvature $|\zeta|$")
        plt.colorbar(cc, ax=ax[i, 2], fraction=0.046)
    fig.suptitle("Fiber-curvature (curl) constraint on the wave lens: bounding "
                 r"$|\zeta|=|\cos\theta\,\theta_x+\sin\theta\,\theta_y|$ removes the "
                 "tight bends at a modest focus cost", y=1.0, fontsize=12)
    plt.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight")
    print("saved", save)
    np.savez("results/dolfinx_lens_curl.npz", g0=np.array(g0), z0=np.array(z0),
             g1=np.array(g1), z1=np.array(z1), zeta_all=zeta_all)


# ----------------------------------------------------------------------- #
#  Through-hole design (paper's prescribed circular voids):  two symmetric
#  circular through-holes are cut in the plate (SIMP soft void, density 0).
#  The fiber lens must still focus the wave past the holes, and -- per the
#  paper's rule -- the fiber toolpaths may not curve tighter than the holes,
#  i.e. |zeta| <= 1/R_hole.  Compare unconstrained vs the through-hole limit.
# ----------------------------------------------------------------------- #
HOLES = [(1.8, Ly/2 - 0.62, 0.28), (1.8, Ly/2 + 0.62, 0.28)]   # (x, y, R), symmetric
# deliberately asymmetric holes (different x, y, radii; NOT mirror-symmetric) so the
# re-optimized fiber field is visibly asymmetric -- proof the design adapts per case
HOLES_ASYM = [(1.5, Ly/2 + 0.70, 0.34), (2.5, Ly/2 - 0.55, 0.24)]


def make_hole(nx=90, omega=OMEGA, eta=0.02, holes=HOLES):
    """LEGACY soft-void hole: a 0/1 density indicator staircased onto the
    structured mesh.  Kept only for comparison -- see make_hole_conforming."""
    ew = make(nx, omega, eta)
    ew.set_zfac(lambda x, y: 1.0 - np.any(
        [np.hypot(x-hx, y-hy) < hr for hx, hy, hr in holes], axis=0).astype(float))
    return ew


def make_hole_conforming(h=0.045, omega=OMEGA, eta=0.02, holes=HOLES):
    """Holes CUT FROM THE GEOMETRY.  The gmsh mesh conforms to each circle, so
    the hole boundary is exactly circular and traction-free (the natural BC of
    the weak form) -- no soft void, no staircase, and the mesh is refined on the
    rim where the scattering actually happens.  This is the reason the
    wave-control study uses dolfinx rather than the structured numpy code."""
    from wavetopo.dolfinx_mesh import rect_mesh
    dom = rect_mesh(Lx, Ly, h, holes=holes, h_hole=0.5*h)
    return _configure(dom, omega, eta)


def run_hole(save="results/dolfinx_lens_hole.png", iters=40):
    from wavetopo.dolfinx_wave import csrbf_operators
    ew = make_hole_conforming(h=0.045)          # conforming, traction-free holes
    cent = centroids(ew)
    supp, mir = symmetric_support(0.33)
    B, Bx, By = csrbf_operators(cent, supp, 0.7)
    sym = lambda v: 0.5*(v - v[mir])
    curlabs = lambda x: np.abs(np.cos(B@x)*(Bx@x) + np.sin(B@x)*(By@x))
    hr = HOLES[0][2]; zeta_all = 1.0/hr                       # through-hole rule

    x0, th0, g0, z0, E0 = optimize_curl(ew, B, Bx, By, sym, iters, zeta_all=None)
    ew.set_theta(th0); ew.solve(); E0f = ew.focus_energy(); T0f = ew.total_energy()
    tri, m0 = field_tri(ew); zc0 = curlabs(x0)
    print(f"hole/unconstrained: gain {E0f/E0:.1f}x  focus {E0f/T0f*100:.2f}%  "
          f"max|zeta|={zc0.max():.2f};  1/R_hole={zeta_all:.2f}", flush=True)

    x1, th1, g1, z1, _ = optimize_curl(ew, B, Bx, By, sym, iters,
                                       zeta_all=zeta_all, mu_max=8.0)
    ew.set_theta(th1); ew.solve(); E1f = ew.focus_energy(); T1f = ew.total_energy()
    _, m1 = field_tri(ew); zc1 = curlabs(x1)
    print(f"hole/constrained:   gain {E1f/E0:.1f}x  focus {E1f/T1f*100:.2f}%  "
          f"max|zeta|={zc1.max():.2f}  (limit {zeta_all:.2f})", flush=True)

    ncell = ew.domain.topology.index_map(2).size_local
    dg0 = np.array([ew.S.dofmap.cell_dofs(c)[0] for c in range(ncell)])
    np.savez("results/dolfinx_lens_hole_data.npz",
             trix=tri.x, triy=tri.y, tris=tri.triangles, cent=cent,
             m0=m0, m1=m1, th0=th0, th1=th1, zc0=zc0[dg0], zc1=zc1[dg0],
             gain0=E0f/E0, gain1=E1f/E0, zmax0=zc0.max(), zmax1=zc1.max(),
             zeta_all=zeta_all, holes=np.array(HOLES),
             g0=np.array(g0), g1=np.array(g1))

    fx, fy, fr = FOCUS
    vmax = np.percentile(np.concatenate([m0, m1]), 99.5)
    zmax = max(zc0.max(), zc1.max())
    tc = np.linspace(0, 2*np.pi, 100)
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    rows = [("unconstrained", m0, th0, x0, E0f/E0),
            (rf"through-hole limit ($|\zeta|\leq 1/R={zeta_all:.1f}$)",
             m1, th1, x1, E1f/E0)]
    for i, (name, m, th, xx, gain) in enumerate(rows):
        tp = ax[i, 0].tripcolor(tri, m, cmap="magma", vmax=vmax, shading="gouraud")
        for hx, hy, rr in HOLES:
            ax[i, 0].plot(hx+rr*np.cos(tc), hy+rr*np.sin(tc), "w-", lw=1.0)
        focus_marker(ax[i, 0])
        ax[i, 0].set_aspect("equal"); ax[i, 0].set_xlim(0, Lx); ax[i, 0].set_ylim(0, Ly)
        ax[i, 0].set_title(f"{name}\nfield  (gain {gain:.0f}$\\times$)")
        plt.colorbar(tp, ax=ax[i, 0], fraction=0.046)
        plot_toolpaths_phase(ax[i, 1], cent, th, (0, Lx), (0, Ly), holes=HOLES,
                             spacing=0.16)
        focus_marker(ax[i, 1])
        ax[i, 1].set_title(rf"toolpaths  (max $|\zeta|$={curlabs(xx).max():.2f})")
        cc = ax[i, 2].tripcolor(tri, facecolors=curlabs(xx)[dg0], cmap="viridis",
                                vmax=zmax)
        for hx, hy, rr in HOLES:
            ax[i, 2].plot(hx+rr*np.cos(tc), hy+rr*np.sin(tc), "w-", lw=1.0)
        ax[i, 2].set_aspect("equal"); ax[i, 2].set_xlim(0, Lx); ax[i, 2].set_ylim(0, Ly)
        ax[i, 2].set_title(r"fiber curvature $|\zeta|$")
        plt.colorbar(cc, ax=ax[i, 2], fraction=0.046)
    fig.suptitle("Wave lens with two prescribed circular through-holes: the fiber "
                 r"toolpaths route around the voids and, under $|\zeta|\leq 1/R$, "
                 "avoid bends tighter than the holes", y=1.0, fontsize=12)
    plt.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight")
    print("saved", save, flush=True)


# ----------------------------------------------------------------------- #
#  Asymmetric through-holes: the mirror symmetry is dropped and the design is
#  re-optimized from scratch, so the fiber field adapts asymmetrically to the two
#  differently-placed/sized voids -- a genuine per-geometry re-optimization.
# ----------------------------------------------------------------------- #
def run_hole_asym(save="results/dolfinx_lens_hole_asym.png", iters=55):
    from wavetopo.dolfinx_wave import csrbf_operators
    ew = make_hole_conforming(h=0.045, holes=HOLES_ASYM)
    cent = centroids(ew)
    B, supp = support_map(cent, (0, Lx), (0, Ly), 0.33, 0.7)   # plain (no symmetry)
    B, Bx, By = csrbf_operators(cent, supp, 0.7)
    ident = lambda v: v                                        # no symmetry projector
    curlabs = lambda x: np.abs(np.cos(B@x)*(Bx@x) + np.sin(B@x)*(By@x))
    hr = min(h[2] for h in HOLES_ASYM); zeta_all = 1.0/hr

    ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve()
    E0b = ew.focus_energy(); tri, m0 = field_tri(ew)
    x1, th1, g1, z1, E0 = optimize_curl(ew, B, Bx, By, ident, iters,
                                        zeta_all=zeta_all, mu_max=8.0)
    ew.set_theta(th1); ew.solve(); E1 = ew.focus_energy(); T1 = ew.total_energy()
    _, m1 = field_tri(ew); zc1 = curlabs(x1)
    print(f"asym holes: gain {E1/E0:.1f}x  focus {E1/T1*100:.2f}%  "
          f"max|zeta|={zc1.max():.2f} (limit {zeta_all:.2f})", flush=True)
    np.savez("results/dolfinx_lens_hole_asym_data.npz",
             trix=tri.x, triy=tri.y, tris=tri.triangles, cent=cent,
             m0=m0, m1=m1, th1=th1, gain=E1/E0, zmax=zc1.max(),
             zeta_all=zeta_all, holes=np.array(HOLES_ASYM))

    fx, fy, fr = FOCUS; vmax = np.percentile(m1, 99.5)
    tc = np.linspace(0, 2*np.pi, 100)
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    for a_, m, t in [(ax[0, 0], m0, "straight fibers (baseline)"),
                     (ax[0, 1], m1, f"re-optimized lens  gain {E1/E0:.0f}$\\times$")]:
        tp = a_.tripcolor(tri, m, cmap="magma", vmax=vmax, shading="gouraud")
        for hx, hy, r in HOLES_ASYM:
            a_.plot(hx+r*np.cos(tc), hy+r*np.sin(tc), "w-", lw=1.0)
        focus_marker(a_)
        a_.set_aspect("equal"); a_.set_xlim(0, Lx); a_.set_ylim(0, Ly); a_.set_title(t)
        plt.colorbar(tp, ax=a_, fraction=0.046)
    plot_toolpaths_phase(ax[1, 0], cent, th1, (0, Lx), (0, Ly), holes=HOLES_ASYM,
                         spacing=0.16)
    focus_marker(ax[1, 0])
    ax[1, 0].set_title(rf"fiber toolpaths (asymmetric)  max$|\zeta|$={zc1.max():.2f}")
    plot_director_field(ax[1, 1], cent, th1, (0, Lx), (0, Ly), holes=HOLES_ASYM, n=30)
    focus_marker(ax[1, 1])
    ax[1, 1].set_title("anisotropy orientation map")
    fig.suptitle("Wave lens with two ASYMMETRIC through-holes: the design is "
                 "re-optimized per geometry, so the fiber field adapts asymmetrically "
                 "around the differently placed voids", y=1.0, fontsize=12)
    plt.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight")
    print("saved", save, flush=True)


# ----------------------------------------------------------------------- #
#  Multi-target beam shaping: MAXIMIZE energy in two focus regions and MINIMIZE
#  it in a third, via a signed region weight w=+1 on the targets, -LAM on the
#  null; maximizing J=\int w|u|^2 focuses two beams and darkens the middle.
# ----------------------------------------------------------------------- #
FOC_A = (0.80*Lx, Ly/2 + 0.62, 0.16)      # maximize
FOC_B = (0.80*Lx, Ly/2 - 0.62, 0.16)      # maximize (mirror of A)
NULLR = (0.80*Lx, Ly/2, 0.16)             # minimize
LAM = 3.0                                  # weight on suppressing the null


def _box(x, y, b):
    return ((np.abs(x-b[0]) < b[2]) & (np.abs(y-b[1]) < b[2])).astype(float)


def make_multi(nx=110, omega=OMEGA, eta=0.02):
    ew = make(nx, omega, eta)
    ew.set_region(lambda x, y: _box(x, y, FOC_A) + _box(x, y, FOC_B)
                  - LAM*_box(x, y, NULLR))
    return ew


def run_multi(save="results/dolfinx_lens_multi.png", iters=75):
    ew = make_multi(nx=110)
    cent = centroids(ew)
    supp, mir = symmetric_support(0.33)
    B, supp = support_map(cent, (0, Lx), (0, Ly), 0.33, 0.7, supp=supp)
    sym = lambda v: 0.5*(v - v[mir])
    e_of = lambda boxes: (ew.set_region(lambda x, y: sum(_box(x, y, b) for b in boxes))
                          or ew.focus_energy())

    ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve()
    tA0, tN0 = e_of([FOC_A, FOC_B]), e_of([NULLR])
    ew.set_region(lambda x, y: _box(x, y, FOC_A) + _box(x, y, FOC_B)
                  - LAM*_box(x, y, NULLR))
    tri, m0 = field_tri(ew); J0 = ew.focus_energy()
    print(f"multi baseline: targets={tA0:.3e} null={tN0:.3e} contrast={tA0/tN0:.1f}",
          flush=True)

    mma = MMA(np.full(B.shape[1], -np.pi/2), np.full(B.shape[1], np.pi/2), move=0.10)
    x = np.zeros(B.shape[1]); best = (J0, np.zeros(ew.theta.x.array.size)); hist = []
    for it in range(iters):
        th = B @ x; ew.set_theta(th); ew.solve()
        J = ew.focus_energy(); hist.append(J)
        if J > best[0]:
            best = (J, th.copy())
        g = sym(B.T @ ew.focus_grad())
        x = sym(mma.update(x, -g))                # maximize J
        if it % 10 == 0 or it == iters-1:
            print(f"[{it:3d}] J={J:.3e}", flush=True)
    thopt = best[1]; ew.set_theta(thopt); ew.solve()
    _, m1 = field_tri(ew)
    tA1, tN1 = e_of([FOC_A, FOC_B]), e_of([NULLR])
    print(f"multi optimized: targets {tA1/tA0:.1f}x brighter; target/null contrast "
          f"{tA0/tN0:.1f}->{tA1/tN1:.1f} ({(tA1/tN1)/(tA0/tN0):.1f}x improvement)",
          flush=True)
    np.savez("results/dolfinx_lens_multi_data.npz",
             trix=tri.x, triy=tri.y, tris=tri.triangles, cent=cent, thopt=thopt,
             m0=m0, m1=m1, hist=np.array(hist), tA0=tA0, tN0=tN0, tA1=tA1, tN1=tN1)

    vmax = np.percentile(m1, 99.5); tc = np.linspace(0, 2*np.pi, 100)
    def marks(a_):
        for b, c in [(FOC_A, "cyan"), (FOC_B, "cyan"), (NULLR, "red")]:
            a_.add_patch(plt.Circle((b[0], b[1]), b[2], ec=c, fc="none",
                         lw=1.3, ls="--", alpha=0.9, zorder=6))
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    for a_, m, t in [(ax[0, 0], m0, "straight fibers (baseline)"),
                     (ax[0, 1], m1, f"two-focus + null lens\ncontrast "
                      f"{tA0/tN0:.0f}$\\to${tA1/tN1:.0f}")]:
        tp = a_.tripcolor(tri, m, cmap="magma", vmax=vmax, shading="gouraud")
        marks(a_)
        a_.set_aspect("equal"); a_.set_xlim(0, Lx); a_.set_ylim(0, Ly); a_.set_title(t)
        plt.colorbar(tp, ax=a_, fraction=0.046)
    plot_toolpaths_phase(ax[1, 0], cent, thopt, (0, Lx), (0, Ly), spacing=0.16)
    marks(ax[1, 0]); ax[1, 0].set_title("fiber toolpaths (beam splitter)")
    ax[1, 1].plot(np.arange(1, len(hist)+1), np.array(hist)/abs(hist[0]), "b-", lw=1.8)
    ax[1, 1].set_xlabel("MMA iteration"); ax[1, 1].set_ylabel(r"$J/|J_0|$")
    ax[1, 1].set_title("convergence"); ax[1, 1].grid(alpha=0.3)
    ax[1, 1].set_box_aspect(0.62)
    fig.suptitle("Multi-target beam shaping: maximize energy in TWO focus regions "
                 "(cyan) and minimize it in a THIRD (red) --- a fiber beam splitter "
                 "with a dark centre", y=1.0, fontsize=12)
    plt.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight")
    print("saved", save, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "fdcheck": fdcheck, "curl": run_curl, "curlcheck": curlcheck,
     "hole": run_hole, "holeasym": run_hole_asym, "multi": run_multi}[cmd]()
