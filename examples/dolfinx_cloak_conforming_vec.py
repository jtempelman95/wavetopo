"""
Elastic cloak of a CONFORMING circular void -- IN-PLANE (vector) orthotropic
physics.

This is the geometrically honest version of examples/dolfinx_cloak_soft.py.  There
the void is a soft SIMP density hole staircased onto a structured mesh; here the
disk is CUT FROM THE GEOMETRY, so the mesh conforms to the circle and its boundary
is exactly circular and traction-free (the natural BC of the weak form).  That
removes the staircase corners, which otherwise scatter on their own and blur the
radius R_V that the curvature rule zeta_all = 1/R_V is tied to.

Because the holed mesh has no material inside the void, the undisturbed reference
field cannot be obtained by setting the density to one.  It is solved on a
SEPARATE hole-free mesh of the same plate and transferred to the holed mesh by
non-matching interpolation.

    XDG_CACHE_HOME=/home/jrt/wavetopo/.fenics-cache PYTHONPATH=/home/jrt/wavetopo \
    /home/jrt/miniforge3/envs/dolfinx_complex/bin/python3 \
        examples/dolfinx_cloak_conforming_vec.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import ufl
from dolfinx import fem

from wavetopo.dolfinx_elastic import ElasticWave
from wavetopo.dolfinx_mesh import rect_mesh
from wavetopo.dolfinx_wave import support_map
from wavetopo.dolfinx_viz import plot_toolpaths_phase, plot_director_field
from wavetopo.cfrp_optimizer import MMA

# Long domain at omega=50: the dominant wavelength is ~0.67 (an anisotropic mixed
# P/S mode between the fast along-fiber and slow cross-fiber speeds), so an 8.0
# wide plate carries ~12 wavelengths and the far field the cloak must reconstruct
# is a rich standing-wave pattern rather than a few lobes.
# Taller domain: the design shell must sit clear of the absorbing sponge above
# and below the void.  On the old 6x3 plate the shell (R=1.4 about y=1.5) ran
# from y=0.1 to 2.9 while the sponge already began at y=0.55 -- the shell was
# clipped top and bottom, so the design had no room to wrap the void vertically.
Lx, Ly = 8.0, 5.0
MARGIN = 0.6                       # absorbing sponge margin
CX, CY = 3.2, Ly/2                 # void sits upstream of centre: long far field
RV = 0.55                          # void radius (cut from the geometry)
R_DES = 1.50                       # DESIGN SHELL: orientation may change ONLY here
RC = 1.70                          # observation starts outside the design shell
OMEGA, ETA = 50.0, 0.02
HMESH = 0.045                      # ~15 P1 elements per wavelength at omega=50
ITERS = 200


def field_tri(ew):
    """Triangulation + |u| for plotting, on that object's own mesh."""
    dom = ew.domain
    coords = dom.geometry.x
    tris = dom.geometry.dofmap.reshape(-1, 3) if hasattr(dom.geometry.dofmap, "reshape") \
        else dom.geometry.dofmap
    V = fem.functionspace(dom, ("Lagrange", 1))
    mag = fem.Function(V)
    ur = ufl.as_vector([ew.U[0], ew.U[1]])
    ui = ufl.as_vector([ew.U[2], ew.U[3]])
    mag.interpolate(fem.Expression(ufl.sqrt(ufl.dot(ur, ur) + ufl.dot(ui, ui)),
                                   V.element.interpolation_points()))
    return mtri.Triangulation(coords[:, 0], coords[:, 1], tris), mag.x.array


def centroids(ew):
    dom = ew.domain
    dom.topology.create_connectivity(2, 0)
    conn = dom.topology.connectivity(2, 0)
    xy = dom.geometry.x[:, :2]
    n = dom.topology.index_map(2).size_local
    c = np.zeros((ew.theta.x.array.size, 2))
    for e in range(n):
        c[ew.S.dofmap.cell_dofs(e)[0]] = xy[conn.links(e)].mean(0)
    return c


def _configure(ew):
    m = MARGIN
    ew.set_sponge(lambda x, y: np.maximum(0, np.maximum.reduce([
        (x-(Lx-m))/m, (m-y)/m, (y-(Ly-m))/m]))**2 * 70.0)
    ew.set_source(lambda x, y: 30.0*np.exp(-80*(x-0.1)**2)*np.ones_like(y))
    ew.set_theta(np.zeros(ew.theta.x.array.size))
    return ew


def build():
    """Holed mesh + hole-free reference mesh, with the reference field
    transferred by non-matching interpolation."""
    mh = rect_mesh(Lx, Ly, HMESH, holes=[(CX, CY, RV)], h_hole=0.5*HMESH)
    mf = rect_mesh(Lx, Ly, HMESH)
    mk = lambda dom: _configure(ElasticWave(
        dom, omega=OMEGA, Ef1=131., Ef2=9., G12=5., nu12=0.27,
        rho=1.6, rho_m=1.2, eta=ETA, penal=3.0))
    ew, ref = mk(mh), mk(mf)

    ref.solve()                                    # undisturbed field, no void
    idata = fem.create_nonmatching_meshes_interpolation_data(
        ew.Uref.function_space.mesh._cpp_object,
        ew.Uref.function_space.element,
        ref.U.function_space.mesh._cpp_object, 1e-6)   # padding (dolfinx 0.7.x)
    ew.Uref.interpolate(ref.U, nmm_interpolation_data=idata)
    ew.Uref.x.scatter_forward()

    m = MARGIN
    ew.set_region(lambda x, y: ((np.hypot(x-CX, y-CY) > RC) & (x > m) &
                                (x < Lx-m) & (y > m) & (y < Ly-m)).astype(float))
    return ew, ref


def main(iters=ITERS, move=0.15, warm=None):
    """move: MMA move limit.  The default 0.15 makes the harder (omega=50, 6x3)
    problem oscillate -- the reduction sawtooths, crashing and recovering to a
    higher peak, with the envelope still rising at the iteration cap.  A smaller
    move plus a warm start from a previous best design converges it smoothly.

    warm: path to a previous *_data.npz; its per-cell ``thopt`` is projected back
    onto the support values by least squares (B is row-stochastic, not square)."""
    ew, ref = build()
    ncell = ew.theta.x.array.size
    print(f"conforming mesh: {ncell} cells; hole-free reference mesh: "
          f"{ref.theta.x.array.size} cells", flush=True)

    trf, mref = field_tri(ref)                     # reference on its own mesh
    ew.set_theta(np.zeros(ncell)); ew.solve()
    J0 = ew.cloak_mismatch(); tri, munc = field_tri(ew)

    cent = centroids(ew)
    B, supp = support_map(cent, (0, Lx), (0, Ly), spacing=0.28, r=0.6)
    # Restrict the DESIGN to the shell r < R_DES.  Pinning the support values
    # alone is not enough: the CS-RBF has radius 0.6, so an active support near
    # the shell edge still rotates fibers well outside it.  We therefore also
    # mask the per-cell orientation by a smooth taper that reaches exactly zero
    # at r = R_DES, and carry the same mask through the chain rule.  Outside the
    # red circle the fibers are then EXACTLY the straight baseline.
    rc_cell = np.hypot(cent[:, 0]-CX, cent[:, 1]-CY)
    w_t = 0.20                                     # taper band
    tap = np.clip((R_DES - rc_cell)/w_t, 0.0, 1.0)
    tap = tap*tap*(3 - 2*tap)                      # smoothstep, C1 at both ends
    active = np.hypot(supp[:, 0]-CX, supp[:, 1]-CY) < R_DES + 0.6
    lo = np.where(active, -np.pi/2, 0.0); hi = np.where(active, np.pi/2, 0.0)
    print(f"{int(active.sum())}/{B.shape[1]} active support pts; design shell "
          f"R_DES={R_DES} ({int((tap>0).sum())}/{ncell} cells free); "
          f"uncloaked J0={J0:.3e}", flush=True)

    # adapt=True: contract the move limit after any step that worsened J.  The
    # objective valley here is smooth but steep (a line scan gives 8.2x -> 24x ->
    # 32.5x over the last 15% of the path to the optimum), so a fixed large step
    # overshoots it and J jumps -- the dips in the convergence curve.
    mma = MMA(lo, hi, move=move, adapt=True)
    x = np.zeros(B.shape[1]); best = (J0, np.zeros(ncell)); hist = []
    if warm is not None and os.path.exists(warm):
        # B is a scipy.sparse csr_matrix (support_map), so np.linalg.lstsq cannot
        # take it -- use the sparse least-squares solver.  Clipping to [lo,hi]
        # also re-zeros the supports outside the design shell.
        from scipy.sparse.linalg import lsqr
        th_w = np.load(warm)["thopt"]
        x = np.clip(lsqr(B, th_w, atol=1e-12, btol=1e-12)[0], lo, hi)
        print(f"warm-started from {warm} (move={move}); "
              f"projection residual {np.linalg.norm(B@x - th_w)/max(np.linalg.norm(th_w),1e-30):.2e}",
              flush=True)
    for it in range(iters):
        th = tap*(B @ x); ew.set_theta(th); ew.solve()
        J = ew.cloak_mismatch(); hist.append(J0/J)
        if J < best[0]:
            best = (J, th.copy())
        x = mma.update(x, B.T @ (tap*ew.cloak_grad()), f=J)  # mask in the chain rule
        if it % 10 == 0 or it == iters-1:
            print(f"[{it:3d}] scatter J={J:.3e}  reduction={J0/J:.1f}x "
                  f"(best {J0/best[0]:.1f}x)", flush=True)

    thopt = best[1]
    ew.set_theta(thopt); ew.solve(); J1 = ew.cloak_mismatch()
    outside = np.abs(thopt[rc_cell >= R_DES]).max() if (rc_cell >= R_DES).any() else 0.0
    print(f"max |theta| OUTSIDE the design shell: {outside:.2e} (must be ~0)", flush=True)
    tri, mcl = field_tri(ew); red = J0/J1
    print(f"CONFORMING vector cloak: J0={J0:.3e} -> J1={J1:.3e} "
          f"reduction={red:.1f}x", flush=True)
    np.savez("results/dolfinx_cloak_conforming_data.npz",
             trix=tri.x, triy=tri.y, tris=tri.triangles,
             trfx=trf.x, trfy=trf.y, trfs=trf.triangles,
             cent=cent, thopt=thopt, mref=mref, munc=munc, mcl=mcl,
             hist=np.array(hist), J0=J0, J1=J1, red=red, RV=RV, RC=RC,
             R_DES=R_DES, tap=tap, CX=CX, CY=CY, Lx=Lx, Ly=Ly,
             hmesh=HMESH)

    vmax = np.percentile(mref, 99.5)
    fig, ax = plt.subplots(2, 3, figsize=(19, 8.5))
    tc = np.linspace(0, 2*np.pi, 200)
    for a_, t_, m_, ttl in [
            (ax[0, 0], trf, mref, "reference (hole-free plate)"),
            (ax[0, 1], tri, munc, f"uncloaked void\nscatter $J_0={J0:.2e}$"),
            (ax[0, 2], tri, mcl, f"orientation-cloaked\n$J_1={J1:.2e}$  "
                                 f"({red:.0f}$\\times$ less)")]:
        tp = a_.tripcolor(t_, m_, cmap="magma", vmax=vmax, shading="gouraud")
        a_.plot(CX+RV*np.cos(tc), CY+RV*np.sin(tc), "c-", lw=1.0)
        a_.plot(CX+R_DES*np.cos(tc), CY+R_DES*np.sin(tc), "r--", lw=1.4)
        a_.plot(CX+RC*np.cos(tc), CY+RC*np.sin(tc), "w:", lw=0.8)
        a_.set_aspect("equal"); a_.set_xlim(0, Lx); a_.set_ylim(0, Ly)
        a_.set_title(ttl); plt.colorbar(tp, ax=a_, fraction=0.03)
    plot_toolpaths_phase(ax[1, 0], cent, thopt, (0, Lx), (0, Ly),
                         holes=(CX, CY, RV), spacing=0.16, n=240)
    ax[1, 0].add_patch(plt.Circle((CX, CY), R_DES, ec="red", fc="none",
                       lw=1.4, ls="--"))
    ax[1, 0].set_title("optimized fiber toolpaths\n(orientation designed ONLY "
                       "inside the red shell)")
    lc = plot_director_field(ax[1, 1], cent, thopt, (0, Lx), (0, Ly),
                             holes=(CX, CY, RV), n=30)
    ax[1, 1].add_patch(plt.Circle((CX, CY), R_DES, ec="red", fc="none",
                       lw=1.4, ls="--"))
    ax[1, 1].set_title("anisotropy orientation map\n(fiber director field)")
    cb = plt.colorbar(lc, ax=ax[1, 1], fraction=0.03, ticks=[0, np.pi/2, np.pi])
    cb.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])
    ax[1, 2].plot(np.arange(1, len(hist)+1), hist, "b-", lw=1.8)
    ax[1, 2].set_xlabel("MMA iteration")
    ax[1, 2].set_ylabel(r"scatter reduction  $J_0/J$")
    _wtxt = "" if warm is None else "  [WARM-STARTED: curve begins at the\n" \
        "seed design, earlier history not shown]"
    ax[1, 2].set_title(f"convergence (final {red:.0f}$\\times$){_wtxt}",
                       fontsize=10 if warm else None)
    ax[1, 2].grid(alpha=0.3); ax[1, 2].set_box_aspect(0.7)
    fig.suptitle("dolfinx in-plane (vector) elastic cloak of a CONFORMING "
                 f"traction-free void ($\\omega$={OMEGA:.0f}): {red:.0f}$\\times$ "
                 "scatter reduction", y=1.0, fontsize=13)
    plt.tight_layout()
    # distinct from examples/dolfinx_cloak_conforming.py (the older SCALAR study),
    # which writes results/dolfinx_cloak_conforming.png -- do not clobber it
    fig.savefig("results/dolfinx_cloak_conforming_vec.png", dpi=140,
                bbox_inches="tight")
    print("saved results/dolfinx_cloak_conforming_vec.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--move", type=float, default=0.15)
    ap.add_argument("--warm", default=None)
    a = ap.parse_args()
    main(iters=a.iters, move=a.move, warm=a.warm)
