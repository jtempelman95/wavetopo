"""
Elastic cloak of a soft (SIMP) void -- IN-PLANE (vector) orthotropic physics,
the faithful dolfinx port of examples/wave_cloak.py.  A fiber-composite plate
contains a void (density z=0) that scatters an incident plane wave.  Optimizing
the fiber orientation in a shell around the void makes the far field match the
void-free reference -- the void becomes ~invisible.  Single material; only the
curved fiber toolpaths do the work.

    .../dolfinx_complex/bin/python3 examples/dolfinx_cloak_soft.py
"""
import sys
import numpy as np
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from wavetopo.dolfinx_elastic import ElasticWave
from wavetopo.dolfinx_wave import support_map, csrbf_operators, curl_penalty
from wavetopo.dolfinx_viz import (plot_toolpaths, plot_toolpaths_phase,
                                 plot_director_field)
from wavetopo.cfrp_optimizer import MMA

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


def make(nx=150, ny=100, omega=OMEGA, eta=ETA):
    """Build the soft-void cloak problem: absorbing sponge, left-edge plane-wave
    drive, void-free reference stored, far-field observation region, and the SIMP
    soft void.  Returns the ready-to-solve ElasticWave (zfac already = soft void)."""
    dom = dmesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]],
                                 [nx, ny], dmesh.CellType.triangle)
    ew = ElasticWave(dom, omega=omega, Ef1=131., Ef2=9., G12=5., nu12=0.27,
                     rho=1.6, rho_m=1.2, eta=eta, penal=3.0)
    m = 0.55
    ew.set_sponge(lambda x, y: np.maximum(0, np.maximum.reduce([
        (x-(Lx-m))/m, (m-y)/m, (y-(Ly-m))/m]))**2 * 70.0)
    ew.set_source(lambda x, y: 30.0*np.exp(-80*(x-0.1)**2)*np.ones_like(y))
    # reference field (no void) -> store
    ew.set_zfac(lambda x, y: np.ones_like(x))
    ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve(); ew.store_reference()
    # observation region: far field outside the shell, inside the non-sponge box
    ew.set_region(lambda x, y: ((np.hypot(x-CX, y-CY) > RC) & (x > m) &
                                (x < Lx-m) & (y > m) & (y < Ly-m)).astype(float))
    # soft void
    ew.set_zfac(lambda x, y: 1.0 - (np.hypot(x-CX, y-CY) < RV).astype(float))
    return ew


def main():
    m = 0.55
    ew = make()                       # reference already stored; zfac = soft void
    # reference field for plotting (temporarily restore the void-free structure)
    ew.set_zfac(lambda x, y: np.ones_like(x))
    ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve()
    trf, mref = field_tri(ew)
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
    hist = []
    for it in range(60):
        th = B @ x; ew.set_theta(th); ew.solve()
        J = ew.cloak_mismatch(); hist.append(J0/J)
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
    np.savez("results/dolfinx_cloak_hist.npz", hist=np.array(hist))
    np.savez("results/dolfinx_cloak_soft_data.npz",
             trix=tri.x, triy=tri.y, tris=tri.triangles,
             trfx=trf.x, trfy=trf.y, trfs=trf.triangles,
             cent=cent, thopt=thopt, mref=mref, munc=munc, mcl=mcl,
             hist=np.array(hist), J0=J0, J1=J1, red=red)

    vmax = np.percentile(mref, 99.5)
    fig, ax = plt.subplots(2, 3, figsize=(19, 8.5))
    tc = np.linspace(0, 2*np.pi, 100)
    for a_, t_, m_, ttl in [(ax[0, 0], trf, mref, "reference (no void)"),
                            (ax[0, 1], tri, munc, f"uncloaked void\nscatter $J_0={J0:.2e}$"),
                            (ax[0, 2], tri, mcl, f"orientation-cloaked\n$J_1={J1:.2e}$  "
                             f"({red:.0f}$\\times$ less)")]:
        tp = a_.tripcolor(t_, m_, cmap="magma", vmax=vmax, shading="gouraud")
        a_.plot(CX+RV*np.cos(tc), CY+RV*np.sin(tc), "c-", lw=1.0)
        a_.plot(CX+RC*np.cos(tc), CY+RC*np.sin(tc), "w:", lw=0.8)
        a_.set_aspect("equal"); a_.set_xlim(0, Lx); a_.set_ylim(0, Ly)
        a_.set_title(ttl); plt.colorbar(tp, ax=a_, fraction=0.03)
    plot_toolpaths_phase(ax[1, 0], cent, thopt, (0, Lx), (0, Ly),
                         holes=(CX, CY, RV), spacing=0.16, n=240)
    ax[1, 0].add_patch(plt.Circle((CX, CY), RC, ec="tab:blue", fc="none",
                       lw=1.2, ls=":"))
    ax[1, 0].set_title("optimized fiber toolpaths\n(curving around the void)")
    lc = plot_director_field(ax[1, 1], cent, thopt, (0, Lx), (0, Ly),
                             holes=(CX, CY, RV), n=30)
    ax[1, 1].add_patch(plt.Circle((CX, CY), RC, ec="tab:blue", fc="none",
                       lw=1.2, ls=":"))
    ax[1, 1].set_title("anisotropy orientation map\n(fiber director field)")
    cb = plt.colorbar(lc, ax=ax[1, 1], fraction=0.03,
                      ticks=[0, np.pi/2, np.pi])
    cb.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])
    ax[1, 2].plot(np.arange(1, len(hist)+1), hist, "b-", lw=1.8)
    ax[1, 2].set_xlabel("MMA iteration")
    ax[1, 2].set_ylabel(r"scatter reduction  $J_0/J$")
    ax[1, 2].set_title(f"convergence (final {red:.0f}$\\times$)")
    ax[1, 2].grid(alpha=0.3); ax[1, 2].set_box_aspect(0.7)
    fig.suptitle("dolfinx in-plane (vector) elastic cloak of a soft void "
                 f"($\\omega$={OMEGA:.0f}): {red:.0f}$\\times$ scatter reduction, "
                 "far field restored", y=1.0, fontsize=13)
    plt.tight_layout()
    fig.savefig("results/dolfinx_cloak_soft.png", dpi=140, bbox_inches="tight")
    print("saved results/dolfinx_cloak_soft.png")


# --------------------------------------------------------------------------- #
#  Fiber-curvature (curl) constraint for the cloak.  The void is a prescribed
#  circular through-hole of radius RV, so the paper's manufacturing rule sets the
#  fiber-curvature limit to zeta_all = 1/RV: the fiber toolpaths may not curve
#  tighter than the hole they wrap.  We compare the unconstrained cloak (tight
#  bends hugging the void) against the curl-constrained cloak.
# --------------------------------------------------------------------------- #
def _cloak_support(ew, spacing=0.30, r=0.62):
    cent = centroids(ew)
    B, supp = support_map(cent, (0, Lx), (0, Ly), spacing=spacing, r=r)
    B, Bx, By = csrbf_operators(cent, supp, r)
    active = np.hypot(supp[:, 0]-CX, supp[:, 1]-CY) < RC + 0.3
    lo = np.where(active, -np.pi/2, 0.0); hi = np.where(active, np.pi/2, 0.0)
    return cent, supp, B, Bx, By, lo, hi, active


def optimize_curl_cloak(ew, B, Bx, By, lo, hi, iters, zeta_all=None,
                        mu_max=8.0, move=0.15, mask=None):
    """Minimize the far-field scatter mismatch J (log objective).  Unconstrained
    (zeta_all=None): return the best-objective design.  Constrained: add a
    continuation curl penalty (restricted to the solid ``mask``) and return the
    final continuation iterate -- the natural constrained optimum -- reporting the
    fiber curvature over the solid material only (curvature inside the void has no
    fibers).  Returns (x, theta, red_hist, zmax_hist, J0)."""
    Md = B.shape[1]
    mma = MMA(lo, hi, move=move)
    x = np.zeros(Md)
    ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve(); J0 = ew.cloak_mismatch()
    sel = slice(None) if mask is None else np.asarray(mask, bool)
    best = (np.inf, x.copy(), np.zeros(ew.theta.x.array.size))
    histR, histZ = [], []
    for it in range(iters):
        th = B @ x; ew.set_theta(th); ew.solve()
        J = ew.cloak_mismatch()
        zabs = np.abs(np.cos(th)*(Bx@x) + np.sin(th)*(By@x))
        zmax = zabs[sel].max()
        histR.append(J0/J); histZ.append(zmax)
        if zeta_all is None and J < best[0]:                # unconstrained: best J
            best = (J, x.copy(), th.copy())
        g = B.T @ ew.cloak_grad()                           # minimize J (raw, as main)
        if zeta_all is not None:
            _, dP = curl_penalty(x, B, Bx, By, zeta_all, mask=mask)
            mu = mu_max * min(1.0, (it+1)/(0.4*iters))       # continuation ramp
            scale = np.linalg.norm(g) / (np.linalg.norm(dP) + 1e-30)
            g = g + mu*scale*dP                              # balance penalty vs objective
        x = mma.update(x, g)
        if it % 5 == 0 or it == iters-1:
            tag = "uncon" if zeta_all is None else "con"
            print(f"  [{tag} {it:3d}] reduction {J0/J:.1f}x  max|zeta|={zmax:.2f}",
                  flush=True)
    if zeta_all is None:
        return best[1], best[2], histR, histZ, J0
    return x.copy(), th.copy(), histR, histZ, J0             # constrained: final iterate


def run_curl(save="results/dolfinx_cloak_curl.png", iters=60):
    ew = make(nx=110, ny=73)
    cent, supp, B, Bx, By, lo, hi, active = _cloak_support(ew)
    curlabs = lambda x: np.abs(np.cos(B@x)*(Bx@x) + np.sin(B@x)*(By@x))
    # curvature only means anything where there is material: mask out the void
    sol = np.hypot(cent[:, 0]-CX, cent[:, 1]-CY) > RV
    print(f"cloak curl: {int(active.sum())}/{B.shape[1]} active pts; "
          f"through-hole radius RV={RV} -> 1/RV={1.0/RV:.2f}", flush=True)

    x0, th0, r0, z0, J0 = optimize_curl_cloak(ew, B, Bx, By, lo, hi, iters,
                                              zeta_all=None, mask=sol)
    ew.set_theta(th0); ew.solve(); J0f = ew.cloak_mismatch()
    tri, m0 = field_tri(ew); zc0 = curlabs(x0); zmax0 = zc0[sol].max()
    # through-hole rule limits curvature to 1/RV (fall back to half the solid peak
    # if the hole rule is already met, so the comparison is always binding)
    zeta_all = min(1.0/RV, 0.5*zmax0)
    print(f"unconstrained: reduction {J0/J0f:.1f}x  max|zeta|(solid)={zmax0:.2f};  "
          f"zeta_all={zeta_all:.2f}", flush=True)

    x1, th1, r1, z1, _ = optimize_curl_cloak(ew, B, Bx, By, lo, hi, iters,
                                             zeta_all=zeta_all, mu_max=2.5, mask=sol)
    ew.set_theta(th1); ew.solve(); J1f = ew.cloak_mismatch()
    _, m1 = field_tri(ew); zc1 = curlabs(x1); zmax1 = zc1[sol].max()
    print(f"constrained:   reduction {J0/J1f:.1f}x  max|zeta|(solid)={zmax1:.2f}  "
          f"(limit {zeta_all:.2f})", flush=True)

    ncell = ew.domain.topology.index_map(2).size_local
    dg0 = np.array([ew.S.dofmap.cell_dofs(c)[0] for c in range(ncell)])
    solc = sol[dg0]
    np.savez("results/dolfinx_cloak_curl_data.npz",
             trix=tri.x, triy=tri.y, tris=tri.triangles, cent=cent, solc=solc,
             m0=m0, m1=m1, th0=th0, th1=th1, zc0=zc0[dg0], zc1=zc1[dg0],
             red0=J0/J0f, red1=J0/J1f, zmax0=zmax0, zmax1=zmax1,
             zeta_all=zeta_all, r0=np.array(r0), z0=np.array(z0),
             r1=np.array(r1), z1=np.array(z1))

    vmax = np.percentile(np.concatenate([m0, m1]), 99.5)
    zmax = max(zmax0, zmax1); tc = np.linspace(0, 2*np.pi, 120)
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    hole_tag = (rf"$\approx1/R_V$" if abs(zeta_all - 1.0/RV) < 1e-6 else "")
    rows = [("unconstrained", m0, th0, x0, J0/J0f, zmax0),
            (rf"curl-constrained ($|\zeta|\leq{zeta_all:.2f}${hole_tag})",
             m1, th1, x1, J0/J1f, zmax1)]
    for i, (name, m, th, xx, red, zmx) in enumerate(rows):
        tp = ax[i, 0].tripcolor(tri, m, cmap="magma", vmax=vmax, shading="gouraud")
        ax[i, 0].plot(CX+RV*np.cos(tc), CY+RV*np.sin(tc), "c-", lw=1.0)
        ax[i, 0].plot(CX+RC*np.cos(tc), CY+RC*np.sin(tc), "w:", lw=0.8)
        ax[i, 0].set_aspect("equal"); ax[i, 0].set_xlim(0, Lx); ax[i, 0].set_ylim(0, Ly)
        ax[i, 0].set_title(f"{name}\nfar field  ({red:.0f}$\\times$ less scatter)")
        plt.colorbar(tp, ax=ax[i, 0], fraction=0.03)
        plot_toolpaths_phase(ax[i, 1], cent, th, (0, Lx), (0, Ly),
                             holes=(CX, CY, RV), spacing=0.16, n=240)
        ax[i, 1].add_patch(plt.Circle((CX, CY), RC, ec="tab:blue", fc="none",
                           lw=1.0, ls=":"))
        ax[i, 1].set_title(rf"fiber toolpaths  (max $|\zeta|$={zmx:.2f})")
        zc = curlabs(xx)[dg0].copy(); zc[~solc] = np.nan     # blank the void
        cc = ax[i, 2].tripcolor(tri, facecolors=zc, cmap="viridis", vmax=zmax)
        ax[i, 2].plot(CX+RV*np.cos(tc), CY+RV*np.sin(tc), "c-", lw=1.0)
        ax[i, 2].set_aspect("equal"); ax[i, 2].set_xlim(0, Lx); ax[i, 2].set_ylim(0, Ly)
        ax[i, 2].set_title(r"fiber curvature $|\zeta|$ (solid)")
        plt.colorbar(cc, ax=ax[i, 2], fraction=0.03)
    fig.suptitle("Fiber-curvature (curl) constraint on the elastic cloak: limiting "
                 r"$|\zeta|$ toward $1/R_V$ (the prescribed circular through-hole) "
                 "relaxes the tight bends hugging the void", y=1.0, fontsize=12)
    plt.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight")
    print("saved", save, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": main, "curl": run_curl}[cmd]()
