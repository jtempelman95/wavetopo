"""
dolfinx wave-control: fiber-orientation optimization for (1) energy localization,
(2) cloaking, and (3) broadband (multi-frequency) localization, plus the study of
how anisotropy is what makes orientation design work at all.

Central thesis (see wavetopo/dolfinx_wave.py): the anisotropic shear tensor
mu(theta) is theta-INDEPENDENT when muL=muT (isotropic), so for an isotropic
material the fiber orientation is a dead design variable and orientation
optimization can do nothing.  Every gain reported below is therefore, by
construction, unavailable to the isotropic counterpart.

Run with the dolfinx (FEniCSx) interpreter, e.g.:
    .../dolfinx_complex/bin/python3 examples/dolfinx_wave.py localize
    ... cloak
    ... multifreq        (broadband localization at several frequencies at once)
    ... study            (anisotropy sweep for localization AND cloak)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
from dolfinx import mesh as dmesh

from wavetopo.dolfinx_wave import WaveControl, cell_filter, support_map
from wavetopo.dolfinx_viz import plot_toolpaths
from wavetopo.cfrp_optimizer import MMA

Lx, Ly = 4.0, 3.0
FOCUS = (0.78 * Lx, Ly / 2)


def make(omega, muL, muT, nx=60, eta=0.03):
    dom = dmesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]],
                                 [nx, int(nx*Ly/Lx)], dmesh.CellType.triangle)
    wc = WaveControl(dom, omega=omega, muL=muL, muT=muT, rho=1.6, eta=eta)
    m = 0.6
    wc.set_sponge(lambda x, y: np.maximum(0, np.maximum.reduce([
        (x-(Lx-m))/m, (m-y)/m, (y-(Ly-m))/m]))**2 * 80.0)
    # plane pressure wave launched from the left edge
    wc.set_source(lambda x, y: np.exp(-60*(x-0.12)**2)*np.ones_like(y))
    return wc


def focus_region(wc):
    wc.set_region(lambda x, y: ((np.abs(x-FOCUS[0]) < 0.18) &
                                (np.abs(y-FOCUS[1]) < 0.18)).astype(float))


def tri_of(wc):
    dom = wc.domain
    xy = dom.geometry.x[:, :2]
    cells = dom.topology.connectivity(2, 0).array.reshape(-1, 3)
    return mtri.Triangulation(xy[:, 0], xy[:, 1], cells)


def umag(wc):
    a = wc.U.x.array.reshape(-1, 2)
    return np.hypot(a[:, 0], a[:, 1])


def centroids_dg0(wc):
    dom = wc.domain
    dom.topology.create_connectivity(2, 0)
    xy = dom.geometry.x[:, :2]; conn = dom.topology.connectivity(2, 0)
    ncell = dom.topology.index_map(2).size_local
    cent = np.zeros((wc.theta.x.array.size, 2))
    for c in range(ncell):
        cent[wc.S.dofmap.cell_dofs(c)[0]] = xy[conn.links(c)].mean(0)
    return cent


def optimize(wc, mode, iters=40, move=0.12, P=None):
    """Single-frequency orientation optimization on the LOG-objective (so the
    gradient is O(1), scale-invariant, and above MMA's raa0 regularization --
    otherwise the ~1e-12 raw gradients never move the asymptotes).  An optional
    filter P (physical theta = P @ design) regularizes the fiber toolpaths."""
    Nt = wc.theta.x.array.size
    ndes = P.shape[1] if P is not None else Nt
    mma = MMA(np.full(ndes, -np.pi/2), np.full(ndes, np.pi/2), move=move)
    x = np.zeros(ndes); best = (None, None)
    for it in range(iters):
        th = P @ x if P is not None else x
        wc.set_theta(th); wc.solve()
        if mode == "localize":
            val = wc.focus_energy(); graw = -wc.focus_grad()/max(val, 1e-30)
            if best[0] is None or val > best[0]:
                best = (val, th.copy())
        else:
            val = wc.cloak_mismatch(); graw = wc.cloak_grad()/max(val, 1e-30)
            if best[0] is None or val < best[0]:
                best = (val, th.copy())
        g = P.T @ graw if P is not None else graw
        x = mma.update(x, g)
    return best[1]


# ----------------------------------------------------------------------- #
#  Experiment 1: single-frequency energy localization                     #
# ----------------------------------------------------------------------- #
def run_localize(muL=9.0, muT=2.5, omega=30.0, save="results/dolfinx_localize.png"):
    wc = make(omega, muL, muT, nx=120, eta=0.02); focus_region(wc)
    cent = centroids_dg0(wc)
    P, supp = support_map(cent, (0, Lx), (0, Ly), spacing=0.33, r=0.7)
    print(f"localize: {P.shape[1]} CS-RBF support points, omega={omega}, "
          f"muL/muT={muL/muT:.1f}")
    wc.set_theta(np.zeros(wc.theta.x.array.size)); wc.solve()
    E0, T0 = wc.focus_energy(), wc.total_energy(); m0 = umag(wc)
    thopt = optimize(wc, "localize", iters=70, move=0.10, P=P)
    wc.set_theta(thopt); wc.solve()
    E1, T1 = wc.focus_energy(), wc.total_energy(); m1 = umag(wc)
    gain, f0, f1 = E1/E0, E0/T0, E1/T1
    print(f"localize: E0={E0:.3e} E1={E1:.3e} gain={gain:.1f}x  "
          f"focus-fraction {f0*100:.2f}% -> {f1*100:.2f}%")
    tri = tri_of(wc); vmax = np.percentile(m1, 99.5)
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    for a_, m, t in [(ax[0], m0, f"isotropic-equivalent baseline (straight fibers)\n"
                      f"focus fraction {f0*100:.2f}%"),
                     (ax[1], m1, f"optimized fiber orientation (anisotropic)\n"
                      f"focus fraction {f1*100:.2f}%   gain {gain:.0f}$\\times$")]:
        tp = a_.tripcolor(tri, m, cmap="magma", vmax=vmax, shading="gouraud")
        a_.add_patch(plt.Rectangle((FOCUS[0]-0.18, FOCUS[1]-0.18), 0.36, 0.36,
                                   ec="cyan", fc="none", lw=1.5))
        a_.set_aspect("equal"); a_.set_title(t); plt.colorbar(tp, ax=a_, fraction=0.03)
    plot_toolpaths(ax[2], cent, thopt, (0, Lx), (0, Ly), nseed=18, lw=0.9)
    ax[2].add_patch(plt.Rectangle((FOCUS[0]-0.18, FOCUS[1]-0.18), 0.36, 0.36,
                                  ec="cyan", fc="none", lw=1.5, zorder=6))
    ax[2].set_title("optimized fiber toolpaths\n(the curvilinear lens)")
    fig.suptitle(rf"dolfinx energy localization by fiber-orientation design "
                 rf"($\mu_L/\mu_T={muL/muT:.0f}$)", y=1.02)
    plt.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight")
    print("saved", save)
    return gain


# ----------------------------------------------------------------------- #
#  Experiment 2: cloaking a void                                          #
# ----------------------------------------------------------------------- #
def run_cloak(muL=6.0, muT=2.0, omega=24.0, save="results/dolfinx_cloak.png"):
    wc = make(omega, muL, muT)
    cx, cy = Lx/2, Ly/2
    wc.set_zfac(lambda x, y: np.ones_like(x))
    wc.set_theta(np.zeros(wc.theta.x.array.size)); wc.solve(); wc.store_reference()
    mref = umag(wc)
    wc.set_zfac(lambda x, y: 1.0-(np.hypot(x-cx, y-cy) < 0.45).astype(float))
    wc.set_region(lambda x, y: ((x > cx+0.6) & (x < Lx-0.7)).astype(float))
    wc.set_theta(np.zeros(wc.theta.x.array.size)); wc.solve()
    J0 = wc.cloak_mismatch(); munc = umag(wc)
    thopt = optimize(wc, "cloak", iters=45)
    wc.set_theta(thopt); wc.solve(); J1 = wc.cloak_mismatch(); mcl = umag(wc)
    red = J0/J1
    print(f"cloak: J0={J0:.3e} J1={J1:.3e} reduction={red:.1f}x")
    tri = tri_of(wc); vmax = np.percentile(mref, 99.5)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a_, m, t in [(ax[0], mref, "reference field (no void)"),
                     (ax[1], munc, f"bare void, isotropic-equivalent\n"
                      f"shadow $J={J0:.1e}$"),
                     (ax[2], mcl, f"orientation-cloaked (anisotropic)\n"
                      f"$J={J1:.1e}$   {red:.0f}$\\times$ less shadow")]:
        tp = a_.tripcolor(tri, m, cmap="magma", vmax=vmax, shading="gouraud")
        th_c = np.linspace(0, 2*np.pi, 60)
        a_.plot(cx+0.45*np.cos(th_c), cy+0.45*np.sin(th_c), "c-", lw=1)
        a_.set_aspect("equal"); a_.set_title(t); plt.colorbar(tp, ax=a_, fraction=0.03)
    fig.suptitle(rf"dolfinx elastic cloak by fiber-orientation design "
                 rf"($\mu_L/\mu_T={muL/muT:.0f}$)", y=1.02)
    plt.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight")
    print("saved", save)
    return red


# ----------------------------------------------------------------------- #
#  Experiment 3: broadband (multi-frequency) localization                 #
# ----------------------------------------------------------------------- #
def make_band(omegas, muL, muT, nx=60):
    """A shared mesh/design across several frequencies: one WaveControl per
    omega, all reading the SAME theta array.  A single fiber toolpath that must
    focus every frequency at once -- only anisotropy gives it the authority."""
    wcs = [make(w, muL, muT, nx=nx) for w in omegas]
    for wc in wcs:
        focus_region(wc)
    return wcs


def band_metrics(wcs, th):
    fr = []
    for wc in wcs:
        wc.set_theta(th); wc.solve()
        fr.append(wc.focus_energy()/wc.total_energy())
    return np.array(fr)


def optimize_band(wcs, iters=45, move=0.12):
    Nt = wcs[0].theta.x.array.size
    mma = MMA(np.full(Nt, -np.pi/2), np.full(Nt, np.pi/2), move=move)
    th = np.zeros(Nt); best = (None, None)
    for it in range(iters):
        # sum of log-objectives across the band -> min worst-case-ish focus
        g = np.zeros(Nt); logsum = 0.0; worst = np.inf
        for wc in wcs:
            wc.set_theta(th); wc.solve()
            E = wc.focus_energy(); T = wc.total_energy()
            g += -wc.focus_grad()/max(E, 1e-30)
            logsum += np.log(max(E, 1e-30)); worst = min(worst, E/T)
        if best[0] is None or worst > best[0]:
            best = (worst, th.copy())
        th = mma.update(th, g)
    return best[1]


def run_multifreq(muL=6.0, muT=2.0, omegas=(19.0, 21.5, 24.0),
                  save="results/dolfinx_multifreq.png"):
    wcs = make_band(list(omegas), muL, muT)
    th0 = np.zeros(wcs[0].theta.x.array.size)
    f0 = band_metrics(wcs, th0)
    thopt = optimize_band(wcs, iters=45)
    f1 = band_metrics(wcs, thopt)
    print("multifreq focus fractions (%):")
    for w, a, b in zip(omegas, f0, f1):
        print(f"  omega={w:.1f}: {a*100:.2f}% -> {b*100:.2f}%  ({b/a:.0f}x)")
    print(f"  worst-case: {f0.min()*100:.2f}% -> {f1.min()*100:.2f}%")
    # figure: optimized field at each frequency + the fraction bar chart
    tri = tri_of(wcs[0])
    fig = plt.figure(figsize=(18, 5))
    for i, wc in enumerate(wcs):
        wc.set_theta(thopt); wc.solve(); m = umag(wc)
        ax = fig.add_subplot(1, len(wcs)+1, i+1)
        tp = ax.tripcolor(tri, m, cmap="magma",
                          vmax=np.percentile(m, 99.5), shading="gouraud")
        ax.add_patch(plt.Rectangle((FOCUS[0]-0.18, FOCUS[1]-0.18), 0.36, 0.36,
                                   ec="cyan", fc="none", lw=1.5))
        ax.set_aspect("equal")
        ax.set_title(rf"$\omega={omegas[i]:.1f}$   focus {f1[i]*100:.1f}%")
        plt.colorbar(tp, ax=ax, fraction=0.03)
    ax = fig.add_subplot(1, len(wcs)+1, len(wcs)+1)
    x = np.arange(len(wcs)); w = 0.38
    ax.bar(x-w/2, f0*100, w, label="straight (iso-equiv.)", color="0.6")
    ax.bar(x+w/2, f1*100, w, label="optimized (aniso.)", color="darkred")
    ax.set_xticks(x); ax.set_xticklabels([f"{o:.1f}" for o in omegas])
    ax.set_xlabel(r"$\omega$"); ax.set_ylabel("focus fraction (%)")
    ax.legend(); ax.set_title("one toolpath, whole band")
    fig.suptitle(rf"dolfinx broadband localization: a single fiber orientation "
                 rf"field focusing 3 frequencies ($\mu_L/\mu_T={muL/muT:.0f}$)",
                 y=1.03)
    plt.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight")
    print("saved", save)


# ----------------------------------------------------------------------- #
#  Anisotropy study: performance vs mu_L/mu_T for localization and cloak  #
# ----------------------------------------------------------------------- #
def run_study(save="results/dolfinx_anisotropy.png"):
    ratios = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    loc_base, loc_opt = [], []      # focus fraction (%), baseline vs optimized
    clk_base, clk_opt = [], []      # cloak residual J, baseline vs optimized
    for r in ratios:
        muT = 4.0/np.sqrt(r); muL = 4.0*np.sqrt(r)      # geometric mean fixed = 4
        # -- localization --
        wc = make(22.0, muL, muT); focus_region(wc)
        wc.set_theta(np.zeros(wc.theta.x.array.size)); wc.solve()
        loc_base.append(wc.focus_energy()/wc.total_energy()*100)
        thopt = optimize(wc, "localize", iters=35)
        wc.set_theta(thopt); wc.solve()
        loc_opt.append(wc.focus_energy()/wc.total_energy()*100)
        # -- cloak --
        cx, cy = Lx/2, Ly/2
        wc = make(24.0, muL, muT)
        wc.set_zfac(lambda x, y: np.ones_like(x))
        wc.set_theta(np.zeros(wc.theta.x.array.size)); wc.solve(); wc.store_reference()
        wc.set_zfac(lambda x, y: 1.0-(np.hypot(x-cx, y-cy) < 0.45).astype(float))
        wc.set_region(lambda x, y: ((x > cx+0.6) & (x < Lx-0.7)).astype(float))
        wc.set_theta(np.zeros(wc.theta.x.array.size)); wc.solve()
        clk_base.append(wc.cloak_mismatch())
        thopt = optimize(wc, "cloak", iters=35)
        wc.set_theta(thopt); wc.solve()
        clk_opt.append(wc.cloak_mismatch())
        print(f"  ratio {r:.1f}: localize {loc_base[-1]:.2f}%->{loc_opt[-1]:.2f}% ; "
              f"cloak J {clk_base[-1]:.2e}->{clk_opt[-1]:.2e} "
              f"({clk_base[-1]/clk_opt[-1]:.1f}x)")
    loc_base, loc_opt = np.array(loc_base), np.array(loc_opt)
    clk_base, clk_opt = np.array(clk_base), np.array(clk_opt)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(ratios, loc_base, "s--", color="0.5", label="straight fibers (baseline)")
    ax[0].plot(ratios, loc_opt, "o-", color="darkred", label="optimized orientation")
    ax[0].set_xlabel(r"anisotropy ratio  $\mu_L/\mu_T$")
    ax[0].set_ylabel("focus fraction at target (%)")
    ax[0].set_title("Localization: orientation design authority\n"
                    "vanishes at ratio 1 (isotropic), grows with anisotropy")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ratios, clk_base/clk_opt, "o-", color="navy")
    ax[1].axhline(1, color="k", lw=0.6, ls=":")
    ax[1].set_xlabel(r"anisotropy ratio  $\mu_L/\mu_T$")
    ax[1].set_ylabel(r"shadow reduction  $J_{\rm straight}/J_{\rm opt}$")
    ax[1].set_title("Cloak: shadow-suppression factor\n"
                    "1$\\times$ (no cloaking) at ratio 1, rising with anisotropy")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Anisotropy is the enabling mechanism: at ratio 1 the shear "
                 "tensor is orientation-independent, so isotropic material gets "
                 "nothing from orientation design", y=1.04, fontsize=11)
    plt.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight")
    print("saved", save)
    np.savez("results/dolfinx_anisotropy.npz", ratios=ratios,
             loc_base=loc_base, loc_opt=loc_opt,
             clk_base=clk_base, clk_opt=clk_opt)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "localize"
    {"localize": run_localize, "cloak": run_cloak,
     "multifreq": run_multifreq, "study": run_study}[cmd]()
