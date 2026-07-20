"""
Confirm fiber orientation is a Dirac MASS for the elastic honeycomb:
opposite A/B sublattice tilt (+phi/-phi) breaks inversion and opens the valley
gap at the K-point Dirac cone.  Uses sparse shift-invert (fast).

Outputs (results/):
  valley_cone.png  -- band structure through K: Dirac cone (phi=0) vs gap (phi>0)
  valley_mass.png  -- valley gap vs A/B fiber tilt phi
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.cfrp import QuadMesh, CSRBFMapping, Material
from wavetopo.bloch import BlochProblem
from examples.valley_hall import honeycomb_density
from examples.valley_viz import sublattice_support_angles


def build(nx, radius):
    a = 1.0
    Lx, Ly = np.sqrt(3) * a, 3 * a
    ny = int(round(nx * Ly / Lx))
    mesh = QuadMesh(nx, ny, Lx, Ly)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    sxs = np.linspace(0, Lx, 6, endpoint=False)
    sys_ = np.linspace(0, Ly, 10, endpoint=False)
    SXg, SYg = np.meshgrid(sxs, sys_)
    rbf = CSRBFMapping(mesh, (SXg.ravel(), SYg.ravel()), r_s=0.55,
                       period=(Lx, Ly))
    bp = BlochProblem(mesh, mat, rbf, penal=3.0)
    z = honeycomb_density(mesh, radius, (Lx, Ly))
    supp = (SXg.ravel(), SYg.ravel())
    return mesh, bp, rbf, z, supp, (Lx, Ly)


def find_mass_dirac(bp, z, rbf, supp, period, nev, nky, phi_test):
    """The folded honeycomb Dirac sits on the kx=0 (Gamma-Y) line.  For each
    band pair, find the min-gap ky and test whether an A/B fiber tilt opens it
    -- return the mass-responsive (valley) Dirac (pair, ky, omega)."""
    Lx, Ly = period
    kys = np.linspace(0.05, 2 * np.pi / Ly - 0.05, nky)
    th0 = np.zeros(rbf.M)
    tht = sublattice_support_angles(*supp, period, np.deg2rad(phi_test))
    def gap_p(ky, p):
        return lambda: None
    best = None
    for p in range(1, nev - 1):
        bp.assemble(z, th0)
        g0s = [bp.bands_at_k(0.0, ky, nev)[p + 1] -
               bp.bands_at_k(0.0, ky, nev)[p] for ky in kys]
        j = int(np.argmin(g0s))
        if g0s[j] > 0.08:                      # coarse screen (cone is sharp)
            continue
        # refine ky around the coarse minimum to pin the Dirac touching
        lo = kys[max(0, j - 1)]; hi = kys[min(len(kys) - 1, j + 1)]
        kfine = np.linspace(lo, hi, 25)
        gfine = [bp.bands_at_k(0.0, ky, nev)[p + 1] -
                 bp.bands_at_k(0.0, ky, nev)[p] for ky in kfine]
        jf = int(np.argmin(gfine))
        kyD, g0 = kfine[jf], gfine[jf]
        if g0 > 0.04:
            continue
        wD = bp.bands_at_k(0.0, kyD, nev)
        bp.assemble(z, tht)
        wt = bp.bands_at_k(0.0, kyD, nev)
        opening = (wt[p + 1] - wt[p]) - g0
        if opening > 0.01 and (best is None or opening > best[0]):
            best = (opening, p, kyD, g0, wt[p + 1] - wt[p],
                    0.5 * (wD[p] + wD[p + 1]))
    return best


def line_bands(bp, z, th, k0, kdir, span, npt, nev):
    ks = np.linspace(-span, span, npt)
    bp.assemble(z, th)
    out = []
    for s in ks:
        kx, ky = k0[0] + s * kdir[0], k0[1] + s * kdir[1]
        out.append(bp.bands_at_k(kx, ky, nev))
    return ks, np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=22)
    ap.add_argument("--radius", type=float, default=0.33)
    ap.add_argument("--nev", type=int, default=7)
    ap.add_argument("--nky", type=int, default=31)
    args = ap.parse_args()

    mesh, bp, rbf, z, supp, period = build(args.nx, args.radius)
    Lx, Ly = period
    print(f"mesh {mesh.nelx}x{mesh.nely}, ndof_red={bp.ndof_red}, "
          f"fill={z.mean():.3f}")

    found = find_mass_dirac(bp, z, rbf, supp, period, args.nev, args.nky,
                            phi_test=20.0)
    if found is None:
        print("no mass-responsive Dirac found; try tuning --radius")
        return
    _, p, kyD, g0, gt, wD = found
    print(f"valley Dirac: bands {p}-{p+1} at k=(0,{kyD:.3f}) omega~{wD:.3f}: "
          f"gap(phi=0)={g0:.4f} -> gap(phi=20)={gt:.4f}")

    # ---- cone vs gap band structure through K (along kx and ky) ----
    th0 = np.zeros(rbf.M)
    thm = sublattice_support_angles(*supp, period, np.deg2rad(25))
    sp = 0.45
    nev = args.nev
    kxp0, bx0 = line_bands(bp, z, th0, (0, kyD), (1, 0), sp, 31, nev)
    kyp0, by0 = line_bands(bp, z, th0, (0, kyD), (0, 1), sp, 31, nev)
    kxpm, bxm = line_bands(bp, z, thm, (0, kyD), (1, 0), sp, 31, nev)
    kypm, bym = line_bands(bp, z, thm, (0, kyD), (0, 1), sp, 31, nev)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    # concatenate ky-line (left half) and kx-line (right half) crossing K
    def stitch(ax_, ky_, by_, kx_, bx_, color, title):
        n = len(ky_)
        x1 = np.linspace(-1, 0, n); x2 = np.linspace(0, 1, n)
        ax_.plot(x1, by_[:, p:p + 2], color=color, lw=1.6)
        ax_.plot(x2, bx_[:, p:p + 2], color=color, lw=1.6)
        ax_.plot(x1, by_[:, max(0, p - 1):p], color="0.7", lw=0.8)
        ax_.plot(x2, bx_[:, max(0, p - 1):p], color="0.7", lw=0.8)
        ax_.plot(x1, by_[:, p + 2:p + 3], color="0.7", lw=0.8)
        ax_.plot(x2, bx_[:, p + 2:p + 3], color="0.7", lw=0.8)
        ax_.axvline(0, color="0.6", lw=0.6, ls="--")
        ax_.set_title(title); ax_.set_xticks([-1, 0, 1])
        ax_.set_xticklabels(["Y", "K", "X"])
    stitch(ax[0], kyp0, by0, kxp0, bx0, "tab:blue",
           f"φ = 0:  Dirac cone  (gap {g0:.3f})")
    stitch(ax[1], kypm, bym, kxpm, bxm, "tab:red",
           f"φ = 25°:  fiber-mass valley gap")
    ax[0].set_ylabel(r"$\omega$")
    fig.suptitle("Fiber orientation is a Dirac mass: A/B tilt opens the valley "
                 "gap (elastic honeycomb)", y=1.0)
    plt.tight_layout()
    fig.savefig("results/valley_cone.png", dpi=140, bbox_inches="tight")
    print("saved results/valley_cone.png")

    # ---- gap vs phi ----
    phis = np.linspace(0, 40, 17)
    gaps = []
    for phi in phis:
        thh = sublattice_support_angles(*supp, period, np.deg2rad(phi))
        bp.assemble(z, thh)
        w = bp.bands_at_k(0.0, kyD, nev)
        gaps.append(w[p + 1] - w[p])
    gaps = np.array(gaps)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(phis, gaps, "o-", color="darkred")
    ax.set_xlabel("A/B fiber tilt  φ (deg)")
    ax.set_ylabel(f"valley gap  ω$_{p+1}$ − ω$_{p}$  at K")
    ax.set_title("Fiber orientation tunes the topological valley gap")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig("results/valley_mass.png", dpi=140, bbox_inches="tight")
    print(f"saved results/valley_mass.png  gap(0)={gaps[0]:.4f} "
          f"gap(40)={gaps[-1]:.4f}")


if __name__ == "__main__":
    main()
