"""
Step 1 of the valley-Hall build: honeycomb lattice of fiber-composite "atoms"
in a soft matrix, on an orthogonal 4-atom rectangular supercell (square-mesh
compatible).  Search for the Dirac cone -- the symmetry-protected band
degeneracy that valley topology is built on.

Honeycomb in a rectangular cell (bond length a):  Lx = sqrt(3) a, Ly = 3 a,
four atoms (2 A-sublattice, 2 B-sublattice) at the fractional positions below.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topoopt.cfrp import Material, QuadMesh, CSRBFMapping
from topoopt.bloch import BlochProblem, k_segment


A_FRAC = [(0.0, 0.0), (0.5, 0.5)]            # A sublattice
B_FRAC = [(0.0, 1 / 3), (0.5, 5 / 6)]        # B sublattice


def honeycomb_density(mesh, radius, period):
    """z=1 inside disks at the 4 honeycomb sites (toroidal), else 0."""
    z = np.zeros(mesh.N)
    Lx, Ly = period
    centers = [(fx * Lx, fy * Ly) for (fx, fy) in A_FRAC + B_FRAC]
    for (cx, cy) in centers:
        dx = mesh.cx - cx
        dy = mesh.cy - cy
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)
        z[np.sqrt(dx**2 + dy**2) < radius] = 1.0
    return z


def sublattice_angle(mesh, period, angA, angB, radius):
    """Per-element fiber angle: angA on A-disks, angB on B-disks (else 0).

    Different A/B angles break inversion symmetry -> Dirac mass."""
    th = np.zeros(mesh.N)
    Lx, Ly = period
    for frac, ang in [(A_FRAC, angA), (B_FRAC, angB)]:
        for (fx, fy) in frac:
            cx, cy = fx * Lx, fy * Ly
            dx = mesh.cx - cx; dy = mesh.cy - cy
            dx -= Lx * np.round(dx / Lx); dy -= Ly * np.round(dy / Ly)
            th[np.sqrt(dx**2 + dy**2) < radius] = ang
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=30)
    ap.add_argument("--radius", type=float, default=0.33)
    ap.add_argument("--nbands", type=int, default=8)
    ap.add_argument("--out", default="results/valley_bands.png")
    args = ap.parse_args()

    a = 1.0
    Lx, Ly = np.sqrt(3) * a, 3 * a
    ny = int(round(args.nx * Ly / Lx))
    mesh = QuadMesh(args.nx, ny, Lx, Ly)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    # orientation design fields are periodic (tileable cell)
    xs = np.linspace(0, Lx, 4, endpoint=False)
    ys = np.linspace(0, Ly, 6, endpoint=False)
    SX, SY = np.meshgrid(xs, ys)
    rbf = CSRBFMapping(mesh, (SX.ravel(), SY.ravel()), r_s=0.6, period=(Lx, Ly))
    bp = BlochProblem(mesh, mat, rbf, penal=3.0)

    z = honeycomb_density(mesh, args.radius, (Lx, Ly))
    print(f"mesh {mesh.nelx}x{mesh.nely}  fill fraction={z.mean():.3f}")

    # symmetric honeycomb: uniform fibers (theta=0) -> expect Dirac cone
    bp.assemble(z, np.zeros(rbf.M))

    # BZ of rectangular cell; the hexagonal K folds to (2pi/Lx*?, ...).
    # Scan a 2D grid to locate the minimum gap between bands (nb/2-1, nb/2).
    nb = args.nbands
    nkx, nky = 21, 21
    kxs = np.linspace(0, 2 * np.pi / Lx, nkx)
    kys = np.linspace(0, 2 * np.pi / Ly, nky)
    # focus on the two "Dirac" bands: pick a mid pair (acoustic branches are 0,1)
    pair = (3, 4)
    gap = np.full((nkx, nky), np.nan)
    allw = np.zeros((nkx, nky, nb))
    for i, kx in enumerate(kxs):
        for j, ky in enumerate(kys):
            w = bp.bands_at_k(kx, ky, nb)
            allw[i, j] = w
            gap[i, j] = w[pair[1]] - w[pair[0]]
    imin = np.unravel_index(np.nanargmin(gap), gap.shape)
    print(f"min gap between bands {pair} = {gap[imin]:.4f} "
          f"at k=({kxs[imin[0]]:.3f},{kys[imin[1]]:.3f})")
    # also report min gap for several adjacent pairs to find the Dirac pair
    for p in range(1, nb - 1):
        g = np.min(allw[:, :, p + 1] - allw[:, :, p])
        print(f"  pair {p}-{p+1}: min gap over BZ = {g:.4f}")

    # band structure along Gamma-X-M-Y-Gamma
    G = np.array([0, 0]); X = np.array([np.pi / Lx, 0])
    Mpt = np.array([np.pi / Lx, np.pi / Ly]); Y = np.array([0, np.pi / Ly])
    path = (k_segment(G, X, 30) + k_segment(X, Mpt, 30)[1:]
            + k_segment(Mpt, Y, 30)[1:] + k_segment(Y, G, 30)[1:])
    bands = bp.band_structure(path, nb)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(bands, "k-", lw=1)
    ax[0].set_title("honeycomb bands (uniform fibers)"); ax[0].set_ylabel(r"$\omega$")
    ax[0].set_xticks([0, 29, 58, 87, 116]); ax[0].set_xticklabels(
        [r"$\Gamma$", "X", "M", "Y", r"$\Gamma$"])
    im = ax[1].imshow(gap.T, origin="lower", cmap="viridis",
                      extent=[0, kxs[-1], 0, kys[-1]], aspect="auto")
    ax[1].set_title(f"gap between bands {pair} (dark = Dirac touching)")
    ax[1].set_xlabel("kx"); ax[1].set_ylabel("ky")
    plt.colorbar(im, ax=ax[1])
    plt.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
