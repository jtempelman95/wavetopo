"""
Visualize the honeycomb fiber-composite metamaterial: unit-cell GEOMETRY
(the "atoms") together with the continuous fiber TOOLPATHS, tiled to show the
lattice.

The A and B sublattices carry opposite fiber tilt (+phi / -phi) -- this is the
inversion-symmetry-breaking "Dirac mass" that opens the valley gap.  We use a
smooth, periodic CS-RBF orientation field so the toolpaths are continuous and
manufacturable (the fibers curve gently between sublattices).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.cfrp import QuadMesh, CSRBFMapping
from examples.valley_hall import honeycomb_density, A_FRAC, B_FRAC


def sublattice_support_angles(SX, SY, period, phi):
    """+phi near A sites, -phi near B sites (toroidal), smooth via CS-RBF."""
    Lx, Ly = period
    A = [(fx * Lx, fy * Ly) for fx, fy in A_FRAC]
    B = [(fx * Lx, fy * Ly) for fx, fy in B_FRAC]

    def mind(px, py, pts):
        d = []
        for (cx, cy) in pts:
            dx = px - cx; dy = py - cy
            dx -= Lx * np.round(dx / Lx); dy -= Ly * np.round(dy / Ly)
            d.append(np.hypot(dx, dy))
        return np.min(d, axis=0)

    dA = mind(SX, SY, A); dB = mind(SX, SY, B)
    return np.where(dA < dB, phi, -phi)


def streamlines_tiled(ax, mesh, theta, z, period, ntile=3, dens_thresh=0.5):
    Lx, Ly = period
    tg = theta.reshape(mesh.nely, mesh.nelx)
    zg = z.reshape(mesh.nely, mesh.nelx)
    # tile
    TG = np.tile(tg, (ntile, ntile))
    ZG = np.tile(zg, (ntile, ntile))
    ny, nx = TG.shape
    xs = (np.arange(nx) + 0.5) * mesh.dx
    ys = (np.arange(ny) + 0.5) * mesh.dy
    # atoms background
    ax.imshow(ZG, origin="lower", cmap="Greys", vmin=0, vmax=1.5,
              extent=[0, ntile * Lx, 0, ntile * Ly], aspect="equal", alpha=0.9)
    u = np.cos(TG); v = np.sin(TG)
    ax.streamplot(xs, ys, u, v, density=3.0, color=np.rad2deg(TG),
                  cmap="coolwarm", linewidth=0.8, arrowsize=1e-6)
    # cell boundaries
    for i in range(1, ntile):
        ax.axhline(i * Ly, color="0.5", lw=0.5, ls="--")
        ax.axvline(i * Lx, color="0.5", lw=0.5, ls="--")
    ax.set_xlim(0, ntile * Lx); ax.set_ylim(0, ntile * Ly)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--radius", type=float, default=0.33)
    ap.add_argument("--phi", type=float, default=35.0, help="A/B fiber tilt deg")
    ap.add_argument("--out", default="results/valley_lattice.png")
    args = ap.parse_args()

    a = 1.0
    Lx, Ly = np.sqrt(3) * a, 3 * a
    ny = int(round(args.nx * Ly / Lx))
    mesh = QuadMesh(args.nx, ny, Lx, Ly)
    z = honeycomb_density(mesh, args.radius, (Lx, Ly))

    # smooth periodic orientation field, +phi(A)/-phi(B)
    sxs = np.linspace(0, Lx, 6, endpoint=False)
    sys = np.linspace(0, Ly, 10, endpoint=False)
    SXg, SYg = np.meshgrid(sxs, sys)
    rbf = CSRBFMapping(mesh, (SXg.ravel(), SYg.ravel()), r_s=0.55,
                       period=(Lx, Ly))
    th_hat = sublattice_support_angles(SXg.ravel(), SYg.ravel(), (Lx, Ly),
                                       np.deg2rad(args.phi))
    theta = rbf.theta(th_hat)

    A = [(fx * Lx, fy * Ly) for fx, fy in A_FRAC]
    B = [(fx * Lx, fy * Ly) for fx, fy in B_FRAC]

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    # single cell: atoms + sublattice markers + toolpaths
    zg = z.reshape(mesh.nely, mesh.nelx)
    ax[0].imshow(zg, origin="lower", cmap="Greys", vmin=0, vmax=1.5,
                 extent=[0, Lx, 0, Ly], aspect="equal", alpha=0.9)
    xs = (np.arange(mesh.nelx) + 0.5) * mesh.dx
    ys = (np.arange(mesh.nely) + 0.5) * mesh.dy
    tg = theta.reshape(mesh.nely, mesh.nelx)
    strm = ax[0].streamplot(xs, ys, np.cos(tg), np.sin(tg), density=2.2,
                            color=np.rad2deg(tg), cmap="coolwarm",
                            linewidth=1.0, arrowsize=1e-6)
    plt.colorbar(strm.lines, ax=ax[0], fraction=0.046, label="fiber angle (°)")
    for (cx, cy) in A:
        ax[0].plot(cx, cy, "o", mfc="dodgerblue", mec="k", ms=11)
    for (cx, cy) in B:
        ax[0].plot(cx, cy, "s", mfc="orange", mec="k", ms=10)
    ax[0].set_xlim(0, Lx); ax[0].set_ylim(0, Ly)
    ax[0].set_title(f"unit cell: atoms + fiber toolpaths "
                    f"(A=+{args.phi:.0f}°, B=-{args.phi:.0f}°)")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y")

    streamlines_tiled(ax[1], mesh, theta, z, (Lx, Ly), ntile=3)
    ax[1].set_title("3×3 lattice with continuous fiber toolpaths")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("y")

    fig.suptitle("Honeycomb fiber-composite valley-Hall metamaterial: "
                 "geometry + toolpath", y=0.99, fontsize=14)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out, " fill=", round(float(z.mean()), 3))


if __name__ == "__main__":
    main()
