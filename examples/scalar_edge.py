"""
Scalar (antiplane-shear) valley-Hall EDGE TRANSPORT.

Tile the co-optimized scalar honeycomb cell; create a valley domain wall
(fiber theta -> -theta across it, opposite valley Chern).  Drive a harmonic
source at the wall at mid-gap and compare topological vs isotropic control.
Supports a straight wall and a Z-bent wall (sharp corners).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topoopt.cfrp import QuadMesh, CSRBFMapping
from topoopt.cfrp_problem import density_filter
from topoopt.scalar import MaterialSH, ScalarHarmonic


def load_cell(npz):
    d = np.load(npz)
    nelx_c, nely_c = int(d["nelx"]), int(d["nely"])
    Lx_c, Ly_c = float(d["Lx"]), float(d["Ly"])
    mesh_c = QuadMesh(nelx_c, nely_c, Lx_c, Ly_c)
    sx = np.linspace(0, Lx_c, 6, endpoint=False)
    sy = np.linspace(0, Ly_c, 10, endpoint=False)
    SX, SY = np.meshgrid(sx, sy)
    rbf = CSRBFMapping(mesh_c, (SX.ravel(), SY.ravel()), r_s=0.55,
                       period=(Lx_c, Ly_c))
    P = density_filter(mesh_c, 0.11, period=(Lx_c, Ly_c))
    y = (P @ d["z"]).reshape(nely_c, nelx_c)
    theta = rbf.theta(d["th"]).reshape(nely_c, nelx_c)
    return y, theta, (nelx_c, nely_c, Lx_c, Ly_c), float(d["omega"])


def wall_x(y, kind, Ly, Lx_c, ncx):
    """Vertical valley domain wall x=wall_x(y).  The two valleys are separated
    in ky, so the wall must run along y to keep them distinct (zigzag-type)."""
    xlo, xhi = (ncx // 2 - 1) * Lx_c, (ncx // 2 + 1) * Lx_c
    if kind == "straight":
        return (ncx // 2) * Lx_c
    y1, y2 = 0.40 * Ly, 0.58 * Ly
    if y < y1:
        return xlo
    if y > y2:
        return xhi
    return xlo + (xhi - xlo) * (y - y1) / (y2 - y1)


def nmag(u, mesh):
    return np.abs(u).reshape(mesh.nely + 1, mesh.nelx + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/scalar_complete.npz")
    ap.add_argument("--ncx", type=int, default=10)
    ap.add_argument("--ncy", type=int, default=6)
    ap.add_argument("--wall", choices=["straight", "zigzag"], default="straight")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    yc, thc, (nelx_c, nely_c, Lx_c, Ly_c), omega = load_cell(args.npz)
    ncx, ncy = args.ncx, args.ncy
    mesh = QuadMesh(nelx_c * ncx, nely_c * ncy, Lx_c * ncx, Ly_c * ncy)
    Lx, Ly = mesh.Lx, mesh.Ly
    print(f"domain {mesh.nelx}x{mesh.nely}, nnode={mesh.nnode}, omega={omega:.3f}")

    z = np.tile(yc, (ncy, ncx)).ravel()
    th_tile = np.tile(thc, (ncy, ncx))
    ex = (np.arange(mesh.nelx) + 0.5) * mesh.dx
    ey = (np.arange(mesh.nely) + 0.5) * mesh.dy
    EX, EY = np.meshgrid(ex, ey)
    wx = np.array([[wall_x(y, args.wall, Ly, Lx_c, ncx) for _ in ex]
                   for y in ey])
    sign = np.where(EX > wx, 1.0, -1.0)
    theta = (th_tile * sign).ravel()
    xwall0 = wall_x(1.2 * Ly_c, args.wall, Ly, Lx_c, ncx)        # bottom
    xwallT = wall_x(Ly - 1.5 * Ly_c, args.wall, Ly, Lx_c, ncx)   # top

    # sponge rim
    marg = 0.9 * Lx_c
    dd = np.maximum.reduce([(marg - mesh.cx) / marg, (mesh.cx - (Lx - marg)) / marg,
                            (marg - mesh.cy) / marg, (mesh.cy - (Ly - marg)) / marg])
    cspg = np.maximum(0, dd)**2 * 50.0

    mat = MaterialSH(mu_L=6.0, mu_T=2.0, mu_m=1.0, rho_f=1.6, rho_m=1.2)
    hv = ScalarHarmonic(mesh, mat, omega=omega, eta=0.02)
    hv.set_sponge(cspg)
    # source at bottom end of the (vertical) wall
    xs, ys = mesh.node_xy()
    src = int(np.argmin((xs - xwall0)**2 + (ys - 1.2 * Ly_c)**2))
    F = np.zeros(mesh.nnode, dtype=complex); F[src] = 1.0
    hv.set_load(F)
    box = (xwallT - Lx_c, xwallT + Lx_c, Ly - 2.4 * Ly_c, Ly - 0.9 * Ly_c)

    out = args.out or f"results/scalar_edge_{args.wall}.png"
    fields, T = {}, {}
    for tag, iso in [("topological", False), ("isotropic", True)]:
        hv.set_design(z, theta, isotropic=iso)
        hv.solve()
        fields[tag] = nmag(hv.u, mesh)
        T[tag] = hv.energy_in_box(*box)
    ratio = T["topological"] / max(T["isotropic"], 1e-30)
    print(f"  T topo={T['topological']:.3e}  iso={T['isotropic']:.3e}  "
          f"ratio={ratio:.1f}x")

    vmax = np.percentile(fields["topological"], 99.6)
    yw = np.linspace(0, Ly, 250)
    xw = [wall_x(y, args.wall, Ly, Lx_c, ncx) for y in yw]
    fig, ax = plt.subplots(1, 2, figsize=(13, 7))
    for a_, tag in zip(ax, ["topological", "isotropic"]):
        im = a_.imshow(fields[tag], origin="lower", cmap="magma", vmax=vmax,
                       extent=[0, Lx, 0, Ly], aspect="equal")
        a_.plot(xw, yw, "c--", lw=1, alpha=0.6)
        a_.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0],
                                   box[3] - box[2], ec="lime", fc="none", lw=1.5))
        a_.set_title(f"{tag}  (T={T[tag]:.1e})")
        plt.colorbar(im, ax=a_, fraction=0.035)
    fig.suptitle(f"Scalar valley-Hall edge transport ({args.wall} wall): "
                 f"topological vs isotropic  ({ratio:.0f}x)", y=1.0, fontsize=13)
    plt.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
