"""
Topological valley-Hall EDGE TRANSPORT robustness demo.

Tile the co-optimized honeycomb cell into a finite domain.  A valley domain
wall (fiber tilt +theta above / -theta below -> opposite valley Chern) follows
a Z-shaped path with two sharp bends.  A time-harmonic source at one end of the
wall excites the topological kink mode; we compare:

  (1) TOPOLOGICAL  : anisotropic fiber valley wall  -> wave hugs the wall and
                     turns the corners (backscatter-immune).
  (2) ISOTROPIC    : same geometry, fiber replaced by isotropic stiff phase
                     (no valley gap) -> no guided mode, energy disperses.

Reports transmission to an output box past the bends and saves |u| fields.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.cfrp import QuadMesh, CSRBFMapping, Material
from wavetopo.cfrp_problem import density_filter
from wavetopo.harmonic import HarmonicValley


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
    P = density_filter(mesh_c, 0.12, period=(Lx_c, Ly_c))
    y = (P @ d["z"]).reshape(nely_c, nelx_c)
    theta = rbf.theta(d["th"]).reshape(nely_c, nelx_c)
    return y, theta, (nelx_c, nely_c, Lx_c, Ly_c), float(d["sigma"])


def wall_y(x, x1, x2, ylo, yhi):
    """Z-shaped wall: flat ylo, ramp up, flat yhi (two sharp bends)."""
    if x < x1:
        return ylo
    if x > x2:
        return yhi
    return ylo + (yhi - ylo) * (x - x1) / (x2 - x1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/valley_cooptimized.npz")
    ap.add_argument("--ncx", type=int, default=9)
    ap.add_argument("--ncy", type=int, default=6)
    ap.add_argument("--omega", type=float, default=0.0, help="0 = mid-gap")
    ap.add_argument("--out", default="results/valley_edge.png")
    args = ap.parse_args()

    yc, thc, (nelx_c, nely_c, Lx_c, Ly_c), sigma = load_cell(args.npz)
    omega = args.omega or np.sqrt(sigma)
    print(f"cell {nelx_c}x{nely_c}, omega(mid-gap)={omega:.3f}")

    ncx, ncy = args.ncx, args.ncy
    nelx, nely = nelx_c * ncx, nely_c * ncy
    Lx, Ly = Lx_c * ncx, Ly_c * ncy
    mesh = QuadMesh(nelx, nely, Lx, Ly)
    print(f"domain {nelx}x{nely}={mesh.N} elems, ndof={mesh.ndof}, "
          f"{Lx:.1f} x {Ly:.1f}")

    # tile geometry + fiber
    z = np.tile(yc, (ncy, ncx)).ravel()
    theta_cell = np.tile(thc, (ncy, ncx))           # (nely, nelx)

    # Z-shaped valley domain wall: sign = +1 above wall, -1 below
    x1, x2 = 0.40 * Lx, 0.60 * Lx
    ylo, yhi = (ncy // 2 - 1) * Ly_c, (ncy // 2 + 1) * Ly_c
    sign = np.ones((nely, nelx))
    for j in range(nely):
        ey = (j + 0.5) * mesh.dy
        for i in range(nelx):
            ex = (i + 0.5) * mesh.dx
            sign[j, i] = 1.0 if ey > wall_y(ex, x1, x2, ylo, yhi) else -1.0
    theta = (theta_cell * sign).ravel()

    # absorbing sponge rim
    marg = 0.9 * Lx_c
    cx = mesh.cx; cy = mesh.cy
    d = np.maximum.reduce([(marg - cx) / marg, (cx - (Lx - marg)) / marg,
                           (marg - cy) / marg, (cy - (Ly - marg)) / marg])
    cspg = np.maximum(0.0, d)**2 * 60.0

    # source: vertical point force at the left end of the wall
    sx_pos, sy_pos = 1.1 * Lx_c, ylo
    src_node = mesh_node_near(mesh, sx_pos, sy_pos)
    F = np.zeros(mesh.ndof, dtype=complex)
    F[2 * src_node + 1] = 1.0

    # output box: past the bend, right end at the upper wall level
    box = (Lx - 2.2 * Lx_c, Lx - 0.9 * Lx_c, yhi - Ly_c, yhi + Ly_c)

    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    hv = HarmonicValley(mesh, mat, omega=omega, eta=0.03)
    hv.set_sponge(cspg)
    hv.set_load(F)

    fields = {}
    trans = {}
    for tag, iso in [("topological", False), ("isotropic", True)]:
        hv.set_design(z, theta, isotropic=iso)
        hv.solve()
        fields[tag] = nodal_mag(hv.u, mesh)
        trans[tag] = hv.energy_in_box(*box)
        print(f"  {tag:12s}: transmission(output box) = {trans[tag]:.3e}")
    ratio = trans["topological"] / max(trans["isotropic"], 1e-30)
    print(f"  -> topological / isotropic transmission = {ratio:.1f}x")

    # ---- figure ----
    vmax = np.percentile(fields["topological"], 99.6)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    xw = np.linspace(0, Lx, 200)
    yw = [wall_y(x, x1, x2, ylo, yhi) for x in xw]
    for a_, tag in zip(ax, ["topological", "isotropic"]):
        im = a_.imshow(fields[tag], origin="lower", cmap="magma", vmax=vmax,
                       extent=[0, Lx, 0, Ly], aspect="equal")
        a_.plot(xw, yw, "c--", lw=1.2, alpha=0.7)
        a_.plot(sx_pos, sy_pos, "co", ms=9, mfc="none", mew=2)
        a_.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0],
                                   box[3] - box[2], ec="lime", fc="none", lw=1.6))
        a_.set_title(f"{tag}   (T={trans[tag]:.1e})")
        plt.colorbar(im, ax=a_, fraction=0.035)
    fig.suptitle("Valley-Hall edge transport around sharp bends: topological "
                 f"vs isotropic  (T ratio {ratio:.0f}x)", y=1.0, fontsize=14)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


def mesh_node_near(mesh, x, y):
    xs, ys = mesh.node_xy()
    return int(np.argmin((xs - x)**2 + (ys - y)**2))


def nodal_mag(u, mesh):
    ux = u[0::2]; uy = u[1::2]
    return np.sqrt(np.abs(ux)**2 + np.abs(uy)**2).reshape(mesh.nely + 1,
                                                          mesh.nelx + 1)


if __name__ == "__main__":
    main()
