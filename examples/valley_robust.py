"""
Topological protection test: valley waveguide vs. wall defects.

The mirror-partner (alpha = 15deg | 45deg) valley-Hall domain wall of the
triangular-rod crystal is deliberately damaged and the mid-gap kink-mode
transmission is measured:

  * pristine straight wall,
  * one rod removed on the wall (vacancy),
  * a cluster of rods next to the wall randomly rotated (orientation
    disorder, +/- 12 deg).

Because backscattering requires inter-valley (K -> K') coupling that smooth
defects do not supply, the guided transmission should survive within a small
factor -- the practical signature of topological protection, and the property
that distinguishes the kink mode from an ordinary defect-line guide.

    python examples/valley_robust.py --h 0.09
"""
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.rhombic import _tri_scalar


def rect_mesh(Lx, Ly, h):
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("r"); gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.7 * h)
    gmsh.model.mesh.generate(2)
    nt, nc, _ = gmsh.model.mesh.getNodes(); nc = nc.reshape(-1, 3)[:, :2]
    t2i = {int(t): i for i, t in enumerate(nt)}
    en = gmsh.model.mesh.getElementsByType(2)[1]
    tris = np.array([[t2i[int(en[3 * k + r])] for r in range(3)]
                     for k in range(len(en) // 3)], np.int64)
    gmsh.finalize()
    return nc, tris


def paint_rods(cents, side_fn, *, R=0.44, a=1.0, skip_sites=(),
               angle_override=None):
    """z=1 inside each triangular rod.  ``skip_sites``: rod-centre coords to
    leave out (vacancies).  ``angle_override``: {site_index: alpha_deg}."""
    a1 = a * np.array([1.0, 0.0]); a2 = a * np.array([0.5, np.sqrt(3) / 2])
    xmax, ymax = cents[:, 0].max(), cents[:, 1].max()
    sites = []
    for i in range(-2, int(xmax / a) + 3):
        for j in range(-2, int(ymax / (a * np.sqrt(3) / 2)) + 3):
            sites.append(i * a1 + j * a2 + (a1 + a2) / 2)
    sites = np.array(sites)
    skip = set()
    for sxy in skip_sites:
        skip.add(int(np.argmin(np.hypot(sites[:, 0] - sxy[0],
                                        sites[:, 1] - sxy[1]))))
    over = {}
    if angle_override:
        for sxy, al in angle_override.items():
            over[int(np.argmin(np.hypot(sites[:, 0] - sxy[0],
                                        sites[:, 1] - sxy[1])))] = al
    _, near = cKDTree(sites).query(cents)
    z = np.zeros(cents.shape[0])
    for e in range(cents.shape[0]):
        si = near[e]
        if si in skip:
            continue
        d = cents[e] - sites[si]
        if si in over:
            al = np.deg2rad(over[si])
        else:
            al = np.deg2rad(15.0 if side_fn(cents[e]) else 45.0)
        v = [R * np.array([np.cos(al + np.pi / 2 + 2 * np.pi * k / 3),
                           np.sin(al + np.pi / 2 + 2 * np.pi * k / 3)])
             for k in range(3)]
        s = 0
        for k in range(3):
            e1 = v[(k + 1) % 3] - v[k]
            s += np.sign(e1[0] * (d[1] - v[k][1]) - e1[1] * (d[0] - v[k][0]))
        if abs(s) == 3:
            z[e] = 1.0
    return z, sites


def assemble(npx, tris, z, muf, omega, eta, sponge):
    Nn = npx.shape[0]
    iK = np.kron(tris, np.ones((3, 1), np.int64)).ravel()
    jK = np.kron(tris, np.ones((1, 3), np.int64)).ravel()
    kel = np.zeros((tris.shape[0], 3, 3)); mel = np.zeros_like(kel)
    mref = np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    for e, tri in enumerate(tris):
        xy = npx[tri]
        Kf, ar = _tri_scalar(xy, muf * np.eye(2))
        Km, _ = _tri_scalar(xy, 1.0 * np.eye(2))
        w = z[e]**3
        kel[e] = w * Kf + (1 - w) * Km
        rho = z[e] * 1.6 + (1 - z[e]) * 1.2
        mel[e] = rho * mref * ar
    K = sp.csr_matrix((kel.ravel(), (iK, jK)), shape=(Nn, Nn))
    M = sp.csr_matrix((mel.ravel(), (iK, jK)), shape=(Nn, Nn))
    Md = np.asarray(M.sum(1)).ravel()
    D = (1 + 1j * eta) * K - omega**2 * M + 1j * omega * sp.diags(sponge * Md)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.09)
    ap.add_argument("--Lx", type=float, default=20.0)
    ap.add_argument("--Ly", type=float, default=17.0)
    ap.add_argument("--muf", type=float, default=100.0)
    ap.add_argument("--eta", type=float, default=0.003)
    ap.add_argument("--wmid", type=float, default=7.28)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="results/valley_robust.png")
    args = ap.parse_args()

    npx, tris = rect_mesh(args.Lx, args.Ly, args.h)
    cents = npx[tris].mean(1)
    print(f"mesh: {npx.shape[0]} nodes, {tris.shape[0]} tris")

    # straight lattice-aligned (zigzag) wall along a2
    a2 = np.array([0.5, np.sqrt(3) / 2])
    start = np.array([0.40 * args.Lx, 2.0])
    end = start + ((args.Ly - 2.0 - start[1]) / a2[1]) * a2
    poly = np.array([start, end])
    mid = 0.5 * (start + end)

    def side(r):
        seg = poly[1] - poly[0]
        return (seg[0] * (r[1] - poly[0][1])
                - seg[1] * (r[0] - poly[0][0])) < 0

    rng = np.random.default_rng(args.seed)
    # disorder cluster: 4 sites nearest the wall midpoint, random rotations
    a1v = np.array([1.0, 0.0])
    near_sites = [mid + dv for dv in
                  (0.6 * a1v, -0.6 * a1v, 0.9 * a2, -0.9 * a2)]
    disorder = {tuple(s): float(rng.uniform(-12, 12)
                                + (15.0 if side(s) else 45.0))
                for s in near_sites}

    configs = {
        "pristine wall": dict(skip=(), over=None),
        "missing rod (vacancy)": dict(skip=(mid,), over=None),
        "orientation disorder (+/-12 deg)": dict(skip=(), over=disorder),
    }

    xs, ys = npx[:, 0], npx[:, 1]; mg = 1.8
    du = np.maximum.reduce([(mg - xs) / mg, (xs - (args.Lx - mg)) / mg,
                            (mg - ys) / mg, (ys - (args.Ly - mg)) / mg])
    sponge = np.maximum(0, du)**2 * 25.0
    srcxy = poly[0] + 0.6 * (poly[1] - poly[0]) / np.hypot(*(poly[1] - poly[0]))
    outxy = poly[1] - 0.6 * (poly[1] - poly[0]) / np.hypot(*(poly[1] - poly[0]))
    src = int(np.argmin((xs - srcxy[0])**2 + (ys - srcxy[1])**2))
    probe = np.hypot(xs - outxy[0], ys - outxy[1]) < 1.5

    results = {}
    for name, cfg in configs.items():
        z, _ = paint_rods(cents, side, skip_sites=cfg['skip'],
                          angle_override=cfg['over'])
        D = assemble(npx, tris, z, args.muf, args.wmid, args.eta, sponge)
        F = np.zeros(npx.shape[0], complex); F[src] = 1.0
        u = spla.spsolve(D.tocsc(), F)
        T = float(np.sum(np.abs(u[probe])**2))
        results[name] = dict(u=u, T=T, z=z)
        print(f"{name:38s}  T={T:.3e}")

    T0 = results["pristine wall"]['T']
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))
    for a_, (name, r) in zip(ax, results.items()):
        vmax = np.percentile(np.abs(r['u']), 99.5)
        a_.tripcolor(xs, ys, tris, np.abs(r['u']), cmap="magma", vmax=vmax,
                     shading="gouraud")
        a_.plot(poly[:, 0], poly[:, 1], "c--", lw=1.0, alpha=0.6)
        a_.plot(xs[src], ys[src], "co", ms=9, mfc="none", mew=2)
        a_.add_patch(plt.Circle(outxy, 1.5, ec="lime", fc="none", lw=1.6))
        if name != "pristine wall":
            a_.add_patch(plt.Circle(mid, 1.6, ec="w", fc="none", lw=1.2,
                                    ls=":"))
        a_.set_aspect("equal")
        a_.set_title(f"{name}\nT = {r['T']:.2e}  ({r['T']/T0*100:.0f}% of "
                     "pristine)", fontsize=11)
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("Topological protection of the valley kink mode: mid-gap "
                 f"transmission survives wall damage (omega={args.wmid})",
                 y=0.99, fontsize=13)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)
    np.savez(args.out.replace(".png", ".npz"),
             names=list(results.keys()),
             T=[results[n]['T'] for n in results])


if __name__ == "__main__":
    main()
