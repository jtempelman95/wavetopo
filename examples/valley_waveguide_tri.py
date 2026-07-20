"""
Robust valley-Hall waveguide (triangular-rod crystal) around a sharp bend.

A finite patch of the triangular-rod crystal on a C3-clean unstructured (gmsh)
mesh.  Triangular rods are rotated alpha=15deg on one side of a bent domain
wall and alpha=45deg on the other (mirror-partner domains, opposite valley
Chern), so the interface carries a gap-traversing valley kink mode.  We drive a
source at one end of the wall and

  * sweep frequency -> a transmission PEAK inside the bulk gap (the guided band);
  * show the mid-gap field hugging the wall and turning the corner, vs an
    in-band control that radiates.
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
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.7*h)
    gmsh.model.mesh.generate(2)
    nt, nc, _ = gmsh.model.mesh.getNodes(); nc = nc.reshape(-1, 3)[:, :2]
    t2i = {int(t): i for i, t in enumerate(nt)}
    en = gmsh.model.mesh.getElementsByType(2)[1]
    tris = np.array([[t2i[int(en[3*k+r])] for r in range(3)]
                     for k in range(len(en)//3)], np.int64)
    gmsh.finalize()
    return nc, tris


def paint_rods(cents, side_fn, R=0.44, a=1.0):
    """z=1 for element centroids inside a rotated triangular rod.  ``side_fn(r)``
    returns True on the alpha=15 side of the (lattice-aligned) domain wall,
    False on the alpha=45 side."""
    a1 = a*np.array([1.0, 0.0]); a2 = a*np.array([0.5, np.sqrt(3)/2])
    xmax, ymax = cents[:, 0].max(), cents[:, 1].max()
    sites = []
    for i in range(-2, int(xmax/a)+3):
        for j in range(-2, int(ymax/(a*np.sqrt(3)/2))+3):
            sites.append(i*a1 + j*a2 + (a1+a2)/2)   # rod centres at cell centroids
    sites = np.array(sites)
    _, near = cKDTree(sites).query(cents)
    z = np.zeros(cents.shape[0])
    for e in range(cents.shape[0]):
        d = cents[e] - sites[near[e]]
        al = np.deg2rad(15.0 if side_fn(cents[e]) else 45.0)
        v = [R*np.array([np.cos(al+np.pi/2+2*np.pi*k/3),
                         np.sin(al+np.pi/2+2*np.pi*k/3)]) for k in range(3)]
        s = 0
        for k in range(3):
            e1 = v[(k+1) % 3]-v[k]
            s += np.sign(e1[0]*(d[1]-v[k][1]) - e1[1]*(d[0]-v[k][0]))
        if abs(s) == 3:
            z[e] = 1.0
    return z


def assemble(npx, tris, z, muf, omega, eta, sponge):
    Nn = npx.shape[0]
    iK = np.kron(tris, np.ones((3, 1), np.int64)).ravel()
    jK = np.kron(tris, np.ones((1, 3), np.int64)).ravel()
    kel = np.zeros((tris.shape[0], 3, 3)); mel = np.zeros_like(kel)
    mref = np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 2]])/12.0
    for e, tri in enumerate(tris):
        xy = npx[tri]
        Kf, ar = _tri_scalar(xy, muf*np.eye(2))
        Km, _ = _tri_scalar(xy, 1.0*np.eye(2))
        w = z[e]**3
        kel[e] = w*Kf + (1-w)*Km
        rho = z[e]*1.6 + (1-z[e])*1.2
        mel[e] = rho*mref*ar
    K = sp.csr_matrix((kel.ravel(), (iK, jK)), shape=(Nn, Nn))
    M = sp.csr_matrix((mel.ravel(), (iK, jK)), shape=(Nn, Nn))
    Md = np.asarray(M.sum(1)).ravel()
    return K - omega**2*M + 1j*omega*sp.diags(sponge*Md), M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.08)
    ap.add_argument("--Lx", type=float, default=20.0)
    ap.add_argument("--Ly", type=float, default=17.0)
    ap.add_argument("--muf", type=float, default=100.0)
    ap.add_argument("--eta", type=float, default=0.003)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--bend", action="store_true")
    ap.add_argument("--seg2ang", type=float, default=120.0,
                    help="direction (deg) of the wall's 2nd segment")
    ap.add_argument("--wmid", type=float, default=7.28)
    ap.add_argument("--wband", type=float, default=6.4)
    ap.add_argument("--out", default="results/valley_wg_tri.png")
    args = ap.parse_args()

    npx, tris = rect_mesh(args.Lx, args.Ly, args.h)
    cents = npx[tris].mean(1)
    # lattice-aligned domain wall (zigzag).  Straight: along a2 (60 deg).  Bent:
    # a2 up to the middle, then along (-1/2, sqrt3/2) (120 deg) -> a 60 deg kink.
    a2 = np.array([0.5, np.sqrt(3)/2])
    th2 = np.deg2rad(args.seg2ang); u2 = np.array([np.cos(th2), np.sin(th2)])
    start = np.array([0.40*args.Lx, 2.0])
    smid = (args.Ly*0.5 - start[1]) / a2[1]
    corner = start + smid*a2
    if args.bend:
        L2 = (args.Ly-2.0 - corner[1]) / u2[1] if abs(u2[1]) > 1e-3 else \
            0.30*args.Lx / max(abs(u2[0]), 1e-3)
        top = corner + L2*u2
        poly = np.array([start, corner, top])
    else:
        poly = np.array([start, start + ((args.Ly-2.0-start[1])/a2[1])*a2])

    def signed_side(r):
        bestd, sgn = 1e18, 1.0
        for k in range(len(poly)-1):
            p0, seg = poly[k], poly[k+1]-poly[k]
            t = np.clip(np.dot(r-p0, seg)/np.dot(seg, seg), 0, 1)
            d = np.hypot(*(r-(p0+t*seg)))
            if d < bestd:
                bestd = d; sgn = np.sign(seg[0]*(r[1]-p0[1])-seg[1]*(r[0]-p0[0]))
        return sgn < 0
    z = paint_rods(cents, signed_side, R=0.44)
    print(f"mesh {npx.shape[0]} nodes {tris.shape[0]} tris, fiber frac {z.mean():.3f}")

    xs, ys = npx[:, 0], npx[:, 1]; mg = 1.8
    du = np.maximum.reduce([(mg-xs)/mg, (xs-(args.Lx-mg))/mg,
                            (mg-ys)/mg, (ys-(args.Ly-mg))/mg])
    sponge = np.maximum(0, du)**2*25.0
    srcxy = poly[0] + 0.6*(poly[1]-poly[0])/np.hypot(*(poly[1]-poly[0]))
    outxy = poly[-1] - 0.6*(poly[-1]-poly[-2])/np.hypot(*(poly[-1]-poly[-2]))
    src = int(np.argmin((xs-srcxy[0])**2 + (ys-srcxy[1])**2))
    probe = np.hypot(xs-outxy[0], ys-outxy[1]) < 1.5
    wallpoly = poly

    if args.sweep:
        print("frequency sweep (transmission to far end of wall):")
        for w in np.linspace(6.4, 8.0, 25):
            D, M = assemble(npx, tris, z, args.muf, w, args.eta, sponge)
            F = np.zeros(npx.shape[0], complex); F[src] = 1.0
            u = spla.spsolve(D.tocsc(), F)
            T = float(np.sum(np.abs(u[probe])**2))
            print(f"  w={w:.3f}  T={T:.3e}" + ("  <==GAP" if 7.02 < w < 7.55 else ""))
        return

    def solve(w):
        D, M = assemble(npx, tris, z, args.muf, w, args.eta, sponge)
        F = np.zeros(npx.shape[0], complex); F[src] = 1.0
        u = spla.spsolve(D.tocsc(), F)
        return u, float(np.sum(np.abs(u[probe])**2))
    u_mid, T_mid = solve(args.wmid); u_band, T_band = solve(args.wband)
    print(f"mid-gap T={T_mid:.3e}  in-band T={T_band:.3e}  ratio={T_mid/max(T_band,1e-30):.1f}x")

    fig, ax = plt.subplots(1, 2, figsize=(13, 8))
    for a_, u, tag, T in [(ax[0], u_mid, f"mid-gap w={args.wmid}", T_mid),
                          (ax[1], u_band, f"in-band w={args.wband}", T_band)]:
        vmax = np.percentile(np.abs(u), 99.5)
        a_.tripcolor(npx[:, 0], npx[:, 1], tris, np.abs(u), cmap="magma",
                     vmax=vmax, shading="gouraud")
        a_.plot(wallpoly[:, 0], wallpoly[:, 1], "c--", lw=1.1, alpha=0.7)
        a_.plot(xs[src], ys[src], "co", ms=9, mfc="none", mew=2)
        a_.add_patch(plt.Circle(outxy, 1.5, ec="lime", fc="none", lw=1.6))
        a_.set_aspect("equal"); a_.set_title(f"{tag}  T={T:.1e}")
    fig.suptitle("Triangular-rod valley waveguide around a sharp bend: "
                 f"mid-gap guiding vs in-band  ({T_mid/max(T_band,1e-30):.0f}x)", y=0.98)
    plt.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
