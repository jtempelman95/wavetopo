"""
Forced-response valley-Hall waveguide on a C3-clean UNSTRUCTURED (gmsh) mesh.

A finite rectangular patch is meshed with isotropic Delaunay triangles (which,
unlike a structured grid, respect C3 on average so valley kink modes actually
propagate).  A honeycomb of stiff disks is painted onto the mesh by centroid;
the sublattice imbalance (rA,rB) flips across a BENT domain wall, so the two
sides carry opposite valley Chern and the interface guides a kink mode.

We drive a source at one end of the wall and (i) sweep frequency to locate the
guiding band (a transmission PEAK inside the bulk gap), then (ii) show the
mid-gap field turning the corner vs an in-band control.
"""
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topoopt.scalar import MaterialSH
from topoopt.rhombic import _tri_scalar


def rect_mesh(Lx, Ly, h):
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("rect")
    occ = gmsh.model.occ
    occ.addRectangle(0, 0, 0, Lx, Ly); occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.7 * h)
    gmsh.model.mesh.generate(2)
    nt, nc, _ = gmsh.model.mesh.getNodes(); nc = nc.reshape(-1, 3)[:, :2]
    tag2i = {int(t): i for i, t in enumerate(nt)}
    _, enodes = gmsh.model.mesh.getElementsByType(2)
    tris = np.array([[tag2i[int(t)] for t in enodes[3*k:3*k+3]]
                     for k in range(len(enodes)//3)], np.int64)
    gmsh.finalize()
    return nc, tris


def honeycomb_z(cents, wall_x, rA, rB, b=1.0):
    """Paint honeycomb disks; radii chosen by side of the bent wall x=wall_x(y)."""
    s3 = np.sqrt(3.0); a1 = b*np.array([1.5, s3/2]); a2 = b*np.array([1.5, -s3/2])
    A0 = (a1+a2)/3; B0 = 2*(a1+a2)/3
    # lattice sites covering the patch
    xmax, ymax = cents[:, 0].max(), cents[:, 1].max()
    P = range(-1, int(xmax/1.5)+2); Q = range(-int(ymax/0.9)-2, int(ymax/0.9)+2)
    Asite = np.array([A0+p*a1+q*a2 for p in P for q in Q])
    Bsite = np.array([B0+p*a1+q*a2 for p in P for q in Q])
    z = np.zeros(cents.shape[0])
    dA = np.min(np.linalg.norm(cents[:, None, :]-Asite[None], axis=2), axis=1)
    dB = np.min(np.linalg.norm(cents[:, None, :]-Bsite[None], axis=2), axis=1)
    left = cents[:, 0] < np.array([wall_x(y) for y in cents[:, 1]])
    rAe = np.where(left, rA, rB); rBe = np.where(left, rB, rA)
    z[(dA < rAe) | (dB < rBe)] = 1.0
    return z


def assemble(npx, tris, z, mat, omega, eta, sponge):
    aC = (mat.mu_L+mat.mu_T)/2; mum = mat.mu_m
    Nn = npx.shape[0]
    iK = np.kron(tris, np.ones((3, 1), np.int64)).ravel()
    jK = np.kron(tris, np.ones((1, 3), np.int64)).ravel()
    kel = np.zeros((tris.shape[0], 3, 3)); mel = np.zeros_like(kel)
    mref = np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 2]])/12.0
    for e, tri in enumerate(tris):
        xy = npx[tri]
        Kf, ar = _tri_scalar(xy, aC*np.eye(2))
        Km, _ = _tri_scalar(xy, mum*np.eye(2))
        w = z[e]**3
        kel[e] = w*Kf + (1-w)*Km
        rho = z[e]*mat.rho_f + (1-z[e])*mat.rho_m
        mel[e] = rho*mref*ar
    K = sp.csr_matrix((kel.ravel(), (iK, jK)), shape=(Nn, Nn))
    M = sp.csr_matrix((mel.ravel(), (iK, jK)), shape=(Nn, Nn))
    Md = np.asarray(M.sum(1)).ravel()
    C = sp.diags(sponge*Md)
    return K - omega**2*M + 1j*omega*C, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.10)
    ap.add_argument("--Lx", type=float, default=22.0)
    ap.add_argument("--Ly", type=float, default=16.0)
    ap.add_argument("--rA", type=float, default=0.46)
    ap.add_argument("--rB", type=float, default=0.24)
    ap.add_argument("--eta", type=float, default=0.004)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--wmid", type=float, default=2.28)
    ap.add_argument("--wband", type=float, default=2.05)
    ap.add_argument("--out", default="results/valley_wg_gmsh.png")
    args = ap.parse_args()

    mat = MaterialSH(mu_L=6.0, mu_T=2.0, mu_m=1.0, rho_f=1.6, rho_m=1.2)
    npx, tris = rect_mesh(args.Lx, args.Ly, args.h)
    cents = npx[tris].mean(1)
    cx0 = 0.5*args.Lx; y1, y2 = 0.38*args.Ly, 0.62*args.Ly
    xlo, xhi = 0.34*args.Lx, 0.66*args.Lx
    def wall_x(y):
        if y < y1: return xlo
        if y > y2: return xhi
        return xlo + (xhi-xlo)*(y-y1)/(y2-y1)
    z = honeycomb_z(cents, wall_x, args.rA, args.rB)
    print(f"mesh {npx.shape[0]} nodes, {tris.shape[0]} tris, fiber frac {z.mean():.3f}")

    # sponge ramp near outer boundary (nodal)
    xs, ys = npx[:, 0], npx[:, 1]; mg = 1.6
    du = np.maximum.reduce([(mg-xs)/mg, (xs-(args.Lx-mg))/mg,
                            (mg-ys)/mg, (ys-(args.Ly-mg))/mg])
    sponge = np.maximum(0, du)**2*30.0
    # source at bottom end of wall; probe at top end past the bend
    src = int(np.argmin((xs-wall_x(2.0))**2 + (ys-2.0)**2))
    outxy = np.array([wall_x(args.Ly-2.0), args.Ly-2.0])
    probe = np.hypot(xs-outxy[0], ys-outxy[1]) < 1.3

    if args.sweep:
        print("frequency sweep (transmission to far end of wall):")
        for w in np.linspace(2.0, 2.42, 22):
            D, M = assemble(npx, tris, z, mat, w, args.eta, sponge)
            F = np.zeros(npx.shape[0], complex); F[src] = 1.0
            u = spla.spsolve(D.tocsc(), F)
            T = float(np.sum(np.abs(u[probe])**2))
            tag = " <==GAP" if 2.24 < w < 2.34 else ""
            print(f"  w={w:.3f}  T={T:.3e}{tag}")
        return

    def solve(w):
        D, M = assemble(npx, tris, z, mat, w, args.eta, sponge)
        F = np.zeros(npx.shape[0], complex); F[src] = 1.0
        u = spla.spsolve(D.tocsc(), F)
        return u, float(np.sum(np.abs(u[probe])**2))
    u_mid, T_mid = solve(args.wmid)
    u_band, T_band = solve(args.wband)
    print(f"mid-gap T={T_mid:.3e}  in-band T={T_band:.3e}  ratio={T_mid/max(T_band,1e-30):.1f}x")

    wy = np.linspace(0, args.Ly, 100); wx = [wall_x(y) for y in wy]
    fig, ax = plt.subplots(1, 2, figsize=(13, 7.5))
    for a_, u, tag, T in [(ax[0], u_mid, f"mid-gap w={args.wmid}", T_mid),
                          (ax[1], u_band, f"in-band w={args.wband}", T_band)]:
        vmax = np.percentile(np.abs(u), 99.5)
        a_.tripcolor(npx[:, 0], npx[:, 1], tris, np.abs(u), cmap="magma",
                     vmax=vmax, shading="gouraud")
        a_.plot(wx, wy, "c--", lw=1.1, alpha=0.7)
        a_.plot(xs[src], ys[src], "co", ms=9, mfc="none", mew=2)
        a_.add_patch(plt.Circle(outxy, 1.3, ec="lime", fc="none", lw=1.6))
        a_.set_aspect("equal"); a_.set_title(f"{tag}  T={T:.1e}")
    fig.suptitle("Valley-Hall waveguide around a sharp bend (unstructured mesh): "
                 f"mid-gap guiding vs in-band  ({T_mid/max(T_band,1e-30):.0f}x)",
                 y=0.98)
    plt.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
