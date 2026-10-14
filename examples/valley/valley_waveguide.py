"""
Topological valley-Hall waveguide: robust guiding around a sharp bend.

A finite patch of the rhombic honeycomb (P1 triangular mesh) carries a BENT
domain wall between an A>B domain and its inversion partner B>A.  A time-
harmonic source at one end of the wall, driven at mid-gap, launches the valley
kink mode.  We compare:

  (i)  mid-gap drive  -> energy hugs the wall and turns the corner (topological
       guiding);
  (ii) in-band drive (frequency inside the bulk bands) -> energy radiates into
       the bulk, little reaches the far end.

Transmission is the field energy in a probe box past the bend; the ratio
T(mid-gap)/T(in-band) quantifies the topological robustness.
"""
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.scalar import MaterialSH
from wavetopo.rhombic import _tri_scalar


def build_patch(n, Nx, Ny, mat, rA, rB, bendfrac=(0.35, 0.65)):
    """Finite triangular patch of Nx x Ny rhombic cells with a Z-bent wall.
    Returns node coords, triangles, per-triangle density z, and the wall polyline
    in physical coordinates."""
    s3 = np.sqrt(3.0)
    a1 = np.array([1.5, s3 / 2]); a2 = np.array([1.5, -s3 / 2]); b = 1.0
    NIx, NIy = Nx * n, Ny * n
    ncol, nrow = NIx + 1, NIy + 1
    nnode = ncol * nrow
    I, J = np.meshgrid(np.arange(ncol), np.arange(nrow))
    npx = (I.ravel()[:, None] / n) * a1 + (J.ravel()[:, None] / n) * a2
    def nid(i, j): return j * ncol + i
    tris = []
    for j in range(NIy):
        for i in range(NIx):
            sw = nid(i, j); se = nid(i + 1, j)
            nw = nid(i, j + 1); ne = nid(i + 1, j + 1)
            tris += [[sw, se, ne], [sw, ne, nw]]
    tris = np.array(tris, np.int64)
    cents = npx[tris].mean(1)

    # bent wall in PARAMETRIC coords: i_wall(j) straight, then ramp, then straight
    j1, j2 = bendfrac[0] * NIy, bendfrac[1] * NIy
    i_lo, i_hi = 0.32 * NIx, 0.68 * NIx
    def iwall(j):
        if j < j1: return i_lo
        if j > j2: return i_hi
        return i_lo + (i_hi - i_lo) * (j - j1) / (j2 - j1)
    # element parametric coords
    ecolI = (np.arange(tris.shape[0]) // 2) % NIx
    erowJ = (np.arange(tris.shape[0]) // 2) // NIx
    # sublattice sites
    Asite = []; Bsite = []
    for p in range(-1, Nx + 1):
        for q in range(-1, Ny + 1):
            Asite.append(p * a1 + q * a2)
            Bsite.append(np.array([b, 0.0]) + p * a1 + q * a2)
    Asite = np.array(Asite); Bsite = np.array(Bsite)
    z = np.zeros(tris.shape[0])
    wall_phys = []
    for e in range(tris.shape[0]):
        left = ecolI[e] < iwall(erowJ[e] + 0.5)
        rr = (rA, rB) if left else (rB, rA)
        dA = np.min(np.hypot(*(cents[e] - Asite).T))
        dB = np.min(np.hypot(*(cents[e] - Bsite).T))
        if dA < rr[0] or dB < rr[1]:
            z[e] = 1.0
    # wall polyline (physical) for plotting
    for j in np.linspace(0, NIy, 60):
        wall_phys.append((iwall(j) / n) * a1 + (j / n) * a2)
    wall_phys = np.array(wall_phys)
    return dict(a1=a1, a2=a2, npx=npx, tris=tris, cents=cents, z=z,
                nnode=nnode, wall=wall_phys, iwall=iwall, n=n, Nx=Nx, Ny=Ny)


def assemble(patch, mat, omega, eta, sponge_frac=0.14):
    tris, npx, z = patch['tris'], patch['npx'], patch['z']
    muL, muT, mum = mat.mu_L, mat.mu_T, mat.mu_m
    aC, bC = (muL + muT) / 2, (muL - muT) / 2
    nnode = patch['nnode']
    iK = np.kron(tris, np.ones((3, 1), np.int64)).ravel()
    jK = np.kron(tris, np.ones((1, 3), np.int64)).ravel()
    kel = np.zeros((tris.shape[0], 3, 3)); mel = np.zeros_like(kel)
    massref = np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    area = np.zeros(tris.shape[0])
    for e, tri in enumerate(tris):
        xy = npx[tri]
        Kf, ar = _tri_scalar(xy, aC * np.eye(2))          # theta=0 (isotropic)
        Km, _ = _tri_scalar(xy, mum * np.eye(2))
        w = z[e]**3
        kel[e] = w * Kf + (1 - w) * Km
        rho = z[e] * mat.rho_f + (1 - z[e]) * mat.rho_m
        mel[e] = rho * massref * ar
        area[e] = ar
    K = sp.csr_matrix((kel.ravel(), (iK, jK)), shape=(nnode, nnode))
    M = sp.csr_matrix((mel.ravel(), (iK, jK)), shape=(nnode, nnode))
    # nodal sponge: ramp near the outer boundary of the parallelogram (param)
    ncol = patch['Nx'] * patch['n'] + 1
    nrow = patch['Ny'] * patch['n'] + 1
    ii = np.arange(nnode) % ncol; jj = np.arange(nnode) // ncol
    du = np.maximum.reduce([(sponge_frac*ncol - ii)/(sponge_frac*ncol),
                            (ii-(1-sponge_frac)*ncol)/(sponge_frac*ncol),
                            (sponge_frac*nrow - jj)/(sponge_frac*nrow),
                            (jj-(1-sponge_frac)*nrow)/(sponge_frac*nrow)])
    cnode = np.maximum(0, du)**2 * 40.0
    # lump nodal sponge into a diagonal mass-proportional damping
    Mdiag = np.asarray(M.sum(1)).ravel()
    C = sp.diags(cnode * Mdiag)
    D = K - omega**2 * M + 1j * omega * C
    return D, M, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--Nx", type=int, default=14)
    ap.add_argument("--Ny", type=int, default=16)
    ap.add_argument("--rA", type=float, default=0.46)
    ap.add_argument("--rB", type=float, default=0.22)
    ap.add_argument("--wmid", type=float, default=2.275)
    ap.add_argument("--wband", type=float, default=2.02)
    ap.add_argument("--eta", type=float, default=0.006)
    ap.add_argument("--out", default="results/valley_waveguide.png")
    args = ap.parse_args()

    mat = MaterialSH(mu_L=6.0, mu_T=2.0, mu_m=1.0, rho_f=1.6, rho_m=1.2)
    patch = build_patch(args.n, args.Nx, args.Ny, mat, args.rA, args.rB)
    npx = patch['npx']; wall = patch['wall']
    print(f"patch {patch['nnode']} nodes, {patch['tris'].shape[0]} tris, "
          f"fiber-frac {patch['z'].mean():.3f}")

    # source node: near the bottom end of the wall
    src_xy = wall[3]
    src = int(np.argmin(np.hypot(*(npx - src_xy).T)))
    # probe box: near the top end of the wall (past the bend)
    out_xy = wall[-4]

    def solve_and_probe(omega):
        D, M, K = assemble(patch, mat, omega, args.eta)
        F = np.zeros(patch['nnode'], complex); F[src] = 1.0
        u = spla.spsolve(D.tocsc(), F)
        d = np.hypot(*(npx - out_xy).T)
        T = float(np.sum(np.abs(u[d < 1.2])**2))
        return u, T

    u_mid, T_mid = solve_and_probe(args.wmid)
    u_band, T_band = solve_and_probe(args.wband)
    print(f"transmission: mid-gap(w={args.wmid}) T={T_mid:.3e}  "
          f"in-band(w={args.wband}) T={T_band:.3e}  ratio={T_mid/max(T_band,1e-30):.1f}x")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    for a_, u, tag, T in [(ax[0], u_mid, f"mid-gap w={args.wmid}", T_mid),
                          (ax[1], u_band, f"in-band w={args.wband}", T_band)]:
        mag = np.abs(u)
        vmax = np.percentile(mag, 99.5)
        sc = a_.scatter(npx[:, 0], npx[:, 1], c=mag, s=6, cmap="magma",
                        vmax=vmax)
        a_.plot(wall[:, 0], wall[:, 1], "c--", lw=1.2, alpha=0.8)
        a_.plot(npx[src, 0], npx[src, 1], "co", ms=9, mfc="none", mew=2)
        a_.add_patch(plt.Circle(out_xy, 1.2, ec="lime", fc="none", lw=1.6))
        a_.set_aspect("equal"); a_.set_title(f"{tag}   T={T:.1e}")
    fig.suptitle("Valley-Hall topological waveguide around a sharp bend: "
                 f"mid-gap guiding vs in-band radiation  "
                 f"(T ratio {T_mid/max(T_band,1e-30):.0f}x)", y=1.0, fontsize=13)
    plt.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
