"""
Valley-Hall kink modes on the PRIMITIVE rhombic honeycomb (the definitive test).

Builds a ribbon supercell: Nc rhombic cells stacked along a1 (finite, open),
one cell along a2 (Bloch-periodic).  The left half has sublattice imbalance
A>B (radii rA,rB) and the right half its inversion partner B>A, so the valley
Chern flips sign across the central wall.  By bulk-edge correspondence the
interface must host gap-traversing kink modes localized at the wall.

We sweep the conserved momentum kappa = k.a2 and plot the ribbon bands; bands
that TRAVERSE the bulk gap and are localized at the central cell are the
topological valley kink states.  A P1 triangular (C3-respecting) mesh is used.
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


def build_ribbon(n, Nc, mat, rA, rB, theta_field=None):
    """Triangular mesh of Nc x 1 rhombic cells; density = honeycomb with
    sublattice imbalance flipping at the central column."""
    s3 = np.sqrt(3.0)
    a1 = np.array([1.5, s3 / 2]); a2 = np.array([1.5, -s3 / 2]); b = 1.0
    NI = Nc * n            # columns of elements along a1
    ncol = NI + 1          # nodes along a1
    nrow = n + 1           # nodes along a2
    nnode = ncol * nrow
    # node coords
    I, J = np.meshgrid(np.arange(ncol), np.arange(nrow))
    npx = (I.ravel()[:, None] / n) * a1 + (J.ravel()[:, None] / n) * a2
    def nid(i, j): return j * ncol + i
    # triangles
    tris = []
    for j in range(n):
        for i in range(NI):
            sw = nid(i, j); se = nid(i + 1, j)
            nw = nid(i, j + 1); ne = nid(i + 1, j + 1)
            tris += [[sw, se, ne], [sw, ne, nw]]
    tris = np.array(tris, np.int64)
    cents = npx[tris].mean(1)
    # sublattice sites (A at lattice pts, B at (a1+a2)/3 + lattice)
    Asite = []; Bsite = []
    for p in range(-1, Nc + 1):
        for q in range(-2, 3):
            Asite.append(p * a1 + q * a2)
            Bsite.append(np.array([b, 0.0]) + p * a1 + q * a2)
    Asite = np.array(Asite); Bsite = np.array(Bsite)
    # density: for each triangle, radii depend on its column-cell ic
    z = np.zeros(tris.shape[0])
    # element column i = (e//2) % NI  (2 triangles per parallelogram cell)
    col = (np.arange(tris.shape[0]) // 2) % NI
    ic = col // n                                # cell index along a1 (0..Nc-1)
    left = ic < (Nc // 2)
    for e in range(tris.shape[0]):
        rr = (rA, rB) if left[e] else (rB, rA)
        dA = np.min(np.hypot(*(cents[e] - Asite).T))
        dB = np.min(np.hypot(*(cents[e] - Bsite).T))
        if dA < rr[0] or dB < rr[1]:
            z[e] = 1.0
    # assemble K, M with scalar isotropic (theta=0) unless theta_field given
    muL, muT, mum = mat.mu_L, mat.mu_T, mat.mu_m
    aC, bC = (muL + muT) / 2, (muL - muT) / 2
    theta = np.zeros(tris.shape[0]) if theta_field is None else theta_field
    iK = np.kron(tris, np.ones((3, 1), np.int64)).ravel()
    jK = np.kron(tris, np.ones((1, 3), np.int64)).ravel()
    kel = np.zeros((tris.shape[0], 3, 3)); mel = np.zeros_like(kel)
    massref = np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    for e, tri in enumerate(tris):
        xy = npx[tri]
        th = theta[e]
        mu = (aC * np.eye(2) + bC * np.array([[np.cos(2*th), np.sin(2*th)],
                                              [np.sin(2*th), -np.cos(2*th)]]))
        w = z[e]**3
        Kf, area = _tri_scalar(xy, mu)
        Km, _ = _tri_scalar(xy, mum * np.eye(2))
        kel[e] = w * Kf + (1 - w) * Km
        rho = z[e] * mat.rho_f + (1 - z[e]) * mat.rho_m
        mel[e] = rho * massref * area
    K = sp.csr_matrix((kel.ravel(), (iK, jK)), shape=(nnode, nnode))
    M = sp.csr_matrix((mel.ravel(), (iK, jK)), shape=(nnode, nnode))
    # Reduction: Bloch-periodic along a2 (phase e^{i kappa}); superlattice-
    # periodic along a1 (Gamma, phase 1) so there are NO vacuum edges -- only
    # the two domain walls (A>B|B>A at the centre and B>A|A>B across the seam).
    master = np.arange(nnode)
    sphase = np.zeros(nnode)
    for i in range(ncol):                 # a2 wrap: top row -> bottom row
        master[nid(i, n)] = nid(i, 0); sphase[nid(i, n)] = 1.0
    for j in range(nrow):                 # a1 wrap: last col -> first col
        master[nid(NI, j)] = master[nid(0, j)]
    master[nid(NI, n)] = nid(0, 0); sphase[nid(NI, n)] = 1.0
    uniq, inv = np.unique(master, return_inverse=True)
    info = dict(a1=a1, a2=a2, nnode=nnode, ncol=ncol, nrow=nrow, nid=nid,
                z=z, cents=cents, Nc=Nc, n=n, npx=npx, tris=tris)
    return K, M, inv, sphase, info


def ribbon_bands(K, M, inv, sphase, kappa, nb, sigma):
    """Reduce with a2-phase e^{i kappa} and solve nb bands near sigma."""
    nnode = K.shape[0]; nred = inv.max() + 1
    ph = np.exp(1j * kappa * sphase)
    T = sp.csr_matrix((ph, (np.arange(nnode), inv)), shape=(nnode, nred))
    Kr = (T.conj().T @ K @ T).tocsc()
    Mr = (T.conj().T @ M @ T).tocsc()
    w2, V = spla.eigsh(Kr, k=nb, M=Mr, sigma=sigma, which='LM')
    o = np.argsort(w2.real)
    return np.sqrt(np.clip(w2[o].real, 0, None)), V[:, o], T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--Nc", type=int, default=10)
    ap.add_argument("--rA", type=float, default=0.46)
    ap.add_argument("--rB", type=float, default=0.22)
    ap.add_argument("--nk", type=int, default=41)
    ap.add_argument("--nb", type=int, default=24)
    ap.add_argument("--gap", type=float, nargs=2, default=[2.225, 2.325])
    ap.add_argument("--out", default="results/valley_ribbon.png")
    args = ap.parse_args()

    mat = MaterialSH(mu_L=6.0, mu_T=2.0, mu_m=1.0, rho_f=1.6, rho_m=1.2)
    K, M, inv, sphase, info = build_ribbon(args.n, args.Nc, mat, args.rA, args.rB)
    print(f"ribbon {info['nnode']} nodes, {info['tris'].shape[0]} tris, "
          f"reduced {inv.max()+1}; fiber-frac {info['z'].mean():.3f}")
    sigma = (0.5 * sum(args.gap))**2
    kappas = np.linspace(0, 2 * np.pi, args.nk)
    bands = np.zeros((args.nk, args.nb))
    # interface localization: weight of each mode near EITHER domain wall
    # (centre wall at parametric column NI/2; seam wall at column 0 == NI)
    a1 = info['a1']; NI = args.Nc * args.n
    Icol = np.round(info['npx'][:, 0] * 0).astype(int)   # placeholder
    # recover each node's parametric column from its index
    ncol = NI + 1
    node_col = np.arange(info['nnode']) % ncol
    dcol = np.minimum(np.abs(node_col - NI // 2),
                      np.minimum(node_col, np.abs(node_col - NI)))
    near = (dcol < 1.5 * args.n)[:, None]
    loc = np.zeros((args.nk, args.nb))
    for iq, kap in enumerate(kappas):
        w, V, T = ribbon_bands(K, M, inv, sphase, kap, args.nb, sigma)
        bands[iq] = w
        full = np.abs(T @ V)**2                       # (nnode, nb)
        loc[iq] = (full * near).sum(0) / (full.sum(0) + 1e-30)
    # count in-gap, interface-localized bands
    ingap = (bands > args.gap[0]) & (bands < args.gap[1])
    traversing = np.sum(ingap.any(0) & (loc.max(0) > 0.5))
    print(f"bands entering gap: {np.sum(ingap.any(0))}; "
          f"interface-localized in-gap branches: {traversing}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    kk = kappas / np.pi
    for b in range(args.nb):
        sc = ax[0].scatter(kk, bands[:, b], c=loc[:, b], cmap="turbo",
                           s=7, vmin=0, vmax=1)
    ax[0].axhspan(args.gap[0], args.gap[1], color="gold", alpha=0.18, zorder=0)
    ax[0].axvline(2/3, color="0.6", lw=0.6, ls=":"); ax[0].axvline(4/3, color="0.6", lw=0.6, ls=":")
    ax[0].set_ylim(args.gap[0]-0.25, args.gap[1]+0.25)
    ax[0].set_xlabel(r"$\kappa=k\cdot a_2/\pi$"); ax[0].set_ylabel(r"$\omega$")
    ax[0].set_title(f"rhombic ribbon: kink modes (color=interface localization)")
    plt.colorbar(sc, ax=ax[0], label="fraction near wall")
    # density map of the ribbon; tint the two sublattice domains
    cents = info['cents']
    ecol = (np.arange(info['tris'].shape[0]) // 2) % NI
    dom = (ecol // args.n) < (args.Nc // 2)      # True = A>B domain
    ax[1].scatter(cents[dom, 0], cents[dom, 1], c=info['z'][dom],
                  cmap="Blues", s=5, vmin=-0.3, vmax=1)
    ax[1].scatter(cents[~dom, 0], cents[~dom, 1], c=info['z'][~dom],
                  cmap="Reds", s=5, vmin=-0.3, vmax=1)
    ax[1].set_aspect("equal")
    ax[1].set_title("ribbon: A>B (blue) | B>A (red) domains; walls at each seam")
    plt.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
