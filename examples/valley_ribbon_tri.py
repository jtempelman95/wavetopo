"""
Ribbon band structure of the triangular-rod valley crystal -- the definitive
test for gap-traversing valley kink modes.

Nx cells along a1 (open), one cell along a2 (Bloch-periodic, momentum
kappa = k.a2).  Left half alpha=15deg, right half alpha=45deg (mirror-partner
domains, opposite valley Chern).  Bands localized at the central wall that
TRAVERSE the bulk complete gap are the topological kink states; the bulk valleys
project to kappa = 2pi/3 and 4pi/3.
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
from topoopt.valley_cell import triangle_ribbon


def assemble_ribbon(m, muf):
    npx, tris, z = m["npx"], m["tris"], m["zfib"]
    Nn = npx.shape[0]
    iK = np.kron(tris, np.ones((3, 1), np.int64)).ravel()
    jK = np.kron(tris, np.ones((1, 3), np.int64)).ravel()
    kel = np.zeros((tris.shape[0], 3, 3)); mel = np.zeros_like(kel)
    mref = np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    for e, tri in enumerate(tris):
        xy = npx[tri]
        Kf, ar = _tri_scalar(xy, muf * np.eye(2))
        Km, _ = _tri_scalar(xy, 1.0 * np.eye(2))
        w = z[e] ** 3
        kel[e] = w * Kf + (1 - w) * Km
        rho = z[e] * 1.6 + (1 - z[e]) * 1.2
        mel[e] = rho * mref * ar
    K = sp.csr_matrix((kel.ravel(), (iK, jK)), shape=(Nn, Nn))
    M = sp.csr_matrix((mel.ravel(), (iK, jK)), shape=(Nn, Nn))
    return K, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nx", type=int, default=10)
    ap.add_argument("--muf", type=float, default=100.0)
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--nk", type=int, default=41)
    ap.add_argument("--nb", type=int, default=28)
    ap.add_argument("--gap", type=float, nargs=2, default=[7.02, 7.55])
    ap.add_argument("--out", default="results/valley_ribbon_tri.png")
    args = ap.parse_args()

    m = triangle_ribbon(args.Nx, 15.0, 45.0, R=0.44, h=args.h)
    K, M = assemble_ribbon(m, args.muf)
    Nn = m["npx"].shape[0]; nred = m["n_red"]
    master, sph = m["master"], m["sphase"]
    print(f"ribbon {Nn} nodes, {m['tris'].shape[0]} tris, reduced {nred}, "
          f"fiber frac {m['zfib'].mean():.3f}")
    sigma = (0.5 * sum(args.gap)) ** 2
    xnode = m["npx"][:, 0]; xwall = (args.Nx / 2) * m["a1"][0]
    near = (np.abs(xnode - xwall) < 1.1 * m["a1"][0])

    kaps = np.linspace(0, 2 * np.pi, args.nk)
    bands = np.full((args.nk, args.nb), np.nan); loc = np.zeros((args.nk, args.nb))
    for iq, kap in enumerate(kaps):
        ph = np.exp(1j * kap * sph)
        T = sp.csr_matrix((ph, (np.arange(Nn), master)), shape=(Nn, nred))
        Kr = (T.conj().T @ K @ T).tocsc(); Mr = (T.conj().T @ M @ T).tocsc()
        try:
            w2, V = spla.eigsh(Kr, k=args.nb, M=Mr, sigma=sigma, which='LM')
        except Exception:
            continue
        o = np.argsort(w2.real); w2, V = w2[o].real, V[:, o]
        bands[iq] = np.sqrt(np.clip(w2, 0, None))
        full = np.abs(T @ V) ** 2
        loc[iq] = (full[near].sum(0)) / (full.sum(0) + 1e-30)
    ingap = ((bands > args.gap[0]) & (bands < args.gap[1]))
    nwall = int(np.sum(ingap.any(0) & (loc.max(0) > 0.5)))
    print(f"in-gap branches: {int(np.sum(ingap.any(0)))}; "
          f"wall-localized in-gap branches: {nwall}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    kk = kaps / np.pi
    for b in range(args.nb):
        sc = ax[0].scatter(kk, bands[:, b], c=loc[:, b], cmap="turbo", s=8,
                           vmin=0, vmax=1)
    ax[0].axhspan(*args.gap, color="gold", alpha=0.15, zorder=0)
    for kv in (2/3, 4/3):
        ax[0].axvline(kv, color="0.6", lw=0.6, ls=":")
    ax[0].set_ylim(args.gap[0]-0.6, args.gap[1]+0.6)
    ax[0].set_xlabel(r"$\kappa=k\cdot a_2/\pi$"); ax[0].set_ylabel(r"$\omega$")
    ax[0].set_title("triangular-rod ribbon (color = wall localization)")
    plt.colorbar(sc, ax=ax[0], label="fraction near wall")
    cents = m["npx"][m["tris"]].mean(1)
    ax[1].scatter(cents[:, 0], cents[:, 1], c=m["zfib"], cmap="Greys", s=3, vmin=0, vmax=1)
    ax[1].axvline(xwall, color="r", lw=1, ls="--")
    ax[1].set_aspect("equal"); ax[1].set_title("ribbon: alpha=15 | 45 (wall = red)")
    plt.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
