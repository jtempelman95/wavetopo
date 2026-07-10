"""
Ribbon (supercell) band structure -- the definitive test for valley-Hall kink
modes.  Periodic ALONG the domain wall, finite ACROSS it.  In-gap bands that are
localized at the central wall are the topological kink (edge) states.

We test both wall orientations (wall along x vs along y) and report which hosts
gapless in-gap kink modes; that sets the right wall + operating frequency for the
forced edge-transport demo.
"""
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
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
    sx = np.linspace(0, Lx_c, 6, endpoint=False); sy = np.linspace(0, Ly_c, 10, endpoint=False)
    SX, SY = np.meshgrid(sx, sy)
    rbf = CSRBFMapping(mesh_c, (SX.ravel(), SY.ravel()), r_s=0.55, period=(Lx_c, Ly_c))
    P = density_filter(mesh_c, 0.11, period=(Lx_c, Ly_c))
    y = (P @ d["z"]).reshape(nely_c, nelx_c)
    theta = rbf.theta(d["th"]).reshape(nely_c, nelx_c)
    return y, theta, (nelx_c, nely_c, Lx_c, Ly_c), float(d["omega"])


def ribbon_T(mesh, period_axis, kper, period_len):
    """Reduction matrix: Bloch-periodic along `period_axis` (0=x,1=y), the other
    axis finite (free)."""
    nelx, nely = mesh.nelx, mesh.nely
    nnode = mesh.nnode
    master = np.arange(nnode); phase = np.ones(nnode, complex)
    for iy in range(nely + 1):
        for ix in range(nelx + 1):
            n = iy * (nelx + 1) + ix
            if period_axis == 0 and ix == nelx:
                master[n] = iy * (nelx + 1) + 0; phase[n] = np.exp(1j * kper * period_len)
            elif period_axis == 1 and iy == nely:
                master[n] = 0 * (nelx + 1) + ix; phase[n] = np.exp(1j * kper * period_len)
    uniq, inv = np.unique(master, return_inverse=True)
    T = sp.csr_matrix((phase, (np.arange(nnode), inv)), shape=(nnode, uniq.size))
    return T


def ribbon_bands(npz, orient, ncross=8, nk=41, nb=26):
    yc, thc, (nelx_c, nely_c, Lx_c, Ly_c), omega = load_cell(npz)
    mat = MaterialSH(mu_L=6.0, mu_T=2.0, mu_m=1.0, rho_f=1.6, rho_m=1.2)
    if orient == "y":     # wall along y: periodic in y (1 cell), finite in x (ncross cells)
        mesh = QuadMesh(nelx_c * ncross, nely_c, Lx_c * ncross, Ly_c)
        z = np.tile(yc, (1, ncross)).ravel()
        th = np.tile(thc, (1, ncross))
        ex = (np.arange(mesh.nelx) + 0.5) * mesh.dx
        sign = np.where(ex[None, :] > (ncross // 2) * Lx_c, 1.0, -1.0) * np.ones((mesh.nely, mesh.nelx))
        theta = (th * sign).ravel()
        paxis, plen, kmax = 1, Ly_c, np.pi / Ly_c
    else:                 # wall along x: periodic in x (1 cell), finite in y (ncross cells)
        mesh = QuadMesh(nelx_c, nely_c * ncross, Lx_c, Ly_c * ncross)
        z = np.tile(yc, (ncross, 1)).ravel()
        th = np.tile(thc, (ncross, 1))
        ey = (np.arange(mesh.nely) + 0.5) * mesh.dy
        sign = np.where(ey[:, None] > (ncross // 2) * Ly_c, 1.0, -1.0) * np.ones((mesh.nely, mesh.nelx))
        theta = (th * sign).ravel()
        paxis, plen, kmax = 0, Lx_c, np.pi / Lx_c

    hv = ScalarHarmonic(mesh, mat, omega=1.0)
    hv.set_design(z, theta)
    K, M = hv.K, hv.M
    ks = np.linspace(-kmax, kmax, nk)
    bands = np.zeros((nk, nb))
    vecs = {}
    for i, k in enumerate(ks):
        T = ribbon_T(mesh, paxis, k, plen)
        Kr = (T.conj().T @ K @ T).toarray(); Mr = (T.conj().T @ M @ T).toarray()
        w2, V = sla.eigh(Kr, Mr, subset_by_index=[0, nb - 1])
        bands[i] = np.sqrt(np.clip(w2, 0, None))
        if i == nk // 2:
            vecs = dict(T=T, V=V, mesh=mesh)
    return ks, bands, vecs, omega, (mesh, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/scalar_complete.npz")
    ap.add_argument("--out", default="results/scalar_ribbon.png")
    args = ap.parse_args()
    gap = (2.117, 2.411)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for col, orient in enumerate(["x", "y"]):
        ks, bands, vecs, omega, _ = ribbon_bands(args.npz, orient)
        kmax = ks.max()
        # count in-gap bands at k=0
        mid = bands[len(ks)//2]
        ingap = np.sum((mid > gap[0]) & (mid < gap[1]))
        print(f"wall along {orient}: in-gap bands at k=0 = {ingap}")
        ax[col].plot(ks / kmax, bands, "k-", lw=0.7)
        ax[col].axhspan(gap[0], gap[1], color="orange", alpha=0.25)
        ax[col].set_ylim(gap[0] - 0.4, gap[1] + 0.4)
        ax[col].set_xlabel(r"$k_{\parallel}/(\pi/L)$")
        ax[col].set_title(f"ribbon, wall along {orient}  (in-gap modes: {ingap})")
        ax[col].set_ylabel(r"$\omega$")
    fig.suptitle("Ribbon band structure: in-gap bands crossing the orange gap "
                 "= valley kink (edge) modes", y=1.0)
    plt.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
