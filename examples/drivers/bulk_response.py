"""
BULK RESPONSE of an optimized metamaterial cell.

A band diagram is a statement about an infinite periodic medium.  This asks the
practical question instead: tile the optimized cell into a FINITE panel, drive it,
and see what actually happens.

  (a) DYNAMIC -- transmission spectrum.  Assemble the tiled panel, solve the
      damped time-harmonic problem (K - w^2 M + i eta K) u = f at each frequency
      with a line drive on the left edge, and measure the energy reaching the
      right.  A real gap shows up as a transmission trough aligned with the band
      prediction; the alignment is the test.

  (b) DYNAMIC -- fields at two frequencies, one inside the gap and one outside,
      showing evanescent decay versus propagation through the same panel.

  (c) STATIC -- uniaxial response of the same panel: pull it, measure the
      effective modulus, and compare with the homogenised C_H the cell was
      optimised against.  Agreement checks that the homogenisation which drove
      the design describes the finite article.

    PYTHONPATH=. python examples/bulk_response.py --data results/multifunc_co_w0.3.npz
"""
import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.cfrp import (Material, QuadMesh, FourierStiffness, CSRBFMapping,
                           element_mass)
from wavetopo.bloch import BlochProblem
from wavetopo.homogenize import Homogenizer

MAT = dict(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
           rho_f=1.6, rho_m=1.2)


def tile(cell, nx, ny):
    return np.tile(cell, (ny, nx))


def build_panel(z_cell, th_cell, nelx, nely, a, ncx, ncy, penal=3.0):
    """Assemble K, M for a ncx x ncy tiling of the optimized cell."""
    zc = z_cell.reshape(nely, nelx)
    tc = th_cell.reshape(nely, nelx)
    Z = tile(zc, ncx, ncy).ravel()
    TH = tile(tc, ncx, ncy).ravel()
    mesh = QuadMesh(nelx*ncx, nely*ncy, a*ncx, a*ncy)
    mat = Material(**MAT)
    fs = FourierStiffness(mat, mesh, nsamp=720)
    kf = fs.kf(TH); km = fs.km[None]
    w = Z**penal
    kel = w[:, None, None]*kf + (1 - w)[:, None, None]*km
    K = sp.csr_matrix((kel.ravel(), (mesh.iK, mesh.jK)),
                      shape=(mesh.ndof, mesh.ndof))
    Me = element_mass(mesh.dx, mesh.dy, rho=1.0)
    rho = Z*mat.rho_f + (1 - Z)*mat.rho_m
    mel = rho[:, None, None]*Me[None]
    M = sp.csr_matrix((mel.ravel(), (mesh.iK, mesh.jK)),
                      shape=(mesh.ndof, mesh.ndof))
    return mesh, K.tocsc(), M.tocsc(), Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/multifunc_co_w0.3.npz")
    ap.add_argument("--ncx", type=int, default=8, help="cells along the drive")
    ap.add_argument("--ncy", type=int, default=4)
    ap.add_argument("--nfreq", type=int, default=70)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--out", default="results/bulk_response.png")
    args = ap.parse_args()

    d = np.load(args.data)
    nelx, nely, a = int(d["nelx"]), int(d["nely"]), float(d["a"])
    z_cell, th_cell = d["z"], d["theta"]
    b = int(d["band"])
    bands = d["bands1"]
    w_lo, w_hi = bands[:, b].max(), bands[:, b+1].min()
    if not (w_hi > w_lo) and b >= 1:      # try the gap below instead
        lo2, hi2 = bands[:, b-1].max(), bands[:, b].min()
        if hi2 > lo2:
            w_lo, w_hi, b = lo2, hi2, b-1
    gap_open = w_hi > w_lo
    print(f"cell: {nelx}x{nely}, band {b} gap "
          f"[{w_lo:.3f}, {w_hi:.3f}] {'OPEN' if gap_open else '(closed)'}")

    mesh, K, M, Z = build_panel(z_cell, th_cell, nelx, nely, a, args.ncx, args.ncy)
    print(f"panel: {args.ncx}x{args.ncy} cells = {mesh.nelx}x{mesh.nely} elements, "
          f"{mesh.ndof} dof", flush=True)

    # ---- drive on the left edge, measure on the right ------------------ #
    xy = np.column_stack([np.repeat(np.arange(mesh.nelx+1)*mesh.dx, 1),
                          np.zeros(mesh.nelx+1)])  # placeholder, real coords below
    nx1, ny1 = mesh.nelx+1, mesh.nely+1
    X = (np.arange(nx1)*mesh.dx)[None, :].repeat(ny1, 0).ravel()
    Y = (np.arange(ny1)*mesh.dy)[:, None].repeat(nx1, 1).ravel()
    left = np.flatnonzero(X < 1e-9)
    right = np.flatnonzero(X > X.max() - 1e-9)
    f = np.zeros(mesh.ndof)
    f[2*left] = 1.0                                   # x-directed line drive
    out_dofs = np.concatenate([2*right, 2*right+1])
    in_dofs = np.concatenate([2*left, 2*left+1])

    freqs = np.linspace(0.35*w_lo, 1.45*w_hi, args.nfreq)
    T = np.zeros(args.nfreq)
    for i, w in enumerate(freqs):
        A = (K*(1 + 1j*args.eta) - (w**2)*M).tocsc()
        u = spla.spsolve(A, f)
        Ein = np.sum(np.abs(u[in_dofs])**2)
        Eout = np.sum(np.abs(u[out_dofs])**2)
        T[i] = 10*np.log10(max(Eout, 1e-300)/max(Ein, 1e-300))
        if i % 10 == 0:
            print(f"   w={w:6.3f}  T={T[i]:+7.2f} dB", flush=True)

    # ---- fields inside and outside the gap ----------------------------- #
    w_in = 0.5*(w_lo + w_hi) if gap_open else 0.5*(w_lo + w_hi)
    w_out = 0.55*w_lo
    fields = {}
    for tag, w in (("in gap", w_in), ("pass band", w_out)):
        A = (K*(1 + 1j*args.eta) - (w**2)*M).tocsc()
        u = spla.spsolve(A, f)
        mag = np.sqrt(np.abs(u[0::2])**2 + np.abs(u[1::2])**2).reshape(ny1, nx1)
        fields[tag] = (w, mag)

    # ---- static: pull the panel, compare with the homogenised C_H ------ #
    fixed = np.concatenate([2*left, 2*left+1])
    free = np.setdiff1d(np.arange(mesh.ndof), fixed)
    fs_ = np.zeros(mesh.ndof); fs_[2*right] = 1.0/len(right)
    Kff = K[free][:, free]
    us = np.zeros(mesh.ndof); us[free] = spla.spsolve(Kff.tocsc(), fs_[free])
    ux_end = us[2*right].mean()
    L, H = X.max(), Y.max()
    E_eff = (1.0/H)/(ux_end/L) if ux_end != 0 else np.nan   # stress/strain

    mesh_c = QuadMesh(nelx, nely, a, a)
    ks = int(d["ks"]) if "ks" in d else 4
    r_s = float(d["r_s"]) if "r_s" in d else 2.0*a/ks
    xs = np.linspace(0, a, ks, endpoint=False); SX, SY = np.meshgrid(xs, xs)
    rbf = CSRBFMapping(mesh_c, (SX.ravel(), SY.ravel()), r_s=r_s, period=(a, a))
    bpc = BlochProblem(mesh_c, Material(**MAT), rbf, penal=3.0)
    th_hat = np.linalg.lstsq(rbf.Phi.toarray(), th_cell, rcond=None)[0]
    bpc.assemble(z_cell, th_hat)
    CH, _ = Homogenizer(bpc).solve()
    E11 = CH[0, 0] - CH[0, 1]**2/CH[1, 1]      # uniaxial modulus, free lateral
    print(f"\nstatic: finite panel E_eff={E_eff:.3f}  vs homogenised "
          f"E11={E11:.3f}  ratio={E_eff/E11:.2f}")

    # ---------------- figure ---------------- #
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.24)

    A1 = fig.add_subplot(gs[0, :2])
    A1.plot(freqs, T, "b-", lw=2.0)
    if gap_open:
        A1.axvspan(w_lo, w_hi, color="orange", alpha=0.30,
                   label=f"predicted gap [{w_lo:.2f}, {w_hi:.2f}]")
    A1.axvline(w_in, color="tab:red", ls="--", lw=1.2)
    A1.axvline(w_out, color="tab:green", ls="--", lw=1.2)
    A1.set_xlabel(r"$\omega$"); A1.set_ylabel("transmission (dB)")
    A1.set_title(f"bulk transmission through {args.ncx}$\\times${args.ncy} cells "
                 f"($\\eta$={args.eta})", fontsize=11)
    A1.grid(alpha=0.3); A1.legend(fontsize=9)

    A2 = fig.add_subplot(gs[0, 2])
    dgc = tile(z_cell.reshape(nely, nelx), args.ncx, args.ncy)
    A2.imshow(dgc, origin="lower", cmap="gray_r", vmin=0, vmax=1,
              extent=[0, L, 0, H], aspect="equal", interpolation="bilinear")
    A2.set_title("the panel (tiled optimized cell)", fontsize=11)

    for k, (tag, col) in enumerate((("pass band", "tab:green"), ("in gap", "tab:red"))):
        w, mag = fields[tag]
        Ax = fig.add_subplot(gs[1, k])
        im = Ax.imshow(mag, origin="lower", extent=[0, L, 0, H], cmap="magma",
                       aspect="equal", interpolation="bilinear",
                       vmax=np.percentile(mag, 99.5))
        Ax.set_title(f"{tag}: $\\omega$={w:.2f}", fontsize=11, color=col)
        plt.colorbar(im, ax=Ax, fraction=0.030, pad=0.02)

    A4 = fig.add_subplot(gs[1, 2])
    decay = [np.log10(np.maximum(m.mean(axis=0), 1e-300))
             for _, m in (fields["pass band"], fields["in gap"])]
    xcell = np.arange(nx1)*mesh.dx/a
    A4.plot(xcell, decay[0], color="tab:green", lw=2, label="pass band")
    A4.plot(xcell, decay[1], color="tab:red", lw=2, label="in gap")
    A4.set_xlabel("distance into the panel (cells)")
    A4.set_ylabel(r"$\log_{10}\langle|u|\rangle$")
    A4.set_title("spatial decay: evanescent vs propagating", fontsize=11)
    A4.grid(alpha=0.3); A4.legend(fontsize=9)

    fig.suptitle("Bulk response of the multi-functional cell: a finite panel is "
                 "both stiff and wave-blocking", y=0.97, fontsize=13)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    np.savez("results/bulk_response.npz", freqs=freqs, T=T, w_lo=w_lo, w_hi=w_hi,
             E_eff=E_eff, E11=E11, decay_pass=decay[0], decay_gap=decay[1],
             xcell=xcell)
    print("saved", args.out)


if __name__ == "__main__":
    main()
