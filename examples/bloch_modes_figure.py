"""
Eigenmode VECTOR FIELDS of a periodic cell.

The band figures show where the modes sit in frequency; this shows what they
actually do.  For a chosen wavevector we plot the Bloch displacement
u(x) = (u_x, u_y) of selected bands as a quiver field over the density, plus the
real part, imaginary part and magnitude -- the same abs/re/im options the
wave-control figures offer.

A Bloch eigenvector is complex and defined up to a global phase, so it is gauge-
fixed here (rotated so its largest component is real) before Re/Im are shown;
without that, "Re u" is meaningless because an arbitrary phase mixes Re and Im.

    PYTHONPATH=. python examples/bloch_modes_figure.py \
        --data results/flatband_C3d.npz --bands 2 3 4 --k M
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.cfrp import Material, QuadMesh, CSRBFMapping
from wavetopo.bloch import BlochProblem


def gauge_fix(v):
    """Remove the arbitrary global phase: rotate so the largest-magnitude
    component is purely real and positive."""
    j = np.argmax(np.abs(v))
    return v * np.exp(-1j*np.angle(v[j]))


def nodal_field(bp, vec_red, kx, ky):
    """Reduced eigenvector -> complex (u_x, u_y) on the cell's node grid."""
    full = bp.T(kx, ky) @ vec_red
    m = bp.mesh
    ux = full[0::2].reshape(m.nely+1, m.nelx+1)
    uy = full[1::2].reshape(m.nely+1, m.nelx+1)
    return ux, uy


KPOINTS = {"G": (0.0, 0.0), "X": (np.pi, 0.0), "M": (np.pi, np.pi),
           "Y": (0.0, np.pi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/flatband_C3d.npz")
    ap.add_argument("--bands", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--k", default="M", choices=list(KPOINTS))
    ap.add_argument("--mode", default="abs", choices=["abs", "re", "im"],
                    help="scalar background: |u|, Re u_x or Im u_x")
    ap.add_argument("--quiver-n", type=int, default=22)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.data)
    nelx, nely = int(d["nelx"]), int(d["nely"])
    a = float(d["a"]); z = d["z"]; theta = d["theta"]
    ks = int(d["ks"]) if "ks" in d else 4
    r_s = float(d["r_s"]) if "r_s" in d else 2.0*a/ks

    mesh = QuadMesh(nelx, nely, a, a)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    xs = np.linspace(0, a, ks, endpoint=False); SX, SY = np.meshgrid(xs, xs)
    rbf = CSRBFMapping(mesh, (SX.ravel(), SY.ravel()), r_s=r_s, period=(a, a))
    bp = BlochProblem(mesh, mat, rbf, penal=3.0)
    theta_hat = np.linalg.lstsq(rbf.Phi.toarray(), theta, rcond=None)[0]
    bp.assemble(z, theta_hat)

    kx, ky = KPOINTS[args.k]
    nb = max(args.bands) + 1
    w, _, V, _ = bp.bands_at_k(kx, ky, nb, return_vec=True)

    dg = z.reshape(nely, nelx)
    xs_n = np.linspace(0, a, nelx+1); ys_n = np.linspace(0, a, nely+1)
    step = max(1, (nelx+1)//args.quiver_n)

    ncol = len(args.bands)
    fig, ax = plt.subplots(2, ncol, figsize=(5.0*ncol, 9.0), squeeze=False)
    for c, b in enumerate(args.bands):
        ux, uy = nodal_field(bp, gauge_fix(V[:, b]), kx, ky)
        mag = np.sqrt(np.abs(ux)**2 + np.abs(uy)**2)
        bg = {"abs": mag, "re": ux.real, "im": ux.imag}[args.mode]
        cmap = "magma" if args.mode == "abs" else "RdBu_r"
        lim = dict(vmax=np.percentile(bg, 99)) if args.mode == "abs" else \
            dict(vmin=-np.abs(bg).max(), vmax=np.abs(bg).max())

        A = ax[0][c]
        im = A.imshow(bg, origin="lower", extent=[0, a, 0, a], cmap=cmap,
                      interpolation="bilinear", **lim)
        A.contour(np.linspace(0, a, nelx), np.linspace(0, a, nely), dg,
                  levels=[0.5], colors="w", linewidths=1.0)
        A.quiver(xs_n[::step], ys_n[::step],
                 ux.real[::step, ::step], uy.real[::step, ::step],
                 color="k", scale=None, width=0.004, alpha=0.85)
        A.set_aspect("equal"); A.set_title(
            f"band {b}   $\\omega$={w[b]:.3f}\nRe$\\,\\mathbf{{u}}$ over "
            f"{'$|u|$' if args.mode=='abs' else args.mode}", fontsize=10)
        plt.colorbar(im, ax=A, fraction=0.046, pad=0.02)

        # divergence / curl: which modes are dilatational vs shear
        dux_dx = np.gradient(ux.real, xs_n, axis=1)
        duy_dy = np.gradient(uy.real, ys_n, axis=0)
        dux_dy = np.gradient(ux.real, ys_n, axis=0)
        duy_dx = np.gradient(uy.real, xs_n, axis=1)
        div = dux_dx + duy_dy
        curl = duy_dx - dux_dy
        B = ax[1][c]
        v = max(np.abs(div).max(), np.abs(curl).max())
        im2 = B.imshow(div, origin="lower", extent=[0, a, 0, a], cmap="PuOr_r",
                       vmin=-v, vmax=v, interpolation="bilinear")
        B.contour(np.linspace(0, a, nelx), np.linspace(0, a, nely), dg,
                  levels=[0.5], colors="k", linewidths=0.8)
        B.set_aspect("equal")
        B.set_title(f"band {b}: $\\nabla\\!\\cdot\\!\\mathbf{{u}}$  "
                    f"(dilatational vs shear)\n"
                    f"$\\|\\nabla\\!\\cdot\\!u\\|/\\|\\nabla\\times u\\|$ = "
                    f"{np.linalg.norm(div)/max(np.linalg.norm(curl),1e-30):.2f}",
                    fontsize=10)
        plt.colorbar(im2, ax=B, fraction=0.046, pad=0.02)

    fig.suptitle(f"Bloch eigenmodes at {args.k} — displacement vector field "
                 f"({args.data.split('/')[-1]})", y=0.98, fontsize=13)
    fig.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.05,
                        wspace=0.18, hspace=0.22)
    out = args.out or f"results/bloch_modes_{args.k}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
