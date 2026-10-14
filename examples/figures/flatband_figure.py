"""Flat-band metamaterial figure from results/flatband_data.npz:
 - crisp PROJECTED density (topology), 2x2 tiling;
 - EXPLICIT phase-field tows, masked to the solid, made seamlessly periodic;
 - an AS-MANUFACTURED validation: re-simulate the band structure with the cropped
   (projected) density and the actual fiber toolpath orientation.
Runs in the base numpy/scipy env; no re-optimization (but the validation re-solves the
band structure, so ``build()`` takes a few seconds).

Importable: ``build()`` returns a live Figure (and stashes the as-manufactured arrays on
``fig._manuf``); ``main()`` is the batch CLI.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import zoom

from wavetopo import figlib as F
from wavetopo.cfrp import Material, QuadMesh, CSRBFMapping
from wavetopo.bloch import BlochProblem, ibz_path_square


# ---- exactly cell-PERIODIC tows via an FFT phase + integer tow count ----
def periodic_tows(theta_cell, dens_cell, a, nelx, nely, pitch, ng=220, dthr=0.5):
    dx, dy = a / nelx, a / nely
    th3 = np.tile(theta_cell, (3, 3)); dn3 = np.tile(dens_cell, (3, 3))
    xs3 = (np.arange(3 * nelx) + 0.5) * dx - a; ys3 = (np.arange(3 * nely) + 0.5) * dy - a
    GX3, GY3 = np.meshgrid(xs3, ys3); c3 = np.column_stack([GX3.ravel(), GY3.ravel()])
    gx = (np.arange(ng) + 0.5) * (a / ng); GXc, GYc = np.meshgrid(gx, gx)
    C = griddata(c3, np.cos(2 * th3).ravel(), (GXc, GYc), method="linear")
    S = griddata(c3, np.sin(2 * th3).ravel(), (GXc, GYc), method="linear")
    D = griddata(c3, dn3.ravel(), (GXc, GYc), method="linear")
    ang = 0.5 * np.arctan2(S, C)
    gxf = -np.sin(ang); gyf = np.cos(ang)
    k = 2 * np.pi * np.fft.fftfreq(ng, d=a / ng)
    KX, KY = np.meshgrid(k, k)
    fxh = np.fft.fft2(gxf / pitch); fyh = np.fft.fft2(gyf / pitch)
    den = KX ** 2 + KY ** 2; den[0, 0] = 1.0
    psih = -1j * (KX * fxh + KY * fyh) / den; psih[0, 0] = 0.0
    psi_p = np.real(np.fft.ifft2(psih))
    nx = round(a * gxf.mean() / pitch); ny = round(a * gyf.mean() / pitch)
    Psi = psi_p + nx * GXc / a + ny * GYc / a
    chi = 0.5 + 0.5 * np.cos(2 * np.pi * Psi)
    return np.where(D >= dthr, chi, np.nan)


def build(data="results/flatband_data.npz"):
    F.journal()
    _P = bool(os.environ.get("FIGNOSUP"))
    d = np.load(data)
    bands0, bands1 = d["bands0"], d["bands1"]
    z, theta = d["z"], d["theta"]
    nelx, nely, a = int(d["nelx"]), int(d["nely"]), float(d["a"])
    b = int(d["band"]); w0, w1 = float(d["w0"]), float(d["w1"])
    hist_w, hist_gu, hist_gd = d["hist_w"], d["hist_gu"], d["hist_gd"]
    npath = bands0.shape[0]; n = (npath + 2) // 3
    ticks = [0, n - 1, 2 * n - 2, 3 * n - 3]; labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
    nb = bands1.shape[1]

    beta, eta = 12.0, 0.5
    proj = lambda x: (np.tanh(beta * eta) + np.tanh(beta * (x - eta))) / \
                     (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
    dg = z.reshape(nely, nelx); th = theta.reshape(nely, nelx)
    dgp = proj(dg)

    # as-manufactured validation: recover support angles, re-simulate the bands
    mesh = QuadMesh(nelx, nely, a, a)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3, rho_f=1.6, rho_m=1.2)
    ks = 4; xs0 = np.linspace(0, a, ks, endpoint=False); SX, SY = np.meshgrid(xs0, xs0)
    rbf = CSRBFMapping(mesh, (SX.ravel(), SY.ravel()), r_s=a / 2, period=(a, a))
    bp = BlochProblem(mesh, mat, rbf, penal=3.0)
    theta_hat = np.linalg.lstsq(rbf.Phi.toarray(), theta, rcond=None)[0]
    z_manuf = dgp.ravel()
    bp.assemble(z_manuf, theta_hat)
    ppath, _, _ = ibz_path_square(a, a, n=n)
    bands_v = bp.band_structure(ppath, nb)
    w_v = bands_v[:, b].max() - bands_v[:, b].min()
    gu_v = bands_v[:, b + 1].min() - bands_v[:, b].max()
    gd_v = bands_v[:, b].min() - bands_v[:, b - 1].max()

    chi1 = periodic_tows(th, dgp, a, nelx, nely, 1.0 / 9.0)
    chi_c = np.tile(chi1, (2, 2))

    kk = np.arange(npath)
    fig, ax = plt.subplots(2, 3, figsize=(7.6, 4.02) if _P else (17, 9))
    panels = [(ax[0, 0], bands0, f"baseline (band {b} width {w0:.2f})", None, None),
              (ax[0, 1], bands1, f"optimized SIMP (width {w1:.2f})", float(d["gap_up"]), float(d["gap_dn"])),
              (ax[0, 2], bands_v, f"as-manufactured (width {w_v:.2f})", gu_v, gd_v)]
    for a_, bands, ttl, gu, gd in panels:
        a_.plot(kk, bands, "0.7", lw=1.0); a_.plot(kk, bands[:, b], "b", lw=2.2)
        a_.set_xticks(ticks); a_.set_xticklabels(labels); a_.set_ylabel(r"$\omega$")
        a_.set_title(ttl); a_.grid(alpha=0.3)
        if gu is not None and gu > 0:
            a_.axhspan(bands[:, b].max(), bands[:, b + 1].min(), color="orange", alpha=0.35)
        if gd is not None and gd > 0:
            a_.axhspan(bands[:, b - 1].max(), bands[:, b].min(), color="orange", alpha=0.35)

    dcell = np.clip(zoom(dgp, 6, order=3, mode="grid-wrap"), 0, 1)
    dens_2 = np.tile(dcell, (2, 2))
    im = ax[1, 0].imshow(dens_2, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                         extent=[0, 2 * a, 0, 2 * a], aspect="equal", interpolation="bilinear")
    ax[1, 0].axhline(a, color="r", lw=0.8); ax[1, 0].axvline(a, color="r", lw=0.8)
    ax[1, 0].set_title("cropped density" if _P else "cropped density (topology), 2$\\times$2 tiling")
    plt.colorbar(im, ax=ax[1, 0], fraction=0.046)

    ax[1, 1].imshow(chi_c, origin="lower", extent=[0, 2 * a, 0, 2 * a], cmap="gray_r",
                    interpolation="bilinear", vmin=0, vmax=1, aspect="equal")
    ax[1, 1].axhline(a, color="r", lw=0.8); ax[1, 1].axvline(a, color="r", lw=0.8)
    ax[1, 1].set_title("fiber toolpaths" if _P else "fiber toolpaths (as-manufactured tows), 2$\\times$2 tiling")

    ax[1, 2].plot(hist_w, label="band width"); ax[1, 2].plot(hist_gu, label="gap above")
    ax[1, 2].plot(hist_gd, label="gap below"); ax[1, 2].axhline(0, color="k", lw=0.6)
    ax[1, 2].set_title("convergence"); ax[1, 2].set_xlabel("iter"); ax[1, 2].legend(); ax[1, 2].grid(alpha=0.3)

    if not _P:
        fig.suptitle("Flat-band + band-gap metamaterial: co-optimized topology and fiber "
                     "toolpath, with as-manufactured validation", y=1.0, fontsize=14)
    plt.tight_layout()
    if _P:
        F.panel_labels(ax)
    fig._manuf = dict(bands_v=bands_v, theta_hat=theta_hat, z_manuf=z_manuf, w_v=w_v, gu_v=gu_v, gd_v=gd_v)
    print(f"as-manufactured band {b}: width={w_v:.3f} gap_up={gu_v:+.3f} gap_dn={gd_v:+.3f}")
    if _P:
        F.harmonize_fonts(fig, 0.807)   # uniform printed fonts (manuscript display scale)
    return fig


def main():
    plt.switch_backend("Agg")
    fig = build()
    F.save_pair(fig, "results/flatband_demo.png", "docs/paper/figs/flatband.png",
                label=not bool(os.environ.get("FIGNOSUP")))
    np.savez("results/flatband_manuf.npz", **fig._manuf)
    print("saved flatband figure (periodic tows + as-manufactured validation)")


if __name__ == "__main__":
    main()
