"""Resolution / toolpath-pitch sweep figure (and LaTeX table) from the runs
results/flatband_C{1,2,3,3d}.npz.  For each level: (a) optimized band structure on the
common fine path, (b) crisp projected density 2x2, (c) as-manufactured tows at that
level's pitch, re-simulated.

Importable: ``build()`` returns a live Figure with the per-level rows on ``fig._rows``;
``main()`` also writes docs/paper/sweep_table.tex and results/flatband_sweep.npz.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import zoom

from wavetopo import figlib as F
from wavetopo.cfrp import Material, QuadMesh, CSRBFMapping
from wavetopo.bloch import BlochProblem, ibz_path_square

# CONTROLLED sweep: mesh (n=32), iteration budget (70) and -- for C1..C3 -- the
# aggregation k-set (22 points) are held FIXED; only the design freedom varies.
# C3d repeats C3 with a DENSER k-set (40 points), the control for k-sampling.
LEVELS = [("C1", "results/flatband_C1.npz", 1 / 6),
          ("C2", "results/flatband_C2.npz", 1 / 9),
          ("C3", "results/flatband_C3.npz", 1 / 12),
          ("C3d", "results/flatband_C3d.npz", 1 / 12)]

BETA, ETA = 12.0, 0.5
proj = lambda x: (np.tanh(BETA * ETA) + np.tanh(BETA * (x - ETA))) / \
                 (np.tanh(BETA * ETA) + np.tanh(BETA * (1 - ETA)))


def periodic_tows(theta_cell, dens_cell, a, nelx, nely, pitch, ng=260, dthr=0.5):
    """Exactly cell-periodic tows: solve grad(psi)=g/pitch by FFT on the torus and
    add an INTEGER number of tow cycles per period, so chi tiles seamlessly."""
    dx, dy = a / nelx, a / nely
    th3 = np.tile(theta_cell, (3, 3)); dn3 = np.tile(dens_cell, (3, 3))
    xs3 = (np.arange(3 * nelx) + 0.5) * dx - a
    ys3 = (np.arange(3 * nely) + 0.5) * dy - a
    GX3, GY3 = np.meshgrid(xs3, ys3)
    c3 = np.column_stack([GX3.ravel(), GY3.ravel()])
    gx = (np.arange(ng) + 0.5) * (a / ng); GXc, GYc = np.meshgrid(gx, gx)
    C = griddata(c3, np.cos(2 * th3).ravel(), (GXc, GYc), method="linear")
    S = griddata(c3, np.sin(2 * th3).ravel(), (GXc, GYc), method="linear")
    D = griddata(c3, dn3.ravel(), (GXc, GYc), method="linear")
    ang = 0.5 * np.arctan2(S, C)
    gxf, gyf = -np.sin(ang), np.cos(ang)
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


def build():
    F.journal()
    rows = []
    fig, ax = plt.subplots(3, len(LEVELS), figsize=(5.5 * len(LEVELS), 14))
    for col, (name, path, pitch) in enumerate(LEVELS):
        d = np.load(path)
        bands0, bands1 = d["bands0"], d["bands1"]
        z, theta = d["z"], d["theta"]
        nelx, nely, a = int(d["nelx"]), int(d["nely"]), float(d["a"])
        b = int(d["band"]); w0, w1 = float(d["w0"]), float(d["w1"])
        gu, gd = float(d["gap_up"]), float(d["gap_dn"])
        ks, R, r_s = int(d["ks"]), float(d["R"]), float(d["r_s"])
        nk, iters, M = int(d["nk"]), int(d["iters"]), int(d["M"])
        npath = bands1.shape[0]; n = (npath + 2) // 3
        ticks = [0, n - 1, 2 * n - 2, 3 * n - 3]
        labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
        dg = z.reshape(nely, nelx); th = theta.reshape(nely, nelx)
        dgp = proj(dg)

        mesh = QuadMesh(nelx, nely, a, a)
        mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3, rho_f=1.6, rho_m=1.2)
        xs0 = np.linspace(0, a, ks, endpoint=False); SX, SY = np.meshgrid(xs0, xs0)
        rbf = CSRBFMapping(mesh, (SX.ravel(), SY.ravel()), r_s=r_s, period=(a, a))
        bp = BlochProblem(mesh, mat, rbf, penal=3.0)
        theta_hat = np.linalg.lstsq(rbf.Phi.toarray(), theta, rcond=None)[0]
        bp.assemble(dgp.ravel(), theta_hat)
        ppath, _, _ = ibz_path_square(a, a, n=n)
        bands_v = bp.band_structure(ppath, bands1.shape[1])
        wv = bands_v[:, b].max() - bands_v[:, b].min()
        guv = bands_v[:, b + 1].min() - bands_v[:, b].max()
        gdv = bands_v[:, b].min() - bands_v[:, b - 1].max()
        print(f"{name}: n={nelx} ks={ks} d={pitch:.3f} | w {w0:.3f}->{w1:.3f} | "
              f"mfg w={wv:.3f} gaps {guv:+.3f}/{gdv:+.3f}")
        rows.append(dict(name=name, n=nelx, ks=ks, M=M, R=R, r_s=r_s, d=pitch,
                         nk=nk, iters=iters, w0=w0, w1=w1, gu=gu, gd=gd,
                         wv=wv, guv=guv, gdv=gdv))

        kk = np.arange(npath); A = ax[0, col]
        A.plot(kk, bands1, "0.75", lw=1.0); A.plot(kk, bands1[:, b], "b", lw=2.2)
        if gu > 0:
            A.axhspan(bands1[:, b].max(), bands1[:, b + 1].min(), color="orange", alpha=0.35)
        if gd > 0:
            A.axhspan(bands1[:, b - 1].max(), bands1[:, b].min(), color="orange", alpha=0.35)
        A.set_xticks(ticks); A.set_xticklabels(labels); A.grid(alpha=0.3); A.set_ylabel(r"$\omega$")
        A.set_title(f"{name}: $n$={nelx}, $k_s$={ks} ($M$={M}), $d$={pitch:.3f}\n"
                    f"band {b}: {w0:.2f}$\\to${w1:.2f} "
                    f"({w0/max(w1,1e-9):.1f}$\\times$), gaps {gu:+.2f}/{gd:+.2f}", fontsize=10)

        dcell = np.clip(zoom(dgp, max(1, round(144 / nelx)), order=3, mode="grid-wrap"), 0, 1)
        A = ax[1, col]
        A.imshow(np.tile(dcell, (2, 2)), origin="lower", cmap="gray_r", vmin=0, vmax=1,
                 extent=[0, 2 * a, 0, 2 * a], aspect="equal", interpolation="bilinear")
        A.axhline(a, color="r", lw=0.8); A.axvline(a, color="r", lw=0.8)
        A.set_title(f"cropped density ($R$={R:.3f}), $2\\times2$", fontsize=10)

        chi = periodic_tows(th, dgp, a, nelx, nely, pitch)
        A = ax[2, col]
        A.imshow(np.tile(chi, (2, 2)), origin="lower", cmap="gray_r", vmin=0, vmax=1,
                 extent=[0, 2 * a, 0, 2 * a], aspect="equal", interpolation="bilinear")
        A.axhline(a, color="r", lw=0.8); A.axvline(a, color="r", lw=0.8)
        A.set_title(f"as-manufactured tows, pitch $d$={pitch:.3f}\n"
                    f"re-simulated: $w$={wv:.2f}, gaps {guv:+.2f}/{gdv:+.2f}", fontsize=10)

    fig.suptitle("Flat-band metamaterial: jointly refining mesh $n$, orientation supports "
                 "$k_s$, and tow pitch $d$", y=0.995, fontsize=14)
    plt.tight_layout()
    fig._rows = rows
    return fig


def _write_table(rows, path="docs/paper/sweep_table.tex"):
    lines = [r"% AUTO-GENERATED by examples/figures/flatband_sweep_figure.py -- do not edit.",
             r"\begin{table}[h]", r"\centering", r"\small",
             r"\caption{\textbf{Controlled} design-freedom sweep for the flat-band cell "
             r"(band $b=3$). The analysis mesh ($n=32$) and the iteration budget (70) are "
             r"identical for every row, and C1--C3 additionally share one aggregation "
             r"$\mathbf k$-set ($N_k=22$). \textbf{C3d repeats C3 with a denser $\mathbf k$-set} "
             r"($N_k=40$). $w_0\!\to\!w_1$ is the target-band width before/after optimization; "
             r"the last three columns are the \emph{as-manufactured} re-simulation.}",
             r"\label{tab:sweep}",
             r"\begin{tabular}{lcccccccccccc}", r"\toprule",
             r"& \multicolumn{4}{c}{design freedom} & \multicolumn{1}{c}{sampling}"
             r" & \multicolumn{4}{c}{optimized (SIMP, fine path)}"
             r" & \multicolumn{3}{c}{as-manufactured}\\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-10}\cmidrule(lr){11-13}",
             r"run & $k_s$ & $M$ & $R$ & $d$ & $N_k$ "
             r"& $w_0$ & $w_1$ & $g_\uparrow$ & $g_\downarrow$ "
             r"& $w^{\rm m}$ & $g^{\rm m}_\uparrow$ & $g^{\rm m}_\downarrow$\\",
             r"\midrule"]
    for r in rows:
        npts = 3 * r['nk'] - 2
        lines.append(
            f"{r['name']} & {r['ks']} & {r['M']} & {r['R']:.3f} & "
            f"$1/{round(1/r['d'])}$ & {npts} & "
            f"{r['w0']:.2f} & {r['w1']:.2f} & {r['gu']:+.2f} & {r['gd']:+.2f} & "
            f"{r['wv']:.2f} & {r['guv']:+.2f} & {r['gdv']:+.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    plt.switch_backend("Agg")
    fig = build()
    F.save_pair(fig, "results/flatband_sweep.png", "docs/paper/figs/flatband_sweep.png")
    _write_table(fig._rows)
    np.savez("results/flatband_sweep.npz", **{f"{r['name']}_{k}": v
             for r in fig._rows for k, v in r.items() if k != "name"})
    print("wrote docs/paper/figs/flatband_sweep.png and docs/paper/sweep_table.tex")


if __name__ == "__main__":
    main()
