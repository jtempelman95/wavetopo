"""Dense stiffness--frequency Pareto for the CFRP bracket, with a topology+fiber
thumbnail at each swept point, so the reader sees the part morph from the pure-stiffness
A-frame into the frequency-weighted, modal-stiffened truss as the modal weight w_omega is
turned up.  A single joint co-design traces the whole front; the pure-stiffness end sits
on the proportionate line omega_1^2 ~ stiffness, and frequency weight lifts the design
into the disproportionate region above it."""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing

from examples.cfrp_bracket_modal import setup, HOLES, LOAD_HOLE, LX, LY
from wavetopo.dolfinx_viz import plot_toolpaths_phase
from _paperfig import set_style, internal_ticks

set_style()
NG, MORPH_R, SMOOTH = 700, 0.016, 0.010


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.02)   # match the pe*.npz design mesh (38062 elems)
    ap.add_argument("--out", default="results/cfrp_bracket_pareto.pdf")
    args = ap.parse_args()
    # DENSE Pareto: 15 high-res points (fall back to the coarse 6-point sweep if absent)
    PE = [0.0, 0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.25, 0.35, 0.5, 0.7, 0.95, 1.3, 1.7, 2.2]
    pairs = [(w, f"results/cfrp_bracket_pe{w}.npz") for w in PE]
    pairs = [(w, f) for w, f in pairs if os.path.exists(f)]
    if not pairs:
        EV = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
        pairs = [(w, f"results/cfrp_bracket_ev{w}.npz") for w in EV
                 if os.path.exists(f"results/cfrp_bracket_ev{w}.npz")]
    ws = [w for w, _ in pairs]
    data = [np.load(f) for _, f in pairs]
    S = np.array([1.0 / float(d["c_ratio"]) for d in data])   # stiffness gain
    F = np.array([float(d["fr"]) for d in data])              # frequency gain
    n = len(pairs)
    cols = cm.viridis(np.linspace(0.05, 0.9, n))
    # pick ~6 representative points for the topology thumbnails
    thumb_i = sorted({min(range(n), key=lambda i: abs(ws[i] - tw))
                      for tw in [0.0, 0.06, 0.13, 0.35, 0.95, 2.2]})
    nth = len(thumb_i)

    mesh, mat, P = setup(args.h)
    tri = mtri.Triangulation(mesh.npx[:, 0], mesh.npx[:, 1], mesh.tris)
    cents = np.column_stack([mesh.cx, mesh.cy])
    nn = mesh.npx.shape[0]; wn = np.zeros(nn)
    np.add.at(wn, mesh.tris.ravel(), np.repeat(mesh.area, 3)); wn = np.maximum(wn, 1e-12)
    gx = np.linspace(0, LX, NG); gy = np.linspace(0, LY, int(NG * LY / LX))
    GX, GY = np.meshgrid(gx, gy); px = LX / NG
    rr = max(int(round(MORPH_R / px)), 1); stel = np.ones((2 * rr + 1, 2 * rr + 1))

    def solid_img(y):
        v = np.zeros(nn); np.add.at(v, mesh.tris.ravel(),
                                    np.repeat(mesh.area, 3) * np.repeat(y, 3))
        raw = np.asarray(mtri.LinearTriInterpolator(tri, v / wn)(GX, GY).filled(0.0))
        return gaussian_filter(binary_closing(binary_opening(raw > 0.5, stel), stel)
                               .astype(float), SMOOTH / px)

    fig = plt.figure(figsize=(7.4, 4.9))
    gst = fig.add_gridspec(1, nth, top=0.965, bottom=0.775, left=0.065, right=0.985, wspace=0.05)
    gsb = fig.add_gridspec(1, 1, top=0.63, bottom=0.10, left=0.10, right=0.975)

    # ---- top strip: topology + fiber thumbnails at representative w_omega ----
    for jj, i in enumerate(thumb_i):
        w, d = ws[i], data[i]
        a = fig.add_subplot(gst[0, jj])
        y, th = d["y"], d["th"]
        a.contourf(GX, GY, solid_img(y), levels=[0.5, 1.5], colors=["#e9e9e9"], zorder=0)
        plot_toolpaths_phase(a, cents, th, (0, LX), (0, LY), holes=HOLES,
                             dens=y, dens_thresh=0.5, spacing=0.11, lw=0.35)
        a.contour(GX, GY, solid_img(y), levels=[0.5], colors=[cols[i]], linewidths=2.2,
                  zorder=6)
        for j, (hx, hy, hr) in enumerate(HOLES):
            a.add_patch(plt.Circle((hx, hy), hr, fc="white",
                                   ec="#b3121b" if j == LOAD_HOLE else "k",
                                   lw=1.0, zorder=7))
        a.plot([0, 0], [0.05, LY - 0.05], color="#b3121b", lw=2.5, zorder=8)
        a.set_xlim(-0.05, LX + 0.05); a.set_ylim(-0.05, LY + 0.05)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(cols[i]); s.set_linewidth(2.0)
        tag = "A-frame" if w == 0 else f"$w_\\omega{{=}}{w:g}$"
        a.set_title(f"{tag}\n$\\omega_1\\,{F[i]:.2f}\\times$", fontsize=7.3,
                    color="#222", pad=1.5)

    # ---- bottom: the dense Pareto front (all points) ----
    a = fig.add_subplot(gsb[0, 0])
    xx = np.linspace(S.min() - 0.4, S.max() + 0.4, 100)
    a.plot(xx, np.sqrt(xx), "--", color="#999", lw=1.4,
           label=r"proportionate ($\omega_1^2\propto$ stiffness)")
    a.fill_between(xx, np.sqrt(xx), F.max() + 0.4, color="#6a2c91", alpha=0.05)
    order = np.argsort(ws)   # connect along the EVOLUTION path (increasing w_omega),
    a.plot(S[order], F[order], "-", color="#c9721f", lw=1.6, zorder=3)  # not by stiffness
    a.scatter(S, F, s=55, c=cols, zorder=5, edgecolor="w", linewidth=0.9)
    # label only the representative (thumbnail) points to keep the dense front readable
    for i in thumb_i:
        a.annotate(f"$w_\\omega{{=}}{ws[i]:g}$", (S[i], F[i]),
                   xytext=(S[i], F[i] + 0.12), ha="center", fontsize=7.3, color="#333")
    a.text(0.02, 0.95, "disproportionate region\n(the frequency lever)",
           transform=a.transAxes, color="#6a2c91", fontsize=9, fontweight="bold",
           va="top")
    a.set_xlabel(r"stiffness gain  $c_0/c$  (equal mass)")
    a.set_ylabel(r"fundamental gain  $\omega_1/\omega_1^{\mathrm{base}}$")
    a.set_xlim(S.min() - 0.4, S.max() + 0.5); a.set_ylim(F.min() - 0.25, F.max() + 0.3)
    a.legend(loc="lower right", fontsize=8.5, frameon=False)
    a.grid(alpha=0.25)
    internal_ticks(a, minor=False)

    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    _pv = os.environ.get("FIGPREVIEW")
    if _pv:
        fig.savefig(_pv, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"points ({n}):", [(w, round(float(s), 1), round(float(f), 2))
                             for w, s, f in zip(ws, S, F)])
    print("saved", args.out)


if __name__ == "__main__":
    main()
