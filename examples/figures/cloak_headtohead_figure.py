"""The cloak story in one figure, three acts.  Panel titles are kept SHORT (the full
description is in the LaTeX caption).  Top row: wave fields |u|.  Bottom row: fiber
toolpaths over the curvature |zeta|, plus the worst-fiber curvature into the shell.

Importable builder: ``build()`` returns a live Figure; ``main()`` is the batch CLI.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata

from wavetopo.dolfinx_viz import plot_toolpaths_phase
from examples.figures._paperfig import set_style, internal_ticks, harmonize_paper

SCALE = 0.792   # display scale in the manuscript (includegraphics width / pdf width)


def build(data="results/cloak_headtohead.npz"):
    """Build and return the three-act cloak figure from a saved .npz."""
    d = np.load(data)
    CX, CY, RV, RDES, RC = [float(d[k]) for k in ("CX", "CY", "RV", "R_DES", "RC")]
    Lx, Ly = float(d["Lx"]), float(d["Ly"])
    J0, Jan, Junc, Jopt = [float(d[k]) for k in ("J0", "Jan", "Junc", "Jopt")]
    red_an, red_unc, red_opt = J0 / Jan, J0 / Junc, J0 / Jopt
    zeta_all = 1.0 / RV
    tri = mtri.Triangulation(d["trix"], d["triy"], d["tris"])
    trf = mtri.Triangulation(d["trfx"], d["trfy"], d["trfs"])
    cent, th_an, th_unc, th_opt = d["cent"], d["th_an"], d["th_unc"], d["th_opt"]

    set_style()
    RED, GOLD, GREEN = "#b3121b", "#8a5a00", "#1a7a3a"
    fig = plt.figure(figsize=(8.2, 3.49))   # authored near print width so fonts stay legible
    gs = fig.add_gridspec(2, 1, height_ratios=[0.52, 1.0], hspace=0.07,
                          left=0.015, right=0.985, top=0.95, bottom=0.075)
    gst = gs[0].subgridspec(1, 5, wspace=0.06)
    # 5 cols: three curvemaps, a thin spacer for the colorbar, then the wide data plot
    gsb = gs[1].subgridspec(1, 5, width_ratios=[1, 1, 1, 0.42, 1.7], wspace=0.16)
    vmax = np.percentile(d["uref"], 99.2)

    def wave(ax, T, u, title, tcolor="k"):
        ax.tripcolor(T, u, cmap="magma", vmin=0, vmax=vmax, shading="gouraud", rasterized=True)
        ax.add_patch(plt.Circle((CX, CY), RV, fc="white", ec="k", lw=0.8, zorder=5))
        ax.add_patch(plt.Circle((CX, CY), RDES, fc="none", ec="#39c", ls=":", lw=1.0, zorder=5))
        ax.set_xlim(0.6, Lx - 0.6); ax.set_ylim(0.6, Ly - 0.6); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11, loc="left", color=tcolor)

    wave(fig.add_subplot(gst[0, 0]), trf, d["uref"], "(a) reference")
    wave(fig.add_subplot(gst[0, 1]), tri, d["ustr"], "(b) uncloaked")
    wave(fig.add_subplot(gst[0, 2]), tri, d["uan"], f"(c) analytical, {red_an:.1f}$\\times$", RED)
    wave(fig.add_subplot(gst[0, 3]), tri, d["uunc"], f"(d) unconstrained, {red_unc:.0f}$\\times$", GOLD)
    wave(fig.add_subplot(gst[0, 4]), tri, d["uopt"], f"(e) curl-constrained, {red_opt:.1f}$\\times$", GREEN)

    # ---- bottom: fiber curvature ----
    ng = 460
    gx = np.linspace(CX - RDES, CX + RDES, ng); gy = np.linspace(CY - RDES, CY + RDES, ng)
    GX, GY = np.meshgrid(gx, gy); RP = np.hypot(GX - CX, GY - CY)
    shell = (RP > RV) & (RP < RDES)

    def curl_grid(th_cell):
        c2 = griddata(cent, np.cos(2 * th_cell), (GX, GY), method="linear", fill_value=1.0)
        s2 = griddata(cent, np.sin(2 * th_cell), (GX, GY), method="linear", fill_value=0.0)
        dcy, dcx = np.gradient(c2, gy[1] - gy[0], gx[1] - gx[0])
        dsy, dsx = np.gradient(s2, gy[1] - gy[0], gx[1] - gx[0])
        thx = 0.5 * (c2 * dsx - s2 * dcx); thy = 0.5 * (c2 * dsy - s2 * dcy)
        thg = 0.5 * np.arctan2(s2, c2)
        z = np.abs(np.cos(thg) * thx + np.sin(thg) * thy); z[~shell] = np.nan
        return z

    def curvemap(ax, th_cell, title, tcolor):
        Z = curl_grid(th_cell)
        im = ax.pcolormesh(GX, GY, Z, cmap="inferno", norm=LogNorm(vmin=0.3, vmax=30),
                           shading="auto", rasterized=True)
        plot_toolpaths_phase(ax, cent, th_cell, (CX - RDES, CX + RDES), (CY - RDES, CY + RDES),
                             holes=[(CX, CY, RV)], spacing=0.11, mode="line",
                             color="#39e5ff", lw=0.7)
        ax.add_patch(plt.Circle((CX, CY), RV, fc="white", ec="k", lw=1.0, zorder=6))
        ax.set_xlim(CX - RDES, CX + RDES); ax.set_ylim(CY - RDES, CY + RDES)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11, loc="left", color=tcolor)
        return im

    ax0 = fig.add_subplot(gsb[0, 0]); ax1 = fig.add_subplot(gsb[0, 1]); ax2 = fig.add_subplot(gsb[0, 2])
    curvemap(ax0, th_an, "(f) analytical", RED)
    im = curvemap(ax1, th_unc, "(g) unconstrained", GOLD)
    curvemap(ax2, th_opt, "(h) curl-constrained", GREEN)
    cb = fig.colorbar(im, ax=[ax0, ax1, ax2], fraction=0.028, pad=0.015, extend="max")
    cb.set_label(r"$|\zeta|$")

    # worst-fiber curvature into the shell (the one data plot -> internal ticks all 4 sides)
    ax = fig.add_subplot(gsb[0, 4])
    Zan, Zun, Zop = curl_grid(th_an), curl_grid(th_unc), curl_grid(th_opt)
    fr = np.linspace(0.012, 0.9, 45); rr = RV + fr * (RDES - RV)

    def band(Z, r):
        v = Z[np.abs(RP - r) < 0.025]; v = v[np.isfinite(v)]
        return np.percentile(v, 92) if v.size else np.nan

    ax.plot(fr, [band(Zan, r) for r in rr], color=RED, lw=2.5, label=f"analytical ({red_an:.1f}$\\times$)")
    ax.plot(fr, [band(Zun, r) for r in rr], color="#d99400", lw=2.5, label=f"unconstrained ({red_unc:.0f}$\\times$)")
    ax.plot(fr, [band(Zop, r) for r in rr], color=GREEN, lw=2.5, label=f"curl-constrained ({red_opt:.1f}$\\times$)")
    ax.axhline(zeta_all, color="#333", ls="--", lw=1.6, label=fr"AFP limit $1/R_V={zeta_all:.2f}$")
    ax.fill_between([0, 0.9], zeta_all, 120, color=RED, alpha=0.06)
    ax.set_yscale("log"); ax.set_ylim(0.3, 120); ax.set_xlim(0, 0.9)
    ax.set_xlabel("fraction into the cloaking shell")
    ax.set_ylabel(r"worst-fiber $|\zeta|$ (92nd pct.)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(r"(i) worst-fiber curvature", fontsize=11, loc="left")
    ax.grid(alpha=0.22, which="major")
    internal_ticks(ax, minor=False)
    fig._reductions = (red_an, red_unc, red_opt)
    return harmonize_paper(fig, SCALE)


def main():
    import argparse
    import os
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/cloak_headtohead.npz")
    ap.add_argument("--out", default="results/cloak_headtohead_fig.pdf")
    args = ap.parse_known_args()[0]
    fig = build(args.data)
    fig.savefig(args.out, bbox_inches="tight")
    pv = os.environ.get("FIGPREVIEW")
    if pv:
        fig.savefig(pv, dpi=110, bbox_inches="tight")
    a, u, o = fig._reductions
    print(f"reductions: analytic {a:.1f}x  unconstrained {u:.1f}x  constrained {o:.1f}x")
    print("saved", args.out)


if __name__ == "__main__":
    main()
