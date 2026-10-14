"""Hero figure for the multifunctional CFRP panel: one curvature-feasible, mirror-
symmetric fiber layout that routes incident vibration to a harvester, shields a
payload core, protects the fastener rims -- across an operational frequency band.

Reads results/dolfinx_panel_band_data.npz (fields + design) and
results/dolfinx_panel_sweep.npz (the before/after bandwidth curve).

    PYTHONPATH=/home/jrt/wavetopo python3 examples/figures/dfx_panel_figure.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from wavetopo.dolfinx_viz import plot_director_tows, plot_toolpaths_phase
from wavetopo import figlib as F

DATA = "results/dolfinx_panel_band_data.npz"
SWEEP = "results/dolfinx_panel_sweep.npz"
OUT = "docs/paper/figs/dfx_panel.png"


def _overlays(ax, d, tc):
    px, py, ph = d["pay"]; sx, sy, sh = d["sensor"]; guard = float(d["guard"])
    ax.add_patch(plt.Rectangle((px-ph, py-ph), 2*ph, 2*ph, ec="#66ffcc",
                               fc="none", lw=1.5, ls="--"))
    ax.add_patch(plt.Rectangle((sx-sh, sy-sh), 2*sh, 2*sh, ec="#bfefff",
                               fc="none", lw=1.7, ls="--"))
    for cx, cy, r in d["holes"]:
        ax.plot(cx+r*np.cos(tc), cy+r*np.sin(tc), "w-", lw=1.0)
        ax.plot(cx+(r+guard)*np.cos(tc), cy+(r+guard)*np.sin(tc), "r:", lw=1.2)


def build():
    d = np.load(DATA)
    Lx, Ly = float(d["Lx"]), float(d["Ly"])
    tri = mtri.Triangulation(d["trix"], d["triy"], d["tris"])
    m0, m1 = d["m0"], d["m1"]
    route, shield, prot = float(d["route"]), float(d["shield"]), float(d["prot"])
    stiff = float(d["stiff"]) if "stiff" in d.files else None
    z1, zall = float(d["zmax"]), float(d["zeta_all"])
    holes = [tuple(map(float, h)) for h in d["holes"]]
    freqs = d["freqs"]
    tc = np.linspace(0, 2*np.pi, 200)
    vmax = np.percentile(m1, 99.0)

    _P = bool(os.environ.get("FIGNOSUP"))         # print mode: near-textwidth
    fig = plt.figure(figsize=(7.3, 6.3) if _P else (13.5, 10.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.80],
                          hspace=0.30, wspace=0.19, top=0.955, bottom=0.065)

    # ---- (a) baseline field -------------------------------------------------
    axB = fig.add_subplot(gs[0, 0])
    tp = axB.tripcolor(tri, m0, cmap="magma", vmax=vmax, shading="gouraud")
    _overlays(axB, d, tc)
    axB.set_aspect("equal"); axB.set_xlim(0, Lx); axB.set_ylim(0, Ly)
    axB.set_title("(a) straight fibers (baseline)", fontsize=10)
    plt.colorbar(tp, ax=axB, fraction=0.023, pad=0.01)

    # ---- (b) the money shot: optimized field --------------------------------
    axA = fig.add_subplot(gs[0, 1])
    tp = axA.tripcolor(tri, m1, cmap="magma", vmax=vmax, shading="gouraud")
    _overlays(axA, d, tc)
    axA.set_aspect("equal"); axA.set_xlim(0, Lx); axA.set_ylim(0, Ly)
    axA.set_title("(b) optimized fiber layout", fontsize=10)
    px, py, _ = d["pay"]; sx, sy, _ = d["sensor"]
    axA.annotate("shielded\npayload", (px, py), color="#66ffcc", ha="center",
                 va="center", fontsize=8, fontweight="bold")
    axA.annotate("harvester", (sx, sy-0.6), color="#bfefff", ha="center",
                 fontsize=8, fontweight="bold")
    plt.colorbar(tp, ax=axA, fraction=0.023, pad=0.01)

    # ---- (c) director tows --------------------------------------------------
    axC = fig.add_subplot(gs[1, 0])
    lc = plot_director_tows(axC, d["cent"], d["thopt"], (0, Lx), (0, Ly),
                            holes=holes, spacing=0.16, lw=1.2)
    _overlays(axC, d, tc)
    axC.set_aspect("equal"); axC.set_xlim(0, Lx); axC.set_ylim(0, Ly)
    axC.set_title("(c) director field (the one layout)", fontsize=10)
    cb = plt.colorbar(lc, ax=axC, fraction=0.023, pad=0.01,
                      ticks=[0, np.pi/2, np.pi])
    cb.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])

    # ---- (d) printable fiber toolpaths --------------------------------------
    axT = fig.add_subplot(gs[1, 1])
    plot_toolpaths_phase(axT, d["cent"], d["thopt"], (0, Lx), (0, Ly),
                         holes=holes, spacing=0.16, n=340)
    _overlays(axT, d, tc)
    axT.set_aspect("equal"); axT.set_xlim(0, Lx); axT.set_ylim(0, Ly)
    axT.set_title(f"(d) as-printed toolpaths (max$|\\zeta|$={z1:.1f}$\\leq${zall:.1f})",
                  fontsize=10)

    # ---- (e) operational bandwidth curve -----------------------------------
    axD = fig.add_subplot(gs[2, :])
    if os.path.exists(SWEEP):
        s = np.load(SWEEP)
        fs = s["freqs"]
        styles = {"tone": ("--", "single-tone design"),
                  "band": ("-", "band-optimized design")}
        colors = dict(sensor="#2ca02c", payload="#d62728", rims="#1f77b4")
        for name, (ls, lab) in styles.items():
            if name not in s:
                continue
            c = s[name]
            axD.plot(c[:, 0], c[:, 1], ls, color=colors["sensor"], lw=1.8,
                     label=f"harvester gain ({lab})" if name == "band" else None)
            axD.plot(c[:, 0], c[:, 2], ls, color=colors["payload"], lw=1.8,
                     label=f"payload (shield)" if name == "band" else None)
            axD.plot(c[:, 0], c[:, 3], ls, color=colors["rims"], lw=1.8,
                     label=f"fastener rims (protect)" if name == "band" else None)
        axD.axhline(1.0, color="0.5", lw=0.8, ls=":")
        axD.axvspan(float(freqs.min()), float(freqs.max()), color="0.85", alpha=0.5,
                    zorder=0, label="optimized band")
        axD.set_xlabel(r"drive frequency $\omega$")
        axD.set_ylabel("energy ratio vs. straight fibers")
        axD.set_yscale("log")
        axD.set_ylim(0.055, 6.0)                   # headroom so the legend clears the curves
        axD.set_title("(e) operational bandwidth: solid = band-optimized (flat across "
                      "the band), dashed = single-tone (rims degrade at the edges)",
                      fontsize=10)
        axD.legend(fontsize=7.5, ncol=3, loc="upper center")
        axD.grid(alpha=0.3, which="both")
    else:
        axD.text(0.5, 0.5, "run dolfinx_panel_sweep.py for the bandwidth curve",
                 ha="center", va="center", transform=axD.transAxes)
        axD.axis("off")

    _st = f", {stiff:.2f}$\\times$ stiffer" if stiff else ""
    fig.suptitle(f"A multifunctional CFRP panel — one fiber layout: route "
                 f"{route:.1f}$\\times$, shield {shield:.2f}$\\times$, protect "
                 f"{prot:.2f}$\\times${_st}, across a band", y=0.995,
                 fontsize=12.5, fontweight="bold")
    return fig


def main():
    F.journal()                                   # publication typography
    fig = build()
    os.makedirs("docs/paper/figs", exist_ok=True)
    if os.environ.get("FIGNOSUP"):                # print mode: uniform paper fonts
        F.harmonize_fonts(fig, 0.90)              # 7.2in fig displayed at ~textwidth
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig("results/dolfinx_panel_hero.png", dpi=140, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
