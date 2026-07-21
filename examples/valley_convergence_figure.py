"""
Figure: the valley guided band is a property of the crystal only when the mesh
conforms to the rods.

Panel (a)  transmission along the domain wall vs frequency, for rod-conforming
           and centroid-painted meshes at three resolutions.  The conforming
           curves collapse onto one another and peak inside the independently
           computed bulk gap; the painted curves scatter over orders of
           magnitude and, at the coarsest resolution, peak outside the gap
           altogether.
Panel (b)  wall confinement vs frequency -- the quantity that separates a bound
           kink mode from ordinary bulk transport, which transmission alone
           cannot do.

    python examples/valley_convergence_figure.py
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference categorical palette, slots 1 and 2 in fixed order.  Line style is a
# secondary encoding so the two methods are never distinguished by colour alone.
C_CONF, C_PAINT = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

STYLE = {0.12: dict(lw=1.4, alpha=0.55, marker="o", ms=3.4),
         0.09: dict(lw=2.0, alpha=0.80, marker="s", ms=3.8),
         0.06: dict(lw=2.6, alpha=1.00, marker="^", ms=4.2)}


def tidy(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="results/valley_band_convergence.npz")
    ap.add_argument("--conf", default="results/valley_confinement.npz")
    ap.add_argument("--gap", type=float, nargs=2, default=[7.014, 7.567],
                    help="bulk complete gap from the conforming unit cell "
                         "(41x41 BZ sampling, converged)")
    ap.add_argument("--kink", type=float, nargs=2, default=[7.014, 7.27],
                    help="topologically protected window: the ribbon's "
                         "wall-localized branch INSIDE the complete gap")
    ap.add_argument("--out", default="results/valley_convergence.png")
    args = ap.parse_args()

    d = np.load(args.conv)
    w = d["omega"]
    have_conf = os.path.exists(args.conf)

    fig, axes = plt.subplots(1, 2 if have_conf else 1,
                             figsize=(12.4 if have_conf else 6.6, 4.5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.axvspan(*args.gap, color="#8a8a85", alpha=0.12, lw=0, zorder=0)
    ax.axvspan(*args.kink, color="#1baf7a", alpha=0.16, lw=0, zorder=0)
    ax.text(np.mean(args.kink), 0.965, "protected", ha="center", va="top",
            fontsize=8.5, color=INK2, transform=ax.get_xaxis_transform())
    ax.text(np.mean([args.kink[1], args.gap[1]]), 0.965, "gap only",
            ha="center", va="top", fontsize=8.5, color=INK2,
            transform=ax.get_xaxis_transform())
    for meth, col in (("conforming", C_CONF), ("painted", C_PAINT)):
        for h, st in STYLE.items():
            key = f"{meth}_h{h:g}"
            if key not in d:
                continue
            ax.semilogy(w, np.maximum(d[key], 1e-12), color=col,
                        ls="-" if meth == "conforming" else "--",
                        label=f"{meth}, h={h:g}", **st)
    tidy(ax)
    ax.set_xlabel(r"frequency  $\omega$", color=INK, fontsize=10)
    ax.set_ylabel(r"transmission  $T=\int_{\rm probe}|u|^2\,dA$",
                  color=INK, fontsize=10)
    ax.set_title("(a)  guided-band transmission vs mesh construction",
                 color=INK, fontsize=11, loc="left", pad=9)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left",
              labelcolor=INK2)

    if have_conf:
        c = np.load(args.conf)
        ax2 = axes[1]
        ax2.axvspan(*args.gap, color="#8a8a85", alpha=0.12, lw=0, zorder=0)
        ax2.axvspan(*args.kink, color="#1baf7a", alpha=0.16, lw=0, zorder=0)
        ax2.text(np.mean(args.kink), 0.30, "protected", ha="center",
                 va="center", fontsize=8.5, color=INK2, rotation=90,
                 transform=ax2.get_xaxis_transform())
        ax2.text(7.72, 0.30, "guided, not protected", ha="center", va="center",
                 fontsize=8.5, color=INK2, rotation=90,
                 transform=ax2.get_xaxis_transform())
        ax2.plot(c["omega"], c["C"], color=C_CONF, lw=2.2, marker="s", ms=3.8)
        ax2.axhline(0.5, color=INK2, lw=1.0, ls=":", zorder=1)
        ax2.text(c["omega"].min(), 0.52, "half the interior energy",
                 fontsize=8, color=INK2, va="bottom")
        tidy(ax2)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel(r"frequency  $\omega$", color=INK, fontsize=10)
        ax2.set_ylabel(r"wall confinement  $C$", color=INK, fontsize=10)
        ax2.set_title(r"(b)  is the energy bound to the wall?  "
                      f"(conforming, h={float(c['h']):g})",
                      color=INK, fontsize=11, loc="left", pad=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
