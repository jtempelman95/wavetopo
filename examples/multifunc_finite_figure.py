"""
Figure for examples/multifunc_finite.py + multifunc_periods.py.

Row 1  the designed cell: density, fiber orientation, Bloch bands, and the
       insertion-loss spectrum of the finite panel against the predicted gap.
Row 2  the finite panel itself: a solid reference panel, the same panel driven
       inside the gap, the same driven in the pass band, and insertion loss vs
       the number of periods with a pass-band control.

Insertion loss is IL = T_solid/T_panel against an identical solid panel -- same
source, sponge, buffers and mesh -- so source coupling and fixturing divide out.

    PYTHONPATH=. python examples/multifunc_finite_figure.py
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

C_CO, C_GE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
GAPC = "#1baf7a"


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/multifunc_finite_data.npz")
    ap.add_argument("--periods", default="results/multifunc_periods.npz")
    ap.add_argument("--out", default="results/multifunc_finite.png")
    args = ap.parse_args()

    d = np.load(args.data)
    n = int(d["cell_n"]); Nx, Ny = int(d["Nx"]), int(d["Ny"])
    lo, hi = float(d["gap_lo"]), float(d["gap_hi"])
    w = d["omega"]
    ILco, ILge = d["IL_co"], d["IL_ge"]
    tri = mtri.Triangulation(d["trix"], d["triy"], d["tris"])
    x0 = float(d["nbuf"]) / n
    x1 = x0 + Nx

    fig, ax = plt.subplots(2, 4, figsize=(19.5, 8.0))

    # ---- (a) cell density ------------------------------------------- #
    a = ax[0, 0]
    im = a.imshow(d["cell_y"].reshape(n, n), origin="lower", cmap="bone_r",
                  vmin=0, vmax=1)
    a.set_title(f"(a) co-designed cell\n$C_{{bulk}}$={float(d['C_bulk_co']):.2f}, "
                f"vol=0.50", fontsize=10, loc="left", color=INK)
    a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)

    # ---- (b) orientation --------------------------------------------- #
    a = ax[0, 1]
    th = d["cell_th"].reshape(n, n)
    im = a.imshow(np.rad2deg(th), origin="lower", cmap="twilight",
                  vmin=-90, vmax=90)
    s = max(n // 13, 1)
    yy, xx = np.mgrid[0:n:s, 0:n:s]
    a.quiver(xx, yy, np.cos(th[::s, ::s]), np.sin(th[::s, ::s]), color="k",
             pivot="mid", headwidth=0, headlength=0, headaxislength=0,
             scale=24, width=0.007)
    a.set_title(r"(b) fiber orientation $\theta$", fontsize=10, loc="left",
                color=INK)
    a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)

    # ---- (c) bands ---------------------------------------------------- #
    a = ax[0, 2]
    B = d["bands"]
    a.axhspan(lo, hi, color=GAPC, alpha=0.18, lw=0)
    for j in range(B.shape[1]):
        a.plot(B[:, j], color=C_CO, lw=1.5)
    a.set_xlim(0, B.shape[0] - 1); a.set_xticks([])
    a.set_ylabel(r"$\omega$", color=INK, fontsize=9.5)
    a.set_title(f"(c) Bloch bands\ncomplete gap {hi-lo:+.3f}", fontsize=10,
                loc="left", color=INK)
    tidy(a)

    # ---- (d) IL spectrum ---------------------------------------------- #
    # Raw IL is NOT comparable between designs: the geometry-only panel
    # transmits ~9x less at EVERY frequency, in and out of gap alike, so on raw
    # IL it outranks the co-design for reasons that have nothing to do with a
    # band gap.  Normalizing each curve by its own out-of-own-gap median removes
    # that broadband offset and leaves what the gap itself contributes.
    a = ax[0, 3]
    glo, ghi = float(d["ge_gap_lo"]), float(d["ge_gap_hi"])
    base_co = np.median(ILco[(w < lo) | (w > hi)])
    base_ge = np.median(ILge[(w < glo) | (w > ghi)])
    a.axvspan(lo, hi, color=GAPC, alpha=0.18, lw=0)
    a.axvspan(glo, ghi, color=C_GE, alpha=0.10, lw=0)
    a.axhline(1.0, color=INK2, lw=1.0, ls=":")
    a.semilogy(w, ILco / base_co, color=C_CO, lw=2.1, marker="o", ms=3.2,
               label=f"co-design (gap {float(d['gap_co']):.2f})")
    a.semilogy(w, ILge / base_ge, color=C_GE, lw=1.7, ls="--", marker="s",
               ms=3.0, label=f"geometry only (gap {float(d['gap_ge']):.2f})")
    a.set_xlabel(r"$\omega$", color=INK, fontsize=9.5)
    a.set_ylabel("isolation / own broadband level", color=INK, fontsize=9.5)
    a.set_title(f"(d) {Nx}-period panel: gap contribution\n"
                "green = co-design gap, orange = geom-only gap",
                fontsize=10, loc="left", color=INK)
    a.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower right")
    tidy(a)

    # ---- (e) energy profile along the panel --------------------------- #
    # The 2D fields saturate near the drive, so the decay THROUGH the crystal is
    # hard to read off them.  The y-averaged profile shows it directly, and in
    # the gap it should be a straight line on a log axis (evanescent decay).
    nelx, nely = int(d["nelx"]), int(d["nely"])
    xs = np.linspace(0, float(d["Lx"]), nelx + 1)
    a = ax[1, 0]
    for key, col, ls, lab in (("msolid", INK2, ":", "solid reference"),
                              ("mpass", C_GE, "--", "pass band"),
                              ("mgap", C_CO, "-", "in the gap")):
        prof = (d[key].reshape(nely + 1, nelx + 1) ** 2).mean(axis=0)
        a.semilogy(xs, prof / prof.max(), color=col, ls=ls, lw=2.0, label=lab)
    a.axvspan(x0, x1, color=GAPC, alpha=0.13, lw=0)
    a.set_xlabel("x  (cells)", color=INK, fontsize=9.5)
    a.set_ylabel(r"$\langle|u|^2\rangle_y$  (normalized)", color=INK,
                 fontsize=9.5)
    a.set_title("(e) energy profile through the panel\nshaded = crystal",
                fontsize=10, loc="left", color=INK)
    a.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="lower left")
    tidy(a)

    # ---- (f,g) fields, each normalized to its own peak ----------------- #
    DBFLOOR = -45.0
    for col, (key, lab) in enumerate(
            [("mgap", f"(f) driven IN the gap  $\\omega$="
                      f"{float(d['w_gap']):.3f}"),
             ("mpass", f"(g) driven in pass band  $\\omega$="
                       f"{float(d['w_pass']):.3f}")]):
        v = d[key]
        vdb = 20.0 * np.log10(np.maximum(v, 1e-14) / v.max())
        a = ax[1, col + 1]
        tp = a.tripcolor(tri, np.clip(vdb, DBFLOOR, 0.0), cmap="magma",
                         shading="gouraud", vmin=DBFLOOR, vmax=0.0)
        for xv in (x0, x1):
            a.axvline(xv, color="c", ls="--", lw=1.0, alpha=0.85)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(lab, fontsize=10, loc="left", color=INK)
        cb = fig.colorbar(tp, ax=a, fraction=0.030, pad=0.03)
        cb.set_label("dB re own peak", fontsize=8, color=INK2)

    # ---- (h) IL vs number of periods ---------------------------------- #
    a = ax[1, 3]
    if os.path.exists(args.periods):
        p = np.load(args.periods)
        a.semilogy(p["Nx"], p["IL_gap"], color=C_CO, lw=2.2, marker="o", ms=5,
                   label="in the gap")
        a.semilogy(p["Nx"], p["IL_pass"], color=INK2, lw=1.6, ls="--",
                   marker="s", ms=4, label="pass-band control")
        ka = float(p["kappa_a"])
        a.set_title(f"(h) isolation vs periods\n$\\kappa a$={ka:.2f}/cell",
                    fontsize=10, loc="left", color=INK)
        a.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")
    a.set_xlabel("number of periods $N_x$", color=INK, fontsize=9.5)
    a.set_ylabel(r"insertion loss", color=INK, fontsize=9.5)
    tidy(a)

    ing = (w > lo) & (w < hi)
    fig.suptitle(
        "Co-designed stiff + band-gap metamaterial, verified on a finite "
        f"truncated panel  ({Nx}$\\times${Ny} cells, free top/bottom edges):  "
        f"in-gap insertion loss {ILco[ing].mean():.0f}$\\times$,  "
        f"static $k_{{\\rm eff}}$={float(d['k_eff_co']):.2f}",
        fontsize=12.5, y=0.99, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(args.out, dpi=185, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
