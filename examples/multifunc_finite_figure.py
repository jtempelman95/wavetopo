"""
Figure: a metamaterial co-designed for static stiffness AND a phononic band gap,
verified on a finite truncated panel.

The layout answers three questions in order:

  WHAT WAS DESIGNED      (a) cell geometry, (b) fiber orientation, (c) its bands
  DOES CO-DESIGN HELP    (d) Pareto fronts -- stiffness against gap, co-design vs
                             geometry alone.  A single weighted run CANNOT answer
                             this: it lands on one point of a trade-off curve.
                             Ours reached C_bulk 4.80 with a gap of +0.63 from a
                             seed at C_bulk 6.97 with no gap -- i.e. it bought the
                             gap by spending 31% of the stiffness.  The claim is
                             only meaningful as "at matched gap, does orientation
                             buy stiffness geometry alone cannot reach?"
  DOES THE PART WORK     (e) static load on the truncated panel against a solid
                             panel of identical size; (f) energy through the
                             panel; (g) isolation vs number of periods.

    PYTHONPATH=. python examples/multifunc_finite_figure.py
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

C_CO, C_GE = "#2a78d6", "#eb6834"
INK, INK2, GRID, GAPC = "#0b0b0b", "#52514e", "#d8d7d2", "#1baf7a"


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def load_pareto(dirpath):
    """Read the constrained-continuation fronts (examples/multifunc_front.py),
    which write one JSON list per mode: {mode}.json = [{target, C, gap, vol}...].
    Falls back to the older one-dict-per-file schema if present."""
    co, ge = [], []
    for mode, bucket in (("co", co), ("geom", ge)):
        f = os.path.join(dirpath, f"{mode}.json")
        if os.path.exists(f):
            bucket.extend(json.load(open(f)))
    if not co and not ge:                         # legacy: one dict per file
        for f in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
            r = json.load(open(f))
            (co if r["mode"] == "co" else ge).append(r)
    co.sort(key=lambda r: r["gap"]); ge.sort(key=lambda r: r["gap"])
    return co, ge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/multifunc_finite_data.npz")
    ap.add_argument("--periods", default="results/multifunc_periods.npz")
    ap.add_argument("--pareto", default="results/front_conv")
    ap.add_argument("--seedC", type=float, default=6.969)
    ap.add_argument("--out", default="results/multifunc_finite.png")
    args = ap.parse_args()

    d = np.load(args.data)
    n = int(d["cell_n"]); Nx, Ny = int(d["Nx"]), int(d["Ny"])
    lo, hi = float(d["gap_lo"]), float(d["gap_hi"])
    w, ILco = d["omega"], d["IL_co"]
    tri = mtri.Triangulation(d["trix"], d["triy"], d["tris"])
    nelx, nely = int(d["nelx"]), int(d["nely"])
    x0 = float(d["nbuf"]) / n; x1 = x0 + Nx
    ing = (w > lo) & (w < hi)

    fig, ax = plt.subplots(2, 4, figsize=(19.0, 8.4))

    # (a) cell ---------------------------------------------------------- #
    a = ax[0, 0]
    im = a.imshow(d["cell_y"].reshape(n, n), origin="lower", cmap="bone_r",
                  vmin=0, vmax=1)
    a.set_title(f"(a) co-designed cell\n$C_{{bulk}}$={float(d['C_bulk_co']):.2f}"
                ",  vol=0.50", fontsize=10, loc="left", color=INK)
    a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)

    # (b) orientation --------------------------------------------------- #
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

    # (c) bands --------------------------------------------------------- #
    a = ax[0, 2]
    B = d["bands"]
    a.axhspan(lo, hi, color=GAPC, alpha=0.20, lw=0)
    for j in range(B.shape[1]):
        a.plot(B[:, j], color=C_CO, lw=1.5)
    a.set_xlim(0, B.shape[0] - 1); a.set_xticks([])
    a.set_ylabel(r"$\omega$", color=INK, fontsize=9.5)
    a.set_title(f"(c) Bloch bands\ncomplete gap {hi-lo:+.3f}", fontsize=10,
                loc="left", color=INK)
    tidy(a)

    # (d) the Pareto question -------------------------------------------- #
    a = ax[0, 3]
    have_pareto = os.path.isdir(args.pareto) and glob.glob(
        os.path.join(args.pareto, "*.json"))
    def _key(r):
        return r["C_bulk"] if "C_bulk" in r else r["C"]

    if have_pareto:
        co, ge = load_pareto(args.pareto)
        if co:
            a.plot([r["gap"] for r in co], [_key(r) for r in co], "-o",
                   color=C_CO, lw=2.2, ms=6, label=r"co-design $z+\theta$")
        if ge:
            a.plot([r["gap"] for r in ge], [_key(r) for r in ge], "--s",
                   color=C_GE, lw=1.9, ms=5.5, label=r"geometry only $z$")
        # annotate the achievable-gap ceiling: the one place orientation helps
        if co and ge:
            gmax_ge = max(r["gap"] for r in ge)
            gmax_co = max(r["gap"] for r in co)
            if gmax_co > gmax_ge + 0.02:
                a.axvspan(gmax_ge, gmax_co, color=C_CO, alpha=0.10, lw=0)
                a.text(0.5 * (gmax_ge + gmax_co), a.get_ylim()[0], "co only",
                       fontsize=7.5, color=C_CO, ha="center", va="bottom")
        a.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")
    else:
        a.text(0.5, 0.5, "run examples/multifunc_front.py", ha="center",
               va="center", fontsize=9, color=INK2, transform=a.transAxes)
    a.set_xlabel("complete band gap", color=INK, fontsize=9.5)
    a.set_ylabel(r"homogenized stiffness $C_{\rm bulk}$", color=INK,
                 fontsize=9.5)
    a.set_title("(d) stiffness--gap front:\ngeometry matches/beats co-design",
                fontsize=10, loc="left", color=INK)
    tidy(a)

    # (e) STATIC ---------------------------------------------------------- #
    # Plot STRAIN-ENERGY DENSITY, not displacement.  Displacement under an axial
    # pull is the integral of strain along x -- a smooth ramp (corr with x ~0.98)
    # that looks the same for a solid block and hides the microstructure.  Strain
    # energy lights up the load-bearing ligaments and vanishes in the voids, so
    # it shows how the metamaterial actually carries the load.
    from matplotlib.colors import LogNorm
    a = ax[1, 0]
    if "estatic_tri" in d.files:
        e = d["estatic_tri"]
        vmax = np.percentile(e, 99.5)
        tp = a.tripcolor(tri, facecolors=e, cmap="inferno", shading="flat",
                         norm=LogNorm(vmax=vmax, vmin=vmax / 1e3))
        clab = "strain-energy density (log)"
    else:
        us = d["ustatic"]
        tp = a.tripcolor(tri, us / us.max(), cmap="viridis", shading="gouraud")
        clab = "$|u|$ (norm.)"
    for xv in (x0, x1):
        a.axvline(xv, color="c", ls="--", lw=1.0, alpha=0.85)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    kp = float(d["k_eff_co"])
    ks = float(d["k_solid"]) if "k_solid" in d.files else np.nan
    sub = f"$k_{{eff}}$={kp:.2f}"
    if np.isfinite(ks):
        sub += f"  ({100*kp/ks:.0f}% of solid)"
    a.set_title(f"(e) STATIC load: strain energy\n{sub}", fontsize=10,
                loc="left", color=INK)
    cb = fig.colorbar(tp, ax=a, fraction=0.030, pad=0.03)
    cb.set_label(clab, fontsize=8, color=INK2)

    # (f) energy profile --------------------------------------------------- #
    a = ax[1, 1]
    xs_ = np.linspace(0, float(d["Lx"]), nelx + 1)
    for key, col, ls, lab in (("msolid", INK2, ":", "solid reference"),
                              ("mpass", C_GE, "--", "pass band"),
                              ("mgap", C_CO, "-", "in the gap")):
        prof = (d[key].reshape(nely + 1, nelx + 1) ** 2).mean(axis=0)
        a.semilogy(xs_, prof / prof.max(), color=col, ls=ls, lw=2.0, label=lab)
    a.axvspan(x0, x1, color=GAPC, alpha=0.13, lw=0)
    a.set_xlabel("x  (cells)", color=INK, fontsize=9.5)
    a.set_ylabel(r"$\langle|u|^2\rangle_y$ (norm.)", color=INK, fontsize=9.5)
    a.set_title("(f) DYNAMIC: energy through the panel\nshaded = crystal",
                fontsize=10, loc="left", color=INK)
    a.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="lower left")
    tidy(a)

    # (g) isolation vs periods --------------------------------------------- #
    a = ax[1, 2]
    if os.path.exists(args.periods):
        p = np.load(args.periods)
        a.semilogy(p["Nx"], p["IL_gap"], color=C_CO, lw=2.2, marker="o", ms=5,
                   label="in the gap")
        a.semilogy(p["Nx"], p["IL_pass"], color=INK2, lw=1.6, ls="--",
                   marker="s", ms=4, label="pass-band control")
        a.set_title("(g) isolation vs periods\n$\\kappa a$="
                    f"{float(p['kappa_a']):.2f}/cell", fontsize=10, loc="left",
                    color=INK)
        a.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")
    a.set_xlabel("number of periods $N_x$", color=INK, fontsize=9.5)
    a.set_ylabel("insertion loss", color=INK, fontsize=9.5)
    tidy(a)

    # (h) spectrum ---------------------------------------------------------- #
    a = ax[1, 3]
    a.axvspan(lo, hi, color=GAPC, alpha=0.20, lw=0)
    a.semilogy(w, ILco, color=C_CO, lw=2.1, marker="o", ms=3.2)
    a.set_xlabel(r"$\omega$", color=INK, fontsize=9.5)
    a.set_ylabel(r"insertion loss $T_{\rm solid}/T$", color=INK, fontsize=9.5)
    a.set_title(f"(h) {Nx}-period spectrum\nin-gap mean "
                f"{ILco[ing].mean():.0f}$\\times$", fontsize=10, loc="left",
                color=INK)
    tidy(a)

    fig.suptitle(
        f"Metamaterial co-designed for stiffness and a band gap, on a finite "
        f"{Nx}$\\times${Ny}-cell panel:  gap {hi-lo:.2f},  in-gap isolation "
        f"{ILco[ing].mean():.0f}$\\times$ at {Nx} periods,  static "
        f"$k_{{\\rm eff}}$={float(d['k_eff_co']):.2f}",
        fontsize=12.5, y=0.985, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(args.out, dpi=185, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
