"""Mode-separation study: maximize omega_1 while minimizing omega_2.  A single anisotropic
co-design drives the first two natural frequencies TOGETHER --- the gap omega_2/omega_1
collapses from the baseline value toward 1 (a near-degenerate fundamental pair) as the
mode-2 weight w2 is turned up.  Top: topology + fiber thumbnails along the sweep.  Bottom
left: the gap closing.  Bottom middle/right: the first two mode shapes of the most
mode-merged design --- two near-degenerate modes the fiber design has tuned to coincide."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing

from examples.drivers.cfrp_bracket_modal import setup, HOLES, LOAD_HOLE, LX, LY
from wavetopo.dolfinx_viz import plot_toolpaths_phase

NG, MORPH_R, SMOOTH = 700, 0.016, 0.010


def build(h=0.025):
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    w2s = [0.0, 0.25, 0.5, 1.0, 2.0]
    pairs = [(w, f"results/cfrp_bracket_m2_{w}.npz") for w in w2s]
    pairs = [(w, f) for w, f in pairs if os.path.exists(f)]
    w2s = [w for w, _ in pairs]; data = [np.load(f) for _, f in pairs]
    # stored 'gap' is lam2/lam1 = (omega2/omega1)^2; the FREQUENCY ratio is its sqrt
    gap = np.array([np.sqrt(float(d["gap"])) for d in data])   # omega_2 / omega_1
    plate = np.sqrt(float(data[0]["lam2_0"]) / float(data[0]["lam1_0"]))  # equal-mass plate
    n = len(pairs); cols = cm.plasma(np.linspace(0.05, 0.85, n))

    mesh, mat, P = setup(h)
    tri = mtri.Triangulation(mesh.npx[:, 0], mesh.npx[:, 1], mesh.tris)
    cents = np.column_stack([mesh.cx, mesh.cy])
    nn = mesh.npx.shape[0]; wn = np.zeros(nn)
    np.add.at(wn, mesh.tris.ravel(), np.repeat(mesh.area, 3)); wn = np.maximum(wn, 1e-12)
    gx = np.linspace(0, LX, NG); gy = np.linspace(0, LY, int(NG * LY / LX))
    GX, GY = np.meshgrid(gx, gy); px = LX / NG
    rr = max(int(round(MORPH_R / px)), 1); stel = np.ones((2 * rr + 1, 2 * rr + 1))

    def e2n(ef):
        v = np.zeros(nn); np.add.at(v, mesh.tris.ravel(), np.repeat(mesh.area, 3) * np.repeat(ef, 3))
        return v / wn

    def solid_img(y):
        raw = np.asarray(mtri.LinearTriInterpolator(tri, e2n(y))(GX, GY).filled(0.0))
        return gaussian_filter(binary_closing(binary_opening(raw > 0.5, stel), stel)
                               .astype(float), SMOOTH / px)

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(1, n, top=0.88, bottom=0.60, left=0.05, right=0.98, wspace=0.06)
    gsb = fig.add_gridspec(1, 3, top=0.47, bottom=0.08, left=0.07, right=0.97, wspace=0.16)

    def holes(a, load=True):
        for j, (hx, hy, hr) in enumerate(HOLES):
            a.add_patch(plt.Circle((hx, hy), hr, fc="white",
                                   ec="#b3121b" if (load and j == LOAD_HOLE) else "k",
                                   lw=1.2, zorder=7))
        a.plot([0, 0], [0.05, LY - 0.05], color="#b3121b", lw=2.5, zorder=8)
        a.set_xlim(-0.05, LX + 0.05); a.set_ylim(-0.05, LY + 0.05)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])

    # top strip: topology + fibers vs w2
    for i, (w, d) in enumerate(zip(w2s, data)):
        a = fig.add_subplot(gs[0, i]); y, th = d["y"], d["th"]
        a.contourf(GX, GY, solid_img(y), levels=[0.5, 1.5], colors=["#eaeaea"], zorder=0)
        plot_toolpaths_phase(a, cents, th, (0, LX), (0, LY), holes=HOLES, dens=y,
                             dens_thresh=0.5, spacing=0.11, lw=0.35)
        a.contour(GX, GY, solid_img(y), levels=[0.5], colors=[cols[i]], linewidths=2.2,
                  zorder=6)
        holes(a)
        for s in a.spines.values():
            s.set_edgecolor(cols[i]); s.set_linewidth(2.0)
        a.set_title(f"$w_2{{=}}{w:g}$\n$\\omega_2/\\omega_1={np.sqrt(float(d['gap'])):.2f}$",
                    fontsize=9, pad=2)

    # bottom-left: gap closing (plate baseline -> max-w1 design -> mode-2 merged)
    a = fig.add_subplot(gsb[0, 0])
    a.axhline(plate, color="#c0392b", ls=":", lw=1.6,
              label=f"equal-mass plate ({plate:.2f})")
    a.plot(w2s, gap, "-o", color="#6a2c91", lw=2.4, ms=8,
           label=r"max-$\omega_1$ $+$ min-$\omega_2$")
    a.axhline(1.0, color="#333", ls="--", lw=1.4, label="degenerate ($\\omega_2=\\omega_1$)")
    a.set_xlabel("mode-2 weight $w_2$"); a.set_ylabel(r"frequency gap $\omega_2/\omega_1$")
    a.set_title("(a) the first two modes\nmerge as $w_2$ rises", fontsize=10, loc="left")
    a.legend(fontsize=8.5, frameon=False, loc="center right")
    a.grid(alpha=0.25); a.set_ylim(0.95, plate * 1.08)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    # bottom middle/right: the two modes of the most-merged design.  Both are SYMMETRIC
    # about y=LY/2 (so a |phi| plot makes them look alike); the SIGNED vertical displacement
    # u_y tells them apart -- mode 1 is a whole-body bob (one sign), mode 2 has the body and
    # tip moving OUT OF PHASE (sign reversal).
    dm = data[int(np.argmin(gap))]

    def mode_uy(phi):
        u = phi[1::2]
        return u / np.abs(u).max()
    yv = dm["y"] > 0.5
    Dg = solid_img(dm["y"]); void = Dg < 0.5
    wmg = w2s[int(np.argmin(gap))]
    for k, (phik, lab, desc) in enumerate([(dm["phi"], "1st", "whole body in phase"),
                                           (dm["phi2"], "2nd", "body and tip out of phase")]):
        a = fig.add_subplot(gsb[0, k + 1])
        g = np.asarray(mtri.LinearTriInterpolator(tri, mode_uy(phik))(GX, GY).filled(0.0))
        # orient so each panel's dominant lobe reads positive (sign of an eigenvector is free)
        if np.nansum(g[~void]) < 0:
            g = -g
        a.imshow(np.ma.array(g, mask=void), origin="lower", extent=[0, LX, 0, LY],
                 cmap="RdBu_r", vmin=-1, vmax=1, interpolation="bilinear", zorder=2)
        holes(a, load=False)
        a.set_title(f"({'bc'[k]}) {lab} mode ($w_2{{=}}{wmg:g}$): {desc}\n"
                    f"signed $u_y$ (red up / blue down)", fontsize=9.5, loc="left")

    fig.suptitle("Mode-separation by designed anisotropy: maximizing $\\omega_1$ already "
                 f"pulls the equal-mass plate's gap $\\omega_2/\\omega_1={plate:.1f}$ down to "
                 f"{gap[0]:.2f}, and adding min-$\\omega_2$ closes it further to {gap.min():.2f} "
                 "--- the fiber design tunes the first two modes toward a degenerate pair",
                 fontsize=11.0, y=0.975)
    fig._gaps = [round(float(g), 2) for g in gap]
    return fig


def main():
    import argparse
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.025)
    ap.add_argument("--out", default="results/cfrp_bracket_mode2.png")
    args = ap.parse_known_args()[0]
    fig = build(args.h)
    fig.savefig(args.out, dpi=175, bbox_inches="tight", facecolor="white")
    print(f"gaps: {fig._gaps}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
