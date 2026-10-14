"""Showcase figure for the fully-resolved clean stiffness/compliance CFRP bracket
(the 'hero' A-frame): a genuine, fully load-bearing truss whose optimized fibers wrap
every hole and follow the load, 7.3x stiffer than an equal-mass plate and 44% stiffer
than any straight layup, at printable curvature."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing

from examples.drivers.cfrp_bracket_modal import setup, HOLES, LOAD_HOLE, LX, LY
from wavetopo.dolfinx_viz import plot_toolpaths_phase
from examples.figures._paperfig import set_style, harmonize_paper

SCALE = 0.838   # display scale in the manuscript (includegraphics width / pdf width)

CLR = "#b3121b"
NG, MORPH_R, SMOOTH = 1100, 0.012, 0.008


def build(data="results/cfrp_bracket_hero.npz", h=0.015):
    set_style()
    d = np.load(data); mesh, mat, P = setup(h)
    y, th, phi = d["y"], d["th"], d["phi"]; solid = y > 0.5
    tri = mtri.Triangulation(mesh.npx[:, 0], mesh.npx[:, 1], mesh.tris)
    cents = np.column_stack([mesh.cx, mesh.cy]); B, edof, A = P.fs.B, mesh.edof, mesh.area
    nn = mesh.npx.shape[0]; wn = np.zeros(nn)
    np.add.at(wn, mesh.tris.ravel(), np.repeat(A, 3)); wn = np.maximum(wn, 1e-12)
    gx = np.linspace(0, LX, NG); gy = np.linspace(0, LY, int(NG * LY / LX))
    GX, GY = np.meshgrid(gx, gy); px = LX / NG
    rr = max(int(round(MORPH_R / px)), 1); stel = np.ones((2 * rr + 1, 2 * rr + 1))

    def e2n(ef):
        v = np.zeros(nn); np.add.at(v, mesh.tris.ravel(), np.repeat(A, 3) * np.repeat(ef, 3))
        return v / wn

    def solid_img(yy):
        raw = np.asarray(mtri.LinearTriInterpolator(tri, e2n(yy))(GX, GY).filled(0.0))
        return gaussian_filter(binary_closing(binary_opening(raw > 0.5, stel), stel)
                               .astype(float), SMOOTH / px)
    Dg = solid_img(y); void = Dg < 0.5

    # signed principal stress (load path)
    u = P.static(y, th, want_grad=False)[1]
    c = np.array([np.ones_like(th), np.cos(2 * th), np.sin(2 * th),
                  np.cos(4 * th), np.sin(4 * th)])
    D = np.einsum("se,sij->eij", c, P.fs.C)
    sg = np.einsum("eij,ej->ei", D, np.einsum("epq,eq->ep", B, u[edof]))
    avg = 0.5 * (sg[:, 0] + sg[:, 1]); Rr = np.sqrt((0.5 * (sg[:, 0] - sg[:, 1])) ** 2 + sg[:, 2] ** 2)
    s1, s2 = avg + Rr, avg - Rr; signed = np.where(np.abs(s1) >= np.abs(s2), s1, s2)

    fig, ax = plt.subplots(2, 2, figsize=(10.2, 5.57))
    fig.subplots_adjust(hspace=0.14, wspace=0.02, top=0.92, bottom=0.03)

    def frame(a):
        for i, (hx, hy, hr) in enumerate(HOLES):
            a.add_patch(plt.Circle((hx, hy), hr, fc="white", ec=CLR if i == LOAD_HOLE else "k",
                                   lw=1.7 if i == LOAD_HOLE else 1.1, zorder=8))
        a.plot([0, 0], [0.04, LY - 0.04], color=CLR, lw=5, zorder=9, solid_capstyle="butt")
        hx, hy, hr = HOLES[LOAD_HOLE]
        a.annotate("", xy=(hx, hy - hr - 0.34), xytext=(hx, hy - hr - 0.02),
                   arrowprops=dict(arrowstyle="-|>", color=CLR, lw=2.3), zorder=9)
        a.set_xlim(-0.06, LX + 0.06); a.set_ylim(-0.36, LY + 0.06)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#cccccc")

    # (a) topology + fibers (fibers wrap the holes)
    a = ax[0, 0]
    a.contourf(GX, GY, Dg, levels=[0.5, 1.5], colors=["#efefef"], zorder=0)
    plot_toolpaths_phase(a, cents, th, (0, LX), (0, LY), holes=HOLES, dens=y,
                         dens_thresh=0.5, spacing=0.055, lw=0.5)
    a.contour(GX, GY, Dg, levels=[0.5], colors=["#222"], linewidths=0.9, zorder=6)
    frame(a); a.set_title("(a) A-frame and fiber tows", loc="left")

    # (b) load path (tension/compression) -- every member loaded, no hanging beams
    a = ax[0, 1]
    gimg = np.asarray(mtri.LinearTriInterpolator(tri, e2n(np.where(solid, signed, 0.0)))(GX, GY).filled(0.0))
    vv = np.percentile(np.abs(signed[solid]), 96)
    a.imshow(np.ma.array(gimg, mask=void), origin="lower", extent=[0, LX, 0, LY],
             cmap="RdBu_r", vmin=-vv, vmax=vv, interpolation="bilinear", zorder=2)
    frame(a); a.set_title("(b) load path (red tension, blue compression)", loc="left")

    # (c) fiber curvature |zeta|
    from wavetopo.cfrp_wave import curl_operators, director_curl
    Px, Py = curl_operators(mesh); zeta = director_curl(th, Px, Py)
    zref = 1.0 / min(hr for _, _, hr in HOLES)
    a = ax[1, 0]
    zg = np.asarray(mtri.LinearTriInterpolator(tri, e2n(np.minimum(zeta, zref)))(GX, GY).filled(0.0))
    im = a.imshow(np.ma.array(zg, mask=void), origin="lower", extent=[0, LX, 0, LY],
                  cmap="viridis", vmin=0, vmax=zref, interpolation="bilinear", zorder=2)
    fig.colorbar(im, ax=a, fraction=0.043, pad=0.02, label=r"$|\zeta|$")
    frame(a); a.set_title(r"(c) fiber curvature $|\zeta|$", loc="left")

    # (d) 1st mode
    a = ax[1, 1]
    mn = np.sqrt(phi[0::2] ** 2 + phi[1::2] ** 2); mn = mn / mn.max()
    mg = np.asarray(mtri.LinearTriInterpolator(tri, mn)(GX, GY).filled(0.0))
    a.imshow(np.ma.array(mg, mask=void), origin="lower", extent=[0, LX, 0, LY],
             cmap="magma", vmin=0, vmax=1, interpolation="bilinear", zorder=2)
    frame(a); a.set_title(r"(d) fundamental mode $\omega_1$", loc="left")

    return harmonize_paper(fig, SCALE)


def main():
    import argparse
    import os
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/cfrp_bracket_hero.npz")
    ap.add_argument("--h", type=float, default=0.015)
    ap.add_argument("--out", default="results/cfrp_bracket_hero.pdf")
    args = ap.parse_known_args()[0]
    fig = build(args.data, args.h)
    fig.savefig(args.out, bbox_inches="tight")
    pv = os.environ.get("FIGPREVIEW")
    if pv:
        fig.savefig(pv, dpi=110, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
