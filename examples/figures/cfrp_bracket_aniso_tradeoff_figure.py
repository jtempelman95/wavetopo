"""The anisotropy-versus-printability front for the bracket.  For each design in the
zeta_all sweep (all w_eig=0, same topology target, same mass), plot the compliance gain
over the best straight (0deg) layup against the fiber curvature actually used, |zeta|max.
As the printable curvature limit is relaxed the tows steer harder and the gain climbs
toward the pointwise-optimal asymptote (principal-stress alignment, |zeta|->inf).  Insets
show the tow field at a gentle, the built, and an aggressive point on the front."""
#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing

from examples.drivers.cfrp_bracket_modal import setup, HOLES, LOAD_HOLE, LX, LY
from wavetopo.cfrp_wave import curl_operators, director_curl
from wavetopo.dolfinx_viz import plot_toolpaths_phase
from examples.figures._paperfig import set_style, internal_ticks, harmonize_paper

SCALE = 0.889   # display scale in the manuscript (includegraphics width / pdf width)

ZAS = [2.0, 3.5, 4.8, 7.0, 10.0, 16.0]
NG, MORPH_R, SMOOTH = 620, 0.014, 0.009


def principal_opt(P, y, nt, iters=15):
    """Pointwise-optimal compliance gain: align stiff axis with larger-|sigma| principal."""
    B, edof, C = P.fs.B, P.mesh.edof, P.fs.C
    c0 = P.static(y, np.zeros(nt), want_grad=False)[0]; th = np.zeros(nt); cbest = c0
    for _ in range(iters):
        u = P.static(y, th, want_grad=False)[1]
        c = np.array([np.ones(nt), np.cos(2 * th), np.sin(2 * th),
                      np.cos(4 * th), np.sin(4 * th)])
        D = np.einsum("se,sij->eij", c, C)
        sg = np.einsum("eij,ej->ei", D, np.einsum("epq,eq->ep", B, u[edof]))
        sx, sy, sxy = sg[:, 0], sg[:, 1], sg[:, 2]
        avg = 0.5 * (sx + sy); R = np.sqrt((0.5 * (sx - sy)) ** 2 + sxy ** 2)
        ps1 = 0.5 * np.arctan2(2 * sxy, sx - sy)
        tgt = np.where(np.abs(avg + R) >= np.abs(avg - R), ps1, ps1 + np.pi / 2)
        s = 0.6 * np.sin(2 * th) + 0.4 * np.sin(2 * tgt)
        cc = 0.6 * np.cos(2 * th) + 0.4 * np.cos(2 * tgt)
        th = 0.5 * np.arctan2(s, cc); cbest = min(cbest, P.static(y, th, want_grad=False)[0])
    return 100 * (c0 - cbest) / c0


def build(h=0.02):
    set_style()
    mesh, mat, P = setup(h); nt = mesh.tris.shape[0]
    Px, Py = curl_operators(mesh)
    tri = mtri.Triangulation(mesh.npx[:, 0], mesh.npx[:, 1], mesh.tris)
    cents = np.column_stack([mesh.cx, mesh.cy]); npx = mesh.npx
    nn = npx.shape[0]; wn = np.zeros(nn); np.add.at(wn, mesh.tris.ravel(), np.repeat(mesh.area, 3))
    gx = np.linspace(0, LX, NG); gy = np.linspace(0, LY, int(NG * LY / LX))
    GX, GY = np.meshgrid(gx, gy); px = LX / NG
    rr = max(int(round(MORPH_R / px)), 1); stel = np.ones((2 * rr + 1, 2 * rr + 1))

    def solid_img(y):
        v = np.zeros(nn); np.add.at(v, mesh.tris.ravel(), np.repeat(mesh.area, 3) * np.repeat(y, 3))
        raw = np.asarray(mtri.LinearTriInterpolator(tri, v / np.maximum(wn, 1e-12))(GX, GY).filled(0))
        return gaussian_filter(binary_closing(binary_opening(raw > 0.5, stel), stel).astype(float),
                               SMOOTH / px)

    zmax, gain, steer, data = [], [], [], {}
    pw = None
    for za in ZAS:
        f = f"results/cfrp_bracket_za{za}.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f); y, th = d["y"], d["th"]; solid = y > 0.5
        c0 = P.static(y, np.zeros(nt), want_grad=False)[0]
        c1 = P.static(y, th, want_grad=False)[0]
        zmax.append(float(director_curl(th, Px, Py)[solid].max()))
        gain.append(100 * (c0 - c1) / c0)
        steer.append(np.mean(np.abs(np.degrees(((th[solid] + np.pi / 2) % np.pi) - np.pi / 2))))
        data[za] = (y, th, solid)
        if za == 4.8:
            pw = principal_opt(P, y, nt)
    zmax, gain, steer = np.array(zmax), np.array(gain), np.array(steer)
    if pw is None and data:
        pw = principal_opt(P, list(data.values())[0][0], nt)

    fig = plt.figure(figsize=(6.8, 3.22))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.16,
                          left=0.07, right=0.98, top=0.9, bottom=0.11)
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(pw, ls="--", color="#c0392b", lw=1.8,
               label=f"pointwise optimum ${{\\sim}}{pw:.0f}\\%$ ($|\\zeta|\\to\\infty$)")
    ax.plot(zmax, gain, "-o", color="#1f6fb2", lw=2.4, ms=8, label="printable optimized tows")
    # mark the built design (za=4.8)
    if 4.8 in data:
        i48 = ZAS.index(4.8) if 4.8 in [z for z in ZAS if z in data] else None
    for i, za in enumerate([z for z in ZAS if z in data]):
        ax.annotate(f"$1/R{{=}}{zmax[i]:.1f}$", (zmax[i], gain[i]),
                    xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8,
                    color="#333")
    # the FEATURED dramatic hero (za=9, fine mesh): a distinct point on the front
    ax.scatter([9.0], [57], marker="*", s=560, facecolor="#f5c518", edgecolor="#1a7a3a",
               linewidth=1.8, zorder=7)
    ax.annotate("featured A-frame\n($|\\zeta|{\\leq}9$, 57%, $18^\\circ$)", (9.0, 57),
                xytext=(9.4, 46), fontsize=9.5, color="#1a7a3a",
                arrowprops=dict(arrowstyle="->", color="#1a7a3a"))
    ax.set_xlabel(r"fiber curvature used  $|\zeta|_{\max}=1/R$  (tighter turns $\rightarrow$)")
    ax.set_ylabel("compliance gain over best straight layup (%)")
    ax.set_title("(a)", fontsize=11, loc="left")
    ax.legend(fontsize=9.5, loc="lower right"); ax.grid(alpha=0.22)
    ax.set_ylim(0, pw * 1.1)
    internal_ticks(ax)

    # right: three tow-field insets (gentle, built, aggressive)
    picks = [z for z in [2.0, 4.8, 16.0] if z in data]
    gi = fig.add_gridspec(len(picks), 1, left=0.69, right=0.99, top=0.9, bottom=0.08, hspace=0.22)
    labels = {2.0: "gentle", 4.8: "conservative", 16.0: "aggressive"}
    for r, za in enumerate(picks):
        a = fig.add_subplot(gi[r, 0]); y, th, solid = data[za]
        Dg = solid_img(y)
        a.contourf(GX, GY, Dg, levels=[0.5, 1.5], colors=["#ececec"], zorder=0)
        plot_toolpaths_phase(a, cents, th, (0, LX), (0, LY), holes=HOLES, dens=y,
                             dens_thresh=0.5, spacing=0.075, lw=0.4)
        a.contour(GX, GY, Dg, levels=[0.5], colors=["#333"], linewidths=1.0, zorder=6)
        for j, (hx, hy, hr) in enumerate(HOLES):
            a.add_patch(plt.Circle((hx, hy), hr, fc="white",
                                   ec="#b3121b" if j == LOAD_HOLE else "k", lw=1.0, zorder=7))
        a.plot([0, 0], [0.05, LY - 0.05], color="#b3121b", lw=2.5, zorder=8)
        a.set_xlim(-0.05, LX + 0.05); a.set_ylim(-0.05, LY + 0.05); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        di = [z for z in ZAS if z in data].index(za)
        gk, sk = gain[di], steer[di]
        a.set_title(f"{labels[za]} ($|\\zeta|{{\\leq}}{za:g}$)", fontsize=10, pad=1)

    fig._diag = dict(za=[z for z in ZAS if z in data],
                     zmax=[round(float(z), 1) for z in zmax],
                     gain=[round(float(g)) for g in gain],
                     steer=[round(float(s), 1) for s in steer],
                     pointwise=round(pw))
    return harmonize_paper(fig, SCALE)


def main():
    import argparse
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.02)
    ap.add_argument("--out", default="results/cfrp_bracket_aniso_tradeoff.pdf")
    args = ap.parse_args()
    fig = build(args.h)
    fig.savefig(args.out, bbox_inches="tight")
    pv = os.environ.get("FIGPREVIEW")
    if pv:
        fig.savefig(pv, dpi=110, bbox_inches="tight")
    print("diag:", fig._diag)
    print("saved", args.out)


if __name__ == "__main__":
    main()
