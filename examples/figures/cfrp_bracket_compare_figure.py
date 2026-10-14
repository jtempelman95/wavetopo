"""Publication figure for the CFRP bracket, §11: a STIFFNESS-vs-FREQUENCY comparison
at fixed mass.  Top row is a stiffness/compliance co-design (a clean, fully load-bearing
A-frame whose first-mode gain is PROPORTIONATE, omega_1^2 ~ stiffness).  Bottom row adds
a frequency objective: the optimizer thins the chords and inserts a lightly-loaded inner
"modal stiffener", buying a DISPROPORTIONATE omega_1 (omega_1^2 rises ~3x faster than
stiffness).  The right panel makes the disproportionality explicit.  Both designs use
the same mesh, mass, holes and equal-mass baseline; every field is rendered on a fine
grid with a morphological boundary clean-up."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing

from examples.drivers.cfrp_bracket_modal import setup, HOLES, LOAD_HOLE, LX, LY
from wavetopo.dolfinx_viz import plot_toolpaths_phase

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.titlesize": 10, "axes.titlepad": 4})
INK, MUT, CLR = "#111111", "#6b6b6b", "#b3121b"
TEN, COM = "#b3121b", "#1f6fb2"          # tension red, compression blue
NG, MORPH_R, SMOOTH = 1000, 0.016, 0.009


def node_mag(phi):
    return np.sqrt(phi[0::2] ** 2 + phi[1::2] ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aframe", default="results/cfrp_bracket_aframe.npz")
    ap.add_argument("--freq", default="results/cfrp_bracket_comp.npz")
    ap.add_argument("--h", type=float, default=0.025)
    ap.add_argument("--out", default="results/cfrp_bracket_modal.png")
    args = ap.parse_known_args()[0]

    mesh, mat, P = setup(args.h)
    tri = mtri.Triangulation(mesh.npx[:, 0], mesh.npx[:, 1], mesh.tris)
    cents = np.column_stack([mesh.cx, mesh.cy])
    B, edof, A = P.fs.B, mesh.edof, mesh.area
    nn = mesh.npx.shape[0]; wn = np.zeros(nn)
    np.add.at(wn, mesh.tris.ravel(), np.repeat(A, 3)); wn = np.maximum(wn, 1e-12)
    gx = np.linspace(0, LX, NG); gy = np.linspace(0, LY, int(NG * LY / LX))
    GX, GY = np.meshgrid(gx, gy); px = LX / NG
    rr = max(int(round(MORPH_R / px)), 1); stel = np.ones((2 * rr + 1, 2 * rr + 1))

    def e2n(ef):
        v = np.zeros(nn); np.add.at(v, mesh.tris.ravel(),
                                    np.repeat(A, 3) * np.repeat(ef, 3))
        return v / wn

    def solid_img(y):
        raw = np.asarray(mtri.LinearTriInterpolator(tri, e2n(y))(GX, GY).filled(0.0))
        b = binary_closing(binary_opening(raw > 0.5, stel), stel)
        return gaussian_filter(b.astype(float), SMOOTH / px)

    def signed_principal(y, th):
        u = P.static(y, th, want_grad=False)[1]
        c = np.array([np.ones_like(th), np.cos(2 * th), np.sin(2 * th),
                      np.cos(4 * th), np.sin(4 * th)])
        D = np.einsum("se,sij->eij", c, P.fs.C)
        eps = np.einsum("epq,eq->ep", B, u[edof]); sig = np.einsum("eij,ej->ei", D, eps)
        avg = 0.5 * (sig[:, 0] + sig[:, 1])
        R = np.sqrt((0.5 * (sig[:, 0] - sig[:, 1])) ** 2 + sig[:, 2] ** 2)
        s1, s2 = avg + R, avg - R
        return np.where(np.abs(s1) >= np.abs(s2), s1, s2)

    fig = plt.figure(figsize=(17.5, 8.0))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.82],
                  hspace=0.16, wspace=0.13, left=0.008, right=0.99,
                  top=0.9, bottom=0.03)

    def frame(a, load=True):
        for i, (hx, hy, hr) in enumerate(HOLES):
            ld = load and i == LOAD_HOLE
            a.add_patch(plt.Circle((hx, hy), hr, fc="white", ec=CLR if ld else "k",
                                   lw=1.6 if ld else 1.0, zorder=7))
        a.plot([0, 0], [0.04, LY - 0.04], color=CLR, lw=4.5, zorder=8,
               solid_capstyle="butt")
        if load:
            hx, hy, hr = HOLES[LOAD_HOLE]
            a.annotate("", xy=(hx, hy - hr - 0.32), xytext=(hx, hy - hr - 0.02),
                       arrowprops=dict(arrowstyle="-|>", color=CLR, lw=2), zorder=9)
        a.set_xlim(-0.06, LX + 0.06); a.set_ylim(-0.34, LY + 0.06)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#cccccc")

    def title(a, s, sz=9.8):
        a.set_title(s, loc="left", color=INK, fontsize=sz)

    def row(r, d, tag, headline):
        y, th, phi = d["y"], d["th"], d["phi"]; solid = y > 0.5
        Dg = solid_img(y); void = Dg < 0.5
        fr, cr, sr = float(d["fr"]), float(d["c_ratio"]), float(d["s_ratio"])
        # topology + fiber tows
        a = fig.add_subplot(gs[r, 0])
        a.contourf(GX, GY, Dg, levels=[0.5, 1.5], colors=["#eaeaea"], zorder=1)
        plot_toolpaths_phase(a, cents, th, (0, LX), (0, LY), holes=HOLES,
                             dens=y, dens_thresh=0.5, spacing=0.075, lw=0.5)
        a.contour(GX, GY, Dg, levels=[0.5], colors=["#333333"], linewidths=0.9,
                  zorder=6)
        frame(a); title(a, f"({tag[0]}) {headline}")
        # tension / compression load path
        a = fig.add_subplot(gs[r, 1])
        sg = signed_principal(y, th); gimg = np.asarray(
            mtri.LinearTriInterpolator(tri, e2n(np.where(solid, sg, 0.0)))(GX, GY)
            .filled(0.0))
        vv = np.percentile(np.abs(sg[solid]), 96)
        im = a.imshow(np.ma.array(gimg, mask=void), origin="lower",
                      extent=[0, LX, 0, LY], cmap="RdBu_r", vmin=-vv, vmax=vv,
                      interpolation="bilinear", zorder=2)
        frame(a); title(a, f"({tag[1]}) load path: tension / compression")
        # mode
        a = fig.add_subplot(gs[r, 2])
        mn = node_mag(phi); mn = mn / mn.max()
        gm = np.asarray(mtri.LinearTriInterpolator(tri, mn)(GX, GY).filled(0.0))
        a.imshow(np.ma.array(gm, mask=void), origin="lower", extent=[0, LX, 0, LY],
                 cmap="magma", vmin=0, vmax=1, interpolation="bilinear", zorder=2)
        frame(a); title(a, f"({tag[2]}) 1st mode  |  $\\omega_1$={fr:.2f}$\\times$ "
                           f"equal-mass plate")
        return fr, cr, sr

    fa, ca, sa = row(0, np.load(args.aframe), "abc",
                     "stiffness/compliance A-frame")
    ff, cf, sf = row(1, np.load(args.freq), "efg", "$+$ frequency objective")

    # disproportionality panel (spans both rows)
    a = fig.add_subplot(gs[:, 3])
    labels = ["A-frame\n(stiffness)", "$+$frequency"]
    w2 = [fa ** 2, ff ** 2]; kk = [1 / ca, 1 / cf]
    x = np.arange(2); w = 0.38
    a.bar(x - w / 2, w2, w, color="#6a2c91", label="$\\omega_1^2$ gain")
    a.bar(x + w / 2, kk, w, color="#1f6fb2", label="stiffness gain $c_0/c$")
    for xi, (a2, k) in enumerate(zip(w2, kk)):
        a.text(xi - w / 2, a2 + 0.3, f"{a2:.0f}$\\times$", ha="center", fontsize=9,
               color="#6a2c91")
        a.text(xi + w / 2, k + 0.3, f"{k:.1f}$\\times$", ha="center", fontsize=9,
               color="#1f6fb2")
    a.set_xticks(x)
    a.set_xticklabels(["A-frame\n(proportionate)", "$+$frequency\n(disproportionate)"],
                      fontsize=9)
    a.get_xticklabels()[1].set_color("#6a2c91")
    a.set_ylim(0, max(w2) * 1.16); a.legend(fontsize=8.5, loc="upper center",
                                            frameon=False)
    a.set_title("the disproportionate lever:\n$\\omega_1^2$ vs stiffness (equal mass)",
                loc="left", fontsize=9.5, color=INK)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(alpha=0.25, axis="y")

    fig.suptitle("CFRP bracket, stiffness vs frequency at equal mass: a stiffness/"
                 "compliance co-design gives a clean, fully load-bearing A-frame "
                 f"($\\omega_1$ {fa:.2f}$\\times$, proportionate) --- adding a frequency "
                 f"objective inserts a lightly-loaded modal stiffener that buys a "
                 f"disproportionate $\\omega_1$ {ff:.2f}$\\times$ "
                 f"($\\omega_1^2$ {ff**2:.0f}$\\times$ vs {1/cf:.1f}$\\times$ stiffness)",
                 fontsize=10.8, y=0.975)
    fig.savefig(args.out, dpi=185, bbox_inches="tight", facecolor="white")
    print(f"A-frame: w1 {fa:.2f}x  w1^2 {fa**2:.1f}  stiff {1/ca:.1f}  stress {sa:.2f}")
    print(f"freq:    w1 {ff:.2f}x  w1^2 {ff**2:.1f}  stiff {1/cf:.1f}  stress {sf:.2f}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
