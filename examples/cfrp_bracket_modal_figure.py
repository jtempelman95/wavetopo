"""Publication figure for examples/cfrp_bracket_modal.py -- topology + toolpath
co-design of a CFRP bracket that MAXIMIZES the first natural frequency omega_1 and
MINIMIZES the peak (p-norm von Mises) stress at a fixed mass budget.

The figure is a DIRECT, equal-mass comparison: the bottom row places the optimized
design next to a uniform-density plate of identical mass (same holes, same solid
mounting rings, straight fibers) under the same load, on shared colour scales, so the
frequency, stress and compliance gains are read off the fields, not just asserted.

All fields are resampled onto a fine grid and the solid boundary is the smooth 0.5
density iso-contour (no triangle-facet staircase)."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing

from examples.cfrp_bracket_modal import setup, HOLES, LOAD_HOLE, LX, LY
from wavetopo.cfrp_wave import curl_operators, director_curl
from wavetopo.dolfinx_viz import plot_toolpaths_phase

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.titlesize": 10.5, "axes.titlepad": 5})
INK, MUT = "#111111", "#6b6b6b"
CLR = "#b3121b"
NG = 1200                                           # grid resolution along x
MORPH_R = 0.016                                     # morphological open/close radius
SMOOTH = 0.009                                      # boundary smoothing length


def node_mag(phi):
    return np.sqrt(phi[0::2] ** 2 + phi[1::2] ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/cfrp_bracket_fine.npz")
    ap.add_argument("--h", type=float, default=0.025)
    ap.add_argument("--out", default="results/cfrp_bracket_modal.png")
    args = ap.parse_args()

    d = np.load(args.data)
    mesh, mat, P = setup(args.h)
    tri = mtri.Triangulation(mesh.npx[:, 0], mesh.npx[:, 1], mesh.tris)
    cents = np.column_stack([mesh.cx, mesh.cy])
    y, th, phi, svm = d["y"], d["th"], d["phi"], d["svm"]
    phi0, svm0 = d["phi0"], d["svm0"]
    solid = y > 0.5
    w1o = float(np.sqrt(d["lam"][0])); w1b = w1o / float(d["fr"])

    # --- element -> node (area weighted) so every field can be grid-interpolated ---
    nn = mesh.npx.shape[0]; wn = np.zeros(nn)
    np.add.at(wn, mesh.tris.ravel(), np.repeat(mesh.area, 3))
    wn = np.maximum(wn, 1e-12)

    def e2n(ef):
        v = np.zeros(nn)
        np.add.at(v, mesh.tris.ravel(), np.repeat(mesh.area, 3) * np.repeat(ef, 3))
        return v / wn
    ynode = e2n(y)

    # --- fine regular grid; LinearTriInterpolator masks holes + exterior for free ---
    gx = np.linspace(0, LX, NG); gy = np.linspace(0, LY, int(NG * LY / LX))
    GX, GY = np.meshgrid(gx, gy)

    def togrid(nodal, smooth=0.0):
        gi = mtri.LinearTriInterpolator(tri, nodal)(GX, GY)   # masked array
        if smooth > 0:
            filled = gaussian_filter(gi.filled(0.0), smooth)
            wt = gaussian_filter((~gi.mask).astype(float), smooth)
            out = np.ma.array(filled / np.maximum(wt, 1e-6), mask=gi.mask)
            return out
        return gi

    # smooth solid region: threshold -> morphological open/close (removes whiskers &
    # fills notches below the print scale) -> Gaussian smooth of the boundary.  This
    # is the same 0.5-density level set, de-noised, so the panels read as clean parts.
    px = LX / NG; rr = max(int(round(MORPH_R / px)), 1)
    st = np.ones((2 * rr + 1, 2 * rr + 1))
    Draw = np.asarray(mtri.LinearTriInterpolator(tri, ynode)(GX, GY).filled(0.0))
    Db = binary_closing(binary_opening(Draw > 0.5, st), st)
    Dg = gaussian_filter(Db.astype(float), SMOOTH / px)       # smooth solid indicator
    void = Dg < 0.5                                           # grid void mask

    fig, ax = plt.subplots(2, 4, figsize=(19.5, 7.4))
    fig.subplots_adjust(hspace=0.17, wspace=0.16, left=0.012, right=0.988,
                        top=0.9, bottom=0.04)

    def frame(a, load=True, mounts=True):
        for i, (hx, hy, hr) in enumerate(HOLES):
            ld = load and i == LOAD_HOLE
            a.add_patch(plt.Circle((hx, hy), hr, fc="white", ec=CLR if ld else "k",
                                   lw=1.7 if ld else 1.0, zorder=7))
        a.plot([0, 0], [0.04, LY - 0.04], color=CLR, lw=5, zorder=8,
               solid_capstyle="butt")
        if load:
            hx, hy, hr = HOLES[LOAD_HOLE]
            a.annotate("", xy=(hx, hy - hr - 0.34), xytext=(hx, hy - hr - 0.02),
                       arrowprops=dict(arrowstyle="-|>", color=CLR, lw=2.2), zorder=9)
        a.set_xlim(-0.06, LX + 0.06); a.set_ylim(-0.36, LY + 0.06)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#cccccc")

    def cbar(a, im, lab):
        cb = fig.colorbar(im, ax=a, fraction=0.043, pad=0.02)
        cb.set_label(lab, fontsize=9); cb.ax.tick_params(labelsize=8)
        return cb

    def title(a, s, size=10):
        a.set_title(s, loc="left", color=INK, fontsize=size)

    def field(a, gimg, cmap, vmax, lab, vmin=0.0):
        im = a.imshow(gimg, origin="lower", extent=[0, LX, 0, LY], cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation="bilinear", zorder=2)
        cbar(a, im, lab)
        return im

    # ============================ ROW 1 : the design ============================
    # (a) topology -- smooth solid via the 0.5 iso-contour of the grid density
    a = ax[0, 0]
    a.contourf(GX, GY, Dg, levels=[0.5, 1.5], colors=["#161616"], zorder=2)
    a.contour(GX, GY, Dg, levels=[0.5], colors=["#161616"], linewidths=0.7, zorder=3)
    frame(a)
    title(a, f"(a) topology  |  {float(d['mass']):.2f}$\\times$ solid-plate mass")

    # (b) continuous fiber toolpaths on the structure
    a = ax[0, 1]
    plot_toolpaths_phase(a, cents, th, (0, LX), (0, LY), holes=HOLES,
                         dens=y, dens_thresh=0.5, spacing=0.09, lw=0.55)
    frame(a)
    title(a, "(b) fiber toolpaths (continuous tows)")

    # (c) fiber-curvature map (printability) -- masked to the solid
    a = ax[0, 2]
    Px, Py = curl_operators(mesh)
    zeta = director_curl(th, Px, Py); zref = 1.0 / min(hr for _, _, hr in HOLES)
    zg = togrid(e2n(np.minimum(zeta, zref)))
    zg = np.ma.array(zg, mask=np.ma.getmaskarray(zg) | void)
    field(a, zg, "viridis", zref, "$|\\zeta|$")
    frame(a, load=False)
    title(a, f"(c) fiber curvature $|\\zeta|$  |  max {zeta[solid].max():.1f} "
             f"$<$ 1/$R$={zref:.1f}")

    # (d) convergence
    a = ax[0, 3]
    a.plot(d["hist_fr"], color="#6a2c91", lw=2.3, label="$\\omega_1/\\omega_1^{\\rm base}$")
    a.plot(d["hist_s"], color="#c0392b", lw=2.0, label="peak stress / base")
    a.plot(d["hist_c"], color="#1f6fb2", lw=2.0, label="compliance $c/c_0$")
    a.plot(d["hist_mass"], color="#c9721f", lw=2.0, label="mass / solid")
    a.axhline(1.0, color=MUT, ls=":", lw=1)
    a.set_xlabel("iteration", fontsize=9.5)
    a.legend(fontsize=8.5, loc="center right", frameon=False)
    a.set_xlim(0, len(d["hist_fr"])); a.set_ylim(0, None); a.margins(y=0.05)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(alpha=0.25)
    title(a, f"(d) convergence: $\\omega_1$ up {float(d['fr']):.2f}$\\times$ at "
             f"fixed mass", size=9.8)

    # ================= ROW 2 : equal-mass baseline vs optimized =================
    # shared scales so the eye reads the gain directly
    svmax = float(np.percentile(svm0, 99.0))
    m_o = node_mag(phi); m_b = node_mag(phi0)
    m_o = m_o / m_o.max(); m_b = m_b / m_b.max()             # unit mode shapes

    # (e) baseline stress (uniform equal-mass plate: material everywhere, no void mask)
    a = ax[1, 0]
    field(a, togrid(e2n(svm0)), "inferno", svmax, "$\\sigma_{\\mathrm{vM}}$")
    frame(a)
    title(a, f"(e) baseline plate $\\sigma_{{\\rm vM}}$  |  uniform, equal mass")

    # (f) optimized stress on the SAME scale -- masked to the truss
    a = ax[1, 1]
    sg = togrid(e2n(svm)); sg = np.ma.array(sg, mask=np.ma.getmaskarray(sg) | void)
    field(a, sg, "inferno", svmax, "$\\sigma_{\\mathrm{vM}}$")
    frame(a)
    title(a, f"(f) optimized $\\sigma_{{\\rm vM}}$  |  peak {float(d['s_ratio']):.2f}"
             f"$\\times$ baseline")

    # (g) baseline mode
    a = ax[1, 2]
    field(a, togrid(m_b), "magma", 1.0, "$|\\phi_1|$")
    frame(a)
    title(a, f"(g) baseline mode  |  $\\omega_1$={w1b:.2f}")

    # (h) optimized mode on the SAME scale -- masked to the truss
    a = ax[1, 3]
    mg = togrid(m_o); mg = np.ma.array(mg, mask=np.ma.getmaskarray(mg) | void)
    field(a, mg, "magma", 1.0, "$|\\phi_1|$")
    frame(a)
    title(a, f"(h) optimized mode  |  $\\omega_1$={w1o:.2f} "
             f"({float(d['fr']):.2f}$\\times$)")

    fig.suptitle("Topology $+$ toolpath co-design of a CFRP bracket: maximize the "
                 "fundamental frequency AND minimize peak stress at a fixed mass "
                 f"--- vs an equal-mass uniform plate, $\\omega_1$ rises "
                 f"{float(d['fr']):.2f}$\\times$ ({100*(float(d['fr'])-1):.0f}%) while "
                 f"peak stress falls to {float(d['s_ratio']):.2f}$\\times$ and "
                 f"compliance to {float(d['c_ratio']):.2f}$\\times$", fontsize=11.5,
                 y=0.985)
    fig.savefig(args.out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"omega_1 {w1b:.3f} -> {w1o:.3f} ({float(d['fr']):.2f}x); "
          f"stress {float(d['s_ratio']):.2f}x; c {float(d['c_ratio']):.2f}x; "
          f"mass {float(d['mass']):.2f} vs base {float(d['mass0']):.2f}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
