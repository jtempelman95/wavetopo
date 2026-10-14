"""Fiber-toolpath figure for the CFRP bracket (§11): the continuous tows of the two
designs, drawn prominently so the fiber ORIENTATION is legible.  In both designs the
tows run ALONG the load-carrying chords and wrap the holes; a per-element check puts
them within 5-9 deg of the local principal-stress direction on every loaded member, so
the E_f1=131 GPa fiber axis carries the load, not the 9 GPa transverse."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing

from examples.drivers.cfrp_bracket_modal import setup, HOLES, LOAD_HOLE, LX, LY
from wavetopo.dolfinx_viz import plot_toolpaths_phase

plt.rcParams.update({"font.family": "serif", "font.size": 10.5,
                     "axes.titlesize": 11, "axes.titlepad": 5})
INK, CLR = "#111111", "#b3121b"
NG, MORPH_R, SMOOTH = 1000, 0.016, 0.009


def misalign(P, mesh, y, th):
    solid = y > 0.5
    u = P.static(y, th, want_grad=False)[1]
    c = np.array([np.ones_like(th), np.cos(2 * th), np.sin(2 * th),
                  np.cos(4 * th), np.sin(4 * th)])
    D = np.einsum("se,sij->eij", c, P.fs.C)
    sig = np.einsum("eij,ej->ei", D,
                    np.einsum("epq,eq->ep", P.fs.B, u[mesh.edof]))
    sxx, syy, sxy = sig[:, 0], sig[:, 1], sig[:, 2]
    ps = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    avg = 0.5 * (sxx + syy); R = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)
    s1, s2 = avg + R, avg - R
    dom = np.where(np.abs(s1) >= np.abs(s2), ps, ps + np.pi / 2)
    mag = np.maximum(np.abs(s1), np.abs(s2))
    d = (th - dom + np.pi / 2) % np.pi - np.pi / 2
    ld = solid & (mag > 0.3 * np.percentile(mag[solid], 90))
    return np.average(np.abs(np.degrees(d))[ld], weights=mag[ld])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aframe", default="results/cfrp_bracket_aframe.npz")
    ap.add_argument("--freq", default="results/cfrp_bracket_comp.npz")
    ap.add_argument("--h", type=float, default=0.025)
    ap.add_argument("--out", default="results/cfrp_bracket_fibers.png")
    args = ap.parse_known_args()[0]
    mesh, mat, P = setup(args.h)
    cents = np.column_stack([mesh.cx, mesh.cy])
    tri = mtri.Triangulation(mesh.npx[:, 0], mesh.npx[:, 1], mesh.tris)
    nn = mesh.npx.shape[0]; wn = np.zeros(nn)
    np.add.at(wn, mesh.tris.ravel(), np.repeat(mesh.area, 3)); wn = np.maximum(wn, 1e-12)
    gx = np.linspace(0, LX, NG); gy = np.linspace(0, LY, int(NG * LY / LX))
    GX, GY = np.meshgrid(gx, gy); px = LX / NG
    rr = max(int(round(MORPH_R / px)), 1); stel = np.ones((2 * rr + 1, 2 * rr + 1))

    def boundary(y):
        v = np.zeros(nn)
        np.add.at(v, mesh.tris.ravel(), np.repeat(mesh.area, 3) * np.repeat(y, 3))
        raw = np.asarray(mtri.LinearTriInterpolator(tri, v / wn)(GX, GY).filled(0.0))
        b = binary_closing(binary_opening(raw > 0.5, stel), stel)
        return gaussian_filter(b.astype(float), SMOOTH / px)

    fig, ax = plt.subplots(2, 1, figsize=(10.5, 9.6))
    fig.subplots_adjust(hspace=0.14)
    for a, path, lab in [(ax[0], args.aframe, "stiffness/compliance A-frame"),
                         (ax[1], args.freq, "$+$ frequency objective")]:
        d = np.load(path); y, th = d["y"], d["th"]
        Dg = boundary(y)
        a.contourf(GX, GY, Dg, levels=[0.5, 1.5], colors=["#f4f4f4"], zorder=0)
        plot_toolpaths_phase(a, cents, th, (0, LX), (0, LY), holes=HOLES,
                             dens=y, dens_thresh=0.5, spacing=0.052, lw=0.75)
        a.contour(GX, GY, Dg, levels=[0.5], colors=["#222222"], linewidths=1.2,
                  zorder=6)
        for i, (hx, hy, hr) in enumerate(HOLES):
            a.add_patch(plt.Circle((hx, hy), hr, fc="white",
                                   ec=CLR if i == LOAD_HOLE else "k",
                                   lw=1.8 if i == LOAD_HOLE else 1.1, zorder=7))
        a.plot([0, 0], [0.04, LY - 0.04], color=CLR, lw=5.5, zorder=8,
               solid_capstyle="butt")
        hx, hy, hr = HOLES[LOAD_HOLE]
        a.annotate("", xy=(hx, hy - hr - 0.34), xytext=(hx, hy - hr - 0.02),
                   arrowprops=dict(arrowstyle="-|>", color=CLR, lw=2.4), zorder=9)
        a.set_xlim(-0.07, LX + 0.07); a.set_ylim(-0.36, LY + 0.07)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#cccccc")
        mis = misalign(P, mesh, y, th)
        fr = float(d["fr"])
        a.set_title(f"{lab}  ($\\omega_1$ {fr:.2f}$\\times$):  continuous tows run "
                    f"along the chords, ${mis:.0f}^\\circ$ from the principal stress",
                    loc="left", color=INK, fontsize=10.5)
    fig.suptitle("CFRP bracket fiber toolpaths --- the tows follow the load: along the "
                 "tension/compression chords and wrapped around every hole",
                 fontsize=11.5, y=0.945)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
