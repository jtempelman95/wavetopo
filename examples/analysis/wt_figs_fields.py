r"""Vector (pgf/LaTeX) field figures of the wavetopo manuscript, from saved results:
wt_lens.pdf, wt_cloak.pdf, wt_panel.pdf.  Field maps are rasterized inside the vector PDF,
carry a labelled colourbar, and voids are drawn as clean filled discs.

    python examples/analysis/wt_figs_fields.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_NLTOPT = _os.environ.get("NLTOPT_DIR", _os.path.join(_os.path.dirname(_ROOT), "NLTopt"))
import os
import sys
import numpy as np

sys.path.insert(0, (_NLTOPT + "/examples"))
sys.path.insert(0, (_ROOT + ""))
import paperstyle as ps
ps.use_palatino()                       # wavetopo manuscript is set in Palatino
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from wavetopo.dolfinx_viz import plot_toolpaths_phase

plt.rcParams["contour.negative_linestyle"] = "solid"
R = (_ROOT + "/results")
FIG = (_ROOT + "/docs/paper/figs")
CM = "magma"
FS = 7.5
TGT = "#22b8cf"                                  # target / design-shell rings, all figures


def tri(x, y, t):
    return mtri.Triangulation(x, y, t)


def frame(ax, extent, title):
    ax.set_aspect("equal"); ax.set_xlim(*extent[:2]); ax.set_ylim(*extent[2:])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=FS, pad=3)
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def field(ax, T, m, vmax, title, extent):
    im = ax.tripcolor(T, m, cmap=CM, vmin=0, vmax=vmax, shading="gouraud", rasterized=True)
    frame(ax, extent, title)
    return im


def hole(ax, c, r):
    """Clean void: a filled white disc slightly larger than the inscribed mesh polygon."""
    ax.add_patch(Circle(c, r * 1.01, fc="white", ec="0.35", lw=0.6, zorder=4))


def ring(ax, c, r, color, ls="--", lw=0.9, halo=False):
    t = np.linspace(0, 2 * np.pi, 160)
    x, y = c[0] + r * np.cos(t), c[1] + r * np.sin(t)
    if halo:                                   # solid white underlay for contrast on the dark field
        ax.plot(x, y, color="white", ls="-", lw=lw + 0.9, alpha=0.9, zorder=4.9)
    ax.plot(x, y, color=color, ls=ls, lw=lw, zorder=5)


def square(ax, c, half, color, ls="--", lw=0.9, halo=False):
    """Box region c +/- half (lens objective; panel payload and sensor)."""
    from matplotlib.patches import Rectangle
    if halo:
        ax.add_patch(Rectangle((c[0] - half, c[1] - half), 2 * half, 2 * half, fill=False, ec="white",
                               ls="-", lw=lw + 0.9, alpha=0.9, zorder=4.9))
    ax.add_patch(Rectangle((c[0] - half, c[1] - half), 2 * half, 2 * half, fill=False, ec=color, ls=ls, lw=lw, zorder=5))


def cbar(fig, im, axes, label, extend="neither"):
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015, extend=extend)
    cb.ax.tick_params(labelsize=FS); cb.set_label(label, fontsize=FS)
    return cb


# ---------------------------------------------------------------- lens
d = np.load(os.path.join(R, "dolfinx_lens_data.npz"))
T = tri(d["trix"], d["triy"], d["tris"]); ext = (0, 4, 0, 3)
tgt = ((0.78 * 4, 1.5), 0.18)                    # objective box (0.78 Lx, Ly/2) +/- 0.18
fig = plt.figure(figsize=(ps.FULL, 1.95))
gl = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.045, 0.16, 1], wspace=0.06, left=0.01, right=0.99, bottom=0.03, top=0.88)
ax = [fig.add_subplot(gl[0, 0]), fig.add_subplot(gl[0, 1]), fig.add_subplot(gl[0, 4])]
cax_l = fig.add_subplot(gl[0, 2])
for k, (m, t) in enumerate(((d["m0"], r"{\fontsize{9}{11}\selectfont (\textbf{a})}\ straight fibers"), (d["m1"], r"{\fontsize{9}{11}\selectfont (\textbf{b})}\ steered lens"))):
    im = field(ax[k], T, m / m.max(), 1.0, t, ext)        # each map normalised by its own maximum
    square(ax[k], *tgt, color=TGT, halo=True)
plot_toolpaths_phase(ax[2], d["cent"], d["thopt"], (0, 4), (0, 3), spacing=0.16, mode="line",
                     color=ps.INK, lw=0.4)
square(ax[2], *tgt, color=TGT)
frame(ax[2], ext, r"{\fontsize{9}{11}\selectfont (\textbf{c})}\ tow paths")
cb = fig.colorbar(im, cax=cax_l); cb.ax.tick_params(labelsize=FS)
cb.set_label(r"$|\mathbf{u}|/\max|\mathbf{u}|$ (per panel)", fontsize=FS)
pos = cax_l.get_position(); cax_l.set_position([pos.x0, pos.y0 + 0.06, pos.width * 0.6, pos.height - 0.12])
fig.savefig(os.path.join(FIG, "wt_lens.pdf")); print("wrote wt_lens.pdf")

# ---------------------------------------------------------------- cloak
h = np.load(os.path.join(R, "cloak_headtohead.npz"))
Th = tri(h["trix"], h["triy"], h["tris"]); Tf = tri(h["trfx"], h["trfy"], h["trfs"])
CX, CY, RV, RD = float(h["CX"]), float(h["CY"]), float(h["RV"]), float(h["R_DES"])
ext = (0.6, 7.4, 0.6, 4.4)
vmax = float(np.max(h["uref"]))                          # the reference sets the common scale
J0 = float(h["J0"])
panels = [(Tf, h["uref"], r"{\fontsize{9}{11}\selectfont (\textbf{a})}\ void-free reference"),
          (Th, h["ustr"], r"{\fontsize{9}{11}\selectfont (\textbf{b})}\ void, straight fibers"),
          (Th, h["uunc"], rf"{{\fontsize{{9}}{{11}}\selectfont (\textbf{{c}})}}\ optimized shell, ${J0/float(h['Junc']):.0f}\times$"),
          (Th, h["uan"], rf"{{\fontsize{{9}}{{11}}\selectfont (\textbf{{d}})}}\ transformation directions, ${J0/float(h['Jan']):.1f}\times$"),
          (Th, h["uopt"], rf"{{\fontsize{{9}}{{11}}\selectfont (\textbf{{e}})}}\ curvature-limited, ${J0/float(h['Jopt']):.1f}\times$")]
fig, axs = plt.subplots(2, 3, figsize=(ps.FULL, 3.15))
for k, (Tk, m, t) in enumerate(panels):
    a = axs.ravel()[k]
    im = field(a, Tk, m, vmax, t, ext)
    if k > 0:
        hole(a, (CX, CY), RV)
    if k >= 2:
        ring(a, (CX, CY), RD, color=TGT, halo=True)
a = axs.ravel()[5]
plot_toolpaths_phase(a, h["cent"], h["th_unc"], (0, 8), (0, 5), holes=[(CX, CY, RV)],
                     spacing=0.16, mode="line", color=ps.INK, lw=0.4, n=900)
ring(a, (CX, CY), RD, color=TGT)
hw = 0.5 * (ext[1] - ext[0]) * 0.92; hh = hw * (ext[3] - ext[2]) / (ext[1] - ext[0]) - 0.03   # same aspect as (a)-(e)
a.set_aspect("equal"); a.set_xlim(CX - hw, CX + hw); a.set_ylim(CY - hh, CY + hh)
a.set_xticks([]); a.set_yticks([]); a.set_title(r"{\fontsize{9}{11}\selectfont (\textbf{f})}\ tow paths of (c) around the void", loc="left", fontsize=FS, pad=3)
fig.subplots_adjust(left=0.01, right=0.885, bottom=0.02, top=0.93, wspace=0.05, hspace=0.22)
cax = fig.add_axes([0.905, 0.12, 0.014, 0.74])
cb = fig.colorbar(im, cax=cax, extend="max"); cb.ax.tick_params(labelsize=FS)
cb.set_label(r"$|\mathbf{u}|$ (common scale, clipped at max of (a))", fontsize=FS)
fig.savefig(os.path.join(FIG, "wt_cloak.pdf")); print("wrote wt_cloak.pdf")

# ---------------------------------------------------------------- panel
p = np.load(os.path.join(R, "dolfinx_panel_band_repro_data.npz"))
T = tri(p["trix"], p["triy"], p["tris"]); Lx, Ly = float(p["Lx"]), float(p["Ly"])
holes = [tuple(x) for x in p["holes"]]
C_SEN, C_PAY, C_RIM = ps.STEEL, ps.GREEN, ps.RED      # same palette as Figs. 2-4
W, mw = ps.FULL, 2.75
mh = mw * Ly / Lx
yb, yt = 0.78, 0.78 + mh + 0.36
H = yt + mh + 0.22
fig = plt.figure(figsize=(W, H))
A = lambda x0, y0, w, h: fig.add_axes([x0 / W, y0 / H, w / W, h / H])
ax = [A(0.05, yt, mw, mh), A(0.05 + mw + 0.12, yt, mw, mh), A(0.05, yb, mw, mh),
      A(0.05 + mw + 0.62, yb, W - (0.05 + mw + 0.62) - 0.08, mh)]
cax_p = A(0.05 + 2 * mw + 0.2, yt, 0.08, mh)
ext = (0, Lx, 0, Ly)
for a, m, t in ((ax[0], p["m0"], r"{\fontsize{9}{11}\selectfont (\textbf{a})}\ straight fibers"),
                (ax[1], p["m1"], r"{\fontsize{9}{11}\selectfont (\textbf{b})}\ band-optimized layout")):
    im = field(a, T, m / m.max(), 1.0, t, ext)
    for (hx, hy, hr) in holes:
        hole(a, (hx, hy), hr)
        ring(a, (hx, hy), hr + float(p["guard"]), color=C_RIM, ls=":", lw=1.3, halo=True)
    square(a, p["pay"][:2], p["pay"][2], color=C_PAY, ls="-.", lw=1.1, halo=True)       # regions are boxes
    square(a, p["sensor"][:2], p["sensor"][2], color=C_SEN, ls="--", lw=1.1, halo=True)
    a.text(p["pay"][0], p["pay"][1] + p["pay"][2] + 0.08, "P", color=C_PAY, fontsize=FS, ha="center", va="bottom", zorder=6,
           bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
    a.text(p["sensor"][0], p["sensor"][1] + p["sensor"][2] + 0.08, "S", color=C_SEN, fontsize=FS, ha="center", va="bottom", zorder=6,
           bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
cb = fig.colorbar(im, cax=cax_p)
cb.ax.tick_params(labelsize=FS); cb.set_label(r"$|\mathbf{u}|/\max|\mathbf{u}|$", fontsize=FS)
plot_toolpaths_phase(ax[2], p["cent"], p["thopt"], (0, Lx), (0, Ly), holes=holes, spacing=0.16,
                     mode="line", color=ps.INK, lw=0.4, n=900)
frame(ax[2], ext, rf"{{\fontsize{{9}}{{11}}\selectfont (\textbf{{c}})}}\ tow paths, $\max|\zeta|={float(p['zmax']):.1f}\le{float(p['zeta_all']):.2f}$")
S = np.load(os.path.join(R, "dolfinx_panel_sweep_repro.npz"))
b = ax[3]
b.axvspan(44, 56, color=ps.FILL_MUTE, zorder=0)
for key, ls in (("band", "-"), ("tone", "--")):
    r = S[key]
    b.semilogy(r[:, 0], r[:, 1], ls, color=C_SEN, lw=1.2)
    b.semilogy(r[:, 0], r[:, 2], ls, color=C_PAY, lw=1.2)
    b.semilogy(r[:, 0], r[:, 3], ls, color=C_RIM, lw=1.2)
b.axhline(1, color=ps.GREY, lw=0.6, ls=":")
b.set_xticks([42, 46, 50, 54, 58])
b.set_yticks([0.1, 0.2, 0.5, 1, 2]); b.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"]); b.minorticks_off()
b.set_xlabel(r"drive frequency $\omega$"); b.set_ylabel("energy / straight fibers")
b.set_title(r"{\fontsize{9}{11}\selectfont (\textbf{d})}\ performance across frequency", loc="left", fontsize=FS, pad=3)
ps.grid(b); ps.despine(b)
handles = [Line2D([], [], color=C_SEN, lw=1.2, label="sensor S (map: dashed square)"),
           Line2D([], [], color=C_PAY, lw=1.2, label="payload P (map: dash-dot square)"),
           Line2D([], [], color=C_RIM, lw=1.2, label="fastener rims (map: dotted rings)"),
           Line2D([], [], color=ps.INK, lw=1.2, ls="-", label="band-optimized design"),
           Line2D([], [], color=ps.INK, lw=1.2, ls="--", label="single-frequency design"),
           Patch(color=ps.FILL_MUTE, label="design band")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=FS,
           bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.4, borderaxespad=0.1)
fig.savefig(os.path.join(FIG, "wt_panel.pdf")); print("wrote wt_panel.pdf")
