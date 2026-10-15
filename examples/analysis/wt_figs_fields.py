r"""Vector (pgf/LaTeX) field figures of the wavetopo manuscript, from saved results:
wt_lens.pdf, wt_cloak.pdf, wt_panel.pdf.  Field maps are rasterized inside the vector PDF,
carry a labelled colourbar, and voids are drawn as clean filled discs.

    python examples/analysis/wt_figs_fields.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import os
import sys
import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
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
RY = np.load(os.path.join(R, "wt_r1_rays.npz"), allow_pickle=True)
LMc = np.load(os.path.join(R, "lens_mechanism.npz"))["rows"]
_k = int(np.argmin(np.abs(LMc[:, 0] - 40))); G40, Gp40 = LMc[_k, 7], LMc[_k, 8]
T = tri(d["trix"], d["triy"], d["tris"]); ext = (-0.32, 4, 0, 3)
tgt = ((0.78 * 4, 1.5), 0.18)                    # objective box (0.78 Lx, Ly/2) +/- 0.18


def lens_deco(a):
    for xy in ((3.4, 0, 3.4, 3), (0, 0.6, 4, 0.6), (0, 2.4, 4, 2.4)):          # where the absorbing layer starts
        a.plot(xy[::2], xy[1::2], color="white", lw=0.7, ls=(0, (2, 2)), alpha=0.85, zorder=4)
    for yy in np.linspace(0.75, 2.25, 4):                                       # x-directed line drive
        a.annotate("", xy=(-0.02, yy), xytext=(-0.3, yy), arrowprops=dict(arrowstyle="-|>", color=ps.RED, lw=0.8), zorder=6)


fig = plt.figure(figsize=(ps.FULL, 2.05))
gl = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.045, 0.16, 1], wspace=0.06, left=0.01, right=0.99, bottom=0.03, top=0.86)
ax = [fig.add_subplot(gl[0, 0]), fig.add_subplot(gl[0, 1]), fig.add_subplot(gl[0, 4])]
cax_l = fig.add_subplot(gl[0, 2])
for k, (m, t) in enumerate(((d["m0"], r"{\fontsize{9}{11}\selectfont (\textbf{a})}\ straight fibers"),
                            (d["m1"], rf"{{\fontsize{{9}}{{11}}\selectfont (\textbf{{b}})}}\ steered: $G={G40:.0f}\times$, $G_\varphi={Gp40:.1f}\times$"))):
    im = field(ax[k], T, m / m.max(), 1.0, t, ext)        # each map normalised by its own maximum
    square(ax[k], *tgt, color=TGT, halo=True)
    lens_deco(ax[k])
plot_toolpaths_phase(ax[2], d["cent"], d["thopt"], (0, 4), (0, 3), spacing=0.16, mode="line",
                     color="#8d969e", lw=0.4)
for pth in RY["paper_paths"]:
    if len(pth) > 1:
        ax[2].plot(pth[:, 0], pth[:, 1], color=ps.RED, lw=0.65, zorder=4)
ax[2].plot([3.4, 3.4], [0, 3], color=ps.GREY, lw=0.6, ls=":"); ax[2].plot([0, 4], [0.6, 0.6], color=ps.GREY, lw=0.6, ls=":")
ax[2].plot([0, 4], [2.4, 2.4], color=ps.GREY, lw=0.6, ls=":")
square(ax[2], *tgt, color=TGT, lw=1.3, halo=True)
frame(ax[2], (0, 4, 0, 3), r"{\fontsize{9}{11}\selectfont (\textbf{c})}\ tow paths (gray), energy rays (red)")
cb = fig.colorbar(im, cax=cax_l); cb.ax.tick_params(labelsize=FS)
cb.set_label(r"$|\mathbf{u}|/\max|\mathbf{u}|$ (per panel)", fontsize=FS)
pos = cax_l.get_position(); cax_l.set_position([pos.x0, pos.y0 + 0.06, pos.width * 0.6, pos.height - 0.12])
fig.savefig(os.path.join(FIG, "wt_lens.pdf")); print("wrote wt_lens.pdf")

# ---------------------------------------------------------------- cloak
h = np.load(os.path.join(R, "cloak_headtohead.npz"))
SC = np.load(os.path.join(R, "wt_r1_cloak_fields.npz"))
Th = tri(h["trix"], h["triy"], h["tris"]); Ts = tri(SC["x"], SC["y"], SC["tris"])
CX, CY, RV, RD = float(h["CX"]), float(h["CY"]), float(h["RV"]), float(h["R_DES"])
RCO = float(h["RC"])                                       # far-field observation region: r > R_C
ext = (0.6, 7.4, 0.6, 4.4)
vmax = float(np.max(SC["inc"]))                           # the incident field sets the scale of both rows
lab = lambda L, t: rf"{{\fontsize{{9}}{{11}}\selectfont (\textbf{{{L}}})}}\ " + t
fig, axs = plt.subplots(2, 4, figsize=(ps.FULL, 2.55))
Tf = tri(h["trfx"], h["trfy"], h["trfs"])
top = [(Tf, h["uref"], lab("a", "incident (void-free)")),
       (Th, h["ustr"], lab("b", "void, straight fibers")),
       (Th, h["uunc"], lab("c", rf"optimized shell, ${float(SC['red_optimized']):.0f}\times$"))]
for k, (Tk, m, t) in enumerate(top):
    a = axs[0, k]
    im = field(a, Tk, m, vmax, t, ext)
    if k > 0:
        hole(a, (CX, CY), RV)
    if k == 2:
        ring(a, (CX, CY), RD, color=TGT, halo=True)
a = axs[0, 3]
plot_toolpaths_phase(a, h["cent"], h["th_unc"], (0, 8), (0, 5), holes=[(CX, CY, RV)],
                     spacing=0.16, mode="line", color=ps.INK, lw=0.4, n=900)
ring(a, (CX, CY), RD, color=TGT)
frame(a, ext, lab("d", "tow paths of (c)"))
bot = [("straight", "straight"), ("optimized", "optimized"), ("transformation", "transformation"),
       ("curv-limited", r"curv.-limited")]
for k, (key, t) in enumerate(bot):
    a = axs[1, k]
    red = "" if key == "straight" else rf", ${float(SC['red_' + key]):.{0 if float(SC['red_' + key]) > 10 else 1}f}\times$"
    im2 = field(a, Ts, SC[f"scat_{key}"], vmax, lab("efgh"[k], t + red), ext)
    hole(a, (CX, CY), RV)
    ring(a, (CX, CY), RCO, color="white", ls=":", lw=0.8)
    if key != "straight":
        ring(a, (CX, CY), RD, color=TGT, lw=0.7)
fig.subplots_adjust(left=0.03, right=0.89, bottom=0.02, top=0.92, wspace=0.04, hspace=0.24)
fig.text(0.012, 0.25, r"scattered $|\mathbf{u}-\mathbf{u}_{\rm inc}|$", rotation=90, fontsize=FS, ha="center", va="center")
fig.text(0.012, 0.73, r"total $|\mathbf{u}|$", rotation=90, fontsize=FS, ha="center", va="center")
cax = fig.add_axes([0.905, 0.10, 0.013, 0.78])
cb = fig.colorbar(im, cax=cax, extend="max"); cb.ax.tick_params(labelsize=FS)
cb.set_label(r"$|\mathbf{u}|$, $|\mathbf{u}-\mathbf{u}_{\rm inc}|$ (one scale)", fontsize=FS)
fig.savefig(os.path.join(FIG, "wt_cloak.pdf")); print("wrote wt_cloak.pdf")

# ---------------------------------------------------------------- panel
p = np.load(os.path.join(R, "dolfinx_panel_band_repro_data.npz"))
T = tri(p["trix"], p["triy"], p["tris"]); Lx, Ly = float(p["Lx"]), float(p["Ly"])
holes = [tuple(x) for x in p["holes"]]
C_SEN, C_PAY, C_RIM = ps.STEEL, ps.GREEN, ps.RED      # same palette as Figs. 2-4
W, mw = ps.FULL, 2.75
mh = mw * Ly / Lx
y3 = 0.55; y2 = y3 + 1.55 + 0.55; yt = y2 + mh + 0.42
H = yt + mh + 0.22
fig = plt.figure(figsize=(W, H))
A = lambda x0, y0, w, h: fig.add_axes([x0 / W, y0 / H, w / W, h / H])
ax = [A(0.05, yt, mw, mh), A(0.05 + mw + 0.12, yt, mw, mh), A(0.05, y2, mw, mh),
      A(0.05 + mw + 0.62, y2, W - (0.05 + mw + 0.62) - 0.08, mh)]
cax_p = A(0.05 + 2 * mw + 0.2, yt, 0.08, mh)
ext = (-0.3, Lx, 0, Ly)
for a, m, t in ((ax[0], p["m0"], r"{\fontsize{9}{11}\selectfont (\textbf{a})}\ straight fibers, $\omega=50$"),
                (ax[1], p["m1"], r"{\fontsize{9}{11}\selectfont (\textbf{b})}\ band-optimized layout, $\omega=50$")):
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
    for yy in np.linspace(1.2, 2.8, 4):                                         # Gaussian-beam drive along x
        a.annotate("", xy=(-0.02, yy), xytext=(-0.28, yy), arrowprops=dict(arrowstyle="-|>", color=ps.RED, lw=0.8), zorder=6)
    m_ = 0.6
    a.plot([Lx - m_, Lx - m_], [0, Ly], color="white", lw=0.6, ls=(0, (2, 2)), alpha=0.8, zorder=4)
    a.plot([0, Lx], [m_, m_], color="white", lw=0.6, ls=(0, (2, 2)), alpha=0.8, zorder=4)
    a.plot([0, Lx], [Ly - m_, Ly - m_], color="white", lw=0.6, ls=(0, (2, 2)), alpha=0.8, zorder=4)
cb = fig.colorbar(im, cax=cax_p)
cb.ax.tick_params(labelsize=FS); cb.set_label(r"$|\mathbf{u}|/\max|\mathbf{u}|$", fontsize=FS)
plot_toolpaths_phase(ax[2], p["cent"], p["thopt"], (0, Lx), (0, Ly), holes=holes, spacing=0.16,
                     mode="line", color=ps.INK, lw=0.4, n=900)
frame(ax[2], (0, Lx, 0, Ly), rf"{{\fontsize{{9}}{{11}}\selectfont (\textbf{{c}})}}\ tow paths, $\max|\zeta|={float(p['zmax']):.1f}\le{float(p['zeta_all']):.2f}$")
S = np.load(os.path.join(R, "dolfinx_panel_sweep_repro.npz"))
b = ax[3]
b.axvspan(44, 56, color=ps.FILL_MUTE, zorder=0)
for key, ls in (("band", "-"), ("tone", "--")):
    r = S[key]
    b.semilogy(r[:, 0], r[:, 1], ls, color=C_SEN, lw=1.2)
    b.semilogy(r[:, 0], r[:, 2], ls, color=C_PAY, lw=1.2, marker="s", ms=3.0, mfc="white", markevery=2)
    b.semilogy(r[:, 0], r[:, 3], ls, color=C_RIM, lw=1.2, marker="^", ms=3.4, mfc="white", markevery=2)
b.axhline(1, color=ps.GREY, lw=0.6, ls=":")
b.set_xticks([42, 46, 50, 54, 58])
b.set_yticks([0.1, 0.2, 0.5, 1, 2]); b.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"]); b.minorticks_off()
b.set_ylim(0.07, 9)
b.set_xlabel(r"drive frequency $\omega$"); b.set_ylabel("energy / straight fibers")
b.set_title(r"{\fontsize{9}{11}\selectfont (\textbf{d})}\ performance across frequency", loc="left", fontsize=FS, pad=3)
ps.grid(b); ps.despine(b)
b.legend(handles=[Line2D([], [], color=C_SEN, lw=1.2, label="sensor S"),
                  Line2D([], [], color=C_PAY, lw=1.2, marker="s", ms=3.0, mfc="white", label="payload P"),
                  Line2D([], [], color=C_RIM, lw=1.2, marker="^", ms=3.4, mfc="white", label="rims"),
                  Line2D([], [], color=ps.INK, lw=1.2, ls="-", label="band design"),
                  Line2D([], [], color=ps.INK, lw=1.2, ls="--", label="single-$\\omega$ design")],
         frameon=False, fontsize=6.4, loc="upper center", ncol=3, handlelength=1.6, columnspacing=0.8, borderaxespad=0.1)
# (e) sensor: redistribution vs growth;  (f) band summary
PN = np.load(os.path.join(R, "wt_r1_panel.npz"))
e = A(0.55, y3, 2.55, 1.55)
r = PN["band"]
e.axvspan(44, 56, color=ps.FILL_MUTE, zorder=0)
e.semilogy(r[:, 0], r[:, 1], "-", color=C_SEN, lw=1.3, label="sensor energy")
e.semilogy(r[:, 0], r[:, 5], "--", color=C_SEN, lw=1.1, label="sensor share (redistribution)")
e.semilogy(r[:, 0], r[:, 4], ":", color=ps.GREY, lw=1.3, label="interior growth")
e.axhline(1, color=ps.GREY, lw=0.6, ls=":")
e.set_yticks([0.3, 0.5, 1, 2, 3]); e.set_yticklabels(["0.3", "0.5", "1", "2", "3"]); e.minorticks_off()
e.set_xticks([42, 46, 50, 54, 58]); e.set_xlabel(r"drive frequency $\omega$"); e.set_ylabel("ratio to straight fibers")
e.legend(frameon=False, fontsize=6.4, loc="upper center", handlelength=1.8, borderaxespad=0.1, ncol=2, columnspacing=0.8)
e.set_ylim(0.25, 9)
e.set_title(r"{\fontsize{9}{11}\selectfont (\textbf{e})}\ sensor: redirection, not resonance", loc="left", fontsize=FS, pad=3)
ps.grid(e); ps.despine(e)
f = A(3.75, y3, W - 3.75 - 0.08, 1.55)
inb = (r[:, 0] >= 44) & (r[:, 0] <= 56)
labs = ["sensor", "payload", "rims"]
xs = np.arange(3)
for k, (key, col, off) in enumerate((("band", ps.INK, -0.19), ("tone", "#9aa5ae", 0.19))):
    rr = PN[key][inb]
    mean = [rr[:, 1].mean(), rr[:, 2].mean(), rr[:, 3].mean()]
    worst = [rr[:, 1].min(), rr[:, 2].max(), rr[:, 3].max()]
    f.bar(xs + off, mean, 0.36, color=col, label=("band design" if key == "band" else "single-$\\omega$ design") + ": band mean")
    f.plot(xs + off, worst, "v" if True else "o", color=ps.RED, ms=4.5, ls="none", zorder=5,
           label="worst in band" if key == "band" else None)
f.axhline(1, color=ps.GREY, lw=0.6, ls=":")
f.set_xticks(xs); f.set_xticklabels(labs); f.set_ylabel("energy / straight fibers")
f.set_ylim(0, 2.6)
f.legend(frameon=False, fontsize=6.3, loc="upper right", handlelength=1.4, borderaxespad=0.1)
f.set_title(r"{\fontsize{9}{11}\selectfont (\textbf{f})}\ over the band $[44,56]$", loc="left", fontsize=FS, pad=3)
ps.grid(f, axis="y"); ps.despine(f)
fig.savefig(os.path.join(FIG, "wt_panel.pdf")); print("wrote wt_panel.pdf")
