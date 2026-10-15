r"""Physical schematic of the three wave-control problems and of the mechanism (paper Fig. 1).

(a) the lens: a 4 x 3 CFRP plate driven by an x-directed line source at its left edge, absorbing
    layers on the other three edges, the tow paths of the actual optimized lens, and the
    objective region;
(b) why fibers channel energy: the quasi-P slowness curve of the CFRP lamina (fibers along x).
    A wave whose normal n points 60 deg off the fibers carries its energy along the normal to the
    slowness curve, only a few degrees off the fibers;
(c) the cloak: a traction-free void, the design shell in which orientation may change, and the
    far-field observation region;
(d) the multifunctional panel: fastener holes with their rim regions, payload P, sensor S and a
    Gaussian-beam drive.

    python examples/analysis/wt_schematic.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_NLTOPT = _os.environ.get("NLTOPT_DIR", _os.path.join(_os.path.dirname(_ROOT), "NLTopt"))
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "examples", "analysis")); sys.path.insert(0, ROOT)
import paperstyle as ps
ps.use_palatino()
import matplotlib.pyplot as plt
plt.rcParams["contour.negative_linestyle"] = "solid"
from matplotlib.patches import Rectangle, Circle
from wavetopo.dolfinx_viz import plot_toolpaths_phase

R = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "docs", "paper", "figs")
TGT = "#22b8cf"
SPONGE = "#dfe7ee"


def plate(ax, Lx, Ly, m=0.6, sides=("r", "t", "b")):
    ax.add_patch(Rectangle((0, 0), Lx, Ly, fc="white", ec=ps.INK, lw=1.0, zorder=0))
    for k in range(12):                                                 # absorbing layer, graded shading
        f = (k + 1) / 12; d = m * (1 - k / 12)
        kw = dict(fc=SPONGE, ec="none", alpha=0.18, zorder=1)
        if "r" in sides: ax.add_patch(Rectangle((Lx - d, 0), d, Ly, **kw))
        if "t" in sides: ax.add_patch(Rectangle((0, Ly - d), Lx, d, **kw))
        if "b" in sides: ax.add_patch(Rectangle((0, 0), Lx, d, **kw))
    ax.set_aspect("equal"); ax.axis("off")


def line_source(ax, x, y0, y1, n=7, L=0.35, gauss=None):
    for y in np.linspace(y0, y1, n):
        s = 1.0 if gauss is None else np.exp(-((y - gauss[0]) / gauss[1]) ** 2)
        if s > 0.08:
            ax.annotate("", xy=(x + L * s, y), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", color=ps.RED, lw=0.9),
                        zorder=6)


fig = plt.figure(figsize=(ps.FULL, 4.3))
# ---------------------------------------------------------------- (a) lens
ax = fig.add_axes([0.0, 0.5, 0.5, 0.45])
Lx, Ly = 4.0, 3.0
plate(ax, Lx, Ly)
d = np.load(os.path.join(R, "dolfinx_lens_data.npz"))
plot_toolpaths_phase(ax, d["cent"], d["thopt"], (0, Lx), (0, Ly), spacing=0.16, mode="line", color=ps.GREEN, lw=0.45)
ax.set_xlim(-0.55, Lx + 0.75); ax.set_ylim(-0.45, Ly + 0.15)
line_source(ax, 0.1, 0.3, 2.7)
ax.add_patch(Rectangle((0.78 * Lx - 0.18, Ly / 2 - 0.18), 0.36, 0.36, fill=False, ec=TGT, ls="--", lw=1.2, zorder=7))
ax.text(-0.5, Ly / 2, "line\nsource", color=ps.RED, fontsize=7.5, ha="left", va="center", rotation=90)
ax.text(Lx + 0.05, Ly - 0.3, "absorbing\nlayer", fontsize=7, color=ps.GREY, ha="left", va="top")
ax.text(0.78 * Lx, Ly / 2 + 0.28, "objective", color="#127f93", fontsize=7.5, ha="center", zorder=8,
        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
ax.text(Lx / 2, -0.3, r"$4\times3$, tow paths of the optimized lens", fontsize=7.5, ha="center")
fig.text(0.01, 0.955, r"(\textbf{a}) orientation-only lens", fontsize=8.5)

# ---------------------------------------------------------------- (b) channeling
bx = fig.add_axes([0.55, 0.5, 0.43, 0.45]); bx.set_aspect("equal"); bx.axis("off")
sys.path.insert(0, (_NLTOPT + ""))
from nltopt.laminate import lamina_Q
Q = lamina_Q() / 1e9; rho = 1.6
psi = np.linspace(0, 2 * np.pi, 721)
c = np.array([np.sqrt(np.linalg.eigvalsh(np.array([[Q[0, 0] * np.cos(p) ** 2 + Q[2, 2] * np.sin(p) ** 2,
                                                     (Q[0, 1] + Q[2, 2]) * np.cos(p) * np.sin(p)],
                                                    [(Q[0, 1] + Q[2, 2]) * np.cos(p) * np.sin(p),
                                                     Q[2, 2] * np.cos(p) ** 2 + Q[1, 1] * np.sin(p) ** 2]]))[-1] / rho)
              for p in psi])
sc = 1 / c; sc = sc / sc.max()                                          # slowness, normalized
for yy in np.linspace(-1.05, 1.05, 12):                                 # fibers along x
    bx.plot([-1.25, 1.25], [yy, yy], color=ps.GREEN, lw=0.45, alpha=0.22, zorder=0)
bx.plot(sc * np.cos(psi), sc * np.sin(psi), color=ps.INK, lw=1.3, zorder=2)
p0 = np.deg2rad(60); k0 = np.argmin(np.abs(psi - p0)); s0 = sc[k0]
P0 = np.array([s0 * np.cos(p0), s0 * np.sin(p0)])
bx.annotate("", xy=P0, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ps.STEEL, lw=1.4), zorder=4)
ds = np.gradient(sc, psi)[k0]                                           # outward normal to the slowness curve
t = np.array([ds * np.cos(p0) - s0 * np.sin(p0), ds * np.sin(p0) + s0 * np.cos(p0)])
nrm = np.array([t[1], -t[0]]); nrm /= np.linalg.norm(nrm)
if nrm @ P0 < 0: nrm = -nrm
bx.annotate("", xy=P0 + 0.55 * nrm, xytext=P0, arrowprops=dict(arrowstyle="-|>", color=ps.RED, lw=1.6), zorder=5)
bx.plot(*P0, "o", color=ps.INK, ms=3, zorder=6)
skew = np.rad2deg(np.arctan2(nrm[1], nrm[0]))
bx.text(-0.42, 0.08, "wave normal $\\mathbf n$\n($60^\\circ$ off the fibers)", color=ps.STEEL, fontsize=7.5, ha="right",
        bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))
bx.text(P0[0] + 0.58 * nrm[0], P0[1] + 0.58 * nrm[1] + 0.05, f"energy flux\n(${abs(skew):.0f}^\\circ$ off the fibers)",
        color=ps.RED, fontsize=7.5, ha="left", va="bottom", bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))
bx.text(-1.22, -1.2, "fibers", color=ps.GREEN, fontsize=7.5)
bx.text(0.42, -0.9, "qP slowness\ncurve, CFRP", fontsize=7.5, bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))
bx.set_xlim(-1.3, 1.6); bx.set_ylim(-1.3, 1.3)
fig.text(0.55, 0.955, r"(\textbf{b}) mechanism: energy channels along the fibers", fontsize=8.5)

# ---------------------------------------------------------------- (c) cloak
cx = fig.add_axes([0.0, 0.03, 0.5, 0.4])
Lx, Ly, CX, CY, RV, RD, RC = 8.0, 5.0, 3.2, 2.5, 0.55, 1.5, 1.7
plate(cx, Lx, Ly)
line_source(cx, 0.1, 0.5, 4.5, n=8, L=0.5)
th = np.linspace(0, 2 * np.pi, 200)
cx.add_patch(Circle((CX, CY), RC, fc="none", ec=ps.GREY, ls=":", lw=0.9, zorder=3))
cx.add_patch(Circle((CX, CY), RD, fc="#e9f7f9", ec=TGT, ls="--", lw=1.2, zorder=3))
cx.add_patch(Circle((CX, CY), RV, fc="white", ec=ps.INK, lw=1.0, zorder=4))
cx.text(CX, CY, "void", fontsize=7, ha="center", va="center", zorder=5)
cx.text(CX, CY + RD + 0.08, "design shell", color="#127f93", fontsize=7.5, ha="center", va="bottom", zorder=5,
        bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
cx.text(6.0, 1.15, "far field\nobserved\n($r>R_C$)", fontsize=7.5, color=ps.GREY, ha="center")
cx.text(Lx / 2, -0.45, r"$8\times5$; fibers straight outside the shell", fontsize=7.5, ha="center")
cx.set_xlim(-0.2, Lx + 0.1); cx.set_ylim(-0.75, Ly + 0.1)
fig.text(0.01, 0.455, r"(\textbf{c}) cloak", fontsize=8.5)

# ---------------------------------------------------------------- (d) panel
dx = fig.add_axes([0.52, 0.03, 0.47, 0.4])
Lx, Ly = 7.0, 4.0
plate(dx, Lx, Ly)
line_source(dx, 0.15, 0.6, 3.4, n=11, L=0.55, gauss=(2.0, 0.95))
for (hx, hy) in ((2.55, 1.05), (2.55, 2.95), (4.35, 1.05), (4.35, 2.95)):
    dx.add_patch(Circle((hx, hy), 0.2 + 0.18, fc="#fbe9e7", ec=ps.RED, ls=":", lw=1.0, zorder=3))
    dx.add_patch(Circle((hx, hy), 0.2, fc="white", ec=ps.INK, lw=0.9, zorder=4))
dx.add_patch(Rectangle((3.2 - 0.5, 2.0 - 0.5), 1.0, 1.0, fc="#e6f4ec", ec=ps.GREEN, ls="-.", lw=1.1, zorder=3))
dx.add_patch(Rectangle((5.9 - 0.3, 2.0 - 0.3), 0.6, 0.6, fc="#e5f0f6", ec=ps.STEEL, ls="--", lw=1.1, zorder=3))
dx.text(3.2, 2.0, "P", color=ps.GREEN, fontsize=8, ha="center", va="center", zorder=5)
dx.text(2.62, 2.0, "payload", color=ps.GREEN, fontsize=7, ha="right", va="center", zorder=5)
dx.text(5.9, 2.0, "S", color=ps.STEEL, fontsize=8, ha="center", va="center", zorder=5)
dx.text(5.9, 2.42, "sensor", color=ps.STEEL, fontsize=7, ha="center", va="bottom", zorder=5)
dx.text(3.45, 3.62, "fastener rims", color=ps.RED, fontsize=7, ha="center")
dx.text(Lx / 2, -0.45, r"$7\times4$, Gaussian-beam drive, band $\omega\in[44,56]$", fontsize=7.5, ha="center")
dx.set_xlim(-0.2, Lx + 0.1); dx.set_ylim(-0.75, Ly + 0.1)
fig.text(0.53, 0.455, r"(\textbf{d}) multifunctional panel", fontsize=8.5)

fig.savefig(os.path.join(FIG, "wt_schematic.pdf")); print("wrote wt_schematic.pdf")
