r"""Revision R1 figures: what the lens does across frequency (wt_lens_band.pdf) and the
energy-channeling evidence (wt_flux.pdf).

    python examples/analysis/wt_r1_figs.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import os
import sys
import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paperstyle as ps
ps.use_palatino()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import griddata

R = os.path.join(_ROOT, "results")
FIG = os.path.join(_ROOT, "docs", "paper", "figs")
LX, LY, M = 4.0, 3.0, 0.6
TGT = "#22b8cf"
ABSC = "#7a5195"                                      # better-absorber series

# ================================================================ Fig. 3: lens across frequency
rows = np.load(os.path.join(R, "lens_mechanism.npz"))["rows"]
om, E0, I0, P0, E1, I1, P1, G, Gphi, GP = rows.T
AB = np.load(os.path.join(R, "lens_absorber.npz"))["rows"]       # omega,G_new,Gphi_new,growth_new,G_old,Gphi_old,growth_old
FN = np.load(os.path.join(R, "wt_r1_fine.npy"))                  # omega,G,Gphi,growth
PW = np.load(os.path.join(R, "wt_r1_power.npy"))                 # omega,which, (Pin,Peta,Pc,Pt) x2
SR = np.load(os.path.join(R, "wt_r1_absorber.npy"))              # pad,omega,R, paper(G,Gp,gr), absorber(G,Gp,gr)
RY = np.load(os.path.join(R, "wt_r1_rays.npz"), allow_pickle=True)

fig, axs = plt.subplots(2, 2, figsize=(ps.FULL, 4.3), sharex=True)
(a, b), (c, d) = axs
for ax in axs.flat:
    ax.axvspan(35, 50, color="#eef0f3", lw=0, zorder=0)
    ax.axvline(40, color=ps.GREY, lw=0.8, ls="--", zorder=1)
    ax.axhline(1, color=ps.GREY, lw=0.6, ls=":", zorder=1)
# (a) raw gain
a.semilogy(om, G, "o-", color=ps.INK, ms=3.2, mfc="white", mew=1.0, lw=1.2, label=r"paper lens, layer $|R|\approx0.33$")
a.semilogy(AB[:, 0], AB[:, 1], "D--", color=ABSC, ms=3.0, mfc="white", mew=1.0, lw=1.0, label=r"re-optimized, layer $|R|\le0.13$")
a.set_ylabel(r"target-energy gain $G$"); a.set_ylim(0.1, 300)
a.set_yticks([0.1, 1, 10, 100]); a.set_yticklabels(["0.1", "1", "10", "100"]); a.minorticks_off()
a.legend(frameon=False, fontsize=6.6, loc="lower right", handlelength=1.6)
a.set_title(r"(\textbf{a}) single-frequency gain: narrow peak", loc="left", fontsize=8.5)
# (b) redirection, with the absorber series and the ray prediction
b.plot(om, Gphi, "s-", color=ps.GREEN, ms=3.0, lw=1.3, label=r"paper lens, $|R|\approx0.33$")
b.plot(AB[:, 0], AB[:, 2], "D--", color=ABSC, ms=3.0, mfc="white", mew=1.0, lw=1.0, label=r"re-optimized, $|R|\le0.13$")
for pad, mk, lab in ((1.2, "v", r"paper lens, $|R|\approx0.04$"), (1.8, "^", r"paper lens, $|R|\approx0.01$")):
    s = SR[SR[:, 0] == pad]
    b.plot(s[:, 1], s[:, 4], mk, color=ps.GREEN, ms=4.2, mfc="white", mew=1.0, ls="none", label=lab)
gr0, gr1 = float(RY["paper_G_ray"]), float(RY["absorber_G_ray"])
b.axhspan(min(gr0, gr1), max(gr0, gr1), xmin=0, xmax=1, color=ps.STEEL, alpha=0.18, lw=0, zorder=0.5)
b.text(20.0, max(gr0, gr1) + 0.12, "ray prediction (no reflections)", color=ps.STEEL, fontsize=6.6, ha="left", va="bottom")
b.set_ylabel(r"redirection $G_\varphi$"); b.set_ylim(0, 9.5)
b.set_ylim(0, 11)
b.legend(frameon=False, fontsize=6.4, loc="upper left", handlelength=1.6, ncol=2, columnspacing=0.8)
b.set_title(r"(\textbf{b}) redirection: broadband", loc="left", fontsize=8.5)
# (c) interior field growth, with the fine sweep
c.semilogy(om, I1 / I0, "^-", color=ps.RED, ms=3.0, lw=1.1, label=r"paper lens")
c.semilogy(FN[:, 0], FN[:, 3], ".", color=ps.RED, ms=2.6, label=r"fine sweep, $\Delta\omega=0.25$")
c.semilogy(AB[:, 0], AB[:, 3], "D--", color=ABSC, ms=3.0, mfc="white", mew=1.0, lw=1.0, label=r"re-optimized, $|R|\le0.13$")
c.set_ylabel(r"interior field growth"); c.set_ylim(0.08, 40)
c.set_yticks([0.1, 0.3, 1, 3, 10, 30]); c.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"]); c.minorticks_off()
c.legend(frameon=False, fontsize=6.4, loc="lower right", handlelength=1.6, borderaxespad=0.1)
c.set_xlabel(r"drive frequency $\omega$")
c.set_title(r"(\textbf{c}) whole-field growth: resonant", loc="left", fontsize=8.5)
# (d) power routed into the target
for which, col, mk, ls, lab in ((0, ps.GREEN, "s", "-", r"paper lens, $|R|\approx0.33$"), (1, ABSC, "D", "--", r"re-optimized, $|R|\le0.13$")):
    s = PW[PW[:, 1] == which]
    Pi0, Pi1 = s[:, 5] / s[:, 2], s[:, 9] / s[:, 6]
    d.plot(s[:, 0], Pi1 / Pi0, mk + ls, color=col, ms=3.0, mfc="white" if which else col, mew=1.0, lw=1.1, label=lab)
d.set_ylabel(r"power routed, $\Pi/\Pi_0$"); d.set_ylim(0, 13)
d.set_xlabel(r"drive frequency $\omega$")
d.legend(frameon=False, fontsize=6.6, loc="lower right", handlelength=1.6, borderaxespad=0.1)
d.set_title(r"(\textbf{d}) injected power absorbed in the target", loc="left", fontsize=8.5)
for ax in axs.flat:
    ax.set_xlim(19, 61); ps.grid(ax); ps.despine(ax)
c.text(39.5, 0.09, "design $\\omega$", fontsize=6.6, color=ps.GREY, ha="right", va="bottom")
c.text(49.5, 30, "band average", fontsize=6.6, color=ps.GREY, ha="right", va="top")
fig.tight_layout(h_pad=0.8, w_pad=1.2)
fig.savefig(os.path.join(FIG, "wt_lens_band.pdf")); print("wrote wt_lens_band.pdf")

# ================================================================ Fig. 4: channeling
F = np.load(os.path.join(R, "wt_r1_fluxmap_lens.npy"))         # cx, cy, Px, Py, theta, dev, strong
FS = np.load(os.path.join(R, "wt_r1_fluxmap_straight.npy"))
c_, Pv, th, dev, strong = F[:, :2], F[:, 2:4], F[:, 4], F[:, 5], F[:, 6].astype(bool)
if np.median(FS[:, 2]) < 0:                                    # flux of the straight plate must point +x
    Pv = -Pv
inner = (c_[:, 0] > 0.3) & (c_[:, 0] < LX - M) & (c_[:, 1] > M) & (c_[:, 1] < LY - M)
mag = np.linalg.norm(Pv, axis=1)

fig = plt.figure(figsize=(ps.FULL, 4.4))
# (a) common sampling of fiber tangents and flux vectors
a = fig.add_axes([0.065, 0.56, 0.42, 0.38])
gx, gy = np.linspace(0.25, LX - M - 0.1, 17), np.linspace(M + 0.08, LY - M - 0.08, 9)
GX, GY = np.meshgrid(gx, gy)
U = griddata(c_, Pv[:, 0], (GX, GY)); V = griddata(c_, Pv[:, 1], (GX, GY))
T = griddata(c_, th, (GX, GY), method="nearest")
S = np.hypot(U, V); Sm = S / np.nanmax(S)
a.quiver(GX, GY, np.cos(T), np.sin(T), headwidth=0, headlength=0, headaxislength=0, pivot="middle",
         color="#9aa5ae", scale=11, width=0.0055, zorder=2)
keep = Sm > 0.05
a.quiver(GX[keep], GY[keep], (U / S)[keep], (V / S)[keep], color=ps.GREEN, scale=13, width=0.006,
         headwidth=3.6, headlength=3.6, headaxislength=3.2, pivot="middle", zorder=3)
a.add_patch(plt.Rectangle((0.78 * LX - 0.18, LY / 2 - 0.18), 0.36, 0.36, ec=TGT, fc="none", ls="--", lw=1.0, zorder=5))
a.set_aspect("equal"); a.set_xlim(0, LX - M + 0.05); a.set_ylim(M, LY - M)
a.set_xticks([0, 1, 2, 3]); a.set_yticks([0.6, 1.5, 2.4])
a.set_xlabel(r"$x$", labelpad=1); a.set_ylabel(r"$y$", labelpad=1)
a.legend(handles=[Line2D([], [], color=ps.GREEN, lw=1.2, label=r"flux direction"),
                  Line2D([], [], color="#9aa5ae", lw=1.2, label="fiber direction"),
                  Line2D([], [], color=TGT, lw=1.0, ls="--", label="target")],
         loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=6.8, handlelength=1.4)
a.set_title(r"(\textbf{a}) flux and fibers, $\omega=40$ (same points)", loc="left", fontsize=8.5)
# (b) deviation map, weak-flux cells masked
b = fig.add_axes([0.545, 0.56, 0.38, 0.38])
xx, yy = np.linspace(0.3, LX - M, 220), np.linspace(M, LY - M, 110)
XX, YY = np.meshgrid(xx, yy)
DV = griddata(c_[inner], np.where(strong, dev, np.nan)[inner], (XX, YY), method="nearest")
im = b.pcolormesh(XX, YY, DV, cmap="magma_r", vmin=0, vmax=45, shading="auto", rasterized=True)
b.add_patch(plt.Rectangle((0.78 * LX - 0.18, LY / 2 - 0.18), 0.36, 0.36, ec=TGT, fc="none", ls="--", lw=1.0, zorder=5))
b.set_aspect("equal"); b.set_xlim(0, LX - M + 0.05); b.set_ylim(M, LY - M)
b.set_facecolor("#d9dde2"); b.set_xticks([0, 1, 2, 3]); b.set_yticks([0.6, 1.5, 2.4])
b.set_xlabel(r"$x$", labelpad=1)
cax = fig.add_axes([0.935, 0.56, 0.012, 0.38]); cb = fig.colorbar(im, cax=cax, extend="max")
cb.set_label("flux--fiber angle (deg)", fontsize=7); cb.ax.tick_params(labelsize=6.8)
b.text(0.02, 0.03, r"gray: $|\bm P|<5\%$ of max", transform=b.transAxes, fontsize=6.4, color=ps.INK,
       bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))
b.set_title(r"(\textbf{b}) where the flux leaves the fibers", loc="left", fontsize=8.5)
# (c) weighted histogram
h = fig.add_axes([0.065, 0.10, 0.25, 0.30])
w = mag * (inner & strong)
phiP = np.arctan2(Pv[:, 1], Pv[:, 0])
dx_ = np.degrees(np.abs(np.angle(np.exp(2j * phiP))) / 2)
bins = np.linspace(0, 90, 19)
h.hist(dev[inner & strong], bins=bins, weights=w[inner & strong] / w.sum(), color=ps.GREEN, alpha=0.85, label="flux vs.\\ fiber")
h.hist(dx_[inner & strong], bins=bins, weights=w[inner & strong] / w.sum(), histtype="step", color=ps.INK, lw=1.1, label=r"flux vs.\ drive ($x$)")
h.set_xlabel("angle (deg)"); h.set_ylabel(r"$|\bm{P}|$-weighted share")
h.set_xticks([0, 30, 60, 90]); h.set_xlim(0, 90)
h.legend(frameon=False, fontsize=6.6, loc="upper right")
h.set_title(r"(\textbf{c}) flux-weighted angles", loc="left", fontsize=8.5)
ps.grid(h); ps.despine(h)
# (d) deviation vs local turning rate of the fibers
e = fig.add_axes([0.40, 0.10, 0.25, 0.30])
from scipy.interpolate import LinearNDInterpolator
gxx, gyy = np.linspace(0, LX, 321), np.linspace(0, LY, 241)
GXX, GYY = np.meshgrid(gxx, gyy)
vv = LinearNDInterpolator(c_, np.c_[np.cos(2 * th), np.sin(2 * th)])(GXX, GYY)
TT = 0.5 * np.arctan2(vv[..., 1], vv[..., 0])
ty, tx = np.gradient(TT, gyy[1] - gyy[0], gxx[1] - gxx[0])
from scipy.interpolate import RegularGridInterpolator
Ig = RegularGridInterpolator((gyy, gxx), np.nan_to_num(np.hypot(tx, ty)))
lam = 2 * np.pi * 9.07 / 40.0
turn = Ig(np.c_[c_[:, 1], c_[:, 0]]) * lam                    # radians turned per along-fiber wavelength
sel = inner & strong
edges = np.array([0, 0.5, 1, 2, 3, 4, 6, 10])
mids, mean = [], []
for i in range(len(edges) - 1):
    m_ = sel & (turn >= edges[i]) & (turn < edges[i + 1])
    if m_.sum() > 20:
        mids.append(0.5 * (edges[i] + edges[i + 1])); mean.append(np.sum(w[m_] * dev[m_]) / w[m_].sum())
e.plot(mids, mean, "o-", color=ps.GREEN, ms=3.5, lw=1.2)
e.set_xlabel(r"fiber turning per wavelength, $\lambda_\parallel|\nabla\theta|$ (rad)")
e.set_ylabel("mean flux--fiber angle (deg)"); e.set_ylim(0, None)
e.set_title(r"(\textbf{d}) channeling vs.\ turning rate", loc="left", fontsize=8.5)
ps.grid(e); ps.despine(e)
# (e) skew of a homogeneous lamina
k = fig.add_axes([0.74, 0.10, 0.24, 0.30])
from lamina import lamina_Q, Q_family
for s, col, lab in ((0.0, ps.GREY, r"$E_{f1}/E_{f2}=1$"), (0.5, ps.STEEL, "2.9"), (1.0, ps.INK, "14.6 (CFRP)")):
    Q = Q_family(s, lamina_Q() / 1e9)
    psv = np.linspace(0, np.pi / 2, 400); cc = []
    for p_ in psv:
        n = np.array([np.cos(p_), np.sin(p_)]); N = np.array([[n[0], 0], [0, n[1]], [n[1], n[0]]])
        cc.append(np.sqrt(np.linalg.eigvalsh(N.T @ Q @ N)[-1] / 1.6))
    cc = np.array(cc); dpsi = np.gradient(cc, psv)
    k.plot(np.degrees(psv), np.degrees(psv + np.arctan(dpsi / cc)), color=col, lw=1.3, label=lab)
k.set_xlabel(r"wave normal $\psi$ (deg)"); k.set_ylabel(r"qP energy direction (deg)")
k.set_xlim(0, 90); k.set_ylim(0, 90); k.set_xticks([0, 30, 60, 90]); k.set_yticks([0, 30, 60, 90])
k.legend(frameon=False, fontsize=6.4, loc="upper left", handlelength=1.2)
k.set_title(r"(\textbf{e}) lamina skew", loc="left", fontsize=8.5)
ps.grid(k); ps.despine(k)
fig.savefig(os.path.join(FIG, "wt_flux.pdf")); print("wrote wt_flux.pdf")
print("turning bins:", list(zip(mids, np.round(mean, 1))))
