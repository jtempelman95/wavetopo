r"""Generate every number, table and new figure of the wavetopo manuscript from the
result files (single source of truth):

  docs/paper/wt_numbers.tex          macros \W...
  docs/paper/wt_*_table.tex          lens band, dose, front, cloak band tables
  docs/paper/figs/wt_lens_band.pdf, wt_flux.pdf, wt_limits.pdf

    python examples/analysis/wt_paper_assets.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import os
import sys
import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paperstyle as ps                      # pgf/LaTeX vector style (shared with NLTopt)
ps.use_palatino()                       # wavetopo manuscript is set in Palatino
import matplotlib.pyplot as plt

R = (_ROOT + "/results")
P = (_ROOT + "/docs/paper")
FIG = os.path.join(P, "figs")
M = {}


def put(k, v, fmt="{:.1f}"):
    M[k] = v if isinstance(v, str) else fmt.format(v)


def have(f):
    return os.path.exists(os.path.join(R, f))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    print("wrote figs/" + name + ".pdf")


# ---------------------------------------------------------------- lens across frequency
if have("lens_mechanism.npz"):
    rows = np.load(os.path.join(R, "lens_mechanism.npz"))["rows"]
    om, E0, I0, P0, E1, I1, P1, G, Gphi, GP = rows.T
    k40 = int(np.argmin(np.abs(om - 40)))
    put("WGainForty", G[k40], "{:.0f}"); put("WRedirForty", Gphi[k40])
    put("WGrowthForty", (I1 / I0)[k40]); put("WPowerForty", (P1 / P0)[k40])
    put("WPhiZero", 100 * (E0 / I0)[k40], "{:.2f}"); put("WPhiOne", 100 * (E1 / I1)[k40], "{:.1f}")
    put("WGainMinusFive", G[np.argmin(np.abs(om - 37.5))], "{:.1f}")
    put("WGainPlusFive", G[np.argmin(np.abs(om - 42.5))], "{:.1f}")
    sel = (om >= 37.5) & (om <= 60); put("WRedirBandMin", Gphi[sel].min())
    band = np.isin(np.round(om, 1), [35.0, 40.0, 45.0, 50.0])         # the band average used throughout
    put("WRedirBandMean", Gphi[band].mean())
    with open(os.path.join(P, "wt_lensband_table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{@{}rrrrr@{}}\n\\toprule\n$\\omega$ & $G$ & $G_\\varphi$ & growth & $P/P_0$\\\\ \\midrule\n")
        for i in range(len(om)):
            if om[i] % 5 == 0 or abs(om[i] - 37.5) < 1e-9 or abs(om[i] - 42.5) < 1e-9:
                b = "\\bf " if abs(om[i] - 40) < 1e-9 else ""
                fh.write(f"{b}{om[i]:.1f} & {b}{G[i]:.1f} & {b}{Gphi[i]:.2f} & {b}{I1[i]/I0[i]:.2f} & {b}{P1[i]/P0[i]:.2f}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    fig, ax = plt.subplots(1, 2, figsize=(ps.FULL, 2.4))
    ax[0].semilogy(om, G, "o-", color=ps.INK, ms=3.5, mfc="white", mew=1.0, lw=1.2, zorder=6, label=r"gain $G$")
    ax[0].semilogy(om, Gphi, "s-", color=ps.GREEN, ms=3, lw=1.4, label=r"redirection $G_\varphi$")
    ax[0].semilogy(om, I1 / I0, "^--", color=ps.RED, ms=3, lw=1.0, label="interior field growth")
    ax[0].axvline(40, color=ps.GRID, lw=0.8); ax[0].axhline(1, color=ps.GREY, lw=0.6, ls=":")
    ax[0].set_xlabel(r"drive frequency $\omega$"); ax[0].set_ylabel("ratio to straight fibers")
    ax[0].set_ylim(0.1, 2e3); ax[0].set_yticks([0.1, 1, 10, 100]); ax[0].set_yticklabels(["0.1", "1", "10", "100"]); ax[0].minorticks_off()
    ax[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=2, columnspacing=1.0); ps.panel(ax[0], "a"); ps.grid(ax[0]); ps.despine(ax[0])
    ax[1].plot(om, 100 * E0 / I0, "o-", color=ps.GREY, ms=3, lw=1.2, label="straight fibers")
    ax[1].plot(om, 100 * E1 / I1, "s-", color=ps.GREEN, ms=3, lw=1.4, label="steered lens")
    ax[1].axvline(40, color=ps.GRID, lw=0.8)
    ax[1].set_xlabel(r"drive frequency $\omega$"); ax[1].set_ylabel(r"target share of interior energy (\%)")
    ax[1].set_ylim(0, 9.5); ax[1].legend(frameon=False, fontsize=8, loc="upper left"); ps.panel(ax[1], "b"); ps.grid(ax[1]); ps.despine(ax[1])
    fig.tight_layout(); save(fig, "wt_lens_band")

# ---------------------------------------------------------------- energy flux
if have("lens_flux.npz"):
    F = np.load(os.path.join(R, "lens_flux.npz"))
    put("WFluxDev", F["lens_40"][0]); put("WFluxShare", 100 * F["lens_40"][1], "{:.0f}")
    put("WFluxDevX", F["lens_40"][2])
    put("WFluxDevThirtyFive", F["lens_35"][0]); put("WFluxDevFortyFive", F["lens_45"][0])
    put("WFluxDevFifty", F["lens_50"][0])
    devs = [F[f"lens_{w}"][0] for w in (35, 40, 45, 50)]
    put("WFluxDevBandMin", min(devs)); put("WFluxDevBandMax", max(devs))
if have("lens_flux_lens.npy"):
    from scipy.interpolate import griddata
    d = np.load(os.path.join(R, "lens_flux_lens.npy"))
    c, Pv, th = d[:, :2], -d[:, 2:4], d[:, 4]      # saved with the e^{-iwt} sign; flip to e^{+iwt}
    Lx, Ly, m = 4.0, 3.0, 0.6
    gx, gy = np.linspace(0.05, Lx - m, 140), np.linspace(m, Ly - m, 70)
    GX, GY = np.meshgrid(gx, gy)
    U = griddata(c, Pv[:, 0], (GX, GY)); V = griddata(c, Pv[:, 1], (GX, GY))
    T = griddata(c, th, (GX, GY), method="nearest")
    fig = plt.figure(figsize=(ps.FULL, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.4)
    a = fig.add_subplot(gs[0])
    st = 8
    a.quiver(GX[::st, ::st], GY[::st, ::st], np.cos(T[::st, ::st]), np.sin(T[::st, ::st]),
             headwidth=0, headlength=0, headaxislength=0, pivot="middle", color="#8d969e",
             scale=24, width=0.005)
    spd = np.hypot(U, V)
    a.streamplot(gx, gy, U, V, density=0.8, color=ps.GREEN, linewidth=0.35 + 0.9 * spd / np.nanmax(spd),
                 arrowsize=0.55)
    a.add_patch(plt.Rectangle((0.78 * Lx - 0.18, Ly / 2 - 0.18), 0.36, 0.36, ec="#22b8cf", fc="none", ls="--", lw=1.0, zorder=5))
    a.set_aspect("equal"); a.set_xlim(0, Lx - m + 0.05); a.set_ylim(m, Ly - m)
    a.set_xlabel(r"$x$"); a.set_ylabel(r"$y$"); ps.panel(a, "a"); a.set_anchor("N")
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], color=ps.GREEN, lw=1.0, label=r"flux $\bm{P}$"),
                      Line2D([], [], color="#8d969e", lw=1.0, label="fiber direction"),
                      Line2D([], [], color="#22b8cf", lw=1.0, ls="--", label="objective region")],
             loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, fontsize=8,
             handlelength=1.5, columnspacing=0.9)
    b = fig.add_subplot(gs[1])
    mask = (c[:, 0] > 0.3) & (c[:, 0] < Lx - m) & (c[:, 1] > m) & (c[:, 1] < Ly - m)
    mag = np.linalg.norm(Pv, axis=1)[mask]
    phiP = np.arctan2(Pv[mask, 1], Pv[mask, 0])
    dfib = np.degrees(np.abs(np.angle(np.exp(2j * (phiP - th[mask])))) / 2)
    dx = np.degrees(np.abs(np.angle(np.exp(2j * phiP))) / 2)
    bins = np.linspace(0, 90, 19)
    b.hist(dfib, bins=bins, weights=mag / mag.sum(), color=ps.GREEN, alpha=0.85, label="flux vs.\\ fiber")
    b.hist(dx, bins=bins, weights=mag / mag.sum(), histtype="step", color=ps.INK, lw=1.1, label=r"flux vs.\ $x$")
    b.set_xlabel("angle (deg)"); b.set_ylabel(r"$|\bm{P}|$-weighted share")
    b.set_xticks([0, 30, 60, 90]); b.set_xlim(0, 90)
    b.legend(frameon=False, fontsize=8); ps.panel(b, "b"); ps.grid(b); ps.despine(b)
    cax = fig.add_subplot(gs[2])
    from lamina import lamina_Q, Q_family
    for s, col, lab in ((0.0, ps.GREY, "isotropic"), (0.5, ps.STEEL, r"$E_{f1}/E_{f2}=2.9$"), (1.0, ps.INK, "CFRP")):
        Q = Q_family(s, lamina_Q() / 1e9)
        psv = np.linspace(0, np.pi / 2, 400); cc = []
        for p_ in psv:
            n = np.array([np.cos(p_), np.sin(p_)]); N = np.array([[n[0], 0], [0, n[1]], [n[1], n[0]]])
            cc.append(np.sqrt(np.linalg.eigvalsh(N.T @ Q @ N)[-1] / 1.6))
        cc = np.array(cc); dpsi = np.gradient(cc, psv)
        yy = np.degrees(psv + np.arctan(dpsi / cc)); cax.plot(np.degrees(psv), yy, color=col, lw=1.3, label=lab)
    cax.set_xlabel(r"wave normal $\psi$ (deg)"); cax.set_ylabel(r"qP energy direction (deg)")
    cax.set_xlim(0, 90); cax.set_ylim(0, 90); cax.set_xticks([0, 30, 60, 90]); cax.set_yticks([0, 30, 60, 90])
    cax.legend([r"$E_{f1}/E_{f2}=1$", "2.9", "14.6"], loc="upper center", bbox_to_anchor=(0.4, -0.27),
               ncol=3, frameon=False, fontsize=8, handlelength=1.2, columnspacing=0.6, handletextpad=0.3); ps.panel(cax, "c"); ps.grid(cax); ps.despine(cax)
    fig.subplots_adjust(bottom=0.3, top=0.93, left=0.07, right=0.98); save(fig, "wt_flux")

# ---------------------------------------------------------------- anisotropy dose
from lamina import lamina_Q as _lQ, Q_family as _Qf


def _s20(s, n=3601):
    """Channeling share: fraction of qP wave normals (uniform in [0, 90 deg]) whose energy direction
    lies within 20 deg of the fiber, for the lamina Q(s) (rho = 1.6, Q in GPa)."""
    Q = _Qf(s, _lQ() / 1e9); psv = np.linspace(0, np.pi / 2, n); cc = []
    for p_ in psv:
        nn = np.array([np.cos(p_), np.sin(p_)]); N = np.array([[nn[0], 0], [0, nn[1]], [nn[1], nn[0]]])
        cc.append(np.sqrt(np.linalg.eigvalsh(N.T @ Q @ N)[-1] / 1.6))
    cc = np.array(cc); ed = psv + np.arctan(np.gradient(cc, psv) / cc)
    return 100 * np.mean(np.abs(np.degrees(ed)) <= 20)


s20 = {k: round(_s20(k)) for k in (0.0, 0.25, 0.5, 0.75, 1.0, 1.1)}


def _skew_row(s, n=3601):
    Q = _Qf(s, _lQ() / 1e9); psv = np.linspace(0, np.pi / 2, n); cc = []
    for p_ in psv:
        nn = np.array([np.cos(p_), np.sin(p_)]); N = np.array([[nn[0], 0], [0, nn[1]], [nn[1], nn[0]]])
        cc.append(np.sqrt(np.linalg.eigvalsh(N.T @ Q @ N)[-1] / 1.6))
    cc = np.array(cc); skew = np.degrees(np.abs(np.arctan(np.gradient(cc, psv) / cc)))
    Ex = Q[0, 0] - Q[0, 1] ** 2 / Q[1, 1]; Ey = Q[1, 1] - Q[0, 1] ** 2 / Q[0, 0]
    return Ex / Ey, skew.max(), _s20(s)


_rows = [_skew_row(k) for k in (0.0, 0.25, 0.5, 0.75, 1.0, 1.1)]
with open(os.path.join(P, "wt_skew_table.tex"), "w") as fh:
    fh.write("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n")
    fh.write("$E_{f1}/E_{f2}$ & " + " & ".join(f"{r[0]:.1f}" for r in _rows) + "\\\\\n")
    fh.write("max qP skew & " + " & ".join(f"${r[1]:.0f}^\\circ$" for r in _rows) + "\\\\\n")
    fh.write("$s_{20}$ & " + " & ".join(f"{r[2]:.0f}\\%" for r in _rows) + "\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
put("WSkewMax", _rows[4][1], "{:.0f}"); put("WSTwentyCFRP", _rows[4][2], "{:.0f}")
if have("lens_dose.npy"):
    D = np.load(os.path.join(R, "lens_dose.npy"))
    with open(os.path.join(P, "wt_dose_table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{@{}cccccc@{}}\n\\toprule\n$s$ & $E_{f1}/E_{f2}$ & $s_{20}$ & $G(40)$ & $G_\\varphi(40)$ & $\\langle G_\\varphi\\rangle_{\\rm band}$\\\\ \\midrule\n")
        for s, r, g, gp, gb in D:
            fh.write(f"{s:.2f} & {r:.1f} & {s20[round(s,2)]}\\% & {g:.1f} & {gp:.2f} & {gb:.2f}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    put("WDoseIso", D[0, 4], "{:.2f}")
    kk = lambda v: int(np.argmin(np.abs(D[:, 0] - v)))
    put("WDoseThree", D[kk(0.5), 4]); put("WDoseMax", D[-1, 4])
    put("WDoseGThree", D[kk(0.5), 2], "{:.0f}"); put("WDoseGFive", D[kk(0.75), 2], "{:.0f}")
    put("WDoseCFRP", D[np.argmin(np.abs(D[:, 0] - 1.0)), 4], "{:.1f}")

# ---------------------------------------------------------------- front
CF = np.load(os.path.join(R, "cloak_front.npz")) if have("cloak_front.npz") else None
if have("lens_front.npy"):
    # Feasibility envelope: a design with max|zeta| <= zeta_all is feasible for that limit
    # and for every looser one, so the best redirection at a limit is at least the best over
    # ALL computed designs (both optimizers, all limits) whose actual curvature satisfies it.
    allr = np.vstack([np.load(os.path.join(R, f)) for f in ("lens_front.npy", "lens_front_cont.npy") if have(f)])
    allr = allr[~((allr[:, 1] == 0) & (allr[:, 4] <= 1.0001))]      # drop "returned straight" failures
    import glob as _g
    ROB = {}                                                          # multi-start robustness runs
    for f in sorted(_g.glob(os.path.join(R, "lens_front_robust_*.npy"))):
        rr = np.load(f); ROB[float(rr[0, 0])] = rr
        allr = np.vstack([allr, rr[:, [0, 2, 3, 4, 5]]])
    NSTART = max((len(v) for v in ROB.values()), default=0)
    NDIST = max((len(np.unique(np.round(v[:, 5], 4))) for v in ROB.values()), default=0)   # distinct outcomes
    limits = [np.inf, 4.0, 2.5, 1.5, 1.0, 0.8, 0.6, 0.45, 0.3]
    Fr = []
    for za in limits:
        ok = allr[:, 1] <= 1.02 * za
        k = np.argmax(np.where(ok, allr[:, 4], -1))
        Fr.append((za, allr[k, 1], allr[k, 2], allr[k, 3], allr[k, 4]))
    Fr = np.array(Fr)
    with open(os.path.join(P, "wt_front_table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{@{}ccccccc@{}}\n\\toprule\n$\\zeta_{\\rm all}$ & $R=1/\\zeta_{\\rm all}$ & $\\max|\\zeta|$ & $G(40)$ & $G_\\varphi(40)$ & $\\langle G_\\varphi\\rangle_{\\rm band}$ & range over starts\\\\ \\midrule\n")
        for za, zm, g, gp, gb in Fr:
            rng = (f"{ROB[za][:, 5].min():.2f}--{ROB[za][:, 5].max():.2f}" if za in ROB else "--")
            fh.write(f"{'$\\infty$' if not np.isfinite(za) else f'{za:g}'} & {'0' if not np.isfinite(za) else f'{1/za:.2f}'} & {zm:.2f} & {g:.1f} & {gp:.2f} & {gb:.2f} & {rng}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    put("WFrontUnc", Fr[0, 4]); put("WFrontTight", Fr[-1, 4], "{:.2f}")
    gz = lambda z: Fr[np.argmin(np.abs(np.where(np.isfinite(Fr[:, 0]), Fr[:, 0], 1e9) - z)), 4]
    put("WFrontTwoFive", gz(2.5)); put("WFrontOneFive", gz(1.5)); put("WFrontOne", gz(1.0)); put("WFrontSix", gz(0.6))
    put("WFrontEight", gz(0.8)); put("WFrontFourFive", gz(0.45)); put("WFrontNStart", NSTART, "{:d}"); put("WFrontNDistinct", NDIST, "{:d}")
    if 1.0 in ROB:
        put("WFrontOneMin", ROB[1.0][:, 5].min())
    put("WFrontTightR", f"{1/Fr[-1,0]:.2f}")

# ---------------------------------------------------------------- cloak band
if have("cloak_band.npy"):
    C = np.load(os.path.join(R, "cloak_band.npy"))
    put("WCloakCurvBandMin", C[:, 2].min())
    with open(os.path.join(P, "wt_cloakband_table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{@{}cccc@{}}\n\\toprule\n$\\omega$ & optimized & curvature-limited ($1/R_V$) & transformation directions\\\\ \\midrule\n")
        for om, a, b, c in C:
            bb = "\\bf " if abs(om - 50) < 1e-9 else ""
            fh.write(f"{bb}{om:.1f} & {bb}{a:.1f} & {bb}{b:.1f} & {bb}{c:.1f}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    g = lambda w, j: C[np.argmin(np.abs(C[:, 0] - w)), j]
    put("WCloakPeak", g(50, 1), "{:.0f}"); put("WCloakMinus", g(47.5, 1)); put("WCloakPlus", g(52.5, 1))
    put("WCloakForty", g(40, 1)); put("WCloakAnalyticForty", g(40, 3))
    put("WCloakAnalytic", g(50, 3)); put("WCloakCurv", g(50, 2))
    put("WCloakCurvMinus", min(g(47.5, 2), g(52.5, 2))); put("WCloakCurvPlus", max(g(47.5, 2), g(52.5, 2)))

if have("lens_dose.npy") or have("lens_front.npy"):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    def logticks(a, vals):
        a.xaxis.set_major_locator(FixedLocator(vals)); a.xaxis.set_minor_locator(NullLocator())
        a.xaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in vals]))

    fig, ax = plt.subplots(1, 3, figsize=(ps.FULL, 2.45), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    RED_LAB = r"band-averaged redirection $\langle G_\varphi\rangle$"
    if have("lens_dose.npy"):
        ax[0].plot(D[:, 1], D[:, 4], "o-", color=ps.GREEN, ms=4, lw=1.4)
        ax[0].set_xscale("log"); logticks(ax[0], [1, 2, 3, 5, 10, 20, 30])
        ax[0].set_xlabel(r"lamina anisotropy $E_{f1}/E_{f2}$")
        ax[0].set_ylabel(RED_LAB); ax[0].set_ylim(0, 9.6); ax[0].set_yticks(range(0, 8))
        t = ax[0].twinx(); t.plot(D[:, 1], [s20[round(s, 2)] for s in D[:, 0]], "s--", color=ps.GREY, ms=3, lw=1.0)
        t.set_ylim(0, 128); t.set_yticks([0, 20, 40, 60, 80, 100]); t.set_ylabel(r"channeling share $s_{20}$ (\%)", color=ps.GREY)
        t.tick_params(axis="y", colors=ps.GREY); t.spines["top"].set_visible(False)
        ax[0].legend(handles=[Line2D([], [], color=ps.GREEN, marker="o", ms=4, label="re-optimized lens"),
                              Line2D([], [], color=ps.GREY, marker="s", ls="--", ms=3, label=r"lamina $s_{20}$ (right)")],
                     frameon=False, fontsize=8, loc="upper left", borderaxespad=0.1)
        ps.panel(ax[0], "a"); ps.grid(ax[0])
    if have("lens_front.npy"):
        fin = np.isfinite(Fr[:, 0])
        for _z, _rr in ROB.items():                                  # spread over independent starts
            ax[1].plot([_z, _z], [_rr[:, 5].min(), _rr[:, 5].max()], color=ps.GREEN, alpha=0.35, lw=3,
                       solid_capstyle="butt", zorder=1)
        ax[1].plot([], [], color=ps.GREEN, alpha=0.35, lw=3, label=f"range over {NSTART} starts")
        ax[1].plot(Fr[fin, 0], Fr[fin, 4], "o-", color=ps.GREEN, ms=4, lw=1.4, label="best (envelope)", zorder=3)
        ax[1].axhline(Fr[~fin, 4][0], color=ps.GREEN, lw=0.9, ls=":", label="best, no limit")
        ax[1].set_xscale("log"); logticks(ax[1], [0.3, 0.6, 1, 1.5, 2.5, 4])
        ax[1].set_xlabel(r"curvature limit $\zeta_{\rm all}=1/R$"); ax[1].set_ylabel(r"redirection $\langle G_\varphi\rangle$")
        ax[1].set_ylim(0, 10.8); ax[1].set_yticks(range(0, 8)); ax[1].set_title("lens", fontsize=8.5)
        ax[1].legend(frameon=False, fontsize=8, loc="upper left", ncol=1, handlelength=1.6, borderaxespad=0.1)
        ps.panel(ax[1], "b"); ps.grid(ax[1]); ps.despine(ax[1])
    if CF is not None:
        ax[2].plot(CF["zeta_all"], CF["reduction"], "s-", color=ps.STEEL, ms=4, lw=1.4, label="curvature-limited")
        ax[2].axhline(float(CF["red_unc"]), color=ps.STEEL, lw=0.9, ls=":", label="no limit")
        ax[2].set_xscale("log"); logticks(ax[2], [1, 2, 3, 5])
        ax[2].set_xlabel(r"curvature limit $\zeta_{\rm all}=1/R$"); ax[2].set_ylabel(r"far-field reduction at $\omega=50$")
        ax[2].set_ylim(0, 57); ax[2].set_yticks(range(0, 41, 10)); ax[2].set_title("cloak", fontsize=8.5)
        ax[2].legend(frameon=False, fontsize=8, loc="upper left", ncol=1, handlelength=1.6, borderaxespad=0.1)
        ps.panel(ax[2], "c"); ps.grid(ax[2]); ps.despine(ax[2])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.2, top=0.9, wspace=0.75)
    save(fig, "wt_limits")

# ---------------------------------------------------------------- numerical verification
if have("wt_verify_A.npy"):
    A = np.load(os.path.join(R, "wt_verify_A.npy")); B = np.load(os.path.join(R, "wt_verify_B.npy"))
    Cv = np.load(os.path.join(R, "wt_verify_C.npy")); Dv = np.load(os.path.join(R, "wt_verify_D.npy"))
    a40 = A[A[:, 0] == 40]
    put("WMeshGmin", a40[:, 2].min(), "{:.0f}"); put("WMeshGmax", a40[:, 2].max(), "{:.0f}")
    put("WMeshPmin", a40[:, 3].min(), "{:.2f}"); put("WMeshPmax", a40[:, 3].max(), "{:.2f}")
    b40 = B[B[:, 1] == 40]
    for eta, nm in ((0.01, "One"), (0.02, "Two"), (0.04, "Four")):
        r = b40[np.isclose(b40[:, 0], eta)][0]
        put(f"WEta{nm}G", r[2], "{:.0f}"); put(f"WEta{nm}P", r[3], "{:.1f}"); put(f"WEta{nm}Gr", r[4], "{:.1f}")
    put("WReflMin", Cv[:, 1].min(), "{:.2f}"); put("WReflMax", Cv[:, 1].max(), "{:.2f}")
    put("WReflThickMax", Cv[:, 6].max(), "{:.2f}")
    k40 = np.argmin(np.abs(Cv[:, 0] - 40))
    put("WThickPForty", Cv[k40, 9], "{:.1f}"); put("WThickGrForty", Cv[k40, 10], "{:.1f}")
    put("WThickGForty", Cv[k40, 8], "{:.0f}")
    sel = np.isin(Cv[:, 0], [35.0, 40.0, 45.0, 50.0])
    put("WThickBand", Cv[sel, 9].mean()); put("WThinBand", Cv[sel, 4].mean())
    put("WCloakFineEval", Dv[1, 2], "{:.0f}"); put("WCloakFineCurv", Dv[1, 3], "{:.1f}")
    put("WCloakFineTr", Dv[1, 4], "{:.1f}")
    if have("dolfinx_cloak_conforming_h030_data.npz"):
        put("WCloakFineReopt", float(np.load(os.path.join(R, "dolfinx_cloak_conforming_h030_data.npz"))["red"]), "{:.0f}")
    with open(os.path.join(P, "wt_verify_table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{@{}llccc@{}}\n\\toprule\ncheck & setting & $G$ & $G_\\varphi$ & growth\\\\ \\midrule\n")
        for om_, nx_, G_, P_, g_ in A:
            fh.write(f"lens mesh, $\\omega={om_:.0f}$ & $n_x={nx_:.0f}$ & {G_:.1f} & {P_:.2f} & {g_:.2f}\\\\\n")
        fh.write("\\midrule\n")
        for eta_, om_, G_, P_, g_ in B:
            if om_ in (40.0, 50.0):
                fh.write(f"lens damping, $\\omega={om_:.0f}$ & $\\eta={eta_:.2f}$ & {G_:.1f} & {P_:.2f} & {g_:.2f}\\\\\n")
        fh.write("\\midrule\n")
        for row in Cv:
            if row[0] in (37.5, 40.0, 45.0, 50.0):
                for off, lab in ((0, "0.6"), (5, "1.2")):
                    fh.write(f"lens sponge, $\\omega={row[0]:g}$ & width {lab}, $|R|={row[1+off]:.2f}$ & "
                             f"{row[3+off]:.1f} & {row[4+off]:.2f} & {row[5+off]:.2f}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
if have("lens_flux_control.npy"):
    FC = np.load(os.path.join(R, "lens_flux_control.npy"), allow_pickle=True)
    ctrl = [r for r in FC if str(r[1]) != "lens"]
    put("WCtrlDevMax", max(float(r[2]) for r in ctrl)); put("WCtrlDevMin", min(float(r[2]) for r in ctrl))
    put("WCtrlXMin", min(float(r[4]) for r in ctrl), "{:.0f}"); put("WCtrlXMax", max(float(r[4]) for r in ctrl), "{:.0f}")

# ---------------------------------------------------------------- lens with the better absorber
if have("lens_absorber.npz"):
    AB = np.load(os.path.join(R, "lens_absorber.npz"))["rows"]      # omega,G_new,Gphi_new,growth_new,G_old,Gphi_old,growth_old
    k = np.argmin(np.abs(AB[:, 0] - 40))
    put("WAbsG", AB[k, 1], "{:.0f}"); put("WAbsP", AB[k, 2]); put("WAbsGr", AB[k, 3])
    b = np.isin(AB[:, 0], [35.0, 40.0, 45.0, 50.0]); put("WAbsBand", AB[b, 2].mean()); put("WAbsBandOld", AB[b, 5].mean())
    put("WAbsMinFortySixty", AB[(AB[:, 0] >= 40) & (AB[:, 0] <= 60), 2].min())
    put("WAbsThirtySevenFive", AB[np.argmin(np.abs(AB[:, 0] - 37.5)), 2])
if have("cloak_fine_band.npy"):
    CB = np.load(os.path.join(R, "cloak_fine_band.npy"))
    put("WCloakFineMinus", CB[CB[:, 0] == 47.5, 1][0]); put("WCloakFinePlus", CB[CB[:, 0] == 52.5, 1][0])

# ---------------------------------------------------------------- panel mesh check
if have("panel_mesh_check.npy"):
    PM = np.load(os.path.join(R, "panel_mesh_check.npy"))         # h, omega, sensor, payload, rims
    c, f = PM[PM[:, 0] > 0.05], PM[PM[:, 0] < 0.05]
    put("WPanFineSenMin", f[:, 2].min(), "{:.2f}"); put("WPanFinePayMax", f[:, 3].max(), "{:.2f}")
    put("WPanFineRimMax", f[:, 4].max(), "{:.2f}")
    put("WPanMeshSenDev", 100 * np.max(np.abs(f[:, 2] / c[:, 2] - 1)), "{:.0f}")
    put("WPanMeshPayDev", 100 * np.max(np.abs(f[:, 3] / c[:, 3] - 1)), "{:.0f}")
    put("WPanMeshRimDevAbs", np.max(np.abs(f[:, 4] - c[:, 4])), "{:.2f}")
# elements per slowest (qS) wavelength: min qS speed 1.77, lambda = 2 pi c / omega
put("WElemLens", 2 * np.pi * 1.77 / 40 / (4.0 / 130), "{:.0f}")
put("WElemCloak", 2 * np.pi * 1.77 / 50 / 0.045, "{:.1f}")
put("WElemPanel", 2 * np.pi * 1.77 / 50 / 0.055, "{:.1f}")

# ---------------------------------------------------------------- multifunctional panel
if have("dolfinx_panel_band_repro_data.npz"):
    PB = np.load(os.path.join(R, "dolfinx_panel_band_repro_data.npz"))
    put("WPanRoute", float(PB["route"])); put("WPanShield", float(PB["shield"]), "{:.2f}")
    put("WPanProt", float(PB["prot"]), "{:.2f}"); put("WPanStiff", float(PB["stiff"]), "{:.2f}")
    put("WPanZmax", float(PB["zmax"]), "{:.1f}")
    SW = np.load(os.path.join(R, "dolfinx_panel_sweep_repro.npz"))
    for nm, key in (("Band", "band"), ("Tone", "tone")):
        r = SW[key]; ib = (r[:, 0] >= 44) & (r[:, 0] <= 56)
        k = np.argmin(r[ib, 1]); put(f"WPan{nm}SenMin", r[ib, 1][k], "{:.2f}"); put(f"WPan{nm}SenMinW", r[ib, 0][k], "{:.0f}")
        put(f"WPan{nm}RimMax", r[ib, 3].max(), "{:.2f}"); put(f"WPan{nm}PayMax", r[ib, 2].max(), "{:.2f}")
        ob = ~ib; put(f"WPan{nm}SenOutMin", r[ob, 1].min(), "{:.2f}")
    fq = PB["freqs"]; put("WPanBLo", fq.min(), "{:.0f}"); put("WPanBHi", fq.max(), "{:.0f}"); put("WPanNF", len(fq), "{:d}")

# ---------------------------------------------------------------- write macros
import glob, re
used = set()
for f in glob.glob(os.path.join(P, "ws_*.tex")):
    used |= set(re.findall(r"\\(W[A-Z][A-Za-z]*)", open(f).read()))
for u in sorted(used - set(M)):
    M[u] = r"\textbf{??}"
missing = sorted(u for u in used if M[u] == r"\textbf{??}")
if missing:
    print("MISSING:", " ".join(missing))
with open(os.path.join(P, "wt_numbers.tex"), "w") as fh:
    fh.write("% AUTO-GENERATED by examples/analysis/wt_paper_assets.py\n")
    for k in sorted(M):
        fh.write(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{M[k]}}}\n")
print(f"wrote wt_numbers.tex ({len(M)} macros)")
