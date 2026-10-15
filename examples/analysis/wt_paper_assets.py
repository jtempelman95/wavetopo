r"""Generate every number, table and new figure of the wavetopo manuscript from the
result files (single source of truth):

  docs/paper/wt_numbers.tex          macros \W...
  docs/paper/wt_*_table.tex          lens band, dose, front, cloak band tables
  docs/paper/figs/wt_limits.pdf   (wt_lens_band.pdf and wt_flux.pdf: wt_r1_figs.py)

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
    pass                                                     # figure: wt_r1_figs.py

# ---------------------------------------------------------------- energy flux
if have("lens_flux.npz"):
    F = np.load(os.path.join(R, "lens_flux.npz"))
    put("WFluxDev", F["lens_40"][0]); put("WFluxShare", 100 * F["lens_40"][1], "{:.0f}")
    put("WFluxDevX", F["lens_40"][2])
    put("WFluxDevThirtyFive", F["lens_35"][0]); put("WFluxDevFortyFive", F["lens_45"][0])
    put("WFluxDevFifty", F["lens_50"][0])
    devs = [F[f"lens_{w}"][0] for w in (35, 40, 45, 50)]
    put("WFluxDevBandMin", min(devs)); put("WFluxDevBandMax", max(devs))
# figure wt_flux.pdf: examples/analysis/wt_r1_figs.py

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
    MS = {}
    for _s in D[:, 0]:
        f_ = os.path.join(R, f"wt_r1_dose_ms_{_s:.2f}.npy")
        if os.path.exists(f_):
            MS[round(float(_s), 2)] = np.load(f_)                  # rows: s, band (paper layer), band (layer x2)
    with open(os.path.join(P, "wt_dose_table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{@{}ccccccc@{}}\n\\toprule\n$s$ & $E_{f1}/E_{f2}$ & $s_{20}$ & $G(40)$ & $G_\\varphi(40)$ & "
                 "$\\langle G_\\varphi\\rangle_{\\rm band}$, 4 starts & better absorber, 4 starts\\\\ \\midrule\n")
        for s, r, g, gp, gb in D:
            m_ = MS.get(round(float(s), 2))
            c1 = f"{m_[:, 1].min():.2f}--{m_[:, 1].max():.2f}" if m_ is not None else f"{gb:.2f}"
            c2 = f"{m_[:, 2].min():.2f}--{m_[:, 2].max():.2f}" if m_ is not None else "1.00"
            fh.write(f"{s:.2f} & {r:.1f} & {s20[round(s,2)]}\\% & {g:.1f} & {gp:.2f} & {c1} & {c2}\\\\\n")
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
        ax[0].plot(D[:, 1], D[:, 4], "o:", color=ps.GREEN, ms=3.2, mfc="white", lw=0.9, zorder=2)
        _x = [D[i, 1] for i in range(len(D)) if round(float(D[i, 0]), 2) in MS]
        _m = [MS[round(float(D[i, 0]), 2)] for i in range(len(D)) if round(float(D[i, 0]), 2) in MS]
        for xx_, mm_ in zip(_x, _m):
            ax[0].plot([xx_, xx_], [mm_[:, 1].min(), mm_[:, 1].max()], color=ps.GREEN, alpha=0.35, lw=3, solid_capstyle="butt")
            ax[0].plot([xx_, xx_], [mm_[:, 2].min(), mm_[:, 2].max()], color="#7a5195", alpha=0.35, lw=3, solid_capstyle="butt")
        _bx = [D[0, 1]] + _x; _b1 = [1.0] + [mm_[:, 1].max() for mm_ in _m]; _b2 = [1.0] + [mm_[:, 2].max() for mm_ in _m]
        ax[0].plot(_bx, _b1, "o-", color=ps.GREEN, ms=4, lw=1.4, zorder=3)
        ax[0].plot(_bx, _b2, "D--", color="#7a5195", ms=3.5, mfc="white", mew=1.0, lw=1.1, zorder=3)
        ax[0].set_xscale("log"); logticks(ax[0], [1, 2, 3, 5, 10, 20, 30])
        ax[0].set_xlabel(r"lamina anisotropy $E_{f1}/E_{f2}$")
        ax[0].set_ylabel(RED_LAB); ax[0].set_ylim(0, 12.5); ax[0].set_yticks(range(0, 8))
        t = ax[0].twinx(); t.plot(D[:, 1], [s20[round(s, 2)] for s in D[:, 0]], "s--", color=ps.GREY, ms=3, lw=1.0)
        t.set_ylim(0, 166); t.set_yticks([0, 20, 40, 60, 80, 100]); t.set_ylabel(r"channeling share $s_{20}$ (\%)", color=ps.GREY)
        t.tick_params(axis="y", colors=ps.GREY); t.spines["top"].set_visible(False)
        ax[0].legend(handles=[Line2D([], [], color=ps.GREEN, marker="o", ms=4, label=r"best of 4, $|R|\approx0.33$"),
                              Line2D([], [], color="#7a5195", marker="D", mfc="white", ls="--", ms=3.5, label=r"best of 4, $|R|\le0.13$"),
                              Line2D([], [], color=ps.GREEN, marker="o", mfc="white", ls=":", ms=3.2, label="straight start (one run)"),
                              Line2D([], [], color=ps.GREY, alpha=0.5, lw=3, label="range over starts"),
                              Line2D([], [], color=ps.GREY, marker="s", ls="--", ms=3, label=r"lamina $s_{20}$ (right)")],
                     frameon=False, fontsize=6.6, loc="upper left", borderaxespad=0.1, handlelength=1.6)
        ps.panel(ax[0], "a"); ps.grid(ax[0])
    if have("lens_front.npy"):
        fin = np.isfinite(Fr[:, 0])
        LAM_L = 2 * np.pi * 9.0712 / 40.0                            # along-fiber qP wavelength, lens
        for _z, _rr in ROB.items():                                  # spread over independent starts
            ax[1].plot([1 / (_z * LAM_L)] * 2, [_rr[:, 5].min(), _rr[:, 5].max()], color=ps.GREEN, alpha=0.35, lw=3,
                       solid_capstyle="butt", zorder=1)
        ax[1].plot([], [], color=ps.GREEN, alpha=0.35, lw=3, label=f"range over {NSTART} starts")
        ax[1].plot(1 / (Fr[fin, 0] * LAM_L), Fr[fin, 4], "o-", color=ps.GREEN, ms=4, lw=1.4, label="best found (lower bound)", zorder=3)
        ax[1].axhline(Fr[~fin, 4][0], color=ps.GREEN, lw=0.9, ls=":", label="best, no limit")
        ax[1].plot([0.7, 0.7], [0, 7.6], color=ps.GREY, lw=0.7, ls="--", zorder=0)
        ax[1].text(0.7, 7.7, r"$0.7\lambda_\parallel$", fontsize=6.4, color=ps.GREY, ha="center", va="bottom")
        ax[1].set_xscale("log"); logticks(ax[1], [0.2, 0.5, 1, 2])
        ax[1].set_xlabel(r"min.\ tow radius $R/\lambda_\parallel$"); ax[1].set_ylabel(r"redirection $\langle G_\varphi\rangle$")
        ax[1].set_ylim(0, 12.5); ax[1].set_yticks(range(0, 8)); ax[1].set_title("lens", fontsize=8.5)
        ax[1].legend(frameon=False, fontsize=6.6, loc="upper left", ncol=1, handlelength=1.6, borderaxespad=0.1)
        ps.panel(ax[1], "b"); ps.grid(ax[1]); ps.despine(ax[1])
    if CF is not None:
        LAM_C = 2 * np.pi * 9.0712 / 50.0                            # along-fiber qP wavelength, cloak
        ax[2].plot(1 / (np.asarray(CF["zeta_all"]) * LAM_C), CF["reduction"], "s-", color=ps.STEEL, ms=4, lw=1.4, label="curvature-limited")
        ax[2].axhline(float(CF["red_unc"]), color=ps.STEEL, lw=0.9, ls=":", label="no limit")
        ax[2].set_xscale("log"); logticks(ax[2], [0.2, 0.5, 1])
        ax[2].set_xlabel(r"min.\ tow radius $R/\lambda_\parallel$"); ax[2].set_ylabel(r"far-field reduction at $\omega=50$")
        ax[2].set_ylim(0, 57); ax[2].set_yticks(range(0, 41, 10)); ax[2].set_title("cloak", fontsize=8.5)
        ax[2].legend(frameon=False, fontsize=6.6, loc="upper right", ncol=1, handlelength=1.6, borderaxespad=0.1)
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

# ---------------------------------------------------------------- Revision R1
if have("wt_r1_rays.npz"):
    RY = np.load(os.path.join(R, "wt_r1_rays.npz"), allow_pickle=True)
    put("WRayPaper", float(RY["paper_G_ray"])); put("WRayAbs", float(RY["absorber_G_ray"]))
    put("WRayShareStraight", 100 * float(RY["straight_share"])); put("WRayShareLens", 100 * float(RY["paper_share"]))
if have("wt_r1_absorber.npy"):
    SR = np.load(os.path.join(R, "wt_r1_absorber.npy"))            # pad, omega, R, paper(G,Gp,gr), absorber(G,Gp,gr)
    bands = {}
    for pad in np.unique(SR[:, 0]):
        q = SR[SR[:, 0] == pad]
        bands[pad] = (q[:, 2].max(), q[:, 4].mean(), q[:, 7].mean())
    put("WReflOne", bands[0.6][0], "{:.2f}"); put("WReflTwo", bands[1.2][0], "{:.2f}"); put("WReflThree", bands[1.8][0], "{:.3f}")
    conv = [v for k, v in bands.items() if k >= 0.6]
    allb = [c[1] for c in conv] + [c[2] for c in conv]
    put("WBandConvMin", min(allb)); put("WBandConvMax", max(allb))
if have("wt_r1_fine.npy"):
    FN = np.load(os.path.join(R, "wt_r1_fine.npy"))
    k = int(np.argmax(FN[:, 3])); gmax = FN[k, 3]; half = 0.5 * gmax
    om_f = np.linspace(FN[0, 0], FN[-1, 0], 4001); gr_f = np.interp(om_f, FN[:, 0], FN[:, 3])
    above = om_f[gr_f >= half]; width = above.max() - above.min()
    put("WFinePeak", gmax); put("WFinePeakOm", FN[k, 0]); put("WFineWidth", width); put("WFineQ", FN[k, 0] / width, "{:.0f}")
    put("WFineGpMin", FN[:, 2].min()); put("WFineGpMax", FN[:, 2].max())
if have("wt_r1_power.npy"):
    PW = np.load(os.path.join(R, "wt_r1_power.npy"))
    for which, tag in ((0, "Paper"), (1, "Abs")):
        q = PW[PW[:, 1] == which]; inb = (q[:, 0] >= 35) & (q[:, 0] <= 50)
        rat = (q[:, 9] / q[:, 6]) / (q[:, 5] / q[:, 2])
        put(f"WPow{tag}Min", rat[inb].min()); put(f"WPow{tag}Max", rat[inb].max())
        k40 = int(np.argmin(np.abs(q[:, 0] - 40)))
        put(f"WPow{tag}PiZero", 100 * q[k40, 5] / q[k40, 2], "{:.2f}"); put(f"WPow{tag}PiOne", 100 * q[k40, 9] / q[k40, 6], "{:.1f}")
        put(f"WPow{tag}Forty", rat[k40])
    bal = max(np.max(np.abs(PW[:, 3] + PW[:, 4] - PW[:, 2]) / PW[:, 2]), np.max(np.abs(PW[:, 7] + PW[:, 8] - PW[:, 6]) / PW[:, 6]))
    put("WPowBal", f"10^{{{int(np.ceil(np.log10(bal)))}}}", "{}")
if have("wt_r1_fluxmap.npy"):
    FMd = np.load(os.path.join(R, "wt_r1_fluxmap.npy"), allow_pickle=True).item()
    put("WFluxDevMasked", FMd["lens_40"]["dev_w"]); put("WFluxStrongFrac", 100 * FMd["lens_40"]["frac_strong"], "{:.0f}")
    put("WFluxShareMasked", 100 * FMd["lens_40"]["share20"], "{:.0f}")
    put("WPolLens", 100 * FMd["lens_40"]["pol"], "{:.0f}"); put("WPolStraight", 100 * FMd["straight_40"]["pol"], "{:.0f}")
    pols = [FMd[f"lens_{w}"]["pol"] for w in (35, 40, 45, 50)]
    put("WPolMin", 100 * min(pols), "{:.0f}"); put("WPolMax", 100 * max(pols), "{:.0f}")
if have("wt_r1_asbuilt.npy"):
    AS = np.load(os.path.join(R, "wt_r1_asbuilt.npy")); bm = AS[:, 2:].mean(axis=1)
    put("WAsBuiltMin", bm.min(), "{:.2f}"); put("WAsBuiltMax", bm.max(), "{:.2f}")
    TA = np.load(os.path.join(R, "wt_r1_tau.npy")); c_ = TA[:, :2]
    inner = (c_[:, 0] < 3.4) & (c_[:, 1] > 0.6) & (c_[:, 1] < 2.4)
    put("WTauMax", TA[inner, 2].max(), "{:.0f}"); put("WTauMin", TA[inner, 2].min(), "{:.3f}")
if have("wt_r1_cloak_series.npy"):
    CS = np.load(os.path.join(R, "wt_r1_cloak_series.npy"))
    for i, nm in enumerate(("A", "B", "C", "D")):
        if i < len(CS):
            put(f"WCSer{nm}", CS[i, 2], "{:.0f}"); put(f"WCSerCurv{nm}", CS[i, 3], "{:.1f}"); put(f"WCSerTr{nm}", CS[i, 4], "{:.1f}")
            put(f"WCSerH{nm}", f"{CS[i, 0]:.4g}"); put(f"WCSerN{nm}", f"{int(CS[i, 1]):,}".replace(",", "{,}"))
if have("wt_r1_cloak_sponge.npy"):
    CSP = np.load(os.path.join(R, "wt_r1_cloak_sponge.npy"))
    put("WCSpOpt", CSP[1], "{:.0f}"); put("WCSpCurv", CSP[2], "{:.1f}"); put("WCSpTr", CSP[3], "{:.1f}")
import glob as _gl
_ms = sorted(_gl.glob(os.path.join(R, "wt_r1_dose_ms_*.npy")))
if _ms:
    MSd = {round(float(np.load(f)[0, 0]), 2): np.load(f) for f in _ms}
    put("WDoseMSmaxSpread", max(v[:, 1].max() - v[:, 1].min() for v in MSd.values()))
    put("WDoseHalfMin", MSd[0.5][:, 1].min()); put("WDoseHalfMax", MSd[0.5][:, 1].max())
    put("WDoseBestCFRP", MSd[1.0][:, 1].max()); put("WDoseBestCFRPAbs", MSd[1.0][:, 2].max())
    put("WDoseBestMaxAbs", MSd[max(MSd)][:, 2].max()); put("WDoseBestQuarterAbs", MSd[0.25][:, 2].max())
if have("wt_r1_panel.npz"):
    PN = np.load(os.path.join(R, "wt_r1_panel.npz"))
    b = PN["band"]; inb = (b[:, 0] >= 44) & (b[:, 0] <= 56)
    put("WPanShareMin", b[inb, 5].min()); put("WPanShareMax", b[inb, 5].max())
    put("WPanGrowthMin", b[inb, 4].min(), "{:.2f}"); put("WPanGrowthMax", b[inb, 4].max(), "{:.2f}")
    put("WPanSenMean", b[inb, 1].mean(), "{:.2f}"); put("WPanPayMean", b[inb, 2].mean(), "{:.2f}"); put("WPanRimMean", b[inb, 3].mean(), "{:.2f}")
    t = PN["tone"]
    put("WPanToneRimMean", t[inb, 3].mean(), "{:.2f}"); put("WPanToneRimWorst", t[inb, 3].max(), "{:.2f}")
    put("WPanRimWorst", b[inb, 3].max(), "{:.2f}")

if have("wt_r1_power.npy"):
    for which, tag in ((0, "Paper"), (1, "Abs")):
        q = PW[PW[:, 1] == which]; inb = (q[:, 0] >= 35) & (q[:, 0] <= 50)
        rn = (q[:, 9] / q[:, 7]) / (q[:, 5] / q[:, 3])                 # target share of the STRUCTURAL loss
        put(f"WPowNorm{tag}Min", rn[inb].min()); put(f"WPowNorm{tag}Max", rn[inb].max())
if have("wt_r1_fluxmap.npy"):
    put("WFluxDevMThirtyFive", FMd["lens_35"]["dev_w"]); put("WFluxDevMFortyFive", FMd["lens_45"]["dev_w"])
    put("WFluxDevMFifty", FMd["lens_50"]["dev_w"])
if CF is not None:
    _z = np.asarray(CF["zeta_all"]); _r = np.asarray(CF["reduction"])
    put("WCFrontThreeFive", _r[np.argmin(np.abs(_z - 3.5))]); put("WCFrontUnc", float(CF["red_unc"]))
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
