"""The cloak manufacturability front: far-field reduction J0/J vs the allowed tow curvature
zeta_all.  Constrained optima climb from the gentle regime toward the unconstrained ceiling;
the geometry-tied limit 1/R_V marks the printable design; the analytical transformation
cloak sits at the singular zeta_all -> infinity end that no process reaches."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _paperfig import set_style, internal_ticks

d = np.load("results/cloak_front.npz")
za, red, zmax = d["zeta_all"], d["reduction"], d["zmax"]
red_unc, red_an, RV = float(d["red_unc"]), float(d["red_an"]), float(d["RV"])
zlim = 1.0 / RV

set_style()
fig, ax = plt.subplots(figsize=(5.6, 3.78))

# unconstrained ceiling (singular / infinite-curvature end)
ax.axhline(red_unc, ls="--", color="#8a5a00", lw=1.8,
           label=f"unconstrained optimum ${red_unc:.0f}\\times$ ($\\zeta_{{\\rm all}}\\!\\to\\!\\infty$)")
# analytical transformation cloak: singular curvature, marooned at the far end
ax.axhline(red_an, ls=":", color="#b3121b", lw=1.8,
           label=f"analytical cloak ${red_an:.1f}\\times$ (singular, unreachable)")
# the constrained front
ax.plot(za, red, "-o", color="#1f6fb2", lw=2.6, ms=9, label="curvature-constrained optima (printable)")
for x, y in zip(za, red):
    ax.annotate(f"{y:.0f}$\\times$", (x, y), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color="#1f6fb2")
# geometry-tied printable design at 1/R_V
k = int(np.argmin(np.abs(za - zlim)))
ax.scatter([za[k]], [red[k]], s=260, facecolor="none", edgecolor="#1a7a3a", lw=2.6, zorder=6)
ax.annotate(f"printable design\n$\\zeta_{{\\rm all}}{{=}}1/R_V{{=}}{zlim:.2f}$",
            (za[k], red[k]), xytext=(za[k] + 0.5, red[k] - 6.5), fontsize=9.5, color="#1a7a3a",
            arrowprops=dict(arrowstyle="->", color="#1a7a3a"))

ax.fill_between([za.min() - 0.2, za.max() + 0.6], red_unc, red_unc + 6, color="#8a5a00", alpha=0.05)
ax.set_xlabel(r"allowed tow curvature  $\zeta_{\rm all}=1/R$  (tighter turns $\rightarrow$)")
ax.set_ylabel(r"far-field reduction  $J_0/J$")
ax.set_xlim(za.min() - 0.2, za.max() + 0.6); ax.set_ylim(0, red_unc + 6)
ax.legend(fontsize=9.5, loc="center right")
ax.grid(alpha=0.22)
internal_ticks(ax)
fig.tight_layout()
fig.savefig("results/cloak_front.pdf", bbox_inches="tight")
import os
_pv = os.environ.get("FIGPREVIEW")
if _pv:
    fig.savefig(_pv, dpi=110, bbox_inches="tight")
print("front points:", [(round(float(x), 2), round(float(y), 1)) for x, y in zip(za, red)])
print("saved results/cloak_front.pdf")
