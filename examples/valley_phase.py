"""
Topological phase diagram / optimization of the triangular-rod valley crystal.

The rod rotation alpha is the tunable Dirac mass: the complete valley gap closes
at alpha=30deg (the C3v-symmetric Dirac point = phase boundary) and opens on
either side, with the valley Chern flipping sign across it.  Maximizing the
complete gap over alpha (and rod size R) is a concrete optimization of the
topological phase -- a larger gap gives a more localized, more robust edge mode.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topoopt.valley_cell import triangle_cell
from topoopt.honeycomb_mesh import HoneycombBloch
from topoopt.scalar import MaterialSH


def complete_gap(al, R, muf, h=0.05, nk=17, pair=1):
    mat = MaterialSH(mu_L=muf, mu_T=muf, mu_m=1.0, rho_f=1.6, rho_m=1.2)
    m = triangle_cell(al, R=R, h=h); hb = HoneycombBloch(m, mat); hb.assemble()
    b1, b2 = hb.recip()
    W = np.array([[hb.bands_at_k(*(s*b1+t*b2), pair+2) for t in
                   np.linspace(0, 1, nk, endpoint=False)]
                  for s in np.linspace(0, 1, nk, endpoint=False)]).reshape(-1, pair+2)
    return W[:, pair+1].min() - W[:, pair].max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muf", type=float, default=100.0)
    ap.add_argument("--out", default="results/valley_phase.png")
    args = ap.parse_args()
    alphas = np.arange(6, 55, 3.0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for R, c in [(0.44, "darkred"), (0.40, "steelblue")]:
        g = []
        for al in alphas:
            try:
                g.append(complete_gap(al, R, args.muf))
            except Exception:
                g.append(np.nan)
        g = np.array(g)
        ax[0].plot(alphas, g, "o-", color=c, label=f"R={R}")
        if R == 0.44:
            best = alphas[np.nanargmax(g)]
            print(f"R={R}: max complete gap {np.nanmax(g):.3f} at alpha={best:.0f}"
                  f" ; Dirac (gap<=0) near alpha=30")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].axvline(30, color="0.5", ls=":", lw=1)
    ax[0].annotate("Dirac point\n(phase boundary)", (30, 0.05), fontsize=9,
                   ha="center")
    ax[0].set_xlabel(r"rod rotation $\alpha$ (deg)")
    ax[0].set_ylabel("complete valley gap")
    ax[0].set_title("Topological phase: gap closes at the Dirac mass = 0")
    ax[0].legend()
    # schematic of the two mirror-partner domains + valley Chern signs
    ax[1].axis("off")
    ax[1].text(0.05, 0.85, r"$\alpha < 30^\circ$:  $m>0$,  "
               r"$C_v^K=+\frac{1}{2}$", fontsize=13, transform=ax[1].transAxes)
    ax[1].text(0.05, 0.65, r"$\alpha > 30^\circ$:  $m<0$,  "
               r"$C_v^K=-\frac{1}{2}$", fontsize=13, transform=ax[1].transAxes)
    ax[1].text(0.05, 0.40, "domain wall (15$^\\circ$|45$^\\circ$):\n"
               r"$|\Delta C_v| = 1 \Rightarrow$ one kink mode per valley,"
               "\ntraversing the complete gap (see ribbon).",
               fontsize=12, transform=ax[1].transAxes)
    ax[1].text(0.05, 0.12, r"$m(\alpha)\propto \cos(3\alpha)$ near the Dirac point",
               fontsize=12, transform=ax[1].transAxes)
    fig.suptitle("Triangular-rod valley crystal: rod orientation as the tunable "
                 "topological Dirac mass", y=1.0)
    plt.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
