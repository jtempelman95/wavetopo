"""
Multi-functional metamaterial: does co-designing TOOLPATH with GEOMETRY beat
geometry alone?

Reads the sweep written by examples/multifunc_demo.py and draws
  (a) the stiffness-vs-gap Pareto scatter, co-design against geometry-only;
  (b) the cell (density + fiber directors) for the best co-designed point;
  (c) its band structure with the complete gap shaded;
  (d) the convergence of both objectives.

Both fronts start from the SAME perturbed cell and get the same iteration
budget, so the only difference between a pair of points is whether theta was
free.

    PYTHONPATH=. python examples/multifunc_figure.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from wavetopo import figlib as _F; _F.journal()  # publication typography
import matplotlib.pyplot as plt

from wavetopo import figlib as F

from wavetopo.dolfinx_viz import plot_director_field

RUNS = []
for f in sorted(glob.glob("results/multifunc_*.npz")):
    tag = os.path.basename(f)[10:-4]
    if tag == "smoke":
        continue
    d = np.load(f)
    RUNS.append(dict(tag=tag, d=d, geom=bool(d["geom_only"]),
                     w=float(d["wstiff"]), C=float(d["C_bulk"]),
                     gap=float(d["gap"]), C0=float(d["C_bulk0"]),
                     gap0=float(d["gap0"])))
if not RUNS:
    raise SystemExit("no results/multifunc_*.npz -- run examples/multifunc_demo.py")

co = [r for r in RUNS if not r["geom"]]
gm = [r for r in RUNS if r["geom"]]
co.sort(key=lambda r: r["C"]); gm.sort(key=lambda r: r["C"])

print(f"{'run':12} {'w_stiff':>8} {'C_bulk':>8} {'gap':>8}  gap open?")
for r in sorted(RUNS, key=lambda r: (r["geom"], r["w"])):
    print(f"  {r['tag']:10} {r['w']:8.1f} {r['C']:8.3f} {r['gap']:+8.4f}  "
          f"{'YES' if r['gap'] > 0 else 'no'}")

# best co-designed point that actually has a gap
best = max([r for r in co if r["gap"] > 0] or co, key=lambda r: r["C"])
d = best["d"]
nelx, nely, a = int(d["nelx"]), int(d["nely"]), float(d["a"])
b = int(d["band"])

fig = plt.figure(figsize=(17, 9.6))
gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.26)

# ---- (a) Pareto ------------------------------------------------------- #
A = fig.add_subplot(gs[0, 0])
A.plot([r["C"] for r in gm], [r["gap"] for r in gm], "o--", color="0.45",
       ms=9, lw=1.6, label="geometry only ($\\theta$ frozen)")
A.plot([r["C"] for r in co], [r["gap"] for r in co], "o-", color="tab:blue",
       ms=10, lw=2.0, label="co-design ($z$ and $\\theta$)")
for r in RUNS:
    A.annotate(f"$w$={r['w']:g}", (r["C"], r["gap"]), fontsize=8,
               xytext=(4, 4), textcoords="offset points",
               color="0.35" if r["geom"] else "tab:blue")
A.axhline(0, color="k", lw=0.9)
A.fill_between([min(r["C"] for r in RUNS)*0.9, max(r["C"] for r in RUNS)*1.05],
               0, A.get_ylim()[1], color="tab:green", alpha=0.07)
A.text(0.02, 0.96, "complete gap open", transform=A.transAxes, fontsize=9,
       va="top", color="tab:green")
A.set_xlabel("homogenised bulk stiffness $C_{\\rm bulk}$")
A.set_ylabel("complete band gap")
A.set_title("stiffness / gap trade-off", fontsize=11)
A.grid(alpha=0.3); A.legend(fontsize=9, loc="lower left")

# ---- (b) the co-designed cell ----------------------------------------- #
B = fig.add_subplot(gs[0, 1])
dg = d["z"].reshape(nely, nelx)
B.imshow(np.tile(dg, (2, 2)), origin="lower", cmap="gray_r", vmin=0, vmax=1,
         extent=[0, 2*a, 0, 2*a], aspect="equal", interpolation="bilinear")
B.axhline(a, color="r", lw=0.8); B.axvline(a, color="r", lw=0.8)
B.set_title(f"co-designed cell ($2\\times2$)\n$C_{{\\rm bulk}}$={best['C']:.2f}, "
            f"gap={best['gap']:+.3f}", fontsize=11)

# ---- (c) fiber directors ---------------------------------------------- #
C_ = fig.add_subplot(gs[0, 2])
cx = (np.arange(nelx)+0.5)*a/nelx
cy = (np.arange(nely)+0.5)*a/nely
CX, CY = np.meshgrid(cx, cy)
cent = np.column_stack([CX.ravel(), CY.ravel()])
lc = plot_director_field(C_, cent, d["theta"], (0, a), (0, a), n=26, lw=2.0)
C_.set_aspect("equal"); C_.set_xlim(0, a); C_.set_ylim(0, a)
C_.set_title("fiber orientation (the second lever)", fontsize=11)
cb = plt.colorbar(lc, ax=C_, fraction=0.046, pad=0.02, ticks=[0, np.pi/2, np.pi])
cb.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])

# ---- (d) bands --------------------------------------------------------- #
D = fig.add_subplot(gs[1, 0])
bands0, bands1 = d["bands0"], d["bands1"]
npath = bands1.shape[0]; n = (npath+2)//3
kk = np.arange(npath)
D.plot(kk, bands1, "0.75", lw=1.0)
D.plot(kk, bands1[:, b], "b", lw=2.2)
D.plot(kk, bands1[:, b+1], "b", lw=2.2)
if best["gap"] > 0:
    D.axhspan(bands1[:, b].max(), bands1[:, b+1].min(), color="orange", alpha=0.35)
D.set_xticks([0, n-1, 2*n-2, 3*n-3])
D.set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
D.set_ylabel(r"$\omega$"); D.grid(alpha=0.3)
D.set_title(f"co-designed bands (gap {best['gap']:+.3f})", fontsize=11)

# ---- (e) convergence --------------------------------------------------- #
E = fig.add_subplot(gs[1, 1])
E.plot(d["hist_C"], "tab:red", lw=1.7, label="$C_{\\rm bulk}$")
E.set_xlabel("iteration"); E.set_ylabel("$C_{\\rm bulk}$", color="tab:red")
E.tick_params(axis="y", labelcolor="tab:red")
E2 = E.twinx()
E2.plot(d["hist_gap"], "tab:blue", lw=1.7, label="gap")
E2.axhline(0, color="k", lw=0.8, ls=":")
E2.set_ylabel("gap", color="tab:blue"); E2.tick_params(axis="y", labelcolor="tab:blue")
E.set_title("both objectives rise together", fontsize=11); E.grid(alpha=0.3)

# ---- (f) polarity ------------------------------------------------------ #
F = fig.add_subplot(gs[1, 2])
par = d["parity"]; wg = d["w_gamma"]
cols = ["tab:red" if p > 0 else "tab:blue" for p in par]
F.barh(np.arange(len(par)), par, color=cols)
for i, (p, w) in enumerate(zip(par, wg)):
    F.text(0.02 if p < 0 else -0.02, i, f"$\\omega$={w:.2f}",
           va="center", ha="left" if p < 0 else "right", fontsize=8)
F.axvline(0, color="k", lw=0.9)
F.set_yticks(np.arange(len(par))); F.set_yticklabels([f"band {i}" for i in range(len(par))])
F.set_xlabel(r"inversion parity  $\langle\phi,\mathcal{I}\phi\rangle$")
F.set_xlim(-1.1, 1.1)
F.set_title("band polarity at $\\Gamma$\n(red even, blue odd)", fontsize=11)
F.grid(alpha=0.3, axis="x")

fig.suptitle("Multi-functional metamaterial: stiff AND gapped, by co-designing "
             "geometry with toolpath", y=0.98, fontsize=13)
F.save_pair(fig, "results/multifunc.png", "docs/paper/figs/multifunc.png")
print("\nsaved results/multifunc.png and docs/paper/figs/multifunc.png")
