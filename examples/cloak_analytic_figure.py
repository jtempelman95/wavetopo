"""Why the ideal cloak is unmanufacturable: the analytical transformation cloak
(Pendry/Norris radial map) requires a fiber field whose curvature |zeta| DIVERGES at
the void rim (the tows must wrap the void infinitely tightly), while our curvature-aware
optimization gives a cloak with bounded, printable |zeta| that still restores the far
field.  This is the fiber-steering manifestation of Norris's result that the exact cloak
needs singular material anisotropy at the inner surface."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

d = np.load("/tmp/claude-1000/-home-jrt/449caa35-5093-4f38-bb0f-a3bd611efc21/"
            "scratchpad/cloak_analytic.npz")
X, Y, TH, Z, RP = d["X"], d["Y"], d["TH"], d["Z"], d["RP"]
Rv, Rdes, zeta_all = float(d["Rv"]), float(d["Rdes"]), float(d["zeta_all"])
FEAS = 1.70          # our curvature-optimized cloak solid-max |zeta| (feasible)

plt.rcParams.update({"font.family": "serif", "font.size": 11})
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))

# (a) analytical director field, line segments coloured by |zeta| (log)
a = ax[0]
step = 26
xs = X[::step, ::step]; ys = Y[::step, ::step]; th = TH[::step, ::step]
zz = Z[::step, ::step]; rp = RP[::step, ::step]
m = (rp > Rv) & (rp < Rdes) & np.isfinite(zz)
xs, ys, th, zz = xs[m], ys[m], th[m], zz[m]
L = 0.075
segs = np.stack([np.c_[xs - L * np.cos(th), ys - L * np.sin(th)],
                 np.c_[xs + L * np.cos(th), ys + L * np.sin(th)]], axis=1)
lc = LineCollection(segs, cmap="inferno",
                    norm=LogNorm(vmin=0.5, vmax=50), linewidths=1.4)
lc.set_array(np.clip(zz, 0.5, 50)); a.add_collection(lc)
cb = fig.colorbar(lc, ax=a, fraction=0.046, pad=0.02, extend="max")
cb.set_label(r"$|\zeta|$", fontsize=12)
a.add_patch(plt.Circle((0, 0), Rv, fc="white", ec="k", lw=1.5, zorder=5))
a.add_patch(plt.Circle((0, 0), Rdes, fc="none", ec="#b3121b", ls="--", lw=1.5,
                       zorder=5))
a.set_xlim(-Rdes * 1.02, Rdes * 1.02); a.set_ylim(-Rdes * 1.02, Rdes * 1.02)
a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
a.set_title("(a) ideal transformation cloak: fibers must wrap the void\n"
            r"infinitely tightly --- $|\zeta|\!\to\!\infty$ at the rim", fontsize=11,
            loc="left")

# (b) |zeta| vs distance into shell: analytical divergence vs limit vs our feasible
a = ax[1]
fr = np.linspace(0.02, 0.88, 40)          # fraction into shell from the rim
rr = Rv + fr * (Rdes - Rv)
med = []; mx = []
for r in rr:
    band = (RP > Rv) & (RP < Rdes) & (np.abs(RP - r) < 0.02) & np.isfinite(Z)
    med.append(np.nanmedian(Z[band])); mx.append(np.nanmax(Z[band]))
a.fill_between(fr, med, mx, color="#c0392b", alpha=0.18, label="analytical (median–max)")
a.plot(fr, med, color="#c0392b", lw=2.4, label="ideal cloak $|\\zeta|$ (median)")
a.axhline(zeta_all, color="#333", ls="--", lw=1.6,
          label=fr"AFP limit $\zeta_{{\rm all}}=1/R_V={zeta_all:.2f}$")
a.axhline(FEAS, color="#1f6fb2", lw=2.4,
          label=f"our optimized cloak: $|\\zeta|_{{\\max}}={FEAS:.2f}$ (feasible, 21$\\times$)")
a.set_yscale("log"); a.set_ylim(0.3, 200); a.set_xlim(0, 0.9)
a.set_xlabel("fraction into the cloaking shell (0 = void rim, 1 = outer edge)")
a.set_ylabel(r"fiber curvature $|\zeta|$")
a.legend(fontsize=9.5, loc="upper right", frameon=False)
a.set_title("(b) the ideal cloak's curvature diverges at the rim; the\n"
            "optimized cloak stays under the printability limit", fontsize=11,
            loc="left")
a.grid(alpha=0.25, which="both")

fig.suptitle("The exact (transformation) cloak is unmanufacturable by fiber steering "
             "--- optimization finds a feasible cloak instead", fontsize=12.5, y=1.0)
fig.savefig("results/cloak_analytic.png", dpi=180, bbox_inches="tight",
            facecolor="white")
print("saved results/cloak_analytic.png")
