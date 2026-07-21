"""Rebuild the joint-routing figure from results/dolfinx_guide_joint_data.npz as a
clean 2x2, WITHOUT re-optimizing.

The driver's inline figure used a 3x2 gridspec with the two field panels spanning
both columns.  Because the domain is 8x4 and the axes are aspect-equal, those
full-width axes were far wider than the data, leaving a large dead margin and an
awkward stagger.  A 2x2 grid matches the 2:1 domain aspect, and the toolpath and
director panels get a proportionally larger share -- which is where the detail is.

    PYTHONPATH=/home/jrt/wavetopo /home/jrt/miniforge3/bin/python3 \
        examples/dfx_guide_figure.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
from wavetopo import figlib as _F; _F.journal()  # publication typography
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from wavetopo import figlib as F
from wavetopo.dolfinx_viz import plot_toolpaths_phase, plot_director_field

d = np.load("results/dolfinx_guide_joint_data.npz")
tri = mtri.Triangulation(d["trix"], d["triy"], d["tris"])
cent, th = d["cent"], d["thopt"]
m0, m1 = d["m0"], d["m1"]
JX, JY, RJ = float(d["JX"]), float(d["JY"]), float(d["RJ"])
GUARD = float(d["GUARD"]); EXIT = d["EXIT"]
gain, guard, prot = float(d["gain"]), float(d["guard"]), float(d["prot"])
confd0, confd1 = float(d["confd0"]), float(d["confd1"])
zmax, zeta_all = float(d["zmax"]), float(d["zeta_all"])
Ex0, Eg0 = float(d["Ex0"]), float(d["Eg0"])

# corridor geometry (as in the driver)
Lx, Ly = 8.0, 4.0
AMP, SIG, HALFW = 1.05, 1.60, 0.32
xs = np.linspace(0, Lx, 900)
cy = JY + AMP*np.exp(-((xs - JX)/SIG)**2)
tc = np.linspace(0, 2*np.pi, 240)
vmax = np.percentile(m1, 99.0)


def marks(a_, corridor=True):
    if corridor:
        a_.plot(xs, cy+HALFW, "c--", lw=1.0, alpha=0.9)
        a_.plot(xs, cy-HALFW, "c--", lw=1.0, alpha=0.9)
    a_.plot(JX+RJ*np.cos(tc), JY+RJ*np.sin(tc), "w-", lw=1.1)
    a_.plot(JX+(RJ+GUARD)*np.cos(tc), JY+(RJ+GUARD)*np.sin(tc), "r:", lw=1.4)
    ex, ey, er = EXIT
    a_.add_patch(plt.Rectangle((ex-er, ey-er), 2*er, 2*er, ec="#bfefff",
                               fc="none", lw=1.4, ls="--"))
    a_.set_aspect("equal"); a_.set_xlim(0, Lx); a_.set_ylim(0, Ly)


fig, ax = plt.subplots(2, 2, figsize=(17, 10.2))

tp = ax[0, 0].tripcolor(tri, m0, cmap="magma", vmax=vmax, shading="gouraud")
marks(ax[0, 0])
ax[0, 0].set_title(f"straight fibers: the beam diffracts onto the joint\n"
                   f"downstream confinement {100*confd0:.0f}%, "
                   f"exit/guard = {Ex0/Eg0:.4f}", fontsize=11)
plt.colorbar(tp, ax=ax[0, 0], fraction=0.028, pad=0.01)

tp = ax[0, 1].tripcolor(tri, m1, cmap="magma", vmax=vmax, shading="gouraud")
marks(ax[0, 1])
ax[0, 1].set_title(f"orientation-CHANNELLED: exit {gain:.0f}$\\times$, guard "
                   f"{guard:.2f}$\\times$\ndownstream confinement "
                   f"{100*confd0:.0f}%$\\to${100*confd1:.0f}%, "
                   f"protection {prot:.0f}$\\times$", fontsize=11)
plt.colorbar(tp, ax=ax[0, 1], fraction=0.028, pad=0.01)

# finer tow pitch and grid: this panel is meant to be read closely
plot_toolpaths_phase(ax[1, 0], cent, th, (0, Lx), (0, Ly), holes=(JX, JY, RJ),
                     spacing=0.11, n=460)
ax[1, 0].plot(xs, cy, "c--", lw=1.1, alpha=0.85)
ax[1, 0].plot(JX+(RJ+GUARD)*np.cos(tc), JY+(RJ+GUARD)*np.sin(tc), "r:", lw=1.4)
ax[1, 0].set_aspect("equal"); ax[1, 0].set_xlim(0, Lx); ax[1, 0].set_ylim(0, Ly)
ax[1, 0].set_title(f"fiber toolpaths   (max$|\\zeta|$={zmax:.2f} $\\leq$ "
                   f"$1/R_{{\\rm tow}}$={zeta_all:.2f})", fontsize=11)

lc = plot_director_field(ax[1, 1], cent, th, (0, Lx), (0, Ly),
                         holes=(JX, JY, RJ), n=52, lw=1.9)
ax[1, 1].plot(xs, cy, "c--", lw=1.1, alpha=0.85)
ax[1, 1].set_aspect("equal"); ax[1, 1].set_xlim(0, Lx); ax[1, 1].set_ylim(0, Ly)
ax[1, 1].set_title("anisotropy director field", fontsize=11)
cb = plt.colorbar(lc, ax=ax[1, 1], fraction=0.028, pad=0.01,
                  ticks=[0, np.pi/2, np.pi])
cb.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])

F.layout(fig, "Channelling elastic energy along a curved corridor around a "
              "through-hole joint (localised aperture source, leakage penalised)",
         y=0.97)
F.save_pair(fig, "results/dolfinx_guide_joint.png", "docs/paper/figs/dfx_guide_joint.png")
print("saved guide figure as a 2x2 (director n=52, tow pitch 0.11)")
