"""Rebuild the conforming-cloak figure from results/dolfinx_cloak_conforming_data.npz
WITHOUT re-optimizing.

Changes over the driver's inline figure: the anisotropy director panel is ZOOMED
to the design shell (the only place the orientation is allowed to vary, so the
full-plate view spent most of its area drawing identical horizontal segments) and
drawn on a much finer grid, with the toolpaths given a matching zoomed inset.

    PYTHONPATH=/home/jrt/wavetopo /home/jrt/miniforge3/bin/python3 \
        examples/dfx_cloak_figure.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from wavetopo import figlib as F
from wavetopo.dolfinx_viz import plot_toolpaths_phase, plot_director_field

d = np.load("results/dolfinx_cloak_conforming_data.npz")
tri = mtri.Triangulation(d["trix"], d["triy"], d["tris"])
trf = mtri.Triangulation(d["trfx"], d["trfy"], d["trfs"])
cent, th = d["cent"], d["thopt"]
mref, munc, mcl = d["mref"], d["munc"], d["mcl"]
J0, J1, red = float(d["J0"]), float(d["J1"]), float(d["red"])
CX, CY = float(d["CX"]), float(d["CY"])
RV, RC, R_DES = float(d["RV"]), float(d["RC"]), float(d["R_DES"])
Lx, Ly = float(d["Lx"]), float(d["Ly"])
hist = d["hist"]

tc = np.linspace(0, 2*np.pi, 240)
vmax = np.percentile(mref, 99.5)
# zoom window: the design shell plus a small margin
Z = R_DES + 0.35
zx, zy = (CX-Z, CX+Z), (CY-Z, CY+Z)

fig = plt.figure(figsize=(17.5, 9.6))
gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.22)


def rings(a_, zoom=False):
    a_.plot(CX+RV*np.cos(tc), CY+RV*np.sin(tc), "c-", lw=1.2)
    a_.plot(CX+R_DES*np.cos(tc), CY+R_DES*np.sin(tc), "r--", lw=1.5)
    if not zoom:
        a_.plot(CX+RC*np.cos(tc), CY+RC*np.sin(tc), "w:", lw=0.9)
    a_.set_aspect("equal")


for k, (t_, m_, ttl) in enumerate([
        (trf, mref, "reference (hole-free plate)"),
        (tri, munc, f"uncloaked void   $J_0={J0:.2e}$"),
        (tri, mcl, f"orientation-cloaked   $J_1={J1:.2e}$  ({red:.0f}$\\times$ less)")]):
    a_ = fig.add_subplot(gs[0, k])
    tp = a_.tripcolor(t_, m_, cmap="magma", vmax=vmax, shading="gouraud")
    rings(a_); a_.set_xlim(0, Lx); a_.set_ylim(0, Ly)
    a_.set_title(ttl, fontsize=10)
    plt.colorbar(tp, ax=a_, fraction=0.030, pad=0.01)

# --- toolpaths, zoomed to the shell ---
a_ = fig.add_subplot(gs[1, 0])
plot_toolpaths_phase(a_, cent, th, zx, zy, holes=(CX, CY, RV), spacing=0.10, n=420)
rings(a_, zoom=True); a_.set_xlim(*zx); a_.set_ylim(*zy)
a_.set_title(f"fiber toolpaths, zoomed to the design shell\n"
             f"(straight everywhere outside the red circle)", fontsize=10)

# --- director field, zoomed and dense: this is the panel worth detail ---
a_ = fig.add_subplot(gs[1, 1])
lc = plot_director_field(a_, cent, th, zx, zy, holes=(CX, CY, RV), n=42, lw=2.0)
rings(a_, zoom=True); a_.set_xlim(*zx); a_.set_ylim(*zy)
a_.set_title("anisotropy director field, zoomed\n(orientation designed ONLY inside "
             "the red shell)", fontsize=10)
cb = plt.colorbar(lc, ax=a_, fraction=0.030, pad=0.01, ticks=[0, np.pi/2, np.pi])
cb.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])

a_ = fig.add_subplot(gs[1, 2])
a_.plot(np.arange(1, len(hist)+1), hist, "b-", lw=1.7)
a_.set_xlabel("MMA iteration"); a_.set_ylabel(r"scatter reduction $J_0/J$")
a_.set_title(f"convergence from the straight-fiber baseline\n(final "
             f"{red:.0f}$\\times$, still rising)", fontsize=10)
a_.grid(alpha=0.3); a_.set_box_aspect(0.72)

F.layout(fig, "dolfinx in-plane orthotropic elastic cloak of a CONFORMING "
              "traction-free void: orientation designed only inside the shell",
         y=0.98)
fig.savefig("results/dolfinx_cloak_conforming_vec.png", dpi=145, bbox_inches="tight")
fig.savefig("docs/paper/figs/dfx_cloak_conf.png", dpi=145, bbox_inches="tight")
print(f"saved cloak figure (director zoomed to r<{Z:.2f} about ({CX},{CY}), n=42)")
