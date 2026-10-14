"""Why the frequency-focused objective grows off-load-path 'free-hanging' members.
The eigenvalue objective's element sensitivity dlam_1/drho = phi^T K_e phi - lam_1 phi^T
M_e phi CHANGES SIGN: it is negative at the mode's tip ANTINODE (mass there lowers
omega_1), so the optimizer scoops mass off the tip (b, the bright antinode).  Under a
FIXED volume budget that scooped material has to go somewhere, and the only places with
non-negative sensitivity are the mode's low-motion root corners -- so it re-deposits there
as clamp-grounded OUTRIGGERS.  They are dark in (a) the static load path (zero service
load).  (c) Trimming the surplus lowest-static-load-first: the first ~10% of area is
neutral to BOTH omega_1 and stiffness -- that is the outrigger surplus.  The compliance
objective has no sign flip (more material on a load path always helps), so it never grows
them; that is why the part we build is the w=0 A-frame.  All clamp-grounded, no islands."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from examples.drivers.cfrp_bracket_modal import setup, HOLES, LOAD_HOLE

d = np.load("results/bracket_outrigger_energy.npz")
cx, cy, A = d["cx"], d["cy"], d["area"]; solid = d["solid"]
sse, mse = d["sse"], d["mse"]; LX, LY = float(d["LX"]), float(d["LY"])
npx, tris = d["npx"], d["tris"]
mesh, mat, P = setup(0.025)
tri = mtri.Triangulation(npx[:, 0], npx[:, 1], tris)
nn = npx.shape[0]; wn = np.zeros(nn)
np.add.at(wn, tris.ravel(), np.repeat(A, 3)); wn = np.maximum(wn, 1e-12)


def e2n(ef):
    v = np.zeros(nn); np.add.at(v, tris.ravel(), np.repeat(A, 3) * np.repeat(ef, 3))
    return v / wn


NG = 640
gx = np.linspace(0, LX, NG); gy = np.linspace(0, LY, int(NG * LY / LX))
GX, GY = np.meshgrid(gx, gy)
Dg = np.asarray(mtri.LinearTriInterpolator(tri, e2n(solid.astype(float)))(GX, GY).filled(0))
void = Dg < 0.5
sse_d = (sse / A); mse_d = (mse / A)
sse_d[~solid] = 0; mse_d[~solid] = 0
mL = sse[solid].sum() / A[solid].sum(); mM = mse[solid].sum() / A[solid].sum()

plt.rcParams.update({"font.family": "serif", "font.size": 10.5})
fig = plt.figure(figsize=(16, 4.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.16,
                      left=0.02, right=0.98, top=0.86, bottom=0.14)


def frame(a):
    for i, (hx, hy, hr) in enumerate(HOLES):
        a.add_patch(plt.Circle((hx, hy), hr, fc="white",
                    ec="#b3121b" if i == LOAD_HOLE else "k", lw=1.2, zorder=8))
    a.plot([0, 0], [0.04, LY - 0.04], color="#b3121b", lw=4, zorder=9)
    a.set_xlim(-0.05, LX + 0.05); a.set_ylim(-0.05, LY + 0.05)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])


def emap(ax, dens, mean, title, cmap):
    g = np.asarray(mtri.LinearTriInterpolator(tri, e2n(dens))(GX, GY).filled(0))
    im = ax.imshow(np.ma.array(g / mean, mask=void), origin="lower",
                   extent=[0, LX, 0, LY], cmap=cmap, norm=LogNorm(vmin=0.03, vmax=8),
                   interpolation="bilinear", zorder=2)
    frame(ax); ax.set_title(title, fontsize=10.5, loc="left")
    return im


a0 = fig.add_subplot(gs[0, 0])
im0 = emap(a0, sse_d, mL, "(a) static load path  $u^{T}Ku$: outriggers\ncarry no service load (dark)", "magma")
cb0 = fig.colorbar(im0, ax=a0, fraction=0.043, pad=0.02, extend="both")
cb0.set_label("static SE / mean", fontsize=8.5)
# annotate an outrigger as dead surplus
a0.annotate("outrigger\n(0% static load)", xy=(0.60, LY - 0.28), xytext=(1.35, LY - 0.55),
            fontsize=8.5, color="#ffd166", ha="left",
            arrowprops=dict(arrowstyle="->", color="#ffd166", lw=1.4))

# (b) fundamental mode amplitude -- the antinode is where mass is scooped
phi = d["phi"]; mn = np.sqrt(phi[0::2] ** 2 + phi[1::2] ** 2); mn = mn / mn.max()
a1 = fig.add_subplot(gs[0, 1])
mg = np.asarray(mtri.LinearTriInterpolator(tri, mn)(GX, GY).filled(0.0))
im1 = a1.imshow(np.ma.array(mg, mask=void), origin="lower", extent=[0, LX, 0, LY],
                cmap="viridis", vmin=0, vmax=1, interpolation="bilinear", zorder=2)
frame(a1)
a1.set_title("(b) fundamental mode $|\\varphi_1|$: mass is\nscooped off the tip antinode",
             fontsize=10.5, loc="left")
ia = int(np.argmax(mn)); a1.annotate("antinode\n(mass removed)",
            xy=(npx[ia, 0], npx[ia, 1]), xytext=(1.5, 0.30), fontsize=8.5, color="w",
            ha="left", arrowprops=dict(arrowstyle="->", color="w", lw=1.4))
cb1 = fig.colorbar(im1, ax=a1, fraction=0.043, pad=0.02)
cb1.set_label("$|\\varphi_1|$ (normalized)", fontsize=8.5)

# trim curve
order = np.argsort(sse_d[solid]); si = np.flatnonzero(solid)
cum = np.cumsum(A[si[order]]) / A[solid].sum()
ks = np.linspace(0.02, 0.55, 16); o1, cc = [], []
c0, _ = P.static(d["y"], d["th"], want_grad=False)
lam0 = d["lam1"]
for k in ks:
    y2 = d["y"].copy(); y2[si[order[cum <= k]]] = 1e-3
    lt, _ = P.modal(y2, d["th"], want_grad=False); ct, _ = P.static(y2, d["th"], want_grad=False)
    o1.append(np.sqrt(lt[0] / lam0)); cc.append(c0 / ct)
a2 = fig.add_subplot(gs[0, 2])
a2.plot(ks * 100, o1, "-o", color="#6a2c91", lw=2.3, ms=5, label=r"$\omega_1$ retained")
a2.plot(ks * 100, cc, "-s", color="#c9721f", lw=2.3, ms=5,
        label=r"stiffness retained ($c_0/c$)")
a2.axvspan(0, 10, color="#7a7", alpha=0.13)
a2.text(5, 0.56, "outrigger\nsurplus\n(neutral)", ha="center", fontsize=8.5,
        color="#2a7a2a")
a2.text(34, 0.62, "load-bearing core:\n$\\omega_1$ and stiffness\nfall together", ha="center",
        fontsize=8.5, color="#555")
a2.set_xlabel("solid area trimmed, lowest static-load first (%)")
a2.set_ylabel("fraction retained"); a2.set_ylim(0.5, 1.03); a2.set_xlim(0, 55)
a2.legend(fontsize=9.5, loc="upper right", frameon=False); a2.grid(alpha=0.25)
a2.set_title("(c) trim the idle-looking material lowest-first", fontsize=10.5, loc="left")
for s in ("top", "right"):
    a2.spines[s].set_visible(False)

fig.suptitle("Why the frequency objective grows the 'free-hanging' outriggers: it scoops "
             "mass off the mode's tip antinode (b); the fixed volume budget re-deposits it "
             "as clamp-grounded root outriggers that carry no static load (a) and are\n"
             "neutral to $\\omega_1$ and stiffness alike (c) --- surplus the compliance "
             "A-frame we build never grows", fontsize=11.3, y=0.995)
fig.savefig("results/cfrp_bracket_outrigger.png", dpi=170, bbox_inches="tight",
            facecolor="white")
print(f"neutral (first 10% area): omega1 retained {o1[np.argmin(np.abs(ks-0.10))]:.3f}, "
      f"stiffness retained {cc[np.argmin(np.abs(ks-0.10))]:.3f}")
print("saved results/cfrp_bracket_outrigger.png")
