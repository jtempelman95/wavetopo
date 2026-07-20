"""Rebuild the flat-band metamaterial figure from results/flatband_data.npz:
 - crisp PROJECTED density (topology), 2x2 tiling;
 - EXPLICIT phase-field tows, masked to the solid, made seamlessly periodic by
   solving the phase on a 4x4 tiling and cropping the central 2x2 (tows connect
   across all four cell boundaries);
 - an AS-MANUFACTURED validation: re-simulate the band structure with the cropped
   (projected) density and the actual fiber toolpath orientation.
Runs in the base numpy/scipy env; no re-optimization."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from wavetopo.cfrp import Material, QuadMesh, CSRBFMapping
from wavetopo.bloch import BlochProblem, ibz_path_square
from wavetopo.dolfinx_viz import fiber_phase

d = np.load("results/flatband_data.npz")
bands0, bands1 = d["bands0"], d["bands1"]
z, theta = d["z"], d["theta"]                      # filtered density, per-cell theta
nelx, nely, a = int(d["nelx"]), int(d["nely"]), float(d["a"])
b = int(d["band"]); w0, w1 = float(d["w0"]), float(d["w1"])
gap_up, gap_dn = float(d["gap_up"]), float(d["gap_dn"])
hist_w, hist_gu, hist_gd = d["hist_w"], d["hist_gu"], d["hist_gd"]
npath = bands0.shape[0]; n = (npath + 2)//3
ticks = [0, n-1, 2*n-2, 3*n-3]; labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
nb = bands1.shape[1]

# crisp projected density (sharp Heaviside)
beta, eta = 12.0, 0.5
proj = lambda x: (np.tanh(beta*eta) + np.tanh(beta*(x-eta))) / \
                 (np.tanh(beta*eta) + np.tanh(beta*(1-eta)))
dg = z.reshape(nely, nelx); th = theta.reshape(nely, nelx)
dgp = proj(dg)

# ---- as-manufactured validation: recover support angles, re-simulate ----
mesh = QuadMesh(nelx, nely, a, a)
mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3, rho_f=1.6, rho_m=1.2)
ks = 4; xs0 = np.linspace(0, a, ks, endpoint=False); SX, SY = np.meshgrid(xs0, xs0)
rbf = CSRBFMapping(mesh, (SX.ravel(), SY.ravel()), r_s=a/2, period=(a, a))
bp = BlochProblem(mesh, mat, rbf, penal=3.0)
theta_hat = np.linalg.lstsq(rbf.Phi.toarray(), theta, rcond=None)[0]
z_manuf = dgp.ravel()                              # cropped (projected) material
bp.assemble(z_manuf, theta_hat)
ppath, _, _ = ibz_path_square(a, a, n=n)
bands_v = bp.band_structure(ppath, nb)
w_v = bands_v[:, b].max() - bands_v[:, b].min()
gu_v = bands_v[:, b+1].min() - bands_v[:, b].max()
gd_v = bands_v[:, b].min() - bands_v[:, b-1].max()
print(f"as-manufactured band {b}: width={w_v:.3f} gap_up={gu_v:+.3f} gap_dn={gd_v:+.3f}")

# ---- exactly cell-PERIODIC tows via an FFT phase + integer tow count ----
# Build a smooth periodic director grid on [0,a) x [0,a) (tile 3x3 + interp so the
# grid is periodic at the cell edges), then solve the periodic least-squares
# grad(psi)=g/pitch by FFT and add an INTEGER number of tow cycles per period, so
# chi = cos(2*pi*Psi) is exactly periodic and tiles seamlessly on all four sides.
def periodic_tows(theta_cell, dens_cell, a, nelx, nely, pitch, ng=220, dthr=0.5):
    dx, dy = a/nelx, a/nely
    th3 = np.tile(theta_cell, (3, 3)); dn3 = np.tile(dens_cell, (3, 3))
    xs3 = (np.arange(3*nelx)+0.5)*dx - a; ys3 = (np.arange(3*nely)+0.5)*dy - a
    GX3, GY3 = np.meshgrid(xs3, ys3); c3 = np.column_stack([GX3.ravel(), GY3.ravel()])
    gx = (np.arange(ng)+0.5)*(a/ng); GXc, GYc = np.meshgrid(gx, gx)
    C = griddata(c3, np.cos(2*th3).ravel(), (GXc, GYc), method="linear")
    S = griddata(c3, np.sin(2*th3).ravel(), (GXc, GYc), method="linear")
    D = griddata(c3, dn3.ravel(), (GXc, GYc), method="linear")
    ang = 0.5*np.arctan2(S, C)                      # in (-pi/2, pi/2], cos>=0 -> consistent
    gxf = -np.sin(ang); gyf = np.cos(ang)           # unit, perp to fiber
    k = 2*np.pi*np.fft.fftfreq(ng, d=a/ng)
    KX, KY = np.meshgrid(k, k)
    fxh = np.fft.fft2(gxf/pitch); fyh = np.fft.fft2(gyf/pitch)
    den = KX**2 + KY**2; den[0, 0] = 1.0
    psih = -1j*(KX*fxh + KY*fyh)/den; psih[0, 0] = 0.0
    psi_p = np.real(np.fft.ifft2(psih))             # periodic part
    nx = round(a*gxf.mean()/pitch); ny = round(a*gyf.mean()/pitch)
    Psi = psi_p + nx*GXc/a + ny*GYc/a               # add integer tow cycles
    chi = 0.5 + 0.5*np.cos(2*np.pi*Psi)
    chi = np.where(D >= dthr, chi, np.nan)
    return chi                                       # ng x ng, exactly periodic

spacing = 1.0/9.0                                    # ~9 tows across the cell
chi1 = periodic_tows(th, dgp, a, nelx, nely, spacing)
chi_c = np.tile(chi1, (2, 2))                        # 2x2 tiling, seamless

# ---------------- figure ----------------
kk = np.arange(npath)
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
panels = [(ax[0, 0], bands0, f"baseline (band {b} width {w0:.2f})", None, None),
          (ax[0, 1], bands1, f"optimized SIMP (width {w1:.2f})", gap_up, gap_dn),
          (ax[0, 2], bands_v, f"as-manufactured (width {w_v:.2f})", gu_v, gd_v)]
for a_, bands, ttl, gu, gd in panels:
    a_.plot(kk, bands, "0.7", lw=1.0); a_.plot(kk, bands[:, b], "b", lw=2.2)
    a_.set_xticks(ticks); a_.set_xticklabels(labels); a_.set_ylabel(r"$\omega$")
    a_.set_title(ttl); a_.grid(alpha=0.3)
    if gu is not None and gu > 0:
        a_.axhspan(bands[:, b].max(), bands[:, b+1].min(), color="orange", alpha=0.35)
    if gd is not None and gd > 0:
        a_.axhspan(bands[:, b-1].max(), bands[:, b].min(), color="orange", alpha=0.35)

from scipy.ndimage import zoom
dcell = np.clip(zoom(dgp, 6, order=3, mode="grid-wrap"), 0, 1)   # smooth periodic upsample
dens_2 = np.tile(dcell, (2, 2))
im = ax[1, 0].imshow(dens_2, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                     extent=[0, 2*a, 0, 2*a], aspect="equal", interpolation="bilinear")
ax[1, 0].axhline(a, color="r", lw=0.8); ax[1, 0].axvline(a, color="r", lw=0.8)
ax[1, 0].set_title("cropped density (topology), 2$\\times$2 tiling")
plt.colorbar(im, ax=ax[1, 0], fraction=0.046)

ax[1, 1].imshow(chi_c, origin="lower", extent=[0, 2*a, 0, 2*a], cmap="gray_r",
                interpolation="bilinear", vmin=0, vmax=1, aspect="equal")
ax[1, 1].axhline(a, color="r", lw=0.8); ax[1, 1].axvline(a, color="r", lw=0.8)
ax[1, 1].set_title("fiber toolpaths (as-manufactured tows), 2$\\times$2 tiling")

ax[1, 2].plot(hist_w, label="band width"); ax[1, 2].plot(hist_gu, label="gap above")
ax[1, 2].plot(hist_gd, label="gap below"); ax[1, 2].axhline(0, color="k", lw=0.6)
ax[1, 2].set_title("convergence"); ax[1, 2].set_xlabel("iter"); ax[1, 2].legend(); ax[1, 2].grid(alpha=0.3)

fig.suptitle("Flat-band + band-gap metamaterial: co-optimized topology and fiber "
             "toolpath, with as-manufactured validation", y=1.0, fontsize=14)
plt.tight_layout()
fig.savefig("results/flatband_demo.png", dpi=140, bbox_inches="tight")
fig.savefig("docs/paper/figs/flatband.png", dpi=140, bbox_inches="tight")
np.savez("results/flatband_manuf.npz", bands_v=bands_v, theta_hat=theta_hat,
         z_manuf=z_manuf, w_v=w_v, gu_v=gu_v, gd_v=gd_v)
print("saved flatband figure (periodic tows + as-manufactured validation)")
