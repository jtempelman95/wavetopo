"""
GRAND EXAMPLE, stiffness+mass leg -- DENSITY co-design on top of the wave-optimal
orientation.  Orientation theta* (from dolfinx_panel.py) does the wave routing /
shielding / protection; here we freeze it and design the SIMP density field z to:

  * MINIMISE compliance      -> stiffer under the bolted service load
  * MINIMISE volume (mass)   -> lighter
  * PRESERVE the wave        -> the wave dJ/dz keeps material wherever |u| lives,
                                so material is only removed from dead zones

    f(z) = W_C (c/c0) + W_M (V/V0) - W_W (Jwave/|Jwave0|)      (minimised)

z is filtered (mesh-independent, no checkerboard) and mirror-symmetric about y=Ly/2.
Removing material where the load path and the wave both need it costs the first two
/ last term respectively, so the optimum keeps the load paths and beam corridors
solid and sheds the corners -- a lighter, stiffer, still-multifunctional panel.

    XDG_CACHE_HOME=/home/jrt/wavetopo/.fenics-cache PYTHONPATH=/home/jrt/wavetopo \
    /home/jrt/miniforge3/envs/dolfinx_complex/bin/python3 \
        examples/dolfinx/dolfinx_panel_density.py --warm results/dolfinx_panel_band_data.npz
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.spatial import cKDTree
import ufl
from dolfinx import fem

import dolfinx_panel as P
from wavetopo.dolfinx_static import StaticElastic
from wavetopo.cfrp_optimizer import MMA

ZMIN = 0.15
FILT_R = 0.16
W_C, W_M, W_W = 1.0, 1.2, 0.6


def density_filter(cent, r):
    tree = cKDTree(cent)
    rows, cols, vals = [], [], []
    for i, p in enumerate(cent):
        for j in tree.query_ball_point(p, r):
            rows.append(i); cols.append(j)
            vals.append(max(0.0, r - float(np.hypot(*(p - cent[j])))))
    H = sp.csr_matrix((vals, (rows, cols)), shape=(len(cent), len(cent)))
    return sp.diags(1.0/np.asarray(H.sum(1)).ravel()) @ H


def cell_mirror(cent, Ly):
    _, cmir = cKDTree(cent).query(np.column_stack([cent[:, 0], Ly - cent[:, 1]]))
    return cmir


def cell_areas(S):
    return fem.assemble_vector(fem.form(ufl.TestFunction(S)*ufl.dx)).array.copy()


def main(warm, iters, band, wc, wm, ww):
    freqs = band
    ews, doms = [], []
    for w in freqs:
        dom = P.rect_mesh(P.Lx, P.Ly, P.HMESH, holes=P.HOLES, h_hole=0.5*P.HMESH)
        ews.append(P._attach(dom, w)); doms.append(dom)
    ew0 = ews[0]
    ncell = ew0.theta.x.array.size
    cent = P.centroids(ew0)
    thopt = np.load(warm)["thopt"]
    print(f"mesh {ncell} cells; band={freqs}; theta* from {warm}", flush=True)

    # static: bolted at the holes, service load over the payload
    se = StaticElastic(ew0.domain, Ef1=131., Ef2=9., G12=5., nu12=0.27,
                       rho=1.6, rho_m=1.2, penal=3.0)
    se.clamp(P.HOLES, tol=0.6*P.HMESH)
    px, py, ph = P.PAY
    se.set_load(lambda x, y: np.zeros_like(x),
                lambda x, y: -1.0*((np.abs(x-px) < ph) & (np.abs(y-py) < ph)).astype(float))

    H = density_filter(cent, FILT_R)
    cmir = cell_mirror(cent, P.Ly)
    A = cell_areas(ew0.S)
    V0 = float(A.sum())
    zsym = lambda v: 0.5*(v + v[cmir])

    def set_design(zd):
        zph = H @ zd
        for ew in ews:
            ew.set_theta(thopt); ew.zfac.x.array[:] = zph
        se.set_theta(thopt); se.zfac.x.array[:] = zph
        return zph

    # straight-fiber baseline (theta=0, solid) -> the reference every "x" is against
    for ew in ews:
        ew.set_theta(np.zeros(ncell)); ew.zfac.x.array[:] = 1.0
    NRM = []
    for ew in ews:
        ew.solve(); ES, EP, EF, EO = P.measure(ew)
        NRM.append(dict(S=ES, P=EP, F=EF, O=EO))

    def wave_state():
        Jw, gz_w, packs = 0.0, np.zeros(ncell), []
        for ew, nrm in zip(ews, NRM):
            ew.solve(); ES, EP, EF, EO = P.measure(ew)
            Jw += P.objective(ES, EP, EF, EO, nrm)
            ew.set_region(P.grad_region(nrm, ES))
            _, dz = ew.focus_grads(); gz_w += dz
            packs.append((ES/nrm["S"], EP/nrm["P"], EF/nrm["F"]))
        return Jw/len(ews), gz_w/len(ews), packs

    # reference at the WAVE-OPTIMAL solid design (theta*, z=1): the panel we improve
    set_design(np.ones(ncell))
    Jstar, _, _ = wave_state()
    se.solve(); c0 = se.compliance()
    print(f"reference (theta*, solid): compliance={c0:.4e} volume={V0:.3e} "
          f"Jwave*={Jstar:+.3f}", flush=True)

    def evaluate(zd):
        zph = set_design(zd)
        Jw, gz_w, packs = wave_state()
        se.solve(); c = se.compliance(); gc_z = se.compliance_grad_z()
        V = float(A @ zph)
        # one-sided wave penalty: only penalise dropping BELOW the theta* design
        wp = max(0.0, Jstar - Jw)
        f = wc*(c/c0) + wm*(V/V0) + ww*wp
        gz = wc*gc_z/c0 + wm*A/V0 + (ww*(-gz_w) if wp > 0 else 0.0)
        gz = zsym(H.T @ gz)                            # chain filter + symmetrise
        return f, gz, c/c0, V/V0, np.mean([p[0] for p in packs]), \
            np.mean([p[1] for p in packs]), np.mean([p[2] for p in packs])

    mma = MMA(np.full(ncell, ZMIN), np.full(ncell, 1.0), move=0.06, adapt=True)
    zd = np.ones(ncell)
    best = (np.inf, zd.copy())
    for it in range(iters):
        f, gz, cr, vr, sv, pv, fv = evaluate(zd)
        if f < best[0]:
            best = (f, zd.copy())
        zd = np.clip(zsym(mma.update(zd, gz, f=f)), ZMIN, 1.0)
        if it % 8 == 0 or it == iters-1:
            print(f"[{it:3d}] f={f:+.3f} stiff={1/cr:4.2f}x vol={vr:4.2f} "
                  f"(mass -{100*(1-vr):.0f}%) sensor={sv:4.1f}x payload={pv:4.2f}x "
                  f"rims={fv:4.2f}x move={mma.move:.3f}", flush=True)

    zd = best[1]
    zph = set_design(zd)
    # final metrics + fields
    curve = []
    for ew, nrm in zip(ews, NRM):
        ew.solve(); ES, EP, EF, EO = P.measure(ew)
        curve.append((ew.omega, ES/nrm["S"], EP/nrm["P"], EF/nrm["F"]))
    se.solve(); c1 = se.compliance()
    tri, m1 = P.field_tri(ew0)
    kc = len(ews)//2
    stiff, vf = c0/c1, float(A @ zph)/V0
    print(f"FINAL: stiffness {stiff:.2f}x | mass -{100*(1-vf):.0f}% (vol frac {vf:.2f}) "
          f"| sensor {curve[kc][1]:.1f}x payload {curve[kc][2]:.2f}x "
          f"rims {curve[kc][3]:.2f}x", flush=True)

    np.savez("results/dolfinx_panel_dens_data.npz",
             trix=tri.x, triy=tri.y, tris=tri.triangles, cent=cent,
             thopt=thopt, zphys=zph, m1=m1, stiff=stiff, volfrac=vf,
             route=curve[kc][1], shield=curve[kc][2], prot=curve[kc][3],
             c0=c0, c1=c1, Lx=P.Lx, Ly=P.Ly, holes=np.array(P.HOLES),
             guard=P.GUARD, pay=np.array(P.PAY), sensor=np.array(P.SENSOR),
             src=np.array([P.SRC_X, P.Ly/2, P.SRC_W]), freqs=np.array(freqs),
             ux=se.u.x.array.copy())
    _diag(cent, zph, tri, m1, stiff, vf, curve[kc])


def _diag(cent, zph, tri, m1, stiff, vf, cc):
    tc = np.linspace(0, 2*np.pi, 160)
    fig, ax = plt.subplots(2, 1, figsize=(11, 9))
    # density map
    sc = ax[0].scatter(cent[:, 0], cent[:, 1], c=zph, s=6, cmap="bone_r",
                       vmin=0, vmax=1)
    for cx, cy, r in P.HOLES:
        ax[0].plot(cx+r*np.cos(tc), cy+r*np.sin(tc), "r-", lw=1.0)
    ax[0].set_aspect("equal"); ax[0].set_xlim(0, P.Lx); ax[0].set_ylim(0, P.Ly)
    ax[0].set_title(f"designed density z (material): {stiff:.2f}x stiffer, "
                    f"-{100*(1-vf):.0f}% mass")
    plt.colorbar(sc, ax=ax[0], fraction=0.022, pad=0.01)
    # wave field on the lightened panel
    vmax = np.percentile(m1, 99.0)
    tp = ax[1].tripcolor(tri, m1, cmap="magma", vmax=vmax, shading="gouraud")
    px, py, ph = P.PAY; sx, sy, sh = P.SENSOR
    ax[1].add_patch(plt.Rectangle((px-ph, py-ph), 2*ph, 2*ph, ec="#66ffcc", fc="none", lw=1.4, ls="--"))
    ax[1].add_patch(plt.Rectangle((sx-sh, sy-sh), 2*sh, 2*sh, ec="#bfefff", fc="none", lw=1.6, ls="--"))
    for cx, cy, r in P.HOLES:
        ax[1].plot(cx+r*np.cos(tc), cy+r*np.sin(tc), "w-", lw=1.0)
    ax[1].set_aspect("equal"); ax[1].set_xlim(0, P.Lx); ax[1].set_ylim(0, P.Ly)
    ax[1].set_title(f"wave still routes on the lightened panel: sensor {cc[1]:.1f}x, "
                    f"payload {cc[2]:.2f}x, rims {cc[3]:.2f}x")
    plt.colorbar(tp, ax=ax[1], fraction=0.022, pad=0.01)
    fig.suptitle("Density co-design: lighter + stiffer + still multifunctional", y=0.98)
    fig.savefig("results/dolfinx_panel_dens.png", dpi=130, bbox_inches="tight")
    print("saved results/dolfinx_panel_dens.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", default="results/dolfinx_panel_band_data.npz")
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--band", default="44,50,56")
    ap.add_argument("--wc", type=float, default=W_C)
    ap.add_argument("--wm", type=float, default=W_M)
    ap.add_argument("--ww", type=float, default=W_W)
    a, _ = ap.parse_known_args()
    main(a.warm, a.iters, [float(v) for v in a.band.split(",")], a.wc, a.wm, a.ww)
