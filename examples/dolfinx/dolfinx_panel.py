"""
GRAND EXAMPLE -- a multifunctional CFRP mounting panel whose SINGLE curvature-
feasible fiber layout does several jobs at once, and does them across an
operational frequency BAND (not a single tone):

  (S) ROUTE/HARVEST  reconverge the incident machine vibration onto an on-axis
                     energy-harvesting / SHM sensor patch                (maximise)
  (P) SHIELD         hold a mounted-payload core in a dark pocket -- the beam
                     flows AROUND it                                     (minimise)
  (F) PROTECT        keep the fatigue-critical fastener-hole rims quiet  (minimise)
  (O) CONFINE        a clean routed field, no spraying into the bulk     (minimise)

Everything is ONE orientation field theta(x), laid by AFP, curvature-limited by the
print head (zeta_all = 1/R_tow), and MIRROR-SYMMETRIC about y=Ly/2 so the split-
and-reconverge field is pristine.  The jobs compose into one signed region weight:

    J(w) = W_S*log(E_S/E_S0) - LAM_P*(E_P/E_P0) - LAM_F*(E_F/E_F0) - LAM_O*(E_O/E_O0)

each term normalised to its straight-fiber baseline (O(1), comparable), one adjoint
per solve.  For a BAND we average J over frequencies {w_k}: robustness across the
band, not a single-tone trick.  The per-frequency transmission curve is reported.

    XDG_CACHE_HOME=/home/jrt/wavetopo/.fenics-cache PYTHONPATH=/home/jrt/wavetopo \
    /home/jrt/miniforge3/envs/dolfinx_complex/bin/python3 \
        examples/dolfinx/dolfinx_panel.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import ufl
from dolfinx import fem

from wavetopo.dolfinx_elastic import ElasticWave
from wavetopo.dolfinx_static import StaticElastic
from wavetopo.dolfinx_mesh import rect_mesh
from wavetopo.dolfinx_wave import support_map, csrbf_operators, curl_penalty, curl_of
from wavetopo.cfrp_optimizer import MMA

# ---- domain: a landscape mounting panel, symmetric about y=Ly/2 ------------- #
Lx, Ly = 7.0, 4.0
MARGIN = 0.6
FREQS = [50.0]                            # operational band (single tone for now)
ETA = 0.02
HMESH = 0.055

SRC_X, SRC_W = 0.15, 0.95                 # broad incident beam, centred on y=Ly/2

PAY = (3.2, 2.0, 0.50)                    # shielded payload core (keep dark)
RB, GUARD = 0.20, 0.18
HOLES = [(2.55, 1.05, RB), (2.55, 2.95, RB),
         (4.35, 1.05, RB), (4.35, 2.95, RB)]   # symmetric 4-bolt mount pattern
SENSOR = (5.90, 2.00, 0.30)              # on-axis harvester / SHM patch

R_TOW = 0.16
ZETA_ALL = 1.0/R_TOW

W_S, LAM_P, LAM_F, LAM_O = 1.0, 1.0, 1.0, 0.5

# static stiffness co-design: the panel is BOLTED at the four holes and carries a
# service load over the payload footprint (the mounted module's inertial weight);
# fibers that also align to the load path stiffen it.  STIFF_W=0 disables it.
STIFF_W = 0.0
LOAD_MAG = 1.0                            # downward service load over the payload

OUT_NPZ = "results/dolfinx_panel_data.npz"
OUT_PNG = "results/dolfinx_panel.png"


# ---- regions (all symmetric about y=Ly/2) ----------------------------------- #
def _box(c):
    cx, cy, h = c
    return lambda x, y: ((np.abs(x-cx) < h) & (np.abs(y-cy) < h)).astype(float)


w_sensor = _box(SENSOR)
w_payload = _box(PAY)


def w_rims(x, y):
    m = np.zeros_like(x)
    for cx, cy, r in HOLES:
        rr = np.hypot(x-cx, y-cy)
        m = np.maximum(m, ((rr >= r) & (rr < r+GUARD)).astype(float))
    return m


def w_interior(x, y):
    m = MARGIN
    return ((x > m) & (x < Lx-m) & (y > m) & (y < Ly-m)).astype(float)


# perimeter-routing experiment: a strongly-penalised central keep-out band in the
# mid-span forces the beam to route AROUND the outside instead of through the middle
PERIM = False
LAM_KO = 1.2


def w_keepout(x, y):
    return ((np.abs(y - Ly/2) < 0.85) & (x > 1.3) & (x < 5.2)).astype(float)


def _parts(x, y):
    """Disjoint masks by priority: rims > payload > sensor > outside."""
    F = w_rims(x, y)
    P = w_payload(x, y)*(1.0 - F)
    S = w_sensor(x, y)*(1.0 - F)*(1.0 - P)
    used = np.clip(F + P + S, 0.0, 1.0)
    O = w_interior(x, y)*(1.0 - used)
    return S, P, F, O


def signed_region(nrm):
    def f(x, y):
        S, P, F, O = _parts(x, y)
        w = ((W_S/nrm["S"])*S - (LAM_P/nrm["P"])*P
             - (LAM_F/nrm["F"])*F - (LAM_O/nrm["O"])*O)
        if PERIM:
            w = w - (LAM_KO/nrm["O"])*O*w_keepout(x, y)
        return w
    return f


def grad_region(nrm, curS):
    def f(x, y):
        S, P, F, O = _parts(x, y)
        w = ((W_S/max(curS, 1e-30))*S - (LAM_P/nrm["P"])*P
             - (LAM_F/nrm["F"])*F - (LAM_O/nrm["O"])*O)
        if PERIM:
            w = w - (LAM_KO/nrm["O"])*O*w_keepout(x, y)
        return w
    return f


def objective(ES, EP, EF, EO, nrm):
    return (W_S*np.log(max(ES, 1e-30)/nrm["S"]) - LAM_P*EP/nrm["P"]
            - LAM_F*EF/nrm["F"] - LAM_O*EO/nrm["O"])


# ---- mesh / solver helpers -------------------------------------------------- #
def field_tri(ew):
    dom = ew.domain
    coords = dom.geometry.x
    tris = dom.geometry.dofmap.reshape(-1, 3) \
        if hasattr(dom.geometry.dofmap, "reshape") else dom.geometry.dofmap
    V = fem.functionspace(dom, ("Lagrange", 1))
    mag = fem.Function(V)
    u2 = sum(ew.U[i]*ew.U[i] for i in range(4))
    mag.interpolate(fem.Expression(ufl.sqrt(u2), V.element.interpolation_points()))
    return mtri.Triangulation(coords[:, 0], coords[:, 1], tris), mag.x.array


def centroids(ew):
    dom = ew.domain
    dom.topology.create_connectivity(2, 0)
    conn = dom.topology.connectivity(2, 0)
    xy = dom.geometry.x[:, :2]
    n = dom.topology.index_map(2).size_local
    c = np.zeros((ew.theta.x.array.size, 2))
    for e in range(n):
        c[ew.S.dofmap.cell_dofs(e)[0]] = xy[conn.links(e)].mean(0)
    return c


def symmetric_support(spacing):
    """CS-RBF support grid mirror-symmetric about y=Ly/2, + mirror index map."""
    ax = Ly/2.0
    sx = np.arange(0.0, Lx+1e-9, spacing)
    ny = int(np.ceil(ax/spacing))
    sy = ax + spacing*np.arange(-ny, ny+1)
    SX, SY = np.meshgrid(sx, sy)
    supp = np.column_stack([SX.ravel(), SY.ravel()])
    key = {(round(px, 5), round(py, 5)): i for i, (px, py) in enumerate(supp)}
    mir = np.array([key[(round(px, 5), round(Ly-py, 5))] for px, py in supp])
    return supp, mir


def make_solver(omega):
    dom = rect_mesh(Lx, Ly, HMESH, holes=HOLES, h_hole=0.5*HMESH)
    return _attach(dom, omega), dom


def _attach(dom, omega):
    ew = ElasticWave(dom, omega=omega, Ef1=131., Ef2=9., G12=5., nu12=0.27,
                     rho=1.6, rho_m=1.2, eta=ETA, penal=3.0)
    m = MARGIN
    ew.set_sponge(lambda x, y: np.maximum(0, np.maximum.reduce([
        (x-(Lx-m))/m, (m-y)/m, (y-(Ly-m))/m]))**2 * 80.0)
    ew.set_source(lambda x, y: 55.0*np.exp(-80*(x-SRC_X)**2)
                  * np.exp(-((y-Ly/2)/SRC_W)**2))
    return ew


def energy_in(ew, fn):
    ew.set_region(fn)
    return ew.focus_energy()


def measure(ew):
    ES = energy_in(ew, w_sensor); EP = energy_in(ew, w_payload)
    EF = energy_in(ew, w_rims); EO = energy_in(ew, lambda x, y: _parts(x, y)[3])
    return ES, EP, EF, EO


def main(iters=110, zeta_all=ZETA_ALL, warm=None):
    # one solver per frequency, all on the SAME mesh sharing theta
    ews, doms = [], []
    for w in FREQS:
        dom = rect_mesh(Lx, Ly, HMESH, holes=HOLES, h_hole=0.5*HMESH)
        ews.append(_attach(dom, w)); doms.append(dom)
    ew0 = ews[0]
    ncell = ew0.theta.x.array.size
    cent = centroids(ew0)
    print(f"conforming mesh: {ncell} cells; {len(HOLES)} holes; "
          f"band={FREQS}; zeta_all={zeta_all:.2f}", flush=True)

    supp, mir = symmetric_support(0.34)
    B, supp = support_map(cent, (0, Lx), (0, Ly), 0.34, 0.72, supp=supp)
    B, Bx, By = csrbf_operators(cent, supp, 0.72)
    Md = B.shape[1]
    sym = lambda v: 0.5*(v - v[mir])
    print(f"{Md} symmetric CS-RBF support points", flush=True)

    # per-frequency baselines
    NRM = []
    for ew in ews:
        ew.set_theta(np.zeros(ncell)); ew.solve()
        ES0, EP0, EF0, EO0 = measure(ew)
        NRM.append(dict(S=ES0, P=EP0, F=EF0, O=EO0))
    tri, m0 = field_tri(ew0)
    print("baselines:", [f"{n['S']:.2e}" for n in NRM], flush=True)

    # optional static stiffness co-design: same mesh & theta as the wave solvers
    se = c0 = M0 = None
    if STIFF_W > 0:
        se = StaticElastic(ew0.domain, Ef1=131., Ef2=9., G12=5., nu12=0.27,
                           rho=1.6, rho_m=1.2, penal=3.0)
        se.clamp(HOLES, tol=0.6*HMESH)
        px, py, ph = PAY
        se.set_load(lambda x, y: np.zeros_like(x),
                    lambda x, y: -LOAD_MAG*((np.abs(x-px) < ph)
                                            & (np.abs(y-py) < ph)).astype(float))
        se.set_theta(np.zeros(ncell)); se.solve()
        c0, M0 = se.compliance(), se.mass()
        print(f"stiffness co-design ON (W={STIFF_W}): baseline compliance="
              f"{c0:.4e}, mass={M0:.4e}", flush=True)

    def eval_bandwidth(x):
        """Solve every frequency (+static), return mean objective and gradient."""
        th = B @ x
        Js, gs = [], np.zeros(Md)
        packs = []
        for ew, nrm in zip(ews, NRM):
            ew.set_theta(th); ew.solve()
            ES, EP, EF, EO = measure(ew)
            Jk = objective(ES, EP, EF, EO, nrm)
            ew.set_region(grad_region(nrm, ES))
            gk = B.T @ ew.focus_grad()
            Js.append(Jk); gs = gs + gk
            packs.append((ES/nrm["S"], EP/nrm["P"], EF/nrm["F"]))
        Jw, gw, cr = np.mean(Js), gs/len(ews), None
        if se is not None:                          # minimise compliance too
            se.set_theta(th); se.solve()
            c = se.compliance()
            Jw = Jw - STIFF_W*(c/c0)
            gw = gw - STIFF_W*(B.T @ se.compliance_grad_theta())/c0
            cr = c/c0
        return Jw, gw, packs, cr

    mma = MMA(np.full(Md, -np.pi/2), np.full(Md, np.pi/2), move=0.10, adapt=True)
    x = np.zeros(Md)
    if warm is not None:                        # warm start from a prior theta field
        th_w = np.load(warm)["thopt"]
        x = sym(np.linalg.lstsq(B.toarray(), th_w, rcond=None)[0])
        print(f"warm start from {warm} (lstsq residual "
              f"{np.linalg.norm(B@x - th_w):.3f})", flush=True)
    best = (-np.inf, np.zeros(ncell), np.zeros(Md)); hist, histZ = [], []
    for it in range(iters):
        mu = 0.0 if it < 20 else min(8.0, 0.5*1.28**(it-20))
        J, g, packs, cr = eval_bandwidth(x)
        zeta = curl_of(x, B, Bx, By)[0]; zmax = np.abs(zeta).max()
        hist.append(J); histZ.append(zmax)
        if (zmax <= zeta_all) and J > best[0]:
            best = (J, (B @ x).copy(), x.copy())
        g = sym(g)                              # keep the search in the symmetric subspace
        pen = 0.0
        if mu > 0:
            pen, dP = curl_penalty(x, B, Bx, By, zeta_all)
            g = g - mu*dP
        x = sym(mma.update(x, -g, f=-(J - mu*pen)))
        if it % 10 == 0 or it == iters-1:
            sv = np.mean([p[0] for p in packs]); pv = np.mean([p[1] for p in packs])
            fv = np.mean([p[2] for p in packs])
            cs = f" stiff={1.0/cr:4.2f}x" if cr else ""
            print(f"[{it:3d}] J={J:+.2f} sensor={sv:7.1f}x payload={pv:5.2f}x "
                  f"rims={fv:5.2f}x{cs} max|zeta|={zmax:.2f} mu={mu:.2f} "
                  f"move={mma.move:.3f}", flush=True)

    ok = np.isfinite(best[0])
    thopt, xopt = (best[1], best[2]) if ok else (B @ x, x)
    if not ok:
        print("WARNING: no feasible iterate; reporting final", flush=True)

    # final per-frequency report + fields at the band centre
    route = shield = prot = None
    curve = []
    for ew, nrm in zip(ews, NRM):
        ew.set_theta(thopt); ew.solve()
        ES, EP, EF, EO = measure(ew)
        curve.append((ew.omega, ES/nrm["S"], EP/nrm["P"], EF/nrm["F"]))
    ew0.set_theta(thopt); ew0.solve()
    tri, m1 = field_tri(ew0)
    kc = len(ews)//2
    route, shield, prot = curve[kc][1], curve[kc][2], curve[kc][3]
    z1 = np.abs(curl_of(xopt, B, Bx, By)[0]).max()
    print("per-frequency (omega, sensorx, payloadx, rimsx):", flush=True)
    for w, s, p, f in curve:
        print(f"   w={w:4.1f}: sensor={s:7.1f}x payload={p:5.2f}x rims={f:5.2f}x",
              flush=True)
    stiff = 1.0
    if se is not None:
        se.set_theta(thopt); se.solve()
        c1 = se.compliance(); stiff = c0/c1
        print(f"STIFFNESS: compliance {c0:.4e} -> {c1:.4e}  ({stiff:.2f}x stiffer) "
              f"at fixed mass {M0:.4e}", flush=True)
    print(f"BAND-CENTRE: sensor {route:.0f}x | payload {shield:.2f}x | "
          f"rims {prot:.2f}x | stiffness {stiff:.2f}x | max|zeta|={z1:.2f} "
          f"(limit {zeta_all:.2f})", flush=True)

    np.savez(OUT_NPZ,
             trix=tri.x, triy=tri.y, tris=tri.triangles, cent=cent, thopt=thopt,
             m0=m0, m1=m1, hist=np.array(hist), histZ=np.array(histZ),
             curve=np.array(curve), route=route, shield=shield, prot=prot,
             stiff=stiff, zeta_all=zeta_all, zmax=z1, Lx=Lx, Ly=Ly,
             holes=np.array(HOLES), guard=GUARD, pay=np.array(PAY),
             sensor=np.array(SENSOR), src=np.array([SRC_X, Ly/2, SRC_W]),
             freqs=np.array(FREQS))
    _diag(tri, m0, m1, route, shield, prot, z1, curve)


def _diag(tri, m0, m1, route, shield, prot, z1, curve):
    tc = np.linspace(0, 2*np.pi, 160)
    vmax = np.percentile(m1, 99.0)

    def marks(a_):
        cx, cy, h = PAY
        a_.add_patch(plt.Rectangle((cx-h, cy-h), 2*h, 2*h, ec="#66ffcc",
                                   fc="none", lw=1.4, ls="--"))
        sx, sy, sh = SENSOR
        a_.add_patch(plt.Rectangle((sx-sh, sy-sh), 2*sh, 2*sh, ec="#bfefff",
                                   fc="none", lw=1.6, ls="--"))
        for cx, cy, r in HOLES:
            a_.plot(cx+r*np.cos(tc), cy+r*np.sin(tc), "w-", lw=1.0)
            a_.plot(cx+(r+GUARD)*np.cos(tc), cy+(r+GUARD)*np.sin(tc), "r:", lw=1.2)
        a_.set_aspect("equal"); a_.set_xlim(0, Lx); a_.set_ylim(0, Ly)

    fig, ax = plt.subplots(2, 1, figsize=(11, 9))
    for a_, m_, ttl in [
            (ax[0], m0, "straight fibers (baseline)"),
            (ax[1], m1, f"optimized: sensor {route:.0f}x, payload {shield:.2f}x, "
                        f"rims {prot:.2f}x, max|zeta|={z1:.2f}")]:
        tp = a_.tripcolor(tri, m_, cmap="magma", vmax=vmax, shading="gouraud")
        marks(a_); a_.set_title(ttl, fontsize=11)
        plt.colorbar(tp, ax=a_, fraction=0.022, pad=0.01)
    fig.suptitle("Multifunctional CFRP panel: route / shield / protect -- one "
                 "symmetric fiber layout", y=0.97, fontsize=12)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print("saved", OUT_PNG, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=110)
    ap.add_argument("--lam-p", type=float, default=LAM_P)
    ap.add_argument("--lam-f", type=float, default=LAM_F)
    ap.add_argument("--lam-o", type=float, default=LAM_O)
    ap.add_argument("--band", type=str, default="", help="comma list, e.g. 46,50,54")
    ap.add_argument("--warm", type=str, default="", help="npz with thopt to warm-start")
    ap.add_argument("--tag", type=str, default="", help="output suffix")
    ap.add_argument("--stiff", type=float, default=STIFF_W,
                    help="static stiffness co-design weight (0 = wave only)")
    ap.add_argument("--perimeter", action="store_true",
                    help="force perimeter routing via a central keep-out band")
    a, _ = ap.parse_known_args()
    LAM_P, LAM_F, LAM_O, STIFF_W = a.lam_p, a.lam_f, a.lam_o, a.stiff
    PERIM = a.perimeter
    if a.band:
        FREQS = [float(v) for v in a.band.split(",")]
    if a.tag:
        OUT_NPZ = f"results/dolfinx_panel{a.tag}_data.npz"
        OUT_PNG = f"results/dolfinx_panel{a.tag}.png"
    main(iters=a.iters, warm=(a.warm or None))
