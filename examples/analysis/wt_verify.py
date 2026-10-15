r"""Numerical verification of the wavetopo results (referee requests).

A  mesh convergence of the lens: G, G_phi and interior growth at omega = 40, 50 on the
   structured mesh nx = 130 (paper), 195, 260, with the saved optimized fiber field
   transferred by interpolating (cos 2theta, sin 2theta) -- the field is a smooth CS-RBF
   field, so this is a resampling, not a re-design;
B  damping sensitivity of the lens: eta = 0.01, 0.02 (paper), 0.04;
C  absorbing layer: (i) reflection coefficient of the sponge for the straight-fiber plate,
   fitted on the centerline from the field  A e^{-ikx} + B e^{+ikx}  with the complex
   along-fiber qP wavenumber; (ii) the lens re-evaluated with a sponge TWICE as thick that
   starts at the same place (domain enlarged outward), so the interior is unchanged and
   only the reflection changes;
D  mesh convergence of the cloak: reductions at omega = 50 for h = 0.045 (paper) and 0.03.

    PYTHONPATH=. python  # dolfinx (real PETSc) environment \
        examples/analysis/wt_verify.py [A|B|C|D ...]
Results -> results/wt_verify_*.npy and stdout.
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

sys.path.insert(0, (_ROOT + "/examples/dolfinx"))
import dolfinx_lens as DL
from wavetopo.dolfinx_elastic import ElasticWave, _SUM

M_SP = 0.6
LENS = np.load("results/dolfinx_lens_data.npz")


def transfer(cent_old, th_old, cent_new):
    """Resample a smooth orientation field on new cell centroids (axial: use 2*theta)."""
    v = np.c_[np.cos(2 * th_old), np.sin(2 * th_old)]
    lin = LinearNDInterpolator(cent_old, v)(cent_new)
    near = NearestNDInterpolator(cent_old, v)(cent_new)
    lin = np.where(np.isnan(lin), near, lin)
    return 0.5 * np.arctan2(lin[:, 1], lin[:, 0])


def lens_metrics(ew):
    U = ew.U
    u2 = sum(U[i] * U[i] for i in range(4))
    x = ufl.SpatialCoordinate(ew.domain)
    interior = ufl.conditional(ufl.And(ufl.And(ufl.lt(x[0], DL.Lx - M_SP), ufl.gt(x[1], M_SP)),
                                       ufl.lt(x[1], DL.Ly - M_SP)), 1.0, 0.0)
    interior = interior * ufl.conditional(ufl.gt(x[0], 0.0), 1.0, 0.0)
    Ef = ew.focus_energy()
    Ei = ew.domain.comm.allreduce(fem.assemble_scalar(fem.form(interior * u2 * ufl.dx)), op=_SUM())
    return Ef, Ei


def lens_pair(ew, th):
    n = ew.theta.x.array.size
    ew.set_theta(np.zeros(n)); ew.solve(); E0, I0 = lens_metrics(ew)
    ew.set_theta(th); ew.solve(); E1, I1 = lens_metrics(ew)
    return E1 / E0, (E1 / I1) / (E0 / I0), I1 / I0


def lens_make(nx=130, omega=40.0, eta=0.02, pad=0.0):
    """Lens plate; pad > 0 enlarges the domain outward on the right/top/bottom by `pad`
    and thickens the sponge by the same amount, so the sponge still STARTS at
    x = Lx - 0.6, y = 0.6, y = Ly - 0.6 (interior unchanged)."""
    Lx, Ly, m = DL.Lx, DL.Ly, M_SP
    x1, y0, y1 = Lx + pad, -pad, Ly + pad
    ny = int(nx * (y1 - y0) / Lx)
    dom = dmesh.create_rectangle(MPI.COMM_WORLD, [[0, y0], [x1, y1]], [int(nx * x1 / Lx), ny],
                                 dmesh.CellType.triangle)
    ew = ElasticWave(dom, omega=omega, Ef1=131.0, Ef2=9.0, G12=5.0, nu12=0.27, rho=1.6, eta=eta)
    t = m + pad                                               # sponge thickness
    ew.set_sponge(lambda x, y: np.maximum(0, np.maximum.reduce([
        (x - (Lx - m)) / t, ((m) - y) / t, (y - (Ly - m)) / t])) ** 2 * 80.0)
    ew.set_source(lambda x, y: 30.0 * np.exp(-80 * (x - 0.1) ** 2) * np.ones_like(y))
    fx, fy, fr = DL.FOCUS
    ew.set_region(lambda x, y: ((np.abs(x - fx) < fr) & (np.abs(y - fy) < fr)).astype(float))
    return ew


def lens_theta(ew):
    return transfer(LENS["cent"], LENS["thopt"], DL.centroids(ew))


def centerline_ux(ew, y=1.5, tol=None):
    V0, mp = ew.W.sub(0).collapse()
    V2, mp2 = ew.W.sub(2).collapse()
    X = V0.tabulate_dof_coordinates()
    tol = tol or 0.6 * (DL.Lx / 130)
    k = np.where(np.abs(X[:, 1] - y) < tol)[0]
    o = np.argsort(X[k, 0])
    xs = X[k, 0][o]
    ur = ew.U.x.array[np.asarray(mp)[k[o]]]
    ui = ew.U.x.array[np.asarray(mp2)[k[o]]]
    return xs, ur + 1j * ui


def reflection(ew, omega, eta):
    """|R| at the sponge entrance x_s = Lx - 0.6 from a two-wave fit on 0.5 < x < 3.3."""
    E1, E2, nu12 = 131.0, 9.0, 0.27
    nu21 = nu12 * E2 / E1
    Q11 = E1 / (1 - nu12 * nu21)
    k = omega * np.sqrt(1.6 / (Q11 * (1 + 1j * eta)))
    xs, u = centerline_ux(ew)
    sel = (xs > 0.5) & (xs < DL.Lx - M_SP - 0.1)
    A = np.c_[np.exp(-1j * k * xs[sel]), np.exp(1j * k * xs[sel])]
    c, *_ = np.linalg.lstsq(A, u[sel], rcond=None)
    res = np.linalg.norm(A @ c - u[sel]) / np.linalg.norm(u[sel])
    xsp = DL.Lx - M_SP
    R = abs(c[1] * np.exp(1j * k * xsp)) / abs(c[0] * np.exp(-1j * k * xsp))
    return R, res


def study_A():
    rows = []
    for om in (40.0, 50.0):
        for nx in (130, 195, 260):
            ew = lens_make(nx=nx, omega=om)
            G, Gp, gr = lens_pair(ew, lens_theta(ew))
            rows.append((om, nx, G, Gp, gr))
            print(f"[A] omega={om:4.1f} nx={nx:3d}  G={G:6.2f}  G_phi={Gp:5.2f}  growth={gr:6.2f}", flush=True)
    np.save("results/wt_verify_A.npy", np.array(rows))


def study_B():
    rows = []
    for eta in (0.01, 0.02, 0.04):
        for om in (37.5, 40.0, 42.5, 50.0):
            ew = lens_make(omega=om, eta=eta)
            G, Gp, gr = lens_pair(ew, lens_theta(ew))
            rows.append((eta, om, G, Gp, gr))
            print(f"[B] eta={eta:.2f} omega={om:4.1f}  G={G:6.2f}  G_phi={Gp:5.2f}  growth={gr:6.2f}", flush=True)
    np.save("results/wt_verify_B.npy", np.array(rows))


def study_C():
    rows = []
    for om in (35.0, 37.5, 40.0, 42.5, 45.0, 50.0):
        r = [om]
        for pad in (0.0, 0.6):
            ew = lens_make(omega=om, pad=pad)
            ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve()
            R, res = reflection(ew, om, 0.02)
            G, Gp, gr = lens_pair(ew, lens_theta(ew))
            r += [R, res, G, Gp, gr]
            print(f"[C] omega={om:4.1f} sponge={0.6+pad:.1f}  |R|={R:.3f} (fit res {res:.2f})  "
                  f"G={G:6.2f}  G_phi={Gp:5.2f}  growth={gr:6.2f}", flush=True)
        rows.append(r)
    np.save("results/wt_verify_C.npy", np.array(rows))


def study_D():
    import dolfinx_cloak_conforming_vec as CV
    unc = np.load("results/dolfinx_cloak_conforming_data.npz")
    h2h = np.load("results/cloak_headtohead.npz")
    rows = []
    for hm in (0.045, 0.030):
        CV.HMESH = hm; CV.OMEGA = 50.0
        ew, ref = CV.build()
        cn = CV.centroids(ew); n = ew.theta.x.array.size
        des = {"optimized": transfer(unc["cent"], unc["thopt"], cn),
               "curv-limited": transfer(h2h["cent"], h2h["th_opt"], cn),
               "transformation": transfer(h2h["cent"], h2h["th_an"], cn)}
        ew.set_theta(np.zeros(n)); ew.solve(); J0 = ew.cloak_mismatch()
        r = [hm, n]
        for k_, th in des.items():
            ew.set_theta(th); ew.solve(); r.append(J0 / ew.cloak_mismatch())
        rows.append(r)
        print(f"[D] h={hm:.3f} ({n} cells)  " + "  ".join(f"{k_}={v:5.1f}x" for k_, v in zip(des, r[2:])),
              flush=True)
    np.save("results/wt_verify_D.npy", np.array(rows))


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C", "D"]
    for w in which:
        {"A": study_A, "B": study_B, "C": study_C, "D": study_D}[w]()
