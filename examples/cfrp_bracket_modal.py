r"""
Topology + toolpath co-design of a CFRP bracket to MAXIMIZE the first natural
frequency omega_1 under a MASS (volume) budget -- the counterintuitive design.

Naively, a higher fundamental frequency just means a stiffer structure.  But
omega_1^2 is a Rayleigh quotient phi^T K phi / phi^T M phi, so it responds to WHERE
mass and stiffness sit relative to the mode shape, not only to their totals.  The
eigenvalue sensitivity has two terms:

    dlam_1/dz_e  =  w'(z_e) phi_e^T (k_f - k_m)_e phi_e          (stiffness: add ribs
                                                                 where the mode strains)
                   - lam_1 (rho_f - rho_m) phi_e^T Me0_e phi_e   (mass: REMOVE mass at
                                                                 the antinodes)

so at FIXED mass the optimizer stiffens the strained regions AND scoops mass out of
the mode's antinodes, raising omega_1 DISPROPORTIONATELY.

Maximizing omega_1 ALONE is a trap, though: a Rayleigh quotient rewards any mass that
raises stiffness faster than inertia, so the optimizer grows NEUTRAL OUTRIGGERS --
stubby root cantilevers that nudge omega_1 up by <0.5% but carry no service load.  They
persist at every volume and regrow even when forbidden.  We therefore co-design against
THREE objectives at fixed mass: MAX omega_1 (modal), MIN compliance (so every member is
load-bearing, not a dead outrigger), and MIN p-norm von Mises stress (so the load path
is thick and uniform).  Plus: top/bottom mirror symmetry, clamp-connectivity (only
material grounded to the support is kept), and passive-solid collars backing every hole.
All sensitivities are FD- and isotropy-verified.

  z_hat -> cone filter -> tanh project   z     : TOPOLOGY (crisp)
  theta_hat -> CS-RBF                    theta : TOOLPATH (smooth, printable)

    PYTHONPATH=. python examples/cfrp_bracket_modal.py --h 0.025 --iters 300
"""
import argparse
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from wavetopo.cfrp_wave import (conforming_plate, CFRPWavePlate, cone_filter,
                                curl_operators, curl_penalty, director_curl, Material)
from wavetopo.cfrp import CSRBFMapping
from wavetopo.cfrp_optimizer import MMA


LX, LY = 3.5, 2.0
# two mounting bolt holes at the clamped root, and a THIRD hole at the tip that is
# the loading boundary -- the service load is applied on its rim, so the topology
# must connect material from the mounts to the load.
HOLES = [(0.55, 0.5, 0.20), (0.55, 1.5, 0.20), (2.95, 1.0, 0.26)]
LOAD_HOLE = 2                                       # index of the tip / load hole
# SINGLE solid material (anisotropic fiber); "removed" material is TRUE void, given a
# small Ersatz stiffness/mass (E_MIN, RHO_MIN, ~1% of solid, same stiffness/mass ratio
# so it adds no spurious low modes) only to keep the operators well-conditioned.
RHO_F, RHO_MIN, E_MIN = 1.6, 0.016, 1.31


class _Adapter:
    def __init__(self, mesh):
        self.cx, self.cy, self.N = mesh.cx, mesh.cy, mesh.tris.shape[0]


_EADJ_CACHE = {}


def _edge_adjacency(mesh):
    """Element graph: two triangles are adjacent iff they share an edge (cached)."""
    key = id(mesh)
    if key in _EADJ_CACHE:
        return _EADJ_CACHE[key]
    tris = mesh.tris
    edict = {}
    I, J = [], []
    for e, t in enumerate(tris):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (a, b) if a < b else (b, a)
            if k in edict:
                I += [e, edict[k]]; J += [edict[k], e]
            else:
                edict[k] = e
    ne = tris.shape[0]
    adj = coo_matrix((np.ones(len(I)), (I, J)), shape=(ne, ne)).tocsr()
    _EADJ_CACHE[key] = adj
    return adj


def grounded_mask(mesh, solid, x_clamp=1e-6):
    """Boolean over elements: solid cells CONNECTED (through solid) to the clamp
    edge x=0.  Material not reachable from the support is floating -- it adds mass
    with no stiffness in the clamped eigenproblem and cannot react the load -- so we
    treat only the grounded component as the real structure."""
    adj = _edge_adjacency(mesh)
    idx = np.flatnonzero(solid)
    if idx.size == 0:
        return solid.copy()
    ncomp, lab = connected_components(adj[idx][:, idx], directed=False)
    # a component is grounded if any of its elements has a node on the clamp edge
    node_clamp = mesh.npx[:, 0] < x_clamp
    ecl = node_clamp[mesh.tris].any(axis=1)               # element touches clamp
    grounded = np.zeros(ncomp, bool)
    for k in range(ncomp):
        if ecl[idx[lab == k]].any():
            grounded[k] = True
    keep = np.zeros(solid.shape[0], bool)
    keep[idx] = grounded[lab]
    return keep


def component_report(mesh, solid):
    """(#components, area-fraction of the largest, area-fraction grounded)."""
    adj = _edge_adjacency(mesh)
    idx = np.flatnonzero(solid)
    if idx.size == 0:
        return 0, 0.0, 0.0
    ncomp, lab = connected_components(adj[idx][:, idx], directed=False)
    a = mesh.area[idx]
    ca = np.array([a[lab == k].sum() for k in range(ncomp)])
    gm = grounded_mask(mesh, solid)
    return ncomp, float(ca.max() / ca.sum()), float(mesh.area[gm].sum()
                                                     / mesh.area[solid].sum())


def project(y, beta, eta=0.5):
    num = np.tanh(beta * eta) + np.tanh(beta * (y - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / den


def dproject(y, beta, eta=0.5):
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta * (1 - np.tanh(beta * (y - eta)) ** 2) / den


def setup(h):
    mesh = conforming_plate(LX, LY, h, holes=HOLES)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=E_MIN, nu_m=0.3,
                   rho_f=RHO_F, rho_m=RHO_MIN)
    P = CFRPWavePlate(mesh, mat, omega=1.0, eta=0.0)   # omega/eta unused by modal
    xs, ys = mesh.npx[:, 0], mesh.npx[:, 1]
    left = np.flatnonzero(xs < 1e-6)
    # service load: downward, on the rim of the tip (load) hole, so the topology must
    # route material from the clamped mounts to the load boundary.
    tx, ty, tr = HOLES[LOAD_HOLE]
    rim = np.flatnonzero(np.abs(np.hypot(xs - tx, ys - ty) - tr) < 1.5 * h)
    Fs = np.zeros(mesh.ndof); Fs[2 * rim + 1] = -1.0 / max(len(rim), 1)
    P.set_static(Fs, np.concatenate([2 * left, 2 * left + 1]))
    return mesh, mat, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.04)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--vol", type=float, default=0.40)
    ap.add_argument("--init", type=float, default=None,
                    help="initial uniform density (default = --vol). With a mass "
                         "penalty, start LEAN so material is ADDED only where it earns "
                         "its weight, rather than starting fat and trapping horns.")
    ap.add_argument("--ring", type=float, default=0.13,
                    help="width of the passive SOLID annulus forced around every "
                         "hole, so the load boundary is backed and the mounts solid")
    ap.add_argument("--w-eig", dest="w_eig", type=float, default=1.0)
    ap.add_argument("--w-eig2", dest="w_eig2", type=float, default=0.0,
                    help="weight on MINIMIZING the second eigenfrequency lam_2/lam_2^0; "
                         "with --w-eig this maximizes omega_1 while pushing omega_2 down, "
                         "driving the first two modes toward a degenerate pair (a "
                         "mode-separation study; uses modal_two)")
    ap.add_argument("--w-stress", dest="w_stress", type=float, default=1.5,
                    help="weight on the p-norm von Mises stress (MINIMIZED) --- gives "
                         "a physical, stress-uniform structure rather than a "
                         "compliance-optimal one with thin, overloaded members")
    ap.add_argument("--pnorm", type=float, default=8.0)
    ap.add_argument("--qrelax", type=float, default=0.5)
    ap.add_argument("--cvol", type=float, default=200.0,
                    help="penalty on the one-sided volume CAP (vol <= --vol)")
    ap.add_argument("--w-comp", dest="w_comp", type=float, default=3.0,
                    help="weight on static compliance c/c0 (MINIMIZED): rewards "
                         "load-bearing material, so the fixed budget favours a stiff "
                         "load path over neutral frequency outriggers")
    ap.add_argument("--w-vol", dest="w_vol", type=float, default=0.0,
                    help="linear MASS penalty: material is kept only if its marginal "
                         "frequency/stress benefit outweighs its mass, so useless "
                         "(static-dead, modally inert) members are removed rather than "
                         "parked to fill a fixed volume budget")
    ap.add_argument("--w-grey", dest="w_grey", type=float, default=0.6)
    ap.add_argument("--rmin", type=float, default=0.12,
                    help="density-filter radius: the minimum-length-scale regularizer")
    ap.add_argument("--deta", type=float, default=0.10,
                    help="robust erode/dilate threshold offset -- objectives are "
                         "evaluated on the ERODED projection so thin members that "
                         "vanish under erosion are penalized (Wang-Lazarov-Sigmund)")
    ap.add_argument("--rbf-spc", dest="rbf_spc", type=float, default=0.08,
                    help="fiber-field (CS-RBF) support spacing; fine enough to let the "
                         "tows wrap holes and follow each chord, exploiting the full "
                         "anisotropy (a coarse field flattens the fibers artificially)")
    ap.add_argument("--betamax", type=float, default=24.0)
    ap.add_argument("--no-sym", dest="sym", action="store_false",
                    help="disable mirror symmetry about the mid-height axis y=LY/2 "
                         "(the load and mounts are symmetric, so the optimum is too)")
    ap.add_argument("--w-curl", dest="w_curl", type=float, default=0.2,
                    help="initial curl weight; adapted at runtime to hold |zeta|max at "
                         "--zeta-all (the printability limit), using the full budget")
    ap.add_argument("--curl-ramp", dest="curl_ramp", type=float, default=3.0)
    ap.add_argument("--zeta-all", dest="zeta_all", type=float, default=4.8,
                    help="printable fiber-curvature limit 1/R; the adaptive curl weight "
                         "drives |zeta|max to this value so fibers exploit anisotropy "
                         "maximally while staying manufacturable")
    ap.add_argument("--warm", default=None,
                    help="npz to warm-start the density z from (its 'zopt' field)")
    ap.add_argument("--void", default=None,
                    help="npz with a boolean 'void' mask of elements forced to stay "
                         "VOID -- used to forbid the neutral root appendages that "
                         "single-eigenvalue maximization deposits, so the freed budget "
                         "thickens the load-bearing truss instead")
    ap.add_argument("--out", default="results/cfrp_bracket_modal")
    a = ap.parse_args()

    mesh, mat, P = setup(a.h)
    nt = mesh.tris.shape[0]; A = mesh.area; Asum = A.sum()
    Filt = cone_filter(mesh, a.rmin)
    Px, Py = curl_operators(mesh)

    sx = np.arange(a.rbf_spc / 2, LX, a.rbf_spc)
    sy = np.arange(a.rbf_spc / 2, LY, a.rbf_spc)
    SX, SY = np.meshgrid(sx, sy)
    rbf = CSRBFMapping(_Adapter(mesh), (SX.ravel(), SY.ravel()), r_s=1.6 * a.rbf_spc)
    Mth = rbf.M

    z = np.full(nt, a.init if a.init is not None else a.vol); thh = np.zeros(Mth)
    if a.warm:
        wd = np.load(a.warm)
        z = (wd["zopt"] if "zopt" in wd else wd["z"]).copy()
    m_solid = RHO_F * (LX * LY)
    # passive-SOLID rings wrapping every hole -- the two bolt mounts and, critically,
    # the tip LOAD hole -- so the rim is fully backed by material and the service load
    # is reacted.  Without this the frequency objective strips this tip antinode and
    # the qp stress relaxation (void = no stress) lets it, leaving the load unbacked.
    cx, cy = mesh.cx, mesh.cy
    ring = np.zeros(nt, bool)
    for hx, hy, hr in HOLES:
        dpc = np.hypot(cx - hx, cy - hy)
        ring |= (dpc > hr - 1e-6) & (dpc < hr + a.ring)
    # optional forced-VOID mask (forbid the neutral root appendages)
    void_f = np.load(a.void)["void"].astype(bool) if a.void else np.zeros(nt, bool)
    ring &= ~void_f                                    # void wins over ring if overlap
    active = (~ring & ~void_f).astype(float)
    # mirror map about the mid-height axis y=LY/2: element e <-> element nearest to
    # its reflection (cx, LY-cy).  Density is symmetrized (y_e = y_mir), the fiber
    # ANGLE is antisymmetrized (theta_e = -theta_mir, the reflection of a director),
    # so the design respects the problem's top-bottom symmetry exactly.
    mir = cKDTree(np.column_stack([cx, cy])).query(
        np.column_stack([cx, LY - cy]))[1] if a.sym else np.arange(nt)

    def sym_d(v):                                  # symmetric part (density)
        return 0.5 * (v + v[mir]) if a.sym else v

    def sym_a(v):                                  # antisymmetric part (fiber angle)
        return 0.5 * (v - v[mir]) if a.sym else v

    def apply_passive(yv):
        return np.where(ring, 1.0, np.where(void_f, 0.0, yv)), active

    # EQUAL-MASS baseline: a uniform-density plate with the same holes, solid rings and
    # straight fibers, whose ACTIVE density is chosen so its total volume fraction
    # equals a given budget -- i.e. it weighs the same as the design it is compared to.
    # Every reported ratio is then a pure shape+fiber effect at IDENTICAL mass.  With a
    # mass PENALTY (not an equality constraint) the design settles at its own efficient
    # volume, so the baseline is rebuilt at that achieved volume for an honest ratio.
    ring_area = A[ring].sum(); active_area = A[~ring].sum()

    def equal_mass_baseline(vfrac):
        ru = float(np.clip((vfrac * Asum - ring_area) / max(active_area, 1e-9),
                           0.02, 1.0))
        yb = np.where(ring, 1.0, ru)
        lamb, phib = P.modal(yb, np.zeros(nt), want_grad=False)
        cb = P.static(yb, np.zeros(nt), False)[0]
        pnb, svmb = P.stress(yb, np.zeros(nt), pnorm=a.pnorm, qrelax=a.qrelax,
                             want_grad=False)
        mb = float((A @ (yb * RHO_F + (1 - yb) * RHO_MIN)) / m_solid)
        return dict(rho_u=ru, y0=yb, lam1=float(lamb[0]), lam2=float(lamb[1]),
                    phi0=phib, c0=cb, pn0=float(pnb), svm0=svmb, mass0=mb)

    base = equal_mass_baseline(a.vol)                # drives the ratio scales
    rho_u = base['rho_u']; y0 = base['y0']; lam1_0 = base['lam1']; phi0 = base['phi0']
    lam2_0 = base['lam2']
    c0 = base['c0']; PN0 = base['pn0']; svm0 = base['svm0']; mass0 = base['mass0']
    print(f"bracket {mesh.nnode} nodes, {nt} tris; {Mth} supports; volume cap "
          f"{a.vol}; equal-mass baseline rho_u={rho_u:.3f}, omega_1="
          f"{np.sqrt(lam1_0):.4f}, c0={c0:.3e}, mass={mass0:.3f}x solid", flush=True)

    stage = max(a.iters // 7, 1)
    mma = MMA(np.concatenate([np.zeros(nt), np.full(Mth, -np.pi / 2)]),
              np.concatenate([np.ones(nt), np.full(Mth, np.pi / 2)]),
              move=0.06, adapt=True)
    best = dict(J=np.inf); hist = []; mu_curl = a.w_curl
    lam = np.array([lam1_0, lam2_0]); phi = np.zeros(mesh.ndof)   # for the modal-skip
    t0 = time.time()
    for it in range(a.iters):
        beta = min(a.betamax, 2.0 ** (it // stage))
        yf = sym_d(Filt @ z)                              # symmetrized density field
        # robust erode/dilate: objectives on the ERODED field (thin members that
        # vanish under erosion are penalized -> minimum length scale, no ragged bits);
        # the intermediate field carries the volume/mass and grey measures.
        ye = project(yf, beta, 0.5 + a.deta); dpre = dproject(yf, beta, 0.5 + a.deta)
        yi = project(yf, beta, 0.5); dpri = dproject(yf, beta, 0.5)
        y, act = apply_passive(ye)                        # eroded, for the objectives
        yiv = apply_passive(yi)[0]                         # intermediate, for volume
        th = sym_a(rbf.Phi @ thh)                          # antisymmetrized fiber angle
        # ADAPTIVE curl weight: drive the fiber curvature to the printability limit
        # zeta_all and no further, so the tows use the FULL manufacturability budget
        # and exploit the anisotropy maximally instead of being over-smoothed flat.
        zmax = float(director_curl(th, Px, Py)[y > 0.5].max()) if (y > 0.5).any() else 0.0
        if zmax > a.zeta_all:
            mu_curl = min(mu_curl * 1.10, 1e3)
        else:
            mu_curl = max(mu_curl * 0.95, 1e-5)
        wcurl = mu_curl

        # modal solve: with mode-2 (modal_two) when omega_2 is in the objective; else
        # the fundamental only, and SKIPPED entirely on non-reporting iterations when
        # frequency is not optimized (w_eig=0), since it is then pure reporting.
        if a.w_eig2 != 0.0:
            lam, phi, phi2, dlz, dlt, dlz2, dlt2 = P.modal_two(y, th)
        elif a.w_eig > 0:
            lam, phi, dlz, dlt = P.modal(y, th); dlz2 = dlt2 = 0.0
        elif it % 20 == 0 or it == a.iters - 1:
            lam, phi = P.modal(y, th, want_grad=False); dlz = dlt = dlz2 = dlt2 = 0.0
        else:
            dlz = dlt = dlz2 = dlt2 = 0.0                  # reuse previous lam/phi
        PN, svmr, dsz, dst = P.stress(y, th, pnorm=a.pnorm, qrelax=a.qrelax)
        if a.w_comp > 0:
            c, _, dcz, dct = P.static(y, th, want_grad=True)
        else:
            c = P.static(y, th, want_grad=False)[0]; dcz = dct = 0.0
        vol = float((A @ yiv) / Asum); grey = float((A @ (yi * (1 - yi))) / Asum)
        Mnd = float((A @ (4 * yi * (1 - yi))) / Asum)
        mass = float((A @ (yiv * RHO_F + (1 - yiv) * RHO_MIN)) / m_solid)
        Pi, dPi = curl_penalty(th, Px, Py, A); Pi /= Asum; dPi /= Asum
        lr = float(lam[0]) / lam1_0; sr = PN / PN0; cr = c / c0
        lr2 = float(lam[1]) / lam2_0                      # second-mode ratio
        # MAXIMIZE omega_1 (and optionally MINIMIZE omega_2, --w-eig2) and MINIMIZE peak
        # stress / compliance.  Volume is held at a.vol by the equality penalty
        # cvol*(vol-a.vol)^2; a high stress/compliance weight steers the fixed budget
        # into THICK load-bearing members rather than dead stiffeners.
        J = (-a.w_eig * lr + a.w_eig2 * lr2 + a.w_stress * sr + a.w_comp * cr
             + a.w_vol * vol + a.cvol * (vol - a.vol) ** 2 + a.w_grey * grey + wcurl * Pi)
        gyf = (act * dpre * (-a.w_eig * dlz / lam1_0 + a.w_eig2 * dlz2 / lam2_0
                             + a.w_stress * dsz / PN0 + a.w_comp * dcz / c0)
               + act * dpri * ((a.w_vol + 2 * a.cvol * (vol - a.vol)) * (A / Asum)
                               + a.w_grey * (1 - 2 * yi) * (A / Asum)))
        dz = Filt.T @ sym_d(gyf)                          # backprop through symmetry
        gth = (-a.w_eig * dlt / lam1_0 + a.w_eig2 * dlt2 / lam2_0
               + a.w_stress * dst / PN0 + a.w_comp * dct / c0 + wcurl * dPi)
        dth = rbf.Phi.T @ sym_a(gth)

        fr = np.sqrt(lr); fr2 = np.sqrt(max(lr2, 0.0))
        hist.append(dict(fr=fr, fr2=fr2, gap=float(lam[1] / lam[0]), c=cr, sr=sr,
                         vol=vol, mass=mass, Mnd=Mnd, J=J))
        if (abs(vol - a.vol) < 0.03 and beta >= a.betamax * 0.9 and Mnd < 0.1
                and J < best['J']):
            best = dict(J=J, z=z.copy(), y=yiv.copy(), th=th.copy(), fr=fr, fr2=fr2,
                        c=cr, sr=sr, vol=vol, mass=mass, Mnd=Mnd, lam=lam.copy())
        if it % 20 == 0 or it == a.iters - 1:
            ncomp, fmax, fgnd = component_report(mesh, yiv > 0.5)
            print(f"[{it:3d}] beta={beta:5.1f} w1/w1_0={fr:.4f} w2/w2_0={fr2:.3f} "
                  f"gap21={lam[1]/lam[0]:.2f} stress/PN0={sr:.3f} c/c0={cr:.3f} "
                  f"vol={vol:.3f} Mnd={Mnd:.3f} |z|max={zmax:5.2f} muC={mu_curl:.3f} "
                  f"comps={ncomp} grnd={fgnd:.3f}", flush=True)
        W = mma.update(np.concatenate([z, thh]), np.concatenate([dz, dth]), f=J)
        z, thh = W[:nt], W[nt:]
    r = best if np.isfinite(best['J']) else dict(
        z=z, y=apply_passive(project(Filt @ z, a.betamax, 0.5))[0],
        th=rbf.Phi @ thh, fr=fr, fr2=fr2, c=cr, sr=sr, vol=vol, mass=mass, Mnd=Mnd,
        lam=lam.copy())
    # CONNECTIVITY GUARANTEE: keep only material grounded to the clamp, so the final
    # design has no floating islands.  Re-evaluate the metrics on the cleaned design.
    ncomp0, fmax0, fgnd0 = component_report(mesh, r['y'] > 0.5)
    gm = grounded_mask(mesh, r['y'] > 0.5)
    removed = float(mesh.area[(r['y'] > 0.5) & ~gm].sum() / mesh.area.sum())
    r['y'] = np.where((r['y'] > 0.5) & ~gm, 0.0, r['y'])   # drop floating solid
    ncomp1 = component_report(mesh, r['y'] > 0.5)[0]
    print(f"connectivity: {ncomp0} component(s), largest {100*fmax0:.1f}% area, "
          f"{100*fgnd0:.1f}% grounded -> removed {100*removed:.2f}% floating, "
          f"now {ncomp1} grounded component(s)", flush=True)
    # rebuild the EQUAL-MASS baseline at the design's ACHIEVED volume, so the reported
    # ratios compare against a uniform plate of the same (final) mass.
    lam1_up, c0_up, PN0_up = lam1_0, c0, PN0            # scales used during the loop
    v_ach = float(A @ r['y'] / Asum)
    base = equal_mass_baseline(v_ach)
    lam1_0 = base['lam1']; lam2_0 = base['lam2']; c0 = base['c0']; PN0 = base['pn0']
    y0, phi0, svm0, mass0, rho_u = (base['y0'], base['phi0'], base['svm0'],
                                    base['mass0'], base['rho_u'])
    lam_c, phi, phi2 = P.modal_two(r['y'], r['th'], want_grad=False)
    r['fr'] = float(np.sqrt(lam_c[0] / lam1_0)); r['lam'] = lam_c
    r['fr2'] = float(np.sqrt(lam_c[1] / lam2_0)); r['gap'] = float(lam_c[1] / lam_c[0])
    r['c'] = P.static(r['y'], r['th'], False)[0] / c0
    r['sr'] = P.stress(r['y'], r['th'], pnorm=a.pnorm, qrelax=a.qrelax,
                       want_grad=False)[0] / PN0
    PNf, svmf = P.stress(r['y'], r['th'], pnorm=a.pnorm, qrelax=a.qrelax,
                         want_grad=False)
    print(f"optimized in {time.time()-t0:.0f}s: omega_1 up {r['fr']:.3f}x "
          f"({100*(r['fr']-1):.1f}%), omega_2 {r['fr2']:.3f}x, gap w2/w1={r['gap']:.2f}, "
          f"peak stress {r['sr']:.3f}x, c/c0={r['c']:.3f}, mass={r['mass']:.3f}x "
          f"(baseline {mass0:.3f}x, rho_u={rho_u:.3f}), Mnd={r['Mnd']:.3f}", flush=True)

    np.savez(a.out + ".npz", trix=mesh.npx[:, 0], triy=mesh.npx[:, 1],
             tris=mesh.tris, cx=mesh.cx, cy=mesh.cy, z=r['z'], zopt=r['z'],
             y=r['y'], th=r['th'],
             holes=np.array(HOLES), phi=phi, phi2=phi2, svm=svmf, lam=r['lam'],
             lam1_0=lam1_0, lam2_0=lam2_0, fr=r['fr'], fr2=r['fr2'], gap=r['gap'],
             s_ratio=r['sr'], c_ratio=r['c'], vol=r['vol'], mass=r['mass'],
             hist_fr=[h['fr'] * np.sqrt(lam1_up / lam1_0) for h in hist],
             hist_fr2=[h['fr2'] for h in hist], hist_gap=[h['gap'] for h in hist],
             hist_s=[h['sr'] * PN0_up / PN0 for h in hist],
             hist_c=[h['c'] * c0_up / c0 for h in hist],
             hist_mass=[h['mass'] for h in hist], Lx=LX, Ly=LY,
             y0=y0, phi0=phi0, svm0=svm0, mass0=mass0, rho_u=rho_u,
             c0=c0, pn0=PN0)
    print("saved", a.out + ".npz", flush=True)


if __name__ == "__main__":
    main()
