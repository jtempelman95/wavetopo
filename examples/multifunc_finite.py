r"""
Multi-functional metamaterial, carried through to a FINITE TRUNCATED STRUCTURE.

The co-design of \S multifunc optimizes a *periodic* cell: homogenized stiffness
C_H from the cell problem, and a complete band gap from Bloch analysis.  Both are
infinite-medium idealizations.  A part you could actually build is a finite tile
of that cell, with free edges, a finite number of periods, and no Bloch boundary
condition anywhere.  This script closes that gap:

  STAGE 1  co-design (z, theta) on one cell for stiffness AND complete gap,
           with a geometry-only run (theta frozen) as the control.
  STAGE 2  Bloch bands of the optimized cell -> the predicted gap [w_lo, w_hi].
  STAGE 3  TILE the cell into an Nx x Ny finite panel between two solid buffers.
  STAGE 4  STATIC: clamp the left buffer, pull the right one, measure the
           effective axial stiffness of the truncated panel.  This is the
           "strength" half, measured on the real part rather than homogenized.
  STAGE 5  DYNAMIC: drive the left buffer over a frequency sweep and measure
           energy transmitted to the right buffer.  If the design works, the
           finite panel attenuates inside the predicted gap and passes outside
           it -- with only a handful of periods and free top/bottom edges.

The point of stage 5 is that a Bloch gap is a statement about an infinite
crystal; whether ~6 periods of a truncated, edge-terminated panel actually stops
a wave is a separate question, and the only one a designer cares about.

    PYTHONPATH=. python examples/multifunc_finite.py --iters 90
    PYTHONPATH=. python examples/multifunc_finite.py --reuse     # skip stage 1
"""
import argparse
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from wavetopo.cfrp import Material, QuadMesh, CSRBFMapping
from wavetopo.bloch import BlochProblem, ibz_path_square
from wavetopo.homogenize import Homogenizer
from wavetopo.multifunc_opt import MultiFunctionalOptimizer
from wavetopo.harmonic import HarmonicValley

MAT = dict(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
           rho_f=1.6, rho_m=1.2)


# ------------------------------------------------------------------ #
#  Stage 1-2: co-design one cell, then its bands
# ------------------------------------------------------------------ #
def build_cell(n, ks, a=1.0):
    mesh = QuadMesh(n, n, a, a)
    mat = Material(**MAT)
    xs = np.linspace(0, a, ks, endpoint=False)
    SX, SY = np.meshgrid(xs, xs)
    rbf = CSRBFMapping(mesh, (SX.ravel(), SY.ravel()), r_s=2.0 * a / ks,
                       period=(a, a))
    return mesh, mat, rbf, BlochProblem(mesh, mat, rbf, penal=3.0)


def codesign(args, geom_only=False):
    a = 1.0
    mesh, mat, rbf, bp = build_cell(args.n, args.ks, a)
    kset, _, _ = ibz_path_square(a, a, n=args.nk)
    nb = args.band + 3

    rng = np.random.default_rng(0)
    z0 = np.clip(args.vol + 0.10 * rng.standard_normal(mesh.N), 0.15, 0.95)
    r2 = (mesh.cx - a / 2) ** 2 + (mesh.cy - a / 2) ** 2
    z0[r2 < (0.17 * a) ** 2] = 0.95
    th0 = 0.05 * rng.standard_normal(rbf.M)

    # Normalizers MUST be set explicitly.  Left to default they are taken from
    # the FIRST evaluation, where the seed cell has a badly closed gap: gap_ref
    # becomes |gap_0| ~ 3.4, so once the gap actually opens, improving it from
    # +0.17 to +0.48 changes J by only -0.09 while stiffness 12 -> 16.6 changes
    # it by -0.66.  The gap term goes numerically deaf exactly when it starts to
    # matter, and the optimizer trades the gap away for stiffness.  gap_ref is a
    # TARGET gap width, not the seed's.
    opt = MultiFunctionalOptimizer(
        bp, band=args.band, n_bands=nb, R=args.R, kset=kset,
        w_stiff=args.wstiff, w_gap=args.wgap, v_target=args.vol,
        c_vol=args.c_vol, period=(a, a),
        C_ref=args.C_ref, gap_ref=args.gap_ref)

    if geom_only:
        from wavetopo.cfrp_optimizer import MMA
        mma = MMA(np.zeros(mesh.N), np.ones(mesh.N), move=0.08, adapt=True)
        z, th = z0.copy(), th0.copy()
        best = dict(score=np.inf)
        for it in range(args.iters):
            frac = it / max(args.iters - 1, 1)
            if frac > 0.25:
                g = (frac - 0.25) / 0.75
                opt.beta = min(opt.beta_max, 1.0 * (opt.beta_max ** g))
            J, dz, dt, info = opt.evaluate(z, th)
            if abs(info['vol'] - args.vol) < 0.03 and J < best['score']:
                best = dict(score=J, z=z.copy(), th=th.copy(), info=dict(info))
            if it % 15 == 0:
                print(f"  [geom {it:3d}] C={info['C_bulk']:7.3f} "
                      f"gap={info['gap']:+.4f} vol={info['vol']:.3f}", flush=True)
            z = mma.update(z, dz, f=J)
        res = best if np.isfinite(best['score']) else dict(z=z, th=th, info=info)
    else:
        res = opt.run(z0, th0, max_iter=args.iters, move=0.08)

    y = opt._apply(res['z'])
    bp.assemble(y, res['th'])
    ppath, ticks, labels = ibz_path_square(a, a, n=28)
    bands = bp.band_structure(ppath, nb)
    C, _ = Homogenizer(bp).solve()
    b = args.band
    lo, hi = bands[:, b].max(), bands[:, b + 1].min()
    return dict(z=res['z'], y=y, th=res['th'],
                theta_elem=(rbf.Phi @ res['th']),
                bands=bands, ticks=ticks, labels=labels,
                C=C, C_bulk=Homogenizer.bulk(C),
                gap_lo=float(lo), gap_hi=float(hi), gap=float(hi - lo),
                vol=float(y.mean()), n=args.n)


# ------------------------------------------------------------------ #
#  Stage 3: tile the cell into a finite panel with solid buffers
# ------------------------------------------------------------------ #
def tile_panel(y_cell, th_cell, n, Nx, Ny, nbuf):
    """Return (mesh, z, theta) for a panel of Nx x Ny cells flanked, in x, by
    solid buffers ``nbuf`` elements wide.  Element index is e = ey*nelx + ex, so
    a cell field reshaped (n, n) tiles by np.tile and ravels straight back."""
    Z = np.tile(np.asarray(y_cell).reshape(n, n), (Ny, Nx))
    TH = np.tile(np.asarray(th_cell).reshape(n, n), (Ny, Nx))
    if nbuf:
        pad_z = np.ones((Ny * n, nbuf))
        pad_t = np.zeros((Ny * n, nbuf))
        Z = np.hstack([pad_z, Z, pad_z])
        TH = np.hstack([pad_t, TH, pad_t])
    nely, nelx = Z.shape
    mesh = QuadMesh(nelx, nely, nelx / n, nely / n)   # cell size a=1
    return mesh, Z.ravel(), TH.ravel()


# ------------------------------------------------------------------ #
#  Stage 4: static axial stiffness of the truncated panel
# ------------------------------------------------------------------ #
def static_stiffness(mesh, mat, z, th, penal=3.0):
    """Clamp x=0, apply unit total traction in +x on x=Lx, return the effective
    axial stiffness  k_eff = F_total / mean(u_x on the loaded edge)."""
    hv = HarmonicValley(mesh, mat, omega=0.0, eta=0.0, penal=penal)
    hv.set_design(z, th)
    K = hv.K.tocsr()
    xs, ys = mesh.node_xy()
    left = np.flatnonzero(xs <= 1e-12)
    right = np.flatnonzero(xs >= mesh.Lx - 1e-12)
    fixed = np.concatenate([2 * left, 2 * left + 1])
    free = np.setdiff1d(np.arange(mesh.ndof), fixed)
    F = np.zeros(mesh.ndof)
    F[2 * right] = 1.0 / len(right)                  # unit total load in x
    u = np.zeros(mesh.ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])
    dx = float(np.mean(u[2 * right]))
    return dict(k_eff=1.0 / dx if dx > 0 else np.nan, u=u, dx=dx)


# ------------------------------------------------------------------ #
#  Stage 5: transmission through the truncated panel
# ------------------------------------------------------------------ #
def _panel_setup(mesh, nbuf, n, strength=30.0):
    """Four-sided sponge, full-height plane-wave drive, full-height probe.

    The first version sponged only the x-ends and drove a short line.  Two
    artifacts followed.  (i) The free top/bottom edges let energy travel around
    the crystal as edge waves, so the panel was never really in the way.
    (ii) The two buffer/crystal interfaces formed a Fabry-Perot cavity, making
    transmission oscillate with panel length -- the measured isolation was
    NON-MONOTONIC in the number of periods (0.6, 0.4, 1.4, 2.3, 7.7, 3.4, 5.4),
    which is impossible for evanescent decay and is the signature of a resonant
    measurement rather than a transport one.
    """
    mx = 0.55 * nbuf / n                       # x sponge depth (in the buffers)
    my = 0.45 * mesh.Ly / 6.0                  # y sponge depth
    dx_ = np.maximum((mx - mesh.cx) / mx, (mesh.cx - (mesh.Lx - mx)) / mx)
    dy_ = np.maximum((my - mesh.cy) / my, (mesh.cy - (mesh.Ly - my)) / my)
    c_el = np.maximum(0.0, np.maximum(dx_, dy_)) ** 2 * strength
    xs, ys = mesh.node_xy()
    x_src, x_prb = 0.75 * nbuf / n, mesh.Lx - 0.55 * nbuf / n
    inner = (ys > my) & (ys < mesh.Ly - my)    # exclude the y sponge
    src = np.flatnonzero((np.abs(xs - x_src) < 0.6 * mesh.dx) & inner)
    prb = (np.abs(xs - x_prb) < 0.9 * mesh.dx) & inner
    return c_el, src, prb


def _solve_T(mesh, mat, z, th, om, eta, c_el, src, prb):
    hv = HarmonicValley(mesh, mat, omega=om, eta=eta)
    hv.set_design(z, th)
    hv.set_sponge(c_el)
    F = np.zeros(mesh.ndof, complex)
    F[2 * src] = 1.0                            # x-polarized plane-wave drive
    hv.set_load(F)
    u = hv.solve()
    T = float(np.sum(np.abs(u[0::2][prb]) ** 2 + np.abs(u[1::2][prb]) ** 2))
    return u, T


def transmission_sweep(mesh, mat, z, th, omegas, *, eta, nbuf, n,
                       reference=True):
    """Return (T, T_ref, IL).  ``IL = T_ref/T`` is INSERTION LOSS: transmission
    through the panel divided by transmission through a solid panel of identical
    geometry, source, sponge and mesh.  Normalizing this way divides out source
    coupling, buffer reflection and sponge absorption, all of which are common to
    both runs, and leaves what the crystal itself does.  Raw T alone conflates
    the crystal with its fixturing."""
    c_el, src, prb = _panel_setup(mesh, nbuf, n)
    z_ref = np.ones_like(z)
    th_ref = np.zeros_like(th)
    T, Tr = [], []
    for om in omegas:
        _, t = _solve_T(mesh, mat, z, th, om, eta, c_el, src, prb)
        T.append(t)
        if reference:
            _, tr = _solve_T(mesh, mat, z_ref, th_ref, om, eta, c_el, src, prb)
            Tr.append(tr)
            print(f"   w={om:7.3f}  T={t:.4e}  T_solid={tr:.4e}  "
                  f"IL={tr/max(t,1e-300):8.1f}x", flush=True)
        else:
            print(f"   w={om:7.3f}  T={t:.4e}", flush=True)
    T = np.array(T)
    Tr = np.array(Tr) if reference else np.full_like(T, np.nan)
    return T, Tr, Tr / np.maximum(T, 1e-300)


def quad_to_tri(mesh):
    """Split each quad into two triangles -> (trix, triy, tris) in the layout
    examples/figures_playground.py expects (mtri.Triangulation(trix,triy,tris))."""
    nx, ny = mesh.nelx, mesh.nely
    xs, ys = mesh.node_xy()
    nid = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    n1 = nid[:-1, :-1].ravel(); n2 = nid[:-1, 1:].ravel()
    n3 = nid[1:, 1:].ravel();   n4 = nid[1:, :-1].ravel()
    tris = np.vstack([np.column_stack([n1, n2, n3]),
                      np.column_stack([n1, n3, n4])])
    return xs, ys, tris


def save_playground(path, mesh, z, th, fields, scalars):
    """Write an npz in the figures_playground schema: a triangulation, element
    centroids and orientation for the toolpath/director plotters, nodal field
    magnitudes, and scalars."""
    trix, triy, tris = quad_to_tri(mesh)
    out = dict(trix=trix, triy=triy, tris=tris,
               cent=np.column_stack([mesh.cx, mesh.cy]),
               thopt=np.asarray(th), zdens=np.asarray(z),
               nelx=mesh.nelx, nely=mesh.nely, Lx=mesh.Lx, Ly=mesh.Ly)
    out.update({k: np.asarray(v) for k, v in fields.items()})
    out.update({k: np.asarray(v) for k, v in scalars.items()})
    np.savez(path, **out)
    print("saved", path, f"({len(tris)} cells, {len(out)} keys)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--band", type=int, default=3)
    ap.add_argument("--nk", type=int, default=5)
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--R", type=float, default=0.13)
    ap.add_argument("--wstiff", type=float, default=1.0)
    ap.add_argument("--wgap", type=float, default=3.0)
    ap.add_argument("--C-ref", dest="C_ref", type=float, default=7.0,
                    help="stiffness normalizer (see codesign(); do NOT leave "
                         "the optimizer to infer it from the seed)")
    ap.add_argument("--gap-ref", dest="gap_ref", type=float, default=0.30,
                    help="TARGET gap width used to normalize the gap term")
    ap.add_argument("--c-vol", dest="c_vol", type=float, default=1500.0,
                    help="volume penalty.  MUST scale with the gap term: with "
                         "w_gap/gap_ref = 10 the gap term reaches ~-18, against "
                         "which the old c_vol=80 contributes only 0.9 at a "
                         "volume error of 0.107 -- so volume drifts to 0.61, "
                         "every strong late iterate is rejected as infeasible "
                         "by run()'s |vol-target|<0.03 test, and the returned "
                         "'best feasible' design is a weak early one.")
    ap.add_argument("--vol", type=float, default=0.5)
    ap.add_argument("--Nx", type=int, default=6)
    ap.add_argument("--Ny", type=int, default=4)
    ap.add_argument("--nbuf", type=int, default=28)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--nw", type=int, default=41)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--out", default="results/multifunc_finite")
    args = ap.parse_args()

    dpath = args.out + "_design.npz"
    if args.reuse and os.path.exists(dpath):
        d = dict(np.load(dpath, allow_pickle=True))
        print("reused", dpath, flush=True)
    else:
        t0 = time.time()
        print("STAGE 1  co-design (stiffness + complete gap)", flush=True)
        co = codesign(args, geom_only=False)
        print(f"  co-design : C_bulk={co['C_bulk']:.3f} gap={co['gap']:+.4f} "
              f"[{co['gap_lo']:.3f},{co['gap_hi']:.3f}] vol={co['vol']:.3f}",
              flush=True)
        print("STAGE 1b geometry-only control (theta frozen)", flush=True)
        ge = codesign(args, geom_only=True)
        print(f"  geom-only : C_bulk={ge['C_bulk']:.3f} gap={ge['gap']:+.4f} "
              f"[{ge['gap_lo']:.3f},{ge['gap_hi']:.3f}] vol={ge['vol']:.3f}",
              flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
        d = {f"co_{k}": v for k, v in co.items()}
        d.update({f"ge_{k}": v for k, v in ge.items()})
        np.savez(dpath, **d)
        print("saved", dpath, flush=True)

    mat = Material(**MAT)
    n = int(d['co_n'])
    lo, hi = float(d['co_gap_lo']), float(d['co_gap_hi'])
    print(f"\nSTAGE 2  predicted complete gap: [{lo:.4f}, {hi:.4f}] "
          f"width {hi-lo:+.4f}", flush=True)

    print("STAGE 3  tiling finite panel "
          f"{args.Nx}x{args.Ny} cells + {args.nbuf}-element buffers", flush=True)
    out = {}
    for tag in ("co", "ge"):
        mesh, z, th = tile_panel(d[f'{tag}_y'], d[f'{tag}_theta_elem'], n,
                                 args.Nx, args.Ny, args.nbuf)
        print(f"  [{tag}] panel {mesh.nelx}x{mesh.nely} el "
              f"({mesh.Lx:.1f} x {mesh.Ly:.1f} cells), vol={z.mean():.3f}",
              flush=True)
        st = static_stiffness(mesh, mat, z, th)
        print(f"  [{tag}] STAGE 4 static k_eff = {st['k_eff']:.4g}", flush=True)
        out[f'{tag}_k_eff'] = st['k_eff']
        out[f'{tag}_z'] = z
        out[f'{tag}_th'] = th
        out[f'{tag}_nelx'] = mesh.nelx
        out[f'{tag}_nely'] = mesh.nely

    # dynamic sweep on the co-designed panel (and the control, for contrast).
    # Sort the edges: a design that failed to open a gap reports hi < lo, which
    # would otherwise build a descending linspace and an empty in-gap mask.
    w_lo, w_hi = min(lo, hi), max(lo, hi)
    if hi <= lo:
        print(f"  WARNING: no complete gap (width {hi-lo:+.4f}).  The finite-panel "
              f"demo has nothing to demonstrate; run more iterations.", flush=True)
    span = max(w_hi - w_lo, 0.05)
    omegas = np.linspace(max(w_lo - 1.6 * span, 0.05), w_hi + 1.6 * span, args.nw)
    for tag in ("co", "ge"):
        mesh, z, th = tile_panel(d[f'{tag}_y'], d[f'{tag}_theta_elem'], n,
                                 args.Nx, args.Ny, args.nbuf)
        print(f"\nSTAGE 5  transmission sweep [{tag}]", flush=True)
        T, Tr, IL = transmission_sweep(mesh, mat, z, th, omegas, eta=args.eta,
                                       nbuf=args.nbuf, n=n)
        out[f'{tag}_T'] = T
        out[f'{tag}_Tref'] = Tr
        out[f'{tag}_IL'] = IL
    # ---- playground-compatible export (figures_playground.py schema) ---- #
    w_mid = 0.5 * (w_lo + w_hi)
    outside = (omegas < w_lo) | (omegas > w_hi)
    w_pass = float(omegas[outside][np.argmax(out['co_T'][outside])])
    mesh, z, th = tile_panel(d['co_y'], d['co_theta_elem'], n,
                             args.Nx, args.Ny, args.nbuf)
    c_el, src, prb = _panel_setup(mesh, args.nbuf, n)
    fields = {}
    for key, om in (("mgap", w_mid), ("mpass", w_pass)):
        u, _ = _solve_T(mesh, mat, z, th, om, args.eta, c_el, src, prb)
        fields[key] = np.sqrt(np.abs(u[0::2])**2 + np.abs(u[1::2])**2)
    # solid reference at the in-gap frequency, for a like-for-like visual
    u, _ = _solve_T(mesh, mat, np.ones_like(z), np.zeros_like(th), w_mid,
                    args.eta, c_el, src, prb)
    fields['msolid'] = np.sqrt(np.abs(u[0::2])**2 + np.abs(u[1::2])**2)
    st = static_stiffness(mesh, mat, z, th)
    fields['ustatic'] = np.sqrt(st['u'][0::2]**2 + st['u'][1::2]**2)
    save_playground(
        args.out + "_data.npz", mesh, z, th, fields,
        dict(omega=omegas, T_co=out['co_T'], T_ge=out['ge_T'],
             gap_lo=lo, gap_hi=hi, w_gap=w_mid, w_pass=w_pass,
             IL_co=out['co_IL'], IL_ge=out['ge_IL'],
             Tref_co=out['co_Tref'],
             k_eff_co=out['co_k_eff'], k_eff_ge=out['ge_k_eff'],
             C_bulk_co=float(d['co_C_bulk']), C_bulk_ge=float(d['ge_C_bulk']),
             gap_co=float(d['co_gap']), gap_ge=float(d['ge_gap']),
             cell_y=np.asarray(d['co_y']), cell_th=np.asarray(d['co_theta_elem']),
             cell_n=n, bands=np.asarray(d['co_bands']),
             Nx=args.Nx, Ny=args.Ny, nbuf=args.nbuf))

    out['omega'] = omegas
    out['gap_lo'], out['gap_hi'] = lo, hi
    out['ge_gap_lo'], out['ge_gap_hi'] = float(d['ge_gap_lo']), float(d['ge_gap_hi'])
    np.savez(args.out + ".npz", **out)
    print("\nsaved", args.out + ".npz", flush=True)

    ing = (omegas > w_lo) & (omegas < w_hi)
    if not ing.any():
        print("no frequencies fall inside the predicted gap; nothing to report")
        return
    for tag in ("co", "ge"):
        T, IL = out[f'{tag}_T'], out[f'{tag}_IL']
        print(f"{tag}: in-gap IL={IL[ing].mean():8.1f}x   "
              f"out-of-gap IL={IL[~ing].mean():6.1f}x   "
              f"peak IL={IL.max():.1f}x @ w={omegas[IL.argmax()]:.3f}   "
              f"k_eff={out[f'{tag}_k_eff']:.4g}", flush=True)


if __name__ == "__main__":
    main()
