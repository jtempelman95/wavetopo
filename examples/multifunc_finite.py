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

    opt = MultiFunctionalOptimizer(
        bp, band=args.band, n_bands=nb, R=args.R, kset=kset,
        w_stiff=args.wstiff, w_gap=args.wgap, v_target=args.vol,
        c_vol=80.0, period=(a, a))

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
def transmission_sweep(mesh, mat, z, th, omegas, *, eta, nbuf, n,
                       sponge_el=None, strength=30.0):
    xs, ys = mesh.node_xy()
    # drive a vertical line just inside the left buffer; probe the right buffer
    x_src = 0.35 * nbuf / n
    x_prb = mesh.Lx - 0.5 * nbuf / n
    src = np.flatnonzero(np.abs(xs - x_src) < 0.6 * mesh.dx)
    prb = (np.abs(xs - x_prb) < 0.9 * nbuf / n / 2)
    # quadratic sponge in the outermost part of each buffer
    marg = 0.55 * nbuf / n
    d = np.maximum((marg - mesh.cx) / marg, (mesh.cx - (mesh.Lx - marg)) / marg)
    c_el = np.maximum(0.0, d) ** 2 * strength if sponge_el is None else sponge_el

    T = []
    for om in omegas:
        hv = HarmonicValley(mesh, mat, omega=om, eta=eta)
        hv.set_design(z, th)
        hv.set_sponge(c_el)
        F = np.zeros(mesh.ndof, complex)
        F[2 * src] = 1.0                              # x-polarized line drive
        hv.set_load(F)
        u = hv.solve()
        ux, uy = u[0::2], u[1::2]
        T.append(float(np.sum(np.abs(ux[prb]) ** 2 + np.abs(uy[prb]) ** 2)))
        print(f"   w={om:7.3f}  T={T[-1]:.4e}", flush=True)
    return np.array(T), dict(src=src, prb=prb, c_el=c_el)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--band", type=int, default=3)
    ap.add_argument("--nk", type=int, default=5)
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--R", type=float, default=0.13)
    ap.add_argument("--wstiff", type=float, default=1.0)
    ap.add_argument("--wgap", type=float, default=1.0)
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
        T, aux = transmission_sweep(mesh, mat, z, th, omegas, eta=args.eta,
                                    nbuf=args.nbuf, n=n)
        out[f'{tag}_T'] = T
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
        T = out[f'{tag}_T']
        print(f"{tag}: in-gap mean T={T[ing].mean():.3e}  "
              f"out-of-gap mean T={T[~ing].mean():.3e}  "
              f"attenuation={T[~ing].mean()/max(T[ing].mean(),1e-300):.1f}x  "
              f"k_eff={out[f'{tag}_k_eff']:.4g}", flush=True)


if __name__ == "__main__":
    main()
