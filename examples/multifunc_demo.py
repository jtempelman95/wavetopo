"""
CHALLENGE PROBLEM -- multi-functional metamaterial:
maximise static stiffness AND open a complete phononic band gap, on one cell.

The claim under test is not "we can do both a bit"; it is that CO-DESIGNING
toolpath with geometry beats geometry alone.  So every run is executed twice:

    --codesign   density z AND fiber orientation theta are both free
    --geom-only  theta is frozen (straight fibers); only z is designed

Sweeping the stiffness/gap weight traces a Pareto front for each.  If the
co-design front lies strictly outside the geometry-only front, orientation is
buying something geometry cannot -- it retunes the stiffness ANISOTROPY without
removing material, so the bands move at nearly fixed density.

Band polarity of the gap edges is measured (not optimised) at Gamma via the
inversion parity <phi, I phi>, so the gap can be labelled by the symmetry of the
modes bounding it.

NOTE the starting design is deliberately PERTURBED.  A uniform cell is highly
symmetric and its Gamma modes are four-fold degenerate, where the simple-
eigenvalue derivative phi^H(dK - lambda dM)phi does not apply and the first
gradients are meaningless.

    PYTHONPATH=. python examples/multifunc_demo.py --wstiff 1 --wgap 1 --tag A
"""
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.cfrp import Material, QuadMesh, CSRBFMapping
from wavetopo.bloch import BlochProblem, ibz_path_square
from wavetopo.homogenize import Homogenizer
from wavetopo.multifunc_opt import MultiFunctionalOptimizer


def build(n, ks, a=1.0):
    mesh = QuadMesh(n, n, a, a)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    xs = np.linspace(0, a, ks, endpoint=False)
    SX, SY = np.meshgrid(xs, xs)
    rbf = CSRBFMapping(mesh, (SX.ravel(), SY.ravel()), r_s=2.0*a/ks,
                       period=(a, a))
    return mesh, mat, rbf, BlochProblem(mesh, mat, rbf, penal=3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--band", type=int, default=3)
    ap.add_argument("--nk", type=int, default=5)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--R", type=float, default=0.13)
    ap.add_argument("--wstiff", type=float, default=1.0)
    ap.add_argument("--wgap", type=float, default=1.0)
    ap.add_argument("--vol", type=float, default=0.5)
    ap.add_argument("--geom-only", action="store_true",
                    help="freeze theta: geometry-only baseline")
    ap.add_argument("--tag", default="")
    a = 1.0
    args = ap.parse_args()
    mesh, mat, rbf, bp = build(args.n, args.ks, a)
    kset, _, _ = ibz_path_square(a, a, n=args.nk)
    ppath, ticks, labels = ibz_path_square(a, a, n=24)
    nb = args.band + 3

    # perturbed start: breaks the Gamma degeneracy (see module docstring)
    rng = np.random.default_rng(0)
    z0 = np.clip(args.vol + 0.10*rng.standard_normal(mesh.N), 0.15, 0.95)
    r2 = (mesh.cx - a/2)**2 + (mesh.cy - a/2)**2
    z0[r2 < (0.17*a)**2] = 0.95
    th0 = 0.05*rng.standard_normal(rbf.M)

    opt = MultiFunctionalOptimizer(
        bp, band=args.band, n_bands=nb, R=args.R, kset=kset,
        w_stiff=args.wstiff, w_gap=args.wgap,
        v_target=args.vol, c_vol=80.0, period=(a, a))

    # baseline numbers for the record
    _, info0 = opt.evaluate(z0, th0, want_grad=False)
    bp.assemble(opt._apply(z0), th0)
    bands0 = bp.band_structure(ppath, nb)
    print(f"start: C_bulk={info0['C_bulk']:.3f} gap={info0['gap']:+.4f} "
          f"(geom_only={args.geom_only})", flush=True)

    t0 = time.time()
    if args.geom_only:
        # freeze theta by zeroing its gradient every iteration
        from wavetopo.cfrp_optimizer import MMA
        N = mesh.N
        mma = MMA(np.zeros(N), np.ones(N), move=0.08, adapt=True)
        z, th = z0.copy(), th0.copy()
        best = dict(score=np.inf); hist = []
        for it in range(args.iters):
            J, dz, dt, info = opt.evaluate(z, th)
            hist.append(dict(C=info['C_bulk'], gap=info['gap'], vol=info['vol'], J=J))
            if abs(info['vol']-args.vol) < 0.03 and J < best['score']:
                best = dict(score=J, z=z.copy(), th=th.copy(), info=dict(info))
            if it % 10 == 0 or it == args.iters-1:
                print(f"[{it:3d}] C_bulk={info['C_bulk']:8.3f} gap={info['gap']:+.4f} "
                      f"vol={info['vol']:.3f} J={J:+.4f}", flush=True)
            z = mma.update(z, dz, f=J)
        res = best if np.isfinite(best['score']) else dict(z=z, th=th, info=info)
        res['hist'] = hist
    else:
        res = opt.run(z0, th0, max_iter=args.iters, move=0.08)
    print(f"optimized in {time.time()-t0:.0f}s", flush=True)

    z, th = res['z'], res['th']
    y = opt._apply(z)
    bp.assemble(y, th)
    bands1 = bp.band_structure(ppath, nb)
    H = Homogenizer(bp); C, _ = H.solve()
    Cb = Homogenizer.bulk(C)
    b = args.band
    gap_true = bands1[:, b+1].min() - bands1[:, b].max()
    gap0 = bands0[:, b+1].min() - bands0[:, b].max()
    wg, _, Vg, _ = bp.bands_at_k(0.0, 0.0, nb, return_vec=True)
    par = [opt.parity(Vg[:, j]) for j in range(nb)]
    print(f"FINAL  C_bulk {info0['C_bulk']:.3f} -> {Cb:.3f} | "
          f"gap (fine path) {gap0:+.4f} -> {gap_true:+.4f} | vol={res['info']['vol']:.3f}")
    print(f"       gap-edge parity at Gamma: band {b} P={par[b]:+.2f}, "
          f"band {b+1} P={par[b+1]:+.2f}", flush=True)

    tag = args.tag or ("geom" if args.geom_only else "co")
    np.savez(f"results/multifunc_{tag}.npz",
             z=y, theta=bp._theta.copy(), bands0=bands0, bands1=bands1,
             C=C, C_bulk=Cb, C_bulk0=info0['C_bulk'],
             gap=gap_true, gap0=gap0, parity=np.array(par), w_gamma=wg,
             hist_C=[h['C'] for h in res['hist']],
             hist_gap=[h['gap'] for h in res['hist']],
             nelx=mesh.nelx, nely=mesh.nely, a=a, band=b,
             ks=args.ks, R=args.R, wstiff=args.wstiff, wgap=args.wgap,
             geom_only=args.geom_only, vol=res['info']['vol'])
    print(f"saved results/multifunc_{tag}.npz")


if __name__ == "__main__":
    main()
