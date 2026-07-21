"""
Topological protection test: valley waveguide vs. wall defects.

The mirror-partner (alpha = 15deg | 45deg) valley-Hall domain wall of the
triangular-rod crystal is deliberately damaged and the guided kink-mode
transmission is measured:

  * pristine straight wall,
  * one rod removed on the wall (vacancy),
  * a cluster of rods next to the wall randomly rotated (orientation
    disorder, +/- 12 deg).

Because backscattering requires inter-valley (K -> K') coupling that smooth
defects do not supply, the guided transmission should survive within a small
factor -- the practical signature of topological protection, and the property
that distinguishes the kink mode from an ordinary defect-line guide.

Two corrections relative to the original version of this script:

  * The mesh now CONFORMS to the rods (``wavetopo.valley_guide``) instead of
    assigning phase by element centroid.  The painted construction gave each rod
    a random ~1% area error and displaced its effective centre by up to 3% of R,
    which randomly perturbs the very C3 symmetry that sets the Dirac mass.
  * The frequency must lie in the GUIDED BAND of the conforming crystal, which
    ``examples/valley_band_convergence.py`` locates.  The previous default
    (omega = 7.28) was chosen from a painted sweep; on a conforming mesh the
    band sits at omega ~ 7.13-7.87, so a frequency read off a painted mesh is
    not transferable.

A defect test is only meaningful if the pristine transmission is well clear of
the floor -- otherwise all three configurations report noise and "survives
damage" is vacuous.  This script refuses to run if that is not the case.

    python examples/valley_robust.py --wmid 7.2
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.valley_guide import (tri_rod_layout, conforming_mesh, painted_mesh,
                                   wall_polyline, wall_side_fn, quadratic_sponge,
                                   nodal_areas, solve_transmission)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.09)
    ap.add_argument("--Lx", type=float, default=20.0)
    ap.add_argument("--Ly", type=float, default=17.0)
    ap.add_argument("--R", type=float, default=0.44)
    ap.add_argument("--muf", type=float, default=100.0)
    ap.add_argument("--eta", type=float, default=0.003)
    ap.add_argument("--wmid", type=float, default=7.2,
                    help="drive frequency; must be inside the guided band")
    ap.add_argument("--wfloor", type=float, default=6.6,
                    help="out-of-band control frequency (defines the floor)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--mesh", choices=["conforming", "painted"],
                    default="conforming")
    ap.add_argument("--min-contrast", type=float, default=50.0,
                    help="required pristine in-band / out-of-band ratio")
    ap.add_argument("--out", default="results/valley_robust.png")
    args = ap.parse_args()

    poly = wall_polyline(args.Lx, args.Ly)
    side = wall_side_fn(poly)
    mid = 0.5 * (poly[0] + poly[-1])

    rng = np.random.default_rng(args.seed)
    a1v = np.array([1.0, 0.0])
    a2v = np.array([0.5, np.sqrt(3) / 2])
    near = [mid + d for d in (0.6 * a1v, -0.6 * a1v, 0.9 * a2v, -0.9 * a2v)]
    disorder = {tuple(s): float(rng.uniform(-12, 12)
                                + (15.0 if side(s) else 45.0)) for s in near}

    configs = {
        "pristine wall": dict(skip_sites=(), angle_override=None),
        "missing rod (vacancy)": dict(skip_sites=(mid,), angle_override=None),
        "orientation disorder (+/-12 deg)": dict(skip_sites=(),
                                                 angle_override=disorder),
    }
    mesher = {"conforming": conforming_mesh, "painted": painted_mesh}[args.mesh]

    results = {}
    for name, cfg in configs.items():
        shapes, cen, al = tri_rod_layout(args.Lx, args.Ly, side, R=args.R, **cfg)
        npx, tris, z = mesher(args.Lx, args.Ly, args.h, shapes)
        sponge = quadratic_sponge(npx, args.Lx, args.Ly)
        w = nodal_areas(npx, tris)

        d0, d1 = poly[1] - poly[0], poly[-1] - poly[-2]
        sxy = poly[0] + 0.6 * d0 / np.hypot(*d0)
        oxy = poly[-1] - 0.6 * d1 / np.hypot(*d1)
        src = int(np.argmin((npx[:, 0] - sxy[0])**2 + (npx[:, 1] - sxy[1])**2))
        probe = np.hypot(npx[:, 0] - oxy[0], npx[:, 1] - oxy[1]) < 1.5

        kw = dict(muf=args.muf, eta=args.eta, sponge=sponge, src=src,
                  probe=probe, weights=w)
        u, T = solve_transmission(npx, tris, z, omega=args.wmid, **kw)
        _, T0 = solve_transmission(npx, tris, z, omega=args.wfloor, **kw)
        results[name] = dict(u=u, T=T, Tfloor=T0, npx=npx, tris=tris, z=z,
                             oxy=oxy, src=src)
        print(f"{name:38s}  T={T:.3e}  (out-of-band {T0:.3e}, "
              f"contrast {T/max(T0,1e-300):.0f}x)", flush=True)

    p = results["pristine wall"]
    contrast = p['T'] / max(p['Tfloor'], 1e-300)
    if contrast < args.min_contrast:
        raise SystemExit(
            f"\nABORT: pristine in-band/out-of-band contrast is {contrast:.1f}x, "
            f"below the required {args.min_contrast}x.\n"
            f"omega={args.wmid} is not in the guided band for this mesh, so the "
            f"defect comparison would be noise-vs-noise.\n"
            f"Locate the band first:  python examples/valley_band_convergence.py")

    T0 = p['T']
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))
    for a_, (name, r) in zip(ax, results.items()):
        npx, tris = r['npx'], r['tris']
        vmax = np.percentile(np.abs(r['u']), 99.5)
        a_.tripcolor(npx[:, 0], npx[:, 1], tris, np.abs(r['u']), cmap="magma",
                     vmax=vmax, shading="gouraud")
        a_.plot(poly[:, 0], poly[:, 1], "c--", lw=1.0, alpha=0.6)
        a_.plot(npx[r['src'], 0], npx[r['src'], 1], "co", ms=9, mfc="none", mew=2)
        a_.add_patch(plt.Circle(r['oxy'], 1.5, ec="lime", fc="none", lw=1.6))
        if name != "pristine wall":
            a_.add_patch(plt.Circle(mid, 1.6, ec="w", fc="none", lw=1.2, ls=":"))
        a_.set_aspect("equal")
        a_.set_title(f"{name}\nT = {r['T']:.2e}  ({r['T']/T0*100:.0f}% of "
                     "pristine)", fontsize=11)
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("Topological protection of the valley kink mode: guided-band "
                 f"transmission survives wall damage (omega={args.wmid}, "
                 f"{args.mesh} mesh, pristine contrast {contrast:.0f}x)",
                 y=0.99, fontsize=13)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)
    np.savez(args.out.replace(".png", ".npz"),
             names=list(results.keys()),
             T=[results[n]['T'] for n in results],
             Tfloor=[results[n]['Tfloor'] for n in results],
             omega=args.wmid, mesh=args.mesh, h=args.h)


if __name__ == "__main__":
    main()
