"""
Is the valley-Hall guided band a property of the crystal, or of the mesh?

Sweeps the domain-wall transmission over frequency for the triangular-rod valley
crystal, on ROD-CONFORMING and (legacy) CENTROID-PAINTED meshes, at several
resolutions.  A physical guided band must sit at the same frequencies and carry
the same transmission as h is refined.  A band that moves, or whose transmission
swings by orders of magnitude between resolutions, is a discretization artifact.

This is the control that was missing when the paper's "mid-gap/in-band ratio
~1.5e3" was established from a single painted realization: three painted meshes
of nominally the same crystal gave T = 7.1e-3, 3.9e+1 and 4.2e-6 at omega=7.267
(results/wg_tri_sweep.log, wg_tri_sweep2.log, valley_robust.log).

    python examples/valley_band_convergence.py --quick     # 13 freqs, 2 meshes
    python examples/valley_band_convergence.py             # 25 freqs, 3 meshes
"""
import argparse
import json
import time

import numpy as np

from wavetopo.valley_guide import (tri_rod_layout, conforming_mesh, painted_mesh,
                                   wall_polyline, wall_side_fn, quadratic_sponge,
                                   nodal_areas, solve_transmission)

MESHERS = {"conforming": conforming_mesh, "painted": painted_mesh}


def probe_nodes(npx, poly, *, inset=0.6, radius=1.5):
    """Source node at one end of the wall, probe disk at the other."""
    d0 = poly[1] - poly[0]
    d1 = poly[-1] - poly[-2]
    srcxy = poly[0] + inset * d0 / np.hypot(*d0)
    outxy = poly[-1] - inset * d1 / np.hypot(*d1)
    src = int(np.argmin((npx[:, 0] - srcxy[0]) ** 2 + (npx[:, 1] - srcxy[1]) ** 2))
    probe = np.hypot(npx[:, 0] - outxy[0], npx[:, 1] - outxy[1]) < radius
    return src, probe, srcxy, outxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=float, default=20.0)
    ap.add_argument("--Ly", type=float, default=17.0)
    ap.add_argument("--R", type=float, default=0.44)
    ap.add_argument("--muf", type=float, default=100.0)
    ap.add_argument("--eta", type=float, default=0.003)
    ap.add_argument("--wlo", type=float, default=6.4)
    ap.add_argument("--whi", type=float, default=8.0)
    ap.add_argument("--nw", type=int, default=25)
    ap.add_argument("--hs", type=float, nargs="+", default=[0.12, 0.09, 0.06])
    ap.add_argument("--methods", nargs="+", default=["conforming", "painted"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/valley_band_convergence")
    args = ap.parse_args()
    if args.quick:
        args.nw, args.hs = 13, [0.12, 0.09]

    poly = wall_polyline(args.Lx, args.Ly)
    side = wall_side_fn(poly)
    shapes, cen, al = tri_rod_layout(args.Lx, args.Ly, side, R=args.R)
    ws = np.linspace(args.wlo, args.whi, args.nw)
    print(f"{len(shapes)} rods; {args.nw} frequencies in "
          f"[{args.wlo}, {args.whi}]; eta={args.eta}", flush=True)

    out = {"omega": ws.tolist(), "runs": {}}
    for method in args.methods:
        for h in args.hs:
            key = f"{method}_h{h:g}"
            t0 = time.time()
            npx, tris, z = MESHERS[method](args.Lx, args.Ly, h, shapes)
            src, probe, sxy, oxy = probe_nodes(npx, poly)
            w = nodal_areas(npx, tris)
            print(f"\n[{key}] {npx.shape[0]} nodes {tris.shape[0]} tris "
                  f"(mesh {time.time()-t0:.1f}s), probe {probe.sum()} nodes",
                  flush=True)
            T = []
            for om in ws:
                t1 = time.time()
                _, t = solve_transmission(npx, tris, z, muf=args.muf, omega=om,
                                          eta=args.eta, sponge=quadratic_sponge(
                                              npx, args.Lx, args.Ly),
                                          src=src, probe=probe, weights=w)
                T.append(t)
                print(f"   w={om:6.3f}  T={t:.4e}   ({time.time()-t1:.1f}s)",
                      flush=True)
            out["runs"][key] = {"h": h, "method": method, "T": T,
                                "nodes": int(npx.shape[0]),
                                "tris": int(tris.shape[0])}
            np.savez(args.out + ".npz", **{"omega": ws},
                     **{k: np.array(v["T"]) for k, v in out["runs"].items()})
            with open(args.out + ".json", "w") as f:
                json.dump(out, f, indent=1)
            print(f"[{key}] done in {time.time()-t0:.0f}s", flush=True)
    print("\nsaved", args.out + ".npz/.json", flush=True)


if __name__ == "__main__":
    main()
