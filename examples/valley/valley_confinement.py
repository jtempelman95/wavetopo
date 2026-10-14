"""
Separating a guided kink mode from ordinary bulk transport.

Transmission to a probe at the far end of the domain wall -- the metric the
earlier valley scripts used -- registers energy that arrived by ANY route.  It
therefore reads high both for a confined edge mode inside the bulk gap and for
delocalized bulk propagation above it, and cannot tell them apart.  Measured on
the conforming crystal, above-gap transmission (bulk band 3 starts at omega
~7.54) is only a factor ~3 below in-gap transmission: on transmission alone the
two regimes are barely distinguishable.

This script sweeps frequency recording BOTH

    T  = \int_probe |u|^2 dA                (did energy arrive?)
    C  = \int_{d<width} |u|^2 / \int_int    (is it bound to the wall?)

so the guided band is identified by confinement, cross-checked against the
independently computed bulk gap from the conforming unit cell
(examples/valley_bulk_gap via wavetopo.valley_cell.triangle_cell).

    python examples/valley_confinement.py
"""
import argparse
import json

import numpy as np

from wavetopo.valley_guide import (tri_rod_layout, conforming_mesh,
                                   wall_polyline, wall_side_fn, quadratic_sponge,
                                   nodal_areas, assemble, wall_confinement)
import scipy.sparse.linalg as spla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=float, default=20.0)
    ap.add_argument("--Ly", type=float, default=17.0)
    ap.add_argument("--R", type=float, default=0.44)
    ap.add_argument("--h", type=float, default=0.09)
    ap.add_argument("--muf", type=float, default=100.0)
    ap.add_argument("--eta", type=float, default=0.003)
    ap.add_argument("--width", type=float, default=1.0,
                    help="wall half-width for the confinement integral")
    ap.add_argument("--wlo", type=float, default=6.4)
    ap.add_argument("--whi", type=float, default=8.0)
    ap.add_argument("--nw", type=int, default=33)
    ap.add_argument("--out", default="results/valley_confinement")
    args = ap.parse_args()

    poly = wall_polyline(args.Lx, args.Ly)
    side = wall_side_fn(poly)
    shapes, cen, al = tri_rod_layout(args.Lx, args.Ly, side, R=args.R)
    npx, tris, z = conforming_mesh(args.Lx, args.Ly, args.h, shapes)
    sponge = quadratic_sponge(npx, args.Lx, args.Ly)
    w = nodal_areas(npx, tris)

    d0, d1 = poly[1] - poly[0], poly[-1] - poly[-2]
    sxy = poly[0] + 0.6 * d0 / np.hypot(*d0)
    oxy = poly[-1] - 0.6 * d1 / np.hypot(*d1)
    src = int(np.argmin((npx[:, 0] - sxy[0])**2 + (npx[:, 1] - sxy[1])**2))
    probe = np.hypot(npx[:, 0] - oxy[0], npx[:, 1] - oxy[1]) < 1.5

    print(f"{len(shapes)} rods, {npx.shape[0]} nodes, {tris.shape[0]} tris",
          flush=True)
    print(f"{'omega':>7} {'T':>12} {'confinement':>12}", flush=True)
    ws = np.linspace(args.wlo, args.whi, args.nw)
    T, C = [], []
    for om in ws:
        D, _, _ = assemble(npx, tris, z, muf=args.muf, omega=om, eta=args.eta,
                           sponge=sponge)
        F = np.zeros(npx.shape[0], complex); F[src] = 1.0
        u = spla.spsolve(D.tocsc(), F)
        t = float(np.sum(w[probe] * np.abs(u[probe])**2))
        c = wall_confinement(npx, tris, u, poly, width=args.width,
                             Lx=args.Lx, Ly=args.Ly)
        T.append(t); C.append(c)
        print(f"{om:7.3f} {t:12.4e} {c:12.4f}", flush=True)

    np.savez(args.out + ".npz", omega=ws, T=np.array(T), C=np.array(C),
             h=args.h, width=args.width)
    with open(args.out + ".json", "w") as f:
        json.dump({"omega": ws.tolist(), "T": T, "C": C, "h": args.h}, f, indent=1)
    print("\nsaved", args.out + ".npz/.json", flush=True)


if __name__ == "__main__":
    main()
