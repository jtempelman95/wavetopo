r"""
How many periods does a finite panel need before the band gap does its job?

A Bloch gap says an infinite crystal supports no propagating mode; it says
nothing about how fast a *truncated* panel attenuates.  Inside the gap the
solution is evanescent, |u| ~ exp(-kappa x), so transmitted energy through Nx
periods should fall geometrically,

    T(Nx) ~ T0 * exp(-2 kappa a Nx),

and the design question is the number of cells needed to reach a target
isolation.  This sweeps Nx on the co-designed cell, at the gap-centre frequency
and at a pass-band control, and fits kappa a.

Reuses the design saved by examples/multifunc_finite.py, so no re-optimization.

    PYTHONPATH=. python examples/multifunc_periods.py
"""
import argparse

import numpy as np

from wavetopo.cfrp import Material
from examples.drivers.multifunc_finite import MAT, tile_panel, _panel_setup, _solve_T


def drive(d, n, Nx, Ny, nbuf, om, eta, tag="co"):
    """Insertion loss through Nx periods: transmission with the crystal divided
    by transmission through a solid panel of identical size, source and sponge.
    Both panels are rebuilt at each Nx so the geometry is matched exactly and the
    ratio isolates the crystal."""
    mesh, z, th = tile_panel(d[f'{tag}_y'], d[f'{tag}_theta_elem'], n,
                             Nx, Ny, nbuf)
    mat = Material(**MAT)
    c_el, src, prb = _panel_setup(mesh, nbuf, n)
    _, t = _solve_T(mesh, mat, z, th, om, eta, c_el, src, prb)
    _, tr = _solve_T(mesh, mat, np.ones_like(z), np.zeros_like(th), om, eta,
                     c_el, src, prb)
    return t, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="results/multifunc_finite_design.npz")
    ap.add_argument("--res", default="results/multifunc_finite.npz")
    ap.add_argument("--Ny", type=int, default=4)
    ap.add_argument("--nbuf", type=int, default=28)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--Nxs", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10])
    ap.add_argument("--out", default="results/multifunc_periods")
    args = ap.parse_args()

    d = dict(np.load(args.design, allow_pickle=True))
    r = np.load(args.res, allow_pickle=True)
    n = int(d['co_n'])
    lo, hi = float(r['gap_lo']), float(r['gap_hi'])
    w, Tco = r['omega'], r['co_T']
    w_gap = 0.5 * (lo + hi)
    out = (w < lo) | (w > hi)
    w_pass = float(w[out][np.argmax(Tco[out])])
    print(f"gap [{lo:.4f},{hi:.4f}]  drive in-gap w={w_gap:.4f}, "
          f"pass-band w={w_pass:.4f}\n", flush=True)

    Tg, Tp, ILg, ILp = [], [], [], []
    print(f"{'Nx':>4} {'T(in gap)':>12} {'IL(in gap)':>12} "
          f"{'T(pass)':>12} {'IL(pass)':>10}", flush=True)
    for Nx in args.Nxs:
        tg, trg = drive(d, n, Nx, args.Ny, args.nbuf, w_gap, args.eta)
        tp, trp = drive(d, n, Nx, args.Ny, args.nbuf, w_pass, args.eta)
        Tg.append(tg); Tp.append(tp)
        ILg.append(trg / max(tg, 1e-300)); ILp.append(trp / max(tp, 1e-300))
        print(f"{Nx:4d} {tg:12.4e} {ILg[-1]:11.1f}x {tp:12.4e} "
              f"{ILp[-1]:9.1f}x", flush=True)

    Tg, Tp = np.array(Tg), np.array(Tp)
    ILg, ILp = np.array(ILg), np.array(ILp)
    Nxs = np.array(args.Nxs, float)
    # fit log T = log T0 - 2 kappa a Nx  over the decaying part
    msk = ILg > 0
    if msk.sum() >= 3:
        c = np.polyfit(Nxs[msk], -np.log(ILg[msk]), 1)
        kappa_a = -c[0] / 2.0
        print(f"\nfitted decay: kappa*a = {kappa_a:.4f} per cell "
              f"({-c[0]/np.log(10):.2f} decades per cell)", flush=True)
    else:
        kappa_a = np.nan
    np.savez(args.out + ".npz", Nx=Nxs, T_gap=Tg, T_pass=Tp,
             IL_gap=ILg, IL_pass=ILp,
             w_gap=w_gap, w_pass=w_pass, kappa_a=kappa_a,
             gap_lo=lo, gap_hi=hi)
    print("saved", args.out + ".npz", flush=True)


if __name__ == "__main__":
    main()
