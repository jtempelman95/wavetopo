"""Re-render saved CFRP designs with clean (filtered+projected) density.

Reads the .npz produced by cfrp_reproduce_table3 / cfrp_cantilever and writes
polished 4-panel figures without re-running the optimization.
"""
import argparse
import numpy as np

from topoopt.cfrp import QuadMesh
from topoopt.cfrp_viz import plot_design


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--R", type=float, default=0.075)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    d = np.load(args.npz)
    mesh = QuadMesh(int(d["nelx"]), int(d["nely"]),
                    float(d["Lx"]), float(d["Ly"]))
    res = {k: d[k] for k in ("z", "theta", "chi_hat", "curl", "f", "curl_max")}
    title = args.title or f"CFRP cantilever {mesh.nelx}x{mesh.nely}"
    plot_design(mesh, res, args.out, title=title, R=args.R)


if __name__ == "__main__":
    main()
