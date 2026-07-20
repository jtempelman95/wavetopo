"""
Reproduce the no-void cantilever of Wong et al. (2026), Section 8.1 / Table 3:

                       f       |zeta|max (1/m)
    no constraint     0.42        8.04
    curl <= 2 /m      0.49        2.0

Runs both cases at the paper resolution (240x150 = 36000 elements) and writes a
4-panel figure (density + |curl| for each case) plus a summary table.
"""
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.cfrp_cantilever import build_cantilever, save_npz
from wavetopo.cfrp_optimizer import optimize_cfrp
from wavetopo.cfrp_viz import plot_design


def run_case(nely, zeta_all, max_outer):
    mesh, prob = build_cantilever(nely, d=0.06, beta=100.0, vf_all=0.25)
    z0 = np.full(mesh.N, 0.5)
    th0 = np.zeros(prob.M)
    t0 = time.time()
    res = optimize_cfrp(prob, z0, th0, zeta_all=zeta_all,
                        max_outer=max_outer, mma_iter=5, patience=25)
    res['time'] = time.time() - t0
    res['outer'] = res['outer_iters']
    return mesh, res


def panel(ax, data, mesh, title, cmap, mask=None, vmax=None):
    if mask is not None:
        data = np.ma.array(data, mask=mask)
    im = ax.imshow(data, origin="lower", cmap=cmap, vmax=vmax,
                   extent=[0, mesh.Lx, 0, mesh.Ly], aspect="equal")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)


def main():
    nely = 150
    cases = [("no constraint", None, 90),
             ("curl <= 2 /m", 2.0, 150)]
    tags = {"no constraint": "nocurl", "curl <= 2 /m": "curl2"}
    results = []
    for name, za, mo in cases:
        print(f"\n########## {name} ##########")
        mesh, res = run_case(nely, za, mo)
        print(f">>> {name}: f={res['f']:.4f}  |zeta|max={res['curl_max']:.3f}"
              f"  ({res['outer']} outer, {res['time']:.0f}s)")
        tag = tags[name]
        save_npz(f"results/cfrp_cant_{tag}.npz", mesh, res)
        plot_design(mesh, res, f"results/cfrp_cant_{tag}.png",
                    title=f"CFRP cantilever {mesh.nelx}x{mesh.nely}  ({name})")
        results.append((name, mesh, res))

    # ---- summary table ----
    print("\n================ Table 3 (no-void cantilever) ================")
    print(f"{'case':16s} {'f (ours)':>10s} {'f (paper)':>10s} "
          f"{'|z|max ours':>12s} {'|z|max paper':>13s}")
    paper = {"no constraint": (0.42, 8.04), "curl <= 2 /m": (0.49, 2.0)}
    for name, mesh, res in results:
        pf, pz = paper[name]
        print(f"{name:16s} {res['f']:>10.3f} {pf:>10.2f} "
              f"{res['curl_max']:>12.2f} {pz:>13.2f}")


if __name__ == "__main__":
    main()
