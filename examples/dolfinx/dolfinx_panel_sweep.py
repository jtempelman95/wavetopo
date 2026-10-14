"""Fine-frequency sweep of the panel designs, for the operational-bandwidth curve.
For each design (straight-fiber baseline is the denominator at every omega), report
the sensor / payload / rims energy ratios across a band, for both the single-TONE
design and the BAND-optimized design.  Saves results/dolfinx_panel_sweep.npz.

    XDG_CACHE_HOME=... PYTHONPATH=/home/jrt/wavetopo python3 \
        examples/dolfinx/dolfinx_panel_sweep.py
"""
import sys
import numpy as np

sys.path.insert(0, "/home/jrt/wavetopo/examples/dolfinx")
import dolfinx_panel as P

FREQS = np.arange(41.0, 59.01, 1.0)             # the swept band (fine resolution)
# both designs share the strong-shield objective; they differ only in single-tone
# (optimized at omega=50) vs band-optimized (averaged over 44/50/56)
DESIGNS = {
    "tone": "results/dolfinx_panel_deep_data.npz",
    "band": "results/dolfinx_panel_band_data.npz",
}


def sweep_one(thopt):
    rows = []
    for w in FREQS:
        dom = P.rect_mesh(P.Lx, P.Ly, P.HMESH, holes=P.HOLES, h_hole=0.5*P.HMESH)
        ew = P._attach(dom, float(w))
        ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve()
        ES0, EP0, EF0, EO0 = P.measure(ew)
        ew.set_theta(thopt); ew.solve()
        ES1, EP1, EF1, EO1 = P.measure(ew)
        rows.append((w, ES1/ES0, EP1/EP0, EF1/EF0))
        print(f"  w={w:4.1f}: sensor={ES1/ES0:6.2f}x payload={EP1/EP0:5.2f}x "
              f"rims={EF1/EF0:5.2f}x", flush=True)
    return np.array(rows)


def main():
    out = {"freqs": FREQS}
    for name, path in DESIGNS.items():
        try:
            th = np.load(path)["thopt"]
        except FileNotFoundError:
            print(f"skip {name}: {path} not found", flush=True)
            continue
        print(f"=== {name} design ({path}) ===", flush=True)
        out[name] = sweep_one(th)
    np.savez("results/dolfinx_panel_sweep.npz", **out)
    print("saved results/dolfinx_panel_sweep.npz", flush=True)


if __name__ == "__main__":
    main()
