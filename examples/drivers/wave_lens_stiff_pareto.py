"""Focus-vs-stiffness Pareto for the multifunctional plate: sweep the compliance
limit for the co-design and trace focus gain against stiffness.  Reuses
examples/wave_lens_stiff.py.  python examples/wave_lens_stiff_pareto.py"""
import argparse, json, time
import numpy as np
from examples.drivers.wave_lens_stiff import build, Stiffness, optimize

ap = argparse.ArgumentParser()
ap.add_argument("--nx", type=int, default=120)
ap.add_argument("--omega", type=float, default=40.0)
ap.add_argument("--iters", type=int, default=80)
ap.add_argument("--tols", type=float, nargs="+",
                default=[0.0, 0.10, 0.20, 0.35, 0.55])
ap.add_argument("--out", default="results/wave_lens_stiff_pareto")
a = ap.parse_args()
Lx, Ly = 4.0, 3.0; ny = int(round(a.nx*Ly/Lx))
mesh, hl, focus = build(a.nx, ny, Lx, Ly, a.omega, 0.03, 0.33, 0.7)
hl.M_ = hl.rbf.Phi.shape[1]; stiff = Stiffness(mesh, hl)
th0 = np.zeros(hl.M_); c0 = stiff.value_grad(th0, False)[0]
print(f"straight compliance c0={c0:.4e}", flush=True)
fo = optimize(hl, stiff, th0, mode="focus", iters=a.iters, verbose=False)
rows = [dict(label="straight", G=1.0, stiff=1.0, tol=None),
        dict(label="focus-only", G=fo['G'], stiff=c0/fo['c'], tol="inf")]
print(f"focus-only: G={fo['G']:.1f}x stiff={c0/fo['c']:.2f}x", flush=True)
for tol in a.tols:
    t=time.time()
    co = optimize(hl, stiff, th0, mode="co", iters=a.iters,
                  c_limit=(1+tol)*c0, verbose=False)
    rows.append(dict(label=f"co tol={tol}", G=co['G'], stiff=c0/co['c'], tol=tol))
    print(f"co tol={tol:.2f}: G={co['G']:.1f}x stiff={c0/co['c']:.2f}x "
          f"({time.time()-t:.0f}s)", flush=True)
json.dump(rows, open(a.out+".json","w"), indent=1)
print("saved", a.out+".json", flush=True)
