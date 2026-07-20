"""Benchmark PETSc LU factor solvers on a real wave-control operator.

The dolfinx runs sit at ~100% CPU (one core) no matter what OMP_NUM_THREADS is
set to, because pc.setType("lu") with no factorSolverType selects PETSc's
BUILT-IN SEQUENTIAL LU.  Threads only help dense BLAS, which a sparse LU barely
touches.  This measures whether a threaded third-party factorization (MUMPS) or
a different sequential one (UMFPACK/SuperLU/KLU) is faster on our matrices.

    OMP_NUM_THREADS=8 XDG_CACHE_HOME=/home/jrt/wavetopo/.fenics-cache \
    PYTHONPATH=/home/jrt/wavetopo .../dolfinx_complex/bin/python3 \
        examples/bench_lu.py
"""
import os
import time
import numpy as np
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

import importlib.util
spec = importlib.util.spec_from_file_location(
    "g", "/home/jrt/wavetopo/examples/dolfinx_guide_joint.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

nthr = os.environ.get("OMP_NUM_THREADS", "?")
print(f"OMP_NUM_THREADS={nthr}\n", flush=True)

ew = g.make()
ew.set_theta(np.zeros(ew.theta.x.array.size))
A = assemble_matrix(ew.a_form, bcs=[]); A.assemble()
b = assemble_vector(ew.L_form)
n = A.getSize()[0]
print(f"operator: {n} x {n}, {A.getInfo()['nz_used']:.0f} nonzeros\n", flush=True)

x = A.createVecRight()


def bench(name, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.time()
        ksp = PETSc.KSP().create()
        ksp.setOperators(A); ksp.setType("preonly")
        pc = ksp.getPC(); pc.setType("lu")
        if name is not None:
            pc.setFactorSolverType(name)
            if name == "mumps":
                pc.setUp()
                F = pc.getFactorMatrix()
                F.setMumpsIcntl(13, 1)          # allow OpenMP root factorization
        ksp.setUp(); ksp.solve(b, x)
        ts.append(time.time() - t0)
        ksp.destroy()
    return min(ts), x.norm()


base = None
for nm in (None, "mumps", "umfpack", "superlu", "klu"):
    try:
        t, nrm = bench(nm)
        if base is None:
            base = t
        print(f"  {str(nm):10s} factor+solve {t:6.2f}s   speedup {base/t:4.2f}x   "
              f"|x|={nrm:.6e}", flush=True)
    except Exception as e:
        print(f"  {str(nm):10s} FAILED: {str(e).splitlines()[-1][:70]}", flush=True)

print("""
Note: MUMPS shared-memory speedup depends on the conda build having OpenMP
enabled.  True multi-core scaling for these solves otherwise requires MPI
(mpirun -n N), which needs the driver made rank-safe -- the design vectors,
centroid bookkeeping, plotting and np.savez here all assume serial execution.""")
