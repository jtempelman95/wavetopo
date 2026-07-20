"""
Re-solve every saved design and store the FULL complex field, not just |u|.

The drivers' field_tri() collapses the 4-component state to
sqrt(u_xr^2+u_yr^2+u_xi^2+u_yi^2) before saving, so the data files keep only the
envelope.  That is enough for a heat-map but loses the real and imaginary parts,
the x/y components, the phase, and hence any wavefront plot, polarization plot or
time animation Re[u e^{-i w t}].

The DESIGNS are saved, and a solve is ~1.5 s with the MUMPS factorization, so the
full field is recoverable without re-optimizing: rebuild each problem, set the
stored theta, solve once, interpolate all four components onto the same P1 space
the triangulation uses, and add them to the existing npz.

Adds, for the baseline (suffix 0) and optimized (suffix 1) states:
    ur_x0, ur_y0, ui_x0, ui_y0,  ur_x1, ur_y1, ui_x1, ui_y1

so that, per node,
    u(t) = Re[(ur + i ui) e^{-i w t}] = ur cos(wt) + ui sin(wt)
    |u|  = sqrt(ur_x^2 + ur_y^2 + ui_x^2 + ui_y^2)     (matches the stored m*)
    phase(u_x) = atan2(ui_x, ur_x)

    XDG_CACHE_HOME=/home/jrt/wavetopo/.fenics-cache PYTHONPATH=/home/jrt/wavetopo \
    /home/jrt/miniforge3/envs/dolfinx_complex/bin/python3 \
        examples/resolve_full_fields.py [key ...]
"""
import importlib.util
import os
import sys
import time

import numpy as np
import ufl
from dolfinx import fem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def components(ew):
    """The four nodal components on the SAME P1 space the triangulation uses.

    Interpolating each component (rather than reshaping U.x.array) guarantees the
    node ordering matches trix/triy/tris already stored in the file.
    """
    V = fem.functionspace(ew.domain, ("Lagrange", 1))
    out = []
    for i in range(4):
        f = fem.Function(V)
        f.interpolate(fem.Expression(ew.U[i], V.element.interpolation_points()))
        out.append(f.x.array.copy())
    return out                      # [u_xr, u_yr, u_xi, u_yi]


# key -> (npz, builder returning a ready ElasticWave, theta keys for state 0/1)
def _lens():   return _mod("examples/dolfinx_lens.py", "L")
def _cloakS(): return _mod("examples/dolfinx_cloak_soft.py", "CS")
def _cloakC(): return _mod("examples/dolfinx_cloak_conforming_vec.py", "CC")
def _guide():  return _mod("examples/dolfinx_guide_joint.py", "G")


SPEC = {
    "lens":       ("results/dolfinx_lens_data.npz",
                   lambda: _lens().make(), (None, "thopt")),
    # NB: each run's mesh resolution must match the one it was optimized on --
    # the cell-count guard below catches a mismatch rather than writing garbage.
    "lens_curl":  ("results/dolfinx_lens_curl_data.npz",
                   lambda: _lens().make(nx=80), ("th0", "th1")),
    "lens_multi": ("results/dolfinx_lens_multi_data.npz",
                   lambda: _lens().make_multi(nx=110), (None, "thopt")),
    "lens_hole":  ("results/dolfinx_lens_hole_data.npz",
                   lambda: _lens().make_hole_conforming(h=0.045), ("th0", "th1")),
    "lens_asym":  ("results/dolfinx_lens_hole_asym_data.npz",
                   lambda: _lens().make_hole_conforming(
                       h=0.045, holes=_lens().HOLES_ASYM), (None, "th1")),
    "cloak_curl": ("results/dolfinx_cloak_curl_data.npz",
                   lambda: _cloakS().make(nx=110, ny=73), ("th0", "th1")),
    "cloak":      ("results/dolfinx_cloak_conforming_data.npz",
                   lambda: _cloakC().build()[0], (None, "thopt")),
    "guide":      ("results/dolfinx_guide_joint_data.npz",
                   lambda: _guide().make(), (None, "thopt")),
}


def run(key):
    path, build, (k0, k1) = SPEC[key]
    if not os.path.exists(path):
        print(f"  {key:11} SKIP (no {path})"); return False
    d = dict(np.load(path))
    if "ur_x1" in d:
        print(f"  {key:11} already has the full field"); return True
    t0 = time.time()
    ew = build()
    n = ew.theta.x.array.size
    nnode = len(d["trix"])
    for tag, thk in ((0, k0), (1, k1)):
        th = np.zeros(n) if thk is None else np.asarray(d[thk], float)
        if th.size != n:
            print(f"  {key:11} FAIL: stored '{thk}' has {th.size} cells, mesh "
                  f"has {n} -- geometry changed since the run"); return False
        ew.set_theta(th); ew.solve()
        cx, cy, ix, iy = components(ew)
        if len(cx) != nnode:
            print(f"  {key:11} FAIL: {len(cx)} nodes vs {nnode} stored")
            return False
        d[f"ur_x{tag}"], d[f"ur_y{tag}"] = cx, cy
        d[f"ui_x{tag}"], d[f"ui_y{tag}"] = ix, iy
        # consistency: recomputed |u| must match the stored envelope
        mag = np.sqrt(cx**2 + cy**2 + ix**2 + iy**2)
        ref = d.get(f"m{tag}")
        if ref is None:                     # conforming cloak naming
            ref = d.get("munc" if tag == 0 else "mcl")
        if ref is not None:
            rel = np.linalg.norm(mag-ref)/max(np.linalg.norm(ref), 1e-30)
            flag = "OK" if rel < 1e-6 else f"MISMATCH {rel:.2e}"
            print(f"  {key:11} state {tag}: |u| vs stored m{tag}  {flag}")
    np.savez(path, **d)
    print(f"  {key:11} wrote full field in {time.time()-t0:.0f}s")
    return True


if __name__ == "__main__":
    todo = sys.argv[1:] or list(SPEC)
    bad = [k for k in todo if k not in SPEC]
    if bad:
        raise SystemExit(f"unknown: {bad}\navailable: {list(SPEC)}")
    ok = sum(bool(run(k)) for k in todo)
    print(f"\n{ok}/{len(todo)} data files now carry Re and Im of u")
