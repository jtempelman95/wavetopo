r"""Revision R1 analyses of the cloak (Reviewer R1, M2, M4; Fig. 6).

    series    FIXED designs (optimized, curvature-limited, transformation directions) transferred
              to conforming meshes h = 0.045 (paper), 0.0375, 0.03, 0.025: far-field mismatch
              reduction on each (a refinement series of fixed designs)
    sponge    the same three designs with an absorbing layer twice as wide, on a domain
              enlarged outward so that the interior, the void, the shell and the observation
              region are unchanged (h = 0.045)
    fields    incident (void-free) field and scattered field u - u_ref of straight fibers and of
              the three designs on the paper mesh, saved for the figure

    PYTHONPATH=. python examples/analysis/wt_r1_cloak.py series|sponge|fields
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
import numpy as np
import ufl
from dolfinx import fem

sys.path.insert(0, _os.path.join(_ROOT, "examples", "dolfinx"))
sys.path.insert(0, _os.path.join(_ROOT, "examples", "analysis"))
import dolfinx_cloak_conforming_vec as CV
from wt_verify import transfer
from wavetopo.dolfinx_elastic import ElasticWave

RES = _os.path.join(_ROOT, "results")
UNC = np.load(_os.path.join(RES, "dolfinx_cloak_conforming_data.npz"))
H2H = np.load(_os.path.join(RES, "cloak_headtohead.npz"))


def designs(cn, shift=(0.0, 0.0)):
    c = cn - np.asarray(shift)[None, :]
    return {"optimized": transfer(UNC["cent"], UNC["thopt"], c),
            "curv-limited": transfer(H2H["cent"], H2H["th_opt"], c),
            "transformation": transfer(H2H["cent"], H2H["th_an"], c)}


def taper(cn, th, shift=(0.0, 0.0)):
    """Designs are straight outside the shell: zero any transferred value beyond R_DES."""
    r = np.hypot(cn[:, 0] - shift[0] - CV.CX, cn[:, 1] - shift[1] - CV.CY)
    return np.where(r < CV.R_DES + 0.02, th, 0.0)


def reductions(ew, cn, shift=(0.0, 0.0)):
    n = ew.theta.x.array.size
    ew.set_theta(np.zeros(n)); ew.solve(); J0 = ew.cloak_mismatch()
    out = {}
    for k, th in designs(cn, shift).items():
        ew.set_theta(taper(cn, th, shift)); ew.solve(); out[k] = J0 / ew.cloak_mismatch()
    return out


def study_series():
    rows = []
    for hm in (0.045, 0.0375, 0.03, 0.025):
        CV.HMESH = hm
        ew, ref = CV.build()
        cn = CV.centroids(ew)
        r = reductions(ew, cn)
        rows.append((hm, ew.theta.x.array.size, r["optimized"], r["curv-limited"], r["transformation"]))
        print(f"[series] h={hm:.4f} ({rows[-1][1]} cells): optimized {r['optimized']:5.1f}x  "
              f"curv-limited {r['curv-limited']:5.1f}x  transformation {r['transformation']:5.1f}x", flush=True)
        np.save(_os.path.join(RES, "wt_r1_cloak_series.npy"), np.array(rows))


def study_sponge(pad=0.6):
    from wavetopo.dolfinx_mesh import rect_mesh
    hm = 0.045
    Lx, Ly, m = CV.Lx + pad, CV.Ly + 2 * pad, CV.MARGIN
    cx, cy = CV.CX, CV.CY + pad
    mh = rect_mesh(Lx, Ly, hm, holes=[(cx, cy, CV.RV)], h_hole=0.5 * hm)
    mf = rect_mesh(Lx, Ly, hm)

    def mk(dom):
        ew = ElasticWave(dom, omega=CV.OMEGA, Ef1=131., Ef2=9., G12=5., nu12=0.27,
                         rho=1.6, rho_m=1.2, eta=CV.ETA, penal=3.0)
        t = m + pad                                      # starts where the paper's layer starts
        ew.set_sponge(lambda x, y: np.maximum(0, np.maximum.reduce([
            (x - (CV.Lx - m)) / t, ((m + pad) - y) / t, (y - (CV.Ly - m + pad)) / t])) ** 2 * 70.0)
        ew.set_source(lambda x, y: 30.0 * np.exp(-80 * (x - 0.1) ** 2) * np.ones_like(y))
        ew.set_theta(np.zeros(ew.theta.x.array.size))
        return ew
    ew, ref = mk(mh), mk(mf)
    ref.solve()
    idata = fem.create_nonmatching_meshes_interpolation_data(
        ew.Uref.function_space.mesh._cpp_object, ew.Uref.function_space.element,
        ref.U.function_space.mesh._cpp_object, 1e-6)
    ew.Uref.interpolate(ref.U, nmm_interpolation_data=idata); ew.Uref.x.scatter_forward()
    ew.set_region(lambda x, y: ((np.hypot(x - cx, y - cy) > CV.RC) & (x > m) & (x < CV.Lx - m)
                                & (y > m + pad) & (y < CV.Ly - m + pad)).astype(float))
    cn = CV.centroids(ew)
    r = reductions(ew, cn, shift=(0.0, pad))
    print(f"[sponge] layer {m+pad:.1f} (vs {m:.1f}): optimized {r['optimized']:5.1f}x  "
          f"curv-limited {r['curv-limited']:5.1f}x  transformation {r['transformation']:5.1f}x", flush=True)
    np.save(_os.path.join(RES, "wt_r1_cloak_sponge.npy"),
            np.array([pad, r["optimized"], r["curv-limited"], r["transformation"]]))


def study_fields():
    CV.HMESH = 0.045
    ew, ref = CV.build()
    cn = CV.centroids(ew)
    tri, _ = CV.field_tri(ew)
    V = fem.functionspace(ew.domain, ("Lagrange", 1))

    def mag(expr):
        f = fem.Function(V); f.interpolate(fem.Expression(expr, V.element.interpolation_points()))
        return f.x.array.copy()
    U, R = ew.U, ew.Uref
    d = [U[i] - R[i] for i in range(4)]
    scat = ufl.sqrt(sum(di * di for di in d))
    inc = ufl.sqrt(sum(R[i] * R[i] for i in range(4)))
    out = dict(x=tri.x, y=tri.y, tris=tri.triangles, inc=None)
    ew.set_theta(np.zeros(ew.theta.x.array.size)); ew.solve()
    out["inc"] = mag(inc); out["scat_straight"] = mag(scat); J0 = ew.cloak_mismatch()
    for k, th in designs(cn).items():
        ew.set_theta(taper(cn, th)); ew.solve()
        out[f"scat_{k}"] = mag(scat); out[f"red_{k}"] = J0 / ew.cloak_mismatch()
        print(f"[fields] {k}: reduction {out[f'red_{k}']:.1f}x", flush=True)
    np.savez(_os.path.join(RES, "wt_r1_cloak_fields.npz"), **out)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        {"series": study_series, "sponge": study_sponge, "fields": study_fields}[w]()
