"""Finite-difference verification of the wave-control adjoint gradients:
  * single-frequency focus (HarmonicLens)
  * multi-frequency focus (WaveFocus, sum + max-min)
  * cloak mismatch (HarmonicCloak)

Run:  PYTHONPATH=. python -m tests.test_wave_grad
"""
import numpy as np

from wavetopo.cfrp import Material, QuadMesh, grid_support_points, CSRBFMapping
from wavetopo.harmonic import HarmonicLens, HarmonicCloak
from wavetopo.wave_control import WaveFocus, ramp_sponge


def _plane_load(mesh, obj):
    F = np.zeros(mesh.ndof, dtype=complex)
    for iy in range(mesh.nely + 1):
        n = obj.node_id(0, iy)
        F[2 * n] = 1.0
    return F


def _setup(nx=40, ny=30, Lx=4.0, Ly=3.0):
    mesh = QuadMesh(nx, ny, Lx, Ly)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    sx, sy = grid_support_points(mesh, 0.45, 0.45)
    rbf = CSRBFMapping(mesh, (sx, sy), r_s=0.8)
    return mesh, mat, rbf, (Lx, Ly)


def _maxrel(obj_fun, x, g, idxs, eps=1e-6):
    worst = 0.0
    for k in idxs:
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        fd = (obj_fun(xp) - obj_fun(xm)) / (2 * eps)
        worst = max(worst, abs(g[k] - fd) / (abs(fd) + 1e-30))
    return worst


def test_lens():
    mesh, mat, rbf, (Lx, Ly) = _setup()
    hl = HarmonicLens(mesh, mat, rbf, omega=38.0, eta=0.03)
    hl.set_load(_plane_load(mesh, hl))
    hl.set_sponge(ramp_sponge(mesh, 0.6, 70.0))
    hl.set_focus(hl_focus := np.zeros(mesh.ndof))
    xs, ys = mesh.node_xy()
    sel = (np.abs(xs - 0.78 * Lx) < 0.18) & (np.abs(ys - Ly / 2) < 0.18)
    w = np.zeros(mesh.ndof); w[0::2][sel] = 1; w[1::2][sel] = 1
    hl.set_focus(w)
    rng = np.random.default_rng(0)
    th = 0.3 * rng.standard_normal(rbf.M)
    hl.focus_energy(th)
    g = hl.focus_grad()
    err = _maxrel(hl.focus_energy, th, g, [2, 9, 17, 30])
    print(f"lens focus grad max rel err = {err:.1e}")
    assert err < 1e-5


def test_multifreq():
    mesh, mat, rbf, (Lx, Ly) = _setup()
    wf = WaveFocus(mesh, mat, rbf, [30.0, 42.0, 54.0], eta=0.03)
    wf.set_load(_plane_load(mesh, wf))
    wf.set_sponge(ramp_sponge(mesh, 0.6, 70.0))
    wf.set_targets([wf.focus_weight((0.78 * Lx, Ly / 2), 0.18)
                    for _ in range(3)])
    rng = np.random.default_rng(1)
    th = 0.3 * rng.standard_normal(rbf.M)
    for mode in ("sum", "minmax"):
        wf.objective(th, mode=mode)
        g = wf.grad()
        err = _maxrel(lambda x: wf.objective(x, mode=mode), th, g,
                      [3, 11, 25, 40])
        print(f"multifreq [{mode}] grad max rel err = {err:.1e}")
        assert err < 1e-5


def test_cloak():
    mesh, mat, rbf, (Lx, Ly) = _setup()
    cl = HarmonicCloak(mesh, mat, rbf, omega=36.0, eta=0.03)
    cx, cy = Lx / 2, Ly / 2
    z = np.ones(mesh.N)
    z[np.hypot(mesh.cx - cx, mesh.cy - cy) < 0.45] = 0.0
    cl.set_load(_plane_load(mesh, cl))
    marg = 0.6
    d = np.maximum.reduce([(mesh.cx - (Lx - marg)) / marg,
                           (marg - mesh.cy) / marg,
                           (mesh.cy - (Ly - marg)) / marg])
    cl.set_sponge(np.maximum(0, d)**2 * 60.0)
    cl.set_density(np.ones(mesh.N))
    u_ref = cl.solve(np.zeros(rbf.M))
    xs, ys = mesh.node_xy()
    obs = (xs > cx + 0.6) & (xs < Lx - 0.7)
    w = np.zeros(mesh.ndof); w[0::2][obs] = 1; w[1::2][obs] = 1
    cl.set_density(z)
    cl.set_target(u_ref, w)
    rng = np.random.default_rng(2)
    th = 0.3 * rng.standard_normal(rbf.M)
    cl.objective(th)
    g = cl.grad()
    err = _maxrel(cl.objective, th, g, [3, 10, 25, 40])
    print(f"cloak grad max rel err = {err:.1e}")
    assert err < 1e-5


if __name__ == "__main__":
    test_lens()
    test_multifreq()
    test_cloak()
    print("all wave-control gradient checks passed")
