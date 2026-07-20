"""Finite-difference verification of the manufacturability-constrained
wave-control module (wavetopo.wave_mfg):

  * focus objective + adjoint gradient with true through-hole voids
  * broadband match (cloak) objective, sum + worst-case (softmax) modes
  * curl field / Jacobian on the orientation-only design
  * augmented-Lagrangian curl feasibility on a tiny lens problem

Run:  PYTHONPATH=. python -m tests.test_wave_mfg
"""
import numpy as np

from wavetopo.cfrp import Material, QuadMesh, grid_support_points, CSRBFMapping
from wavetopo.wave_mfg import (HarmonicPlate, curl_field, curl_jacobian,
                              optimize_orientation, ramp_sponge)


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


def test_focus_with_holes():
    mesh, mat, rbf, (Lx, Ly) = _setup()
    z = np.ones(mesh.N)
    z[np.hypot(mesh.cx - 1.4, mesh.cy - 1.5) < 0.3] = 0.0   # a through-hole
    hp = HarmonicPlate(mesh, mat, rbf, [38.0], eta=0.03, z=z)
    hp.set_load(hp.plane_load("left"))
    hp.set_sponge(ramp_sponge(mesh, 0.6, 70.0))
    hp.set_focus(hp.box_weight((0.78 * Lx, Ly / 2), 0.18))
    rng = np.random.default_rng(0)
    th = 0.3 * rng.standard_normal(rbf.M)
    hp.set_baseline(hp.raw_values(th) + 1.0)
    hp.objective(th)
    g = hp.grad()
    err = _maxrel(lambda x: hp.objective(x), th, g, [2, 9, 17, 30])
    print(f"focus-with-hole grad max rel err = {err:.1e}")
    assert err < 1e-5


def test_broadband_match():
    mesh, mat, rbf, (Lx, Ly) = _setup()
    cx, cy = Lx / 2, Ly / 2
    z = np.ones(mesh.N)
    z[np.hypot(mesh.cx - cx, mesh.cy - cy) < 0.45] = 0.0
    omegas = [32.0, 40.0]
    # reference: no hole
    hp = HarmonicPlate(mesh, mat, rbf, omegas, eta=0.03)
    hp.set_load(hp.plane_load("left"))
    hp.set_sponge(ramp_sponge(mesh, 0.6, 60.0))
    u_ref = hp.solve(np.zeros(rbf.M))
    xs, ys = mesh.node_xy()
    obs = (np.hypot(xs - cx, ys - cy) > 1.0) & (xs > 0.6) & (xs < Lx - 0.7) \
        & (ys > 0.6) & (ys < Ly - 0.6)
    w = np.zeros(mesh.ndof); w[0::2][obs] = 1; w[1::2][obs] = 1
    hp.set_density(z)
    hp.set_match(u_ref, w)
    rng = np.random.default_rng(2)
    th = 0.3 * rng.standard_normal(rbf.M)
    hp.set_baseline(hp.raw_values(np.zeros(rbf.M)))
    for mode in ("sum", "minmax"):
        hp.objective(th, mode=mode)
        g = hp.grad()
        err = _maxrel(lambda x: hp.objective(x, mode=mode), th, g,
                      [3, 10, 25, 40])
        print(f"broadband match [{mode}] grad max rel err = {err:.1e}")
        assert err < 1e-5


def test_curl_jacobian():
    mesh, mat, rbf, _ = _setup()
    rng = np.random.default_rng(3)
    th = 0.4 * rng.standard_normal(rbf.M)
    J = curl_jacobian(rbf, th)
    eps = 1e-7
    worst = 0.0
    for k in [1, 8, 20, 33]:
        tp = th.copy(); tp[k] += eps
        tm = th.copy(); tm[k] -= eps
        fd = (curl_field(rbf, tp) - curl_field(rbf, tm)) / (2 * eps)
        col = np.asarray(J[:, k].todense()).ravel()
        worst = max(worst, np.max(np.abs(col - fd)) /
                    (np.max(np.abs(fd)) + 1e-30))
    print(f"curl jacobian max rel err = {worst:.1e}")
    assert worst < 1e-6


def test_al_curl_feasibility():
    mesh, mat, rbf, (Lx, Ly) = _setup(nx=30, ny=22)
    hp = HarmonicPlate(mesh, mat, rbf, [38.0], eta=0.03)
    hp.set_load(hp.plane_load("left"))
    hp.set_sponge(ramp_sponge(mesh, 0.6, 70.0))
    hp.set_focus(hp.box_weight((0.78 * Lx, Ly / 2), 0.2))
    base = hp.raw_values(np.zeros(rbf.M))
    hp.set_baseline(base)
    zeta_all = 1.0
    res = optimize_orientation(hp, np.zeros(rbf.M), zeta_all=zeta_all,
                               mu0=50.0, xi=1.5, max_outer=25, mma_iter=4,
                               verbose=False)
    print(f"AL: J={res['J']:.3f}  |zeta|max={res['curl_max']:.3f} "
          f"(limit {zeta_all})")
    assert res['curl_max'] <= zeta_all * 1.02
    assert res['J'] < -1.0   # improved on the straight-fiber gain of 1


if __name__ == "__main__":
    test_focus_with_holes()
    test_broadband_match()
    test_curl_jacobian()
    test_al_curl_feasibility()
    print("all wave-mfg gradient checks passed")
