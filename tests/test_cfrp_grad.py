"""Finite-difference verification of CFRP objective + constraint gradients."""
import numpy as np

from wavetopo.cfrp import Material, QuadMesh, grid_support_points
from wavetopo.cfrp_problem import CFRPProblem


def build_tiny():
    mesh = QuadMesh(14, 9, 2.4, 1.5)
    mat = Material()
    sx, sy = grid_support_points(mesh, 0.4, 0.4)
    prob = CFRPProblem(mesh, mat, (sx, sy), R=0.18, r_s=0.6, d=0.25,
                       beta=12.0, v_all=0.5, vf_all=0.25)
    # clamp left edge
    fixed = []
    for iy in range(mesh.nely + 1):
        n = prob.node_id(0, iy)
        fixed += [2 * n, 2 * n + 1]
    prob.set_fixed_dofs(fixed)
    # downward load at mid right edge
    F = np.zeros(mesh.ndof)
    n = prob.node_id(mesh.nelx, mesh.nely // 2)
    F[2 * n + 1] = -0.5
    prob.set_load(F)
    return mesh, prob


def central(ffun, x, k, eps):
    xp = x.copy(); xp[k] += eps
    xm = x.copy(); xm[k] -= eps
    return (ffun(xp) - ffun(xm)) / (2 * eps)


def main():
    rng = np.random.default_rng(3)
    mesh, prob = build_tiny()
    z = np.clip(0.5 + 0.25 * rng.standard_normal(mesh.N), 0.1, 1.0)
    th = 0.4 * rng.standard_normal(prob.M)

    prob.forward(z, th)
    dz_an, dth_an = prob.objective_grad()
    eps = 1e-6

    print("=== objective df/dz ===")
    for k in [5, 33, 77, 120]:
        fd = central(lambda zz: prob.forward(zz, th), z, k, eps)
        print(f"  z[{k:3d}] an={dz_an[k]: .6e} fd={fd: .6e} "
              f"rel={abs(dz_an[k]-fd)/(abs(fd)+1e-30):.2e}")

    print("=== objective df/dtheta_hat ===")
    for k in [2, 7, 15, 20]:
        fd = central(lambda tt: prob.forward(z, tt), th, k, eps)
        print(f"  th[{k:3d}] an={dth_an[k]: .6e} fd={fd: .6e} "
              f"rel={abs(dth_an[k]-fd)/(abs(fd)+1e-30):.2e}")

    # ----- volume constraints -----
    prob.forward(z, th)
    vc = prob.volume_constraints()
    vg = prob.volume_grads()
    print("=== vg grad (z) ===  vg =", vc['vg'])
    dz, dth_ = vg['vg']
    for k in [5, 33, 77]:
        fd = central(lambda zz: (prob.forward(zz, th),
                                 prob.volume_constraints()['vg'])[1], z, k, eps)
        print(f"  z[{k:3d}] an={dz[k]: .6e} fd={fd: .6e}")

    print("=== vfg grad (z, theta) ===  vfg =", vc['vfg'])
    dzf, dthf = vg['vfg']
    for k in [5, 77]:
        fd = central(lambda zz: (prob.forward(zz, th),
                                 prob.volume_constraints()['vfg'])[1], z, k, eps)
        print(f"  z[{k:3d}] an={dzf[k]: .6e} fd={fd: .6e}")
    for k in [2, 15]:
        fd = central(lambda tt: (prob.forward(z, tt),
                                 prob.volume_constraints()['vfg'])[1], th, k, eps)
        print(f"  th[{k:3d}] an={dthf[k]: .6e} fd={fd: .6e}")

    # ----- curl jacobian -----
    print("=== curl jacobian d zeta_l / d theta_hat_q ===")
    J = prob.curl_jac(th).toarray()
    # pick the largest-magnitude entries (guaranteed coupled)
    flat = np.dstack(np.unravel_index(np.argsort(-np.abs(J), axis=None),
                                      J.shape))[0][:5]
    for l, q in flat:
        fd = central(lambda tt: prob.curl(tt)[l], th, q, eps)
        print(f"  zeta[{l}]/th[{q}] an={J[l, q]: .6e} fd={fd: .6e} "
              f"rel={abs(J[l, q]-fd)/(abs(fd)+1e-30):.2e}")
    print("max|zeta| =", np.max(np.abs(prob.curl(th))))


if __name__ == "__main__":
    main()
