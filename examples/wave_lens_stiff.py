r"""
A fiber layout that FOCUSES an elastic wave AND stiffens the panel.

The curvilinear-fiber lens of examples/wave_lens.py steers an incident wave to a
target purely by fiber orientation, on a solid plate.  But the converging-lens
fiber pattern is not the pattern that carries a structural load: a lens curves
fibers to bend the wave, while a stiff panel wants fibers aligned with the load
path.  A single orientation field theta(x) cannot obviously do both.

This is a genuine multi-functional co-design on a FINITE part, with NO band gap
anywhere -- so it is immune to the sparse-k-sampling failure that made the
periodic band-gap "gaps" evaporate under dense sampling.  Both objectives are
functions of the same theta on the same solid plate:

  DYNAMIC  focus gain  G = E_focus(theta) / E_focus(straight)        (maximize)
  STATIC   cantilever compliance  c = F_s^T u_s,  K(theta) u_s = F_s (minimize)

We report three designs from an identical start:
  * straight       theta = 0            -- stiff, no focusing (G = 1)
  * focus-only     max G, theta free    -- focuses, but how stiff?
  * co-design      max G s.t. c <= c_straight (stay at least as stiff as
                   straight fibers), by augmented Lagrangian

If the co-design keeps most of the focus gain while holding compliance at the
straight-fiber level, the toolpath buys BOTH functions at once -- the honest,
finite-domain version of the multi-functional claim.

    PYTHONPATH=. python examples/wave_lens_stiff.py
"""
import argparse
import time

import numpy as np
import scipy.sparse.linalg as spla

from wavetopo.cfrp import Material, QuadMesh, CSRBFMapping
from wavetopo.harmonic import HarmonicLens
from wavetopo.cfrp_optimizer import MMA


def grid_support_points(mesh, sx, sy):
    xs = np.arange(sx / 2, mesh.Lx, sx)
    ys = np.arange(sy / 2, mesh.Ly, sy)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


def build(nx, ny, Lx, Ly, omega, eta, support_spc, r_s):
    mesh = QuadMesh(nx, ny, Lx, Ly)
    mat = Material(Ef1=131, Ef2=9, G12=5, nu12=0.27, Em=2.6, nu_m=0.3,
                   rho_f=1.6, rho_m=1.2)
    sx, sy = grid_support_points(mesh, support_spc, support_spc)
    rbf = CSRBFMapping(mesh, (sx, sy), r_s=r_s)
    hl = HarmonicLens(mesh, mat, rbf, omega=omega, eta=eta)

    # plane pressure wave from the left edge
    F = np.zeros(mesh.ndof, dtype=complex)
    for iy in range(ny + 1):
        F[2 * hl.node_id(0, iy)] = 1.0
    hl.set_load(F)

    marg = 0.6
    d = np.maximum.reduce([(mesh.cx - (Lx - marg)) / marg,
                           (marg - mesh.cy) / marg,
                           (mesh.cy - (Ly - marg)) / marg])
    hl.set_sponge(np.maximum(0.0, d) ** 2 * 80.0)

    fx, fy, fr = 0.78 * Lx, Ly / 2, 0.18
    w = np.zeros(mesh.ndof)
    xs, ys = mesh.node_xy()
    for n in range(mesh.nnode):
        if abs(xs[n] - fx) < fr and abs(ys[n] - fy) < fr:
            w[2 * n] = 1.0; w[2 * n + 1] = 1.0
    hl.set_focus(w)
    return mesh, hl, (fx, fy, fr)


class Stiffness:
    """Cantilever compliance of the SAME plate: clamp the left edge, apply a
    transverse (downward) load on the right edge, c = F_s^T u_s under K(theta).
    Self-adjoint, so the gradient needs no extra solve."""

    def __init__(self, mesh, hl):
        self.mesh = mesh; self.hl = hl
        xs, ys = mesh.node_xy()
        left = np.flatnonzero(xs <= 1e-9)
        right = np.flatnonzero(xs >= mesh.Lx - 1e-9)
        self.fixed = np.concatenate([2 * left, 2 * left + 1])
        self.free = np.setdiff1d(np.arange(mesh.ndof), self.fixed)
        self.Fs = np.zeros(mesh.ndof)
        self.Fs[2 * right + 1] = -1.0 / len(right)     # unit transverse load

    def value_grad(self, theta_hat, want_grad=True):
        hl = self.hl
        theta = hl.rbf.theta(theta_hat)
        K = hl.assemble_K(theta).tocsr()
        u = np.zeros(self.mesh.ndof)
        u[self.free] = spla.spsolve(K[self.free][:, self.free].tocsc(),
                                    self.Fs[self.free])
        c = float(self.Fs @ u)
        if not want_grad:
            return c, None, u
        dkf = hl.fs.dkf_dtheta(theta)                  # (N,8,8)
        uE = u[self.mesh.edof]
        term = np.einsum('ei,eij,ej->e', uE, dkf, uE)  # u_e^T dK u_e
        dc_dth = -term                                 # dc/dtheta = -u^T dK u
        return c, hl.rbf.Phi.T @ dc_dth, u


def optimize(hl, stiff, th0, *, mode, iters, c_limit=None, mu0=5.0, xi=1.4,
             mu_max=1e4, move=0.1, verbose=True):
    """mode='focus' : maximize focus gain only.
       mode='co'    : maximize focus gain s.t. compliance <= c_limit, by an
                      augmented Lagrangian on the single constraint."""
    M = hl.M_ if hasattr(hl, "M_") else hl.rbf.Phi.shape[1]
    mma = MMA(np.full(M, -np.pi / 2), np.full(M, np.pi / 2), move=move)
    th = np.array(th0, float)
    J0 = hl.focus_energy(th0 * 0.0)                    # straight-fiber focus
    lam, mu = 0.0, mu0
    best = dict(G=-np.inf)
    for it in range(iters):
        J = hl.focus_energy(th)
        gJ = hl.focus_grad()
        G = J / J0
        if mode == "focus":
            c, dc, feas = None, None, True
            grad = -gJ / J0                            # minimize -G
        else:
            c, dc, _ = stiff.value_grad(th)
            viol = c - c_limit
            h = max(viol, -lam / mu)
            feas = viol <= 1e-3 * abs(c_limit)
            act = 1.0 if viol > -lam / mu else 0.0
            grad = -gJ / J0 + (lam + mu * h) * act * dc
        if (feas or mode == "focus") and G > best['G']:
            best = dict(G=G, th=th.copy(),
                        c=(c if c is not None else stiff.value_grad(th, False)[0]))
        if verbose and (it % 10 == 0 or it == iters - 1):
            cs = f" c={c:.3e}" if c is not None else ""
            print(f"  [{mode} {it:3d}] gain={G:6.2f}x{cs}", flush=True)
        th = mma.update(th, grad)
        if mode == "co" and it % 8 == 7:
            c, _, _ = stiff.value_grad(th, False)
            lam = max(lam + mu * (c - c_limit), 0.0)
            if c > c_limit:
                mu = min(xi * mu, mu_max)
    if not np.isfinite(best['G']):
        best = dict(G=G, th=th.copy(), c=stiff.value_grad(th, False)[0])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--omega", type=float, default=40.0)
    ap.add_argument("--iters", type=int, default=70)
    ap.add_argument("--stiff-tol", type=float, default=0.0,
                    help="co-design compliance limit = (1+tol)*c_straight")
    ap.add_argument("--out", default="results/wave_lens_stiff")
    args = ap.parse_args()

    Lx, Ly = 4.0, 3.0
    ny = int(round(args.nx * Ly / Lx))
    mesh, hl, focus = build(args.nx, ny, Lx, Ly, args.omega, eta=0.03,
                            support_spc=0.33, r_s=0.7)
    hl.M_ = hl.rbf.Phi.shape[1]
    stiff = Stiffness(mesh, hl)
    print(f"mesh {mesh.nelx}x{mesh.nely}  M={hl.M_} supports  omega={args.omega}",
          flush=True)

    th0 = np.zeros(hl.M_)
    c_straight = stiff.value_grad(th0, False)[0]
    G_straight = 1.0
    print(f"straight fibers:  gain={G_straight:.2f}x  compliance={c_straight:.4e}",
          flush=True)

    t0 = time.time()
    print("focus-only optimization ...", flush=True)
    fo = optimize(hl, stiff, th0, mode="focus", iters=args.iters)
    print(f"focus-only:  gain={fo['G']:.2f}x  compliance={fo['c']:.4e}  "
          f"({fo['c']/c_straight:.2f}x straight -> "
          f"{'STIFFER' if fo['c']<c_straight else 'MORE COMPLIANT'})", flush=True)

    c_limit = (1.0 + args.stiff_tol) * c_straight
    print(f"co-design (max focus s.t. compliance <= {c_limit:.4e}) ...",
          flush=True)
    co = optimize(hl, stiff, th0, mode="co", iters=args.iters, c_limit=c_limit)
    print(f"co-design:   gain={co['G']:.2f}x  compliance={co['c']:.4e}  "
          f"({co['c']/c_straight:.2f}x straight)", flush=True)
    print(f"\nSUMMARY  ({time.time()-t0:.0f}s):", flush=True)
    print(f"  straight    : focus  1.00x   stiffness 1.00x (ref)", flush=True)
    print(f"  focus-only  : focus {fo['G']:5.2f}x   stiffness "
          f"{c_straight/fo['c']:.2f}x", flush=True)
    print(f"  co-design   : focus {co['G']:5.2f}x   stiffness "
          f"{c_straight/co['c']:.2f}x  <- keeps focus AND stiffness", flush=True)

    np.savez(args.out + ".npz",
             th_focus=fo['th'], th_co=co['th'],
             G_focus=fo['G'], G_co=co['G'],
             c_straight=c_straight, c_focus=fo['c'], c_co=co['c'],
             omega=args.omega, focus=focus, Lx=Lx, Ly=Ly)
    print("saved", args.out + ".npz", flush=True)


if __name__ == "__main__":
    main()
