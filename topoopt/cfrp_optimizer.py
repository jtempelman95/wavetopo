"""
MMA + Augmented-Lagrangian driver for the toolpath-integrated CFRP problem.

* ``MMA`` : bound-constrained Method of Moving Asymptotes (Svanberg 1987),
  specialized to m = 0 explicit constraints (all constraints are folded into
  the augmented-Lagrangian objective), so the separable subproblem has a
  closed-form primal solution.

* ``optimize_cfrp`` : the outer augmented-Lagrangian loop of Section 6, with a
  toolpath-generation (wave projection) step every design update (Fig. 4).

Equation numbers refer to Wong, Sanders & Rosen, Compos. Struct. 378 (2026).
"""
from __future__ import annotations

import numpy as np

from .cfrp_problem import CFRPProblem


# ---------------------------------------------------------------------- #
#  Bound-constrained MMA (m = 0)                                          #
# ---------------------------------------------------------------------- #
class MMA:
    def __init__(self, xmin, xmax, move=0.15, asyinit=0.2, asyincr=1.2,
                 asydecr=0.7, albefa=0.1, raa0=1e-5):
        self.xmin = np.asarray(xmin, float)
        self.xmax = np.asarray(xmax, float)
        self.span = np.maximum(self.xmax - self.xmin, 1e-12)
        self.move = move
        self.asyinit, self.asyincr, self.asydecr = asyinit, asyincr, asydecr
        self.albefa, self.raa0 = albefa, raa0
        self.xold1 = None
        self.xold2 = None
        self.low = None
        self.upp = None
        self.it = 0

    def update(self, x, df0dx):
        x = np.asarray(x, float)
        self.it += 1
        xmin, xmax, span = self.xmin, self.xmax, self.span

        # --- moving asymptotes ---------------------------------------- #
        if self.it <= 2:
            self.low = x - self.asyinit * span
            self.upp = x + self.asyinit * span
        else:
            zzz = (x - self.xold1) * (self.xold1 - self.xold2)
            gamma = np.ones_like(x)
            gamma[zzz > 0] = self.asyincr
            gamma[zzz < 0] = self.asydecr
            self.low = x - gamma * (self.xold1 - self.low)
            self.upp = x + gamma * (self.upp - self.xold1)
            self.low = np.clip(self.low, x - 10 * span, x - 0.01 * span)
            self.upp = np.clip(self.upp, x + 0.01 * span, x + 10 * span)

        # --- move limits (alpha, beta) -------------------------------- #
        alpha = np.maximum.reduce([xmin,
                                   self.low + self.albefa * (x - self.low),
                                   x - self.move * span])
        beta = np.minimum.reduce([xmax,
                                  self.upp - self.albefa * (self.upp - x),
                                  x + self.move * span])

        # --- approximation coefficients ------------------------------- #
        dfp = np.maximum(df0dx, 0.0)
        dfm = np.maximum(-df0dx, 0.0)
        raa = self.raa0 / span
        p0 = (self.upp - x)**2 * (1.001 * dfp + 0.001 * dfm + raa)
        q0 = (x - self.low)**2 * (0.001 * dfp + 1.001 * dfm + raa)

        # --- closed-form separable minimizer -------------------------- #
        sp_, sq = np.sqrt(p0), np.sqrt(q0)
        xnew = (sp_ * self.low + sq * self.upp) / (sp_ + sq)
        xnew = np.clip(xnew, alpha, beta)

        self.xold2 = self.xold1
        self.xold1 = x.copy()
        return xnew


# ---------------------------------------------------------------------- #
#  Augmented-Lagrangian outer loop                                       #
# ---------------------------------------------------------------------- #
def optimize_cfrp(prob: CFRPProblem, z0, theta_hat0, *,
                  zeta_all=None, lam0=15.0, mu0=5.0, mu_max=1e7, xi=1.25,
                  max_outer=200, mma_iter=5, tolD=1e-3, tolC=5e-3,
                  ftol=2e-3, patience=20, verbose=True, record=None):
    """Run the toolpath-integrated TO. Returns dict with final state/history.

    Constraints g = [vg, vfg(optional), curl_1..curl_N(optional)].
    """
    N, M = prob.N, prob.M
    have_vfg = prob.vf_all is not None
    have_curl = zeta_all is not None

    # ---- assemble constraint bookkeeping ----------------------------- #
    n_global = 1 + (1 if have_vfg else 0)
    n_local = N if have_curl else 0
    Nc = n_global + n_local

    lam = np.full(Nc, lam0)
    mu = mu0

    # design vector W = [z, theta_hat]
    z = np.array(z0, float)
    th = np.array(theta_hat0, float)
    xmin = np.concatenate([np.zeros(N), np.full(M, -np.pi / 2)])
    xmax = np.concatenate([np.ones(N), np.full(M, np.pi / 2)])
    # lock passive elements at 0
    xmin[:N][prob.passive] = 0.0
    xmax[:N][prob.passive] = 0.0
    z[prob.passive] = 0.0

    mma = MMA(xmin, xmax)
    hist = {'f': [], 'curl_max': [], 'vg': [], 'vfg': []}
    best = {'f': np.inf, 'z': z.copy(), 'th': th.copy(), 'curl_max': np.inf}
    stale = 0

    def eval_all(z, th):
        f = prob.forward(z, th)
        vc = prob.volume_constraints()
        g = [vc['vg']]
        if have_vfg:
            g.append(vc['vfg'])
        if have_curl:
            zeta = prob.curl(th)
            g_curl = zeta**2 - zeta_all**2
            g.extend(g_curl.tolist())
        else:
            zeta = prob.curl(th)
        return f, np.array(g), zeta

    def al_grad(z, th, g):
        """Gradient of the AL function wrt [z, th]. eq (30),(23),(24),(46)."""
        dfz, dfth = prob.objective_grad()
        h = np.maximum(g, -lam / mu)                      # eq (24)
        active = (g > -lam / mu)
        factor = (lam + mu * h) * active                  # per-constraint
        # weights: 1 for global, 1/N for local (eq 23)
        w = np.ones(Nc)
        if have_curl:
            w[n_global:] = 1.0 / N

        gz = dfz.copy()
        gth = dfth.copy()
        vg = prob.volume_grads()
        # vg (global j=0): depends on z only
        dz, _ = vg['vg']
        gz += w[0] * factor[0] * dz
        idx = 1
        if have_vfg:
            dzf, dthf = vg['vfg']
            gz += w[idx] * factor[idx] * dzf
            gth += w[idx] * factor[idx] * dthf
            idx += 1
        if have_curl:
            zeta = prob.curl(th)
            J = prob.curl_jac(th)                          # (N,M) sparse
            fac_curl = factor[idx:]                        # length N
            # d(zeta^2)/dth = 2 zeta J ; weighted sum over local constraints
            gth += w[idx] * (J.T @ (2.0 * zeta * fac_curl))
        gz[prob.passive] = 0.0
        return np.concatenate([gz, gth])

    converged = False
    total_inner = 0
    for k in range(max_outer):
        # ---- inner MMA loop (minimize AL for mma_iter steps) --------- #
        z_start, th_start = z.copy(), th.copy()
        for _ in range(mma_iter):
            f, g, zeta = eval_all(z, th)
            grad = al_grad(z, th, g)
            W = np.concatenate([z, th])
            Wn = mma.update(W, grad)
            z, th = Wn[:N], Wn[N:]
            z[prob.passive] = 0.0
            total_inner += 1

        # ---- evaluate at end of subproblem --------------------------- #
        f, g, zeta = eval_all(z, th)
        curl_max = float(np.max(np.abs(zeta)))
        hist['f'].append(f)
        hist['curl_max'].append(curl_max)
        hist['vg'].append(g[0])
        hist['vfg'].append(g[1] if have_vfg else np.nan)
        if record is not None:
            record(k, prob, z, th, f, curl_max)

        # ---- feasibility + best-design tracking ---------------------- #
        viol = max(g[0], 0.0)
        if have_vfg:
            viol = max(viol, g[1])
        curl_ok = (not have_curl) or (curl_max / zeta_all < 1.0 + tolC)
        feasible = (viol <= ftol) and curl_ok
        if feasible and f < best['f'] - 1e-6:
            best = {'f': f, 'z': z.copy(), 'th': th.copy(),
                    'curl_max': curl_max}
            stale = 0
        else:
            stale += 1

        # ---- AL multiplier / penalty update, eq (26)-(27) ------------ #
        # gate mu growth on actual violation to avoid post-feasibility
        # ill-conditioning (the penalty blowing up after constraints are met)
        h = np.maximum(g, -lam / mu)
        lam = np.maximum(lam + mu * h, 0.0)
        if not (viol <= ftol and curl_ok):
            mu = min(xi * mu, mu_max)

        # ---- convergence test, eq (28)-(29) -------------------------- #
        dchange = max(np.mean(np.abs(z - z_start)),
                      np.mean(np.abs(th - th_start)))
        if verbose:
            msg = (f"[outer {k:3d}] f={f:.4f} |zeta|max={curl_max:6.3f} "
                   f"vg={g[0]:+.4f} dW={dchange:.2e} mu={mu:.1e} "
                   f"best={best['f']:.4f}")
            if have_vfg:
                msg += f" vfg={g[1]:+.4f}"
            print(msg)
        if dchange < tolD and curl_ok and k > 3:
            converged = True
            if verbose:
                print(f"Converged at outer iter {k}.")
            break
        if stale >= patience and k > 10:
            if verbose:
                print(f"No feasible improvement for {patience} iters; stopping.")
            break

    # return the best feasible design found (falls back to last if none)
    if np.isfinite(best['f']):
        z, th = best['z'], best['th']
    f, g, zeta = eval_all(z, th)
    return dict(z=z, theta_hat=th, theta=prob.rbf.theta(th), f=f,
                curl=zeta, curl_max=float(np.max(np.abs(zeta))),
                chi_hat=prob._state['chi_hat'], hist=hist,
                converged=converged, outer_iters=k + 1,
                inner_iters=total_inner, best_f=best['f'])
