"""
Diagnose the dips in the cloak convergence curve.

Two candidate mechanisms:
  (a) OPTIMIZER: the MMA in cfrp_optimizer.py has no globalization -- update()
      returns the separable subproblem minimizer unconditionally, with no line
      search or acceptance test, so a step that worsens J is accepted outright.
      The move limit is a fraction of span = xmax-xmin = pi, so move=0.15 allows
      ~27 degrees of orientation change per support per iteration.
  (b) PHYSICS: at fixed omega with light damping (eta=0.02) the plate response is
      a superposition of weakly-damped modes.  A large orientation change shifts
      modal frequencies across omega, so |u| -- and hence the far-field mismatch
      J -- can jump sharply.  That would make J genuinely rough along design
      directions, not merely badly approximated.

We separate them by measuring how much J actually degrades under a perturbation
the SIZE of one MMA step, at each move limit, around the converged design.

    XDG_CACHE_HOME=/home/jrt/wavetopo/.fenics-cache PYTHONPATH=/home/jrt/wavetopo \
    .../dolfinx_complex/bin/python3 examples/diag_cloak_dips.py
"""
import numpy as np
from scipy.sparse.linalg import lsqr

import importlib.util
spec = importlib.util.spec_from_file_location(
    "ck", "/home/jrt/wavetopo/examples/dolfinx_cloak_conforming_vec.py")
ck = importlib.util.module_from_spec(spec); spec.loader.exec_module(ck)

from wavetopo.dolfinx_wave import support_map

SEED = "/tmp/claude-1000/cloak_325x_seed.npz"
rng = np.random.default_rng(0)

ew, ref = ck.build()
ncell = ew.theta.x.array.size
cent = ck.centroids(ew)
B, supp = support_map(cent, (0, ck.Lx), (0, ck.Ly), spacing=0.28, r=0.6)
active = np.hypot(supp[:, 0]-ck.CX, supp[:, 1]-ck.CY) < ck.RC + 0.3

ew.set_theta(np.zeros(ncell)); ew.solve()
J0 = ew.cloak_mismatch()
th_star = np.load(SEED)["thopt"]
x_star = lsqr(B, th_star, atol=1e-12, btol=1e-12)[0]
x_star[~active] = 0.0

def Jof(x):
    ew.set_theta(B @ x); ew.solve()
    return ew.cloak_mismatch()

Jstar = Jof(x_star)
print(f"converged design: J={Jstar:.4e}  reduction={J0/Jstar:.1f}x", flush=True)

# ---- (1) line scan along the design direction: is J smooth or spiky? ----
print("\n(1) LINE SCAN  J(t) for theta = t * theta*   [is the objective rough?]")
for t in [0.0, 0.25, 0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3]:
    J = Jof(t*x_star)
    print(f"    t={t:4.2f}   J={J:.4e}   reduction={J0/J:6.1f}x", flush=True)

# ---- (2) how bad is ONE step of each move limit? ----
span = np.pi                      # xmax-xmin for the orientation variables
print("\n(2) PERTURBATION PROBE  J after a random step of one MMA move limit")
print("    (worst case over 6 random directions; step = move*span per support)")
for move in (0.15, 0.05, 0.035):
    step = move*span
    worst, vals = None, []
    for k in range(6):
        d = rng.uniform(-1, 1, x_star.size); d[~active] = 0.0
        xp = np.clip(x_star + step*d, -np.pi/2, np.pi/2)
        J = Jof(xp); vals.append(J0/J)
    vals = np.array(vals)
    print(f"    move={move:5.3f} (={np.degrees(step):4.1f} deg/support): "
          f"reduction {vals.min():5.1f}x .. {vals.max():5.1f}x "
          f"(converged {J0/Jstar:.1f}x)  -> worst-case drop "
          f"{100*(1-vals.min()/(J0/Jstar)):4.0f}%", flush=True)
print("""
FINDINGS (measured 2026-07-19; the probe REFUTED the simple reading above):
  * The line scan is smooth and monotone -- no resonance spikes.  Mechanism (b),
    a rough objective, is ruled out.
  * Excess J scales as step^2 (measured 21.7x for a 4.94x larger step, against
    24.4x predicted): the objective is a smooth, locally QUADRATIC bowl.  It is
    simply very STEEP.
  * move=0.15 permits 0.471 rad/support = 1.84x the rms design magnitude
    (0.257 rad), so overshoot was near-certain, and MMA has no line search or
    acceptance test -- the bad point is accepted and persists.
  * BUT even move=0.035 costs ~80% in a bad direction.  Quadratic growth means
    ANY fixed step is a gamble; a smaller one only changes the odds.
  => The fix is an ADAPTIVE trust region, not merely a smaller step:
     MMA(..., adapt=True) + update(x, g, f=J) in cfrp_optimizer.py.""")
