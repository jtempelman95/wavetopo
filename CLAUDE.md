# CLAUDE.md — Autonomous Wave Control Research Agent

## Directive

Work continuously without asking for permission or confirmation. The user is unavailable. Your job is to produce physically meaningful results and improve the codebase until they exist.

**A result is not done until the wave field makes physical sense.** After every optimization run, plot the displacement magnitude |u(x)| and inspect it:
- For a lens: energy must visibly concentrate at the target region. A diffuse or uniform field means the optimizer converged to a bad local minimum — perturb and re-run.
- For a cloak: the shadow behind the void must be suppressed. If the scattered field looks the same with and without optimized fibers, the objective or adjoint has a bug — find and fix it.
- For a waveguide/channel: energy must travel along the channel with visible confinement. Leakage spreading uniformly into the bulk means the design has not converged — keep iterating.
- For any other geometry: describe what physical success looks like, check the field against it, and only move on when it's there.

If a field looks wrong, do not move on. Fix the code or re-run with better parameters.

RESULTS MUST LOOK PHYSIAL- IF WE HAVE CLEARLY DICSONNECTED OR BARELY CONNECTED DOMAINS IN TOPOLOGY PROBLEMS, THATS AN ISSUE!

---

## Autonomy Rules

1. **Never stop to ask.** If something is ambiguous, make a reasonable engineering judgment and proceed.
2. **Fix bugs yourself.** If a run errors or produces nonsense, read the traceback, identify the cause, patch the code, and re-run.
3. **Keep iterating.** If the objective has not plateaued (< 5% change over 50 iterations), keep running. Do not report a result mid-convergence.
4. **Restart intelligently.** If optimization stalls at a bad local minimum (J flat, field looks wrong), add small random noise (σ = 0.1 rad) to θ̂ and resume MMA — do not restart from scratch.
5. **Save everything.** After a good result: save the field plot as a PDF, save raw arrays as `.npz`, write the key metric into the LaTeX source. Then move to the next task.
6. **Live instructions override this file.** If the user left explicit direction in the current session, follow that first.

---

## Sanity Checks — Run Before Trusting Any New Gradient

### Finite-difference gradient check
```python
eps = 1e-5
for i in range(min(5, n_support)):
    theta_plus = theta_hat.copy(); theta_plus[i] += eps
    theta_minus = theta_hat.copy(); theta_minus[i] -= eps
    dJ_fd = (evaluate_objective(theta_plus) - evaluate_objective(theta_minus)) / (2 * eps)
    rel_err = abs(dJ_analytic[i] - dJ_fd) / (abs(dJ_fd) + 1e-12)
    assert rel_err < 1e-4, f"Gradient wrong at DoF {i}: rel_err={rel_err:.2e}"
```
If this fails, do not run optimization. Fix the adjoint first.

### Isotropy-annihilation check (run after any adjoint code change)
When **Q**_f is set to isotropic (Q_11 = Q_22, Q_66 = (Q_11 − Q_12)/2), every ∂J/∂θ_ℓ must be zero. A non-zero value means a bug in the sensitivity chain — find it before continuing.

---

## Physical Standards

- Curvature-feasible design: |ζ|_max ≤ ζ_all in the final result. If not, tighten curvature continuation (raise μ_max, restart from current θ̂).
- Conforming mesh for any geometric feature (holes, channels, voids): use gmsh/OCC, never SIMP soft-void on a structured grid. Staircase boundaries scatter and corrupt gradient-based objectives.
- Every result must be compared to the straight-fiber (θ ≡ 0) baseline. Report the gain ratio, not just the absolute metric.
- Toolpaths must be visually smooth and printable. Run `plot_toolpaths_phase` and look at them.

---

## Material Parameters (fixed)
```
E_f1 = 131 GPa,  E_f2 = 9 GPa,  G_12 = 5 GPa,  ν_12 = 0.27
rho = 1.6,  eta = 0.02–0.03 (structural damping)
State space: [P1]^4 real four-field split (no complex PETSc)
```
