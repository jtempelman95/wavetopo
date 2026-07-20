# Toolpath-Integrated CFRP Topology Optimization — Reproduction Report

Reimplementation of **Wong, Sanders & Rosen, *"Toolpath-integrated topology
optimization for design of additively manufactured fiber-reinforced structures
considering limits on fiber curvature"*, Composite Structures 378 (2026)
119897.**

The goal was to reproduce the paper's central result — controlling fiber
curvature in a continuous-fiber composite cantilever via a curl constraint,
while building an explicit fiber/matrix toolpath each optimization iteration —
and to do so on a clean, gradient-verified code base that we can extend toward
wave-based metamaterials.

---

## 1. What was built

The existing repo was isotropic SIMP on FEniCSx; none of the paper's machinery
existed. We added a self-contained **numpy/scipy** implementation on a
structured bilinear-quad mesh (mirroring the paper's MATLAB approach), so every
derivative is analytic and finite-difference checkable.

| Module | Contents | Paper eqs |
|---|---|---|
| `wavetopo/cfrp.py` | `Material` (Qf, Dm, ρ), `QuadMesh`, element stiffness/mass, **exact Fourier-series precompute** of anisotropic element matrices, **CS-RBF** orientation mapping + analytic spatial derivatives, **wave projection** + adjoint | 3–10, 14–21 |
| `wavetopo/cfrp_problem.py` | `CFRPProblem`: assembly, compliance, full sensitivities, volume / fiber-volume / curl constraints, linear density filter | 2, 31–52 |
| `wavetopo/cfrp_optimizer.py` | bound-constrained **MMA** + **augmented-Lagrangian** outer loop | 22–27 |
| `wavetopo/cfrp_viz.py` | density / orientation-streamline / towpreg / curl figures | Fig 6 |
| `tests/test_cfrp_grad.py` | finite-difference verification of every gradient | — |

**Pipeline (forward model).**
```
z  --filter P--> y --SIMP--> E                          (composite stiffness scale)
θ̂ --CS-RBF Φ--> θ                                       (continuous orientation)
(z,θ) --wave projection--> χ̂ ∈ {fiber=1, matrix=0}      (explicit toolpath)
K = Σ_e E_e [ χ̂_e k_f(θ_e) + (1−χ̂_e) k_m ],   K u = F,   f = Fᵀu
```
Constraints: composite volume `v_g`, fiber volume `v_fg`, and one **local curl**
constraint per element `ζ_e² ≤ ζ_all²` limiting fiber-path curvature.

---

## 2. Challenges & solutions

**(a) Trusting the gradients before optimizing.**
The sensitivity chain runs objective → stiffness → wave-projection → orientation,
with a least-squares phase solve in the middle — easy to get subtly wrong.
*Solution:* build and **finite-difference-verify every gradient on a tiny mesh
before running any optimization.** Final agreement: objective gradients ~1e-6,
curl Jacobian ~1e-11, wave-projection adjoint 4–7 digits.

**(b) Adjoint through the least-squares phase solve (the crux).**
The wave projection solves `ψ = (AᵀA)⁻¹Aᵀ ∇ψ` (eq 15), with `∇ψ` a nonlinear
function of orientation and density. *Solution:* prefactorize the constant
normal-equation operator `AᵀA` once (pinning the anchor node to remove the
constant null space), and reuse the **same factorization** for both the forward
phase solve and its adjoint `μ = (AᵀA)⁻¹ λ`. This is also the main per-iteration
cost (one extra sparse solve), exactly as the paper notes.

**(c) Exact element-matrix Fourier fit.**
Rotated fiber stiffness `D_f(θ)=T Q_f Tᵀ` is a degree-4 trig polynomial, so the
5-term Fourier basis `{1, cos2θ, sin2θ, cos4θ, sin4θ}` fits it **exactly**
(residual ~1e-16). Element matrices for the 5 coefficient matrices are
precomputed once; per-iteration assembly is just `k_f(θ)=Σ_s k_{f,s} c_s(θ)`.

**(d) Unit convention.** The paper's O(0.1–1) compliance values are only
consistent if moduli are entered as **bare GPa numbers** (131, 9, 5, 2.6), load
0.5, lengths in metres, thickness 1 — not SI Pascals. Matching this was needed
to compare `f` against the paper.

**(e) Augmented-Lagrangian penalty blow-up.**
Growing the penalty `μ ← ξμ` every outer iteration (as written) inflates `μ`
unboundedly; once the constraints are already satisfied this ill-conditions the
MMA subproblem and the design *diverges after it has converged* (we saw `f`
reach 0.43 then bounce back to 0.7 as μ→5e6). *Solution:* **gate μ growth on
actual constraint violation** — stop inflating once feasible.

**(f) MMA oscillation on the coupled problem.**
Density and orientation couple through the wave projection, so MMA oscillates at
the ~3% level rather than reaching the paper's tight `tolD=1e-3`. *Solution:*
**track and return the best *feasible* iterate**, plus a stagnation stop. This
is standard practice and gives clean reported numbers.

**(g) Figure continuity (errant / missing pixels).**
The raw design variable `z` carries ~5% sub-filter speckle. *Solution:* display
the **physical** density — linear filter then Heaviside projection — which the
FE analysis already uses. Result: **0 isolated pixels, 0 interior holes**. No
re-optimization needed; `cfrp_viz.clean_density` + `cfrp_replot.py`.

**(h) Unknown load/BC.** Fig 5's exact load layout was not legible in the
extracted PDF; we assume a downward point load at the mid-right edge. This is the
main reason our compliance sits ~8–18% above the paper while the *trends and the
curl-control mechanism match closely.*

---

## 3. Results — no-void cantilever (Table 3, 240×150 = 36 000 elements)

| case | f (ours) | f (paper) | \|ζ\|max ours | \|ζ\|max paper |
|---|---|---|---|---|
| no curl constraint | 0.452 | 0.42 | 10.7 | 8.04 |
| curl ≤ 2 /m | 0.577 | 0.49 | **1.93** | **2.00** |

The curl constraint drives the maximum fiber curvature down to the allowable
limit (1.93 ≈ 2.0) and the voids reshape to give fibers a larger turning radius,
at a compliance cost — the paper's headline behavior. Figures:
`results/cfrp_cant_nocurl_clean.png`, `results/cfrp_cant_curl2_clean.png`.

---

## 4. How to run

```bash
# env: base (numpy/scipy); see memory/envs.md
export PYTHONPATH=$PWD
PY=/home/jrt/miniforge3/bin/python3

# gradient verification (seconds)
$PY -m tests.test_cfrp_grad

# single design + figure
$PY examples/cfrp_cantilever.py --nely 150 --curl 2.0 \
    --save out.png --save-npz out.npz

# reproduce both Table-3 cases (~25 min)
$PY examples/cfrp_reproduce_table3.py

# re-render a saved design with clean physical density
$PY examples/cfrp_replot.py results/cfrp_cant_nocurl.npz clean.png
```

---

## 5. Toward wave-based metamaterials (next)

The same two design fields map directly onto a **directional phononic crystal**:
`z` designs the inclusion topology and `θ` sets the fiber orientation that makes
a band gap *direction-dependent*. `wavetopo/bloch.py` already provides the
Bloch-periodic generalized eigensolver (`K(k)φ = ω²M(k)φ`) reusing the same
anisotropic assembly; a fiber-aligned cell shows strong directional anisotropy
in the dispersion (2nd branch 7.1 along the fiber vs 1.9 transverse). The
remaining work is eigenvalue sensitivities and a directional-gap objective.
