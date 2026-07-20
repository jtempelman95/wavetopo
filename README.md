# wavetopo

**Topology and fiber-orientation optimization of continuous-fiber composites, for
elastic wave control.**

Design the *local fiber angle* `θ(x)` of a continuous-fiber composite plate — one
material, no topology changes — and steer in-plane elastic waves with it: focus
them, cloak a void, route them around a bolt hole, or flatten a phononic band.
Every design is delivered as a manufacturable toolpath under a printer-realistic
fiber-curvature limit.

---

## The one idea this rests on

The rotated orthotropic plane-stress stiffness is

```
D_f(θ) = T(θ) Q_f T(θ)ᵀ
```

and `∂D_f/∂θ ≡ 0` **exactly when `Q_f` is isotropic**. Every design sensitivity in
this repository is assembled from `∂D_f/∂θ`, so for an isotropic plate the
gradient vanishes identically and the optimizer returns the baseline unchanged.

Fiber orthotropy (`E_f1/E_f2 ≈ 15`, a wave-speed ratio of ≈3.8) is therefore the
*necessary and sufficient* mechanism for orientation-based wave control — not an
enhancement to it. The isotropic case is a genuine null control, not a strawman.

---

## Results at a glance

| study | driver | outcome |
|---|---|---|
| Curvilinear wave lens | `examples/dolfinx_lens.py` | **78×** focus-energy gain |
| …under `\|ζ\| ≤ ζ_all` | `dolfinx_lens.py curl` | 12× — the manufacturability price |
| Multi-target (2 foci + null) | `dolfinx_lens.py multi` | target/null contrast 1.9 → 19.2 |
| Two through-holes (conforming) | `dolfinx_lens.py hole` | 49× / 17× constrained, feasible |
| Asymmetric holes | `dolfinx_lens.py holeasym` | 50×, re-solved per geometry |
| Elastic cloak (conforming void) | `dolfinx_cloak_conforming_vec.py` | **36×** scatter reduction |
| Energy routing around a joint | `dolfinx_guide_joint.py` | 698× exit, joint 83% quieter, 97% confinement |
| Flat band + gaps | `flatband_demo.py` | width 2.78 → 0.59, both gaps open |

All numbers are measured, reproducible from the tracked data, and reported with
their caveats (see *Honest limitations*).

---

## Install

Two environments, deliberately separate:

```bash
# 1) core: pure numpy/scipy — Bloch metamaterials, all figure generation
conda create -n wavetopo python=3.11 numpy scipy matplotlib
pip install -e .

# 2) FE wave control: FEniCSx/dolfinx 0.7.3 with a REAL PETSc build + gmsh
conda create -n dolfinx_complex -c conda-forge fenics-dolfinx=0.7.3 python-gmsh
```

Everything except the `dolfinx_*` drivers runs in the core environment — including
**every figure**, because the design data is tracked.

```bash
export PYTHONPATH=/path/to/wavetopo
export XDG_CACHE_HOME=/path/to/wavetopo/.fenics-cache   # dolfinx JIT cache
```

---

## Quick start

```bash
# rebuild every paper figure from tracked data (no PDE solves, ~10 min)
python examples/make_figures.py --all

# see what is stale (source data newer than the published figure)
python examples/make_figures.py --list

# just one, while restyling
python examples/make_figures.py lens
```

Interactive, for hand-tuning:

| script | use |
|---|---|
| `examples/figures_playground.py` | **every knob exposed.** One `# %%` cell per figure, plotting code inline. Open in VS Code, `Shift+Enter`. |
| `examples/figures_interactive.py` | thinner: calls the shared builders per cell |
| `examples/make_figures.py` | batch/CLI, fixed styling — what publishes the paper figures |

---

## Repository layout

```
wavetopo/                 the package
  cfrp.py                 Q4 mesh, orthotropic Q_f, CS-RBF map, wave projection
  bloch.py                Bloch reduction T(k), band structure, eigen sensitivities
  flatband_opt.py         flat-band + band-gap co-optimizer (density + orientation)
  cfrp_optimizer.py       MMA (with an optional adaptive trust region)
  dolfinx_elastic.py      in-plane vector wave control, real 4-field split, adjoint
  dolfinx_mesh.py         conforming gmsh meshes with N circular holes cut out
  dolfinx_wave.py         CS-RBF operators, fiber-curvature (curl) constraint
  dolfinx_viz.py          phase-field toolpath delineation, director maps
  figlib.py               shared figure primitives (panels, colorbars, layout)

examples/                 drivers (dolfinx_*) and figure tooling
docs/paper/               the write-up
  dolfinx_wave_control.tex   full derivations, algorithms, results
  slides.tex                 beamer talk
  results_macros.tex         result numbers, \input by BOTH — cannot drift
results/                  *_data.npz design data (tracked); *.png (regenerated)
tests/                    gradient verification against finite differences
```

---

## Method, in brief

**Physics.** Time-harmonic in-plane elastodynamics, genuine vector fields:

```
[(1 + iη)K(θ) − ω²M + iωC] u = f
```

with structural damping `η` and a quadratic absorbing sponge `C`.

**Real four-field split.** The installed PETSc is a *real* build, so the complex
system is solved as

```
[ A  −B ] [u_r]   [f_r]         A = K − ω²M
[ B   A ] [u_i] = [f_i]         B = ηK + ωC
```

This is exact (realification of `a+ib`), and — the part that matters —
`𝒜ᵀ` is the realification of `(A+iB)ᴴ`, so **the transposed real solve returns
the true complex adjoint**, not an approximation. Cost ≈2× a native complex LU;
with complex PETSc the whole construction collapses and no result changes.

**Sensitivity.** Discrete adjoint by UFL differentiation of the residual,
FD-verified to ~1e-6. One LU factorization serves forward *and* adjoint.

**Manufacturing is in the loop, not after it.** Orientation is parametrized by a
compactly-supported RBF (Wendland C²); fiber-path curvature
`ζ = (n·∇)θ` is bounded by `ζ_all`; toolpaths are delineated by wave projection
`χ = ½ + ½cos(2πψ/d)` with `∇ψ ⟂` the fibers. On periodic cells the phase is made
*exactly* cell-periodic (FFT Helmholtz projection + an integer number of tow
cycles), and designs are re-simulated **as-manufactured**.

**Two solvers, stated plainly.** Wave control runs in dolfinx (unstructured,
conforming, PETSc + UFL adjoints). Bloch band structure runs in an independent
numpy/scipy Q4 code (structured, exact periodicity, dense `eigh`). They share the
constitutive model and the design machinery but no assembly or linear algebra —
each is a check on the other.

---

## Things that were learned the hard way

Recorded because each cost real time and each failed *silently*:

- **Conforming meshes matter.** A SIMP soft void staircases the boundary to one
  element; it scatters off its own corners and blurs the radius the curvature
  rule `ζ_all = 1/R` is tied to. Cutting the disk from the geometry roughly
  doubled the achievable cloaking.
- **Region specification is where the bugs hide.** Overlapping reward and penalty
  regions silently penalize the very path energy is asked to follow. Rewarding
  *accumulation* instead of *transport* makes the optimizer pool energy upstream
  and never rotate the downstream fibers. Both return healthy headline numbers.
- **Unbounded rewards defeat their own objective.** A delivery reward linear in
  `E_exit` reached 6300× and outweighed the joint-protection penalty 760:1 — the
  design bought delivery by making the joint 20× *busier*. A `log` reward fixes it.
- **Design freedom must be matched by validation sampling.** 64 orientation
  variables against 22 Brillouin-zone wavevectors *overfits*: the true bandwidth
  came out worse than the coarsest design. Diagnostic, free: since KS aggregation
  can only over-estimate a maximum, a fine-path width *exceeding* the aggregate
  proves the band is excursing between samples.
- **MMA has no globalization.** It accepts a step that worsens the objective. The
  cloak's convergence dips were step size, not physics (excess J scales as
  step²); an adaptive trust region took 23.5× → 39.6× with zero dips.
- **MUMPS over PETSc's built-in LU is 4.4× faster** with identical answers, and
  `OMP_NUM_THREADS` does nothing for sparse LU — real multicore needs MPI.

---

## Honest limitations

- The cloak is **not converged** — still improving at the iteration cap. The
  reported figure is a lower bound, not an optimum.
- The asymmetric-hole design ends **1.7% above** its curvature limit: the
  constraint is a smooth penalty under continuation, not a hard bound.
- The `§7.3` cloak curvature study still uses the **soft-void** mesh, so its
  reduction factors are not comparable to the conforming cloak's.
- The drivers save only `|u|`; run `examples/resolve_full_fields.py` to recover
  Re/Im components (it re-solves saved designs, ~25 s total).
- Cloak numbers across sections are **not** mutually comparable — domain size,
  frequency and whether the design is shell-confined all differ, and each is
  stated where it is reported.

---

## Reference

The manufacturing machinery — CS-RBF orientation field, wave-projection
toolpaths, and the curvature constraint tied to a minimum turning radius — is
adopted from:

> M. Wong, C. Sanders, D. Rosen, *Toolpath-integrated topology optimization for
> design of additively manufactured continuous-fiber composites*,
> **Composite Structures** (2025).

That work optimizes *static* compliance. The contribution here is to carry the
framework into **time-harmonic elastic wave control** and periodic metamaterials,
with a complete discrete-adjoint derivation and a verifiable FEniCSx realization.

*The reference PDF is copyrighted by Elsevier and is deliberately not
redistributed in this repository.*

---

## Building the documents

```bash
cd docs/paper
pdflatex dolfinx_wave_control.tex && pdflatex dolfinx_wave_control.tex
pdflatex slides.tex && pdflatex slides.tex
```

Both use `newtxtext`/`newtxmath`. It is loaded unconditionally so MiKTeX's
install-on-demand can fetch it; set `\newtxfalse` in the preamble to build with
default fonts instead. `amssymb` is deliberately **not** loaded alongside
newtxmath (which supplies those symbols itself and clashes otherwise).
