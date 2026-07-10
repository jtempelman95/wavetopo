# topoopt

Topology **and fiber-orientation** optimization of continuous-fiber composites,
for quasi-static design and **elastic wave control**. The fiber *toolpath*
(orientation field) and structural *topology* (density field) are optimized
together; only the physics/objective changes between applications.

The wave-control core is pure **numpy/scipy** (no FEniCSx needed) and lives in
the `topoopt/` package; all sensitivities are adjoint-based and
finite-difference verified (`tests/test_wave_grad.py`, `tests/test_cfrp_grad.py`).

## Capabilities & how to run

Use the base conda env (`/home/jrt/miniforge3/bin/python3`, numpy/scipy) and set
`PYTHONPATH=$PWD`. Figures land in `results/`.

| Deliverable | Script | Result |
|---|---|---|
| **CFRP validation** (Wong et al. 2026: curvature-constrained cantilever) | `examples/cfrp_reproduce_table3.py` | `results/cfrp_cant_*_clean.png` |
| **Wave-energy localization** (single-freq lens, 20× focus) | `examples/wave_lens.py` | `results/wave_lens.png` |
| **Broadband localization** (multi-freq max–min lens) | `examples/wave_lens_multifreq.py` | `results/wave_lens_multifreq.png` |
| **Elastic cloak** (void + toolpath, 15× less scatter) | `examples/wave_cloak.py` | `results/wave_cloak.png` |
| **Topological valley waveguide** (triangular-rod, orientation = Dirac mass) | `examples/valley_phase.py`, `examples/valley_ribbon_tri.py`, `examples/valley_waveguide_tri.py` | `results/valley_phase.png`, `valley_ribbon_tri.png`, `valley_wg_tri_straight.png` |
| **Periodic metamaterials** (Bloch bands, band gaps) | `examples/valley_mass.py`, `examples/bandgap_demo.py` | `results/valley_*.png` |

```bash
export PYTHONPATH=$PWD
PY=/home/jrt/miniforge3/bin/python3
$PY -m tests.test_wave_grad          # verify wave-control gradients
$PY examples/wave_lens_multifreq.py  # broadband lens (~3 min)
$PY examples/wave_cloak.py           # elastic cloak (~1 min)
```

### Paper

`docs/paper/toolpath_wave.tex` — full write-up (method, adjoints, all results,
honest topology negative result). Build: `cd docs/paper && pdflatex toolpath_wave`.

### Package map

- **Modeling**: `cfrp.py` (anisotropic FE, exact Fourier stiffness, CS-RBF
  orientation map, wave-projection toolpaths), `scalar.py` (antiplane-shear).
- **Quasi-static CFRP**: `cfrp_problem.py`, `cfrp_optimizer.py` (MMA + augmented
  Lagrangian), `cfrp_viz.py`.
- **Wave control**: `harmonic.py` (`HarmonicLens`, `HarmonicCloak`),
  `wave_control.py` (`WaveFocus`, single/multi-frequency).
- **Periodic (Bloch)**: `bloch.py` (bands, FHS Berry curvature, eigen
  sensitivities), `bandgap_opt.py`, `valley_opt.py`, `rhombic.py`.
- **Topological valley-Hall**: `honeycomb_mesh.py` (periodic gmsh mesh +
  `HoneycombBloch`), `valley_cell.py` (triangular-rod cell/ribbon; rod rotation
  = tunable Dirac mass). Requires `gmsh` (in the base env).

---

## Legacy: SIMP compliance (FEniCSx / dolfinx)

The original module implements classical **SIMP** with a **Helmholtz PDE filter**
and **Optimality Criteria** density update (canonical cantilever benchmark).
Requires a FEniCSx environment.

---

## Problem: Cantilever Beam

```
  ╔══════════════════════════════════╗
  ║                                  ║ ← F (downward, right edge)
  ║          find optimal ρ(x)       ║
  ║                                  ║
  ╚══════════════════════════════════╝
  ↑ fixed (clamped left edge)
```

| Parameter | Value |
|-----------|-------|
| Domain    | 2 × 1 (length × height) |
| BC        | Left edge clamped; right edge uniform downward load F = 1 |
| Material  | E₀ = 1, ν = 0.3 (plane stress) |
| Objective | Minimize compliance C = f^T u |
| Constraint| Volume fraction ≤ vf (default 40%) |

---

## Method

### SIMP Penalization

The Young's modulus of each element is interpolated as:

```
E(ρ_e) = E_min + ρ_e^p · (E₀ − E_min)
```

where p = 3 forces intermediate densities towards 0 or 1.

### Helmholtz Filter

A PDE filter (Lazarov & Sigmund 2016) regularises the density field and
prevents checkerboard patterns:

```
−r² Δρ̃ + ρ̃ = ρ     (homogeneous Neumann BC)
r = R_min / (2√3)
```

### Optimality Criteria Update

```
ρ_e^{new} = clip( ρ_e · (|dC/dρ_e| / λ)^η , ρ_e − move, ρ_e + move )
```

λ is found by bisection to satisfy the volume constraint exactly each
iteration; η = 0.5 is a numerical damping factor.

---

## Installation

```bash
# 1. Create the dolfinx conda environment (if not already present)
conda create -n dolfinx_env -c conda-forge fenics-dolfinx=0.7 mpi4py petsc4py matplotlib pytest
conda activate dolfinx_env

# 2. Clone the repo
git clone <repo-url>
cd topoopt
```

No pip install is needed; just run from the repo root.

---

## Quick Start

```bash
# Run the cantilever example (60×120 mesh, ~2–5 min)
conda run -n dolfinx_complex python examples/cantilever.py

# Coarser mesh for testing
conda run -n dolfinx_complex python examples/cantilever.py --nx 30

# Save VTK output
conda run -n dolfinx_complex python examples/cantilever.py --save-vtk

# All options
conda run -n dolfinx_complex python examples/cantilever.py --help
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--nx` | 60 | Elements in y-direction (x gets 2×nx) |
| `--vf` | 0.4 | Target volume fraction |
| `--penal` | 3.0 | SIMP penalization exponent |
| `--r-min` | 0.04 | Filter radius (fraction of beam height) |
| `--max-iter` | 100 | Maximum OC iterations |
| `--tol` | 1e-3 | Convergence tolerance (max density change) |
| `--no-plot` | — | Suppress matplotlib output |
| `--save-vtk` | — | Write XDMF/VTK output |

---

## Running Tests

```bash
conda run -n dolfinx_complex python -m pytest tests/ -v
```

Tests cover:
- FE mesh cell count
- Compliance positivity
- Solid stiffer than void
- Sensitivity sign
- Filter invariance on uniform fields
- Filter smoothing of checkerboard
- OC volume constraint satisfaction
- Integration test: 15-iter optimization on 16×8 mesh

---

## Repository Layout

```
topoopt/
├── topoopt/
│   ├── __init__.py       # public API
│   ├── fem.py            # ElasticityProblem (FEniCSx)
│   ├── filters.py        # HelmholtzFilter, ProjectionFilter
│   └── optimizer.py      # SIMPOptimizer (OC method)
├── examples/
│   └── cantilever.py     # Cantilever beam example
├── tests/
│   ├── test_fem.py       # Unit tests
│   └── test_integration.py  # Integration test
├── setup.py
└── README.md
```

---

## References

1. Bendsøe, M. P. & Sigmund, O. (2004). *Topology Optimization: Theory,
   Methods and Applications*. Springer.
2. Lazarov, B. S. & Sigmund, O. (2016). Filters in topology optimization based
   on Helmholtz-type differential equations. *IJNME* 86(6), 765–781.
3. Wang, F., Lazarov, B. S. & Sigmund, O. (2011). On projection methods,
   convergence and robust formulations in topology optimization using the SIMP
   method. *SMO* 43, 767–784.
4. Andreassen, E., et al. (2011). Efficient topology optimization in MATLAB
   using 88 lines of code. *SMO* 43, 1–16.
