# Reproducing the wavetopo paper

Solves use a dolfinx (FEniCSx) environment with a real-scalar PETSc; post-processing and figures use a
plain Python with numpy/scipy/matplotlib and LaTeX (pgf figures). Run from the repository root with
`PYTHONPATH=.` (and `XDG_CACHE_HOME=.fenics-cache` for dolfinx). The repository is self-contained:
the plotting style (`examples/analysis/paperstyle.py`) and lamina helpers (`examples/analysis/lamina.py`)
are included.

Paper-producing scripts (all write into `results/`):
- designs: `examples/dolfinx/dolfinx_lens.py`, `dolfinx_cloak_conforming_vec.py`,
  `dolfinx_panel.py --band 44,47,50,53,56 --lam-p 3 --stiff 0.3 --tag _band_repro` (and `--band 50 ... --tag _tone_repro`),
  `dolfinx_panel_sweep.py --repro`
- analysis: `examples/analysis/lens_mechanism.py`, `lens_flux.py`, `lens_flux_control.py`, `lens_dose.py`,
  `lens_front.py` (+ `lens_front_robust.py <zeta>`), `cloak_band.py`, `wt_verify.py A B C D`, `cloak_fine.py`,
  `cloak_fine_band.py`, `lens_absorber.py`, `panel_mesh_check.py`
- assets: `examples/analysis/wt_paper_assets.py` (numbers, tables, summary figures),
  `wt_figs_fields.py` (field figures), `wt_schematic.py` (Fig. 1); then `pdflatex dolfinx_wave_control`.

Scripts under `examples/drivers/` and `examples/figures/` belong to earlier exploratory studies.
