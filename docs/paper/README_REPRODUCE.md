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
- revision R1 analyses (dolfinx env): `examples/analysis/wt_r1_lens.py power fluxmap fine absorber asbuilt`,
  `wt_r1_cloak.py series sponge fields`, `wt_r1_dose_ms.py <s>` for s in 0.25 0.50 0.75 1.00 1.10,
  `wt_r1_panel.py`; plain Python: `wt_r1_rays.py` (ray prediction)
- assets: `examples/analysis/wt_paper_assets.py` (numbers, tables, Fig. 5), `wt_r1_figs.py` (Figs. 3, 4),
  `wt_figs_fields.py` (Figs. 2, 6, 7), `wt_schematic.py` (Fig. 1); then three passes of
  `pdflatex dolfinx_wave_control`.

The submitted version is archived in `docs/paper/archive/R0/` (git tag `R0`); this revision is tag `R1`.

Scripts under `examples/drivers/` and `examples/figures/` belong to earlier exploratory studies.
