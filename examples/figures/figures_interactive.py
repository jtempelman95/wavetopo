# %% [markdown]
# # Paper figures — interactive
#
# Run these cells in the VS Code **Python Interactive** window (`Shift+Enter`).
# Every figure is rebuilt from `results/*_data.npz`, so nothing here re-solves a
# PDE — restyle and re-run a cell in a second.
#
# * **Interpreter:** the base conda env (`/home/jrt/miniforge3/bin/python3`).
#   dolfinx is *not* needed — the drivers already saved the triangulation, the
#   fields and the design.
# * **Batch equivalent:** `python examples/figures/make_figures.py --all`
#
# Each cell calls the same builder the batch driver uses, so what you see here is
# exactly what gets published. The last section shows how to compose a custom
# variant from the `figlib` primitives instead.

# %%
# --- setup: run once -------------------------------------------------------
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    globals().get("__file__", os.path.join(os.getcwd(), "examples", "figures", "_"))))))
for p in (ROOT, os.path.join(ROOT, "examples"), os.path.join(ROOT, "examples", "figures")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import make_figures as M          # noqa: E402  (import-safe: does not set Agg)
from wavetopo import figlib as F  # noqa: E402

# False while experimenting -> writes results/ only, leaves docs/paper/figs alone
PUBLISH = False

print("cwd:", os.getcwd())
print("backend:", plt.get_backend(), "  (should NOT be 'agg' in VS Code)")
print()
for name in M.FIGS:
    st, miss = M.status(name)
    print(f"  {name:12} {st:8} {M.PUBLISHED[name]}")

# %% [markdown]
# ## Style knobs
# Mutate `F.STYLE`, then re-run any figure cell. Nothing else needs reloading.

# %%
F.STYLE.update(
    cmap_field="magma",      # try: inferno, viridis, cividis, turbo
    cmap_curv="viridis",
    director_n=44,           # density of the fiber-director segments
    director_lw=2.0,
    tow_pitch=0.13,          # phase-field tow spacing (smaller = finer tows)
    tow_n=420,
    field_pctl=99.0,         # lower => more saturated fields
    dpi=145,
)
F.STYLE

# %% [markdown]
# ## Manufacturable-paper figures (standalone builders)
# These six live in their own `*_figure.py` modules (not `make_figures`). Each
# `build()` returns a live figure with no backend forced and nothing saved, so you
# restyle and re-run. Edit the module, then `importlib.reload(...)` to pick it up.
# Style for these is `_paperfig.set_style()` (called inside each `build`); the
# shield **re-solves** its harmonic systems (~1 min).

# %%
import importlib   # noqa: E402
import cfrp_bracket_aniso_tradeoff_figure as _trade   # noqa: E402
import cfrp_bracket_hero_figure as _hero              # noqa: E402
import cfrp_bracket_pareto_figure as _pareto          # noqa: E402
import cloak_headtohead_figure as _cloak              # noqa: E402
import cloak_front_figure as _cfront                  # noqa: E402
import cfrp_wave_shield_figure as _shield             # noqa: E402
# after editing a module's build(), reload it: importlib.reload(_trade)

# %% Figure 2 — the static (manufacturability) front
fig = _trade.build(); plt.show()

# %% Figure 3 — the printable static design (hero A-frame)
fig = _hero.build(); plt.show()

# %% Figure 4 — the stiffness–frequency Pareto (the frequency lever)
fig = _pareto.build(); plt.show()

# %% Figure 5 — the cloak in three acts
fig = _cloak.build(); plt.show()

# %% Figure 6 — the cloak's manufacturability front
fig = _cfront.build(); plt.show()

# %% Figure 9 — the vibration shield (re-solves, ~1 min)
fig = _shield.build(); plt.show()

# %% [markdown]
# ## Wave lens

# %%
fig = M.build_lens(PUBLISH)
plt.show()

# %% [markdown]
# ## Curvature-constrained lens
# Rows: unconstrained vs `|zeta| <= zeta_all`. Columns: field, toolpaths, curvature.

# %%
fig = M.build_lens_curl(PUBLISH)
plt.show()

# %% [markdown]
# ## Multi-target beam shaping (two foci + a null)

# %%
fig = M.build_lens_multi(PUBLISH)
plt.show()

# %% [markdown]
# ## Lens with two prescribed through-holes

# %%
fig = M.build_lens_hole(PUBLISH)
plt.show()

# %% [markdown]
# ## Asymmetric through-holes (design re-optimized per geometry)

# %%
fig = M.build_lens_hole_asym(PUBLISH)
plt.show()

# %% [markdown]
# ## Cloak: the void as the prescribed through-hole
# Soft-void variant. The void is masked out of the tows and blanked in the
# curvature map using the saved `solc` solid mask — this file stores **no**
# geometry keys, which is how a generic rebuild once dropped the scatterer
# entirely.

# %%
fig = M.build_cloak_curl(PUBLISH)
plt.show()

# %% [markdown]
# ## Figures with dedicated modules (now LIVE, not PNG)
# The cloak, guide, orient, flatband and sweep figures live in their own modules,
# each with a `build()` that returns a live figure (edit the module + `reload`, or
# tweak the returned `fig`/`ax`). flatband and sweep re-simulate band structures,
# so they take a few seconds to a minute.

# %%
import importlib   # noqa: E402
import dfx_cloak_figure as _dcloak      # noqa: E402
import dfx_guide_figure as _dguide      # noqa: E402
import dfx_orient_figure as _dorient    # noqa: E402
import flatband_figure as _flat         # noqa: E402
import flatband_sweep_figure as _fsweep  # noqa: E402

# %% Conforming elastic cloak
fig = _dcloak.build(); plt.show()

# %% Energy guided around a joint
fig = _dguide.build(); plt.show()

# %% Anisotropy director maps (all designs)
fig = _dorient.build(); plt.show()

# %% Flat-band metamaterial (re-simulates, ~10 s)
fig = _flat.build(); plt.show()

# %% Resolution/pitch sweep (re-simulates 4 levels, ~1 min)
fig = _fsweep.build(); plt.show()

# %% [markdown]
# ---
# ## Custom variants
# The builders are convenience wrappers. To compose your own panel set, use the
# `figlib` primitives directly — everything you need is in the data file.

# %%
d, tri, xl, yl = F.load("results/dolfinx_cloak_conforming_data.npz")
CX, CY = float(d["CX"]), float(d["CY"])
RV, R_DES = float(d["RV"]), float(d["R_DES"])
Z = R_DES + 0.35

F.STYLE["cmap_field"] = "inferno"                   # <- tweak me
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
F.field_panel(ax[0], tri, d["mcl"], xl, yl,
              f"cloaked field ({float(d['red']):.0f}$\\times$ less scatter)")
F.dir_panel(ax[1], d["cent"], d["thopt"], (CX-Z, CX+Z), (CY-Z, CY+Z),
            "director field, zoomed to the design shell",
            holes=[(CX, CY, RV)], n=48)
plt.tight_layout()
plt.show()
F.STYLE["cmap_field"] = "magma"                     # restore

# %% [markdown]
# ## Publish everything
# Runs the batch driver, writing both `results/` and `docs/paper/figs/`.

# %%
import subprocess  # noqa: E402
subprocess.run([sys.executable, "examples/figures/make_figures.py", "--all"],
               check=True, env=dict(os.environ, PYTHONPATH=ROOT))
