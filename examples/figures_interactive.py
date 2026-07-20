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
# * **Batch equivalent:** `python examples/make_figures.py --all`
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(
    globals().get("__file__", os.path.join(os.getcwd(), "examples", "_")))))
for p in (ROOT, os.path.join(ROOT, "examples")):
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
# ## Figures that delegate to their own scripts
# The cloak, guide, orient, flatband and sweep figures have dedicated builders
# (the last two genuinely re-simulate band structures, so they take minutes).
# They write files rather than returning a live figure, so we display the PNG.

# %%
from IPython.display import Image, display   # noqa: E402

M.FIGS["cloak"][1](PUBLISH)
display(Image("results/dolfinx_cloak_conforming_vec.png"))

# %%
M.FIGS["guide"][1](PUBLISH)
display(Image("results/dolfinx_guide_joint.png"))

# %%
M.FIGS["orient"][1](PUBLISH)
display(Image("results/dfx_orient.png"))

# %%
M.FIGS["flatband"][1](PUBLISH)          # minutes: re-simulates
display(Image("results/flatband_demo.png"))

# %%
M.FIGS["sweep"][1](PUBLISH)             # minutes: re-simulates 4 levels
display(Image("results/flatband_sweep.png"))

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
subprocess.run([sys.executable, "examples/make_figures.py", "--all"],
               check=True, env=dict(os.environ, PYTHONPATH=ROOT))
