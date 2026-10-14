# %% [markdown]
# # Figure playground — every figure in BOTH papers, live
#
# Run these cells in the VS Code **Python Interactive** window (`Shift+Enter`).
# Every cell calls a figure's `build()` and returns a **live** matplotlib figure —
# nothing is a static PNG, so you can modify the display on the fly.
#
# **Two ways to modify a figure:**
# 1. **Restyle globally, then re-run a cell.** Edit the *Global knobs* cell
#    (`F.STYLE`, rcParams, `FIGNOSUP`) and re-run any figure cell.
# 2. **Change the construction.** Edit the figure's `build()` in its module
#    (e.g. `cfrp_bracket_hero_figure.py`), then re-run its cell — each cell
#    `importlib.reload(...)`s the module first, so your edits take effect live.
#
# You can also grab the returned `fig`/`ax` and tweak titles, limits, colours
# inline (see the *Inline tweak demo* at the bottom).
#
# Figures are grouped by paper. Figures shared by both papers are listed once,
# under the **manuscript**, and tagged `[also omnibus]`.
#
# * **Interpreter:** the base conda env (`/home/jrt/miniforge3/bin/python3`);
#   dolfinx is not needed — the drivers already saved every field/design.
# * **Batch publish equivalent:** `python examples/figures/make_figures.py --all`
#   plus the standalone `*_figure.py` CLIs (or the *Publish everything* cell).

# %% Setup — run once
import os
import sys
import importlib

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    globals().get("__file__", os.path.join(os.getcwd(), "examples", "figures", "_"))))))
for p in (ROOT, os.path.join(ROOT, "examples"), os.path.join(ROOT, "examples", "figures")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

# print-mode figures (small, no suptitle) — the variants the manuscript uses.
# Set "" for the large on-screen/omnibus variants.
os.environ["FIGNOSUP"] = "1"
# False = experiment (render only); True = also write results/ and docs/paper/figs/
PUBLISH = False

import make_figures as M                         # noqa: E402  the 6 figlib builders
from wavetopo import figlib as F                 # noqa: E402  style + primitives

# standalone builders (one module each) — imported once, reloaded per cell
import cfrp_bracket_aniso_tradeoff_figure as _trade   # noqa: E402
import cfrp_bracket_hero_figure as _hero              # noqa: E402
import cfrp_bracket_pareto_figure as _pareto          # noqa: E402
import cfrp_bracket_mode2_figure as _mode2            # noqa: E402
import cloak_headtohead_figure as _cloak              # noqa: E402
import cloak_front_figure as _cfront                  # noqa: E402
import cfrp_wave_shield_figure as _shield             # noqa: E402
import dfx_cloak_figure as _dcloak                    # noqa: E402
import dfx_guide_figure as _dguide                    # noqa: E402
import dfx_orient_figure as _dorient                  # noqa: E402
import dfx_panel_figure as _panel                     # noqa: E402
import flatband_figure as _flat                       # noqa: E402
import flatband_sweep_figure as _fsweep               # noqa: E402

print("cwd:", os.getcwd())
print("backend:", plt.get_backend(), " (should NOT be 'agg' in VS Code)")
print("FIGNOSUP:", os.environ.get("FIGNOSUP"), " PUBLISH:", PUBLISH)

# %% [markdown]
# ## Global knobs
# Mutate these, then re-run any figure cell. `F.STYLE` drives the figlib figures
# (lens / cloak-curl); rcParams + `_paperfig.set_style()` drive the standalone ones.

# %%
F.STYLE.update(
    cmap_field="magma",      # inferno, viridis, cividis, turbo
    cmap_curv="viridis",
    director_n=44,           # density of fiber-director segments
    director_lw=2.0,
    tow_pitch=0.13,          # phase-field tow spacing (smaller = finer tows)
    tow_n=420,
    field_pctl=99.0,         # lower => more saturated fields
    dpi=145,
)
# example rcParam override for the standalone figures (applied on next build):
plt.rcParams.update({"axes.titlesize": 10})
F.STYLE

# %% [markdown]
# # ══════════════ MANUSCRIPT (manufacturable_wave_control) ══════════════
# The eleven figures of the working manuscript, in paper order.

# %% [markdown]
# ### Fig 2 — the static (manufacturability) front · bracket

# %%
importlib.reload(_trade); fig = _trade.build(); plt.show()

# %% [markdown]
# ### Fig 3 — the printable static design (hero A-frame)  [also omnibus]

# %%
importlib.reload(_hero); fig = _hero.build(); plt.show()

# %% [markdown]
# ### Fig 4 — the stiffness–frequency Pareto (frequency lever)  [also omnibus]

# %%
importlib.reload(_pareto); fig = _pareto.build(); plt.show()

# %% [markdown]
# ### Fig 5 — the cloak in three acts  [also omnibus]

# %%
importlib.reload(_cloak); fig = _cloak.build(); plt.show()

# %% [markdown]
# ### Fig 6 — the cloak's manufacturability front

# %%
importlib.reload(_cfront); fig = _cfront.build(); plt.show()

# %% [markdown]
# ### Fig 7 — curvature-constrained wave lens  (figlib)  [also omnibus]

# %%
importlib.reload(M); fig = M.build_lens_curl(PUBLISH); plt.show()

# %% [markdown]
# ### Fig 8 — multi-target beam shaping (two foci + a null)  (figlib)  [also omnibus]

# %%
importlib.reload(M); fig = M.build_lens_multi(PUBLISH); plt.show()

# %% [markdown]
# ### Fig 9 — the vibration shield (re-solves harmonic systems, ~1 min)  [also omnibus]

# %%
importlib.reload(_shield); fig = _shield.build(); plt.show()

# %% [markdown]
# ### Fig 10 — energy routed around a joint  [also omnibus]

# %%
importlib.reload(_dguide); fig = _dguide.build(); plt.show()

# %% [markdown]
# ### Fig 11 — flat-band metamaterial (re-simulates bands, ~10 s)  [also omnibus]

# %%
importlib.reload(_flat); fig = _flat.build(); plt.show()

# %% [markdown]
# ### Fig 12 — the multifunctional panel (route / shield / protect / stiffen, band)  [also omnibus]
# Reads results/dolfinx_panel_band_data.npz + dolfinx_panel_sweep.npz. To restyle:
# edit dfx_panel_figure.build() (panel titles, overlays, the bandwidth curve) or
# the region weights/geometry in examples/dolfinx/dolfinx_panel.py then re-run the
# driver. FIGNOSUP toggles the print layout (7.3x6.3, harmonized) vs the large one.

# %%
importlib.reload(_panel); fig = _panel.build(); plt.show()

# %% [markdown]
# # ══════════════ OMNIBUS-ONLY (dolfinx_wave_control) ══════════════
# Figures that appear only in the omnibus paper (the shared ones are above).

# %% [markdown]
# ### Wave lens (baseline curvilinear-fiber lens)  (figlib)

# %%
importlib.reload(M); fig = M.build_lens(PUBLISH); plt.show()

# %% [markdown]
# ### Conforming elastic cloak (dfx_cloak_conf)

# %%
importlib.reload(_dcloak); fig = _dcloak.build(); plt.show()

# %% [markdown]
# ### Anisotropy director maps (all designs)  (dfx_orient)

# %%
importlib.reload(_dorient); fig = _dorient.build(); plt.show()

# %% [markdown]
# ### Lens with two prescribed through-holes  (figlib)

# %%
importlib.reload(M); fig = M.build_lens_hole(PUBLISH); plt.show()

# %% [markdown]
# ### Lens with two ASYMMETRIC through-holes (re-optimized per geometry)  (figlib)

# %%
importlib.reload(M); fig = M.build_lens_hole_asym(PUBLISH); plt.show()

# %% [markdown]
# ### Cloak-curvature study (soft-void variant)  (figlib)

# %%
importlib.reload(M); fig = M.build_cloak_curl(PUBLISH); plt.show()

# %% [markdown]
# ### Resolution / tow-pitch sweep (re-simulates 4 levels, ~1 min)  (flatband_sweep)

# %%
importlib.reload(_fsweep); fig = _fsweep.build(); plt.show()

# %% [markdown]
# ### Mode-separation study (bracket, ω₂/ω₁ → 1)  (cfrp_bracket_mode2)

# %%
importlib.reload(_mode2); fig = _mode2.build(); plt.show()

# %% [markdown]
# ---
# ## Inline tweak demo
# Grab the returned `fig`/`ax` and restyle without touching the module. Example on
# the cloak-front figure: recolour the front curve and retitle, live.

# %%
importlib.reload(_cfront)
fig = _cfront.build()
ax = fig.axes[0]
for ln in ax.get_lines():
    if ln.get_marker() == "o":          # the constrained-front curve
        ln.set_color("#0a7d4f"); ln.set_markerfacecolor("#0a7d4f")
ax.set_xlabel("allowed tow curvature $\\zeta_{\\rm all}$  (my label)")
fig.canvas.draw_idle()
plt.show()

# %% [markdown]
# ## Publish everything (writes results/ and docs/paper/figs/)
# The figlib figures go through the batch driver; the standalone ones through
# their own CLIs. Equivalent to running each `*_figure.py` + `make_figures --all`.

# %%
import subprocess  # noqa: E402
env = dict(os.environ, PYTHONPATH=ROOT)
subprocess.run([sys.executable, "examples/figures/make_figures.py", "--all"], check=True, env=env)
for script, extra in [
    ("cfrp_bracket_aniso_tradeoff_figure.py", []),
    ("cfrp_bracket_hero_figure.py", ["--data", "results/cfrp_bracket_hero_aniso.npz",
                                     "--out", "results/cfrp_bracket_hero.pdf"]),
    ("cfrp_bracket_pareto_figure.py", []),
    ("cfrp_bracket_mode2_figure.py", []),
    ("cloak_headtohead_figure.py", []),
    ("cloak_front_figure.py", []),
    ("cfrp_wave_shield_figure.py", []),
]:
    subprocess.run([sys.executable, f"examples/figures/{script}", *extra], check=True, env=env)
print("published all figures")
