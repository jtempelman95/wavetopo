"""
topoopt — topology + fiber-orientation optimization of continuous-fiber
composites, for quasi-static design and elastic wave control.

Pure numpy/scipy core (no FeniCSx required):

  Modeling
    cfrp            Material, QuadMesh, FourierStiffness, CSRBFMapping,
                    WaveProjection, element_stiffness/mass
    scalar          antiplane-shear (scalar) analogue: MaterialSH, ScalarBloch

  Quasi-static CFRP (Wong et al. 2026 replication)
    cfrp_problem    CFRPProblem (compliance, volume, curl constraints)
    cfrp_optimizer  MMA + augmented-Lagrangian driver
    cfrp_viz        density / streamline / toolpath / curl figures

  Wave control (time-harmonic)
    harmonic        HarmonicLens (focus), HarmonicCloak (void cloak)
    wave_control    WaveFocus — single- & multi-frequency energy localization

  Periodic metamaterials (Bloch)
    bloch           BlochProblem, FHS berry_curvature, eigen sensitivities
    bandgap_opt     directional band-gap optimizer
    valley_opt      valley-gap / complete-gap co-optimizer
    rhombic         primitive rhombic honeycomb cell
"""

__version__ = "0.2.0"

# --- pure numpy/scipy wave-control + CFRP API (always available) --------- #
from .cfrp import (Material, QuadMesh, FourierStiffness, CSRBFMapping,
                   WaveProjection, grid_support_points,
                   element_stiffness, element_mass)
from .cfrp_problem import CFRPProblem, density_filter
from .cfrp_optimizer import MMA, optimize_cfrp
from .harmonic import HarmonicLens, HarmonicCloak
from .wave_control import WaveFocus, ramp_sponge

# --- dolfinx-based legacy modules (optional) ----------------------------- #
try:  # pragma: no cover - environment dependent
    from .fem import ElasticityProblem
    from .filters import ConeFilter, HelmholtzFilter, ProjectionFilter
    from .optimizer import SIMPOptimizer
    from .visualize import (OptimizationRecorder, density_snapshot,
                            save_convergence_plot)
    _DOLFINX_OK = True
except Exception:  # noqa: BLE001 - dolfinx not installed in this env
    _DOLFINX_OK = False

__all__ = [
    # modeling
    "Material", "QuadMesh", "FourierStiffness", "CSRBFMapping",
    "WaveProjection", "grid_support_points", "element_stiffness", "element_mass",
    # quasi-static CFRP
    "CFRPProblem", "density_filter", "MMA", "optimize_cfrp",
    # wave control
    "HarmonicLens", "HarmonicCloak", "WaveFocus", "ramp_sponge",
]
