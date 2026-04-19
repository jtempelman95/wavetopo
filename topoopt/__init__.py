"""
topoopt — Topology Optimization with FEniCSx
"""

from .fem import ElasticityProblem
from .filters import HelmholtzFilter
from .optimizer import SIMPOptimizer
from .visualize import OptimizationRecorder, density_snapshot, save_convergence_plot

__version__ = "0.1.0"
__all__ = [
    "ElasticityProblem",
    "HelmholtzFilter",
    "SIMPOptimizer",
    "OptimizationRecorder",
    "density_snapshot",
    "save_convergence_plot",
]
