"""
topoopt — Topology Optimization with FEniCSx
"""

from .fem import ElasticityProblem
from .filters import HelmholtzFilter
from .optimizer import SIMPOptimizer

__version__ = "0.1.0"
__all__ = ["ElasticityProblem", "HelmholtzFilter", "SIMPOptimizer"]
