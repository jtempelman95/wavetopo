"""
Integration test: run a short optimization on a coarse mesh and verify
that the result is physically sensible.

Run with:
    conda run -n dolfinx_complex python -m pytest tests/test_integration.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from topoopt import ElasticityProblem, HelmholtzFilter, SIMPOptimizer


def test_cantilever_optimization():
    """
    Run 15 OC iterations on a 16×8 mesh (fast) and check:
    1. Compliance is positive throughout.
    2. Final volume fraction is within 2% of target.
    3. Compliance at the end is lower than at the start.
    """
    nx, ny = 16, 8
    vf = 0.40

    fem_prob = ElasticityProblem(nx=nx, ny=ny, lx=2.0, ly=1.0, penal=3.0)
    filt = HelmholtzFilter(fem_prob.DG0, r_min=0.1)

    def fem_solve(rho):
        rho_tilde = filt.apply(rho)
        np.clip(rho_tilde, 1e-3, 1.0, out=rho_tilde)
        fem_prob.update_density(rho_tilde)
        u, C = fem_prob.solve()
        dc_tilde = fem_prob.compliance_sensitivity(u)
        dc = filt.sensitivity_chain(dc_tilde, rho)
        return C, dc

    opt = SIMPOptimizer(n_elem=nx * ny, vf=vf, max_iter=15, tol=1e-4)
    result = opt.optimize(fem_solve)

    # Check all compliances are positive
    assert all(c > 0 for c in result.compliance_history), \
        "All compliance values must be positive"

    # Check volume constraint
    final_vol = result.densities.mean()
    assert abs(final_vol - vf) < 0.02, \
        f"Volume fraction {final_vol:.4f} not within 2% of target {vf}"

    # Check compliance decreases
    C_init = result.compliance_history[0]
    C_final = result.compliance_history[-1]
    assert C_final < C_init, \
        f"Final compliance {C_final:.4f} should be less than initial {C_init:.4f}"

    print(f"\n  C_init={C_init:.4f}, C_final={C_final:.4f}, "
          f"vol={final_vol:.4f}, iters={result.n_iterations}")
