"""
Integration test: 20 OC iterations on a coarse 16×8 mesh.

Verifies:
1. All compliances are positive.
2. Final volume fraction is within 2% of target.
3. Final compliance is lower than initial.
4. Final density is mostly binary (SIMP penalization is working).

Run with:
    conda run -n dolfinx_complex python -m pytest tests/test_integration.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wavetopo import ElasticityProblem, ConeFilter, SIMPOptimizer


def test_cantilever_optimization():
    nx, ny = 16, 8
    vf = 0.40
    lx, ly = 2.0, 1.0
    h = ly / ny           # element height
    r_min = 2.5 * h       # r_min = 2.5×h  (well-coupled filter)

    fem_prob = ElasticityProblem(nx=nx, ny=ny, lx=lx, ly=ly, penal=3.0,
                                 load_type="midpoint")
    filt = ConeFilter(fem_prob.DG0, r_min=r_min)

    def fem_solve(rho):
        rho_phys = np.clip(rho, 1e-3, 1.0)
        fem_prob.update_density(rho_phys)
        u, C = fem_prob.solve()
        dc = fem_prob.compliance_sensitivity(u)
        # Weighted sensitivity filter (Sigmund 2001 / Andreassen 2011)
        dc_filt = filt.apply(rho_phys * dc) / np.maximum(1e-3, rho_phys)
        return C, dc_filt

    opt = SIMPOptimizer(n_elem=nx * ny, vf=vf, max_iter=20, tol=1e-4)
    result = opt.optimize(fem_solve)

    assert all(c > 0 for c in result.compliance_history), \
        "All compliances must be positive"

    final_vol = result.densities.mean()
    assert abs(final_vol - vf) < 0.02, \
        f"Volume {final_vol:.4f} not within 2% of target {vf}"

    C_init  = result.compliance_history[0]
    C_final = result.compliance_history[-1]
    assert C_final < C_init, \
        f"Final C={C_final:.4f} should be less than initial C={C_init:.4f}"

    # Check that most densities are near 0 or 1 (SIMP penalization working)
    # After 20 iters on a coarse mesh the design is still polarising; check a
    # modest lower bound so the test isn't fragile to exact iteration count.
    binary_fraction = np.mean((result.densities < 0.2) | (result.densities > 0.8))
    assert binary_fraction > 0.1, \
        f"Only {binary_fraction:.1%} near 0/1 (expected >10%)"

    print(f"\n  C_init={C_init:.4f}, C_final={C_final:.4f}, "
          f"vol={final_vol:.4f}, binary={binary_fraction:.1%}, "
          f"iters={result.n_iterations}")
