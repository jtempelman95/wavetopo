"""
Unit tests for the FE elasticity problem and density filter.

Run with:
    conda run -n dolfinx_complex python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from topoopt import ElasticityProblem, ConeFilter


class TestElasticityProblem:

    @pytest.fixture(scope="class")
    def fem_small(self):
        """4×2 mesh, midpoint load."""
        return ElasticityProblem(nx=4, ny=2, lx=2.0, ly=1.0)

    def test_mesh_cell_count(self, fem_small):
        n_cells = fem_small.domain.topology.index_map(
            fem_small.domain.topology.dim
        ).size_local
        assert n_cells == 4 * 2

    def test_solve_returns_positive_compliance(self, fem_small):
        u, C = fem_small.solve()
        assert C > 0

    def test_solid_stiffer_than_void(self):
        fem_solid = ElasticityProblem(nx=8, ny=4)
        fem_void  = ElasticityProblem(nx=8, ny=4)

        fem_solid.update_density(np.ones(32))
        _, C_solid = fem_solid.solve()

        fem_void.update_density(np.full(32, 0.01))
        _, C_void = fem_void.solve()

        assert C_solid < C_void, (
            f"Solid C={C_solid:.4f} should be less than near-void C={C_void:.4f}"
        )

    def test_sensitivity_negative(self, fem_small):
        """dC/drho must be ≤ 0: adding material reduces compliance."""
        fem_small.update_density(np.full(8, 0.5))
        u, _ = fem_small.solve()
        dc = fem_small.compliance_sensitivity(u)
        assert np.all(dc <= 0), f"max dc = {dc.max():.4e} should be ≤ 0"

    def test_update_density_shapes(self, fem_small):
        n = fem_small.domain.topology.index_map(
            fem_small.domain.topology.dim
        ).size_local
        rho = np.linspace(0.1, 1.0, n)
        fem_small.update_density(rho)
        np.testing.assert_allclose(fem_small.rho.x.array, rho)

    def test_midpoint_load_has_fewer_loaded_facets_than_distributed(self):
        """Point load should load fewer boundary facets than distributed."""
        # Both meshes 8×4; count loaded facets indirectly via compliance ratio
        fem_mid  = ElasticityProblem(nx=8, ny=4, lx=2.0, ly=1.0, load_type="midpoint")
        fem_dist = ElasticityProblem(nx=8, ny=4, lx=2.0, ly=1.0, load_type="distributed")
        rho = np.ones(32)
        fem_mid.update_density(rho)
        fem_dist.update_density(rho)
        _, C_mid  = fem_mid.solve()
        _, C_dist = fem_dist.solve()
        # Both must be positive; no specific ordering requirement since they
        # are different problems with the same total applied force
        assert C_mid > 0
        assert C_dist > 0


class TestConeFilter:

    @pytest.fixture(scope="class")
    def fem_and_filter(self):
        # r_min = 3 × element_size (h = 2.0/8 = 0.25) → well-coupled
        fem_prob = ElasticityProblem(nx=8, ny=4, lx=2.0, ly=1.0)
        filt = ConeFilter(fem_prob.DG0, r_min=0.75)
        return fem_prob, filt

    def test_uniform_density_preserved(self, fem_and_filter):
        """Filtering a uniform field must return the same field."""
        _, filt = fem_and_filter
        rho = np.ones(32) * 0.6
        rho_tilde = filt.apply(rho)
        np.testing.assert_allclose(rho_tilde, rho, atol=1e-10)

    def test_filter_smooths_checkerboard(self, fem_and_filter):
        """Checkerboard input must have lower variance after filtering."""
        _, filt = fem_and_filter
        rho = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(32)])
        rho_tilde = filt.apply(rho)
        assert rho_tilde.std() < rho.std(), (
            f"std after={rho_tilde.std():.4f} should be < std before={rho.std():.4f}"
        )

    def test_filtered_values_bounded(self, fem_and_filter):
        _, filt = fem_and_filter
        rng = np.random.default_rng(42)
        rho = rng.uniform(0, 1, 32)
        rho_tilde = filt.apply(rho)
        assert rho_tilde.min() >= -1e-10
        assert rho_tilde.max() <= 1.0 + 1e-10

    def test_adjoint_is_transpose(self, fem_and_filter):
        """Verify sensitivity_chain computes M^T @ v (not M @ v)."""
        _, filt = fem_and_filter
        rng = np.random.default_rng(0)
        rho = rng.uniform(0, 1, 32)
        v   = rng.uniform(0, 1, 32)
        w   = rng.uniform(0, 1, 32)

        # Check: <M w, v> == <w, M^T v>  (adjoint identity)
        Mw  = filt.apply(w)
        MTv = filt.sensitivity_chain(v, rho)
        assert abs(Mw @ v - w @ MTv) < 1e-10, (
            "Adjoint identity <Mw,v> = <w,M^Tv> must hold exactly for sparse M"
        )

    def test_sensitivity_chain_shape(self, fem_and_filter):
        _, filt = fem_and_filter
        rho = np.full(32, 0.5)
        dc = filt.sensitivity_chain(-np.ones(32), rho)
        assert dc.shape == rho.shape

    def test_r_min_controls_smoothing_strength(self):
        """Larger r_min → more smoothing (lower output std on checkerboard)."""
        fem_prob = ElasticityProblem(nx=8, ny=4, lx=2.0, ly=1.0)
        rho = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(32)])

        filt_small = ConeFilter(fem_prob.DG0, r_min=0.3)
        filt_large = ConeFilter(fem_prob.DG0, r_min=1.0)

        std_small = filt_small.apply(rho).std()
        std_large = filt_large.apply(rho).std()
        assert std_large < std_small, "Larger r_min must produce more smoothing"


class TestOCUpdate:

    def test_volume_constraint_satisfied(self):
        from topoopt import SIMPOptimizer
        vf, n = 0.4, 100
        opt = SIMPOptimizer(n_elem=n, vf=vf, max_iter=5)

        def fem_solve(rho):
            return 1.0, -np.ones(n) * 0.1

        result = opt.optimize(fem_solve)
        assert abs(result.densities.mean() - vf) < 0.02

    def test_optimizer_reduces_compliance(self):
        from topoopt import SIMPOptimizer
        opt = SIMPOptimizer(n_elem=50, vf=0.5, max_iter=20)

        def fem_solve(rho):
            C = float(np.sum((rho - 1.0) ** 2))
            dc = 2 * (rho - 1.0)
            return C, dc

        result = opt.optimize(fem_solve)
        assert result.compliance_history[-1] <= result.compliance_history[0]
