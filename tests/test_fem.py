"""
Unit tests for the FE elasticity problem.

Run with:
    conda run -n dolfinx_complex python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from topoopt import ElasticityProblem, HelmholtzFilter


class TestElasticityProblem:
    """Basic sanity checks on the FE problem."""

    @pytest.fixture(scope="class")
    def fem_small(self):
        """Tiny 4×2 mesh for fast tests."""
        return ElasticityProblem(nx=4, ny=2, lx=2.0, ly=1.0)

    def test_mesh_cell_count(self, fem_small):
        n_cells = fem_small.domain.topology.index_map(
            fem_small.domain.topology.dim
        ).size_local
        assert n_cells == 4 * 2, f"Expected 8 cells, got {n_cells}"

    def test_solve_returns_positive_compliance(self, fem_small):
        u, C = fem_small.solve()
        assert C > 0, "Compliance must be positive"

    def test_solid_stiffer_than_void(self):
        """Full solid should be stiffer (lower compliance) than near-void."""
        fem_solid = ElasticityProblem(nx=8, ny=4)
        fem_void = ElasticityProblem(nx=8, ny=4, E_min=1e-9)

        fem_solid.update_density(np.ones(8 * 4))
        _, C_solid = fem_solid.solve()

        fem_void.update_density(np.full(8 * 4, 0.01))
        _, C_void = fem_void.solve()

        assert C_solid < C_void, (
            f"Solid compliance {C_solid:.4f} should be less than "
            f"near-void compliance {C_void:.4f}"
        )

    def test_sensitivity_negative(self, fem_small):
        """dC/drho should be ≤ 0 everywhere (adding material reduces compliance)."""
        fem_small.update_density(np.full(8, 0.5))
        u, _ = fem_small.solve()
        dc = fem_small.compliance_sensitivity(u)
        assert np.all(dc <= 0), "Sensitivity dC/drho must be non-positive"

    def test_update_density_shapes(self, fem_small):
        n = fem_small.domain.topology.index_map(
            fem_small.domain.topology.dim
        ).size_local
        rho = np.linspace(0.1, 1.0, n)
        fem_small.update_density(rho)
        np.testing.assert_allclose(fem_small.rho.x.array, rho)


class TestHelmholtzFilter:
    """Tests for the PDE-based density filter."""

    @pytest.fixture(scope="class")
    def fem_and_filter(self):
        # r_min=0.5 > element_size≈0.25 so the Helmholtz filter couples neighbours
        fem_prob = ElasticityProblem(nx=8, ny=4, lx=2.0, ly=1.0)
        filt = HelmholtzFilter(fem_prob.DG0, r_min=0.5)
        return fem_prob, filt

    def test_uniform_density_preserved(self, fem_and_filter):
        """Filtering a uniform field should return the same field."""
        _, filt = fem_and_filter
        rho = np.ones(8 * 4) * 0.6
        rho_tilde = filt.apply(rho)
        np.testing.assert_allclose(rho_tilde, rho, atol=1e-6,
                                   err_msg="Uniform density should be filter-invariant")

    def test_filter_smooths_checkerboard(self, fem_and_filter):
        """Checkerboard density should have lower variance after filtering."""
        _, filt = fem_and_filter
        n = 8 * 4
        rho = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)])
        rho_tilde = filt.apply(rho)
        assert rho_tilde.std() < rho.std(), (
            "Filter should reduce variance of checkerboard pattern"
        )

    def test_filtered_values_bounded(self, fem_and_filter):
        """Filtered densities should stay in [0, 1] for input in [0, 1]."""
        _, filt = fem_and_filter
        rng = np.random.default_rng(42)
        rho = rng.uniform(0, 1, 8 * 4)
        rho_tilde = filt.apply(rho)
        assert rho_tilde.min() >= -1e-10
        assert rho_tilde.max() <= 1.0 + 1e-10

    def test_sensitivity_chain_shape(self, fem_and_filter):
        _, filt = fem_and_filter
        rho = np.full(8 * 4, 0.5)
        dc_tilde = np.ones(8 * 4) * -1.0
        dc = filt.sensitivity_chain(dc_tilde, rho)
        assert dc.shape == rho.shape


class TestOCUpdate:
    """Tests for the Optimality Criteria optimizer update rule."""

    def test_volume_constraint_satisfied(self):
        from topoopt import SIMPOptimizer
        vf = 0.4
        n = 100
        opt = SIMPOptimizer(n_elem=n, vf=vf, max_iter=5)

        call_count = [0]

        def fem_solve(rho):
            call_count[0] += 1
            # Fake: all sensitivities equal → uniform update
            return 1.0, -np.ones(n) * 0.1

        result = opt.optimize(fem_solve)
        final_vol = result.densities.mean()
        assert abs(final_vol - vf) < 0.02, (
            f"Volume fraction {final_vol:.4f} should be close to {vf}"
        )

    def test_optimizer_reduces_compliance(self):
        """Compliance should generally decrease over iterations (not guaranteed
        monotonically with OC but should decrease overall)."""
        from topoopt import SIMPOptimizer
        opt = SIMPOptimizer(n_elem=50, vf=0.5, max_iter=20)

        rng = np.random.default_rng(0)

        def fem_solve(rho):
            # Fake parabolic landscape: C = sum((rho - 1)^2)
            C = float(np.sum((rho - 1.0) ** 2))
            dc = 2 * (rho - 1.0)
            return C, dc

        result = opt.optimize(fem_solve)
        assert result.compliance_history[-1] <= result.compliance_history[0], (
            "Compliance should not increase over optimization"
        )
