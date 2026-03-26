"""
Unit tests for pytrist.units.UnitConverter.

Tests verify:
  1. Conversion factor formulas against analytic results.
  2. Derived physical quantity formulas.
  3. Array conversion methods (length, time, speed, field_B, field_E).
  4. Self-consistency: round-trip conversions recover the original values.
  5. Edge cases (sigma=1, mass_ratio=1 i.e. pair plasma).

All expected values are computed from first principles using the definitions
in units.py — no look-up tables.
"""

import numpy as np
import pytest

from pytrist.units import UnitConverter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uc_standard():
    """Realistic magnetised ion-electron plasma (σ=0.1, mi/me=100)."""
    return UnitConverter(c_omp=10.0, sigma=0.1, mass_ratio=100.0, CC=0.45)


@pytest.fixture
def uc_pair():
    """Pair plasma: mi = me, so mass_ratio = 1."""
    return UnitConverter(c_omp=8.0, sigma=1.0, mass_ratio=1.0, CC=0.45)


@pytest.fixture
def uc_hi_sigma():
    """Highly magnetised case: σ = 10."""
    return UnitConverter(c_omp=20.0, sigma=10.0, mass_ratio=50.0, CC=0.45)


# ---------------------------------------------------------------------------
# Derived physical quantities
# ---------------------------------------------------------------------------

class TestDerivedQuantities:
    """Verify formulas for di_in_cells, vAi_over_c, Omega_ci_over_wpe."""

    def test_di_in_cells_standard(self, uc_standard):
        # di = c_omp × √(mi/me) = 10 × √100 = 10 × 10 = 100
        expected = 10.0 * np.sqrt(100.0)
        assert uc_standard.di_in_cells == pytest.approx(expected, rel=1e-12)

    def test_di_in_cells_pair(self, uc_pair):
        # di = 8 × √1 = 8
        expected = 8.0 * 1.0
        assert uc_pair.di_in_cells == pytest.approx(expected, rel=1e-12)

    def test_di_in_cells_hi_sigma(self, uc_hi_sigma):
        # di = 20 × √50
        expected = 20.0 * np.sqrt(50.0)
        assert uc_hi_sigma.di_in_cells == pytest.approx(expected, rel=1e-12)

    def test_vAi_over_c_standard(self, uc_standard):
        # vAi/c = √(σ/mi_me) = √(0.1/100) = √(0.001)
        expected = np.sqrt(0.1 / 100.0)
        assert uc_standard.vAi_over_c == pytest.approx(expected, rel=1e-12)

    def test_vAi_over_c_pair(self, uc_pair):
        # vAi/c = √(1/1) = 1  (Alfvén = c in pair plasma with σ=1)
        assert uc_pair.vAi_over_c == pytest.approx(1.0, rel=1e-12)

    def test_Omega_ci_over_wpe_standard(self, uc_standard):
        # Ωci/ωpe = √σ / mi_me = √0.1 / 100
        expected = np.sqrt(0.1) / 100.0
        assert uc_standard.Omega_ci_over_wpe == pytest.approx(expected, rel=1e-12)

    def test_Omega_ci_over_wpe_pair(self, uc_pair):
        # Ωci/ωpe = √1 / 1 = 1
        assert uc_pair.Omega_ci_over_wpe == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Conversion factors
# ---------------------------------------------------------------------------

class TestConversionFactors:
    """Verify the three primary conversion factor properties."""

    def test_cell_to_di_is_reciprocal_of_di_in_cells(self, uc_standard):
        # cell_to_di = 1 / di_in_cells
        assert uc_standard.cell_to_di == pytest.approx(
            1.0 / uc_standard.di_in_cells, rel=1e-12
        )

    def test_cell_to_di_formula(self, uc_standard):
        # 1 / (c_omp × √mi_me)
        expected = 1.0 / (10.0 * np.sqrt(100.0))
        assert uc_standard.cell_to_di == pytest.approx(expected, rel=1e-12)

    def test_wpe_to_wci_formula(self, uc_standard):
        # √σ / mi_me = √0.1 / 100
        expected = np.sqrt(0.1) / 100.0
        assert uc_standard.wpe_to_wci == pytest.approx(expected, rel=1e-12)

    def test_wpe_to_wci_equals_Omega_ci_over_wpe(self, uc_standard):
        # They are the same quantity by definition
        assert uc_standard.wpe_to_wci == pytest.approx(
            uc_standard.Omega_ci_over_wpe, rel=1e-12
        )

    def test_c_to_vAi_formula(self, uc_standard):
        # √(mi_me / σ) = √(100 / 0.1) = √1000
        expected = np.sqrt(100.0 / 0.1)
        assert uc_standard.c_to_vAi == pytest.approx(expected, rel=1e-12)

    def test_c_to_vAi_is_reciprocal_of_vAi_over_c(self, uc_standard):
        # v[vAi] = v[c] / (vAi/c)  →  c_to_vAi = 1 / vAi_over_c
        assert uc_standard.c_to_vAi == pytest.approx(
            1.0 / uc_standard.vAi_over_c, rel=1e-12
        )

    def test_pair_c_to_vAi(self, uc_pair):
        # σ=1, mi_me=1 → vAi=c → c_to_vAi = 1
        assert uc_pair.c_to_vAi == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Self-consistency: vAi × Ωci / ωpe relations
# ---------------------------------------------------------------------------

class TestPhysicalConsistency:
    """Physical cross-checks between derived quantities."""

    def test_vAi_sigma_relation(self, uc_standard):
        """vAi² / c² = σ / mi_me  (definition of σ for uniform B)."""
        uc = uc_standard
        assert uc.vAi_over_c**2 == pytest.approx(uc.sigma / uc.mass_ratio, rel=1e-12)

    def test_Omega_ci_relation(self, uc_standard):
        """Ωci/ωpe = √σ / mi_me  (from ωce = √σ ωpe and Ωci = ωce/mi_me)."""
        uc = uc_standard
        assert uc.Omega_ci_over_wpe == pytest.approx(
            np.sqrt(uc.sigma) / uc.mass_ratio, rel=1e-12
        )

    def test_di_de_relation(self, uc_standard):
        """di = de × √(mi/me)  →  di_in_cells = c_omp × √mi_me."""
        uc = uc_standard
        assert uc.di_in_cells == pytest.approx(uc.c_omp * np.sqrt(uc.mass_ratio), rel=1e-12)

    def test_conversion_factors_all_positive(self, uc_standard, uc_pair, uc_hi_sigma):
        for uc in [uc_standard, uc_pair, uc_hi_sigma]:
            assert uc.cell_to_di > 0
            assert uc.wpe_to_wci > 0
            assert uc.c_to_vAi > 0


# ---------------------------------------------------------------------------
# Array conversion methods
# ---------------------------------------------------------------------------

class TestLengthConversion:
    def test_scalar_conversion(self, uc_standard):
        # 100 cells should equal 1 di (since di_in_cells = 100)
        result = uc_standard.length(100.0)
        assert result == pytest.approx(1.0, rel=1e-12)

    def test_array_conversion(self, uc_standard):
        cells = np.array([0.0, 50.0, 100.0, 200.0])
        expected = cells * uc_standard.cell_to_di
        np.testing.assert_allclose(uc_standard.length(cells), expected, rtol=1e-12)

    def test_scalar_returns_float(self, uc_standard):
        result = uc_standard.length(50.0)
        assert isinstance(result, float)

    def test_array_returns_ndarray(self, uc_standard):
        result = uc_standard.length(np.array([50.0]))
        assert isinstance(result, np.ndarray)

    def test_zero_maps_to_zero(self, uc_standard):
        assert uc_standard.length(0.0) == pytest.approx(0.0)

    def test_list_input(self, uc_standard):
        result = uc_standard.length([10, 20, 30])
        expected = np.array([10, 20, 30]) * uc_standard.cell_to_di
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestTimeConversion:
    def test_formula(self, uc_standard):
        # t[1/ωpe] = 1 → t[1/Ωci] = wpe_to_wci
        result = uc_standard.time(1.0)
        assert result == pytest.approx(uc_standard.wpe_to_wci, rel=1e-12)

    def test_array_input(self, uc_standard):
        t_code = np.linspace(0, 1000, 50)
        expected = t_code * uc_standard.wpe_to_wci
        np.testing.assert_allclose(uc_standard.time(t_code), expected, rtol=1e-12)

    def test_scalar_returns_float(self, uc_standard):
        assert isinstance(uc_standard.time(10.0), float)

    def test_array_returns_ndarray(self, uc_standard):
        assert isinstance(uc_standard.time(np.array([10.0])), np.ndarray)

    def test_pair_plasma(self, uc_pair):
        # σ=1, mi_me=1 → wpe_to_wci = 1 → t[Ωci] = t[ωpe]
        t = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(uc_pair.time(t), t, rtol=1e-12)


class TestSpeedConversion:
    def test_formula(self, uc_standard):
        # v[c] = 1 → v[vAi] = c_to_vAi
        result = uc_standard.speed(1.0)
        assert result == pytest.approx(uc_standard.c_to_vAi, rel=1e-12)

    def test_Alfven_speed_in_ion_units(self, uc_standard):
        # vAi/c in code → should give exactly 1.0 in ion units
        vAi_in_c = uc_standard.vAi_over_c
        result = uc_standard.speed(vAi_in_c)
        assert float(result) == pytest.approx(1.0, rel=1e-12)

    def test_array_conversion(self, uc_standard):
        v = np.array([0.0, 0.1, 0.5, 1.0])
        expected = v * uc_standard.c_to_vAi
        np.testing.assert_allclose(uc_standard.speed(v), expected, rtol=1e-12)

    def test_pair_plasma(self, uc_pair):
        # vAi = c → c_to_vAi = 1 → no change
        v = np.array([0.3, 0.6, 0.9])
        np.testing.assert_allclose(uc_pair.speed(v), v, rtol=1e-12)


class TestFieldConversion:
    def test_field_B_returns_float_array(self, uc_standard):
        arr = np.array([1.0, 2.0, 3.0])
        result = uc_standard.field_B(arr)
        assert result.dtype == np.float64

    def test_field_E_returns_float_array(self, uc_standard):
        arr = np.array([0.5, -0.5])
        result = uc_standard.field_E(arr)
        assert result.dtype == np.float64

    def test_field_B_normalises_by_B0(self, uc_standard):
        """field_B divides by B0 so that the upstream field ≈ 1."""
        B0 = uc_standard.B0
        arr = np.array([0.0, B0, 2 * B0, -B0])
        expected = np.array([0.0, 1.0, 2.0, -1.0])
        np.testing.assert_allclose(uc_standard.field_B(arr), expected, rtol=1e-12)

    def test_field_E_normalises_by_E0(self, uc_standard):
        """field_E divides by E0 = B0 × vAi_over_c (natural ion E unit)."""
        E0 = uc_standard.B0 * uc_standard.vAi_over_c
        arr = np.array([0.0, E0, -E0])
        expected = np.array([0.0, 1.0, -1.0])
        np.testing.assert_allclose(uc_standard.field_E(arr), expected, rtol=1e-12)

    def test_B0_formula(self, uc_standard):
        """B0 = CC^2 * sqrt(sigma) / c_omp (Gaussian PIC convention)."""
        expected = 0.45 ** 2 * np.sqrt(0.1) / 10.0
        assert uc_standard.B0 == pytest.approx(expected, rel=1e-12)

    def test_field_B_scalar_input(self, uc_standard):
        result = uc_standard.field_B(uc_standard.B0)
        assert float(result) == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Converting to ion units and back should recover original values."""

    def test_length_round_trip(self, uc_standard):
        x_cells = np.array([1.0, 10.0, 100.0, 500.0])
        x_di = uc_standard.length(x_cells)
        x_cells_back = x_di / uc_standard.cell_to_di
        np.testing.assert_allclose(x_cells_back, x_cells, rtol=1e-12)

    def test_time_round_trip(self, uc_standard):
        t_wpe = np.array([0.0, 1.0, 100.0, 1e4])
        t_wci = uc_standard.time(t_wpe)
        t_wpe_back = t_wci / uc_standard.wpe_to_wci
        np.testing.assert_allclose(t_wpe_back, t_wpe, rtol=1e-12)

    def test_speed_round_trip(self, uc_standard):
        v_c = np.array([0.0, 0.1, 0.5, 1.0])
        v_vAi = uc_standard.speed(v_c)
        v_c_back = v_vAi / uc_standard.c_to_vAi
        np.testing.assert_allclose(v_c_back, v_c, rtol=1e-12)


# ---------------------------------------------------------------------------
# Parameter storage
# ---------------------------------------------------------------------------

class TestParameterStorage:
    def test_stored_parameters(self, uc_standard):
        assert uc_standard.c_omp == 10.0
        assert uc_standard.sigma == 0.1
        assert uc_standard.mass_ratio == 100.0
        assert uc_standard.CC == 0.45

    def test_repr_contains_params(self, uc_standard):
        r = repr(uc_standard)
        assert "c_omp=10.0" in r
        assert "sigma=0.1" in r

    def test_summary_is_string(self, uc_standard):
        s = uc_standard.summary()
        assert isinstance(s, str)
        assert "cell_to_di" in s
        assert "wpe_to_wci" in s
        assert "c_to_vAi" in s


# ---------------------------------------------------------------------------
# Numerical sanity checks with explicit hand-computed values
# ---------------------------------------------------------------------------

class TestNumericalValues:
    """Hard-coded expected values computed by hand for specific parameters."""

    def test_cell_to_di_value(self):
        # c_omp=10, mi_me=100 → di=100 cells → cell_to_di = 0.01
        uc = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
        assert uc.cell_to_di == pytest.approx(0.01, rel=1e-12)

    def test_wpe_to_wci_value(self):
        # σ=0.1, mi_me=100 → wpe_to_wci = √0.1 / 100 ≈ 3.162e-3
        uc = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
        expected = np.sqrt(0.1) / 100.0
        assert uc.wpe_to_wci == pytest.approx(expected, rel=1e-12)

    def test_c_to_vAi_value(self):
        # σ=0.1, mi_me=100 → c_to_vAi = √(100/0.1) = √1000 ≈ 31.62
        uc = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
        expected = np.sqrt(100.0 / 0.1)
        assert uc.c_to_vAi == pytest.approx(expected, rel=1e-12)

    def test_length_100cells_is_1di(self):
        # di = 100 cells → 100 cells = 1 di
        uc = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
        assert uc.length(100.0) == pytest.approx(1.0, rel=1e-12)

    def test_time_1wpe_in_wci_units(self):
        # 1/ωpe → Ωci: multiply by √σ/mi_me = √0.1/100
        uc = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
        expected = np.sqrt(0.1) / 100.0
        assert uc.time(1.0) == pytest.approx(expected, rel=1e-12)

    def test_speed_vAi_is_1(self):
        # Speed equal to vAi should map to exactly 1.0 in ion units
        uc = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
        vAi_in_c = np.sqrt(0.1 / 100.0)
        assert uc.speed(vAi_in_c) == pytest.approx(1.0, rel=1e-12)
