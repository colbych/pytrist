"""
Unit conversion utilities for Tristan-MP V2 simulations.

Tristan-V2 uses an internal (code) unit system:
  - Lengths:  grid cells
  - Times:    1/ωpe  (inverse electron plasma frequency)
  - Speeds:   c      (speed of light)
  - B fields: normalised so  σ = B0² c_omp² / (CC² × 4π n0 me c²)

The "ion units" we convert to are based on the *y*-component of the
background magnetic field (By), which defines natural scales for
magnetised ion physics:

  Spatial scale  — ion inertial length
      di = de × √(mi/me)  =  c_omp × √mass_ratio   [cells]

  Time scale — inverse ion cyclotron frequency (from By)
      Ωci_y = ωce_y / mass_ratio  =  √σ × ωpe / mass_ratio
      → period  Tci = mass_ratio / (√σ ωpe)

  Speed scale — ion Alfvén speed (from By)
      vAi = c × √(σ / mass_ratio)

The conversion *factors* (multiply a code-unit quantity to get ion units):

      cell_to_di  = 1 / (c_omp × √mass_ratio)
      wpe_to_wci  = √σ / mass_ratio
      c_to_vAi    = √(mass_ratio / σ)   [so v[vAi] = v[c] / (vAi/c)]

For electromagnetic fields:
  - B fields are normalised to B0; the upstream guide field equals 1 in ion units.
  - E fields are normalised to E0 = B0 × vAi_over_c, the natural ion electric
    field unit set by ideal MHD (E ~ v/c × B ~ vAi/c × B0).
    With this choice, (E×B)/B0² = S_ion directly, since
    (E/E0) × (B/B0) = (E×B) / (E0 × B0) = (E×B) / (B0² × vAi_over_c).
"""

from __future__ import annotations

import numpy as np


class UnitConverter:
    """Convert arrays between Tristan-V2 code units and ion units.

    Parameters
    ----------
    c_omp : float
        Electron skin depth / inertial length in grid cells (de).
    sigma : float
        Magnetisation parameter  σ = ωce²/ωpe²  = B0² c_omp² / CC².
    mass_ratio : float
        Ion-to-electron mass ratio  mi/me.
    CC : float
        Speed of light in code units (typically ~0.45).
    """

    def __init__(
        self,
        c_omp: float,
        sigma: float,
        mass_ratio: float,
        CC: float,
    ) -> None:
        self._c_omp = float(c_omp)
        self._sigma = float(sigma)
        self._mass_ratio = float(mass_ratio)
        self._CC = float(CC)

    # ------------------------------------------------------------------
    # Raw parameters
    # ------------------------------------------------------------------

    @property
    def c_omp(self) -> float:
        """Electron skin depth in grid cells."""
        return self._c_omp

    @property
    def sigma(self) -> float:
        """Magnetisation parameter σ = ωce²/ωpe²."""
        return self._sigma

    @property
    def mass_ratio(self) -> float:
        """Ion-to-electron mass ratio mi/me."""
        return self._mass_ratio

    @property
    def CC(self) -> float:
        """Speed of light in code units."""
        return self._CC

    # ------------------------------------------------------------------
    # Derived physical quantities
    # ------------------------------------------------------------------

    @property
    def di_in_cells(self) -> float:
        """Ion inertial length in grid cells: di = de × √(mi/me)."""
        return self._c_omp * np.sqrt(self._mass_ratio)

    @property
    def vAi_over_c(self) -> float:
        """Ion Alfvén speed as a fraction of c: vAi/c = √(σ/mi_me)."""
        return np.sqrt(self._sigma / self._mass_ratio)

    @property
    def Omega_ci_over_wpe(self) -> float:
        """Ion cyclotron frequency in units of ωpe: Ωci/ωpe = √σ/mi_me."""
        return np.sqrt(self._sigma) / self._mass_ratio

    # ------------------------------------------------------------------
    # Conversion factors
    # ------------------------------------------------------------------

    @property
    def cell_to_di(self) -> float:
        """Factor: cells → di.

        x[di] = x[cells] × cell_to_di

        Derivation:
            di = c_omp × √(mi/me) cells
            → 1 cell = 1 / (c_omp √(mi/me)) di
        """
        return 1.0 / (self._c_omp * np.sqrt(self._mass_ratio))

    @property
    def wpe_to_wci(self) -> float:
        """Factor: 1/ωpe → 1/Ωci_y (i.e., t[Ωci_y] = t[ωpe] × wpe_to_wci).

        Derivation:
            Ωci_y = ωce_y / (mi/me)  = (√σ ωpe) / (mi/me)
            t[Ωci_y] = t[1/ωpe] × (Ωci_y / ωpe) = t[1/ωpe] × √σ / (mi/me)
        """
        return np.sqrt(self._sigma) / self._mass_ratio

    @property
    def c_to_vAi(self) -> float:
        """Factor: c → vAi  (i.e., v[vAi] = v[c] × c_to_vAi).

        Derivation:
            vAi = c × √(σ / (mi/me))
            v[vAi] = v[c] / (vAi/c) = v[c] / √(σ/(mi/me))
                   = v[c] × √((mi/me)/σ)
        """
        return np.sqrt(self._mass_ratio / self._sigma)

    @property
    def B0(self) -> float:
        """Background magnetic field magnitude in Tristan code units.

        Tristan-V2 uses Gaussian-unit Maxwell equations with an explicit
        speed of light (``∂B/∂t = -CC ∇×E``), so the cyclotron frequency
        in code units is ``ωce = B_code / CC`` (not ``B_code``).  Solving
        ``σ = ωce² / ωpe² = (B0/CC)² / (CC/c_omp)²``:

            B0 = CC² × √σ / c_omp
        """
        return self._CC ** 2 * np.sqrt(self._sigma) / self._c_omp

    @property
    def E0(self) -> float:
        """Natural ion electric field unit in Tristan code units.

        In ideal MHD the electric field scales as E ~ (v/c) × B ~ (vAi/c) × B0,
        so the natural ion unit is::

            E0 = B0 × vAi_over_c = B0 × √(σ / mass_ratio)

        With this normalisation, ``(field_E × field_B) = S_ion`` for the
        ExB Poynting flux without any additional factor.
        """
        return self.B0 * self.vAi_over_c

    @property
    def step_to_wci(self) -> float:
        """Factor: code step → 1/Ωci_y.

        Tristan-V2 history and params files store time as an integer step
        counter.  The physical time in 1/ωpe is  t = step × CC/c_omp
        (since ωpe = CC/c_omp in code units).  Converting further to
        1/Ωci_y gives the overall factor:

            step[1/Ωci_y] = step × (CC/c_omp) × (√σ/mass_ratio)
                           = step × CC/c_omp × wpe_to_wci
        """
        return (self._CC / self._c_omp) * self.wpe_to_wci

    # ------------------------------------------------------------------
    # Array conversion methods
    # ------------------------------------------------------------------

    def length(self, arr):
        """Convert length from cells to ion inertial lengths (di).

        Parameters
        ----------
        arr : float or array-like
            Position data in grid cells.

        Returns
        -------
        float or numpy.ndarray
            Position data in di.  Returns a scalar float when *arr* is a
            scalar; returns an ndarray when *arr* is array-like.
        """
        a = np.asarray(arr, dtype=float)
        result = a * self.cell_to_di
        return float(result) if result.ndim == 0 else result

    def time(self, arr):
        """Convert time from 1/ωpe to 1/Ωci_y.

        Parameters
        ----------
        arr : float or array-like
            Time data in units of 1/ωpe.

        Returns
        -------
        float or numpy.ndarray
            Time data in units of 1/Ωci_y (ion cyclotron periods / 2π).
            Returns a scalar float when *arr* is a scalar.
        """
        a = np.asarray(arr, dtype=float)
        result = a * self.wpe_to_wci
        return float(result) if result.ndim == 0 else result

    def speed(self, arr):
        """Convert speed from c to ion Alfvén speed vAi.

        Parameters
        ----------
        arr : float or array-like
            Velocity data in units of c.

        Returns
        -------
        float or numpy.ndarray
            Velocity data in units of vAi.  Returns a scalar float when
            *arr* is a scalar.
        """
        a = np.asarray(arr, dtype=float)
        result = a * self.c_to_vAi
        return float(result) if result.ndim == 0 else result

    def field_B(self, arr):
        """Normalise a magnetic field array to B0.

        Tristan-V2 stores B fields in raw code units where the upstream
        guide field has magnitude B0 = CC × √σ / c_omp ≈ O(0.01).
        Dividing by B0 gives dimensionless values where the upstream
        By ≈ 1 (and B0 ≡ 1 in ion units).

        Parameters
        ----------
        arr : array-like
            Magnetic field data in Tristan code units.

        Returns
        -------
        numpy.ndarray
            Magnetic field normalised to B0 (upstream ≈ 1).
        """
        return np.asarray(arr, dtype=float) / self.B0

    def field_E(self, arr):
        """Normalise an electric field array to E0 = B0 × vAi_over_c.

        The natural ion electric field unit is set by ideal MHD:
        E ~ (v/c) × B ~ (vAi/c) × B0.  With this normalisation an ExB
        equilibrium at drift velocity v_ExB gives field_E × field_B = v_ExB/vAi
        directly, and the Poynting flux in ion units is simply field_E × field_B.

        For the standard test case (vx = 0.01c, By = B0):
            Ez_eq = -0.01 × B0  →  field_E(Ez_eq) = -0.01/vAi_over_c = -2.5

        Parameters
        ----------
        arr : array-like
            Electric field data in Tristan code units.

        Returns
        -------
        numpy.ndarray
            Electric field normalised to E0 = B0 × vAi_over_c.
        """
        return np.asarray(arr, dtype=float) / self.E0

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UnitConverter("
            f"c_omp={self._c_omp}, "
            f"sigma={self._sigma}, "
            f"mass_ratio={self._mass_ratio}, "
            f"CC={self._CC})"
        )

    def summary(self) -> str:
        """Return a human-readable summary of derived scales."""
        lines = [
            "UnitConverter summary",
            "─" * 40,
            f"  c_omp (de in cells)    = {self._c_omp:.4g}",
            f"  sigma (ωce²/ωpe²)      = {self._sigma:.4g}",
            f"  mass_ratio (mi/me)     = {self._mass_ratio:.4g}",
            f"  CC (c in code units)   = {self._CC:.4g}",
            "",
            "Derived scales:",
            f"  B0 (code units)        = {self.B0:.6g}",
            f"  di in cells            = {self.di_in_cells:.4g}",
            f"  vAi / c                = {self.vAi_over_c:.4g}",
            f"  Ωci / ωpe              = {self.Omega_ci_over_wpe:.4g}",
            "",
            "Conversion factors (code → ion):",
            f"  cell_to_di             = {self.cell_to_di:.6g}",
            f"  wpe_to_wci             = {self.wpe_to_wci:.6g}",
            f"  c_to_vAi               = {self.c_to_vAi:.6g}",
            f"  field_B:   ÷ B0        = {self.B0:.6g}",
            f"  field_E:   ÷ E0        = {self.E0:.6g}",
        ]
        return "\n".join(lines)
