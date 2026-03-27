"""
Derived diagnostic quantities computed from field-file moment tensors.

``FieldMoments`` is the field-file analogue of ``ParticleMoments``: it reads
the moment-tensor datasets written by Tristan-MP V2 (``dens``, ``TXX``,
``jprtX``, ``QX``, …) and exposes physically meaningful derived quantities
with optional ion-unit conversion.

Usage::

    fm = sim.field_moments(step=10)

    vel = fm.bulk_velocity(1, units='ion')   # {'x','y','z'} in vAi
    T   = fm.temperature_tensor(2)           # {'xx','yy','zz',…} code units
    rho = fm.charge_density(1, units='ion')  # normalised to n0
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .fields import FieldSnapshot
    from .units import UnitConverter


class FieldMoments:
    """Derived quantities from field-file moment tensors.

    Parameters
    ----------
    field_snapshot : FieldSnapshot
        A code-unit field snapshot (``units='code'``).  The snapshot must
        have a ``params`` object attached so that ``_species_mass``,
        ``_species_charge``, and ``_n0`` are populated.
    unit_converter : UnitConverter, optional
        Required for ``units='ion'``.  Falls back to the converter already
        attached to *field_snapshot*.
    n0 : float, optional
        Reference number density ``ppc0/2`` in code units.  Falls back to
        ``field_snapshot._n0``.
    """

    def __init__(
        self,
        field_snapshot: FieldSnapshot,
        unit_converter: UnitConverter | None = None,
        n0: float | None = None,
    ) -> None:
        self._flds = field_snapshot
        self.uc: UnitConverter | None = (
            unit_converter if unit_converter is not None else field_snapshot.uc
        )
        self._n0: float | None = (
            n0 if n0 is not None else field_snapshot._n0
        )
        self._cache: dict[tuple, object] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_ion_ready(self) -> None:
        """Raise if ion-unit conversion is not possible."""
        if self.uc is None:
            raise ValueError(
                "unit_converter is required for units='ion'. "
                "Provide it when constructing FieldMoments, or attach a "
                "params file to the FieldSnapshot."
            )
        if self._n0 is None:
            raise ValueError(
                "n0 (= ppc0/2) is required for ion-unit conversion. "
                "Attach a params file when constructing the FieldSnapshot, "
                "or pass n0= to FieldMoments."
            )

    def _stress_raw(self, species_id: int) -> dict[str, np.ndarray]:
        """Raw stress-tensor components in code units, cached.

        Off-diagonal components absent from the file are substituted with
        zero arrays and a ``RuntimeWarning`` is emitted.
        """
        key = ("stress_raw", species_id)
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]

        sid = species_id
        result: dict[str, np.ndarray] = {}

        for comp in ("XX", "YY", "ZZ"):
            result[comp.lower()] = self._flds._load(f"T{comp}{sid}")

        for comp in ("XY", "XZ", "YZ"):
            dataset = f"T{comp}{sid}"
            if dataset in self._flds.keys:
                result[comp.lower()] = self._flds._load(dataset)
            else:
                shape = result["xx"].shape
                result[comp.lower()] = np.zeros(shape, dtype=result["xx"].dtype)
                warnings.warn(
                    f"{dataset} not found in {self._flds.filepath.name}; "
                    "treating off-diagonal component as zero.",
                    RuntimeWarning,
                    stacklevel=3,
                )

        self._cache[key] = result
        return result

    def _heat_flux_raw(self, species_id: int) -> dict[str, np.ndarray]:
        """Raw third-moment QX/QY/QZ in code units, cached.

        Tristan-MP V2 stores ``ρ_s <|v|² v_i>`` without a factor of ½ and
        without bulk-flow subtraction.  Used by ``EnergyFlux.heat_flux()``.
        """
        key = ("heat_flux_raw", species_id)
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]

        sid = species_id
        result: dict[str, np.ndarray] = {
            "x": self._flds._load(f"QX{sid}"),
            "y": self._flds._load(f"QY{sid}"),
            "z": self._flds._load(f"QZ{sid}"),
        }
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Public diagnostic methods
    # ------------------------------------------------------------------

    def bulk_velocity(
        self, species_id: int, units: str = "code"
    ) -> dict[str, np.ndarray]:
        """Bulk (mean) velocity of species *s*, dict with keys 'x','y','z'.

        Computed from the charge-current density and mass density::

            U_i = (m_k / q_k) × jprt_i / dens_k

        In code units: ``[c]``.

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
            ``'ion'`` returns velocity in units of ``vAi``.
        """
        cache_key = ("vel_code", species_id)
        if cache_key not in self._cache:
            sid = species_id
            m_k = self._flds._species_mass.get(sid)
            q_k = self._flds._species_charge.get(sid)
            if m_k is None or q_k is None:
                raise ValueError(
                    f"Mass and/or charge for species {sid} not found. "
                    "Provide a params file when constructing the FieldSnapshot."
                )
            if q_k == 0:
                raise ValueError(
                    f"Species {sid} has charge 0; cannot compute bulk velocity."
                )
            jx   = self._flds._load(f"jprtX{sid}")
            jy   = self._flds._load(f"jprtY{sid}")
            jz   = self._flds._load(f"jprtZ{sid}")
            dens = self._flds._load(f"dens{sid}")
            safe_inv = np.where(dens > 0, 1.0 / np.where(dens > 0, dens, 1.0), 0.0)
            fac = m_k / q_k
            self._cache[cache_key] = {
                "x": fac * jx * safe_inv,
                "y": fac * jy * safe_inv,
                "z": fac * jz * safe_inv,
            }
        result = self._cache[cache_key]
        if units == "ion":
            self._check_ion_ready()
            f = self.uc.c_to_vAi
            return {"x": result["x"] * f, "y": result["y"] * f, "z": result["z"] * f}
        return result

    def charge_density(
        self, species_id: int, units: str = "code"
    ) -> np.ndarray:
        """Charge density of species *s*: ρ_k = q_k n_k.

        In code units: ``[q_k / cell³]``.
        In ion units: normalised to ``n0``, giving ``q_k n_k / n0`` (≈ ±1
        upstream for a neutral background plasma).

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
        """
        cache_key = ("charge_density_code", species_id)
        if cache_key not in self._cache:
            sid = species_id
            m_k = self._flds._species_mass.get(sid)
            q_k = self._flds._species_charge.get(sid)
            if m_k is None or q_k is None:
                raise ValueError(
                    f"Mass and/or charge for species {sid} not found. "
                    "Provide a params file when constructing the FieldSnapshot."
                )
            dens = self._flds._load(f"dens{sid}")
            self._cache[cache_key] = (q_k / m_k) * dens
        arr = self._cache[cache_key]
        if units == "ion":
            self._check_ion_ready()
            return arr / self._n0
        return arr

    def pressure_tensor(
        self, species_id: int, units: str = "code"
    ) -> dict[str, np.ndarray]:
        """Pressure tensor P_ij = Π_ij − ρ_s U_i U_j, keys 'xx'…'yz'.

        Subtracts the bulk-flow contribution from the full lab-frame stress
        tensor (TXX, TYY, … stored by Tristan-MP V2) to give the true thermal
        pressure tensor in the bulk frame.

        In code units: ``[m_k n_k c²]``.

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
            ``'ion'`` normalises to ``[n0 mi vAi²]``.
        """
        cache_key = ("pressure_code", species_id)
        if cache_key not in self._cache:
            stress = self._stress_raw(species_id)
            dens   = self._flds._load(f"dens{species_id}")
            vel    = self.bulk_velocity(species_id)
            vx, vy, vz = vel["x"], vel["y"], vel["z"]
            self._cache[cache_key] = {
                "xx": stress["xx"] - dens * vx * vx,
                "yy": stress["yy"] - dens * vy * vy,
                "zz": stress["zz"] - dens * vz * vz,
                "xy": stress["xy"] - dens * vx * vy,
                "xz": stress["xz"] - dens * vx * vz,
                "yz": stress["yz"] - dens * vy * vz,
            }
        result = self._cache[cache_key]
        if units == "ion":
            self._check_ion_ready()
            f = self.uc.c_to_vAi ** 2 / (self._n0 * self.uc.mass_ratio)
            return {comp: result[comp] * f for comp in result}
        return result

    def temperature_tensor(
        self, species_id: int, units: str = "code"
    ) -> dict[str, np.ndarray]:
        """Temperature tensor T_ij = P_ij / n_k, keys 'xx'…'yz'.

        Divides the pressure tensor by the number density ``n_k = dens_k / m_k``
        to give the temperature per particle.

        In code units: ``[m_k c²]``.
        In ion units: ``[mi vAi²]`` for all species, using the factor
        ``c_to_vAi² / mass_ratio`` (which correctly accounts for both electron
        and ion mass ratios).

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
            ``'ion'`` normalises to ``[mi vAi²]``.
        """
        cache_key = ("temp_tensor_code", species_id)
        if cache_key not in self._cache:
            sid = species_id
            m_k = self._flds._species_mass.get(sid)
            if m_k is None:
                raise ValueError(
                    f"Mass for species {sid} not found. "
                    "Provide a params file when constructing the FieldSnapshot."
                )
            P    = self.pressure_tensor(sid)
            dens = self._flds._load(f"dens{sid}")
            # n_k = dens / m_k  →  T_ij = P_ij / n_k = P_ij × m_k / dens
            safe_inv = np.where(dens > 0, 1.0 / np.where(dens > 0, dens, 1.0), 0.0)
            self._cache[cache_key] = {
                comp: P[comp] * m_k * safe_inv for comp in P
            }
        result = self._cache[cache_key]
        if units == "ion":
            self._check_ion_ready()
            f = self.uc.c_to_vAi ** 2 / self.uc.mass_ratio
            return {comp: result[comp] * f for comp in result}
        return result

    # ------------------------------------------------------------------
    # Cache management / repr
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Release all cached intermediate arrays."""
        self._cache.clear()

    def __repr__(self) -> str:
        step = self._flds.step
        uc_info = "with UnitConverter" if self.uc is not None else "no UnitConverter"
        n0_info = f"n0={self._n0}" if self._n0 is not None else "n0=unknown"
        return f"FieldMoments(step={step}, {uc_info}, {n0_info})"
