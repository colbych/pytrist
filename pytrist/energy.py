"""
Energy density flux analysis for Tristan-MP V2 simulation output.

For each particle species and the electromagnetic field, computes the terms in
the energy flux decomposition::

    Q_s = q_KE + q_enthalpy + q_heat   (particle species s)
    S   = (CC/4π) E × B                (Poynting flux)

All methods accept ``units='code'`` (default) or ``units='ion'``.

Ion-unit normalisation for particle terms::

    energy density  [n0 mi vAi²]  →  divide by (n0 × mass_ratio), multiply by c_to_vAi²
    energy flux     [n0 mi vAi³]  →  divide by (n0 × mass_ratio), multiply by c_to_vAi³

Ion-unit normalisation for Poynting flux uses the Gaussian Alfvén relation
B0² = 4π n0 mi vAi² so that Poynting and particle fluxes share the same unit::

    S [n0 mi vAi³] = CC × (E×B) / (4π × n0 × mass_ratio × vAi_code²)

Usage example::

    ef = sim.energy_flux(step=10)

    ke_ion = ef.bulk_ke_density(2, units='ion')       # ions, shape (nz, ny, nx)
    q_enth = ef.enthalpy_flux(2, units='ion')          # {'x','y','z'}
    S      = ef.poynting_flux(units='ion')             # {'x','y','z'}
    total  = ef.total_particle_energy_flux(2, units='ion')
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .fields import FieldSnapshot
    from .units import UnitConverter


class EnergyFlux:
    """Energy density flux terms computed from field-file moment tensors.

    Parameters
    ----------
    field_snapshot : FieldSnapshot
        A code-unit field snapshot (``units='code'``).  The snapshot must have
        been loaded with a ``params`` object so that ``_species_mass``,
        ``_species_charge``, and ``_n0`` are populated.
    unit_converter : UnitConverter, optional
        Required when ``units='ion'`` is requested.  Falls back to the
        converter already attached to *field_snapshot* if not provided.
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
                "Provide it when constructing EnergyFlux, or attach a params "
                "file to the FieldSnapshot so Simulation can build one."
            )
        if self._n0 is None:
            raise ValueError(
                "n0 (= ppc0/2) is required for ion-unit conversion. "
                "Attach a params file when constructing the FieldSnapshot, "
                "or pass n0= to EnergyFlux."
            )

    def _energy_density_factor(self) -> float:
        """Ion-unit factor for energy density: / (n0 × mr) × c_to_vAi²."""
        return self.uc.c_to_vAi ** 2 / (self._n0 * self.uc.mass_ratio)

    def _energy_flux_factor(self) -> float:
        """Ion-unit factor for energy flux: / (n0 × mr) × c_to_vAi³."""
        return self.uc.c_to_vAi ** 3 / (self._n0 * self.uc.mass_ratio)

    def _bulk_velocity_raw(self, species_id: int) -> dict[str, np.ndarray]:
        """Bulk velocity in code units [c], cached."""
        key = ("vel_raw", species_id)
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]

        sid = species_id
        m_k = self._flds._species_mass.get(sid)
        q_k = self._flds._species_charge.get(sid)
        if m_k is None or q_k is None:
            raise ValueError(
                f"Mass and/or charge for species {sid} not found. "
                "Provide a params file when constructing the FieldSnapshot."
            )
        if q_k == 0:
            raise ValueError(f"Species {sid} has charge 0; cannot compute bulk velocity.")

        jx = self._flds._load(f"jprtX{sid}")
        jy = self._flds._load(f"jprtY{sid}")
        jz = self._flds._load(f"jprtZ{sid}")
        dens = self._flds._load(f"dens{sid}")

        safe_inv = np.where(dens > 0, 1.0 / np.where(dens > 0, dens, 1.0), 0.0)
        fac = m_k / q_k
        result: dict[str, np.ndarray] = {
            "vx": fac * jx * safe_inv,
            "vy": fac * jy * safe_inv,
            "vz": fac * jz * safe_inv,
        }
        self._cache[key] = result
        return result

    def _stress_raw(self, species_id: int) -> dict[str, np.ndarray]:
        """Raw stress-tensor components in code units, cached.

        Off-diagonal components that are absent from the file are substituted
        with zero arrays and a RuntimeWarning is emitted.
        """
        key = ("stress_raw", species_id)
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]

        sid = species_id
        result: dict[str, np.ndarray] = {}

        # Diagonal components are required
        for comp in ("XX", "YY", "ZZ"):
            dataset = f"T{comp}{sid}"
            result[comp.lower()] = self._flds._load(dataset)

        # Off-diagonal components are optional; substitute zero with a warning
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
        """Raw heat-flux components QX/QY/QZ in code units, cached."""
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

    def _pressure_tensor_raw(self, species_id: int) -> dict[str, np.ndarray]:
        """Pressure tensor P_ij = Π_ij − ρ_s U_i U_j in code units, cached.

        Subtracts the bulk-flow contribution from the full stress-energy
        tensor (TXX, TYY, … as stored by Tristan) to give the true thermal
        pressure tensor in the bulk frame.  Off-diagonal components that are
        absent from the file are substituted with zero (via ``_stress_raw``).
        """
        key = ("pressure_raw", species_id)
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]

        stress = self._stress_raw(species_id)
        dens   = self._flds._load(f"dens{species_id}")
        vel    = self._bulk_velocity_raw(species_id)
        vx, vy, vz = vel["vx"], vel["vy"], vel["vz"]

        self._cache[key] = {
            "xx": stress["xx"] - dens * vx * vx,
            "yy": stress["yy"] - dens * vy * vy,
            "zz": stress["zz"] - dens * vz * vz,
            "xy": stress["xy"] - dens * vx * vy,
            "xz": stress["xz"] - dens * vx * vz,
            "yz": stress["yz"] - dens * vy * vz,
        }
        return self._cache[key]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Scalar energy densities
    # ------------------------------------------------------------------

    def bulk_ke_density(self, species_id: int, units: str = "code") -> np.ndarray:
        """Bulk kinetic energy density: ½ dens_s |U_s|².

        In code units: ``[m_k n_k c²]`` — same dimension as stress-tensor
        diagonal components.

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
            ``'ion'`` normalises to ``[n0 mi vAi²]``.
        """
        cache_key = ("ke_density_code", species_id)
        if cache_key not in self._cache:
            dens = self._flds._load(f"dens{species_id}")
            vel = self._bulk_velocity_raw(species_id)
            self._cache[cache_key] = 0.5 * dens * (
                vel["vx"] ** 2 + vel["vy"] ** 2 + vel["vz"] ** 2
            )
        arr = self._cache[cache_key]
        if units == "ion":
            self._check_ion_ready()
            return arr * self._energy_density_factor()
        return arr

    def internal_energy_density(self, species_id: int, units: str = "code") -> np.ndarray:
        """Thermal energy density: ½ Tr(P_s) = ½ (TXX + TYY + TZZ).

        In code units: ``[m_k n_k c²]``.  For an isotropic Maxwellian this
        equals ``(3/2) n k_B T``.

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
            ``'ion'`` normalises to ``[n0 mi vAi²]``.
        """
        cache_key = ("u_th_code", species_id)
        if cache_key not in self._cache:
            P = self._pressure_tensor_raw(species_id)
            self._cache[cache_key] = 0.5 * (P["xx"] + P["yy"] + P["zz"])
        arr = self._cache[cache_key]
        if units == "ion":
            self._check_ion_ready()
            return arr * self._energy_density_factor()
        return arr

    # ------------------------------------------------------------------
    # Vector energy fluxes  (return {'x', 'y', 'z'})
    # ------------------------------------------------------------------

    def bulk_ke_flux(
        self, species_id: int, units: str = "code"
    ) -> dict[str, np.ndarray]:
        """Bulk kinetic energy flux: ½ dens_s |U_s|² × U_s.

        In code units: ``[m_k n_k c³]`` — same dimension as heat-flux
        components.

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
            ``'ion'`` normalises to ``[n0 mi vAi³]``.
        """
        cache_key = ("ke_flux_code", species_id)
        if cache_key not in self._cache:
            ke  = self.bulk_ke_density(species_id)
            vel = self._bulk_velocity_raw(species_id)
            self._cache[cache_key] = {
                "x": ke * vel["vx"],
                "y": ke * vel["vy"],
                "z": ke * vel["vz"],
            }
        result = self._cache[cache_key]
        if units == "ion":
            self._check_ion_ready()
            f = self._energy_flux_factor()
            return {"x": result["x"] * f, "y": result["y"] * f, "z": result["z"] * f}
        return result

    def internal_energy_flux(
        self, species_id: int, units: str = "code"
    ) -> dict[str, np.ndarray]:
        """Internal (thermal) energy flux: u_th_s × U_s.

        Advection of thermal energy density by the bulk flow.
        ``u_th = ½ Tr(P_s)`` is the scalar thermal energy density.

        In code units: ``[m_k n_k c³]``.

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
            ``'ion'`` normalises to ``[n0 mi vAi³]``.
        """
        cache_key = ("u_th_flux_code", species_id)
        if cache_key not in self._cache:
            u_th = self.internal_energy_density(species_id)
            vel = self._bulk_velocity_raw(species_id)
            self._cache[cache_key] = {
                "x": u_th * vel["vx"],
                "y": u_th * vel["vy"],
                "z": u_th * vel["vz"],
            }
        result = self._cache[cache_key]
        if units == "ion":
            self._check_ion_ready()
            f = self._energy_flux_factor()
            return {"x": result["x"] * f, "y": result["y"] * f, "z": result["z"] * f}
        return result

    def enthalpy_flux(
        self, species_id: int, units: str = "code"
    ) -> dict[str, np.ndarray]:
        """Enthalpy (pressure-tensor work) flux: P_s · U_s.

        Full pressure-tensor contracted with bulk velocity::

            q_x = TXX vx + TXY vy + TXZ vz
            q_y = TXY vx + TYY vy + TYZ vz
            q_z = TXZ vx + TYZ vy + TZZ vz

        For isotropic pressure this reduces to ``p_iso × U``.  The anisotropic
        contribution from off-diagonal terms is included when the corresponding
        datasets (TXY, TXZ, TYZ) are present; otherwise a ``RuntimeWarning``
        is emitted and those terms are set to zero.

        In code units: ``[m_k n_k c³]``.

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
            ``'ion'`` normalises to ``[n0 mi vAi³]``.
        """
        cache_key = ("enthalpy_flux_code", species_id)
        if cache_key not in self._cache:
            P   = self._pressure_tensor_raw(species_id)
            vel = self._bulk_velocity_raw(species_id)
            vx, vy, vz = vel["vx"], vel["vy"], vel["vz"]
            self._cache[cache_key] = {
                "x": P["xx"] * vx + P["xy"] * vy + P["xz"] * vz,
                "y": P["xy"] * vx + P["yy"] * vy + P["yz"] * vz,
                "z": P["xz"] * vx + P["yz"] * vy + P["zz"] * vz,
            }
        result = self._cache[cache_key]
        if units == "ion":
            self._check_ion_ready()
            f = self._energy_flux_factor()
            return {"x": result["x"] * f, "y": result["y"] * f, "z": result["z"] * f}
        return result

    def heat_flux(
        self, species_id: int, units: str = "code"
    ) -> dict[str, np.ndarray]:
        """Third-order heat-flux cumulant: QX_s, QY_s, QZ_s.

        This is the irreducible heat flux (random thermal transport), distinct
        from the ``enthalpy_flux`` (pressure × bulk velocity).

        In code units: ``[m_k n_k c³]``.

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
            ``'ion'`` normalises to ``[n0 mi vAi³]``.
        """
        cache_key = ("heat_flux_code", species_id)
        if cache_key not in self._cache:
            # QX/QY/QZ store the raw third moment ρ_s <|v|² v_i> — no ½ factor
            # and no bulk-flow subtraction (analogous to TXX storing the full
            # second moment rather than the pressure tensor).
            # Extract the true heat-flux cumulant via the identity:
            #   ρ <|v|² v_i> = 2 q_KE_i + 2 q_enthalpy_i + 2 q_IE_i + 2 q_heat_i
            # => q_heat_i = ½ Q_raw_i − q_KE_i − q_enthalpy_i − q_IE_i
            raw  = self._heat_flux_raw(species_id)
            ke   = self.bulk_ke_flux(species_id)
            enth = self.enthalpy_flux(species_id)
            ie   = self.internal_energy_flux(species_id)
            self._cache[cache_key] = {
                "x": 0.5 * raw["x"] - ke["x"] - enth["x"] - ie["x"],
                "y": 0.5 * raw["y"] - ke["y"] - enth["y"] - ie["y"],
                "z": 0.5 * raw["z"] - ke["z"] - enth["z"] - ie["z"],
            }
        result = self._cache[cache_key]
        if units == "ion":
            self._check_ion_ready()
            f = self._energy_flux_factor()
            return {"x": result["x"] * f, "y": result["y"] * f, "z": result["z"] * f}
        return result

    def poynting_flux(self, units: str = "code") -> dict[str, np.ndarray]:
        """Electromagnetic energy flux: S = (CC/4π) E × B.

        In code units, returns ``CC × (E × B)`` (the Gaussian Poynting vector
        with the explicit speed-of-light factor).

        In ion units, normalises to ``[n_phys mi vAi³]`` using the Gaussian
        Alfvén relation ``B0² = 4π n_phys mi vAi²`` (where ``n_phys`` is the
        physical background density, not ``ppc0/2``).  The 4π factors cancel
        exactly, giving::

            S_ion = (E×B) / (B0² × vAi_over_c)
                  = CC × (E×B) / (CC × B0 × E0)

        where ``E0 = B0 × vAi_over_c`` is the natural ion electric field unit.
        With this normalisation an ExB drift at velocity v gives S_ion = v/vAi.

        Unlike particle energy fluxes this does **not** require ``n0 = ppc0/2``;
        only a ``UnitConverter`` is needed.

        Parameters
        ----------
        units : {'code', 'ion'}
            ``'ion'`` normalises to ``[n_phys mi vAi³]``.
        """
        cache_key = ("poynting_flux_code",)
        if cache_key not in self._cache:
            if self.uc is None:
                raise ValueError(
                    "unit_converter is required to compute poynting_flux. "
                    "Provide one when constructing EnergyFlux."
                )
            ex = self._flds._load("ex")
            ey = self._flds._load("ey")
            ez = self._flds._load("ez")
            bx = self._flds._load("bx")
            by = self._flds._load("by")
            bz = self._flds._load("bz")
            CC = self.uc.CC
            self._cache[cache_key] = {
                "x": CC * (ey * bz - ez * by),
                "y": CC * (ez * bx - ex * bz),
                "z": CC * (ex * by - ey * bx),
            }
        result = self._cache[cache_key]
        if units == "ion":
            if self.uc is None:
                raise ValueError("unit_converter is required for units='ion'.")
            f = 1.0 / (self.uc.CC * self.uc.B0 * self.uc.E0)
            return {"x": result["x"] * f, "y": result["y"] * f, "z": result["z"] * f}
        return result

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def total_particle_energy_flux(
        self, species_id: int, units: str = "code"
    ) -> dict[str, np.ndarray]:
        """Total particle energy flux: q_KE + q_enthalpy + q_heat.

        Sum of all three transport contributions for species *s*::

            Q_s = bulk_ke_flux + enthalpy_flux + heat_flux

        In code units: ``[m_k n_k c³]``.

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
            ``'ion'`` normalises to ``[n0 mi vAi³]``.
        """
        cache_key = ("total_prtl_flux_code", species_id)
        if cache_key not in self._cache:
            ke   = self.bulk_ke_flux(species_id)
            enth = self.enthalpy_flux(species_id)
            heat = self.heat_flux(species_id)
            self._cache[cache_key] = {
                "x": ke["x"] + enth["x"] + heat["x"],
                "y": ke["y"] + enth["y"] + heat["y"],
                "z": ke["z"] + enth["z"] + heat["z"],
            }
        result = self._cache[cache_key]
        if units == "ion":
            self._check_ion_ready()
            f = self._energy_flux_factor()
            return {"x": result["x"] * f, "y": result["y"] * f, "z": result["z"] * f}
        return result

    def total_energy_flux(
        self,
        species_ids: list[int] | None = None,
        units: str = "code",
    ) -> dict[str, np.ndarray]:
        """Total energy flux: Poynting + sum of particle fluxes over species.

        Parameters
        ----------
        species_ids : list of int, optional
            Species to include.  Defaults to ``[1, 2]``.
        units : {'code', 'ion'}
            ``'ion'`` normalises particle fluxes to ``[n0 mi vAi³]`` and
            Poynting to ``[n_phys mi vAi³]``.
        """
        if species_ids is None:
            species_ids = [1, 2]
        result = {ax: arr.copy() for ax, arr in self.poynting_flux(units=units).items()}
        for sid in species_ids:
            p = self.total_particle_energy_flux(sid, units=units)
            for ax in ("x", "y", "z"):
                result[ax] = result[ax] + p[ax]
        return result

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Release all cached intermediate arrays."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        step = self._flds.step
        uc_info = "with UnitConverter" if self.uc is not None else "no UnitConverter"
        n0_info = f"n0={self._n0}" if self._n0 is not None else "n0=unknown"
        return f"EnergyFlux(step={step}, {uc_info}, {n0_info})"
