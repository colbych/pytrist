"""
Field data reader for Tristan-MP V2 simulation output.

Field snapshots are stored in HDF5 files named ``flds.tot.NNNNN`` where
NNNNN is the zero-padded 5-digit step number.

Datasets present in a typical field file
-----------------------------------------
Electromagnetic:   ex, ey, ez, bx, by, bz, jx, jy, jz
Per species:       dens_1, dens_2, enrg_1, enrg_2, ...
                   (species index suffix varies by run)

Moment tensor datasets (when enabled in Tristan output config)
--------------------------------------------------------------
Momentum density:  T0X{k}, T0Y{k}, T0Z{k}     — first-order moment
Diagonal stress:   TXX{k}, TYY{k}, TZZ{k}     — pressure tensor diagonal
Off-diagonal:      TXY{k}, TXZ{k}, TYZ{k}     — shear stress
Heat flux:         QX{k},  QY{k},  QZ{k}       — third-order moment

All arrays are 3-D (nz, ny, nx) even for 2-D runs (nz==1).

Usage example::

    from pytrist.fields import FieldSnapshot

    snap = FieldSnapshot("/path/to/flds.tot.00001")
    bx = snap.bx          # numpy array, loaded lazily
    by = snap["by"]       # dict-like access also supported

    # With unit converter:
    from pytrist.units import UnitConverter
    uc = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
    snap = FieldSnapshot("/path/to/flds.tot.00001", unit_converter=uc)
    bx_ion = snap.bx      # same values; unit label changes
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .units import UnitConverter

# Fields normalised to B0 or E0 when units='ion'
_B_KEYS: frozenset[str] = frozenset({"bx", "by", "bz"})
_E_KEYS: frozenset[str] = frozenset({"ex", "ey", "ez"})

# Regex patterns for per-species moment fields (both dens1 and dens_1 style)
_DENS_RE = re.compile(r"^dens_?(\d+)$")             # dens1, dens_1, dens2, …
_VEL_RE = re.compile(r"^vel([xyz])_?(\d*)$")        # velx, velx1, velx_1, …
_STRESS_RE = re.compile(r"^T([XYZ0]{2})_?(\d+)$")  # TXX1, TYY2, TXZ1, T0X1, …
_HEAT_FLUX_RE = re.compile(r"^Q([XYZ])_?(\d+)$")   # QX1, QY2, QZ1, …


class FieldSnapshot:
    """Lazy-loading container for a single field snapshot.

    Parameters
    ----------
    filepath : str or Path
        Path to a ``flds.tot.NNNNN`` HDF5 file.
    unit_converter : UnitConverter, optional
        Required when ``units='ion'``.  Also available as ``self.uc``
        for manual conversions.
    params : SimParams, optional
        Associated simulation parameters snapshot.  Used to populate
        the ``time`` property when the field file itself does not store
        a time attribute.
    units : {'code', 'ion'}, optional
        Unit system for returned field arrays.

        * ``'code'`` (default) — raw Tristan code units.  B and E fields
          have magnitude ~B0 ≈ CC√σ/c_omp upstream.
        * ``'ion'`` — applies physical normalisation automatically:

          - **B fields** (bx/by/bz): divided by B0 → upstream By ≈ 1.
          - **E fields** (ex/ey/ez): divided by E0 = B0 × vAi_over_c → equilibrium
            ExB field Ez = −v_ExB/c × By gives field_E(Ez) = −v_ExB/vAi.
          - **Number density** (dens1, dens_1, …): divided by ``m_k × n0``
            (where ``n0 = ppc0/2``) → dimensionless, upstream ≈ 1.
          - **Bulk velocity** (velx, vely, velz, velx1, …): multiplied by
            ``c_to_vAi = √(mass_ratio/σ)`` → units of vAi.
          - **Stress tensor** (TXX1, TYY2, TXZ1, T0X1, …): divided by
            ``n0 × mass_ratio`` then multiplied by ``c_to_vAi²`` → units of
            ``(n/n0) × (m_k/m_i) × vAi²``.  Dividing by density gives
            temperature ``T = (m_k/m_i) ⟨δv²⟩ c_to_vAi²`` in units of
            m_i vAi², equal for both species in an equal-temperature plasma
            (e.g. T_e = T_i ≈ 0.1 m_i vAi² upstream if initialized equal).
          - **Heat flux** (QX1, QY2, …): divided by ``n0 × mass_ratio`` then
            multiplied by ``c_to_vAi³`` → units of ``(n/n0) × (m_k/m_i) × vAi³``.
            This is consistent with the stress-tensor normalization with one
            additional power of velocity.
          - All other fields (currents, energy density, etc.) are returned
            unchanged.

        Requires ``unit_converter`` to be set when ``units='ion'``.
    """

    def __init__(
        self,
        filepath: str | Path,
        unit_converter: UnitConverter | None = None,
        params=None,
        units: str = "code",
    ) -> None:
        if units not in ("code", "ion"):
            raise ValueError(f"units must be 'code' or 'ion', got {units!r}")
        if units == "ion" and unit_converter is None:
            raise ValueError("unit_converter must be provided when units='ion'.")
        self.filepath = Path(filepath).resolve()
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Field file not found: {self.filepath}\n"
                "Expected a file named like 'flds.tot.NNNNN'."
            )
        self.uc = unit_converter
        self._params = params
        self.units = units

        # Cache for lazily loaded arrays
        self._cache: dict[str, np.ndarray] = {}
        # Available dataset names (populated on first access)
        self._keys: list[str] | None = None

        # Extract step number from filename suffix (e.g. "flds.tot.00042")
        parts = self.filepath.name.split(".")
        try:
            self.step: int = int(parts[-1])
        except (ValueError, IndexError):
            self.step = -1

        # Extract n0, per-species masses, and charges from params
        self._n0: float | None = None
        self._species_mass: dict[int, float] = {}
        self._species_charge: dict[int, float] = {}
        if params is not None:
            for key in ("plasma:ppc0", "ppc0", "ppc"):
                try:
                    val = params[key]
                    if val is not None:
                        self._n0 = float(val) / 2.0
                        break
                except (KeyError, TypeError, AttributeError):
                    pass
            for sid in range(1, 9):
                for mk in (f"particles:m{sid}", f"m{sid}"):
                    try:
                        val = params[mk]
                        if val is not None:
                            self._species_mass[sid] = float(val)
                            break
                    except (KeyError, TypeError, AttributeError):
                        pass
                for ck in (f"particles:ch{sid}", f"particles:q{sid}", f"ch{sid}", f"q{sid}"):
                    try:
                        val = params[ck]
                        if val is not None:
                            self._species_charge[sid] = float(val)
                            break
                    except (KeyError, TypeError, AttributeError):
                        pass

    # ------------------------------------------------------------------
    # Key / dataset discovery
    # ------------------------------------------------------------------

    @property
    def keys(self) -> list[str]:
        """Names of all datasets available in this file."""
        if self._keys is None:
            with h5py.File(self.filepath, "r") as f:
                self._keys = list(f.keys())
        return self._keys

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    # ------------------------------------------------------------------
    # Lazy data access
    # ------------------------------------------------------------------

    def _load(self, key: str) -> np.ndarray:
        """Load a single dataset from the HDF5 file (with caching)."""
        if key not in self._cache:
            if key not in self.keys:
                raise KeyError(
                    f"Field '{key}' not found in {self.filepath.name}.\n"
                    f"Available fields: {self.keys}"
                )
            with h5py.File(self.filepath, "r") as f:
                self._cache[key] = f[key][()]
        return self._cache[key]

    def _load_field(self, key: str) -> np.ndarray:
        """Load *key*, applying unit normalisation when units='ion'."""
        arr = self._load(key)
        if self.units != "ion":
            return arr

        # B fields → divide by B0 (upstream By ≈ 1)
        if key in _B_KEYS:
            return self.uc.field_B(arr)

        # E fields → divide by E0 = B0 × vAi_over_c
        if key in _E_KEYS:
            return self.uc.field_E(arr)

        # Number density dens{k} or dens_{k} → divide by m_k × n0
        m = _DENS_RE.match(key)
        if m:
            sid = int(m.group(1))
            m_k = self._species_mass.get(sid, 1.0)
            n0 = self._n0 if self._n0 is not None else 1.0
            return arr / (m_k * n0)

        # Bulk velocity vel{x/y/z}[{k}] → multiply by c_to_vAi
        if _VEL_RE.match(key):
            return arr * self.uc.c_to_vAi

        # Stress tensor T{ij}{k} → T = (m_k/m_i) ⟨δv²⟩ c_to_vAi²  [m_i vAi²]
        # TXX_k (code) ≈ m_k × n × ⟨δv²⟩.
        # Dividing by n0 × m_i and multiplying by c_to_vAi² gives pressure
        # (n/n0) × (m_k/m_i) × ⟨δv²⟩ × c_to_vAi².  After dividing by density
        # the intensive temperature T = (m_k/m_i) ⟨δv²⟩ c_to_vAi² is equal
        # for both species in an equal-temperature plasma (T_e = T_i).
        m = _STRESS_RE.match(key)
        if m:
            n0 = self._n0 if self._n0 is not None else 1.0
            return arr / (n0 * self.uc.mass_ratio) * self.uc.c_to_vAi ** 2

        # Heat flux Q{i}{k} → normalized by n0 × m_i × vAi³
        # Q_k (code) ≈ m_k × n × ⟨v²⟩ × v_i, one extra velocity factor vs stress.
        m = _HEAT_FLUX_RE.match(key)
        if m:
            n0 = self._n0 if self._n0 is not None else 1.0
            return arr / (n0 * self.uc.mass_ratio) * self.uc.c_to_vAi ** 3

        return arr

    def __getitem__(self, key: str) -> np.ndarray:
        return self._load_field(key)

    def __getattr__(self, key: str) -> np.ndarray:
        # Avoid infinite recursion for private / dunder attributes
        if key.startswith("_") or key in ("filepath", "uc", "step", "keys"):
            raise AttributeError(key)
        # Only intercept if key looks like a field name (no dots)
        if "." not in key:
            try:
                return self._load_field(key)
            except KeyError as exc:
                raise AttributeError(str(exc)) from exc
        raise AttributeError(key)

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    @property
    def time(self) -> float | None:
        """Simulation time in 1/ωpe for this snapshot.

        Tries, in order:
        1. A ``time`` dataset inside the HDF5 file.
        2. The associated ``SimParams`` object (if provided).
        3. Returns ``None`` if unavailable.
        """
        if "time" in self.keys:
            t = self._load("time")
            if isinstance(t, np.ndarray):
                t = float(t.flat[0])
            return float(t)
        if self._params is not None:
            try:
                return float(self._params.time)
            except AttributeError:
                pass
        return None

    @property
    def time_ion(self) -> float | None:
        """Simulation time converted to ion cyclotron periods (1/Ωci_y).

        Returns ``None`` if no UnitConverter or no time information is
        available.
        """
        if self.uc is None or self.time is None:
            return None
        return float(self.uc.time(self.time))

    # ------------------------------------------------------------------
    # Convenience field properties (electromagnetic)
    # ------------------------------------------------------------------

    @property
    def ex(self) -> np.ndarray:
        """Electric field x-component (normalised to E0 = B0×vAi_over_c when units='ion')."""
        return self._load_field("ex")

    @property
    def ey(self) -> np.ndarray:
        """Electric field y-component (normalised to E0 = B0×vAi_over_c when units='ion')."""
        return self._load_field("ey")

    @property
    def ez(self) -> np.ndarray:
        """Electric field z-component (normalised to E0 = B0×vAi_over_c when units='ion')."""
        return self._load_field("ez")

    @property
    def bx(self) -> np.ndarray:
        """Magnetic field x-component (normalised to B0 when units='ion')."""
        return self._load_field("bx")

    @property
    def by(self) -> np.ndarray:
        """Magnetic field y-component (normalised to B0 when units='ion')."""
        return self._load_field("by")

    @property
    def bz(self) -> np.ndarray:
        """Magnetic field z-component (normalised to B0 when units='ion')."""
        return self._load_field("bz")

    # ------------------------------------------------------------------
    # Derived electromagnetic quantities
    # ------------------------------------------------------------------

    def B_squared(self) -> np.ndarray:
        """Scalar magnetic energy density \|B\|².

        In ion units returns ``(B/B0)²``, so the upstream guide field gives 1.
        """
        return self.bx**2 + self.by**2 + self.bz**2

    def E_dot_B(self) -> np.ndarray:
        """Scalar E·B.

        In ion units: ``(E/E0)·(B/B0)``.  Zero for a pure ExB configuration.
        """
        return self.ex*self.bx + self.ey*self.by + self.ez*self.bz

    def ExB_drift(self) -> dict[str, np.ndarray]:
        """ExB drift velocity (E×B)/B², dict with keys 'x', 'y', 'z'.

        In ion units the result has units of E0/B0 = vAi/c, so an
        equilibrium drift at 0.01c gives 0.01/vAi_over_c (e.g. 2.5 for
        vAi = 0.004c).  Returns zero where B = 0.
        """
        b2 = self.B_squared()
        safe_b2 = np.where(b2 > 0, b2, np.inf)
        return {
            "x": (self.ey*self.bz - self.ez*self.by) / safe_b2,
            "y": (self.ez*self.bx - self.ex*self.bz) / safe_b2,
            "z": (self.ex*self.by - self.ey*self.bx) / safe_b2,
        }

    def B_hat(self) -> dict[str, np.ndarray]:
        """Magnetic field unit vector B/\|B\|, dict with keys 'x', 'y', 'z'.

        Dimensionless — identical in code and ion units since the B0
        normalisation cancels.  Returns zero where B = 0.
        """
        b_mag = np.sqrt(self.B_squared())
        safe_b = np.where(b_mag > 0, b_mag, np.inf)
        return {
            "x": self.bx / safe_b,
            "y": self.by / safe_b,
            "z": self.bz / safe_b,
        }

    def psi(self) -> np.ndarray:
        """Magnetic flux function ψ = A_z computed in the x-y plane.

        Uses a two-step integration with reference point ψ(0, 0) = 0:

        1. Integrate −By along x at y = 0:
               ψ(x, 0) = −∫₀ˣ By(x', 0) dx'
        2. Integrate Bx along y at each x:
               ψ(x, y) = ψ(x, 0) + ∫₀ʸ Bx(x, y') dy'

        For 3D arrays the integration is performed in the x-y plane
        (the last two axes of the field array).

        In code units ψ has dimensions ``[B_code × cell]``.
        In ion units ψ has dimensions ``[di]`` (since B → B/B0 and
        dx → cell_to_di).

        Assumes square cells (dx = dy), which Tristan-MP V2 always uses.
        """
        if self.units == "ion":
            if self.uc is None:
                raise ValueError(
                    "unit_converter is required for psi() in ion units."
                )
            dx = self.uc.cell_to_di
        else:
            dx = 1.0

        # Step 1: ψ(x, 0) — shifted cumsum of −By along x at y = 0
        by_y0 = self.by[..., 0:1, :]                                   # (..., 1, nx)
        by_y0_s = np.concatenate(
            [np.zeros_like(by_y0[..., :1]), by_y0[..., :-1]], axis=-1
        )
        psi_y0 = -np.cumsum(by_y0_s, axis=-1) * dx                     # (..., 1, nx)

        # Step 2: add ∫₀ʸ Bx dy — shifted cumsum of Bx along y
        bx_s = np.concatenate(
            [np.zeros_like(self.bx[..., :1, :]), self.bx[..., :-1, :]], axis=-2
        )
        return psi_y0 + np.cumsum(bx_s, axis=-2) * dx

    @property
    def jx(self) -> np.ndarray:
        """Current density x-component (code units)."""
        return self._load("jx")

    @property
    def jy(self) -> np.ndarray:
        """Current density y-component (code units)."""
        return self._load("jy")

    @property
    def jz(self) -> np.ndarray:
        """Current density z-component (code units)."""
        return self._load("jz")

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def b_magnitude(self) -> np.ndarray:
        """Total magnetic field magnitude \|B\|."""
        return np.sqrt(self.bx**2 + self.by**2 + self.bz**2)

    @property
    def e_magnitude(self) -> np.ndarray:
        """Total electric field magnitude \|E\|."""
        return np.sqrt(self.ex**2 + self.ey**2 + self.ez**2)

    def bulk_velocity(
        self,
        species_id: int,
    ) -> dict[str, np.ndarray]:
        """Bulk velocity of a species computed from the field-file current density.

        Uses the relation  v_k = (m_k / q_k) × jprt_k / dens_k  where
        ``jprtX{k}`` is the charge-current density and ``dens{k}`` is the
        mass density stored in the field file.

        Parameters
        ----------
        species_id : int
            Species index (1 = electrons, 2 = ions, etc.).

        Returns
        -------
        dict with keys 'vx', 'vy', 'vz'
            Bulk velocity arrays.  Units depend on ``self.units``:
            * ``'code'`` — units of c.
            * ``'ion'``  — units of vAi.

        Raises
        ------
        ValueError
            If species mass or charge cannot be determined.
        KeyError
            If the required field-file datasets are absent.
        """
        sid = species_id
        m_k = self._species_mass.get(sid)
        q_k = self._species_charge.get(sid)
        if m_k is None:
            raise ValueError(
                f"Mass for species {sid} not found. "
                "Provide a params file when constructing FieldSnapshot."
            )
        if q_k is None:
            raise ValueError(
                f"Charge for species {sid} not found. "
                "Provide a params file when constructing FieldSnapshot."
            )
        if q_k == 0:
            raise ValueError(f"Species {sid} has zero charge; bulk velocity undefined.")

        # Load raw (code-unit) current and density
        jx = self._load(f"jprtX{sid}")
        jy = self._load(f"jprtY{sid}")
        jz = self._load(f"jprtZ{sid}")
        dens = self._load(f"dens{sid}")

        # Safe inverse density (avoid dividing by zero in empty cells)
        inv_dens = np.where(dens > 0, 1.0 / np.where(dens > 0, dens, 1.0), 0.0)

        # v_k = (m_k / q_k) * j_k / dens_k    [units: c]
        factor = m_k / q_k
        vx = factor * jx * inv_dens
        vy = factor * jy * inv_dens
        vz = factor * jz * inv_dens

        if self.units == "ion" and self.uc is not None:
            vx = vx * self.uc.c_to_vAi
            vy = vy * self.uc.c_to_vAi
            vz = vz * self.uc.c_to_vAi

        return {"vx": vx, "vy": vy, "vz": vz}

    def stress_tensor(self, species_id: int) -> dict[str, np.ndarray]:
        """Stress tensor (and momentum density) for a species from the field file.

        Returns all nine moment-tensor components that Tristan stores per species:
        the three momentum-density components (T0X/Y/Z), the three diagonal
        pressure-tensor components (TXX/YY/ZZ), and the three off-diagonal
        shear-stress components (TXY/XZ/YZ).

        Only components present in the file are included; missing components are
        silently omitted (e.g. if the run was configured without ``write_Tij``).

        Parameters
        ----------
        species_id : int
            Species index (1 = electrons, 2 = ions, etc.).

        Returns
        -------
        dict with a subset of keys: 't0x', 't0y', 't0z',
            'txx', 'tyy', 'tzz', 'txy', 'txz', 'tyz'
            Arrays are normalized when ``units='ion'``:
            divided by ``n0 × mass_ratio`` and multiplied by ``c_to_vAi²``.

        Raises
        ------
        KeyError
            If none of the expected stress-tensor datasets are present.
        """
        components = {
            "t0x": f"T0X{species_id}",
            "t0y": f"T0Y{species_id}",
            "t0z": f"T0Z{species_id}",
            "txx": f"TXX{species_id}",
            "tyy": f"TYY{species_id}",
            "tzz": f"TZZ{species_id}",
            "txy": f"TXY{species_id}",
            "txz": f"TXZ{species_id}",
            "tyz": f"TYZ{species_id}",
        }
        result = {}
        for out_key, hdf_key in components.items():
            if hdf_key in self.keys:
                result[out_key] = self._load_field(hdf_key)
        if not result:
            raise KeyError(
                f"No stress-tensor datasets found for species {species_id} "
                f"in {self.filepath.name}.\n"
                f"Expected keys like 'TXX{species_id}', 'T0X{species_id}', etc.\n"
                f"Available fields: {self.keys}"
            )
        return result

    def heat_flux(self, species_id: int) -> dict[str, np.ndarray]:
        """Heat flux vector (third-order moment) for a species.

        Reads the ``QX{k}``, ``QY{k}``, ``QZ{k}`` datasets written by Tristan
        when ``write_Qi`` is enabled in the output configuration.

        Parameters
        ----------
        species_id : int
            Species index (1 = electrons, 2 = ions, etc.).

        Returns
        -------
        dict with keys 'qx', 'qy', 'qz'
            Arrays are normalized when ``units='ion'``:
            divided by ``n0 × mass_ratio`` and multiplied by ``c_to_vAi³``.

        Raises
        ------
        KeyError
            If the heat-flux datasets are not present in the file.
        """
        components = {
            "qx": f"QX{species_id}",
            "qy": f"QY{species_id}",
            "qz": f"QZ{species_id}",
        }
        result = {}
        for out_key, hdf_key in components.items():
            if hdf_key in self.keys:
                result[out_key] = self._load_field(hdf_key)
        if not result:
            raise KeyError(
                f"No heat-flux datasets found for species {species_id} "
                f"in {self.filepath.name}.\n"
                f"Expected keys like 'QX{species_id}', 'QY{species_id}', 'QZ{species_id}'.\n"
                f"Available fields: {self.keys}"
            )
        return result

    # ------------------------------------------------------------------
    # Bulk release / cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Release all cached arrays from memory."""
        self._cache.clear()

    def preload(self, keys: list[str] | None = None) -> None:
        """Eagerly load datasets into the cache.

        Parameters
        ----------
        keys : list of str, optional
            Specific field names to load.  Defaults to all available fields.
        """
        for key in (keys or self.keys):
            self._load(key)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        t = self.time
        t_str = f"{t:.2f}" if t is not None else "?"
        return (
            f"FieldSnapshot(step={self.step}, "
            f"time={t_str}/ωpe, "
            f"units='{self.units}', "
            f"file='{self.filepath.name}')"
        )


class FieldLoader:
    """Manages lazy access to multiple field snapshots.

    Parameters
    ----------
    filepaths : list of Path
        Sorted list of ``flds.tot.NNNNN`` paths.
    unit_converter : UnitConverter, optional
        Passed through to each ``FieldSnapshot``.
    params_map : dict[int, SimParams], optional
        Mapping from step number to SimParams, for time information.
    units : {'code', 'ion'}, optional
        Passed through to each ``FieldSnapshot``.
    """

    def __init__(
        self,
        filepaths: list[Path],
        unit_converter: UnitConverter | None = None,
        params_map: dict | None = None,
        units: str = "code",
    ) -> None:
        self._filepaths: dict[int, Path] = {}
        self.uc = unit_converter
        self._params_map = params_map or {}
        self._units = units
        self._snapshots: dict[int, FieldSnapshot] = {}

        for fp in filepaths:
            parts = fp.name.split(".")
            try:
                step = int(parts[-1])
                self._filepaths[step] = fp
            except (ValueError, IndexError):
                pass

    @property
    def steps(self) -> list[int]:
        """Sorted list of available step numbers."""
        return sorted(self._filepaths.keys())

    def __getitem__(self, step: int) -> FieldSnapshot:
        if step not in self._snapshots:
            if step not in self._filepaths:
                raise KeyError(
                    f"Field snapshot for step {step} not found. "
                    f"Available steps: {self.steps}"
                )
            self._snapshots[step] = FieldSnapshot(
                self._filepaths[step],
                unit_converter=self.uc,
                params=self._params_map.get(step),
                units=self._units,
            )
        return self._snapshots[step]

    def __len__(self) -> int:
        return len(self._filepaths)

    def __repr__(self) -> str:
        return f"FieldLoader(n_steps={len(self)}, steps={self.steps})"
