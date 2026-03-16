"""
Field data reader for Tristan-MP V2 simulation output.

Field snapshots are stored in HDF5 files named ``flds.tot.NNNNN`` where
NNNNN is the zero-padded 5-digit step number.

Datasets present in a typical field file
-----------------------------------------
Electromagnetic:   ex, ey, ez, bx, by, bz, jx, jy, jz
Per species:       dens_1, dens_2, enrg_1, enrg_2, ...
                   (species index suffix varies by run)

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

# Fields that are normalised by B0 when units='ion'
_EM_KEYS: frozenset[str] = frozenset({"bx", "by", "bz", "ex", "ey", "ez"})

# Regex patterns for per-species moment fields (both dens1 and dens_1 style)
_DENS_RE = re.compile(r"^dens_?(\d+)$")          # dens1, dens_1, dens2, …
_VEL_RE = re.compile(r"^vel([xyz])_?(\d*)$")     # velx, velx1, velx_1, …
_STRESS_RE = re.compile(r"^T([XYZ0]{2})_?(\d+)$")  # TXX1, TYY2, TXZ1, T0X1, …


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

          - **B / E fields** (bx/by/bz/ex/ey/ez): divided by B0 → upstream By ≈ 1.
          - **Number density** (dens1, dens_1, …): divided by ``m_k × n0``
            (where ``n0 = ppc0/2``) → dimensionless, upstream ≈ 1.
          - **Bulk velocity** (velx, vely, velz, velx1, …): multiplied by
            ``c_to_vAi = √(mass_ratio/σ)`` → units of vAi.
          - **Stress tensor** (TXX1, TYY2, TXZ1, …): divided by ``m_k × n0``
            then multiplied by ``c_to_vAi²`` → units of n0 × vAi².
            Dividing a stress component by density gives temperature in vAi².
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

        # Extract n0 and per-species masses from params (used for ion-unit normalisation)
        self._n0: float | None = None
        self._species_mass: dict[int, float] = {}
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

        # EM fields → divide by B0 (upstream By ≈ 1)
        if key in _EM_KEYS:
            return self.uc.field_B(arr)

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

        # Stress tensor T{ij}{k} → divide by m_k × n0, convert c² → vAi²
        m = _STRESS_RE.match(key)
        if m:
            sid = int(m.group(2))
            m_k = self._species_mass.get(sid, 1.0)
            n0 = self._n0 if self._n0 is not None else 1.0
            return arr / (m_k * n0) * self.uc.c_to_vAi ** 2

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
        """Electric field x-component (normalised to B0 when units='ion')."""
        return self._load_field("ex")

    @property
    def ey(self) -> np.ndarray:
        """Electric field y-component (normalised to B0 when units='ion')."""
        return self._load_field("ey")

    @property
    def ez(self) -> np.ndarray:
        """Electric field z-component (normalised to B0 when units='ion')."""
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
        """Total magnetic field magnitude |B|."""
        return np.sqrt(self.bx**2 + self.by**2 + self.bz**2)

    @property
    def e_magnitude(self) -> np.ndarray:
        """Total electric field magnitude |E|."""
        return np.sqrt(self.ex**2 + self.ey**2 + self.ez**2)

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
