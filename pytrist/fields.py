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

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .units import UnitConverter


class FieldSnapshot:
    """Lazy-loading container for a single field snapshot.

    Parameters
    ----------
    filepath : str or Path
        Path to a ``flds.tot.NNNNN`` HDF5 file.
    unit_converter : UnitConverter, optional
        If provided, field arrays will be returned in ion units.
        (For EM fields the numerical values are unchanged; for derived
        quantities such as density the converter is available via
        ``self.uc``.)
    params : SimParams, optional
        Associated simulation parameters snapshot.  Used to populate
        the ``time`` property when the field file itself does not store
        a time attribute.
    """

    def __init__(
        self,
        filepath: str | Path,
        unit_converter: UnitConverter | None = None,
        params=None,
    ) -> None:
        self.filepath = Path(filepath).resolve()
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Field file not found: {self.filepath}\n"
                "Expected a file named like 'flds.tot.NNNNN'."
            )
        self.uc = unit_converter
        self._params = params

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

    def __getitem__(self, key: str) -> np.ndarray:
        return self._load(key)

    def __getattr__(self, key: str) -> np.ndarray:
        # Avoid infinite recursion for private / dunder attributes
        if key.startswith("_") or key in ("filepath", "uc", "step", "keys"):
            raise AttributeError(key)
        # Only intercept if key looks like a field name (no dots)
        if "." not in key:
            try:
                return self._load(key)
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
        """Electric field x-component (code units)."""
        return self._load("ex")

    @property
    def ey(self) -> np.ndarray:
        """Electric field y-component (code units)."""
        return self._load("ey")

    @property
    def ez(self) -> np.ndarray:
        """Electric field z-component (code units)."""
        return self._load("ez")

    @property
    def bx(self) -> np.ndarray:
        """Magnetic field x-component (code units, normalised to B0)."""
        return self._load("bx")

    @property
    def by(self) -> np.ndarray:
        """Magnetic field y-component (code units, normalised to B0)."""
        return self._load("by")

    @property
    def bz(self) -> np.ndarray:
        """Magnetic field z-component (code units, normalised to B0)."""
        return self._load("bz")

    @property
    def jx(self) -> np.ndarray:
        """Current density x-component."""
        return self._load("jx")

    @property
    def jy(self) -> np.ndarray:
        """Current density y-component."""
        return self._load("jy")

    @property
    def jz(self) -> np.ndarray:
        """Current density z-component."""
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
    """

    def __init__(
        self,
        filepaths: list[Path],
        unit_converter: UnitConverter | None = None,
        params_map: dict | None = None,
    ) -> None:
        self._filepaths: dict[int, Path] = {}
        self.uc = unit_converter
        self._params_map = params_map or {}
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
            )
        return self._snapshots[step]

    def __len__(self) -> int:
        return len(self._filepaths)

    def __repr__(self) -> str:
        return f"FieldLoader(n_steps={len(self)}, steps={self.steps})"
