"""
Parameter reader for Tristan-MP V2 simulation output.

Params files are named ``params.NNNNN`` where NNNNN is the zero-padded
5-digit step number.  Each file is an HDF5 file where every dataset is a
scalar (or 1-element array) holding a simulation parameter.

Usage example::

    from pytrist.params import SimParams

    p = SimParams("/path/to/output/params.00001")
    print(p.c_omp, p.sigma, p.mass_ratio)
    print(p["ppc0"])          # dict-like access
    print(list(p.keys()))     # all available parameters
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Parameter name aliases
# Tristan-V2 occasionally uses different names for the same quantity across
# versions.  Map canonical names → list of alternatives to try.
# ---------------------------------------------------------------------------
_ALIASES: dict[str, list[str]] = {
    "mass_ratio": ["mi_me", "mass_ratio", "mi/me"],
    "c_omp": ["c_omp", "comp"],
    "sigma": ["sigma"],
    "CC": ["CC", "c", "speed_of_light"],
    "ppc0": ["ppc0", "ppc"],
    "time": ["time"],
    "output_interval": ["output/interval", "interval"],
}


class SimParams:
    """Dictionary-like container for Tristan-V2 simulation parameters.

    Parameters
    ----------
    filepath : str or Path
        Path to a ``params.NNNNN`` HDF5 file.

    Attributes
    ----------
    step : int
        Step number extracted from the filename suffix.
    filepath : Path
        Absolute path to the params file.
    """

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath).resolve()
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Params file not found: {self.filepath}\n"
                "Expected a file named like 'params.NNNNN' (5-digit step number)."
            )

        self._data: dict[str, Any] = {}
        self._load()

        # Extract step number from filename (e.g. "params.00042" → 42)
        suffix = self.filepath.suffix  # ".00042"
        try:
            self.step: int = int(suffix.lstrip("."))
        except ValueError:
            self.step = -1

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read all datasets from the HDF5 file into self._data."""
        with h5py.File(self.filepath, "r") as f:
            self._load_group(f, prefix="")

    def _load_group(self, group: h5py.Group, prefix: str) -> None:
        """Recursively load datasets from an HDF5 group."""
        for key in group.keys():
            item = group[key]
            full_key = f"{prefix}/{key}".lstrip("/") if prefix else key
            if isinstance(item, h5py.Dataset):
                val = item[()]
                # Unwrap 0-d or 1-element arrays to Python scalars
                if isinstance(val, np.ndarray):
                    if val.ndim == 0:
                        val = val.item()
                    elif val.size == 1:
                        val = val.flat[0]
                        if isinstance(val, (np.integer,)):
                            val = int(val)
                        elif isinstance(val, (np.floating,)):
                            val = float(val)
                elif isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                self._data[full_key] = val
            elif isinstance(item, h5py.Group):
                self._load_group(item, prefix=full_key)

    # ------------------------------------------------------------------
    # dict-like interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        if key in self._data:
            return self._data[key]
        # Try aliases
        for canonical, alternatives in _ALIASES.items():
            if key in alternatives:
                for alt in alternatives:
                    if alt in self._data:
                        return self._data[alt]
        raise KeyError(
            f"Parameter '{key}' not found in {self.filepath.name}. "
            f"Available keys: {list(self._data.keys())}"
        )

    def __contains__(self, key: object) -> bool:
        if key in self._data:
            return True
        if isinstance(key, str):
            for _, alternatives in _ALIASES.items():
                if key in alternatives:
                    return any(alt in self._data for alt in alternatives)
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Return value for *key*, or *default* if not present."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        """Return all raw parameter names."""
        return list(self._data.keys())

    def values(self):
        """Return all parameter values."""
        return self._data.values()

    def items(self):
        """Return (key, value) pairs."""
        return self._data.items()

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------
    # Convenience properties for commonly used parameters
    # ------------------------------------------------------------------

    def _get_alias(self, canonical: str) -> Any:
        """Look up a parameter using its canonical name and aliases."""
        alternatives = _ALIASES.get(canonical, [canonical])
        for alt in alternatives:
            if alt in self._data:
                return self._data[alt]
        raise AttributeError(
            f"Parameter '{canonical}' (aliases: {alternatives}) not found "
            f"in {self.filepath.name}."
        )

    @property
    def c_omp(self) -> float:
        """Electron skin depth in grid cells (de)."""
        return float(self._get_alias("c_omp"))

    @property
    def sigma(self) -> float:
        """Magnetisation parameter σ = ωce²/ωpe²."""
        return float(self._get_alias("sigma"))

    @property
    def mass_ratio(self) -> float:
        """Ion-to-electron mass ratio mi/me."""
        return float(self._get_alias("mass_ratio"))

    @property
    def CC(self) -> float:
        """Speed of light in code units."""
        return float(self._get_alias("CC"))

    @property
    def time(self) -> float:
        """Simulation time in units of 1/ωpe."""
        return float(self._get_alias("time"))

    @property
    def ppc0(self) -> int:
        """Initial particles per cell."""
        return int(self._get_alias("ppc0"))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SimParams(step={self.step}, "
            f"file='{self.filepath.name}', "
            f"n_params={len(self._data)})"
        )

    def summary(self) -> str:
        """Return a human-readable summary of key parameters."""
        lines = [
            f"SimParams — {self.filepath.name}  (step {self.step})",
            "─" * 50,
        ]
        for key in ("c_omp", "sigma", "mass_ratio", "CC", "ppc0", "time"):
            try:
                val = self._get_alias(key)
                lines.append(f"  {key:<20s} = {val}")
            except AttributeError:
                lines.append(f"  {key:<20s} = <not found>")
        lines.append(f"\n  Total parameters loaded: {len(self._data)}")
        return "\n".join(lines)
