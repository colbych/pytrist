"""
Particle data reader for Tristan-MP V2 simulation output.

Particle snapshots are stored in HDF5 files named ``prtl.tot.NNNNN``.

Datasets in a typical particle file
-------------------------------------
x, y, z     — positions in grid cells
u, v, w     — 4-velocities (γβ) in units of c
wei         — macro-particle weight (number of physical particles per macro)
ind         — particle index within MPI tile
proc        — MPI rank that owns the particle
ch          — charge (±1)
inde        — unique index (optional)
Species distinction is typically indicated by separate datasets named
  x1, y1, z1, u1, v1, w1, wei1 … for species 1
  x2, y2, z2, u2, v2, w2, wei2 … for species 2
OR by a flat array with a companion integer dataset ``ind`` / ``proc``
used to split by species.  This reader handles both naming schemes.

Usage example::

    from pytrist.particles import ParticleSnapshot

    snap = ParticleSnapshot("/path/to/prtl.tot.00001")

    # Access species 1 particles
    sp1 = snap.species(1)
    print(sp1["x"], sp1["u"])

    # Access all particles (flat arrays, no species filtering)
    print(snap["x"], snap["wei"])

    # With unit converter
    from pytrist.units import UnitConverter
    uc = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
    snap = ParticleSnapshot("/path/to/prtl.tot.00001", unit_converter=uc)
    sp1_ion = snap.species(1, units="ion")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .units import UnitConverter

# Datasets that hold position information (in cells)
_POSITION_KEYS = {"x", "y", "z"}

# Datasets that hold velocity/momentum information (in units of c)
_VELOCITY_KEYS = {"u", "v", "w"}


class ParticleSnapshot:
    """Lazy-loading container for a single particle snapshot.

    Parameters
    ----------
    filepath : str or Path
        Path to a ``prtl.tot.NNNNN`` HDF5 file.
    unit_converter : UnitConverter, optional
        If provided, positions/velocities can be converted to ion units
        via the ``units`` argument of :meth:`species`.
    params : SimParams, optional
        Associated simulation parameters.
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
                f"Particle file not found: {self.filepath}\n"
                "Expected a file named like 'prtl.tot.NNNNN'."
            )
        self.uc = unit_converter
        self._params = params
        self._cache: dict[str, np.ndarray] = {}
        self._keys: list[str] | None = None

        parts = self.filepath.name.split(".")
        try:
            self.step: int = int(parts[-1])
        except (ValueError, IndexError):
            self.step = -1

    # ------------------------------------------------------------------
    # Key discovery
    # ------------------------------------------------------------------

    @property
    def keys(self) -> list[str]:
        """Names of all datasets in this file."""
        if self._keys is None:
            with h5py.File(self.filepath, "r") as f:
                self._keys = list(f.keys())
        return self._keys

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self, key: str) -> np.ndarray:
        if key not in self._cache:
            if key not in self.keys:
                raise KeyError(
                    f"Dataset '{key}' not found in {self.filepath.name}.\n"
                    f"Available datasets: {self.keys}"
                )
            with h5py.File(self.filepath, "r") as f:
                self._cache[key] = f[key][()]
        return self._cache[key]

    def __getitem__(self, key: str) -> np.ndarray:
        return self._load(key)

    # ------------------------------------------------------------------
    # Species detection
    # ------------------------------------------------------------------

    def _detect_naming_scheme(self) -> str:
        """Determine how species are encoded in this file.

        Returns
        -------
        str
            ``"suffixed"``  – datasets named x1, y1, u1 … x2, y2, u2 …
            ``"flat"``      – single x, y, u … arrays; species coded in
                              a companion integer array (``proc`` or ``ind``).
        """
        if "x1" in self.keys or "x_1" in self.keys:
            return "suffixed"
        return "flat"

    @property
    def n_species(self) -> int:
        """Inferred number of particle species."""
        if self._detect_naming_scheme() == "suffixed":
            count = 0
            for key in self.keys:
                # count keys of the form 'x1', 'x2', 'x10', …
                if key.startswith("x") and key[1:].isdigit():
                    count += 1
                elif key.startswith("x_") and key[2:].isdigit():
                    count += 1
            return count
        # Flat scheme: look for a 'ch' (charge) array or assume 2 species
        return 2  # conservative default

    def _suffixed_key(self, base: str, species_id: int) -> str:
        """Return the dataset name for *base* variable of species *species_id*.

        Tries both ``base + str(id)`` and ``base + "_" + str(id)``.
        """
        k1 = f"{base}{species_id}"
        k2 = f"{base}_{species_id}"
        if k1 in self.keys:
            return k1
        if k2 in self.keys:
            return k2
        raise KeyError(
            f"No dataset found for species {species_id}, variable '{base}'.\n"
            f"Tried '{k1}' and '{k2}'.\n"
            f"Available keys: {self.keys}"
        )

    # ------------------------------------------------------------------
    # Species access
    # ------------------------------------------------------------------

    def species(
        self,
        species_id: int,
        units: str = "code",
    ) -> dict[str, np.ndarray]:
        """Return a dictionary of arrays for the requested species.

        Parameters
        ----------
        species_id : int
            Species index (1-based, as used in Tristan-V2 naming).
        units : {'code', 'ion'}
            Whether to return data in code units or ion units.  If
            ``'ion'`` but no ``UnitConverter`` was provided at construction
            time, a ``ValueError`` is raised.

        Returns
        -------
        dict
            Keys: ``x, y, z`` (positions), ``u, v, w`` (4-velocities),
            ``wei`` (weights), plus any extra datasets present for this
            species.
        """
        if units == "ion" and self.uc is None:
            raise ValueError(
                "unit_converter must be provided to return data in ion units."
            )

        scheme = self._detect_naming_scheme()
        result: dict[str, np.ndarray] = {}

        if scheme == "suffixed":
            # Datasets are named x1, y1, z1, u1, v1, w1, wei1, …
            base_keys = ["x", "y", "z", "u", "v", "w", "wei", "ch", "ind", "proc"]
            for base in base_keys:
                try:
                    result[base] = self._load(self._suffixed_key(base, species_id))
                except KeyError:
                    pass  # Optional keys may not be present for every species
        else:
            # Flat scheme: all species share the same arrays.
            # Attempt to split by a species-indicator array.
            split_arr = self._find_species_indicator()
            mask = split_arr == species_id if split_arr is not None else None

            for key in ["x", "y", "z", "u", "v", "w", "wei", "ch", "ind", "proc"]:
                if key not in self.keys:
                    continue
                arr = self._load(key)
                result[key] = arr[mask] if mask is not None else arr

        # Apply unit conversions
        if units == "ion" and self.uc is not None:
            for key in list(result.keys()):
                if key in _POSITION_KEYS:
                    result[key] = self.uc.length(result[key])
                elif key in _VELOCITY_KEYS:
                    result[key] = self.uc.speed(result[key])

        return result

    def _find_species_indicator(self) -> np.ndarray | None:
        """Try to find an integer array that indexes particle species.

        Looks for 'proc', 'ind', 'ch', or 'sp' datasets.
        """
        # 'ch' holds ±1 charge; electrons are -1 (species 1 by convention),
        # ions are +1 (species 2).  Remap to 1-based species IDs.
        if "ch" in self.keys:
            ch = self._load("ch")
            # Map: -1 → 1 (electrons), +1 → 2 (ions)
            sp = np.where(ch < 0, 1, 2)
            return sp
        if "sp" in self.keys:
            return self._load("sp")
        # Cannot determine species from flat arrays; return None
        return None

    # ------------------------------------------------------------------
    # Direct array access (all particles, no species split)
    # ------------------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        return self._load("x")

    @property
    def y(self) -> np.ndarray:
        return self._load("y")

    @property
    def z(self) -> np.ndarray:
        return self._load("z")

    @property
    def u(self) -> np.ndarray:
        return self._load("u")

    @property
    def v(self) -> np.ndarray:
        return self._load("v")

    @property
    def w(self) -> np.ndarray:
        return self._load("w")

    @property
    def wei(self) -> np.ndarray:
        return self._load("wei")

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def gamma(self, species_id: int | None = None) -> np.ndarray:
        """Lorentz factor γ = √(1 + u² + v² + w²).

        Parameters
        ----------
        species_id : int, optional
            If given, compute γ only for that species.
        """
        if species_id is not None:
            sp = self.species(species_id, units="code")
            u, v, w = sp.get("u"), sp.get("v"), sp.get("w")
        else:
            u = self._load("u") if "u" in self.keys else None
            v = self._load("v") if "v" in self.keys else None
            w = self._load("w") if "w" in self.keys else None

        if u is None:
            raise ValueError("Velocity dataset 'u' not found.")
        u2 = u**2
        v2 = v**2 if v is not None else 0.0
        w2 = w**2 if w is not None else 0.0
        return np.sqrt(1.0 + u2 + v2 + w2)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Release all cached arrays from memory."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = None
        if "x" in self.keys:
            try:
                n = len(self._load("x"))
            except Exception:
                pass
        n_str = str(n) if n is not None else "?"
        return (
            f"ParticleSnapshot(step={self.step}, "
            f"n_particles≈{n_str}, "
            f"file='{self.filepath.name}')"
        )
