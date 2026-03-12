"""
Energy spectra reader for Tristan-MP V2 simulation output.

Spectra snapshots are stored in HDF5 files named ``spec.tot.NNNNN``.

Typical datasets in a spectra file
-------------------------------------
``xsl``     — energy bin edges or centres
``dens_sp1``, ``dens_sp2``  — number spectrum dN/dγ for each species
``crnt_sp1``, …             — current/energy-weighted spectrum
``spe_log`` (bool flag)     — whether x-axis is logarithmically spaced

Usage example::

    from pytrist.spectra import SpectraSnapshot

    snap = SpectraSnapshot("/path/to/spec.tot.00001")
    gamma = snap.gamma_bins      # energy axis (γ)
    dn_dgamma = snap.spectrum(1) # dN/dγ for species 1

    # With unit converter (currently no conversion needed for spectra,
    # but the UnitConverter is stored for future use)
    from pytrist.units import UnitConverter
    uc = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
    snap = SpectraSnapshot("/path/to/spec.tot.00001", unit_converter=uc)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .units import UnitConverter


class SpectraSnapshot:
    """Lazy-loading container for a single spectra snapshot.

    Parameters
    ----------
    filepath : str or Path
        Path to a ``spec.tot.NNNNN`` HDF5 file.
    unit_converter : UnitConverter, optional
        Stored for future use (energy spectra currently need no unit
        conversion, as γ is dimensionless).
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
                f"Spectra file not found: {self.filepath}\n"
                "Expected a file named like 'spec.tot.NNNNN'."
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
    # Energy axis
    # ------------------------------------------------------------------

    @property
    def gamma_bins(self) -> np.ndarray:
        """Energy bin centres (Lorentz factor γ).

        Tries ``xsl`` first, then ``gamma``, then ``e_bins``.
        Returns the raw array; whether it is bin edges or centres depends
        on the simulation version.
        """
        for candidate in ["xsl", "gamma", "e_bins", "e"]:
            if candidate in self.keys:
                return self._load(candidate)
        raise KeyError(
            f"No energy-axis dataset found in {self.filepath.name}. "
            f"Available keys: {self.keys}"
        )

    @property
    def log_spaced(self) -> bool:
        """True if the energy axis is logarithmically spaced."""
        if "spe_log" in self.keys:
            val = self._load("spe_log")
            return bool(val) if np.ndim(val) == 0 else bool(val.flat[0])
        # Guess from the energy axis
        try:
            x = self.gamma_bins
            if len(x) < 2:
                return False
            ratios = np.diff(x) / x[:-1]
            return float(np.std(ratios)) < 0.01 * float(np.mean(ratios))
        except KeyError:
            return False

    # ------------------------------------------------------------------
    # Species spectra
    # ------------------------------------------------------------------

    def _detect_species_keys(self) -> dict[int, str]:
        """Return mapping {species_id: dataset_name} for number spectra."""
        mapping: dict[int, str] = {}
        for key in self.keys:
            # Patterns: dens_sp1, dens_1, dens1, dn_sp1, sp1, …
            for prefix in ["dens_sp", "dens_", "dn_sp", "sp", "n_sp"]:
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    if suffix.isdigit():
                        mapping[int(suffix)] = key
                        break
        return mapping

    @property
    def species_ids(self) -> list[int]:
        """List of species IDs for which spectra are available."""
        return sorted(self._detect_species_keys().keys())

    def spectrum(self, species_id: int) -> np.ndarray:
        """Return the number spectrum dN/dγ for the requested species.

        Parameters
        ----------
        species_id : int
            1-based species index.

        Returns
        -------
        numpy.ndarray
            Spectral density array, same length as :attr:`gamma_bins`.
        """
        mapping = self._detect_species_keys()
        if species_id not in mapping:
            raise KeyError(
                f"No spectrum found for species {species_id} in "
                f"{self.filepath.name}.\n"
                f"Detected species: {sorted(mapping.keys())}\n"
                f"Available keys: {self.keys}"
            )
        return self._load(mapping[species_id])

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    @property
    def time(self) -> float | None:
        """Simulation time in 1/ωpe, or None if unavailable."""
        if "time" in self.keys:
            t = self._load("time")
            return float(t.flat[0]) if isinstance(t, np.ndarray) else float(t)
        if self._params is not None:
            try:
                return float(self._params.time)
            except AttributeError:
                pass
        return None

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Release all cached arrays from memory."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        t = self.time
        t_str = f"{t:.2f}" if t is not None else "?"
        return (
            f"SpectraSnapshot(step={self.step}, "
            f"time={t_str}/ωpe, "
            f"file='{self.filepath.name}')"
        )
