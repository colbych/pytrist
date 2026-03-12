"""
History file reader for Tristan-MP V2 simulation output.

The ``history`` file is a tab-separated text file with a single header line
and one data row per output step.  Column names are read from the header.

Typical columns (names vary by Tristan version/configuration):
  - ``t``          or ``time``  — simulation time (1/ωpe)
  - ``lx``, ``ly``             — box size (cells)
  - ``totKinE``    or similar  — total kinetic energy
  - ``totEmE``                 — total EM energy
  - ``bx_max``, ``by_max``, …  — field extrema

Usage example::

    from pytrist.history import History

    hist = History("/path/to/output/history")
    print(hist.column_names)

    t   = hist["t"]          # simulation time array
    eke = hist["totKinE"]

    # Convert time axis to ion units
    from pytrist.units import UnitConverter
    uc = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
    hist = History("/path/to/output/history", unit_converter=uc)
    t_ion = hist.time_ion    # time in 1/Ωci_y
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .units import UnitConverter

# Common names for the time column
_TIME_COLUMN_CANDIDATES = ["t", "time", "step", "it"]


class History:
    """Container for Tristan-V2 history (time-series) data.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``history`` file.
    unit_converter : UnitConverter, optional
        If provided, enables :attr:`time_ion`.
    comment_char : str
        Lines beginning with this character are treated as comments and
        skipped.  Default ``'#'``.
    """

    def __init__(
        self,
        filepath: str | Path,
        unit_converter: UnitConverter | None = None,
        comment_char: str = "#",
    ) -> None:
        self.filepath = Path(filepath).resolve()
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"History file not found: {self.filepath}\n"
                "Expected a file named 'history' in the simulation output directory."
            )
        self.uc = unit_converter
        self._comment_char = comment_char
        self._data: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return
        self._data = self._parse()

    def _parse(self) -> dict[str, np.ndarray]:
        """Parse the history file and return a dict of column arrays."""
        lines: list[str] = []
        header: list[str] | None = None

        with open(self.filepath, "r") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(self._comment_char):
                    continue
                if header is None:
                    # First non-comment line is the header
                    header = self._parse_header(line)
                    continue
                lines.append(line)

        if header is None:
            raise ValueError(
                f"Could not find a header line in {self.filepath}. "
                "Expected a tab/space-separated line with column names."
            )

        if not lines:
            # Return empty arrays
            return {col: np.array([]) for col in header}

        # Build 2-D float array from data lines
        rows: list[list[float]] = []
        for line in lines:
            try:
                row = [float(v) for v in line.split()]
            except ValueError:
                # Skip malformed lines
                continue
            rows.append(row)

        if not rows:
            return {col: np.array([]) for col in header}

        arr = np.array(rows, dtype=float)  # shape (n_rows, n_cols)
        n_cols = min(len(header), arr.shape[1])

        data: dict[str, np.ndarray] = {}
        for i, col in enumerate(header[:n_cols]):
            data[col] = arr[:, i]

        # If more columns than header names, store extras with generated names
        for i in range(n_cols, arr.shape[1]):
            data[f"col_{i}"] = arr[:, i]

        return data

    @staticmethod
    def _parse_header(line: str) -> list[str]:
        """Split a header line into column names.

        Handles tab-separated and space-separated formats; strips leading
        ``#`` if present.
        """
        line = line.lstrip("#").strip()
        if "\t" in line:
            cols = [c.strip() for c in line.split("\t")]
        else:
            cols = line.split()
        return [c for c in cols if c]  # drop empty strings

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        """Names of all columns in the history file."""
        self._ensure_loaded()
        return list(self._data.keys())

    @property
    def n_steps(self) -> int:
        """Number of data rows (time steps) in the file."""
        self._ensure_loaded()
        if not self._data:
            return 0
        col = next(iter(self._data.values()))
        return len(col)

    def __getitem__(self, key: str) -> np.ndarray:
        self._ensure_loaded()
        if key not in self._data:
            raise KeyError(
                f"Column '{key}' not found in {self.filepath.name}.\n"
                f"Available columns: {self.column_names}"
            )
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        self._ensure_loaded()
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        self._ensure_loaded()
        return iter(self._data)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._data)

    def get(self, key: str, default=None):
        """Return column *key* or *default* if not present."""
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        """Iterate over (column_name, array) pairs."""
        self._ensure_loaded()
        return self._data.items()

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    @property
    def _time_key(self) -> str | None:
        """Name of the time column, or None if not found."""
        self._ensure_loaded()
        for candidate in _TIME_COLUMN_CANDIDATES:
            if candidate in self._data:
                return candidate
        return None

    @property
    def time(self) -> np.ndarray | None:
        """Simulation time column in code units (1/ωpe).

        Returns ``None`` if no time column is detected.
        """
        key = self._time_key
        if key is None:
            return None
        return self._data[key]

    @property
    def time_ion(self) -> np.ndarray | None:
        """Time column converted to ion cyclotron periods (1/Ωci_y).

        Returns ``None`` if no UnitConverter or no time column.
        """
        if self.uc is None or self.time is None:
            return None
        return self.uc.time(self.time)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            n = self.n_steps
            cols = self.column_names
        except Exception:
            n, cols = "?", []
        return (
            f"History(n_steps={n}, "
            f"columns={cols}, "
            f"file='{self.filepath.name}')"
        )
