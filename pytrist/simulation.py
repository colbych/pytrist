"""
Primary entry point for reading Tristan-MP V2 simulation output.

Typical usage::

    import pytrist

    sim = pytrist.Simulation("/path/to/output/")

    # Inspect available steps
    print(sim.steps)          # [1, 2, 3, …]
    print(sim.times)          # corresponding times in 1/ωpe

    # Unit converter built from simulation parameters
    uc = sim.unit_converter

    # Read parameters for the last step
    p = sim.params()          # SimParams object
    print(p.sigma, p.c_omp)

    # Read field snapshot
    flds = sim.fields(step=10)
    bx = flds.bx              # numpy array

    # Read particles
    prtl = sim.particles(step=10)
    sp1 = prtl.species(1)     # dict of arrays for electrons

    # Read history
    hist = sim.history()
    t = hist["t"]

    # One-liner field access with optional unit conversion
    bx_ion = sim.read_field("bx", step=10, units="ion")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .fields import FieldSnapshot
from .history import History
from .moments import ParticleMoments
from .params import SimParams
from .particles import ParticleSnapshot
from .spectra import SpectraSnapshot
from .units import UnitConverter

# ---------------------------------------------------------------------------
# File-pattern helpers
# ---------------------------------------------------------------------------

# Regex matching a 5-digit zero-padded step suffix: "00042"
_STEP_RE = re.compile(r"^(\d{5})$")

# Subdirectory names Tristan-V2 may use for each output type
_SUBDIRS = {
    "flds": ["flds", "fields", "."],
    "prtl": ["prtl", "particles", "."],
    "spec": ["spec", "spectra", "."],
    "params": [".", "params"],
}


def _find_files(
    output_dir: Path,
    prefix: str,
    subdirs: list[str],
) -> list[Path]:
    """Search *output_dir* (and optional subdirectories) for files matching
    ``prefix.NNNNN`` and return them sorted by step number.

    Parameters
    ----------
    output_dir : Path
        Root simulation output directory.
    prefix : str
        File prefix, e.g. ``"flds.tot"``, ``"prtl.tot"``, ``"params"``.
    subdirs : list[str]
        Subdirectory names to search, in priority order.
        Use ``"."`` to search the root *output_dir* directly.
    """
    found: dict[int, Path] = {}

    search_dirs: list[Path] = []
    for sd in subdirs:
        candidate = output_dir if sd == "." else output_dir / sd
        if candidate.is_dir():
            search_dirs.append(candidate)

    for search_dir in search_dirs:
        for fp in search_dir.iterdir():
            if not fp.is_file():
                continue
            name = fp.name
            # File must start with the prefix followed by a dot
            if not name.startswith(prefix + "."):
                continue
            suffix = name[len(prefix) + 1:]  # characters after "prefix."
            m = _STEP_RE.match(suffix)
            if m:
                step = int(m.group(1))
                # Prefer root-level files over subdir files (first found wins)
                if step not in found:
                    found[step] = fp

    return [found[s] for s in sorted(found.keys())]


# ---------------------------------------------------------------------------
# Main Simulation class
# ---------------------------------------------------------------------------


class Simulation:
    """Primary entry point for a Tristan-MP V2 simulation output directory.

    Parameters
    ----------
    output_dir : str or Path
        Path to the directory containing simulation output files.
    params_step : int or None
        Which step to use when building the :attr:`unit_converter`.
        Defaults to the last available step.
    """

    def __init__(
        self,
        output_dir: str | Path,
        params_step: int | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        if not self.output_dir.is_dir():
            raise NotADirectoryError(
                f"Output directory not found: {self.output_dir}"
            )

        self._params_step = params_step  # step used for unit conversion

        # Lazy caches
        self._fld_paths: list[Path] | None = None
        self._prtl_paths: list[Path] | None = None
        self._spec_paths: list[Path] | None = None
        self._param_paths: list[Path] | None = None

        self._params_cache: dict[int, SimParams] = {}
        self._fields_cache: dict[int, FieldSnapshot] = {}
        self._particles_cache: dict[int, ParticleSnapshot] = {}
        self._spectra_cache: dict[int, SpectraSnapshot] = {}
        self._moments_cache: dict[int, ParticleMoments] = {}
        self._history_cache: History | None = None
        self._unit_converter_cache: UnitConverter | None = None

    # ------------------------------------------------------------------
    # Internal file discovery
    # ------------------------------------------------------------------

    def _scan_flds(self) -> list[Path]:
        if self._fld_paths is None:
            self._fld_paths = _find_files(
                self.output_dir, "flds.tot", _SUBDIRS["flds"]
            )
        return self._fld_paths

    def _scan_prtl(self) -> list[Path]:
        if self._prtl_paths is None:
            self._prtl_paths = _find_files(
                self.output_dir, "prtl.tot", _SUBDIRS["prtl"]
            )
        return self._prtl_paths

    def _scan_spec(self) -> list[Path]:
        if self._spec_paths is None:
            self._spec_paths = _find_files(
                self.output_dir, "spec.tot", _SUBDIRS["spec"]
            )
        return self._spec_paths

    def _scan_params(self) -> list[Path]:
        if self._param_paths is None:
            self._param_paths = _find_files(
                self.output_dir, "params", _SUBDIRS["params"]
            )
        return self._param_paths

    def _step_from_path(self, fp: Path) -> int:
        suffix = fp.name.split(".")[-1]
        return int(suffix)

    # ------------------------------------------------------------------
    # Steps & times
    # ------------------------------------------------------------------

    @property
    def steps(self) -> list[int]:
        """Sorted list of output step numbers detected from field files.

        Falls back to particle or params files if no field files are found.
        """
        fld_steps = [self._step_from_path(p) for p in self._scan_flds()]
        if fld_steps:
            return fld_steps
        prtl_steps = [self._step_from_path(p) for p in self._scan_prtl()]
        if prtl_steps:
            return prtl_steps
        param_steps = [self._step_from_path(p) for p in self._scan_params()]
        return param_steps

    @property
    def times(self) -> list[float]:
        """Simulation times (1/ωpe) for each step in :attr:`steps`.

        Reads the ``time`` parameter from each params file.  Returns NaN
        for steps where the params file is not available.
        """
        result: list[float] = []
        for step in self.steps:
            try:
                p = self.params(step)
                result.append(p.time)
            except (FileNotFoundError, KeyError, AttributeError):
                result.append(float("nan"))
        return result

    # ------------------------------------------------------------------
    # Unit converter
    # ------------------------------------------------------------------

    @property
    def unit_converter(self) -> UnitConverter:
        """UnitConverter built from simulation parameters.

        Uses the step specified by ``params_step`` at construction time,
        defaulting to the last available params step.
        """
        if self._unit_converter_cache is not None:
            return self._unit_converter_cache

        param_steps = [self._step_from_path(p) for p in self._scan_params()]
        if not param_steps:
            raise FileNotFoundError(
                f"No params files found in {self.output_dir}. "
                "Cannot build UnitConverter without simulation parameters."
            )

        step = self._params_step if self._params_step is not None else param_steps[-1]
        p = self.params(step)

        self._unit_converter_cache = UnitConverter(
            c_omp=p.c_omp,
            sigma=p.sigma,
            mass_ratio=p.mass_ratio,
            CC=p.CC,
        )
        return self._unit_converter_cache

    # ------------------------------------------------------------------
    # Reader methods
    # ------------------------------------------------------------------

    def params(self, step: int | None = None) -> SimParams:
        """Return SimParams for *step* (default: last available step).

        Parameters
        ----------
        step : int, optional
            Step number.  Defaults to the last available params step.
        """
        param_paths = self._scan_params()
        if not param_paths:
            raise FileNotFoundError(
                f"No params files found in {self.output_dir} or its "
                f"subdirectories.  Expected files named 'params.NNNNN'."
            )
        step_to_path = {self._step_from_path(p): p for p in param_paths}

        if step is None:
            step = max(step_to_path.keys())

        if step not in step_to_path:
            raise KeyError(
                f"Params file for step {step} not found. "
                f"Available steps: {sorted(step_to_path.keys())}"
            )

        if step not in self._params_cache:
            self._params_cache[step] = SimParams(step_to_path[step])
        return self._params_cache[step]

    def fields(self, step: int) -> FieldSnapshot:
        """Return a FieldSnapshot for *step*.

        The unit converter is always attached so that ``snap.time_ion``
        and manual ``sim.unit_converter.length(arr)`` calls work without
        needing to reload the snapshot.  Raw arrays are always in code
        units; use :attr:`unit_converter` to convert them.

        Parameters
        ----------
        step : int
            Step number.
        """
        if step not in self._fields_cache:
            fld_paths = {self._step_from_path(p): p for p in self._scan_flds()}
            if step not in fld_paths:
                raise KeyError(
                    f"Field snapshot for step {step} not found. "
                    f"Available steps: {sorted(fld_paths.keys())}"
                )
            try:
                uc = self.unit_converter
            except FileNotFoundError:
                uc = None
            try:
                p = self.params(step)
            except (FileNotFoundError, KeyError):
                p = None
            self._fields_cache[step] = FieldSnapshot(
                fld_paths[step], unit_converter=uc, params=p
            )
        return self._fields_cache[step]

    def particles(self, step: int) -> ParticleSnapshot:
        """Return a ParticleSnapshot for *step*.

        The unit converter is always attached so that
        ``prtl.species(1, units='ion')`` works without needing to reload.
        Raw arrays are always in code units.

        Parameters
        ----------
        step : int
            Step number.
        """
        if step not in self._particles_cache:
            prtl_paths = {self._step_from_path(p): p for p in self._scan_prtl()}
            if step not in prtl_paths:
                raise KeyError(
                    f"Particle snapshot for step {step} not found. "
                    f"Available steps: {sorted(prtl_paths.keys())}"
                )
            try:
                uc = self.unit_converter
            except FileNotFoundError:
                uc = None
            try:
                p = self.params(step)
            except (FileNotFoundError, KeyError):
                p = None
            self._particles_cache[step] = ParticleSnapshot(
                prtl_paths[step], unit_converter=uc, params=p
            )
        return self._particles_cache[step]

    def spectra(
        self,
        step: int,
    ) -> SpectraSnapshot:
        """Return a SpectraSnapshot for *step*.

        Parameters
        ----------
        step : int
            Step number.
        """
        if step not in self._spectra_cache:
            spec_paths = {self._step_from_path(p): p for p in self._scan_spec()}
            if step not in spec_paths:
                raise KeyError(
                    f"Spectra snapshot for step {step} not found. "
                    f"Available steps: {sorted(spec_paths.keys())}"
                )
            try:
                p = self.params(step)
            except (FileNotFoundError, KeyError):
                p = None
            self._spectra_cache[step] = SpectraSnapshot(
                spec_paths[step], unit_converter=None, params=p
            )
        return self._spectra_cache[step]

    def history(self) -> History:
        """Return the History object for this simulation.

        Searches for a file named ``history`` in the output directory.
        """
        if self._history_cache is not None:
            return self._history_cache

        # Try common locations
        candidates = [
            self.output_dir / "history",
            self.output_dir / "history.txt",
        ]
        for fp in candidates:
            if fp.exists():
                try:
                    uc = self.unit_converter
                except FileNotFoundError:
                    uc = None
                self._history_cache = History(fp, unit_converter=uc)
                return self._history_cache

        raise FileNotFoundError(
            f"History file not found in {self.output_dir}. "
            "Tried: " + ", ".join(str(c) for c in candidates)
        )

    def moments(
        self,
        step: int,
        region: tuple[int, int, int, int] | None = None,
    ) -> ParticleMoments:
        """Return a :class:`~pytrist.moments.ParticleMoments` for *step*.

        Full-box requests are cached; sub-region requests are not.

        Parameters
        ----------
        step : int
            Step number.
        region : tuple of int, optional
            ``(x0, x1, y0, y1)`` in cell indices (half-open).
            Defaults to the full simulation box.
        """
        cache_key: int | None = step if region is None else None
        if cache_key is not None and cache_key in self._moments_cache:
            return self._moments_cache[cache_key]

        prtl = self.particles(step)
        try:
            uc = self.unit_converter
        except FileNotFoundError:
            uc = None
        try:
            p = self.params(step)
        except (FileNotFoundError, KeyError):
            p = None

        obj = ParticleMoments(prtl, params=p, unit_converter=uc, region=region)
        if cache_key is not None:
            self._moments_cache[cache_key] = obj
        return obj

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def read_field(
        self,
        name: str,
        step: int,
        units: str = "code",
    ) -> np.ndarray:
        """Load a single field array, optionally converting to ion units.

        Parameters
        ----------
        name : str
            Field name, e.g. ``'bx'``, ``'dens_1'``.
        step : int
            Step number.
        units : {'code', 'ion'}
            ``'code'`` returns raw code-unit values.
            ``'ion'`` converts the array:
              - B-field components → :meth:`UnitConverter.field_B`
              - E-field components → :meth:`UnitConverter.field_E`
              - Density/other      → returned as-is (no conversion defined)

        Returns
        -------
        numpy.ndarray
        """
        snap = self.fields(step)
        arr = snap[name]

        if units == "ion":
            uc = self.unit_converter
            n = name.lower().lstrip("_")
            if n.startswith(("bx", "by", "bz")):
                arr = uc.field_B(arr)
            elif n.startswith(("ex", "ey", "ez")):
                arr = uc.field_E(arr)
            # Density, energy, current: no conversion defined by default

        return arr

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the simulation."""
        lines = [
            f"Simulation: {self.output_dir}",
            "─" * 60,
        ]
        steps = self.steps
        lines.append(f"  Available steps    : {len(steps)}")
        if steps:
            lines.append(f"  Step range         : {steps[0]} – {steps[-1]}")

        # Counts per type
        n_flds = len(self._scan_flds())
        n_prtl = len(self._scan_prtl())
        n_spec = len(self._scan_spec())
        n_params = len(self._scan_params())
        lines.append(f"  Field files        : {n_flds}")
        lines.append(f"  Particle files     : {n_prtl}")
        lines.append(f"  Spectra files      : {n_spec}")
        lines.append(f"  Params files       : {n_params}")

        # Unit converter info (only if params available)
        try:
            uc = self.unit_converter
            lines.append("")
            lines.append(uc.summary())
        except FileNotFoundError:
            lines.append("\n  (No params files — unit converter unavailable)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        n = len(self.steps)
        return (
            f"Simulation(dir='{self.output_dir.name}', "
            f"n_steps={n})"
        )
