"""
Fluid moments computed from particle data for Tristan-MP V2 simulations.

Moments are computed by binning macro-particles onto the simulation grid
using a nearest-grid-point (NGP) scheme (``np.histogram2d``).  Results will
differ slightly from Tristan's own field outputs (``dens1``, ``TXX1``, …),
which use a cloud-in-cell (CIC) deposit with spatial smoothing, but the
two should agree to within ~5–20% depending on the quantity and the spatial
scale being averaged over.

Typical usage::

    import pytrist

    sim  = pytrist.Simulation("/path/to/output/")
    moms = sim.moments(step=10)

    n_e  = moms.density(1)                   # (ny, nx), dimensionless, ≈1 upstream
    V    = moms.bulk_velocity(2)             # dict with 'vx','vy','vz' in c
    T    = moms.temperature(1)               # (ny, nx), c²
    P    = moms.temperature_tensor(1)        # dict with 'xx','yy',… in c²
    q    = moms.heat_flux(1)                 # dict with 'x','y','z' in c³

    # Ion units
    T_ion = moms.temperature(1, units='ion') # c² → vAi²

    # Sub-region
    moms_cs = sim.moments(step=10, region=(100, 300, 350, 450))
    T_cs = moms_cs.temperature(1)            # shape (100, 200)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .units import UnitConverter

if TYPE_CHECKING:
    from .particles import ParticleSnapshot


def _get_stride(params) -> int:
    """Extract the particle output stride from a SimParams object."""
    for key in ("output:stride", "output/stride", "stride"):
        try:
            val = params[key]
            if val is not None:
                return int(val)
        except (KeyError, TypeError, AttributeError):
            pass
    return 1


def _get_n0(params) -> float | None:
    """Return ppc0/2 from params, or None if unavailable."""
    for key in ("plasma:ppc0", "ppc0", "ppc"):
        try:
            val = params[key]
            if val is not None:
                return float(val) / 2.0
        except (KeyError, TypeError, AttributeError):
            pass
    return None


class ParticleMoments:
    """Fluid moments computed from a :class:`~pytrist.particles.ParticleSnapshot`.

    Moments are computed lazily on first access and cached.  All quantities
    are returned as 2-D arrays of shape ``(ny_region, nx_region)`` matching
    the field-file convention ``(nz, ny, nx)`` with ``nz=1`` suppressed.

    Parameters
    ----------
    particle_snapshot : ParticleSnapshot
        Source particle data.  Positions must be in code units (cells).
    params : SimParams, optional
        Simulation parameters.  Used to read the grid size, output stride,
        and reference density n0 = ppc0/2.
    unit_converter : UnitConverter, optional
        Required for ``units='ion'`` conversions.
    region : tuple of int, optional
        Sub-region ``(x0, x1, y0, y1)`` in cell indices, half-open
        ``[x0, x1) × [y0, y1)``.  Defaults to the full box.
    n0 : float, optional
        Reference single-species density (ppc0/2).  If not provided,
        read from params.  Required for moment computation.
    """

    def __init__(
        self,
        particle_snapshot: "ParticleSnapshot",
        params=None,
        unit_converter: UnitConverter | None = None,
        region: tuple[int, int, int, int] | None = None,
        n0: float | None = None,
    ) -> None:
        self._prtl = particle_snapshot
        self._params = params
        self.uc = unit_converter
        self._cache: dict[tuple, np.ndarray] = {}

        self._stride = _get_stride(params)
        if n0 is not None:
            self._n0: float | None = float(n0)
        else:
            self._n0 = _get_n0(params)
        nx, ny = self._read_grid_size()
        self._nx_full = nx
        self._ny_full = ny

        if region is None:
            self._x0, self._x1 = 0, nx
            self._y0, self._y1 = 0, ny
        else:
            self._x0, self._x1, self._y0, self._y1 = region

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _read_grid_size(self) -> tuple[int, int]:
        """Return (nx, ny) from params or infer from particle positions."""
        p = self._params
        if p is not None:
            for nx_key in ("grid:mx0", "grid:nx", "nx"):
                for ny_key in ("grid:my0", "grid:ny", "ny"):
                    try:
                        nx = int(p[nx_key])
                        ny = int(p[ny_key])
                        return nx, ny
                    except (KeyError, TypeError, AttributeError):
                        pass

        # Infer from particle positions (last resort)
        sp = self._prtl.species(1, units="code")
        x = sp.get("x")
        y = sp.get("y")
        if x is not None and y is not None:
            return int(np.ceil(x.max())) + 1, int(np.ceil(y.max())) + 1
        raise ValueError(
            "Cannot determine grid size: no params provided and no particles found."
        )

    @property
    def x_edges(self) -> np.ndarray:
        """Cell-index bin edges in x, shape ``(nx_region + 1,)``."""
        return np.arange(self._x0, self._x1 + 1, dtype=float)

    @property
    def y_edges(self) -> np.ndarray:
        """Cell-index bin edges in y, shape ``(ny_region + 1,)``."""
        return np.arange(self._y0, self._y1 + 1, dtype=float)

    @property
    def x_centers(self) -> np.ndarray:
        """Cell-centre positions in x, shape ``(nx_region,)``."""
        return np.arange(self._x0, self._x1, dtype=float) + 0.5

    @property
    def y_centers(self) -> np.ndarray:
        """Cell-centre positions in y, shape ``(ny_region,)``."""
        return np.arange(self._y0, self._y1, dtype=float) + 0.5

    # ------------------------------------------------------------------
    # Core moment engine
    # ------------------------------------------------------------------

    def _compute_moments(self, species_id: int) -> None:
        """Compute and cache all moments for *species_id*.

        Populates ``self._cache`` with keys
        ``("density", sid)``, ``("Vx", sid)``, …, ``("qz", sid)``.
        """
        sid = species_id
        sp = self._prtl.species(sid, units="code")

        x = np.asarray(sp["x"], dtype=np.float64)
        y = np.asarray(sp["y"], dtype=np.float64)
        u = np.asarray(sp["u"], dtype=np.float64)
        v = np.asarray(sp["v"], dtype=np.float64)
        w = np.asarray(sp["w"], dtype=np.float64)
        wei = np.asarray(sp["wei"], dtype=np.float64)

        xe = self.x_edges
        ye = self.y_edges
        x0, x1 = self._x0, self._x1
        y0, y1 = self._y0, self._y1

        mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        x, y, u, v, w, wei = x[mask], y[mask], u[mask], v[mask], w[mask], wei[mask]

        def _bin2d(weights: np.ndarray) -> np.ndarray:
            """Bin weighted particles; returns (ny_region, nx_region)."""
            H, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=weights)
            return H.T  # histogram2d returns (nx, ny), transpose → (ny, nx)

        # ----- density -----
        if self._n0 is None:
            raise ValueError(
                "n0 (reference density = ppc0/2) is required. "
                "Provide params or pass n0= explicitly."
            )
        sum_wei = _bin2d(wei)
        n = (sum_wei * self._stride) / self._n0   # dimensionless, ≈1 upstream
        self._cache[("density", sid)] = n

        # ----- bulk velocity -----
        valid = sum_wei > 0
        Vx = np.where(valid, _bin2d(wei * u) / sum_wei, 0.0)
        Vy = np.where(valid, _bin2d(wei * v) / sum_wei, 0.0)
        Vz = np.where(valid, _bin2d(wei * w) / sum_wei, 0.0)
        self._cache[("Vx", sid)] = Vx
        self._cache[("Vy", sid)] = Vy
        self._cache[("Vz", sid)] = Vz

        # ----- second-order moments <ui uj> -----
        avg_uu = np.where(valid, _bin2d(wei * u * u) / sum_wei, 0.0)
        avg_vv = np.where(valid, _bin2d(wei * v * v) / sum_wei, 0.0)
        avg_ww = np.where(valid, _bin2d(wei * w * w) / sum_wei, 0.0)
        avg_uv = np.where(valid, _bin2d(wei * u * v) / sum_wei, 0.0)
        avg_uw = np.where(valid, _bin2d(wei * u * w) / sum_wei, 0.0)
        avg_vw = np.where(valid, _bin2d(wei * v * w) / sum_wei, 0.0)

        # ----- pressure tensor: P_ij = n × (<ui uj> - Vi Vj) -----
        Txx = avg_uu - Vx * Vx
        Tyy = avg_vv - Vy * Vy
        Tzz = avg_ww - Vz * Vz
        Txy = avg_uv - Vx * Vy
        Txz = avg_uw - Vx * Vz
        Tyz = avg_vw - Vy * Vz

        self._cache[("Pxx", sid)] = n * Txx
        self._cache[("Pyy", sid)] = n * Tyy
        self._cache[("Pzz", sid)] = n * Tzz
        self._cache[("Pxy", sid)] = n * Txy
        self._cache[("Pxz", sid)] = n * Txz
        self._cache[("Pyz", sid)] = n * Tyz

        # ----- heat flux: Q_i = n × ½<|δu|² δu_i> -----
        u2 = u * u + v * v + w * w  # |u|² per particle
        avg_u2 = np.where(valid, _bin2d(wei * u2) / sum_wei, 0.0)
        avg_u2u = np.where(valid, _bin2d(wei * u2 * u) / sum_wei, 0.0)
        avg_u2v = np.where(valid, _bin2d(wei * u2 * v) / sum_wei, 0.0)
        avg_u2w = np.where(valid, _bin2d(wei * u2 * w) / sum_wei, 0.0)

        V2 = Vx * Vx + Vy * Vy + Vz * Vz
        cross_x = avg_uu * Vx + avg_uv * Vy + avg_uw * Vz
        cross_y = avg_uv * Vx + avg_vv * Vy + avg_vw * Vz
        cross_z = avg_uw * Vx + avg_vw * Vy + avg_ww * Vz

        qx = 0.5 * (avg_u2u - avg_u2 * Vx - 2.0 * cross_x + 2.0 * V2 * Vx)
        qy = 0.5 * (avg_u2v - avg_u2 * Vy - 2.0 * cross_y + 2.0 * V2 * Vy)
        qz = 0.5 * (avg_u2w - avg_u2 * Vz - 2.0 * cross_z + 2.0 * V2 * Vz)

        self._cache[("qx", sid)] = n * qx
        self._cache[("qy", sid)] = n * qy
        self._cache[("qz", sid)] = n * qz

    def _ensure_computed(self, species_id: int) -> None:
        if ("density", species_id) not in self._cache:
            self._compute_moments(species_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def density(self, species_id: int, units: str = "code") -> np.ndarray:
        """Dimensionless number density normalized to n0 = ppc0/2.

        Upstream value ≈ 1.  No conversion is applied for ``units='ion'``
        (density is dimensionless in both unit systems).

        Parameters
        ----------
        species_id : int
            Species index (1-based).
        units : {'code', 'ion'}
            Has no effect on the numerical values but is accepted for
            API consistency.

        Returns
        -------
        numpy.ndarray, shape (ny_region, nx_region)
        """
        self._ensure_computed(species_id)
        return self._cache[("density", species_id)].copy()

    def bulk_velocity(
        self, species_id: int, units: str = "code"
    ) -> dict[str, np.ndarray]:
        """Bulk velocity components.

        Parameters
        ----------
        species_id : int
        units : {'code', 'ion'}
            ``'code'`` → c;  ``'ion'`` → vAi.

        Returns
        -------
        dict with keys ``'vx'``, ``'vy'``, ``'vz'``, each shape (ny, nx).
        """
        self._ensure_computed(species_id)
        sid = species_id
        vx = self._cache[("Vx", sid)].copy()
        vy = self._cache[("Vy", sid)].copy()
        vz = self._cache[("Vz", sid)].copy()

        if units == "ion":
            if self.uc is None:
                raise ValueError(
                    "unit_converter must be provided for units='ion'."
                )
            f = self.uc.c_to_vAi
            vx, vy, vz = vx * f, vy * f, vz * f

        return {"vx": vx, "vy": vy, "vz": vz}

    def temperature_tensor(
        self, species_id: int, units: str = "code"
    ) -> dict[str, np.ndarray]:
        """Pressure tensor  P_ij = n × (<ui uj> − Vi Vj).

        Extensive quantity: multiply the intensive temperature tensor by the
        dimensionless density n/n0.  Upstream value ≈ T_upstream.

        Parameters
        ----------
        species_id : int
        units : {'code', 'ion'}
            ``'code'`` → c²;  ``'ion'`` → vAi².

        Returns
        -------
        dict with keys ``'xx'``, ``'yy'``, ``'zz'``, ``'xy'``, ``'xz'``, ``'yz'``,
        each shape (ny, nx).
        """
        self._ensure_computed(species_id)
        sid = species_id
        result = {
            "xx": self._cache[("Pxx", sid)].copy(),
            "yy": self._cache[("Pyy", sid)].copy(),
            "zz": self._cache[("Pzz", sid)].copy(),
            "xy": self._cache[("Pxy", sid)].copy(),
            "xz": self._cache[("Pxz", sid)].copy(),
            "yz": self._cache[("Pyz", sid)].copy(),
        }

        if units == "ion":
            if self.uc is None:
                raise ValueError(
                    "unit_converter must be provided for units='ion'."
                )
            f = self.uc.c_to_vAi ** 2
            result = {k: v * f for k, v in result.items()}

        return result

    def temperature(self, species_id: int, units: str = "code") -> np.ndarray:
        """Scalar temperature  T = Tr(P) / (3 n).

        Recovers the intensive per-particle temperature from the extensive
        pressure tensor.

        Parameters
        ----------
        species_id : int
        units : {'code', 'ion'}
            ``'code'`` → c²;  ``'ion'`` → vAi².

        Returns
        -------
        numpy.ndarray, shape (ny_region, nx_region)
        """
        P = self.temperature_tensor(species_id, units=units)
        n = self.density(species_id)   # dimensionless, no unit conversion
        valid = n > 0
        return np.where(
            valid,
            (P["xx"] + P["yy"] + P["zz"]) / (3.0 * np.where(valid, n, 1.0)),
            0.0,
        )

    def heat_flux(
        self, species_id: int, units: str = "code"
    ) -> dict[str, np.ndarray]:
        """Extensive heat flux  Q_i = n × ½<|δu|² δu_i>.

        Extensive quantity: the intensive heat flux multiplied by n/n0.

        The intensive heat flux is defined as the third-order velocity moment
        in the bulk-flow frame::

            qi = ½(<|δu|² δui>)   where  δu = u − V

        Parameters
        ----------
        species_id : int
        units : {'code', 'ion'}
            ``'code'`` → c³;  ``'ion'`` → vAi³.

        Returns
        -------
        dict with keys ``'x'``, ``'y'``, ``'z'``, each shape (ny, nx).
        """
        self._ensure_computed(species_id)
        sid = species_id
        qx = self._cache[("qx", sid)].copy()
        qy = self._cache[("qy", sid)].copy()
        qz = self._cache[("qz", sid)].copy()

        if units == "ion":
            if self.uc is None:
                raise ValueError(
                    "unit_converter must be provided for units='ion'."
                )
            f = self.uc.c_to_vAi ** 3
            qx, qy, qz = qx * f, qy * f, qz * f

        return {"x": qx, "y": qy, "z": qz}

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Release all cached moment arrays from memory."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        nx = self._x1 - self._x0
        ny = self._y1 - self._y0
        region_str = f"({self._x0}:{self._x1}, {self._y0}:{self._y1})"
        return (
            f"ParticleMoments(step={self._prtl.step}, "
            f"grid=({ny}, {nx}), region={region_str}, "
            f"stride={self._stride})"
        )
