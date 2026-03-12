# CLAUDE.md — pytrist Developer Guide

This file is the primary reference for Claude Code sessions working on
`pytrist`, a Python reader for Tristan-MP V2 plasma PIC simulation output.

---

## 1. Package Purpose and Architecture

`pytrist` reads HDF5 output files produced by the Tristan-MP V2 particle-in-cell
(PIC) plasma physics code and exposes the data as NumPy arrays with optional
unit conversion to physically meaningful ion units.

### Design philosophy

- **Lazy loading**: HDF5 files are opened only when data is first accessed,
  then cached.  Never preload data the user did not ask for.
- **Minimal dependencies**: Core functionality requires only `numpy` and `h5py`.
  `xarray` support is optional and kept separate.
- **Flat API**: All common operations are reachable through `Simulation`.
  Individual reader classes (`FieldSnapshot`, etc.) can also be used directly.
- **Robust file discovery**: Handles both flat output directories and the
  subdirectory layout (`flds/`, `prtl/`, `spec/`).
- **Clear errors**: FileNotFoundError messages tell the user what file was
  expected and where it was searched for.

### Module map

```
pytrist/
├── __init__.py        — Re-exports the public API
├── simulation.py      — Simulation class (primary entry point)
├── fields.py          — FieldSnapshot, FieldLoader
├── particles.py       — ParticleSnapshot
├── params.py          — SimParams
├── history.py         — History
├── spectra.py         — SpectraSnapshot
└── units.py           — UnitConverter
```

---

## 2. Tristan-V2 Output File Organisation

### File naming

All output files use a 5-digit zero-padded step number as the suffix:

| File pattern        | Content                                       |
|---------------------|-----------------------------------------------|
| `flds.tot.NNNNN`    | Electromagnetic fields on the grid (HDF5)     |
| `prtl.tot.NNNNN`    | Macro-particle positions and velocities (HDF5)|
| `params.NNNNN`      | Scalar simulation parameters (HDF5)           |
| `spec.tot.NNNNN`    | Energy spectra (HDF5, optional)               |
| `history`           | Time-series diagnostics (tab-separated text)  |

Files may live directly in the output directory or in subdirectories named
`flds/`, `prtl/`, `spec/`.  `_find_files()` in `simulation.py` handles both.

### HDF5 field file datasets

Typical datasets in `flds.tot.NNNNN` (all 3-D arrays of shape `(nz, ny, nx)`):

- EM fields: `ex ey ez bx by bz jx jy jz`
- Per species: `dens_1 dens_2 enrg_1 enrg_2` (index = species ID)

### HDF5 particle file datasets

Two naming schemes exist across Tristan versions:

- **Suffixed** (most common): `x1 y1 z1 u1 v1 w1 wei1` for species 1,
  `x2 y2 z2 …` for species 2, etc.
- **Flat**: single `x y z u v w wei` arrays for all particles; species
  distinguished by a companion `ch` (charge sign: −1 = electrons) or `sp`
  (integer species ID) array.

`ParticleSnapshot._detect_naming_scheme()` determines which layout is used.

### History file format

A plain-text file with:
- Possibly leading `#` comment lines (skipped)
- A header line with tab- or space-separated column names
- One data row per output step

Common column names: `t`, `lx`, `ly`, `totKinE`, `totEmE`, `bx_max`, `by_max`.

### Params file datasets

Each dataset is a scalar (or 1-element array) holding one simulation parameter.
Some parameters live in HDF5 groups (e.g., `output/interval`).  `SimParams`
flattens the group hierarchy into `/`-separated key names.

---

## 3. Unit System — Detailed Mathematics

### Code (internal) units

| Quantity | Code unit               |
|----------|-------------------------|
| Length   | grid cell               |
| Time     | 1/ωpe (inverse electron plasma frequency) |
| Speed    | c (speed of light)      |
| B field  | normalised to B0 (background field magnitude) |

The magnetisation parameter σ is defined as:

```
σ = ωce² / ωpe²  =  (eB0/mec)² / ωpe²
```

In code units, where c_omp = de = c/ωpe (in cells) and CC = c (in code
cells/timestep):

```
σ = B0² × c_omp² / CC²
```

### Ion units (target system)

The natural scale for magnetised ion physics is set by the background
magnetic field (typically By in Tristan-V2 reconnection runs):

**Ion inertial length** (spatial scale):
```
di = de × √(mi/me) = c_omp × √mass_ratio   [in grid cells]
```

**Ion cyclotron frequency** (time scale):
```
Ωci_y = ωce_y / (mi/me) = (√σ × ωpe) / mass_ratio
```

**Ion Alfvén speed** (velocity scale):
```
vAi = c × √(σ / mass_ratio)
```

### Conversion factors

To convert a quantity FROM code units TO ion units, multiply by the factor:

| Quantity  | Factor formula                      | Variable name   |
|-----------|-------------------------------------|-----------------|
| Length    | `1 / (c_omp × √mass_ratio)`         | `cell_to_di`    |
| Time      | `√σ / mass_ratio`                   | `wpe_to_wci`    |
| Speed     | `√(mass_ratio / σ)`                 | `c_to_vAi`      |

**Length derivation:**
```
di = c_omp × √mass_ratio  [cells]
x[di] = x[cells] / di_in_cells = x[cells] × (1 / (c_omp × √mass_ratio))
```

**Time derivation:**
```
Ωci_y = √σ × ωpe / mass_ratio
t[Ωci_y] = t[1/ωpe] × (Ωci_y / ωpe) = t[1/ωpe] × √σ / mass_ratio
```

**Speed derivation:**
```
vAi = c × √(σ / mass_ratio)
v[vAi] = v[c] × c / vAi = v[c] / (vAi/c) = v[c] / √(σ/mass_ratio)
       = v[c] × √(mass_ratio / σ)
```

**Electromagnetic fields:**
B fields are stored normalised to B0 (the background field strength).  By
definition, B0 = 1 in both code and ion units, so the conversion factor is
1.0 and numerical values are unchanged.  The same applies to E fields, which
are normalised identically.

### Numeric example

For `c_omp=10, σ=0.1, mass_ratio=100, CC=0.45`:

| Quantity         | Value                             |
|------------------|-----------------------------------|
| di_in_cells      | 10 × √100 = 100 cells             |
| vAi/c            | √(0.1/100) = √0.001 ≈ 0.03162    |
| Ωci/ωpe          | √0.1/100 ≈ 3.162×10⁻³            |
| cell_to_di       | 1/100 = 0.01                      |
| wpe_to_wci       | √0.1/100 ≈ 3.162×10⁻³            |
| c_to_vAi         | √(100/0.1) = √1000 ≈ 31.62       |

---

## 4. Key Classes and Their Relationships

```
Simulation
├── uses: _find_files()          — file discovery utility (module level)
├── returns: SimParams           — via .params(step)
├── returns: FieldSnapshot       — via .fields(step)
│   └── uses: UnitConverter      — attached as .uc
├── returns: ParticleSnapshot    — via .particles(step)
│   └── uses: UnitConverter      — attached as .uc
├── returns: SpectraSnapshot     — via .spectra(step)
├── returns: History             — via .history()
│   └── uses: UnitConverter      — for .time_ion
└── creates: UnitConverter       — via .unit_converter (from SimParams)
```

### Instantiation flow

1. User creates `Simulation(output_dir)`.
2. First access to `.steps` triggers `_scan_flds()` which calls `_find_files()`.
3. First access to `.unit_converter` reads the last params file via `.params()`.
4. `UnitConverter` is built from `c_omp, sigma, mass_ratio, CC`.
5. Calls to `.fields(step)` return a `FieldSnapshot` with the converter attached.

### Caching strategy

- `Simulation` caches `SimParams`, `FieldSnapshot`, `ParticleSnapshot`,
  `SpectraSnapshot`, and `History` objects in dicts keyed by step number.
- Individual snapshots cache raw NumPy arrays in `self._cache` dicts.
- `clear_cache()` methods allow releasing memory when processing many steps.

---

## 5. Common Usage Patterns

### Basic inspection

```python
import pytrist

sim = pytrist.Simulation("/path/to/output/")
print(sim.steps)           # [1, 2, 3, …, 100]
print(sim.times)           # [50.0, 100.0, …] in 1/ωpe
print(sim.summary())
```

### Reading fields

```python
flds = sim.fields(step=10)
bx = flds.bx               # ndarray, shape (nz, ny, nx)
b_mag = flds.b_magnitude    # computed on the fly

# Custom field (e.g. particle density species 1)
dens = flds["dens_1"]
```

### Reading particles

```python
prtl = sim.particles(step=10)
sp1 = prtl.species(1)      # dict: {x, y, z, u, v, w, wei, ...}
x_e = sp1["x"]             # electron x-positions in cells
gamma_e = prtl.gamma(species_id=1)

sp2 = prtl.species(2)      # ions
```

### Unit conversion

```python
uc = sim.unit_converter
print(uc.summary())

# Convert manually:
x_di   = uc.length(sp1["x"])    # cells → di
t_wci  = uc.time(sim.times)     # 1/ωpe → 1/Ωci_y
v_vAi  = uc.speed(sp1["u"])     # c → vAi

# Or request ion units from .species():
sp1_ion = prtl.species(1, units="ion")  # x,y,z in di; u,v,w in vAi

# One-liner for fields:
bx_ion = sim.read_field("bx", step=10, units="ion")
```

### History time series

```python
hist = sim.history()
t    = hist["t"]           # time in 1/ωpe
eke  = hist["totKinE"]
t_ion = hist.time_ion      # converted to 1/Ωci_y (needs unit_converter)
```

### Spectra

```python
spec = sim.spectra(step=10)
gamma_bins = spec.gamma_bins
dn_de = spec.spectrum(1)   # dN/dγ for species 1
```

### Memory management

```python
# Process many steps without accumulating arrays:
for step in sim.steps:
    flds = sim.fields(step)
    do_analysis(flds.bx, flds.by)
    flds.clear_cache()
```

### Using individual classes directly

```python
from pytrist.fields import FieldSnapshot
from pytrist.units import UnitConverter

uc = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
snap = FieldSnapshot("/data/flds.tot.00010", unit_converter=uc)
print(snap.bx.shape)
```

---

## 6. How to Extend the Package

### Adding a new field type

Field files may contain datasets not yet enumerated as named properties.
They are always accessible via `snap["dataset_name"]` or `snap.dataset_name`
(attribute access falls through to `__getattr__` which calls `_load`).

To add a named property (for IDE autocompletion and documentation):

```python
# In fields.py, inside FieldSnapshot:
@property
def new_field(self) -> np.ndarray:
    """Description of the new field."""
    return self._load("new_field_dataset_name")
```

### Adding new unit conversions

Add a method to `UnitConverter` in `units.py`:

```python
def density(self, arr, n0: float) -> np.ndarray:
    """Convert number density from code units (particles/cell³) to n/n0."""
    # 1 code cell³ contains n0×ppc0 physical particles; adjust as needed.
    return np.asarray(arr) / n0
```

Then call it in the relevant snapshot's conversion logic.

### Adding xarray support

The optional `xarray` extra is reserved for returning labeled arrays.  A
suggested implementation:

```python
# In fields.py
def to_xarray(self, fields=None):
    """Return an xr.Dataset of the requested field arrays."""
    import xarray as xr
    fields = fields or ["bx", "by", "bz", "ex", "ey", "ez"]
    data_vars = {f: (["z", "y", "x"], self._load(f)) for f in fields
                 if f in self.keys}
    return xr.Dataset(data_vars, attrs={"step": self.step, "time": self.time})
```

### Supporting a new output type

1. Create a new module `pytrist/new_output.py` with a class following the
   same pattern as `FieldSnapshot`:
   - `__init__` takes `filepath` + optional `unit_converter` + `params`
   - Lazy loading via `_load(key)` with `self._cache`
   - `self.step` extracted from filename suffix
2. Add file discovery in `simulation.py` (a `_scan_new_output()` method and
   a `new_output(step)` reader method mirroring `fields()`).
3. Export the class from `__init__.py`.

### Supporting a new params field name variation

Edit `_ALIASES` in `params.py`:

```python
_ALIASES: dict[str, list[str]] = {
    ...
    "new_param": ["new_param", "new_param_v2", "OLD_NAME"],
    ...
}
```

Then add a convenience property:

```python
@property
def new_param(self) -> float:
    """Description."""
    return float(self._get_alias("new_param"))
```

---

## 7. Testing Approach

### Running tests

```bash
# From the repo root:
pip install -e ".[dev]"
pytest tests/
pytest tests/ -v --tb=short
pytest tests/ --cov=pytrist --cov-report=term-missing
```

### Test structure

`tests/test_units.py` — pure math tests, no HDF5 files required.  Covers:
- Derived quantity formulas (`di_in_cells`, `vAi_over_c`, `Omega_ci_over_wpe`)
- Conversion factor formulas (`cell_to_di`, `wpe_to_wci`, `c_to_vAi`)
- Physical consistency checks (σ definition, di/de relation)
- Array conversion methods (`length`, `time`, `speed`, `field_B`, `field_E`)
- Round-trip recovery (convert then invert)
- Hard-coded numerical values for the standard parameters

### Writing tests for file-reading classes

HDF5 test files can be created with `h5py` in a pytest fixture:

```python
import h5py
import numpy as np
import pytest
from pathlib import Path

@pytest.fixture
def tmp_flds_file(tmp_path):
    fp = tmp_path / "flds.tot.00001"
    with h5py.File(fp, "w") as f:
        f.create_dataset("bx", data=np.ones((4, 4, 4)))
        f.create_dataset("by", data=np.zeros((4, 4, 4)))
        f.create_dataset("time", data=np.float32(100.0))
    return fp

def test_field_snapshot_bx(tmp_flds_file):
    from pytrist.fields import FieldSnapshot
    snap = FieldSnapshot(tmp_flds_file)
    assert snap.bx.shape == (4, 4, 4)
    assert snap.step == 1
```

This pattern can be applied to `params.py`, `particles.py`, `spectra.py`,
and `simulation.py` tests.

---

## 8. Known Limitations and Future Work

- **Slice files** (`slice/` subdirectory): not yet supported.
- **Restart files**: not needed for analysis; not implemented.
- **MPI-split output** (non-merged files): `pytrist` assumes merged
  (`tot`) files produced by Tristan-V2's built-in merge step.
- **xarray integration**: stubs exist but full implementation is pending.
- **Species auto-detection in flat scheme**: relies on the `ch` (charge)
  dataset; if absent, `_find_species_indicator()` returns None and
  `species()` returns unsplit arrays.
- **3-D coordinates for field arrays**: coordinate arrays (cell x, y, z
  positions) are not stored by Tristan-V2; callers must construct them
  as `np.arange(nx)` etc.
