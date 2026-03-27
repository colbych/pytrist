# pytrist

A Python reader for [Tristan-MP V2](https://github.com/PrincetonUniversity/tristan-mp-v2) plasma particle-in-cell (PIC) simulation output.

`pytrist` loads simulation data from HDF5 output files and optionally converts quantities into physically meaningful **ion-based units** — ion inertial length, inverse ion cyclotron frequency, and ion Alfvén speed — instead of the code's internal electron units.

## Features

- Read electromagnetic fields, particle data, energy spectra, and history files
- Lazy loading: HDF5 files are only read when data is first accessed
- Unit conversion to ion-based units via the mass ratio and magnetization σ
- Works on laptops and HPCs alike — only `numpy` and `h5py` required
- Simple, flat API designed for interactive analysis and scripting

## Installation

```bash
# From GitHub (recommended for HPCs):
pip install git+https://github.com/colbych/pytrist.git

# For development (editable install from a local clone):
git clone https://github.com/colbych/pytrist.git
cd pytrist
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, numpy ≥ 1.21, h5py ≥ 3.0

## Quick start

```python
import pytrist

# Point to your simulation output directory
sim = pytrist.Simulation("/path/to/output/")

# See what steps are available
print(sim.steps)   # [1, 2, 3, ..., 100]
print(sim.times)   # corresponding times in 1/ωpe

# Print a summary including unit conversion scales
print(sim.summary())
```

## Reading data

### Electromagnetic fields

Fields are stored in `flds.tot.NNNNN` files as 3-D arrays of shape `(nz, ny, nx)`.

```python
flds = sim.fields(step=10)

bx = flds.bx          # magnetic field x-component (numpy array)
by = flds.by
bz = flds.bz

ex = flds.ex          # electric field components
ey = flds.ey
ez = flds.ez

jx = flds.jx          # current density components

# Derived quantities
b_mag = flds.b_magnitude    # |B| = sqrt(bx² + by² + bz²)
e_mag = flds.e_magnitude    # |E|

# Particle density and energy density for each species
dens_e = flds["dens_1"]     # electrons (species 1)
dens_i = flds["dens_2"]     # ions (species 2)

# Any dataset in the HDF5 file is accessible by name
print(flds.keys)            # list all available fields
```

### Particle data

Particles are stored in `prtl.tot.NNNNN` files. Note that Tristan-V2 writes only a fraction of particles (controlled by `tot_output_stride` in the input file).

```python
prtl = sim.particles(step=10)

# Access data for a specific species (1-based indexing)
# Species 1 = electrons, Species 2 = ions (by default)
electrons = prtl.species(1)
ions      = prtl.species(2)

# Each species dict contains numpy arrays:
x = electrons["x"]    # x-position (cells)
y = electrons["y"]    # y-position (cells)
u = electrons["u"]    # x-velocity (4-velocity γβx, units of c)
v = electrons["v"]    # y-velocity
w = electrons["w"]    # z-velocity
wei = electrons["wei"]  # macro-particle weight

# Lorentz factor γ = sqrt(1 + u² + v² + w²)
gamma_e = prtl.gamma(species_id=1)
```

### History file

The `history` file contains global time-series diagnostics.

```python
hist = sim.history()

print(hist.column_names)    # list all columns

t    = hist["t"]            # simulation time (1/ωpe)
ekin = hist["totKinE"]      # total kinetic energy
eem  = hist["totEmE"]       # total EM energy
```

### Energy spectra

```python
spec = sim.spectra(step=10)

gamma_bins = spec.gamma_bins       # energy axis (Lorentz factor γ)
dn_dgamma  = spec.spectrum(1)      # dN/dγ for species 1 (electrons)
ion_spec   = spec.spectrum(2)      # ions
```

### Derived EM quantities (FieldSnapshot)

Several commonly needed quantities can be computed directly from the field snapshot:

```python
flds = sim.fields(step=10)

b2   = flds.B_squared()             # |B|², shape (nz, ny, nx)
edb  = flds.E_dot_B()               # E·B invariant
bhat = flds.B_hat()                 # unit vector {'x','y','z'}, dimensionless
exb  = flds.ExB_drift(units='ion')  # ExB drift velocity {'x','y','z'} in vAi
psi  = flds.psi(units='ion')        # magnetic flux function ψ (x-y plane) in d_i
```

`psi` is integrated from the origin as `ψ(x,0) = −∫By dx` then `ψ(x,y) = ψ(x,0) + ∫Bx dy`, so contours of `psi` are magnetic field lines in 2-D runs.

### Field-file moment diagnostics (FieldMoments)

Tristan-V2 writes per-species moment tensors (density, current, stress tensor, heat flux) to the field file. `FieldMoments` turns these into physical quantities:

```python
fm = sim.field_moments(step=10)

# Bulk velocity {'x','y','z'} — code units [c] or ion units [vAi]
vel = fm.bulk_velocity(1, units='ion')    # electrons
vel = fm.bulk_velocity(2, units='ion')    # ions

# Charge density ρ = q n
rho = fm.charge_density(1, units='ion')  # normalised to n0

# Pressure tensor {'xx','yy','zz','xy','xz','yz'}
P = fm.pressure_tensor(1, units='ion')   # [n0 mi vAi²]

# Temperature tensor (pressure / number density)
T = fm.temperature_tensor(1, units='ion')  # [mi vAi²] for all species
```

`FieldMoments` requires a params file to be present (for species mass/charge and n0). The instance is cached and shared with `EnergyFlux`, so calling both in one analysis session does not reload any data.

### Energy flux decomposition (EnergyFlux)

`EnergyFlux` computes all terms in the particle and electromagnetic energy flux budget:

```python
ef = sim.energy_flux(step=10)

# Bulk kinetic energy density and flux
ke  = ef.bulk_ke_density(2, units='ion')   # ½ ρ |U|², shape (nz, ny, nx)
qke = ef.bulk_ke_flux(2, units='ion')      # ke × U, dict {'x','y','z'}

# Thermal (internal) energy density and flux
u_th  = ef.internal_energy_density(2, units='ion')
q_ie  = ef.internal_energy_flux(2, units='ion')

# Enthalpy flux (pressure-tensor work): P·U
q_enth = ef.enthalpy_flux(2, units='ion')

# Irreducible heat flux cumulant
q_heat = ef.heat_flux(2, units='ion')

# Electromagnetic Poynting flux
S = ef.poynting_flux(units='ion')          # CC (E×B), dict {'x','y','z'}

# Totals
Q_prtl = ef.total_particle_energy_flux(2, units='ion')  # ke + enthalpy + heat
Q_tot  = ef.total_energy_flux(species_ids=[1, 2], units='ion')  # Poynting + all species
```

All methods accept `units='code'` (default) or `units='ion'`. Particle fluxes are normalised to `[n0 mi vAi³]`; Poynting flux uses the Gaussian Alfvén relation and does not require n0.

### Simulation parameters

```python
p = sim.params()           # last available step by default
p = sim.params(step=1)     # specific step

print(p.c_omp)             # electron skin depth in cells
print(p.sigma)             # magnetisation σ = ωce²/ωpe²
print(p.mass_ratio)        # ion-to-electron mass ratio mi/me
print(p.CC)                # speed of light in code units
print(p.time)              # simulation time in 1/ωpe

# Dict-like access to any parameter in the file
print(p["ppc0"])
print(list(p.keys()))
```

## Unit conversion

Tristan-V2 uses internal units based on electron scales: lengths in grid cells, times in 1/ωpe (inverse electron plasma frequency), and speeds in units of c. `pytrist` can convert to physically motivated **ion units** determined by the background magnetic field (By):

| Quantity | Code unit | Ion unit | Conversion factor |
|----------|-----------|----------|-------------------|
| Length | grid cell | ion inertial length d_i | `1 / (c_omp × √mass_ratio)` |
| Time | 1/ωpe | 1/Ωci (ion cyclotron period) | `√σ / mass_ratio` |
| Speed | c | vAi (ion Alfvén speed) | `√(mass_ratio / σ)` |
| B field | B0 | B0 | 1 (unchanged) |
| E field | B0 | E0 = B0 × vAi/c | `1 / (vAi/c)` |

where σ = ωce²/ωpe² is the magnetisation parameter.  The E field ion unit E0 = B0 × vAi/c is chosen so that an ExB drift at vAi has magnitude 1, and `field_E(E) × field_B(B)` gives the Poynting flux directly in ion units.

### Using the UnitConverter

```python
uc = sim.unit_converter
print(uc.summary())

# Convert manually
x_di  = uc.length(electrons["x"])   # cells → ion inertial lengths
t_wci = uc.time(sim.times)          # 1/ωpe → 1/Ωci
v_vAi = uc.speed(electrons["u"])    # c → ion Alfvén speed

# Access derived scales
print(uc.di_in_cells)        # ion inertial length in grid cells
print(uc.vAi_over_c)         # ion Alfvén speed as a fraction of c
print(uc.Omega_ci_over_wpe)  # Ωci in units of ωpe
```

### Ion-unit particle data

```python
# Request ion units directly from species()
electrons_ion = prtl.species(1, units="ion")
# x, y, z are now in d_i; u, v, w are now in vAi
```

### Ion-unit history

```python
hist = sim.history()
t_ion = hist.time_ion    # time axis in 1/Ωci (requires unit_converter)
```

### One-liner field access

```python
bx_ion = sim.read_field("bx", step=10, units="ion")
```

## Working with many steps

`pytrist` caches snapshots, so re-accessing the same step is free. To process many steps without accumulating data in memory, call `clear_cache()`:

```python
import numpy as np

b_rms = []
for step in sim.steps:
    flds = sim.fields(step)
    b_rms.append(np.sqrt(np.mean(flds.b_magnitude**2)))
    flds.clear_cache()   # free the loaded arrays before moving on

b_rms = np.array(b_rms)
```

## Using classes directly

All reader classes can be used independently of `Simulation`:

```python
from pytrist.fields   import FieldSnapshot
from pytrist.units    import UnitConverter

uc   = UnitConverter(c_omp=10, sigma=0.1, mass_ratio=100, CC=0.45)
snap = FieldSnapshot("/data/output/flds.tot.00010", unit_converter=uc)
bx   = snap.bx
```

## Examples

The `examples/` directory contains self-contained scripts and interactive notebooks.

**Scripts** (`examples/scripts/`):

| Script | Description |
|--------|-------------|
| `01_inspect_simulation.py` | Explore an output directory and print a summary |
| `02_plot_fields.py` | 2-D colour plots of Bz, density, and field lines |
| `03_energy_history.py` | Plot energy partition over time |
| `04_particle_phase_space.py` | Phase-space scatter plot |
| `05_energy_spectra.py` | Plot particle energy spectra |
| `06_unit_conversion.py` | Demonstrate all unit conversion utilities |
| `07_field_moments.py` | Spatial maps of bulk velocity and temperature tensor |
| `08_energy_flux.py` | Energy flux decomposition profiles |

**Notebooks** (`examples/notebooks/`):

| Notebook | Description |
|----------|-------------|
| `01_getting_started.ipynb` | Load a simulation, read fields and particles, plot |
| `02_unit_conversion.ipynb` | Unit system walkthrough with worked examples |
| `03_energy_analysis.ipynb` | Energy flux decomposition and budget verification |

## Contributing

Bug reports and pull requests are welcome at https://github.com/colbych/pytrist.

## License

MIT
