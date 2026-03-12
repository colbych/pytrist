"""
06_unit_conversion.py
=====================
Demonstrate all unit conversions available in pytrist.UnitConverter.

This script does not require simulation output files — it constructs a
UnitConverter directly from representative parameters and shows how each
conversion works.

Background
----------
Tristan-MP V2 internal units:
  Lengths  →  grid cells
  Times    →  1/ωpe  (inverse electron plasma frequency)
  Speeds   →  c      (speed of light)

Ion units (what this converter produces):
  Lengths  →  di    (ion inertial length)
  Times    →  1/Ωci (inverse ion cyclotron frequency, based on By)
  Speeds   →  vAi   (ion Alfvén speed, based on By)

Key parameters:
  c_omp      = electron skin depth de in grid cells  (de = c/ωpe)
  sigma (σ)  = magnetisation  ωce²/ωpe²  (≈ B0² c_omp² / CC²)
  mass_ratio = mi/me  (ion-to-electron mass ratio)
  CC         = speed of light in code units (~0.45)
"""

import numpy as np
import pytrist
from pytrist import UnitConverter


def demo_converter(c_omp: float, sigma: float, mass_ratio: float, CC: float) -> None:
    uc = UnitConverter(c_omp=c_omp, sigma=sigma, mass_ratio=mass_ratio, CC=CC)

    print("=" * 60)
    print(uc.summary())
    print()

    # ------------------------------------------------------------------
    # Conversion factors
    # ------------------------------------------------------------------
    print("Conversion factors (multiply code-unit value by factor):")
    print(f"  Length:  1 cell   = {uc.cell_to_di:.6g} di")
    print(f"  Time:    1/ωpe    = {uc.wpe_to_wci:.6g} /Ωci")
    print(f"  Speed:   1 c      = {uc.c_to_vAi:.6g} vAi")
    print()

    # ------------------------------------------------------------------
    # Length conversion example
    # ------------------------------------------------------------------
    # A wave with wavelength = 100 cells.  In di?
    wavelength_cells = 100.0
    wavelength_di    = uc.length(wavelength_cells)[0]
    print(f"Wavelength example:")
    print(f"  λ = {wavelength_cells} cells  →  {wavelength_di:.4f} di")
    print()

    # ------------------------------------------------------------------
    # Time conversion example
    # ------------------------------------------------------------------
    # Output times at steps 1..5 (assuming output every 100 ωpe^-1)
    t_code = np.arange(1, 6) * 100.0          # [100, 200, ..., 500] / ωpe
    t_ion  = uc.time(t_code)                   # in 1/Ωci
    print("Time conversion (output interval = 100/ωpe):")
    print(f"  {'t [1/ωpe]':>12}  {'t [1/Ωci]':>12}")
    for tc, ti in zip(t_code, t_ion):
        print(f"  {tc:>12.1f}  {ti:>12.6f}")
    print()

    # ------------------------------------------------------------------
    # Speed conversion example
    # ------------------------------------------------------------------
    # Bulk flow speeds as a fraction of c
    speeds_c   = np.array([0.0, 0.01, 0.05, 0.10, 0.45])  # in units of c
    speeds_vAi = uc.speed(speeds_c)
    print("Speed conversion:")
    print(f"  {'v [c]':>8}  {'v [vAi]':>10}")
    for vc, va in zip(speeds_c, speeds_vAi):
        print(f"  {vc:>8.3f}  {va:>10.4f}")
    print()

    # ------------------------------------------------------------------
    # Derived physical scales
    # ------------------------------------------------------------------
    print("Physical scales:")
    print(f"  Ion inertial length : di = {uc.di_in_cells:.2f} cells")
    print(f"  Ion Alfvén speed    : vAi = {uc.vAi_over_c:.4g} c")
    print(f"  Ion cyclotron freq  : Ωci = {uc.Omega_ci_over_wpe:.4g} ωpe")
    print()

    # ------------------------------------------------------------------
    # Recover code units from ion units (round-trip check)
    # ------------------------------------------------------------------
    x_cells   = 250.0
    x_di      = uc.length(x_cells)[0]
    x_back    = x_di / uc.cell_to_di          # invert
    print(f"Round-trip check (length):")
    print(f"  {x_cells} cells → {x_di:.6f} di → {x_back:.6f} cells")
    print()


def main() -> None:
    print("Example 1 — Typical reconnection run (non-relativistic ions)")
    print("  c_omp=10, σ=0.1, mi/me=100, CC=0.45")
    demo_converter(c_omp=10.0, sigma=0.1, mass_ratio=100.0, CC=0.45)

    print("Example 2 — Magnetised electrons, lighter ions")
    print("  c_omp=15, σ=1.0, mi/me=25, CC=0.45")
    demo_converter(c_omp=15.0, sigma=1.0, mass_ratio=25.0, CC=0.45)

    # ------------------------------------------------------------------
    # Using UnitConverter via a Simulation object
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Using via Simulation object:")
    print()
    print("    import pytrist")
    print("    sim = pytrist.Simulation('/path/to/output/')")
    print("    uc  = sim.unit_converter")
    print("    print(uc.summary())")
    print()
    print("    # Convert history time axis")
    print("    hist   = sim.history()")
    print("    t_ion  = uc.time(hist['t'])")
    print()
    print("    # Convert particle positions")
    print("    prtl   = sim.particles(step=10)")
    print("    sp1    = prtl.species(1, units='ion')")
    print("    x_di   = sp1['x']    # already in di")


if __name__ == "__main__":
    main()
