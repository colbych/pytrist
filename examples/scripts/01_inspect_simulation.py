"""
01_inspect_simulation.py
========================
Explore a Tristan-MP V2 output directory: list available steps,
print simulation parameters, and show the unit conversion scales.

Usage
-----
    python examples/01_inspect_simulation.py /path/to/output/
"""

import sys
import pytrist


def main(output_dir: str) -> None:
    sim = pytrist.Simulation(output_dir)

    # --- High-level summary ---
    print(sim.summary())
    print()

    # --- Available output steps ---
    steps = sim.steps
    print(f"Output steps: {steps[0]} … {steps[-1]}  ({len(steps)} total)")
    print()

    # --- Simulation parameters (last step) ---
    p = sim.params()
    print(p.summary())
    print()

    # --- Unit converter ---
    uc = sim.unit_converter
    print(uc.summary())
    print()

    # --- Map code times to ion cyclotron times ---
    times_code = sim.times          # list of floats in 1/ωpe
    times_ion  = uc.time(times_code)

    print("First 5 output times:")
    print(f"  {'step':>6}  {'t [1/ωpe]':>12}  {'t [1/Ωci]':>12}")
    for step, tc, ti in zip(steps[:5], times_code[:5], times_ion[:5]):
        print(f"  {step:>6}  {tc:>12.1f}  {ti:>12.4f}")

    # --- Fields available in first step ---
    flds = sim.fields(step=steps[0])
    print(f"\nFields available at step {steps[0]}: {flds.keys}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
