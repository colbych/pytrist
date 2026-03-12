"""
03_energy_history.py
====================
Plot the time evolution of kinetic and electromagnetic energy from the
simulation history file.

The time axis is shown in inverse ion cyclotron times (1/Ωci).

Usage
-----
    python examples/03_energy_history.py /path/to/output/
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import pytrist


# Common column name variants across Tristan versions
_KIN_CANDIDATES = ["totKinE", "kinE", "kin_energy", "Ekin"]
_EM_CANDIDATES  = ["totEmE",  "emE",  "em_energy",  "Eem", "totBE", "totEE"]


def _find_column(hist: pytrist.History, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in hist:
            return name
    return None


def main(output_dir: str) -> None:
    sim  = pytrist.Simulation(output_dir)
    uc   = sim.unit_converter
    hist = sim.history()

    print("History columns:", hist.column_names)

    # --- Time axis ---
    t_code = hist.time
    if t_code is None:
        print("ERROR: no time column found in history file.")
        sys.exit(1)
    t_ion = uc.time(t_code)   # convert 1/ωpe → 1/Ωci

    # --- Energy columns ---
    kin_key = _find_column(hist, _KIN_CANDIDATES)
    em_key  = _find_column(hist, _EM_CANDIDATES)

    if kin_key is None and em_key is None:
        print("No recognised energy columns found.")
        print("Available columns:", hist.column_names)
        sys.exit(1)

    # --- Normalise to initial total energy ---
    energies = {}
    if kin_key:
        energies["Kinetic"] = hist[kin_key]
    if em_key:
        energies["EM"] = hist[em_key]

    e0 = sum(e[0] for e in energies.values())   # initial total energy

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    for label, energy in energies.items():
        ax.plot(t_ion, energy / e0, label=label, linewidth=1.5)

    if len(energies) > 1:
        total = sum(energies.values())
        ax.plot(t_ion, total / e0, "k--", linewidth=1, label="Total", alpha=0.7)

    ax.set_xlabel(r"$t \; [\Omega_{ci}^{-1}]$")
    ax.set_ylabel(r"Energy / $E_0$")
    ax.set_title("Energy partition")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = "energy_history.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="Simulation output directory")
    args = parser.parse_args()
    main(args.output_dir)
