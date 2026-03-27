"""
05_energy_spectra.py
====================
Plot particle energy spectra (dN/dγ vs γ) at one or more output steps.

Usage
-----
    python examples/05_energy_spectra.py /path/to/output/ --steps 10 50 100
    python examples/05_energy_spectra.py /path/to/output/              # last step only
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pytrist


def main(output_dir: str, steps: list[int] | None, species_ids: list[int]) -> None:
    sim = pytrist.Simulation(output_dir)
    uc  = sim.unit_converter

    if not steps:
        steps = [sim.steps[-1]]

    # Validate steps against available spectra
    available = set(sim.steps)
    steps = [s for s in steps if s in available]
    if not steps:
        print("None of the requested steps have spectrum files.")
        return

    colors = cm.viridis(np.linspace(0.1, 0.9, len(steps)))

    fig, axes = plt.subplots(
        1, len(species_ids), figsize=(5 * len(species_ids), 4),
        constrained_layout=True, squeeze=False,
    )

    for col_idx, sp_id in enumerate(species_ids):
        ax = axes[0, col_idx]
        label_name = "electrons" if sp_id == 1 else f"species {sp_id}"

        for step, color in zip(steps, colors):
            p     = sim.params(step)
            t_ion = uc.time(p.time)

            try:
                spec = sim.spectra(step)
            except KeyError:
                print(f"  No spectra file for step {step}, skipping.")
                continue

            try:
                gamma = spec.gamma_bins
                dn    = spec.spectrum(sp_id)
            except KeyError as e:
                print(f"  {e}")
                continue

            # Filter out zero/negative values for log-log plot
            mask = (gamma > 0) & (dn > 0)
            if not np.any(mask):
                continue

            ax.loglog(
                gamma[mask], dn[mask],
                color=color,
                linewidth=1.5,
                label=rf"$t = {t_ion:.2f}\,\Omega_{{ci}}^{{-1}}$",
            )

        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$dN/d\gamma$")
        ax.set_title(f"Energy spectrum — {label_name}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.2)

    out = "energy_spectra.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="Simulation output directory")
    parser.add_argument("--steps",   type=int, nargs="+", default=None,
                        help="Step numbers to plot (default: last step)")
    parser.add_argument("--species", type=int, nargs="+", default=[1, 2],
                        help="Species IDs to plot (default: 1 2)")
    args = parser.parse_args()
    main(args.output_dir, args.steps, args.species)
