"""
04_particle_phase_space.py
==========================
Phase-space scatter plot: particle x-position vs. x-velocity (ux),
coloured by weight, for a chosen species and output step.

Positions are plotted in ion inertial lengths (di);
velocities are plotted in ion Alfvén speed units (vAi).

Usage
-----
    python examples/04_particle_phase_space.py /path/to/output/ --step 10 --species 1
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pytrist


def main(output_dir: str, step: int | None, species_id: int) -> None:
    sim = pytrist.Simulation(output_dir)
    uc  = sim.unit_converter

    if step is None:
        step = sim.steps[-1]

    p    = sim.params(step)
    t_ion = uc.time(p.time)

    # Load particle data in ion units
    prtl = sim.particles(step)
    sp   = prtl.species(species_id, units="ion")

    if "x" not in sp or "u" not in sp:
        print("ERROR: x or u dataset not found for this species.")
        return

    x   = sp["x"]   # positions in di
    u   = sp["u"]   # x-velocity in vAi
    wei = sp.get("wei", np.ones_like(x))

    # Downsample if too many particles to plot comfortably
    MAX_PLOT = 200_000
    if len(x) > MAX_PLOT:
        idx = np.random.choice(len(x), MAX_PLOT, replace=False)
        x, u, wei = x[idx], u[idx], wei[idx]
        print(f"Downsampled to {MAX_PLOT} particles for plotting.")

    label = "electrons" if species_id == 1 else f"ions (sp {species_id})"
    print(f"Step {step}  |  t = {t_ion:.3f} / Ωci  |  {len(x)} {label}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)

    sc = ax.scatter(
        x, u, c=np.log10(np.maximum(wei, 1e-10)),
        s=0.5, alpha=0.4, cmap="viridis", rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, label=r"$\log_{10}(w)$")

    ax.set_xlabel(r"$x \; [d_i]$")
    ax.set_ylabel(r"$u_x \; [v_{Ai}]$")
    ax.set_title(
        rf"Phase space ({label})"
        rf"  —  $t = {t_ion:.2f}\,\Omega_{{ci}}^{{-1}}$"
    )

    out = f"phase_space_sp{species_id}_step{step:05d}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="Simulation output directory")
    parser.add_argument("--step",    type=int, default=None, help="Step number (default: last)")
    parser.add_argument("--species", type=int, default=1,    help="Species ID (default: 1 = electrons)")
    args = parser.parse_args()
    main(args.output_dir, args.step, args.species)
