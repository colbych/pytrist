"""
02_plot_fields.py
=================
2-D colour plots of the out-of-plane magnetic field (Bz) and electron
number density at a chosen output step.

Axes are labelled in ion inertial lengths (di) and ion cyclotron times (1/Ωci).

Usage
-----
    python examples/02_plot_fields.py /path/to/output/ --step 10
    python examples/02_plot_fields.py /path/to/output/              # last step
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pytrist


def main(output_dir: str, step: int | None) -> None:
    sim = pytrist.Simulation(output_dir)
    uc  = sim.unit_converter

    if step is None:
        step = sim.steps[-1]

    flds = sim.fields(step)
    p    = sim.params(step)

    t_ion = uc.time(p.time)
    print(f"Step {step}  |  t = {p.time:.1f} / ωpe  =  {t_ion:.3f} / Ωci")

    # --- Load fields ---
    bz   = flds.bz                  # out-of-plane B (2-D or 3-D array)
    # Squeeze out the z-dimension for 2-D runs
    if bz.ndim == 3:
        bz = bz[0]

    # Electron density (species 1); fall back gracefully if not present
    dens_key = "dens_1" if "dens_1" in flds.keys else None
    if dens_key:
        dens = flds[dens_key]
        if dens.ndim == 3:
            dens = dens[0]

    # --- Build coordinate axes in ion inertial lengths ---
    ny, nx = bz.shape
    x = np.arange(nx) * uc.cell_to_di   # cells → di
    y = np.arange(ny) * uc.cell_to_di

    # --- Plot ---
    ncols = 2 if dens_key else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    # Bz panel
    ax = axes[0]
    vlim = np.percentile(np.abs(bz), 99)
    im = ax.pcolormesh(x, y, bz, cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto")
    fig.colorbar(im, ax=ax, label=r"$B_z / B_0$")
    ax.set_xlabel(r"$x \; [d_i]$")
    ax.set_ylabel(r"$y \; [d_i]$")
    ax.set_title(rf"$B_z$  —  $t = {t_ion:.2f}\,\Omega_{{ci}}^{{-1}}$")
    ax.set_aspect("equal")

    # Density panel
    if dens_key:
        ax = axes[1]
        vlim_d = np.percentile(dens[dens > 0], 99) if np.any(dens > 0) else 1.0
        im2 = ax.pcolormesh(
            x, y, dens,
            norm=mcolors.LogNorm(vmin=max(dens[dens > 0].min(), vlim_d * 1e-3),
                                 vmax=vlim_d),
            cmap="plasma",
            shading="auto",
        )
        fig.colorbar(im2, ax=ax, label=r"$n_e \; / \; n_0$")
        ax.set_xlabel(r"$x \; [d_i]$")
        ax.set_ylabel(r"$y \; [d_i]$")
        ax.set_title(rf"Electron density  —  $t = {t_ion:.2f}\,\Omega_{{ci}}^{{-1}}$")
        ax.set_aspect("equal")

    out = f"fields_step{step:05d}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="Simulation output directory")
    parser.add_argument("--step", type=int, default=None, help="Step number (default: last)")
    args = parser.parse_args()
    main(args.output_dir, args.step)
