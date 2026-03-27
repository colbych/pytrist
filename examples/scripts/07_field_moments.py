"""
07_field_moments.py
===================
Spatial maps of field-file moment diagnostics for a chosen species and step.

Six panels arranged in a 2×3 grid:
  Top row   — bulk velocity Ux, Uy and charge density ρ/n0
  Bottom row — temperature tensor diagonal Txx, Tyy, Tzz

All quantities shown in ion units (vAi, mi vAi², n0).
Magnetic field lines are overlaid as contours of the flux function ψ.

Requires a params file for the chosen step (species mass, charge, and n0).

Usage
-----
    python examples/scripts/07_field_moments.py /path/to/output/ --step 10 --species 1
    python examples/scripts/07_field_moments.py /path/to/output/   # last step, electrons
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pytrist


def _squeeze(arr: np.ndarray) -> np.ndarray:
    """Return the x-y slice (last two axes) for 3-D field arrays."""
    return arr[0] if arr.ndim == 3 else arr


def main(output_dir: str, step: int | None, species_id: int) -> None:
    sim = pytrist.Simulation(output_dir)
    uc  = sim.unit_converter

    if step is None:
        step = sim.steps[-1]

    p     = sim.params(step)
    t_ion = uc.time(p.time)
    print(f"Step {step}  |  t = {p.time:.1f} / ωpe  =  {t_ion:.3f} / Ωci")

    flds = sim.fields(step)
    fm   = sim.field_moments(step)

    label = "electrons" if species_id == 1 else f"ions (sp {species_id})"
    print(f"Species: {label}")

    # --- Coordinate axes in di ---
    sample = _squeeze(flds.bz)
    ny, nx = sample.shape
    x = np.arange(nx) * uc.cell_to_di
    y = np.arange(ny) * uc.cell_to_di

    # --- Magnetic flux function for field-line contours ---
    try:
        psi = _squeeze(flds.psi())
    except Exception:
        psi = None

    # --- Compute all diagnostics in ion units ---
    vel   = fm.bulk_velocity(species_id, units="ion")
    T     = fm.temperature_tensor(species_id, units="ion")
    rho   = fm.charge_density(species_id, units="ion")

    panels = [
        (_squeeze(vel["x"]),  r"$U_x \; [v_{Ai}]$",            "RdBu_r",  True),
        (_squeeze(vel["y"]),  r"$U_y \; [v_{Ai}]$",            "RdBu_r",  True),
        (_squeeze(rho),       r"$\rho \; / \; n_0$",            "RdBu_r",  True),
        (_squeeze(T["xx"]),   r"$T_{xx} \; [m_i v_{Ai}^2]$",   "plasma",  False),
        (_squeeze(T["yy"]),   r"$T_{yy} \; [m_i v_{Ai}^2]$",   "plasma",  False),
        (_squeeze(T["zz"]),   r"$T_{zz} \; [m_i v_{Ai}^2]$",   "plasma",  False),
    ]

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    fig.suptitle(
        rf"{label}  —  $t = {t_ion:.2f}\,\Omega_{{ci}}^{{-1}}$",
        fontsize=13,
    )

    for ax, (data, cblabel, cmap, symmetric) in zip(axes.flat, panels):
        if symmetric:
            vlim = np.percentile(np.abs(data), 99)
            norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim) if vlim > 0 else None
            im = ax.pcolormesh(x, y, data, cmap=cmap, norm=norm, shading="auto")
        else:
            vmax = np.percentile(data[data > 0], 99) if np.any(data > 0) else 1.0
            vmin = max(data[data > 0].min(), vmax * 1e-3) if np.any(data > 0) else 0.0
            im = ax.pcolormesh(
                x, y, data,
                norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=cmap,
                shading="auto",
            )

        fig.colorbar(im, ax=ax, label=cblabel, shrink=0.9)

        if psi is not None:
            ax.contour(x, y, psi, levels=20, colors="k", linewidths=0.4, alpha=0.5)

        ax.set_xlabel(r"$x \; [d_i]$")
        ax.set_ylabel(r"$y \; [d_i]$")
        ax.set_aspect("equal")

    out = f"field_moments_sp{species_id}_step{step:05d}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_dir", help="Simulation output directory")
    parser.add_argument("--step",    type=int, default=None,
                        help="Step number (default: last)")
    parser.add_argument("--species", type=int, default=1,
                        help="Species ID (default: 1 = electrons)")
    args = parser.parse_args()
    main(args.output_dir, args.step, args.species)
