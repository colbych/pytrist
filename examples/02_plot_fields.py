"""
02_plot_fields.py
=================
2-D colour plots of the out-of-plane magnetic field (Bz) and electron
number density at a chosen output step.

Axes are labelled in ion inertial lengths (di) and ion cyclotron times (1/Ωci).
Magnetic field lines are shown as contours of the flux function ψ = Az.

Usage
-----
    python examples/02_plot_fields.py /path/to/output/ --step 10
    python examples/02_plot_fields.py /path/to/output/              # last step
"""

import argparse

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
    bz = flds.bz
    if bz.ndim == 3:
        bz = bz[0]

    # Magnetic flux function for field-line contours (code units — topology only)
    try:
        psi = flds.psi()
        if psi.ndim == 3:
            psi = psi[0]
    except Exception:
        psi = None

    # Electron number density normalised to n0
    # dens_1 stores mass density m_e * n_e; m_e = 1 in code units, so
    # dens_1 / n0 gives n_e / n0 directly.
    dens_key = "dens_1" if "dens_1" in flds.keys else None
    dens = None
    dens_label = None
    if dens_key:
        raw = flds[dens_key]
        if raw.ndim == 3:
            raw = raw[0]
        n0 = flds._n0
        if n0 is not None and n0 > 0:
            dens = raw / n0
            dens_label = r"$n_e \; / \; n_0$"
        else:
            # No n0 available — normalise to upstream value (99th percentile)
            # and label accordingly
            ref = np.percentile(raw[raw > 0], 99) if np.any(raw > 0) else 1.0
            dens = raw / ref
            dens_label = r"$n_e$ (norm. to upstream)"

    # --- Build coordinate axes in ion inertial lengths ---
    ny, nx = bz.shape
    x = np.arange(nx) * uc.cell_to_di
    y = np.arange(ny) * uc.cell_to_di

    # --- Plot ---
    ncols = 2 if dens is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    # Bz panel with ψ contours
    ax = axes[0]
    vlim = np.percentile(np.abs(bz), 99)
    im = ax.pcolormesh(x, y, bz, cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto")
    fig.colorbar(im, ax=ax, label=r"$B_z / B_0$")
    if psi is not None:
        ax.contour(x, y, psi, levels=20, colors="k", linewidths=0.5, alpha=0.6)
    ax.set_xlabel(r"$x \; [d_i]$")
    ax.set_ylabel(r"$y \; [d_i]$")
    ax.set_title(rf"$B_z$  —  $t = {t_ion:.2f}\,\Omega_{{ci}}^{{-1}}$")
    ax.set_aspect("equal")

    # Density panel
    if dens is not None:
        ax = axes[1]
        pos = dens[dens > 0]
        vmax_d = np.percentile(pos, 99) if len(pos) else 1.0
        vmin_d = max(pos.min(), vmax_d * 1e-3) if len(pos) else 1e-3
        im2 = ax.pcolormesh(
            x, y, dens,
            norm=mcolors.LogNorm(vmin=vmin_d, vmax=vmax_d),
            cmap="plasma",
            shading="auto",
        )
        fig.colorbar(im2, ax=ax, label=dens_label)
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
