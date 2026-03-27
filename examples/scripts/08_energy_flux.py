"""
08_energy_flux.py
=================
Y-averaged energy flux profiles for a chosen species and step.

Seven panels arranged in a 3×3 grid (last two slots empty):
  Row 1 — Bulk KE flux, Internal energy flux, Enthalpy flux
  Row 2 — Heat flux,    Poynting flux,        Total particle flux
  Row 3 — ½Q_raw residual check (should be ~0 if identity holds)

All quantities in ion units [n0 mi vAi³].  Profiles are averaged over y
and plotted against x in di.

The ½Q_raw identity check verifies:
    ½ Q_raw_x  =  q_KE_x + q_enth_x + q_IE_x + q_heat_x

A non-zero residual indicates a numerical issue or missing datasets.

Requires a params file for the chosen step (species mass, charge, and n0).

Usage
-----
    python examples/scripts/08_energy_flux.py /path/to/output/ --step 10 --species 2
    python examples/scripts/08_energy_flux.py /path/to/output/   # last step, ions
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import pytrist


def _yavg(arr: np.ndarray) -> np.ndarray:
    """Return the y-average of a 2-D or 3-D array (result is 1-D in x)."""
    a = arr[0] if arr.ndim == 3 else arr
    return a.mean(axis=0)


def main(output_dir: str, step: int | None, species_id: int) -> None:
    sim = pytrist.Simulation(output_dir)
    uc  = sim.unit_converter

    if step is None:
        step = sim.steps[-1]

    p     = sim.params(step)
    t_ion = uc.time(p.time)
    print(f"Step {step}  |  t = {p.time:.1f} / ωpe  =  {t_ion:.3f} / Ωci")

    ef  = sim.energy_flux(step)
    flds = sim.fields(step)

    label = "electrons" if species_id == 1 else f"ions (sp {species_id})"
    print(f"Species: {label}")

    # --- x-axis in di ---
    sample = flds.bz
    if sample.ndim == 3:
        sample = sample[0]
    ny, nx = sample.shape
    x = np.arange(nx) * uc.cell_to_di

    # --- Compute all x-component fluxes, y-averaged ---
    ke_x   = _yavg(ef.bulk_ke_flux(species_id, units="ion")["x"])
    ie_x   = _yavg(ef.internal_energy_flux(species_id, units="ion")["x"])
    enth_x = _yavg(ef.enthalpy_flux(species_id, units="ion")["x"])
    heat_x = _yavg(ef.heat_flux(species_id, units="ion")["x"])
    S_x    = _yavg(ef.poynting_flux(units="ion")["x"])
    tot_x  = _yavg(ef.total_particle_energy_flux(species_id, units="ion")["x"])

    # --- ½Q_raw identity check ---
    # total_particle_energy_flux = q_KE + q_enth + q_heat
    # internal_energy_flux is separate; ½Q_raw = tot_prtl + ie
    q_raw_half = tot_x + ie_x
    residual   = q_raw_half - (ke_x + enth_x + ie_x + heat_x)

    panels = [
        (ke_x,    r"$q_{\rm KE}$",   "Bulk KE flux"),
        (ie_x,    r"$q_{\rm IE}$",   "Internal energy flux"),
        (enth_x,  r"$q_{\rm enth}$", "Enthalpy flux"),
        (heat_x,  r"$q_{\rm heat}$", "Heat flux"),
        (S_x,     r"$S_x$",          "Poynting flux"),
        (tot_x,   r"$Q_{\rm prtl}$", "Total particle flux"),
        (residual, r"residual",       r"$\frac{1}{2}Q_{\rm raw} - (q_{\rm KE} + q_{\rm enth} + q_{\rm IE} + q_{\rm heat})$"),
    ]

    # --- Also plot ½Q_raw on the identity-check panel for reference ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(
        rf"{label}  —  $t = {t_ion:.2f}\,\Omega_{{ci}}^{{-1}}$  "
        r"(y-averaged, $x$-flux, ion units $[n_0 m_i v_{Ai}^3]$)",
        fontsize=12,
    )

    all_axes = axes.flat
    for i, (data, ylabel, title) in enumerate(panels):
        ax = next(all_axes)
        ax.plot(x, data, lw=1.5)
        if i == 6:
            # Also show ½Q_raw on the residual panel
            ax.plot(x, q_raw_half, lw=1, ls="--", alpha=0.6, label=r"$\frac{1}{2}Q_{\rm raw}$")
            ax.axhline(0, color="k", lw=0.6, ls=":")
            ax.legend(fontsize=8)
        ax.set_xlabel(r"$x \; [d_i]$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)

    # Hide the two unused axes (panels 8 and 9)
    for ax in [next(all_axes), next(all_axes)]:
        ax.set_visible(False)

    out = f"energy_flux_sp{species_id}_step{step:05d}.png"
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
    parser.add_argument("--species", type=int, default=2,
                        help="Species ID (default: 2 = ions)")
    args = parser.parse_args()
    main(args.output_dir, args.step, args.species)
