#!/usr/bin/env python3
"""
Batch field overview for Tristan-MP V2 simulations.

Generates PNG figures for all major field quantities (B, E, ExB drift,
bulk flows, density, temperature tensors) for a single output step.
Equivalent to the pytrist_field_overview.ipynb notebook but runs
non-interactively and saves all figures to disk.

Usage
-----
    python field_overview.py /path/to/output 10
    python field_overview.py /path/to/output 10 --fig-dir /scratch/figs/step10
    python field_overview.py /path/to/output 10 --dpi 200

Output directory
----------------
By default figures are saved to::

    <output_dir>/figures/step_<NNNNN>/

Each figure is a PNG named descriptively (e.g. ``01_B_field.png``).
A summary of key upstream diagnostics is printed to stdout.
"""

import argparse
import sys
from pathlib import Path

# ── Parse arguments before importing matplotlib so we can set the backend ────
parser = argparse.ArgumentParser(
    description="Generate pytrist field overview PNGs for one simulation step.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("output_dir", help="Simulation output directory")
parser.add_argument("step", type=int, help="Step number to plot")
parser.add_argument(
    "--fig-dir", default=None,
    help="Directory to save figures (default: <output_dir>/figures/step_NNNNN)",
)
parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
parser.add_argument(
    "--pytrist-path", default=None,
    help="Path to prepend to sys.path (if pytrist is not installed)",
)
args = parser.parse_args()

if args.pytrist_path:
    sys.path.insert(0, args.pytrist_path)

OUTPUT   = args.output_dir
STEP     = args.step
DPI      = args.dpi
FIG_DIR  = (
    Path(args.fig_dir)
    if args.fig_dir
    else Path(OUTPUT) / "figures" / f"step_{STEP:05d}"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"Figures → {FIG_DIR}")

import matplotlib
matplotlib.use("Agg")  # non-interactive — must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pytrist

plt.rcParams.update({
    "figure.dpi": DPI,
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "axes.titlesize": 10,
})

# ── Figure-saving helper ──────────────────────────────────────────────────────
_fig_index = 0

def _savefig(name: str) -> None:
    global _fig_index
    _fig_index += 1
    fname = FIG_DIR / f"{_fig_index:02d}_{name}.png"
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [{_fig_index:02d}] {fname.name}")


# ── Load simulation ───────────────────────────────────────────────────────────
print(f"\nLoading step {STEP} from {OUTPUT}")
sim  = pytrist.Simulation(OUTPUT)
uc   = sim.unit_converter
p    = sim.params(STEP)
flds = sim.fields(STEP, units="ion")
moms = sim.moments(STEP)

nx = int(p["grid:mx0"])
ny = int(p["grid:my0"])
x  = np.arange(nx) * uc.cell_to_di
y  = np.arange(ny) * uc.cell_to_di

t_wci = sim.times[STEP] * uc.wpe_to_wci
print(f"Step {STEP}   t = {sim.times[STEP]:.1f} / wpe  =  {t_wci:.3f} / Wci")
print(f"Grid: {nx} x {ny} cells  =  {x.max():.2f} x {y.max():.2f} di")
print(f"c_to_vAi = {uc.c_to_vAi:.4f}   B0 = {uc.B0:.6f}   n0 = {p.ppc0/2:.0f}")
print()


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _pcolor(ax, arr, cmap, symmetric=False, label="", vmax=None):
    if vmax is None:
        vmax = np.nanpercentile(np.abs(arr), 99.5)
    if symmetric:
        im = ax.pcolormesh(x, y, arr, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.pcolormesh(x, y, arr, cmap=cmap, vmin=0, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(label, fontsize=8)
    ax.set_xlabel("x (di)", fontsize=8)
    ax.set_ylabel("y (di)", fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def _row(titles, arrays, cmap="RdBu_r", symmetric=True, label="", vmax=None, figsize=None):
    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=figsize or (4 * n, 6), tight_layout=True)
    if n == 1:
        axes = [axes]
    for ax, title, arr in zip(axes, titles, arrays):
        _pcolor(ax, arr, cmap=cmap, symmetric=symmetric, label=label, vmax=vmax)
        ax.set_title(title)
        ax.set_aspect("equal")
    return fig, axes


# ── 1. Magnetic field ─────────────────────────────────────────────────────────
print("Plotting magnetic field ...")
bx = flds.bx[0]
by = flds.by[0]
bz = flds.bz[0]

_row(["bx (B0)", "by (B0)", "bz (B0)"], [bx, by, bz], label="B / B0")
plt.suptitle(f"Magnetic field — step {STEP}  (t = {t_wci:.3f} / Wci)", y=1.02)
_savefig("B_field")
print(f"  by upstream (x ~ nx/2): {by[:, nx//2-10:nx//2+10].mean():.4f}  (expect ~ 1)")


# ── 2. Electric field ─────────────────────────────────────────────────────────
print("Plotting electric field ...")
ex = flds.ex[0]
ey = flds.ey[0]
ez = flds.ez[0]

_row(["ex (B0)", "ey (B0)", "ez (B0)"], [ex, ey, ez], label="E / B0")
plt.suptitle(f"Electric field — step {STEP}", y=1.02)
_savefig("E_field")


# ── 3. E×B drift ─────────────────────────────────────────────────────────────
print("Plotting ExB drift ...")
B2      = bx**2 + by**2 + bz**2
B2_safe = np.where(B2 > 1e-4, B2, np.nan)

vExB_x = (ey * bz - ez * by) / B2_safe
vExB_y = (ez * bx - ex * bz) / B2_safe
vExB_z = (ex * by - ey * bx) / B2_safe

_row(["vExB_x (vAi)", "vExB_y (vAi)", "vExB_z (vAi)"],
     [vExB_x, vExB_y, vExB_z], label="v_ExB (vAi)")
plt.suptitle(f"E×B drift velocity — step {STEP}", y=1.02)
_savefig("ExB_drift")

# Use the upper quartile of B2 to define "strong field" cells — robust to
# any overall B magnitude (B0 may differ across simulations).
B2_thresh  = np.nanpercentile(B2, 50)   # median B2; upstream cells are above this
up_mask_b  = B2 > B2_thresh
print(f"  Upstream |vExB_x| (|B|² > median): {np.nanmedian(np.abs(vExB_x[up_mask_b])):.4f} vAi  (expect ~ 0)")
print(f"  Upstream |vExB_z| (|B|² > median): {np.nanmedian(np.abs(vExB_z[up_mask_b])):.4f} vAi  (expect ~ 0)")


# ── 4. Ion bulk flow (field file) ─────────────────────────────────────────────
print("Plotting ion bulk flow (field file) ...")
Vi = flds.bulk_velocity(2)

_row(["Vi_x (vAi)", "Vi_y (vAi)", "Vi_z (vAi)"],
     [Vi["vx"][0], Vi["vy"][0], Vi["vz"][0]], label="v_i (vAi)")
plt.suptitle(f"Ion bulk flow — step {STEP}", y=1.02)
_savefig("Vi_field")


# ── 5. Electron bulk flow (field file) ────────────────────────────────────────
print("Plotting electron bulk flow (field file) ...")
Ve = flds.bulk_velocity(1)

_row(["Ve_x (vAi)", "Ve_y (vAi)", "Ve_z (vAi)"],
     [Ve["vx"][0], Ve["vy"][0], Ve["vz"][0]], label="v_e (vAi)")
plt.suptitle(f"Electron bulk flow — step {STEP}", y=1.02)
_savefig("Ve_field")


# ── 6. ExB vs bulk flow diagnostics (text only) ───────────────────────────────
# Use a percentile-based threshold so the check works for any B normalization.
strong_B = (B2 > B2_thresh) & np.isfinite(vExB_x)

print(f"\nIdeal-MHD check (|B|² > median = {B2_thresh:.4f} B0²):")
if strong_B.sum() == 0:
    print("  WARNING: no finite ExB cells above the B threshold — skipping.")
else:
    rms_ExB = np.nanstd(vExB_x[strong_B])
    rms_Ve  = np.nanstd(Ve["vx"][0][strong_B])
    rms_Vi  = np.nanstd(Vi["vx"][0][strong_B])
    print(f"  RMS |vExB_x| = {rms_ExB:.4f} vAi")
    print(f"  RMS |Ve_x|   = {rms_Ve:.4f} vAi")
    print(f"  RMS |Vi_x|   = {rms_Vi:.4f} vAi")

# upstream_T_code is optional: some simulations don't store it in params.
upstream_T_code = None
for key in ("problem:upstream_T", "problem:T0", "plasma:T0", "upstream_T"):
    try:
        upstream_T_code = float(p[key])
        print(f"  upstream_T = {upstream_T_code:.6g} c²  (from params key '{key}')")
        break
    except (KeyError, TypeError):
        pass

if upstream_T_code is not None and strong_B.sum() > 0:
    v_th_vAi  = np.sqrt(upstream_T_code) * uc.c_to_vAi
    shot_noise = v_th_vAi / np.sqrt(p.ppc0 / 2)
    print(f"  Shot-noise floor ~ {shot_noise:.4f} vAi  (v_th={v_th_vAi:.3f} vAi)")
    if rms_Ve < 2 * shot_noise:
        print("  WARNING: Ve_x is noise-dominated at this step.")
    else:
        mask_e = strong_B & np.isfinite(Ve["vx"][0])
        mask_i = strong_B & np.isfinite(Vi["vx"][0])
        slope_e = np.polyfit(Ve["vx"][0][mask_e], vExB_x[mask_e], 1)
        slope_i = np.polyfit(Vi["vx"][0][mask_i], vExB_x[mask_i], 1)
        print(f"  vExB_x = {slope_e[0]:.3f} x Ve_x + {slope_e[1]:.4f}  (ideal MHD -> +1.0)")
        print(f"  vExB_x = {slope_i[0]:.3f} x Vi_x + {slope_i[1]:.4f}")


# ── 7. Bulk flow: field-file vs moments ───────────────────────────────────────
print("\nPlotting bulk flow comparison (field vs moments) ...")
Vi_moms = moms.bulk_velocity(2, units="ion")
Ve_moms = moms.bulk_velocity(1, units="ion")

comps  = ["vx", "vy", "vz"]
labels = ["x",  "y",  "z"]

for sid, name, Vf, Vm in [(2, "Ion", Vi, Vi_moms), (1, "Electron", Ve, Ve_moms)]:
    vmax = max(
        np.nanpercentile(np.abs(Vf["vx"][0]), 99.5),
        np.nanpercentile(np.abs(Vm["vx"]),    99.5),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), tight_layout=True)
    for col, (comp, lab) in enumerate(zip(comps, labels)):
        _pcolor(axes[0, col], Vf[comp][0], cmap="RdBu_r", symmetric=True,
                label=f"V{lab} (vAi)", vmax=vmax)
        axes[0, col].set_title(f"{name} V{lab} — field file (CIC)")

        _pcolor(axes[1, col], Vm[comp], cmap="RdBu_r", symmetric=True,
                label=f"V{lab} (vAi)", vmax=vmax)
        axes[1, col].set_title(f"{name} V{lab} — moments (NGP)")

        for ax in axes[:, col]:
            ax.set_aspect("equal")

    plt.suptitle(
        f"{name} bulk flow  —  field-file vs particle-moments  —  step {STEP}",
        y=1.01,
    )
    _savefig(f"V{name[0]}_field_vs_moms")

    print(f"  {name} bulk flow  RMS(fld - mom) / RMS(fld):")
    for comp, lab in zip(comps, labels):
        populated = Vm[comp] != 0.0
        if populated.sum() > 100:
            diff_rms = np.sqrt(np.mean((Vf[comp][0][populated] - Vm[comp][populated]) ** 2))
            fld_rms  = np.sqrt(np.mean(Vf[comp][0][populated] ** 2))
            ratio    = diff_rms / fld_rms if fld_rms > 0 else float("nan")
            print(f"    V{lab}: {ratio:.3f}   (0 = perfect; <0.2 typical NGP vs CIC)")


# ── 8. ExB vs bulk flow regression ───────────────────────────────────────────
strong_B2 = (B2 > B2_thresh) & np.isfinite(vExB_z)
print("\nExB vs bulk flow regression:")
mask = strong_B2 & np.isfinite(Ve["vz"][0])
if mask.sum() > 10:
    slope_e = np.polyfit(Ve["vx"][0][mask], vExB_x[mask], 1)
    print(f"  vExB_x = {slope_e[0]:.3f} x Ve_x + {slope_e[1]:.4f} vAi  (n={mask.sum()})")
mask = strong_B2 & np.isfinite(Vi["vx"][0])
if mask.sum() > 10:
    slope_i = np.polyfit(Vi["vx"][0][mask], vExB_x[mask], 1)
    print(f"  vExB_x = {slope_i[0]:.3f} x Vi_x + {slope_i[1]:.4f} vAi  (n={mask.sum()})")


# ── 9. Density ────────────────────────────────────────────────────────────────
print("\nPlotting density ...")
n_e = flds["dens1"][0]
n_i = flds["dens2"][0]

_row(["n_e / n0", "n_i / n0"], [n_e, n_i],
     cmap="inferno", symmetric=False, label="n / n0")
plt.suptitle(f"Number density (upstream ~ 1) — step {STEP}", y=1.02)
_savefig("density")

up = slice(nx // 2 - 15, nx // 2 + 15)
print(f"  n_e upstream: {n_e[:, up].mean():.4f}   n_i upstream: {n_i[:, up].mean():.4f}  (expect ~ 1)")


# ── 10. Electron temperature tensor (field file) ──────────────────────────────
print("\nPlotting electron temperature tensor (field file) ...")
eps = 0.05
n_e_safe = np.where(n_e > eps, n_e, np.nan)

Te = {}
for comp, key in [("xx", "TXX1"), ("yy", "TYY1"), ("zz", "TZZ1"),
                  ("xy", "TXY1"), ("xz", "TXZ1"), ("yz", "TYZ1")]:
    Te[comp] = flds[key][0] / n_e_safe

upstream_Te = (
    upstream_T_code * uc.c_to_vAi ** 2 / uc.mass_ratio
    if upstream_T_code is not None else None
)
up_mask = (n_e > 0.8) & (n_e < 1.5)
cs_mask = (n_e > 1.5) & np.isfinite(Te["xx"])

expect_str = f"  (expect ~ {upstream_Te:.4f})" if upstream_Te is not None else ""
print(f"  Upstream T_e{expect_str}:")
for c in ["xx", "yy", "zz"]:
    print(f"    T_e,{c} = {np.nanmedian(Te[c][up_mask]):.4f} m_i vAi²")
print("  Current-sheet T_e (T_zz > T_xx,yy expected from Ez heating):")
for c in ["xx", "yy", "zz"]:
    print(f"    T_e,{c} = {np.nanmedian(Te[c][cs_mask]):.4f} m_i vAi²")

fig, axes = plt.subplots(2, 3, figsize=(13, 7), tight_layout=True)
for ax, (comp, arr) in zip(axes.flat, Te.items()):
    _pcolor(ax, arr,
            cmap="RdBu_r" if comp[0] != comp[1] else "inferno",
            symmetric=(comp[0] != comp[1]),
            label="T_e (m_i vAi²)")
    ax.set_title(f"T_e,{comp} (m_i vAi²)")
plt.suptitle(f"Electron temperature tensor — step {STEP}", y=1.02)
_savefig("Te_tensor_field")


# ── 11. Ion temperature tensor (field file) ───────────────────────────────────
print("\nPlotting ion temperature tensor (field file) ...")
n_i_safe = np.where(n_i > eps, n_i, np.nan)

Ti = {}
for comp, key in [("xx", "TXX2"), ("yy", "TYY2"), ("zz", "TZZ2"),
                  ("xy", "TXY2"), ("xz", "TXZ2"), ("yz", "TYZ2")]:
    Ti[comp] = flds[key][0] / n_i_safe

expect_str = f"  (expect {upstream_Te:.4f})" if upstream_Te is not None else ""
print(f"  Upstream T_i,xx = {np.nanmedian(Ti['xx'][up_mask]):.4f} m_i vAi²{expect_str}")

fig, axes = plt.subplots(2, 3, figsize=(13, 7), tight_layout=True)
for ax, (comp, arr) in zip(axes.flat, Ti.items()):
    _pcolor(ax, arr,
            cmap="RdBu_r" if comp[0] != comp[1] else "inferno",
            symmetric=(comp[0] != comp[1]),
            label="T_i (m_i vAi²)")
    ax.set_title(f"T_i,{comp} (m_i vAi²)")
plt.suptitle(f"Ion temperature tensor — step {STEP}", y=1.02)
_savefig("Ti_tensor_field")


# ── 12. Scalar temperature from moments ───────────────────────────────────────
print("\nPlotting scalar temperature from moments ...")
T_e = moms.temperature(1, units="ion")
T_i = moms.temperature(2, units="ion")

upstream_T_vAi = (
    upstream_T_code * uc.c_to_vAi ** 2 / uc.mass_ratio
    if upstream_T_code is not None else None
)
expect_str = f"  (expect {upstream_T_vAi:.4f})" if upstream_T_vAi is not None else ""
print(f"  T_e median (populated): {np.median(T_e[T_e > 0]):.4f} m_i vAi²{expect_str}")
print(f"  T_i median (populated): {np.median(T_i[T_i > 0]):.4f} m_i vAi²{expect_str}")

_row(["T_e (m_i vAi²)", "T_i (m_i vAi²)"], [T_e, T_i],
     cmap="inferno", symmetric=False, label="T (m_i vAi²)")
plt.suptitle(f"Scalar temperature from particle moments — step {STEP}", y=1.02)
_savefig("T_scalar_moms")


# ── 13. Diagonal temperature tensor from moments ──────────────────────────────
print("\nPlotting temperature tensor from moments ...")
n_e_moms = moms.density(1)
n_i_moms = moms.density(2)

P_e = moms.temperature_tensor(1, units="ion")
P_i = moms.temperature_tensor(2, units="ion")

n_e_moms_safe = np.where(n_e_moms > 0, n_e_moms, np.mean(n_e_moms) / 1e3)
n_i_moms_safe = np.where(n_i_moms > 0, n_i_moms, np.mean(n_i_moms) / 1e3)

Te_xx = P_e["xx"] / n_e_moms_safe
Te_yy = P_e["yy"] / n_e_moms_safe
Te_zz = P_e["zz"] / n_e_moms_safe
Ti_xx = P_i["xx"] / n_i_moms_safe
Ti_yy = P_i["yy"] / n_i_moms_safe
Ti_zz = P_i["zz"] / n_i_moms_safe

fig, axes = plt.subplots(2, 3, figsize=(13, 7), tight_layout=True)
for ax, arr, title in zip(axes[0], [Te_xx, Te_yy, Te_zz],
                          ["T_e,xx (m_i vAi²)", "T_e,yy (m_i vAi²)", "T_e,zz (m_i vAi²)"]):
    _pcolor(ax, arr, cmap="inferno", symmetric=False, label="T_e (m_i vAi²)")
    ax.set_title(title)
for ax, arr, title in zip(axes[1], [Ti_xx, Ti_yy, Ti_zz],
                          ["T_i,xx (m_i vAi²)", "T_i,yy (m_i vAi²)", "T_i,zz (m_i vAi²)"]):
    _pcolor(ax, arr, cmap="inferno", symmetric=False, label="T_i (m_i vAi²)")
    ax.set_title(title)
plt.suptitle(f"Diagonal temperature from particle moments — step {STEP}", y=1.02)
_savefig("T_tensor_moms")


# ── 14–15. Temperature comparison: field vs moments (2D and 1D) ───────────────
print("\nPlotting temperature comparison (field vs moments) ...")
upstream_T = upstream_T_vAi   # may be None if not in params

species_info = [
    ("Electron", Te, {"xx": Te_xx, "yy": Te_yy, "zz": Te_zz}, Ve, n_e, 1),
    ("Ion",      Ti, {"xx": Ti_xx, "yy": Ti_yy, "zz": Ti_zz}, Vi, n_i, 2),
]

for name, fld_T, moms_T_ij, Vf, n_fld, sid in species_info:
    m_k = float(p[f"particles:m{sid}"])
    mk_over_mi = m_k / uc.mass_ratio

    fld_corr = {
        "xx": fld_T["xx"] - mk_over_mi * Vf["vx"][0] ** 2,
        "yy": fld_T["yy"] - mk_over_mi * Vf["vy"][0] ** 2,
        "zz": fld_T["zz"] - mk_over_mi * Vf["vz"][0] ** 2,
    }

    # 2D comparison
    vmax = max(
        np.nanpercentile(np.abs(fld_corr["xx"]), 99.5),
        np.nanpercentile(np.abs(moms_T_ij["xx"]), 99.5),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), tight_layout=True)
    for col, comp in enumerate(["xx", "yy", "zz"]):
        _pcolor(axes[0, col], fld_corr[comp], cmap="inferno", symmetric=False,
                label="T (m_i vAi²)", vmax=vmax)
        axes[0, col].set_title(f"T_{name[0].lower()},{comp} — field (CIC, V²-corrected)")
        _pcolor(axes[1, col], moms_T_ij[comp], cmap="inferno", symmetric=False,
                label="T (m_i vAi²)", vmax=vmax)
        axes[1, col].set_title(f"T_{name[0].lower()},{comp} — moments (NGP)")
        for ax in axes[:, col]:
            ax.set_aspect("equal")
    plt.suptitle(
        f"{name} temperature  —  field vs moments  —  step {STEP}  (t = {t_wci:.3f} / Wci)",
        y=1.01,
    )
    _savefig(f"T{name[0]}_field_vs_moms_2D")

    # Upstream diagnostics
    up_mask2 = (n_fld > 0.8) & (n_fld < 1.5)
    expect_str = f"  upstream expect ~ {upstream_T:.4f} m_i vAi²" if upstream_T is not None else ""
    print(f"  {name} (m_k/m_i = {mk_over_mi:.3f}){expect_str}:")
    for comp in ["xx", "yy", "zz"]:
        fld_up  = np.nanmedian(fld_corr[comp][up_mask2])
        moms_up = np.nanmedian(moms_T_ij[comp][up_mask2 & (moms_T_ij[comp] > 0)])
        ratio   = moms_up / fld_up if fld_up > 0 else float("nan")
        print(f"    T_{comp}  field={fld_up:.4f}  moms={moms_up:.4f}  moms/field={ratio:.3f}")

    # 1D y-averaged comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), tight_layout=True)
    for ax, comp in zip(axes, ["xx", "yy", "zz"]):
        fld_1d  = np.nanmean(fld_corr[comp],  axis=0)
        moms_1d = np.nanmean(moms_T_ij[comp], axis=0)
        ax.plot(x, fld_1d,  lw=1.5, label="field file (CIC)")
        ax.plot(x, moms_1d, lw=1.5, label="moments (NGP)", alpha=0.8)
        if upstream_T is not None:
            ax.axhline(upstream_T, color="k", lw=1, ls="--", label=f"upstream ({upstream_T:.3f})")
        ax.set_xlabel("x (di)")
        ax.set_ylabel("T (m_i vAi²)")
        ax.set_title(f"T_{name[0].lower()},{comp}")
        ax.legend(fontsize=8)
    plt.suptitle(
        f"{name} temperature — y-averaged — step {STEP}  (t = {t_wci:.3f} / Wci)",
        y=1.02,
    )
    _savefig(f"T{name[0]}_field_vs_moms_1D")

print(f"\nDone. {_fig_index} figures saved to {FIG_DIR}")
