"""Do the 6 flights actually see the domain the same way?

Motivated by the single-flight-vs-joint comparison just set up
(run_single_flight_inversions.sh / single_vs_joint_check.py): a day's
posterior category flux is only meaningful to compare against another day's
if both days' data actually have comparable *sensitivity* to that category's
source locations. A flight whose footprints barely cross the landfill/WWTP
region has a landfill/wastewater posterior that's essentially still the
prior, not a real, data-informed estimate -- comparing it at face value
against a well-sampled day would be comparing an estimate to noise.

This is a no-resolve check, cheap on purpose: it needs only each flight's raw
Jacobian (one streamed pass each, via JacobianFile.cell_column_sums -- no
operator materialization, no solve) plus the prior emission fields already
used to build the inversion's prior. Two questions:

1. Spatial pattern: is each flight's total per-cell sensitivity map (summed
   over its own receptors) similar in *shape* to the other flights', or does
   flight-to-flight variation in flight path / wind / mixing height make each
   day look at a genuinely different part of the domain?
2. Per-category signal: for each flight and each source category, how much
   "expected explained enhancement" (sensitivity x prior emission, summed
   over core cells) does that flight's data actually carry? A day with near-
   zero signal for a category has an uninformative, prior-driven posterior
   for it, regardless of what the number says.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, ".")

from adapters.gridded_state import GriddedState
from adapters.jacobian_operator import JacobianFile
from goe.config import Config
from halo_oe.emissions import group_priors_on_grid
from halo_oe.groups import keyword_map_from_config

CONFIG_PATH = "config.ini"
FLIGHTS = ["20230726_1", "20230726_2", "20230728_1", "20230728_2", "20230805", "20230809"]


def main():
    cfg = Config(CONFIG_PATH)
    jac_dir = cfg.get("jacobian", "dir")
    bbox = cfg.get_literal("domain", "bbox")
    emissions_path = cfg.get("emissions", "path")
    inventory = cfg.get("emissions", "inventory")
    kwmap = keyword_map_from_config(cfg)

    grid = None
    core = None
    sens_by_flight = {}
    for fid in FLIGHTS:
        jf = JacobianFile(os.path.join(jac_dir, f"{fid}.nc"))
        if grid is None:
            grid = jf.grid
            core = GriddedState(grid, grid.bbox_mask(*bbox), name="core")
        elif jf.grid.shape != grid.shape:
            raise ValueError(f"{fid}: grid {jf.grid.shape} != {grid.shape}")
        sens = jf.cell_column_sums()   # flat (lat-major), full native grid, one streamed pass
        sens_by_flight[fid] = sens[core.active]   # restrict to the domain actually being solved
        print(f"  {fid}: n_receptors={jf.n_receptors}  "
              f"core-sensitivity sum={sens_by_flight[fid].sum():.4g}")
        jf.close()

    group_fields, assignment = group_priors_on_grid(emissions_path, inventory, grid, keyword_map=kwmap)
    categories = sorted(group_fields)
    print(f"\ncategories: {categories}")

    # --- (1) spatial pattern: pairwise cosine similarity of core-restricted
    # raw sensitivity maps across flights ---------------------------------
    S = np.stack([sens_by_flight[fid] for fid in FLIGHTS])   # (n_flights, n_core_cells)
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    Sn = S / np.where(norms == 0, 1.0, norms)
    sim = Sn @ Sn.T
    print("\npairwise cosine similarity of core-sensitivity maps (raw, unweighted):")
    header = "              " + " ".join(f"{f:>14}" for f in FLIGHTS)
    print(header)
    for i, fid in enumerate(FLIGHTS):
        print(f"{fid:>14} " + " ".join(f"{sim[i, j]:>14.3f}" for j in range(len(FLIGHTS))))

    # --- (2) per-category expected signal per flight ----------------------
    totals = np.zeros((len(FLIGHTS), len(categories)))
    for fi, fid in enumerate(FLIGHTS):
        for ci, cat in enumerate(categories):
            prior_c = group_fields[cat].reshape(-1)[core.active]
            totals[fi, ci] = float(np.sum(sens_by_flight[fid] * prior_c))

    print(f"\nper-flight, per-category expected signal (sensitivity x prior emission, core-integrated):")
    print("              " + " ".join(f"{c:>14}" for c in categories))
    for fi, fid in enumerate(FLIGHTS):
        print(f"{fid:>14} " + " ".join(f"{totals[fi, ci]:>14.4g}" for ci in range(len(categories))))

    frac = totals / np.maximum(totals.sum(axis=1, keepdims=True), 1e-300)
    print(f"\nsame, as each flight's own share across categories (rows sum to 1):")
    print("              " + " ".join(f"{c:>14}" for c in categories))
    for fi, fid in enumerate(FLIGHTS):
        print(f"{fid:>14} " + " ".join(f"{frac[fi, ci]:>14.3f}" for ci in range(len(categories))))

    # --- plots -------------------------------------------------------------
    clat, clon = core.active_lat, core.active_lon
    ext = [clon.min(), clon.max(), clat.min(), clat.max()]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    vmax = max(s.max() for s in sens_by_flight.values())
    for ax, fid in zip(axes.ravel(), FLIGHTS):
        field = np.full(grid.shape, np.nan).reshape(-1)
        field[core.active] = sens_by_flight[fid]
        field = field.reshape(grid.shape)
        rows = np.where(core.mask.any(1))[0]; cols = np.where(core.mask.any(0))[0]
        sub = field[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        pos = sub[sub > 0]
        im = ax.imshow(np.where(sub > 0, sub, np.nan), origin="lower", extent=ext, aspect="auto",
                        cmap="viridis", norm=LogNorm(vmin=max(pos.min(), 1e-12) if pos.size else 1e-12, vmax=vmax))
        ax.set_title(fid); ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, shrink=0.85, label="core sensitivity (log)")
    fig.suptitle("Per-flight raw sensitivity (Σ receptors H[:,cell]), core domain, log scale")
    plt.savefig("runs/flight_sensitivity_maps.png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, origin="lower", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(FLIGHTS))); ax.set_xticklabels(FLIGHTS, rotation=45, ha="right")
    ax.set_yticks(range(len(FLIGHTS))); ax.set_yticklabels(FLIGHTS)
    for i in range(len(FLIGHTS)):
        for j in range(len(FLIGHTS)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                     color="w" if sim[i, j] < 0.6 else "k", fontsize=8)
    ax.set_title("Cosine similarity of core sensitivity maps, flight vs. flight")
    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig("runs/flight_sensitivity_similarity.png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(2 * len(categories), 6) + 2, 5), constrained_layout=True)
    x = np.arange(len(categories)); w = 0.8 / len(FLIGHTS)
    for fi, fid in enumerate(FLIGHTS):
        ax.bar(x + fi * w, totals[fi], width=w, label=fid)
    ax.set_xticks(x + w * (len(FLIGHTS) - 1) / 2); ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("expected signal (sensitivity x prior emission, core-integrated)")
    ax.set_title("Per-flight, per-category sensitivity to the prior -- how much each\n"
                 "day's data actually 'sees' of each source category")
    ax.legend(fontsize=8)
    plt.savefig("runs/flight_sensitivity_by_category.png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    print("\nplots -> runs/flight_sensitivity_maps.png, runs/flight_sensitivity_similarity.png, "
          "runs/flight_sensitivity_by_category.png")


if __name__ == "__main__":
    main()
