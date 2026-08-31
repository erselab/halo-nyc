#!/usr/bin/env bash
# Re-solve each of the 6 flights on its own (single-flight state, not shared
# with the other 5 days) and save each as its own bundle, to compare against
# what the joint 6-flight inversion (runs/legtest_legoffset_6flight) produces
# for that same flight.
#
# Why this matters: the joint inversion fits ONE shared set of per-cell
# category scale factors to all 6 flights' data at once -- it implicitly
# assumes the same underlying emission field (up to those scale factors)
# explains every day. If real emissions vary day to day, that joint fit is a
# compromise that could systematically mismatch every individual day in a way
# that looks exactly like the persistent per-flight residual structure this
# investigation has been chasing, but for a different reason (a modeling
# assumption, not a missing source or bad background). This has never been
# tested -- every substantive run so far (the MDM tuning sweep, every §5-§18
# diagnostic) used the joint 6-flight solve.
#
# Uses runs/legtest_legoffset_6flight/config.ini -- the ACTUAL settings
# behind every diagnostic in RESIDUAL_INVESTIGATION.md -- as the base config,
# not the live config.ini (which had drifted: mdm_stddev=0.03 vs the bundle's
# 0.025, use_leg_offsets=false vs the bundle's true, now fixed back to true in
# the live config too -- see the investigation doc). The two stale directory
# paths recorded in that old bundle (predating the /scratch/scrowel3 ->
# /scratch/scrowel3_lab move) are overridden, and use_leg_offsets is pinned
# explicitly rather than left to whatever the base config happens to say, so
# this is otherwise an exact apples-to-apples comparison: same error model,
# same kriged leg-offset background, same decomposition, same category
# priors -- the only thing that changes is how many flights' data inform the
# shared state.
#
# [plotting] in the base config is already enabled (posterior, residuals,
# background, flux_summary, leg_offsets), so each single-flight bundle gets
# the full diagnostic set automatically, including the new prior-residual
# panel in residuals_map.png.
#
# Run from the bayes_opt directory. Each flight is one real solve (its own
# ~12.6GB Jacobian read + inversion) -- expect this to take a while;
# run in the background or via sbatch if your session has a wall-time limit.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3
BASE_CONFIG=runs/legtest_legoffset_6flight/config.ini
OUT_BASE="$(pwd)/runs"
JAC_DIR=/scratch/scrowel3_lab/halo-nyc/stilt/harvard_jacobians
FLIGHT_DATA_DIR=/scratch/scrowel3_lab/halo-nyc/flight_data

FLIGHTS=(20230726_1 20230726_2 20230728_1 20230728_2 20230805 20230809)

for fid in "${FLIGHTS[@]}"; do
    echo "=== ${fid} ==="
    "$PYTHON" run_halo.py "$BASE_CONFIG" \
        --flights "$fid" \
        --save "${OUT_BASE}/single_${fid}" \
        --set jacobian.dir="$JAC_DIR" \
        --set background.flight_data_dir="$FLIGHT_DATA_DIR" \
        --set background.use_leg_offsets=true
done

echo
echo "done -- single-flight bundles written to:"
for fid in "${FLIGHTS[@]}"; do
    echo "  runs/single_${fid}"
done
