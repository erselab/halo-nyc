#!/usr/bin/env bash
# Single-day inversions for all 6 flights, using leg-offset background
# fitting and the "best" prior/observational parameters this investigation
# has actually found -- run twice per flight, once with outlier filtering
# off (the current, always-used setting) and once on, for direct comparison.
#
# "Best parameters," honestly stated: most knobs tested in this investigation
# were CONFIRMED at their current defaults, not improved on --
# mdm_stddev/measurement_stddev (Sec 1's 49-job sweep), category_spatial
# natural_gas=5km (Sec 15, Sec 22.3), category_uncertainty default=1.0. The
# one parameter with a motivated alternative is mdm_correlation_length_km:
# Sec 14b's empirical estimate is ~2.25km vs. the current 1.5km -- used here,
# but Sec 24's own suggestions list flags it explicitly as preliminary, not
# implementation-ready (only 15 events, confounded with leg-level bias at
# longer lengths per Sec 15). Treat this run as a real test of that
# parameter across more data, not as adopting a settled value.
#
# Outlier filtering has never been turned on anywhere in this investigation
# (outlier_threshold=0 throughout, confirmed directly -- see the "is the
# outlier filter on" check). No prior run establishes what threshold to use
# if turned on; outlier_threshold=3.0 (a standard gross-error cutoff) with
# outlier_kind=innovation (already the configured kind, Sec 6's own
# recommended one) is used here as a reasonable default -- adjust and rerun
# if a different threshold is wanted.
#
# Base config is the reference bundle's own saved config.ini (the exact
# settings behind every diagnostic in RESIDUAL_INVESTIGATION.md), with the
# two stale directory paths overridden (same fix applied throughout this
# investigation) and use_leg_offsets pinned explicitly for robustness against
# base-config drift, matching run_single_flight_inversions.sh's convention.
#
# 12 real solves (6 flights x 2 outlier settings) -- run in the background or
# via sbatch if your session has a wall-time limit.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3
BASE_CONFIG=runs/legtest_legoffset_6flight/config.ini
OUT_BASE="$(pwd)/runs"
JAC_DIR=/scratch/scrowel3_lab/halo-nyc/stilt/harvard_jacobians
FLIGHT_DATA_DIR=/scratch/scrowel3_lab/halo-nyc/flight_data
MDM_CORR_LEN_KM=2.25
OUTLIER_THRESHOLD_ON=3.0

FLIGHTS=(20230726_1 20230726_2 20230728_1 20230728_2 20230805 20230809)

for fid in "${FLIGHTS[@]}"; do
    for variant in off on; do
        if [ "$variant" = "off" ]; then
            thresh=0
        else
            thresh=$OUTLIER_THRESHOLD_ON
        fi
        save_name="single_${fid}_bestparams_outlier_${variant}"
        echo "=== ${fid}  outlier=${variant} (threshold=${thresh}) -> ${save_name} ==="
        "$PYTHON" run_halo.py "$BASE_CONFIG" \
            --flights "$fid" \
            --save "${OUT_BASE}/${save_name}" \
            --set jacobian.dir="$JAC_DIR" \
            --set background.flight_data_dir="$FLIGHT_DATA_DIR" \
            --set background.use_leg_offsets=true \
            --set observations.mdm_correlation_length_km="$MDM_CORR_LEN_KM" \
            --set observations.outlier_kind=innovation \
            --set observations.outlier_threshold="$thresh"
    done
done

echo
echo "done -- bundles written to:"
for fid in "${FLIGHTS[@]}"; do
    for variant in off on; do
        echo "  runs/single_${fid}_bestparams_outlier_${variant}"
    done
done
