#!/bin/bash
# Regenerate all [plotting] diagnostics (posterior, residuals, background, flux_summary)
# for every completed tuning-sweep bundle under runs/tune_*, via --plot-only (no re-solve).
cd /scratch/scrowel3/halo-nyc/bayes_opt
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3
export MPLBACKEND=Agg

LOG=runs/plot_all_tuning.log
EXPECTED="posterior.png residuals_map.png residuals_autocorr.png background.png flux_maps.png flux_totals.png"

is_complete() {
    for f in $EXPECTED; do
        [ -f "$1/$f" ] || return 1
    done
    return 0
}

total=0
for d in runs/tune_*; do
    [ -f "$d/layout.json" ] && total=$((total+1))
done

echo "$(date '+%H:%M:%S')  === resuming batch (skipping already-complete bundles) ===" | tee -a "$LOG"

n=0
for d in runs/tune_*; do
    if [ ! -f "$d/layout.json" ]; then
        echo "$(date '+%H:%M:%S')  SKIP (no bundle): $d" | tee -a "$LOG"
        continue
    fi
    n=$((n+1))
    if is_complete "$d"; then
        echo "$(date '+%H:%M:%S')  [$n/$total] already complete, skipping: $d" | tee -a "$LOG"
        continue
    fi
    echo "$(date '+%H:%M:%S')  [$n/$total] plotting $d" | tee -a "$LOG"
    "$PYTHON" run_halo.py --plot-only "$d" --plot-diagnostics >> "$LOG" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
        echo "$(date '+%H:%M:%S')  FAILED (exit $status): $d" | tee -a "$LOG"
    fi
done
echo "$(date '+%H:%M:%S')  ALL DONE ($n bundles processed)" | tee -a "$LOG"