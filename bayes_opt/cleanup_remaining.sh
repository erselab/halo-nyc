#!/bin/bash
# Wait for the plot_all_tuning batch to finish, then remove factors.npz from
# the remaining (non-top-5) 0.04/0.05 ppm bundles it was still processing.
cd /scratch/scrowel3/halo-nyc/bayes_opt
LOG=runs/cleanup_remaining.log
: > "$LOG"

while ! grep -q "ALL DONE" runs/plot_all_tuning.log 2>/dev/null; do
    sleep 60
done
echo "$(date '+%H:%M:%S') batch finished, cleaning up remaining bundles" | tee -a "$LOG"

KEEP="tune_0.025ppm_1.5km tune_0.025ppm_1km tune_0.025ppm_2km tune_0.025ppm_3km tune_0.02ppm_3km"
freed=0
count=0
for d in runs/tune_0.04ppm_* runs/tune_0.05ppm_*; do
    name=$(basename "$d")
    keep=false
    for k in $KEEP; do [ "$name" = "$k" ] && keep=true; done
    if [ "$keep" = false ] && [ -f "$d/factors.npz" ]; then
        sz=$(stat -c %s "$d/factors.npz")
        rm -f "$d/factors.npz"
        freed=$((freed+sz))
        count=$((count+1))
    fi
done
echo "$(date '+%H:%M:%S') removed factors.npz from $count more bundles, freed $(numfmt --to=iec $freed)" | tee -a "$LOG"
du -sh runs/ >> "$LOG" 2>&1
echo "CLEANUP DONE" | tee -a "$LOG"
