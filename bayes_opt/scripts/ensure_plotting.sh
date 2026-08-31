#!/bin/bash
# Make sure the tuning-sweep plotting batch (plot_all_tuning.sh) is running,
# relaunching it detached if it died. Safe to run as many times as you like —
# it refuses to start a second copy, and plot_all_tuning.sh itself skips any
# runs/tune_* bundle that already has all 6 expected PNGs, so a relaunch just
# picks up where the last one left off.
cd /scratch/scrowel3/halo-nyc/bayes_opt || exit 1

if pgrep -f "bash plot_all_tuning.sh" > /dev/null; then
    echo "Already running (pid $(pgrep -f 'bash plot_all_tuning.sh' | head -1)) — not starting another copy."
else
    echo "Not running — relaunching detached (setsid+nohup+disown, survives terminal/session exit)."
    setsid nohup bash plot_all_tuning.sh > /dev/null 2>&1 < /dev/null &
    disown -a
    sleep 2
    echo "Launched pid $(pgrep -f 'bash plot_all_tuning.sh' | head -1)"
fi

echo
echo "--- progress ---"
n=0; incomplete=0
for d in runs/tune_*; do
    [ -f "$d/layout.json" ] || continue
    cnt=$(ls "$d"/*.png 2>/dev/null | wc -l)
    if [ "$cnt" = "6" ]; then n=$((n+1)); else incomplete=$((incomplete+1)); fi
done
echo "complete: $n   remaining: $incomplete"
echo
echo "--- tail of runs/plot_all_tuning.log ---"
tail -8 runs/plot_all_tuning.log
