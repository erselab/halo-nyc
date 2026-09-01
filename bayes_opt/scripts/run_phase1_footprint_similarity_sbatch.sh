#!/bin/bash
#SBATCH --job-name=footprint_phase1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH --time=02:00:00
#SBATCH -o runs/footprint_phase1.%j.out
#SBATCH -e runs/footprint_phase1.%j.err
#SBATCH -p atmos2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user="sean.crowell@rochester.edu"

# Runs Phase 1 of the "attack hypothesis (a)" plan (see RESIDUAL_
# INVESTIGATION.md / the approved plan): footprint_similarity_space_time.py
# extended to all 6 flights, materializing each flight's full (n_receptors x
# n_cells) footprint matrix in turn (~15GB at float32 each, freed before the
# next flight) -- the reason this needs its own allocation rather than
# running inline. As its own SLURM job, this is independent of any
# interactive session's wall-clock, same pattern as plot_all_tuning_sbatch.sh.
#
# Artifacts this produces (everything needed to analyze Phase 1's result,
# regardless of how this async run's stdout/stderr end up being handled):
#   runs/footprint_similarity_phase1_summary.txt   <- clean, guaranteed text
#       artifact: per-flight n_receptors/n_pairs, the half-max decay-length
#       table (the actual Phase 1 deliverable), and the raw 1D decay curve
#       per flight for re-analysis without re-running.
#   runs/footprint_similarity_decay_length.png     <- decay-length bar chart,
#       805/809 highlighted red -- the headline comparison.
#   runs/footprint_similarity_decay_1d.png         <- all 6 flights' decay
#       curves overlaid.
#   runs/footprint_similarity_space_time.png       <- per-flight 2D
#       (distance, time-gap) heatmaps (6 panels).
#   runs/footprint_similarity_time_slices.png      <- per-flight 1D slices
#       at fixed distances.
#   runs/footprint_phase1.<jobid>.out/.err         <- raw stdout/stderr, in
#       case anything needs debugging (progress prints, per-flight timing).

set -euo pipefail
cd /scratch/scrowel3_lab/halo-nyc/bayes_opt
/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3 footprint_similarity_space_time.py
