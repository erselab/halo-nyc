#!/bin/bash
#SBATCH --job-name=plot_tuning
#SBATCH --ntasks=1
#SBATCH --mem 100G
#SBATCH --cpus-per-task=16
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH -o output_plot.%j
#SBATCH -e error_plot.%j
#SBATCH -p atmos2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user="sean.crowell@rochester.edu"
#SBATCH --exclusive

# Runs the resumable tuning-sweep plotting batch as its own SLURM allocation,
# independent of any interactive session's wall-clock. plot_all_tuning.sh
# skips any runs/tune_* bundle that already has all 6 expected PNGs, so this
# is safe to (re)submit if it ever needs to be resumed again.
cd /scratch/scrowel3/halo-nyc/bayes_opt
bash plot_all_tuning.sh
