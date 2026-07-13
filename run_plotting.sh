#!/bin/bash
#SBATCH --job-name=run_halo_tuning
#SBATCH --ntasks=1
#SBATCH --mem 100G
#SBATCH --cpus-per-task=16
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -p atmos2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user="sean.crowell@rochester.edu"
#SBATCH --exclusive

eval "$(conda shell.bash hook)"
cd /scratch/scrowel3/halo-nyc/bayes_opt 
conda activate analysis
bash ensure_plotting.sh
