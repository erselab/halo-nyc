#!/bin/bash
#SBATCH --job-name=run_halo_tuning
#SBATCH --ntasks=1
#SBATCH --mem 100G
#SBATCH --cpus-per-task=16
#SBATCH -N 1
#SBATCH --time=01:000:00
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -p atmos2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user="sean.crowell@rochester.edu"
#SBATCH --exclusive

eval "$(conda shell.bash hook)"
cd /scratch/scrowel3/halo-nyc/bayes_opt 
conda activate analysis

echo "TUNING RUN: MDM_SIG"${1}"ppm MDM_LAM"${2}"km"
python run_halo.py config.ini --tune --set observations.mdm_stddev=$1 --set observations.mdm_correlation_length_km=$2 --save tune_${1}ppm_${2}km 
