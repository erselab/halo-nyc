#!/bin/bash
#SBATCH --job-name=run_halo_tuning
#SBATCH --ntasks=1
#SBATCH --mem 100G
#SBATCH --cpus-per-task=16
#SBATCH -N 1
#SBATCH --time=02:00:00
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
#python run_halo.py config.ini --tune --set observations.mdm_stddev=$1 --set observations.mdm_correlation_length_km=$2 --save tune_${1}ppm_${2}km 
rm -rf runs/legtest_legoffset_6flight_outliers && MPLBACKEND=Agg /gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3 run_halo.py config.ini --tune --save legtest_legoffset_6flight_outliers \
    --set background.use_leg_offsets=true \
    --set observations.mdm_stddev=0.025 \
    --set observations.mdm_correlation_length_km=1.5 \
    --set observations.outlier_threshold=3.0 \
    --plot-diagnostics