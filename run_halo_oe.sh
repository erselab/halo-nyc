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
python run_halo.py config.ini --flights 20230726_1,20230726_2,20230728_1,20230728_2,20230805,20230809 --tune --set observations.mdm_stddev=0.030 --set observations.mdm_correlation_length_km=1.5 
