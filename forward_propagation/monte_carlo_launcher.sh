#!/bin/bash -l
#SBATCH --job-name=monte_carlo_runner
#SBATCH --partition batch
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out

source ../setup-env.sh
../print-env.sh

parallel --jobs 4 srun -n 32 python3 single_run.py {} ::: {0..4} 
