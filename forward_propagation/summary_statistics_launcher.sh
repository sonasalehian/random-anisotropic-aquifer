#!/bin/bash -l
#SBATCH --job-name=summary_statistics
#SBATCH -p batch
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x-%j.out
set -e

source ../setup-env.sh
../print-env.sh

srun python summary_statistics.py
