#!/bin/bash -l
#SBATCH --job-name=model
#SBATCH -p batch
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=logs/%x-%j.out
set -e

source ../setup-env.sh
../print-env.sh

srun python model.py
srun python read_write_checkpoints.py
