#!/bin/bash -l
#SBATCH --job-name=model
#SBATCH -p batch
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

source ../setup-env.sh
../print-env.sh

srun -c 1 python model.py
