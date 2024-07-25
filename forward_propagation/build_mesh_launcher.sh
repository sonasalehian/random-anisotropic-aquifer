#!/bin/bash -l
#SBATCH --job-name=build_mesh
#SBATCH -p batch
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out

source ../setup-env.sh
../print-env.sh

srun python build_mesh.py
