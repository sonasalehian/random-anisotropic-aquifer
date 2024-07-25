#!/bin/bash -l
#SBATCH --job-name=monte_carlo_runner
#SBATCH --partition batch
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out

source ../setup-env.sh
../print-env.sh

parallel --jobs 4 "srun -N 1 -n 32 python3 -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.size)'" ::: {0..3} 
