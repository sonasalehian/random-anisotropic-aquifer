#!/bin/bash -l
#SBATCH --job-name=monte_carlo_runner
#SBATCH --partition batch
#SBATCH --time=03:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out
set -e

# TODO: Automate this.
# Instructions on job sizing (aion).
#
# One run takes ~10 minutes on aion using 32 tasks (MPI ranks) per run.
# One node (128 cores) can run 4 jobs simultaneously.
# Therefore one node can execute ~24 runs each hour.
# Define the total number of runs (e.g. 2000) and the desired number of nodes
# (e.g. 32 - max 64 on batch).
# Wall time = 2000/(32 * 24) = 2.6 hours
# 
# Instructions on GNU parallel sizing (aion).
#
# The parameter passed to --jobs should be equal to
# (nodes*jobs per node)/(mpi ranks per run).

source ../setup-env.sh
../print-env.sh

echo "Starting parallel..."

PARALLEL_JOBLOG_FILE="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-parallel.txt"
parallel --delay 0.2 --jobs 128 \
    --joblog ${PARALLEL_JOBLOG_FILE} \
    "srun --output=logs/%x-%j/%x-%j-{}.txt --exclusive -N 1 -n 32 python3 single_run.py {}" \
    ::: {0..2000}

echo "Finished."
