#!/bin/bash -l
#SBATCH --job-name=checkpoint_parallel_r
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-18:00:00
#SBATCH --ntasks-per-node=28
#SBATCH -p batch
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user

source setup-env.sh

parallel --jobs $JOBS_PER_NODE srun -n 14 -c 1 python3 random_ahc_tensor_checkpoint_r.py {} ::: {2200..2400}
