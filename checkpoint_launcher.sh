#!/bin/bash -l
#SBATCH --job-name=checkpoint_parallel
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:30:00
#SBATCH --ntasks-per-node=28
#SBATCH -p batch
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sona.salehian.001@student.uni.lu

source $SCRATCH/spack/share/spack/setup-env.sh
spack env activate fenicsx-main-20230214

export WDIR=$SCRATCH/stochastic_model
cd $WDIR

spack env status
scontrol show job $SLURM_JOB_ID

parallel --jobs 2 srun -n 14 python3 random_ahc_tensor_checkpoint.py {} ::: {43..46}

