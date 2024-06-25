#!/bin/bash -l
#SBATCH --job-name=checkpoint_parallel_r
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-18:00:00
#SBATCH --ntasks-per-node=28
#SBATCH -p batch
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sona.salehian.001@student.uni.lu

source $SCRATCH/spack/share/spack/setup-env.sh
spack env activate fenicsx-main-20230214

export WDIR=$SCRATCH/stochastic_model
cd $WDIR

# set number of jobs based on number of cores available and number of threads per job
export JOBS_PER_NODE=$(( $SLURM_CPUS_ON_NODE / $SLURM_CPUS_PER_TASK ))

spack env status
scontrol show job ${SLURM_JOB_ID}

parallel --jobs $JOBS_PER_NODE srun -n 14 -c 1 python3 random_ahc_tensor_checkpoint_r.py {} ::: {801..1000}

