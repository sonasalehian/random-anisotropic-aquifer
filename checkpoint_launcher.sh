#!/bin/bash -l
#SBATCH --job-name=checkpoint_parallel
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=28
#SBATCH -p batch

source $SCRATCH/spack/share/spack/setup-env.sh

spack env activate fenicsx-main-20230214

# python3 -m pip install git+https://github.com/jorgensd/adios4dolfinx@main

export WDIR=$SCRATCH/stochastic_model
cd $WDIR

# set number of jobs based on number of cores available and number of threads per job
export JOBS_PER_NODE=$(( $SLURM_CPUS_ON_NODE / $SLURM_CPUS_PER_TASK ))

echo "fenicsx-main-20230214 env"
echo $SLURM_CPUS_ON_NODE
echo $SLURM_CPUS_PER_TASK
echo $JOBS_PER_NODE 
echo "Spack, batch, n=14, c=1, t=2:00:00,0..3"

parallel --jobs $JOBS_PER_NODE srun -n 14 -c 1 python3 random_ahc_tensor_checkpoint.py {} ::: {0..3}

