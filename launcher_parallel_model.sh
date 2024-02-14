#!/bin/bash -l
#SBATCH --job-name=spack_parallel
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=28
#SBATCH -p batch

## Command(s) to run (example):
# module load gnu-parallel/2019.03.22

# source ${SCRATCH}fenicsx-iris-gompi-32-20230116-r2/bin/env-fenics.sh

source $SCRATCH/spack/share/spack/setup-env.sh
spack env activate fenicsx-main-20230214

export WDIR=$SCRATCH/stochastic_model
cd $WDIR

# set number of jobs based on number of cores available and number of threads per job
export JOBS_PER_NODE=$(( $SLURM_CPUS_ON_NODE / $SLURM_CPUS_PER_TASK ))

echo "spack env: fenicsx-main-20230214"
echo $SLURM_CPUS_ON_NODE
echo $SLURM_CPUS_PER_TASK
echo $JOBS_PER_NODE 
echo "batch, n=8, c=1, t=4:00:00,0..2"

# parallel --jobs $JOBS_PER_NODE --slf hostfile --wd $WDIR --joblog task.log --resume --progress -a task.lst sh run-blast.sh {} output/{/.}.blst $SLURM_CPUS_PER_TASK

parallel --jobs $JOBS_PER_NODE srun -n 8 -c 1 python3 random_ahc_tensor.py {} ::: {0..2}

