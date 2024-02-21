#!/bin/bash -l
#SBATCH --job-name=spack_parallel
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=28
#SBATCH -p batch

## Command(s) to run (example):
# module load gnu-parallel/2019.03.22

source $SCRATCH/spack/share/spack/setup-env.sh

spack env activate fenicsx-main-20230206

export WDIR=$SCRATCH/stochastic_model
cd $WDIR

# set number of jobs based on number of cores available and number of threads per job
export JOBS_PER_NODE=$(( $SLURM_CPUS_ON_NODE / $SLURM_CPUS_PER_TASK ))

echo "fenicsx-main-20230206 env"
echo $SLURM_CPUS_ON_NODE
echo $SLURM_CPUS_PER_TASK
echo $JOBS_PER_NODE 
echo "Spack, batch, n=14, c=1, t=16:00:00,0..100"

# parallel --jobs $JOBS_PER_NODE --slf hostfile --wd $WDIR --joblog task.log --resume --progress -a task.lst sh run-blast.sh {} output/{/.}.blst $SLURM_CPUS_PER_TASK

parallel --jobs $JOBS_PER_NODE srun -n 14 -c 1 python3 random_ahc_tensor.py {} ::: {0..100}
# srun -c 1 -n 14 python3 random_ahc_tensor.py
