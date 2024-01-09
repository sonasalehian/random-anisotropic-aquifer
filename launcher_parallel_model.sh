#!/bin/bash -l
#SBATCH --job-name=my_parallel_job
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=07:00:00
#SBATCH --ntasks-per-node=28
#SBATCH -p bigmem

## Command(s) to run (example):
# module load gnu-parallel/2019.03.22

source ${SCRATCH}fenicsx-iris-gompi-32-0.7.2/bin/env-fenics.sh

export WDIR=${HOME}/stochastic_model
cd $WDIR

# set number of jobs based on number of cores available and number of threads per job
export JOBS_PER_NODE=$(( $SLURM_CPUS_ON_NODE / $SLURM_CPUS_PER_TASK ))

echo $SLURM_CPUS_ON_NODE
echo $SLURM_CPUS_PER_TASK
echo $JOBS_PER_NODE 
echo "bigmem, n=4, t=7:00:00"

# parallel --jobs $JOBS_PER_NODE --slf hostfile --wd $WDIR --joblog task.log --resume --progress -a task.lst sh run-blast.sh {} output/{/.}.blst $SLURM_CPUS_PER_TASK

parallel --jobs $JOBS_PER_NODE srun -n 4 -c 1 python3 random_ahc.py {} ::: {6..54}

