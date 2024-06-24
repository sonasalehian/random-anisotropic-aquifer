#!/bin/bash -l
#SBATCH --job-name=statistical_analysis_test
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
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

echo "spack env: fenicsx-main-20230214"
echo $SLURM_CPUS_ON_NODE
echo $SLURM_CPUS_PER_TASK
echo $JOBS_PER_NODE 
echo "batch, c=1, t=8:00:00"
echo "0-4"

srun -c 1 python3 testing_mean.py
