#!/bin/bash -l
#SBATCH --job-name=checkpoint_parallel
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=28
#SBATCH -p batch

# JSH: Useful.
# SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
# SBATCH --mail-user=email@ufl.edu       # Where to send mail

source $SCRATCH/spack/share/spack/setup-env.sh

# JSH: Could you put the spack.yml under version control?
spack env activate fenicsx-main-20230214

# JSH: Add to requirements.txt file. Would also recommend changing @main to a
# specific commit hash.
# see e.g. https://stackoverflow.com/questions/16584552/how-to-state-in-requirements-txt-a-direct-github-source
# python3 -m pip install git+https://github.com/jorgensd/adios4dolfinx@main

export WDIR=$SCRATCH/stochastic_model
cd $WDIR

# set number of jobs based on number of cores available and number of threads per job
export JOBS_PER_NODE=$(( $SLURM_CPUS_ON_NODE / $SLURM_CPUS_PER_TASK ))

# JSH: spack env status
echo "fenicsx-main-20230214 env"
# JSH: scontrol show job ${SLURM_JOB_ID} will get you all this information and more.
echo $SLURM_CPUS_ON_NODE
echo $SLURM_CPUS_PER_TASK
echo $JOBS_PER_NODE 
# JSH: scontrol show job ${SLURM_JOB_ID} will get you all this information and more.
echo "Spack, batch, n=14, c=1, t=2:00:00,4..9"

# JSH: Why 4 to 9? And it is hardcoded?
# JSH: Why is 14 hardcoded?
parallel --jobs $JOBS_PER_NODE srun -n 14 -c 1 python3 random_ahc_tensor_checkpoint.py {} ::: {4..9}
