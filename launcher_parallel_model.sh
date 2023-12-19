#!/bin/bash -l
#SBATCH --job-name=my_parallel_job
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=error.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --ntasks-per-node=14
#SBATCH -p batch

## Command(s) to run (example):
# module load gnu-parallel/2019.03.22

source ${SCRATCH}fenicsx-iris-gompi-32-0.7.2/bin/env-fenics.sh

export WDIR=${HOME}/stochastic_model
cd $WDIR

# # set number of jobs based on number of cores available and number of threads per job
export JOBS_PER_NODE=$(( $SLURM_CPUS_ON_NODE / $SLURM_CPUS_PER_TASK ))

# Read parameters from files
random_value=$(cat hpc_output/random_values.txt)
output_directory=$(cat hpc_output/output_directories.txt)

# echo $SLURM_JOB_NODELIST |sed s/\,/\\n/g > hostfile

# parallel --jobs $JOBS_PER_NODE --slf hostfile --wd $WDIR --joblog task.log --resume --progress -a task.lst sh run-blast.sh {} output/{/.}.blst $SLURM_CPUS_PER_TASK

parallel --jobs $JOBS_PER_NODE --wd $WDIR --joblog task.log --resume --progress -a <(paste -d' ' <(echo "$param1") <(echo "$param2")) python3 random_ihc.py {} $SLURM_CPUS_PER_TASK


# # Run statistical analysis
# python3 statistical_analysis.py hpc_output/random_values.txt

# # Run model.py in parallel
# cat "$file_path" | parallel python3 random_ihc.py {}

# # Read parameters from files
# random_value=$(cat hpc_output/random_values.txt)
# output_directory=$(cat hpc_output/output_directories.txt)

# # Run the script for each pair of parameters
# parallel -a <(paste -d' ' <(echo "$param1") <(echo "$param2")) -j4 python3 random_ihc.py

# echo "Results are calculated"

