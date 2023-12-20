#!/bin/bash
#SBATCH --job-name=simple_parallel
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --ntasks-per-node=14
#SBATCH -p batch

# ## Command(s) to run (example):
# module load gnu-parallel/2019.03.22
# module load 

source ${SCRATCH}fenicsx-iris-gompi-32-0.7.2/bin/env-fenics.sh

# Remove existing result.txt
rm hpc_output/result.txt

# Generate random parameters
python3 random_parameters.py

# Set the file path
file_path='hpc_output/random_values.txt'

# Run statistical analysis
python3 statistical_analysis.py hpc_output/random_values.txt

# # Run function.py in parallel
cat "$file_path" | parallel python3 function.py {}

# # Read parameters from files
# param=$(cat hpc_output/random_values.txt)


# # Run the script for each pair of parameters
# parallel --jobs $JOBS_PER_NODE -a hpc_output/random_values.txt python3 function.py {}

echo "Results are calculated"

# Run statistical analysis for the result
python3 statistical_analysis.py hpc_output/result.txt
