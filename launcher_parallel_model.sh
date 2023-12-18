#!/bin/bash

# Remove existing result.txt
# rm hpc_output/result.txt

# Generate random parameters
python3 random_parameters.py

# Set the file path
file_path='hpc_output/random_values.txt'

# Run statistical analysis
python3 statistical_analysis.py hpc_output/random_values.txt

# # Run model.py in parallel
# cat "$file_path" | parallel python3 random_ihc.py {}

# Read parameters from files
random_value=$(cat hpc_output/random_values.txt)
output_directory=$(cat hpc_output/output_directories.txt)

# Run the script for each pair of parameters
paste -d' ' <(echo "$random_value") <(echo "$output_directory") | xargs -n 2 python3 random_ihc.py

echo "Results are calculated"

# Run statistical analysis for the result
# python3 statistical_analysis.py hpc_output/result.txt
