#!/bin/bash

# Remove existing result.txt
rm hpc_output/result.txt

# Generate random parameters
python3 random_parameters.py

# Set the file path
file_path='hpc_output/random_values.txt'

# Run statistical analysis
python3 statistical_analysis.py hpc_output/random_values.txt

# Run function.py in parallel
cat "$file_path" | parallel python3 function.py {}

echo "Results are calculated"

# Run statistical analysis for the result
python3 statistical_analysis.py hpc_output/result.txt
