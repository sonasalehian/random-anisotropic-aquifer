#!/bin/bash -l
#SBATCH --job-name=angle_model
#SBATCH -p batch
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/%x-%j.out
set -e

source ../setup-env.sh
../print-env.sh

python generate_data_from_histogram.py
python model_selection.py
python calibration.py
