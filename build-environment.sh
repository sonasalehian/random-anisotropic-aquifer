#!/bin/bash -l
#SBATCH --job-name=build-environment
#SBATCH --partition batch
#SBATCH --time=0-4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out
set -e

source $HOME/spack/share/spack/setup-env.sh

./print-env.sh

spack env activate --create ./spack_env
cp spack.yaml spack_env/
spack concretize
spack install -j16
pip install -r requirements.txt

# Apply patch to mpi4py
INIT_FILE = $(find $SPACK_ENV/.spack-env -type f -name '__init__.py' | grep mpi4py)
patch -u $INIT_FILE -i mpi4py-patch-unilu.patch
