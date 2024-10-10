#!/bin/bash -l
# This file is sourced at the start of job launch scripts and can be adjusted
# to your environment.
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"

source $HOME/spack/share/spack/setup-env.sh
spack env activate -p -d ./spack_env

# Workaround for broken Python module find for gmsh on uni.lu cluster
export PYTHONPATH=$SPACK_ENV/.spack-env/view/lib64/:$PYTHONPATH

# Workaround for broken Python module find for adios2
export PYTHONPATH=$(find $SPACK_ENV/.spack-env -type d -name 'site-packages' | grep venv):$PYTHONPATH

cd -
