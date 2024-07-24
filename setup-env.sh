#!/bin/bash -l
# This file is sourced at the start of job launch scripts and can be adjusted
# to your environment.
source $SCRATCH/spack/share/spack/setup-env.sh
spack env activate aquifer_env

# Workaround for broken Python module find for gmsh on uni.lu cluster
export PYTHONPATH=$SPACK_ENV/.spack-env/view/lib64/:$PYTHONPATH

# TODO: Add workaround for broken Python module find for adios2
