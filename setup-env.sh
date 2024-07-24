#!/bin/bash -l
# This file is sourced at the start of job launch scripts and can be adjusted
# to your environment.
source $SCRATCH/spack/share/spack/setup-env.sh
spack env activate aquifer_env

# Workarounds for broken Python module find for gmsh
export PYTHONPATH=$SPACK_ENV/.spack-env/view/lib64/:$PYTHONPATH

# Output useful information at start of job
spack env status
scontrol show job $SLURM_JOB_ID
