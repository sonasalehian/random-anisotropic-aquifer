#!/bin/bash -l
# Output useful information at start of job
spack env status
scontrol show job $SLURM_JOB_ID
