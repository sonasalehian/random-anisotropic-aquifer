#!/bin/bash -l
# Output useful information at start of job
scontrol show job $SLURM_JOB_ID
spack env status
git status --short -b
git log --oneline -3
