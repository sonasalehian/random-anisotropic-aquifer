# A stochastic model of an underground aquifer

## Calibration of Bayesian angle model

## Mesh generation

Use gmsh to generate the aquifer mesh:

    srun -n 1 python3 build_mesh.py

The output is written in DOLFINx-compatible XDMF to `output/mesh/*`.

## Software environment

This stochastic part of this study involves a Monte Carlo forward uncertainty
analysis through a finite element model. This is only feasible on a HPC. The
instructions below are designed around a HPC environment with SLURM and Spack
available.

A Spack environment file `spack.yaml` is provided to install the basic
dependencies.

    spack env create aquifer_env spack.yml
    spack env activate aquifer_env
    spack install -j16 # Adjust build threads to your environment.
    python -m pip install -r requirements.txt

To quickly bring up the environment:

    . setup-env.sh

### gmsh workaround (uni.lu)

It is necessary to add the Python gmsh interface location to `PYTHONPATH` on
uni.lu HPC cluster as it is installed in `lib64/` not `lib/`.

    export PYTHONPATH=$SPACK_ENV/.spack-env/view/lib64/:$PYTHONPATH

### adios2 workaroud (spack issue?)

It seems necessary to add the Python adios2 interface location to `PYTHONPATH`
as it is not found via Spack's venv integration.

    export PYTHONPATH=$SPACK_ENV/.spack-env/view
