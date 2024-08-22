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
`v0.22` available at `$HOME/spack`.

A Spack environment file `spack.yaml` is provided to install the basic
dependencies.

    sbatch build-environment.sh

To quickly bring up the environment:

    . setup-env.sh

### MPI issue (uni.lu issue)

Running:

    from mpi4py import MPI

on the uni.lu HPC leads to `error: plugin_load_from_file` etc. This can be
fixed by applying 'mpi4py-patch-unilu.patch` to `mpi4py` `__init__.py` file.

    export INIT_FILE=$(find $SPACK_ENV/.spack-env -type f -name '__init__.py' | grep mpi4py)
    patch -u $INIT_FILE -i mpi4py-patch-unilu.patch

This fix is already applied in `build-environment.sh`.

### gmsh workaround (uni.lu/spack issue)

It is necessary to add the Python gmsh interface location to `PYTHONPATH` on
uni.lu HPC cluster as it is installed in `lib64/` not `lib/`.

    export PYTHONPATH=$SPACK_ENV/.spack-env/view/lib64/:$PYTHONPATH

This fix is already applied in `setup-env.sh`.

### adios2 workaroud (spack issue)

It seems necessary to add the Python adios2 interface location to `PYTHONPATH`
as it is not found via Spack's venv integration.
 
    export PYTHONPATH=$(find $SPACK_ENV/.spack-env -type d -name 'site-packages' | grep venv):$PYTHONPATH

This fix is already applied in `setup-env.sh`.
