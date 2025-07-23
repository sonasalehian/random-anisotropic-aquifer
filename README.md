# Supplementary material: A random model of anisotropic hydraulic conductivity tailored to the InSAR-based analysis of aquifers 

The `random-anisotropic-aquifer` repository contains code implementing a stochastic model
developed for anisotropic hydraulic conductivity (AHC) of an aquifer system that is
designed around the requirements of InSAR-based analysis of aquifer systems.

For full details, please refer to the corresponding paper: [A random model of
anisotropic hydraulic conductivity tailored to the InSAR-based analysis of
aquifers](https://hdl.handle.net/10993/xxxxx).

This repository was also permanently archived at:
[doi:10.5281/zenodo.14170026](10.5281/zenodo.14170026).

## Citing

Please consider citing this code and related paper if you find it useful.

    @misc{salehian_supplement-ahc_2025,
          title = {Supplementary material: A random model of anisotropic hydraulic conductivity tailored to the InSAR-based analysis of aquifers},
          author = {Salehian Ghamsari, Sona, and Hale, Jack S.},
          month = jul,
          year = {2025},
          doi = {10.5281/zenodo.14170026},
          keywords = {stochastic model, anisotropy hydraulic conductivity, symmetric positive definite, 
	          mixture of von Mises distribution, poroelastic model, FEniCS, finite element methods},
    }

## Authors (alphabetical)

Sona Salehian Ghamsari, University of Luxembourg, Luxembourg.

Jack S. Hale, University of Luxembourg, Luxembourg.

## Description

### Model of principal direction

All scripts necessary to calibrate the principal direction models are available in the
`angle_model/` directory. 

A `.sh` file is included for running tasks on a High-Performance Computing (HPC) system. This 
script enables you to:

1.	Generate outcrop data from the rose diagram using
    `generate_data_from_histogram.py`.
2.	Compare prior models to determine the best fit using `model_selection.py`.
3.	Produce random angles based on the selected model using `calibration.py`.

To execute, simply run:

	sbatch angle_model_launcher.sh

### Model of AHC

All scripts required for generating random AHC tensors are located in the
`tensor_model/` directory. To execute these scripts on an HPC system, use the
provided `tensor_model_launcher.sh` file. This script runs:

1.	`generate_random_ahct.py` to generate the three sets of random AHC tensors.
2.	`plot_elliptics.py` to create elliptical representations for each set.

To run the script on an HPC, use the following command:
	
    sbatch tensor_model_launcher.sh

### Mesh generation

The scripts for generating the aquifer mesh are located in the
`forward_propagation/` directory. 

Use gmsh to generate the aquifer mesh by submitting the job script:
	
	sbatch build_mesh_launcher.sh

The generated mesh is written in a DOLFINx-compatible XDMF format and saved in 
`forward_propagation/output/mesh`.

### Forward uncertainty analysis

The uncertainty in the m

All the following scripts are located in the `forward_propagation/` directory.

Use the following command to run the Monte Carlo simulations for uncertainty
propagation: 
	
    sbatch monte_carlo_launcher.sh

Execute this command to calculate statistics of LOS displacements:

	sbatch summary_statistics_launcher.sh

To simulate the model with Anderson Junction AHC data, use:

	sbatch model_launcher.sh

First, ensure ParaView is installed on your system. Then, use the following
command to generate figures from the statistical analysis:
	
    pvpython 02_[name_of_file].py

## Software environment

This stochastic part of this study involves a Monte Carlo forward uncertainty
analysis through a finite element model. This is only feasible on an HPC. The
instructions below are designed around an HPC environment with SLURM and Spack
`v0.22` available at `$HOME/spack`.

A Spack environment file `spack.yaml` is provided to define the necessary
dependencies. To install them, submit the following SLURM job:

    sbatch build-environment.sh
 
Once the dependencies are installed, you can quickly bring up the environment
by sourcing the provided `setup-env.sh` script:

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

## License

AHC-Stochastic-Model is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version. This program is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
Public License for more details. You should have received a copy of the GNU
Lesser General Public License along with AHC-Stochastic-Model. If not,
see http://www.gnu.org/licenses/.
