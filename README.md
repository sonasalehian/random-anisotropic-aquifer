# Stochastic model

# How to use
Create a Docker container, named for instance `dolfinx-checkpoint`. Use the v0.7.0 tag to get the versiov 7 branch of DOLFINx to be compatible with our codes.

    docker run -ti -v $(pwd):/root/shared -w /root/shared --name=dolfinx-checkpoint ghcr.io/fenics/dolfinx/dolfinx:v0.7.0

# Installing dependencies
Install `adios4dolfinx` on top of DOLFINx:

    python3 -m pip install adios4dolfinx[test]@git+https://github.com/jorgensd/adios4dolfinx@v0.7.3

or

    pip install -r requirements.txt

This docker container can be opened later with:

    docker container start -i dolfinx-checkpoint
