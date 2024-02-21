import basix
import basix.ufl
import dolfinx
import numpy as np
from mpi4py import MPI

import adios4dolfinx


mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([100, 150, 40])],
                        [20, 6, 6], cell_type=dolfinx.mesh.CellType.tetrahedron)  
el = basix.ufl.element("Lagrange", "tetrahedron", 1)
V = dolfinx.fem.functionspace(mesh, el)
uh = dolfinx.fem.Function(V)


def f(x):
    return x[0]**2+x[1]**2

uh.interpolate(f)
uh.name = "uh"

filename = 'output/mesh_checkpoint.bp'

adios4dolfinx.write_mesh(mesh, filename)
adios4dolfinx.write_function(uh, filename, time=0.0)
