import os, sys
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

import basix
import basix.ufl
import dolfinx
from mpi4py import MPI
import numpy as np
import adios4dolfinx
from default_parameters import parameters

t = parameters["t"]
T = parameters["T"]
num_steps = parameters["num_steps"]
dt = T / num_steps  # time step size
T2 = parameters["T2"]
num_steps2 = parameters["num_steps2"]
dt2 = T2 / num_steps2

random_folder = 'random_r'
n = 0
filename = f'../output/{random_folder}/random_ahc_{n}/los_submesh_checkpoint.bp'
engine = "BP4"
MPI.COMM_WORLD.Barrier()
submesh = adios4dolfinx.read_mesh(
    MPI.COMM_WORLD, filename, engine, dolfinx.mesh.GhostMode.shared_facet
)
U_sub = dolfinx.fem.functionspace(submesh, basix.ufl.element("Lagrange", "tetrahedron", 1))
# print(U_sub.dofmap.index_map.size_local)
# print(U_sub.dofmap.index_map.num_ghosts)
u_los = dolfinx.fem.Function(U_sub)
output = dolfinx.fem.Function(U_sub)

sub_file_vtx = dolfinx.io.VTXWriter(submesh.comm, f'../output/{random_folder}/ahc_output_{n}.bp', [output], engine)

for i in range(num_steps):
    t += dt
    output.x.array[:] = 0
    output.x.scatter_forward()
    if (i+1) % 20 == 0:
        u_los.name = "u_n_sub"
        adios4dolfinx.read_function(u_los, filename, engine, time=t)
        output.x.array[:] = u_los.x.array
        output.x.scatter_forward()
        sub_file_vtx.write(t)

for i in range(num_steps2):
    t += dt2
    output.x.array[:] = 0
    output.x.scatter_forward()
    if (i+1) % 20 == 0:
        u_los.name = "u_n_sub"
        adios4dolfinx.read_function(u_los, filename, engine, time=t)
        output.x.array[:] = u_los.x.array
        output.x.scatter_forward()
        sub_file_vtx.write(t)

sub_file_vtx.close()
