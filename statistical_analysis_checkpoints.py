import os, sys
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

import basix
import basix.ufl
import dolfinx
from mpi4py import MPI
import numpy as np

import adios4dolfinx

from default_parameters import create_default_parameters

parameters = create_default_parameters()
t = parameters["t"]
T = parameters["T"]
num_steps = parameters["num_steps"]
dt = T / num_steps  # time step size
T2 = parameters["T2"]
num_steps2 = parameters["num_steps2"]
dt2 = T2 / num_steps2

random_folder = 'random_r'
n_outputs = 2400
n_0 = 2200
filename = f'./output/{random_folder}/random_ahc_{n_0}/los_submesh_checkpoint.bp'
engine = "BP4"
MPI.COMM_WORLD.Barrier()
submesh = adios4dolfinx.read_mesh(
    MPI.COMM_WORLD, filename, engine, dolfinx.mesh.GhostMode.shared_facet
)
U_sub = dolfinx.fem.functionspace(submesh, basix.ufl.element("Lagrange", "tetrahedron", 1))
print(U_sub.dofmap.index_map.size_local)
print(U_sub.dofmap.index_map.num_ghosts)
u_loss = [dolfinx.fem.Function(U_sub) for _ in range(n_0, n_outputs)]
print(len(u_loss))
u_los = dolfinx.fem.Function(U_sub)
u_los_mean = dolfinx.fem.Function(U_sub)

sub_file_vtx = dolfinx.io.VTXWriter(submesh.comm, f'./output/{random_folder}/final_mean{n_0}-{n_outputs}.bp', [u_los_mean], engine)

filename_mean = f'./output/{random_folder}/mean_for_std{n_0}-{n_outputs}.bp'
adios4dolfinx.write_mesh(submesh, filename_mean)


for i in range(num_steps):
    t += dt
    if (i+1) % 20 == 0:
        # print("time_step:", i)
        u_los_mean.x.array[:] = 0
        for n in range(n_0, n_outputs+1):
            # print("n:", n)
            u_los.name = "u_n_sub"
            filename = f'./output/{random_folder}/random_ahc_{n}/los_submesh_checkpoint.bp'
            adios4dolfinx.read_function(u_los, filename, engine, time=t)
            u_los_mean.x.array[:] += u_los.x.array
            u_los_mean.x.scatter_forward()

        u_los_mean.x.array[:] /= (n_outputs+1-n_0)
        sub_file_vtx.write(t)
        adios4dolfinx.write_function(u_los_mean, filename_mean, time=t)

for i in range(num_steps2):
    t += dt2
    if (i+1) % 20 == 0:
        # print("time_step:", i)
        u_los_mean.x.array[:] = 0
        for n in range(n_0, n_outputs+1):
            u_los.name = "u_n_sub"
            filename = f'./output/{random_folder}/random_ahc_{n}/los_submesh_checkpoint.bp'
            adios4dolfinx.read_function(u_los, filename, engine, time=t)
            u_los_mean.x.array[:] += u_los.x.array
            u_los_mean.x.scatter_forward()

        u_los_mean.x.array[:] /= (n_outputs+1-n_0)
        sub_file_vtx.write(t)
        adios4dolfinx.write_function(u_los_mean, filename_mean, time=t)

sub_file_vtx.close()
