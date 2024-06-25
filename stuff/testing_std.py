import os, sys
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

import basix
import basix.ufl
import ufl
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

random_folder = 'test'
n_outputs = 4
n_0 = 0
n = n_0
filename_mean = f'./output/{random_folder}/mean_for_std{n_0}-{n_outputs}.bp'
# filename = f'./output/{random_folder}/random_ahc_{n}/los_submesh_checkpoint.bp'
engine = "BP4"
# MPI.COMM_WORLD.Barrier()
# submesh = adios4dolfinx.read_mesh(
#     MPI.COMM_WORLD, filename, engine, dolfinx.mesh.GhostMode.shared_facet
# )
domain = dolfinx.mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([20, 20, 5])],
                            [20, 6, 6], cell_type=dolfinx.mesh.CellType.tetrahedron)
U_sub = dolfinx.fem.functionspace(domain, basix.ufl.element("Lagrange", "tetrahedron", 1))
print(U_sub.dofmap.index_map.size_local)
print(U_sub.dofmap.index_map.num_ghosts)
u_los = dolfinx.fem.Function(U_sub)
# u_loss = [dolfinx.fem.Function(U_sub) for _ in range(n, n_outputs)]
u_los_mean = dolfinx.fem.Function(U_sub)
u_los_std = dolfinx.fem.Function(U_sub)

sub_file_vtx = dolfinx.io.VTXWriter(domain.comm, f"./output/{random_folder}/final_std{n_0}-{n_outputs}.bp", [u_los_std], engine="BP4")

for i in range(num_steps):
    t += dt
    if (i+1) % 20 == 0:
        adios4dolfinx.read_function(u_los_mean, filename_mean, engine, time=t)
        u_los_std.x.array[:] = 0
        u_los_std.x.scatter_forward()
        for n in range(n_0, n_outputs+1):
            u_los.x.array[:] = n
            u_los_std.x.array[:] += (u_los.x.array - u_los_mean.x.array)**2
            u_los_std.x.scatter_forward()

        n = n_0
        u_los_std.x.array[:] /= (n_outputs+1-n_0)
        u_los_std.x.array[:] = np.sqrt(u_los_std.x.array[:])
        sub_file_vtx.write(t)

for i in range(num_steps2):
    t += dt2
    if (i+1) % 20 == 0:
        adios4dolfinx.read_function(u_los_mean, filename_mean, engine, time=t)
        u_los_std.x.array[:] = 0
        u_los_std.x.scatter_forward()
        for n in range(n_0, n_outputs+1):
            u_los.x.array[:] = n
            u_los_std.x.array[:] += (u_los.x.array - u_los_mean.x.array)**2
            u_los_std.x.scatter_forward()

        n = n_0
        u_los_std.x.array[:] /= (n_outputs+1-n_0)
        u_los_std.x.array[:] = np.sqrt(u_los_std.x.array[:])
        sub_file_vtx.write(t)

sub_file_vtx.close()
