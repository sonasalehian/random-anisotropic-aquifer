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
dt2 = (T2-T) / num_steps2

n_outputs = 10
n = 0
filename = f'./output/random_s_los/random_ahc_{n}/submesh_checkpoint.bp'
engine = "BP4"
MPI.COMM_WORLD.Barrier()
submesh = adios4dolfinx.read_mesh(
    MPI.COMM_WORLD, filename, engine, dolfinx.mesh.GhostMode.shared_facet
)
W = dolfinx.fem.functionspace(submesh, basix.ufl.element("Lagrange", "tetrahedron", 1))
u_los_mean = dolfinx.fem.Function(W)
U_sub = dolfinx.fem.functionspace(submesh, basix.ufl.element("Lagrange", "tetrahedron", 1))
u_los = dolfinx.fem.Function(U_sub)
u_mean = dolfinx.fem.Function(U_sub)
u_los.name = "u_n_sub"

sub_file_vtx = dolfinx.io.VTXWriter(submesh.comm, f"./output/random_s_los/final_mean.bp", [u_los_mean], engine="BP4")
# sub_file_vtx2 = dolfinx.io.VTXWriter(submesh.comm, f"./output/random_s_los/random_ahc_{n}/submesh_checkpoint2.bp", [u_n_sub], engine="BP4")

for i in range(num_steps):
    t += dt
    if (i+1) % 20 == 0:
        for n in range(n_outputs):
            filename = f'./output/random_s_los/random_ahc_{n}/submesh_checkpoint.bp'
            adios4dolfinx.read_function(u_los, filename, engine, time=t)
            u_mean = ufl.algebra.Sum(u_mean, u_los)
        u_mean = ufl.algebra.Division(u_mean, n_outputs)
        u_mean_expr = dolfinx.fem.Expression(u_mean, W.element.interpolation_points())    
        u_los_mean.interpolate(u_mean_expr)
        sub_file_vtx.write(t)
        # sub_file_vtx2.write(t)
for i in range(num_steps2):
    t += dt2
    if (i+1) % 20 == 0:
        for n in range(n_outputs):
            filename = f'./output/random_s_los/random_ahc_{n}/submesh_checkpoint.bp'
            adios4dolfinx.read_function(u_los, filename, engine, time=t)
            u_mean = ufl.algebra.Sum(u_mean, u_los)
        u_mean = ufl.algebra.Division(u_mean, n_outputs)
        u_mean_expr = dolfinx.fem.Expression(u_mean, W.element.interpolation_points())    
        u_los_mean.interpolate(u_mean_expr)
        sub_file_vtx.write(t)
        # sub_file_vtx2.write(t)


sub_file_vtx.close()
# sub_file_vtx2.close()
