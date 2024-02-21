import basix
import basix.ufl
import ufl
import dolfinx
from mpi4py import MPI
import numpy as np

import adios4dolfinx

from default_parameters import parameters


# def get_dtype(in_dtype: np.dtype, complex: bool):
#     dtype: numpy.typing.DTypeLike
#     if in_dtype == np.float32:
#         if complex:
#             dtype = np.complex64
#         else:
#             dtype = np.float32
#     elif in_dtype == np.float64:
#         if complex:
#             dtype = np.complex128
#         else:
#             dtype = np.float64
#     else:
#         raise ValueError("Unsuported dtype")
#     return dtype


t = parameters["t"]
T = parameters["T"]
num_steps = parameters["num_steps"]
dt = T / num_steps  # time step size
T2 = parameters["T2"]
num_steps2 = parameters["num_steps2"]
dt2 = (T2-T) / num_steps2


n = 2
filename = f'./output/random_s_los/random_ahc_{n}/submesh_checkpoint.bp'
engine = "BP4"
MPI.COMM_WORLD.Barrier()
submesh = adios4dolfinx.read_mesh(
    MPI.COMM_WORLD, filename, engine, dolfinx.mesh.GhostMode.shared_facet
)
# dtype = get_dtype(mesh.geometry.x.dtype, complex)
W = dolfinx.fem.functionspace(submesh, basix.ufl.element("Lagrange", "tetrahedron", 1))
u_los_h = dolfinx.fem.Function(W)
U_sub = dolfinx.fem.functionspace(submesh, basix.ufl.element("Lagrange", "tetrahedron", 1))
u_n_sub = dolfinx.fem.Function(U_sub)
u_n_sub.name = "u_n_sub"

sub_file_vtx = dolfinx.io.VTXWriter(submesh.comm, f"./output/random_s_los/random_ahc_{n}/submesh_checkpoint2.bp", [u_los_h], engine="BP4")
sub_file_vtx2 = dolfinx.io.VTXWriter(submesh.comm, f"./output/random_s_los/random_ahc_{n}/submesh_checkpoint2_2.bp", [u_n_sub], engine="BP4")

for i in range(num_steps):
    t += dt
    if (i+1) % 20 == 0:
        adios4dolfinx.read_function(u_n_sub, filename, engine, time=t)
        # print(u_n_sub.x.array)
        u_los = ufl.algebra.Sum(u_n_sub, u_n_sub)
        u_los_expr = dolfinx.fem.Expression(u_los, W.element.interpolation_points())    
        u_los_h.interpolate(u_los_expr)
        sub_file_vtx.write(t)
        sub_file_vtx2.write(t)
for i in range(num_steps2):
    t += dt2
    if (i+1) % 20 == 0:
        adios4dolfinx.read_function(u_n_sub, filename, engine, time=t)
        u_los = ufl.algebra.Sum(u_n_sub, u_n_sub)
        u_los_expr = dolfinx.fem.Expression(u_los, W.element.interpolation_points())    
        u_los_h.interpolate(u_los_expr)
        sub_file_vtx.write(t)
        sub_file_vtx2.write(t)


sub_file_vtx.close()
