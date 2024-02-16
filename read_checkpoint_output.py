import basix
import basix.ufl
import dolfinx
from mpi4py import MPI
import numpy as np

import adios4dolfinx

from default_parameters import parameters


def get_dtype(in_dtype: np.dtype, complex: bool):
    dtype: numpy.typing.DTypeLike
    if in_dtype == np.float32:
        if complex:
            dtype = np.complex64
        else:
            dtype = np.float32
    elif in_dtype == np.float64:
        if complex:
            dtype = np.complex128
        else:
            dtype = np.float64
    else:
        raise ValueError("Unsuported dtype")
    return dtype


t = parameters["t"]
T = parameters["T"]
num_steps = parameters["num_steps"]
dt = T / num_steps  # time step size
T2 = parameters["T2"]
num_steps2 = parameters["num_steps2"]
dt2 = (T2-T) / num_steps2


n = 0
filename = f'./output/random_s_los/random_ahc_{n}/submesh_checkpoint.bp'
engine = "BP4"
# comm = [MPI.COMM_SELF, MPI.COMM_WORLD]
MPI.COMM_WORLD.Barrier()
mesh = adios4dolfinx.read_mesh(
    MPI.COMM_WORLD, filename, engine, dolfinx.mesh.GhostMode.shared_facet
)
dtype = get_dtype(mesh.geometry.x.dtype, complex)
el = basix.ufl.element("Lagrange", "tetrahedron", 1)
V = dolfinx.fem.functionspace(mesh, el)
v = dolfinx.fem.Function(V, dtype=dtype)
v.name = "uh"

sub_file_vtx = dolfinx.io.VTXWriter(mesh.comm, f"./output/random_s_los/random_ahc_{n}/submesh_checkpoint2.bp", [v], engine="BP4")

for i in range(num_steps):
    t += dt
    if (i+1) % 20 == 0:
        adios4dolfinx.read_function(v, filename, engine, time=t)
        # sub_file_vtx.write(t)

# v_ex = dolfinx.fem.Function(V)

# def f(x):
#     return x[0]**2+x[1]**2

# v_ex.interpolate(f)

sub_file_vtx.close()

# res = np.finfo(dtype).resolution
# assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)