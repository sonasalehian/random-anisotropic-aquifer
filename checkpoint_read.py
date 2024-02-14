import basix
import basix.ufl
import dolfinx
from mpi4py import MPI

import adios4dolfinx


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

# dtype = get_dtype(mesh.geometry.x.dtype, complex)
filename = f"output/mesh_checkpoint.bp"
engine = "BP4"
comm = [MPI.COMM_SELF, MPI.COMM_WORLD]
MPI.COMM_WORLD.Barrier()
mesh = adios4dolfinx.read_mesh(
    MPI.COMM_WORLD, filename, engine, dolfinx.mesh.GhostMode.shared_facet
)
el = basix.ufl.element("Discontinuous Lagrange", "tetrahedron", 1)
V = dolfinx.fem.functionspace(mesh, el)
v = dolfinx.fem.Function(V)
v.name = "uh"
adios4dolfinx.read_function(v, filename, engine)
v_ex = dolfinx.fem.Function(V)


def f(x):
    return x[0]**2+x[1]**2

v_ex.interpolate(f)

# sub_file_vtx = dolfinx.io.VTXWriter(mesh.comm, "output/mesh_checkpoint2.bp", [v], engine="BP4")
# sub_file_vtx.write()
# sub_file_vtx.close()

# res = np.finfo(dtype).resolution
# assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)