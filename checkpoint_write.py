import itertools
import pathlib

import basix
import basix.ufl
import dolfinx
import numpy as np
import pytest
from mpi4py import MPI
import numpy.typing

import adios4dolfinx

# from .test_utils import read_function, write_function, get_dtype, read_function_time_dep, write_function_time_dep


# def write_function(mesh, el, f, dtype, name="uh", append: bool = False) -> str:

mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([100, 150, 40])],
                        [20, 6, 6], cell_type=dolfinx.mesh.CellType.tetrahedron)  
el = basix.ufl.element("Discontinuous Lagrange", "tetrahedron", 0)
V = dolfinx.fem.functionspace(mesh, el)
uh = dolfinx.fem.Function(V)


def f(x):
    return x[0]**2+x[1]**2

uh.interpolate(f)
uh.name = "uh"
# el_hash = (
#     V.element.signature()
#     .replace(" ", "")
#     .replace(",", "")
#     .replace("(", "")
#     .replace(")", "")
#     .replace("[", "")
#     .replace("]", "")
# )

# file_hash = f"{el_hash}_{np.dtype(dtype).name}"
# filename = pathlib.Path(f"output/mesh_{file_hash}.bp")
filename = pathlib.Path(f"output/mesh_checkpoint.bp")
# if mesh.comm.size != 1:
#     if not append:
adios4dolfinx.write_mesh(mesh, filename)
adios4dolfinx.write_function(uh, filename, time=0.0)
# else:
#     if MPI.COMM_WORLD.rank == 0:
#         if not append:
#             adios4dolfinx.write_mesh(mesh, filename)
#         adios4dolfinx.write_function(uh, filename, time=0.0)

# return file_hash


# def read_function(comm, el, f, hash, dtype, name="uh"):
#     filename = f"output/mesh_{hash}.bp"
#     engine = "BP4"
#     mesh = adios4dolfinx.read_mesh(
#         comm, filename, engine, dolfinx.mesh.GhostMode.shared_facet
#     )
#     V = dolfinx.fem.functionspace(mesh, el)
#     v = dolfinx.fem.Function(V, dtype=dtype)
#     v.name = name
#     adios4dolfinx.read_function(v, filename, engine)
#     v_ex = dolfinx.fem.Function(V, dtype=dtype)
#     v_ex.interpolate(f)

#     res = np.finfo(dtype).resolution
#     assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)


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




# # ----------------------------------------------------------------------------------


# dtypes = [np.float64, np.float32]  # Mesh geometry dtypes
# write_comm = [MPI.COMM_SELF, MPI.COMM_WORLD]  # Communicators for creating mesh

# two_dimensional_cell_types = [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
# three_dimensional_cell_types = [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]

# two_dim_combinations = itertools.product(dtypes, two_dimensional_cell_types, write_comm)
# three_dim_combinations = itertools.product(dtypes, three_dimensional_cell_types, write_comm)


# @pytest.fixture(params=two_dim_combinations, scope="module")
# def mesh_2D(request):
#     dtype, cell_type, write_comm = request.param
#     mesh = dolfinx.mesh.create_unit_square(write_comm, 10, 10, cell_type=cell_type, dtype=dtype)
#     return mesh


# @pytest.fixture(params=three_dim_combinations, scope="module")
# def mesh_3D(request):
#     dtype, cell_type, write_comm = request.param
#     M = 5
#     mesh = dolfinx.mesh.create_unit_cube(write_comm, M, M, M, cell_type=cell_type, dtype=dtype)
#     return mesh


# @pytest.mark.parametrize("complex", [True, False])
# @pytest.mark.parametrize("family", ["Lagrange", "DG"])
# @pytest.mark.parametrize("degree", [1, 4])
# @pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
# def test_read_write_P_2D(read_comm, family, degree, complex, mesh_2D):
#     mesh = mesh_2D
#     f_dtype = get_dtype(mesh.geometry.x.dtype, complex)

#     el = basix.ufl.element(family,
#                            mesh.ufl_cell().cellname(),
#                            degree,
#                            basix.LagrangeVariant.gll_warped,
#                            shape=(mesh.geometry.dim, ),
#                            dtype=mesh.geometry.x.dtype)

#     def f(x):
#         values = np.empty((2, x.shape[1]), dtype=f_dtype)
#         values[0] = np.full(x.shape[1], np.pi) + x[0] + x[1] * 1j
#         values[1] = x[0] + 3j * x[1]
#         return values

#     hash = write_function(mesh, el, f, f_dtype)
#     MPI.COMM_WORLD.Barrier()
#     read_function(read_comm, el, f, hash, f_dtype)


# @pytest.mark.parametrize("complex", [True, False])
# @pytest.mark.parametrize("family", ["Lagrange", "DG"])
# @pytest.mark.parametrize("degree", [1, 4])
# @pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
# def test_read_write_P_3D(read_comm, family, degree, complex, mesh_3D):
#     mesh = mesh_3D
#     f_dtype = get_dtype(mesh.geometry.x.dtype, complex)
#     el = basix.ufl.element(family,
#                            mesh.ufl_cell().cellname(),
#                            degree,
#                            basix.LagrangeVariant.gll_warped,
#                            shape=(mesh.geometry.dim, ))

#     def f(x):
#         values = np.empty((3, x.shape[1]), dtype=f_dtype)
#         values[0] = np.pi + x[0] + 2j*x[2]
#         values[1] = x[1] + 2 * x[0]
#         values[2] = 1j*x[1] + np.cos(x[2])
#         return values

#     hash = write_function(mesh, el, f, f_dtype)

#     MPI.COMM_WORLD.Barrier()
#     read_function(read_comm, el, f, hash, f_dtype)


# @pytest.mark.parametrize("complex", [True, False])
# @pytest.mark.parametrize("family", ["Lagrange", "DG"])
# @pytest.mark.parametrize("degree", [1, 4])
# @pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
# def test_read_write_P_2D_time(read_comm, family, degree, complex, mesh_2D):
#     mesh = mesh_2D
#     f_dtype = get_dtype(mesh.geometry.x.dtype, complex)

#     el = basix.ufl.element(family,
#                            mesh.ufl_cell().cellname(),
#                            degree,
#                            basix.LagrangeVariant.gll_warped,
#                            shape=(mesh.geometry.dim, ),
#                            dtype=mesh.geometry.x.dtype)

#     def f0(x):
#         values = np.empty((2, x.shape[1]), dtype=f_dtype)
#         values[0] = np.full(x.shape[1], np.pi) + x[0] + x[1] * 1j
#         values[1] = x[0] + 3j * x[1]
#         return values

#     def f1(x):
#         values = np.empty((2, x.shape[1]), dtype=f_dtype)
#         values[0] = 2*np.full(x.shape[1], np.pi) + x[0] + x[1] * 1j
#         values[1] = -x[0] + 3j * x[1] + 2*x[1]
#         return values

#     t0 = 0.8
#     t1 = 0.6
#     hash = write_function_time_dep(mesh, el, f0, f1, t0, t1, f_dtype)
#     MPI.COMM_WORLD.Barrier()
#     read_function_time_dep(read_comm, el, f0, f1, t0, t1, hash, f_dtype)


# @pytest.mark.parametrize("complex", [True, False])
# @pytest.mark.parametrize("family", ["Lagrange", "DG"])
# @pytest.mark.parametrize("degree", [1, 4])
# @pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
# def test_read_write_P_3D_time(read_comm, family, degree, complex, mesh_3D):
#     mesh = mesh_3D
#     f_dtype = get_dtype(mesh.geometry.x.dtype, complex)
#     el = basix.ufl.element(family,
#                            mesh.ufl_cell().cellname(),
#                            degree,
#                            basix.LagrangeVariant.gll_warped,
#                            shape=(mesh.geometry.dim, ))

#     def f(x):
#         values = np.empty((3, x.shape[1]), dtype=f_dtype)
#         values[0] = np.pi + x[0] + 2j*x[2]
#         values[1] = x[1] + 2 * x[0]
#         values[2] = 1j*x[1] + np.cos(x[2])
#         return values

#     def g(x):
#         values = np.empty((3, x.shape[1]), dtype=f_dtype)
#         values[0] = x[0] + np.pi * 2j*x[2]
#         values[1] = 1j*x[2] + 2 * x[0]
#         values[2] = x[0] + 1j*np.cos(x[1])
#         return values

#     t0 = 0.1
#     t1 = 1.3
#     hash = write_function_time_dep(mesh, el, g, f, t0, t1, f_dtype)
#     MPI.COMM_WORLD.Barrier()
#     read_function_time_dep(read_comm, el, g, f, t0, t1, hash, f_dtype)