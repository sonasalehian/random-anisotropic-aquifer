import basix
import basix.ufl
import dolfinx
import numpy as np
from mpi4py import MPI

import adios4dolfinx

from mpi4py import MPI
import numpy as np
import dolfinx

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




mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
dtype = get_dtype(mesh.geometry.x.dtype, complex)


V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2, )))
u = dolfinx.fem.Function(V)

u.interpolate(lambda x: (x[0], np.sin(x[1])))
u.x.scatter_forward()

mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
cells = dolfinx.mesh.compute_incident_entities(
    mesh.topology, boundary_facets, mesh.topology.dim-1, mesh.topology.dim)


submesh, cell_map, _, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim, cells)

V_sub = dolfinx.fem.functionspace(submesh, V.ufl_element())
u_sub = dolfinx.fem.Function(V_sub)

num_sub_cells = submesh.topology.index_map(submesh.topology.dim).size_local
for cell in range(num_sub_cells):
    sub_dofs = V_sub.dofmap.cell_dofs(cell)
    parent_dofs = V.dofmap.cell_dofs(cell_map[cell])
    assert V_sub.dofmap.bs == V.dofmap.bs
    for parent, child in zip(parent_dofs, sub_dofs):
        for b in range(V_sub.dofmap.bs):
            u_sub.x.array[child*V_sub.dofmap.bs +
                          b] = u.x.array[parent*V.dofmap.bs+b]

u_sub.x.scatter_forward()
u_sub.name = "u_sub"


filename = 'output/submesh_checkpoint_MWE.bp'

adios4dolfinx.write_mesh(submesh, filename)
adios4dolfinx.write_function(u_sub, filename, time=0.0)


# Read the checkpoint file
filename = "output/submesh_checkpoint_MWE.bp"
engine = "BP4"
MPI.COMM_WORLD.Barrier()
submesh = adios4dolfinx.read_mesh(
    MPI.COMM_WORLD, filename, engine, dolfinx.mesh.GhostMode.shared_facet
)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2, )))
V_sub = dolfinx.fem.functionspace(submesh, V.ufl_element())

# el = basix.ufl.element("Lagrange", 1, (2, ))
# V_sub = dolfinx.fem.functionspace(submesh, el)
v_sub = dolfinx.fem.Function(V_sub)
v_sub.name = "u_sub"
adios4dolfinx.read_function(v_sub, filename, engine)
# v_ex = dolfinx.fem.Function(V)


# def f(x):
#     return x[0]**2+x[1]**2

# v_ex.interpolate(f)
t = 0
sub_file_vtx = dolfinx.io.VTXWriter(submesh.comm, "output/submesh_checkpoint_MWE2.bp", [v_sub], engine="BP4")
sub_file_vtx.write(t)
sub_file_vtx.close()

res = np.finfo(dtype).resolution
assert np.allclose(v_sub.x.array, u_sub.x.array, atol=10 * res, rtol=10 * res)