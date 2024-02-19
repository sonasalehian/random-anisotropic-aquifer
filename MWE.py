import basix
import basix.ufl
import dolfinx
import numpy as np
from mpi4py import MPI

import adios4dolfinx

from mpi4py import MPI
import numpy as np
import dolfinx


mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)


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

v_sub = dolfinx.fem.Function(V_sub)
adios4dolfinx.read_function(v_sub, filename, engine)

assert np.allclose(v_sub.x.array, u_sub.x.array)
