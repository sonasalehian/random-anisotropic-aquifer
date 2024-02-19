import basix
import basix.ufl
import dolfinx
import numpy as np
from mpi4py import MPI

import adios4dolfinx

from mpi4py import MPI
import numpy as np
import dolfinx

import dolfinx.io

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)


V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u = dolfinx.fem.Function(V)

u.interpolate(lambda x: np.sin(x[1]))
u.x.scatter_forward()

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
cells = dolfinx.mesh.compute_incident_entities(
    mesh.topology, boundary_facets, mesh.topology.dim - 1, mesh.topology.dim)


submesh, cell_map, _, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim, cells)

print(submesh.topology.dim)

V_sub = dolfinx.fem.functionspace(submesh, ("Lagrange", 1))
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

with dolfinx.io.XDMFFile(submesh.comm, "output/before_checkpoint.xdmf", "w") as f:
    f.write_mesh(submesh)
    f.write_function(u_sub)

filename = "output/checkpoint.bp"
adios4dolfinx.write_mesh(submesh, filename)
adios4dolfinx.write_function(u_sub, filename)

# Read the checkpoint file
engine = "BP4"
submesh = adios4dolfinx.read_mesh(
    MPI.COMM_WORLD, filename, engine, dolfinx.mesh.GhostMode.shared_facet
)
V_sub = dolfinx.fem.functionspace(submesh, ("Lagrange", 1))
v_sub = dolfinx.fem.Function(V_sub)
adios4dolfinx.read_function(v_sub, filename, engine)

assert np.allclose(np.linalg.norm(v_sub.x.array), np.linalg.norm(u_sub.x.array))

with dolfinx.io.XDMFFile(mesh.comm, "output/after_checkpoint.xdmf", "w") as f:
    f.write_mesh(submesh)
    f.write_function(v_sub)
