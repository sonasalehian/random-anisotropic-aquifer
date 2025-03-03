from mpi4py import MPI

import adios4dolfinx
import numpy as np

import basix
import basix.ufl
import dolfinx

random_folder = "random_scaling_and_rotation_intermediate_ratio1-7/run_3609"
filename_output_ts = f"output/{random_folder}/output_ts.npy"
ts = np.load(filename_output_ts)

filename = f"output/{random_folder}/solution.bp"
engine = "BP4"
# MPI.COMM_WORLD.Barrier()
submesh = adios4dolfinx.read_mesh(
    filename, MPI.COMM_WORLD, engine, dolfinx.mesh.GhostMode.shared_facet
)
U_sub = dolfinx.fem.functionspace(submesh, basix.ufl.element("Lagrange", "tetrahedron", 1))
u_los = dolfinx.fem.Function(U_sub)
output = dolfinx.fem.Function(U_sub)

sub_file_vtx = dolfinx.io.VTXWriter(
    submesh.comm, f"output/{random_folder}/solution_readable.bp", [output], engine
)

for t in ts:
    # output.x.array[:] = 0
    # output.x.scatter_forward()
    u_los.name = "u_los_sub"
    adios4dolfinx.read_function(filename, u_los, engine, time=t)
    output.x.array[:] = u_los.x.array
    # output.x.scatter_forward()
    sub_file_vtx.write(t)

sub_file_vtx.close()
