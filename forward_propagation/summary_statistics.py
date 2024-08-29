import adios4dolfinx
import basix
import basix.ufl
import dolfinx
import os
import numpy as np
from mpi4py import MPI

def count_files_starting_with(folder_path, prefix):
    return len([f for f in os.listdir(folder_path) if f.startswith(prefix)])

random_folder = "random_sr"

# Number of outputs
n_0 = 0
folder_path = f"{os.getenv('SCRATCH')}/stochastic_model/forward_propagation/output/{random_folder}"

prefix = 'run_'
n_outputs = count_files_starting_with(folder_path, prefix)

# Load time steps
filename_output_ts = f"{os.getenv('SCRATCH')}/stochastic_model/forward_propagation/output/{random_folder}/run_{str(n_0).zfill(4)}/output_ts.npy"
ts = np.load(filename_output_ts)

# Read mesh
filename = f"{os.getenv('SCRATCH')}/stochastic_model/forward_propagation/output/{random_folder}/run_{str(n_0).zfill(4)}/solution.bp"
engine = "BP4"
MPI.COMM_WORLD.Barrier()
submesh = adios4dolfinx.read_mesh(
    MPI.COMM_WORLD, filename, engine, dolfinx.mesh.GhostMode.shared_facet
)
U_sub = dolfinx.fem.functionspace(submesh, basix.ufl.element("Lagrange", "tetrahedron", 1))
u_los = dolfinx.fem.Function(U_sub)
u_los_mean = dolfinx.fem.Function(U_sub)
u_los_std = dolfinx.fem.Function(U_sub)

file_vtx_mean = dolfinx.io.VTXWriter(
    submesh.comm, f"{os.getenv('SCRATCH')}/stochastic_model/forward_propagation/output/{random_folder}/final_mean{n_0}-{n_outputs-1}.bp", [u_los_mean], engine
)

file_vtx_std = dolfinx.io.VTXWriter(
    submesh.comm, f"{os.getenv('SCRATCH')}/stochastic_model/forward_propagation/output/{random_folder}/final_std{n_0}-{n_outputs-1}.bp", [u_los_std], engine
)

for t in ts:
    # Mean calculation
    u_los_mean.x.array[:] = 0
    for n in range(n_0, n_outputs):
        u_los.name = "u_n_sub"
        filename = f"{os.getenv('SCRATCH')}/stochastic_model/forward_propagation/output/{random_folder}/run_{str(n).zfill(4)}/solution.bp"
        adios4dolfinx.read_function(u_los, filename, engine, time=t)
        u_los_mean.x.array[:] += u_los.x.array
        u_los_mean.x.scatter_forward()

    u_los_mean.x.array[:] /= n_outputs - n_0
    file_vtx_mean.write(t)
    
    # Standard deviation calculation
    u_los_std.x.array[:] = 0
    u_los_std.x.scatter_forward()
    for n in range(n_0, n_outputs):
        u_los.name = "u_n_sub"
        filename = f"{os.getenv('SCRATCH')}/stochastic_model/forward_propagation/output/{random_folder}/run_{str(n).zfill(4)}/solution.bp"
        adios4dolfinx.read_function(u_los, filename, engine, time=t)
        u_los_std.x.array[:] += (u_los.x.array - u_los_mean.x.array) ** 2
        u_los_std.x.scatter_forward()

    u_los_std.x.array[:] /= n_outputs - n_0
    u_los_std.x.array[:] = np.sqrt(u_los_std.x.array[:])
    file_vtx_std.write(t)

file_vtx_mean.close()
file_vtx_std.close()
