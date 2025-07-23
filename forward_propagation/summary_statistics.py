import os

from mpi4py import MPI

import adios4dolfinx
import numpy as np

import basix
import basix.ufl
import dolfinx

from utils import print_root

# Root folder with outputs
random_folder = "random_scaling_and_rotation"
folder_path = f"{os.getenv('SCRATCH')}/stochastic_model/forward_propagation/output/{random_folder}/"

# Number of outputs
num_outputs = len([f for f in os.listdir(folder_path) if f.startswith("run_")])

print_root(f"Found {num_outputs} samples in {folder_path}.")

# Load time steps
filename_output_ts = f"{folder_path}/run_{str(0).zfill(4)}/output_ts.npy"
ts = np.load(filename_output_ts)

# Read mesh
filename = f"{folder_path}/run_{str(0).zfill(4)}/solution.bp"
engine = "BP4"
submesh = adios4dolfinx.read_mesh(
    filename, MPI.COMM_WORLD, engine, dolfinx.mesh.GhostMode.shared_facet
)
U_sub = dolfinx.fem.functionspace(submesh, basix.ufl.element("Lagrange", "tetrahedron", 1))
u_los = dolfinx.fem.Function(U_sub)
u_los_mean = dolfinx.fem.Function(U_sub)
u_los_std = dolfinx.fem.Function(U_sub)

file_vtx_mean = dolfinx.io.VTXWriter(
    submesh.comm,
    f"{folder_path}/mean.bp",
    [u_los_mean],
    engine,
)

file_vtx_std = dolfinx.io.VTXWriter(
    submesh.comm,
    f"{folder_path}/std.bp",
    [u_los_std],
    engine,
)

for t in ts:
    print_root(f"Timestep: {t}")
    
    print_root(f"Computing mean...")
    u_los_mean.x.array[:] = 0.0
    
    for n in range(0, num_outputs):
        u_los.name = "u_los_sub"
        filename = f"{folder_path}/run_{str(n).zfill(4)}/solution.bp"
        adios4dolfinx.read_function(filename, u_los, engine, time=t)
        u_los_mean.x.array[:] += u_los.x.array
        u_los_mean.x.scatter_forward()

    u_los_mean.x.array[:] /= num_outputs
    file_vtx_mean.write(t)

    print_root(f"Computing standard deviation...")
    u_los_std.x.array[:] = 0.0
    
    for n in range(0, num_outputs):
        u_los.name = "u_los_sub"
        filename = f"{folder_path}/run_{str(n).zfill(4)}/solution.bp"
        adios4dolfinx.read_function(filename, u_los, engine, time=t)
        u_los_std.x.array[:] += (u_los.x.array - u_los_mean.x.array) ** 2
        u_los_std.x.scatter_forward()

    u_los_std.x.array[:] /= (num_outputs-1)
    u_los_std.x.array[:] = np.sqrt(u_los_std.x.array[:])
    file_vtx_std.write(t)

file_vtx_mean.close()
file_vtx_std.close()
print_root(f"mean and std calculated successfully.")
