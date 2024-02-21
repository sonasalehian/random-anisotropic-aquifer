from default_parameters import parameters
from model_load_mesh_submesh_ktensor_checkpoint import solve
import sys
import numpy as np

# n = int(sys.argv[1])
n = 2
file_path = 'hpc_output/random_hc_fixed_scaling.csv'

# Load the list of arrays from the CSV file
k_flat = np.loadtxt(file_path, delimiter=',')

# Adjust the tansor in parameters
parameters["k_x_aqfr"] = k_flat[n, 0]
parameters["k_xy_aqfr"] = k_flat[n, 1]
parameters["k_yx_aqfr"] = k_flat[n, 2]
parameters["k_y_aqfr"] = k_flat[n, 3]

parameters["output_dir"] = f'./output/random_s_los/random_ahc_{n}'

solve(parameters)

