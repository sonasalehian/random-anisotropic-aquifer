import sys

import numpy as np
from default_parameters import create_default_parameters
from model import solve

parameters = create_default_parameters()

n = int(sys.argv[1])

# Load the list of arrays
#k_flat = np.load("output/data/ahct_random_scaling_and_rotation.npy")

# Adjust the tansor in parameters
#parameters["k_x_aqfr"] = k_flat[n, 0]
#parameters["k_xy_aqfr"] = k_flat[n, 1]
#parameters["k_yx_aqfr"] = k_flat[n, 2]
#parameters["k_y_aqfr"] = k_flat[n, 3]

parameters["output_dir"] = f"output/run_{str(n).zfill(4)}"

solve(parameters)
