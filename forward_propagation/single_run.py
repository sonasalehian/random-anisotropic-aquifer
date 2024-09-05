import os
import sys

import numpy as np
from default_parameters import create_default_parameters
from model import solve

parameters = create_default_parameters()

random_folder = "random_s"
n = int(sys.argv[1])

# Load the list of arrays
if random_folder == "random_s":
    k = np.load("../tensor_model/output/ahct_random_scaling.npy")
elif random_folder == "random_r":
    k = np.load("../tensor_model/output/ahct_random_rotation.npy")
elif random_folder == "random_sr":
    k = np.load("../tensor_model/output/ahct_random_scaling_and_rotation.npy")

# Adjust the tansor in parameters
parameters["k_x_aqfr"] = k[n, 0, 0]
parameters["k_xy_aqfr"] = k[n, 0, 1]
parameters["k_yx_aqfr"] = k[n, 1, 0]
parameters["k_y_aqfr"] = k[n, 1, 1]

parameters["output_dir"] = (
    f"{os.getenv('SCRATCH')}/stochastic_model/forward_propagation/output/{random_folder}/run_{str(n).zfill(4)}"
)

solve(parameters)
