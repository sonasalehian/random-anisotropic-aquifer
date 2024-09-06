import os
import sys

import numpy as np
from default_parameters import create_default_parameters
from model import solve

parameters = create_default_parameters()

random_type = "random_scaling_rotation"
n = int(sys.argv[1])

k = np.load("../tensor_model/output/ahct_{random_type}.npy")

# Adjust the tansor in parameters
parameters["k_x_aqfr"] = k[n, 0, 0]
parameters["k_xy_aqfr"] = k[n, 0, 1]
parameters["k_yx_aqfr"] = k[n, 1, 0]
parameters["k_y_aqfr"] = k[n, 1, 1]

parameters["output_dir"] = (
    f"{os.getenv('SCRATCH')}/stochastic_model/forward_propagation/output/{random_type}/run_{str(n).zfill(4)}/"
)

solve(parameters)
