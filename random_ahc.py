from default_parameters import parameters
from model_load_mesh import solve
import sys
import numpy as np

def read_numbers_from_file(n, file_path):
    elliptic_radius = np.loadtxt(file_path, delimiter=',')
    
    # Assuming the columns are numeric; adjust indexing if needed
    k_x_aqfr = elliptic_radius[n, 1]
    k_y_aqfr = elliptic_radius[n, 0]

    return k_x_aqfr, k_y_aqfr

n = int(sys.argv[1])
file_path = 'hpc_output/random_hc_fixed_orientation.csv'

random_value_x, random_value_y = read_numbers_from_file(n, file_path)

# Format the values to print in scientific notation with two decimal places
formatted_random_value_x = "{:.1e}".format(random_value_x)
formatted_random_value_y = "{:.1e}".format(random_value_y)

parameters["k_x_aqfr"] = random_value_x
parameters["k_y_aqfr"] = random_value_y

parameters["output_dir"] = f'./output/random_ahc_{formatted_random_value_x}_{formatted_random_value_y}'

solve(parameters)
