from default_parameters import parameters
from model_load_mesh import solve
import sys

if __name__ == "__main__":
    hc = float(sys.argv[1])  # Read parameter from command line argument

    parameters["k_x_aqfr"] = hc
    parameters["k_y_aqfr"] = hc

    parameters["output_dir"] = './output/random_ihc'

    solve(parameters)
