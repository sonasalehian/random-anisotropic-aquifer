from default_parameters import parameters
from model_load_mesh import solve
import sys

if __name__ == "__main__":
    # Check if the correct number of command line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python process_parameters.py <param1> <param2>")
        sys.exit(1)  # Exit with an error code

    random_value = float(sys.argv[1])
    output_directory = sys.argv[2]

    parameters["k_x_aqfr"] = random_value
    parameters["k_y_aqfr"] = random_value

    parameters["output_dir"] = f'./output/{output_directory}'

    solve(parameters)
