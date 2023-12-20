from default_parameters import parameters
from model_load_mesh import solve
import sys

def read_numbers_from_file(n):
    file_path = 'hpc_output/random_values.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()
        numbers = [float(line.strip()) for line in lines]
        number = numbers[n]
    return number

n = int(sys.argv[1])

random_value = read_numbers_from_file(n)

parameters["k_x_aqfr"] = random_value
parameters["k_y_aqfr"] = random_value

parameters["output_dir"] = f'./output/random_ihc_{n}'

solve(parameters)
