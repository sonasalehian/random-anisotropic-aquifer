from default_parameters import parameters
from model_load_mesh import solve

parameters["output_dir"] = './output/load_mesh'

solve(parameters)
