import sys

def model_computation(parameter):
    # Replace this with your actual model computation
    return parameter**2

if __name__ == "__main__":
    parameter = float(sys.argv[1])  # Read parameter from command line argument

    result = model_computation(parameter)

# print(f"Model computation for parameter {parameter}: {result}")

file_name = 'hpc_output/result.txt'

with open(file_name, 'a') as file:
    file.write(f"{result}\n")
