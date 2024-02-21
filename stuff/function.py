import sys

def process_parameters(param1):
    # Replace this with your logic
    result = param1**2
    return result

if __name__ == "__main__":
    # # Check if the correct number of command line arguments is provided
    # if len(sys.argv) != 3:
    #     print("Usage: python process_parameters.py <param1> <param2>")
    #     sys.exit(1)  # Exit with an error code

    random_value = float(sys.argv[1])
    # output_directory = sys.argv[2]

    result = process_parameters(random_value)

    file_name = f'hpc_output/result.txt'

    with open(file_name, 'a') as file:
        file.write(f"{result}\n")
