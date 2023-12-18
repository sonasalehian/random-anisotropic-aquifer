from scipy.stats import lognorm
import os
import numpy as np

def stretch_to_range(values, new_min, new_max):
    # Assuming values are between 0.7 and 1.4
    old_min, old_max = np.min(values), np.max(values)
    
    # Perform linear transformation to stretch the values to the new range
    stretched_values = (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    
    return stretched_values

# Generate random values with lognorm
sigma = 0.1 # standard deviation

x = lognorm.rvs(sigma, size=2)
lower_limit = 4.7E-13
upper_limit = 1.1E-11

hc = stretch_to_range(x, lower_limit, upper_limit)

# Save the random values
file_name = 'hpc_output/random_values.txt'

if os.path.exists(file_name):
  os.remove(file_name)
else:
  print("The file does not exist")

with open(file_name, 'w') as file:
    for number in hc:
        file.write(f"{number}\n")

print(f"Random values have been saved to {file_name}")
