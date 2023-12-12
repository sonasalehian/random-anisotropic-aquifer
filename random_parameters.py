from scipy.stats import lognorm
import os

# Generate random values with lognorm
sigma = 1.2 # standard deviation

x = lognorm.rvs(sigma, size=10000)

# Save the random values
file_name = 'hpc_output/random_values.txt'

if os.path.exists(file_name):
  os.remove(file_name)
else:
  print("The file does not exist")

with open(file_name, 'w') as file:
    for number in x:
        file.write(f"{number}\n")

print(f"Random values have been saved to {file_name}")
