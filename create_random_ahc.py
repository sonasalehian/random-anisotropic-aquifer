import numpy as np
import matplotlib.pyplot as plt
import os 

# Target mean for exponential distribution
target_mean = 1e-12

# Parameters for normal distribution
std_dev_normal = 1  # Standard deviation for normal distribution

# Calculate the mean and variance for the normal distribution
mean_normal_1 = np.log(target_mean) - 0.5 * (std_dev_normal**2)

# Generate random values from a normal distribution
num_samples = 1000
hydraulic_conductivity = np.zeros([num_samples, 2])
normal_values_1 = np.random.normal(loc=mean_normal_1, scale=std_dev_normal, size=num_samples)

# Calculate exponential values
print(np.mean(normal_values_1))
hydraulic_conductivity[:, 0] = np.exp(normal_values_1)

# Generate second random value to calculate second hydraulic conductivity
std_dev_normal_2 = 0.8
mean_normal_2 = np.log(1.5) - 0.5 * (std_dev_normal_2**2)
normal_values_2 = np.random.normal(loc=mean_normal_2, scale=std_dev_normal_2, size=num_samples)
print(np.mean(normal_values_2))
hydraulic_conductivity[:, 1] = np.exp(normal_values_1 + np.exp(normal_values_2))

# Save the random values
file_name = 'hpc_output/random_hc_fixed_orientation.csv'

if os.path.exists(file_name):
  os.remove(file_name)
else:
  print("The random_hc_fixed_orientation file does not exist to remove")

# Save the data to a CSV file
np.savetxt('hpc_output/random_hc_fixed_orientation.csv', hydraulic_conductivity, delimiter=',')

print(f"Random values have been saved to {file_name}")

# Calculate mean and standard deviation of the resulting values
mean_hc_1 = np.mean(hydraulic_conductivity[:, 0])
std_hc_1 = np.std(hydraulic_conductivity[:, 0])
mean_hc_2 = np.mean(hydraulic_conductivity[:, 1])
std_hc_2 = np.std(hydraulic_conductivity[:, 1])

# Plot the histogram of generated values
plt.hist(hydraulic_conductivity[:, 0], bins=30, density=True, alpha=0.5, color='b')
plt.title('Hydraulic Conductivity Distribution')
plt.xlabel('Hydraulic Conductivity')
plt.ylabel('Frequency')
plt.show()

# Print the results
print(f"Target Mean: {target_mean}")
print(f"Actual Mean Hydraulic Conductivity: {mean_hc_1}, {mean_hc_2}")
print(f"Standard Deviation: {std_hc_1}, {std_hc_2}")
