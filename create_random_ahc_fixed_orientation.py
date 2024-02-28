# JSH: You should be able to recover this case by having one generation
# function and setting the dispersion parameter low. 
import numpy as np
import matplotlib.pyplot as plt
import os

# Target mean for exponential distribution
target_mean_1 = 4.7e-13
target_mean_2 = 1.1e-11

# Parameters for normal distribution
std_dev_normal = 0.1  # Standard deviation for normal distribution

# Calculate the mean and variance for the normal distribution
mean_normal_1 = np.log(target_mean_1) - 0.5 * (std_dev_normal**2)
mean_normal_2 = np.log(target_mean_2) - 0.5 * (std_dev_normal**2)

# Initialize variables
num_samples = 1000
final_hydraulic_conductivity = np.zeros([num_samples, 2])
filtered_count = 0
max_attempts = 10000  # Maximum attempts to reach 1000 valid samples

while filtered_count < num_samples and max_attempts > 0:
    # Generate random values from a normal distribution
    hydraulic_conductivity = np.zeros([num_samples, 2])
    normal_values_1 = np.random.normal(loc=mean_normal_1, scale=std_dev_normal, size=num_samples)
    normal_values_2 = np.random.normal(loc=mean_normal_2, scale=std_dev_normal, size=num_samples)

    # Calculate exponential values
    hydraulic_conductivity[:, 0] = np.exp(normal_values_1)
    hydraulic_conductivity[:, 1] = np.exp(normal_values_2)

    # Apply condition to remove values differing more than 20% from target means
    condition_1 = np.abs(hydraulic_conductivity[:, 0] - target_mean_1) <= 0.2 * target_mean_1
    condition_2 = np.abs(hydraulic_conductivity[:, 1] - target_mean_2) <= 0.2 * target_mean_2
   # Add valid samples to final_hydraulic_conductivity
    valid_samples = hydraulic_conductivity[condition_1 & condition_2]
    count_to_fill = min(num_samples - filtered_count, len(valid_samples))
    final_hydraulic_conductivity[filtered_count:filtered_count + count_to_fill] = valid_samples[:count_to_fill]


    # Update count and attempts
    filtered_count += count_to_fill 
    max_attempts -= 1

# Save the random values after filtering
file_name_filtered = 'output/data/random_hc_20filtered_fixed_orientation.csv'

if os.path.exists(file_name_filtered):
    os.remove(file_name_filtered)
else:
    print("The random_hc_filtered file does not exist to remove")

# Save the filtered data to a CSV file
np.savetxt(file_name_filtered, final_hydraulic_conductivity, delimiter=',')

print(f"Filtered values have been saved to {file_name_filtered}")
print(max_attempts)

# Calculate mean and standard deviation of the resulting values
mean_hc_1 = np.mean(final_hydraulic_conductivity[:, 0])
std_hc_1 = np.std(final_hydraulic_conductivity[:, 0])
mean_hc_2 = np.mean(final_hydraulic_conductivity[:, 1])
std_hc_2 = np.std(final_hydraulic_conductivity[:, 1])

# Print the results
print(f"Target Mean: {target_mean_1}, {target_mean_2}")
print(f"Actual Mean Hydraulic Conductivity: {mean_hc_1}, {mean_hc_2}")
print(f"Standard Deviation: {std_hc_1}, {std_hc_2}")
print(final_hydraulic_conductivity.shape)

# Plot the histogram of filtered values
plt.figure(1)
plt.hist(final_hydraulic_conductivity[:, 0], bins=300, density=True, alpha=0.5, color='b', label='Distribution k_yy', range=(3e-13, 7e-13))
plt.title('Random Hydraulic Conductivity Distribution')
plt.xlabel('Hydraulic Conductivity')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("output/plots/Distribution_yy.png")  # save as png

plt.figure(2)
plt.hist(final_hydraulic_conductivity[:, 1], bins=100, density=True, alpha=0.5, color='r', label='Distribution k_xx', range=(8e-12, 1.4e-11))
plt.title('Random Hydraulic Conductivity Distribution')
plt.xlabel('Hydraulic Conductivity')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("output/plots/Distribution_xx.png")  # save as png
