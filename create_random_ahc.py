import numpy as np
import matplotlib.pyplot as plt
import os

# Set the parameters for the von Mises distribution
mean_angle = 0  # Mean angle in radians
kappa = 324  # Concentration parameter, controls the dispersion of the distribution ~10 deg

# Generate random samples from the von Mises distribution
num_samples = 1000
random_angles = np.random.vonmises(mean_angle, kappa, num_samples)

# Convert angles to degrees for visualization (optional)
random_angles_deg = np.degrees(random_angles)

# Plot the histogram of the generated angles
plt.hist(random_angles_deg, bins=30, density=True, alpha=0.5, color='b')
plt.title('Random Rotation Angles from von Mises Distribution')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.savefig("output/plots/Rotation_angle_distribution.png")  # save as png

k_initial = np.array([[1.1e-11,0],[0,4.7e-13]])


# Target mean for exponential distribution
target_mean_1 = k_initial[0, 0]
target_mean_2 = k_initial[1, 1]

# Parameters for normal distribution
std_dev_normal = 0.1  # Standard deviation for normal distribution

# Calculate the mean and variance for the normal distribution
mean_normal_1 = np.log(target_mean_1) - 0.5 * (std_dev_normal**2)
mean_normal_2 = np.log(target_mean_2) - 0.5 * (std_dev_normal**2)

# Initialize variables
num_samples = 1000
final_eigenvalues = np.zeros([num_samples, 2])
filtered_count = 0
max_attempts = 10000  # Maximum attempts to reach 1000 valid samples

while filtered_count < num_samples and max_attempts > 0:
    # Generate random values from a normal distribution
    eigenvalues = np.zeros([num_samples, 2])
    normal_values_1 = np.random.normal(loc=mean_normal_1, scale=std_dev_normal, size=num_samples)
    normal_values_2 = np.random.normal(loc=mean_normal_2, scale=std_dev_normal, size=num_samples)

    # Calculate exponential values
    eigenvalues[:, 0] = np.exp(normal_values_1)
    eigenvalues[:, 1] = np.exp(normal_values_2)

    # Apply condition to remove values differing more than 20% from target means
    condition_1 = np.abs(eigenvalues[:, 0] - target_mean_1) <= 0.2 * target_mean_1
    condition_2 = np.abs(eigenvalues[:, 1] - target_mean_2) <= 0.2 * target_mean_2
   # Add valid samples to final_eigenvalues
    valid_samples = eigenvalues[condition_1 & condition_2]
    count_to_fill = min(num_samples - filtered_count, len(valid_samples))
    final_eigenvalues[filtered_count:filtered_count + count_to_fill] = valid_samples[:count_to_fill]

    # Update count and attempts
    filtered_count += count_to_fill 
    max_attempts -= 1

print(final_eigenvalues)
# k_s = np.array([[final_eigenvalues[:, 0], 0], [0, final_eigenvalues[:, 1]]])
k = [np.empty((2, 2)) for _ in range(num_samples)]
for i, angle in enumerate(random_angles):
    W = np.array([[0, -angle], [angle, 0]])
    R_r = np.identity(2) + (np.sin(angle)/angle) * W + ((1- np.cos(angle))/angle**2) * np.dot(W, W)
    k_s = np.array([[final_eigenvalues[i, 0], 0], [0, final_eigenvalues[i, 1]]])
    k[i] = np.dot(R_r, np.dot(k_s, np.transpose(R_r)))

# Save the random hydraulic conductivity tensor
file_name = 'output/data/random_hc.csv'

if os.path.exists(file_name):
    os.remove(file_name)
else:
    print(f"The {file_name} does not exist to remove")

# Save the list of arrays to a CSV file
np.savetxt(file_name, np.array(k).reshape(num_samples, -1), delimiter=',')

print(f"Random hydraulic conductivity tensors have been saved to {file_name}")

