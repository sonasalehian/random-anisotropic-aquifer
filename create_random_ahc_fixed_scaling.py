import numpy as np
import matplotlib.pyplot as plt
import os

# Set the parameters for the von Mises distribution
mean_angle = 0  # Mean angle in radians
kappa = 324  # Concentration parameter, controls the dispersion of the distribution

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
plt.savefig("hpc_output/Rotation_angle_distribution.png")  # save as png

k_initial = np.array([[1.1e-11,0],[0,4.7e-13]])
R = [np.empty((2, 2)) for _ in range(num_samples)]
k = [np.empty((2, 2)) for _ in range(num_samples)]
for i, angle in enumerate(random_angles):
    W = np.array([[0, -angle], [angle, 0]])
    R = np.identity(2) + (np.sin(angle)/angle) * W + ((1- np.cos(angle))/angle**2) * np.dot(W, W)
    k[i] = np.dot(R, np.dot(k_initial, np.transpose(R)))

# Save the random hydraulic conductivity tensor
file_name = 'hpc_output/random_hc_fixed_scaling.csv'

if os.path.exists(file_name):
    os.remove(file_name)
else:
    print("The random_hc_fixed_scaling file does not exist to remove")

# Save the list of arrays to a CSV file
np.savetxt(file_name, np.array(k).reshape(num_samples, -1), delimiter=',')

print(f"Filtered values have been saved to {file_name}")
