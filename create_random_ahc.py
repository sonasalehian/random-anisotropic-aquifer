# JSH: There is a lot going on in this script. Could the core routine be
# converted into a reusable function(s)? For example, a function like the
# existing random generators which take the distribution parameters and
# num_samples.
import numpy as np
import matplotlib.pyplot as plt
import os

# Set the parameters for the von Mises distribution
mean_angle = 0  # Mean angle in radians
# JSH: Why not parametrise with variance and then convert to dispersion? not so
# easy to work with dispersion. 
# JSH: Where is 324 (or its sigma^2) taken from?
kappa = 324  # Concentration parameter, controls the dispersion of the distribution ~10 deg

# Generate random samples from the von Mises distribution
num_samples = 1000
random_angles = np.random.vonmises(mean_angle, kappa, num_samples)

# JSH: Plotting should be done in a separate script.
# JSH: From radians to degrees
# Convert angles to degrees for visualization (optional)
random_angles_deg = np.degrees(random_angles)

# Plot the histogram of the generated angles
plt.hist(random_angles_deg, bins=30, density=True, alpha=0.5, color='b')
plt.title('Random Rotation Angles from von Mises Distribution')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
# JSH: Comment is unnecessary, obvious by reading code.
# JSH: Don't use caps in filenames. Rotation -> rotation.
plt.savefig("output/plots/Rotation_angle_distribution.png")  # save as png

# JSH: Where were these values taken from?
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
# JSH: Repeated variable.
num_samples = 1000
# JSH: Remove all of this filtering stuff.
final_eigenvalues = np.zeros([num_samples, 2])
filtered_count = 0
max_attempts = 10000  # Maximum attempts to reach 1000 valid samples

# JSH: You cannot do this filtering, you are not left with a well-defined set
# of samples from a pdf. We need to look at appropriate parameters above.
# Take a look at e.g. truncated normal to understand why:
# https://en.wikipedia.org/wiki/Truncated_normal_distribution
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
    # JSH: logical and in Python is simply `and`. & does something different (bitwise and).
    valid_samples = eigenvalues[condition_1 & condition_2]
    count_to_fill = min(num_samples - filtered_count, len(valid_samples))
    final_eigenvalues[filtered_count:filtered_count + count_to_fill] = valid_samples[:count_to_fill]

    # Update count and attempts
    filtered_count += count_to_fill 
    max_attempts -= 1

# JSH: Remove, potentially huge output to log file.
print(final_eigenvalues)
# k_s = np.array([[final_eigenvalues[:, 0], 0], [0, final_eigenvalues[:, 1]]])

# JSH: numpy supports three dimensional arrays.
# JSH: don't use np.empty - anything you don't set specifically will be random!
k = [np.empty((2, 2)) for _ in range(num_samples)]
# JSH: Needs some further explanation.
for i, angle in enumerate(random_angles):
    W = np.array([[0, -angle], [angle, 0]])
    R_r = np.identity(2) + (np.sin(angle)/angle) * W + ((1 - np.cos(angle))/angle**2) * np.dot(W, W)
    k_s = np.array([[final_eigenvalues[i, 0], 0], [0, final_eigenvalues[i, 1]]])
    k[i] = np.dot(R_r, np.dot(k_s, np.transpose(R_r)))

# Save the random hydraulic conductivity tensor
# JSH: This should go near the top of the file so it can be changed easily?
file_name = 'output/data/random_hc.csv'

if os.path.exists(file_name):
    # JSH: Don't delete files silently - dangerous, check if it exists at start
    # and raise an error e.g. RuntimeError.
    os.remove(file_name)
else:
    # JSH: Remove.
    print(f"The {file_name} does not exist to remove")

# Save the list of arrays to a CSV file
# JSH: If you are reading and writing into numpy, just use np.save or np.savez
# Then you do not need to reshape, as CSV only supports 1D or 2D arrays.
# Also when you read it back in, it will be in its 'natural' shape.
np.savetxt(file_name, np.array(k).reshape(num_samples, -1), delimiter=',')

print(f"Random hydraulic conductivity tensors have been saved to {file_name}")
