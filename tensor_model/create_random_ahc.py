import os

import numpy as np

# Load generated random rotation angle
random_angles = np.load("../angle_model/output/random_rotation_angle.npy")
random_angles = random_angles - np.radians(110.0)
num_samples = len(random_angles)

eigenvalues = np.load("output/random_scaling.npy")

k = [np.zeros((2, 2)) for _ in range(num_samples)]
for i, angle in enumerate(random_angles):
    W = np.array([[0, -angle], [angle, 0]])
    R_r = np.identity(2) + (np.sin(angle) / angle) * W + ((1 - np.cos(angle)) / angle**2) * (W @ W)
    k_s = np.array([[eigenvalues[i, 0], 0], [0, eigenvalues[i, 3]]])
    k[i] = R_r @ (k_s @ np.transpose(R_r))

# Save the random hydraulic conductivity tensor
file_name = "output/data/random_scaling_and_rotation.npy"

if os.path.exists(file_name):
    os.remove(file_name)

# Save the list of arrays to a CSV file
np.save(file_name, np.array(k).reshape(num_samples, -1))

print(f"Random hydraulic conductivity tensors have been saved to {file_name}")
