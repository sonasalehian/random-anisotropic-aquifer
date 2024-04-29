import numpy as np
import matplotlib.pyplot as plt
import os
from create_random_ahc_fixed_orientation import generating_random_eigenvalues

# Load generated random rotation angle
random_angles = np.load('./output/data/random_rotation_angle.npy')
random_angles = random_angles - np.radians(110.0)
num_samples = len(random_angles)

eigenvalues = np.load('./output/data/ahct_random_scaling.npy')

k = [np.zeros((2, 2)) for _ in range(num_samples)]
for i, angle in enumerate(random_angles):
    W = np.array([[0, -angle], [angle, 0]])
    R_r = np.identity(2) + (np.sin(angle)/angle) * W + ((1- np.cos(angle))/angle**2) * np.dot(W, W)
    k_s = np.array([[eigenvalues[i, 0], 0], [0, eigenvalues[i, 3]]])
    k[i] = np.dot(R_r, np.dot(k_s, np.transpose(R_r)))

# Save the random hydraulic conductivity tensor
file_name = 'output/data/ahct_random_scaling_and_rotation.npy'

if os.path.exists(file_name):
    os.remove(file_name)
else:
    print(f"The {file_name} does not exist to remove")

# Save the list of arrays to a CSV file
np.save(file_name, np.array(k).reshape(num_samples, -1))

print(f"Random hydraulic conductivity tensors have been saved to {file_name}")

