import numpy as np
import os
from scipy import stats
from default_parameters import create_default_parameters

parameters = create_default_parameters()

def generating_random_scaling(mu_1, mu_2, std, num_samples):
    hydraulic_conductivity = np.zeros([num_samples, 4])

    # Generate random values from a lognorm distribution
    lambda_1 = stats.lognorm(s=std, scale=mu_1)
    lambda_2 = stats.lognorm(s=std, scale=mu_2)

    hydraulic_conductivity[:, 0] = lambda_1.rvs(size=num_samples)
    hydraulic_conductivity[:, 3] = lambda_2.rvs(size=num_samples)

    return hydraulic_conductivity

def generating_random_rotation(random_angles, eigenvalues):
    num_samples = len(random_angles)
    k = [np.zeros((2, 2)) for _ in range(num_samples)]
    for i, angle in enumerate(random_angles):
        W = np.array([[0, -angle], [angle, 0]])
        R_r = np.identity(2) + (np.sin(angle)/angle) * W + ((1- np.cos(angle))/angle**2) * (W @ W)
        k_s = np.array([[eigenvalues[i, 0], 0], [0, eigenvalues[i, 3]]])
        k[i] = R_r @ (k_s @ np.transpose(R_r))
    return k

def write_result(hydraulic_conductivity, file_name):
    if os.path.exists(file_name):
        os.remove(file_name)

    np.save(file_name, hydraulic_conductivity)
    print(f"Random values have been saved to {file_name}")

# -----------------Step 1: random AHC tensor with random scaling -------------------
# Initialize variables
mu_1 = parameters["k_x_aqfr"] # Mean for first eigenvalue
mu_2 = parameters["k_y_aqfr"] # Mean for second eigenvalue
std = 0.08  # Standard deviation for distribution
num_samples = 8000

eigenvalues = generating_random_scaling(mu_1, mu_2, std, num_samples)
k_s = eigenvalues

# Save the random values
file_name = 'output/data/ahct_random_scaling.npy'
write_result(k_s, file_name)

# -----------------Step 2: random AHC tensor with random orientation -------------------
# Load generated random rotation angle
random_angles = np.load('./output/data/random_rotation_angle.npy')
random_angles = random_angles - np.radians(110.0)

# Fixed scaling
k_initial = np.array([parameters["k_x_aqfr"], 0, 0, parameters["k_y_aqfr"]])
fixed_eigenvalues = np.zeros([num_samples, 4])
fixed_eigenvalues = np.tile(k_initial, (num_samples, 1))

k_r = generating_random_rotation(random_angles, fixed_eigenvalues)

file_name = 'output/data/ahct_random_rotation.npy'
write_result(np.array(k_r).reshape(num_samples, -1), file_name)

# -----------------Step 3: random AHC tensor with random scaling and orientation -------------------
k_sr = generating_random_rotation(random_angles, eigenvalues)

file_name = 'output/data/ahct_random_scaling_and_rotation.npy'
write_result(np.array(k_sr).reshape(num_samples, -1), file_name)