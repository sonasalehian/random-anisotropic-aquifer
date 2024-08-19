import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import sys
sys.path.insert(0, "../forward_propagation")
from default_parameters import create_default_parameters
import scienceplots

plt.style.use(['science'])

parameters = create_default_parameters()

def generating_random_scaling(mu_1, mu_2, std, num_samples):
    hydraulic_conductivity = np.zeros((num_samples, 2, 2))

    # Generate random values from a lognorm distribution
    lambda_1 = stats.lognorm(s=std, scale=mu_1)
    lambda_2 = stats.lognorm(s=std, scale=mu_2)

    hydraulic_conductivity[:, 0, 0] = lambda_1.rvs(size=num_samples)
    hydraulic_conductivity[:, 1, 1] = lambda_2.rvs(size=num_samples)

    return hydraulic_conductivity, lambda_1, lambda_2

def generating_random_rotation(random_angles, k_s):
    num_samples = len(random_angles)
    
    # Initialize the 3D array for the output
    k = np.zeros((num_samples, 2, 2))

    for i, angle in enumerate(random_angles):
        W = np.array([[0, -angle], [angle, 0]])
        # Calculate the rotation matrix R_r
        if angle != 0:
            R_r = np.identity(2) + (np.sin(angle)/angle) * W + ((1 - np.cos(angle))/angle**2) * (W @ W)
        else:
            R_r = np.identity(2)  # Avoid division by zero for angle == 0
        # Compute the rotated matrix
        k[i, :, :] = R_r @ (k_s[i, :, :] @ np.transpose(R_r))
    
    return k

def plot_pdfs(lambda_1, lambda_2, mu_1, mu_2, filename_1, filename_2):
    # Plot the PDF of lognormal distribution
    x_1 = np.linspace(0.6*mu_1, 1.4*mu_1, 1000)  
    x_2 = np.linspace(0.6*mu_2, 1.4*mu_2, 1000)  
    pdf_1 = lambda_1.pdf(x_1)
    pdf_2 = lambda_2.pdf(x_2)

    # Plot the histogram of random values
    fig = plt.figure(figsize=(5.5, 3))
    plt.figure(1)
    plt.plot(x_1, pdf_1, 'indianred', label=r'PDF $k_{xx}$')
    plt.plot([mu_1*0.79, mu_1*1.21], [0, 0], marker='o', color='darkred', markersize=8, label=r'$k_{xx}\pm 21\%$')
    plt.xlabel(r'Hydraulic conductivity ($m^3skg^{-1}$)')
    plt.ylabel('Probability density')
    plt.legend()
    plt.savefig(filename_1)  # save as png
    print(f"{filename_1} have been ploted.")

    fig = plt.figure(figsize=(5.5, 3))
    plt.figure(2)
    plt.plot(x_2, pdf_2, 'c-', label=r'PDF $k_{yy}$')
    plt.plot([mu_2*0.81, mu_2*1.19], [0, 0], marker='o', color='darkcyan', markersize=8, label=r'$k_{yy}\pm 19\%$')
    plt.xlabel('Hydraulic conductivity($m^3skg^{-1}$)')
    plt.ylabel('Probability density')
    plt.legend()
    plt.savefig(filename_2)  # save as png
    print(f"{filename_2} have been ploted.")

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

eigenvalues, lambda_1, lambda_2 = generating_random_scaling(mu_1, mu_2, std, num_samples)
k_s = eigenvalues

filename_1 = "output/Distribution_xx.pdf"
filename_2 = "output/Distribution_yy.pdf"
plot_pdfs(lambda_1, lambda_2, mu_1, mu_2, filename_1, filename_2)

# Save the random values
file_name = 'output/ahct_random_scaling.npy'
write_result(k_s, file_name)

# -----------------Step 2: random AHC tensor with random orientation -------------------
# Load generated random rotation angle
random_angles = np.load('../angle_model/output/random_rotation_angle.npy')
random_angles = random_angles - np.radians(110.0)

# Fixed scaling
k_initial = np.array([[parameters["k_x_aqfr"], 0], [0, parameters["k_y_aqfr"]]])
fixed_eigenvalues = np.zeros((num_samples, 2, 2))
fixed_eigenvalues[:] = k_initial


k_r = generating_random_rotation(random_angles, fixed_eigenvalues)

file_name = 'output/ahct_random_rotation.npy'
write_result(k_r, file_name)

# -----------------Step 3: random AHC tensor with random scaling and orientation -------------------
k_sr = generating_random_rotation(random_angles, k_s)

file_name = 'output/ahct_random_scaling_and_rotation.npy'
write_result(k_sr, file_name)
