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

def generating_random_eigenvalues(mu_1, mu_2, std, num_samples):
    hydraulic_conductivity = np.zeros([num_samples, 4])

    # Generate random values from a lognorm distribution
    lambda_1 = stats.lognorm(s=std, scale=mu_1)
    lambda_2 = stats.lognorm(s=std, scale=mu_2)

    hydraulic_conductivity[:, 0] = lambda_1.rvs(size=num_samples)
    hydraulic_conductivity[:, 3] = lambda_2.rvs(size=num_samples)

    return hydraulic_conductivity, lambda_1, lambda_2

def plot_result(lambda_1, lambda_2, filename_1, filename_2):
    # Plot the PDF of lognormal distribution
    x_1 = np.linspace(0.6e-11, 1.6e-11, 1000)  
    x_2 = np.linspace(2.7e-13, 6.7e-13, 1000)  
    pdf_1 = lambda_1.pdf(x_1)
    pdf_2 = lambda_2.pdf(x_2)

    # Plot the histogram of random values
    fig = plt.figure(figsize=(5.5, 3))
    plt.figure(1)
    # plt.hist(hydraulic_conductivity[:, 0], bins=100, density=True, alpha=0.5, color='b', label=r'Distribution $k_{xx}$', range=(8e-12, 1.4e-11))
    plt.plot(x_1, pdf_1, 'indianred', label=r'PDF $k_{xx}$')
    plt.plot([mu_1*0.79, mu_1*1.21], [0, 0], marker='o', color='darkred', markersize=8, label=r'$k_{xx}\pm 21\%$')
    # plt.title('Random Hydraulic Conductivity Distribution')
    plt.xlabel(r'Hydraulic conductivity ($m^3skg^{-1}$)')
    plt.ylabel('Probability density')
    plt.legend()
    plt.savefig(filename_1)  # save as png

    fig = plt.figure(figsize=(5.5, 3))
    plt.figure(2)
    # plt.hist(hydraulic_conductivity[:, 3], bins=100, density=True, alpha=0.5, color='r', label=r'Distribution $k_{yy}$', range=(3e-13, 7e-13))
    plt.plot(x_2, pdf_2, 'c-', label=r'PDF $k_{yy}$')
    plt.plot([mu_2*0.81, mu_2*1.19], [0, 0], marker='o', color='darkcyan', markersize=8, label=r'$k_{yy}\pm 19\%$')
    # plt.title('Random Hydraulic Conductivity Distribution')
    plt.xlabel('Hydraulic conductivity($m^3skg^{-1}$)')
    plt.ylabel('Probability density')
    plt.legend()
    plt.savefig(filename_2)  # save as png

def write_result(hydraulic_conductivity, file_name):
    if os.path.exists(file_name):
        os.remove(file_name)

    np.save(file_name, hydraulic_conductivity)
    print(f"Random values have been saved to {file_name}")

# Initialize variables
mu_1 = parameters["k_x_aqfr"] # Mean for first eigenvalue
mu_2 = parameters["k_y_aqfr"] # Mean for second eigenvalue
std = 0.08  # Standard deviation for distribution
num_samples = 8000

hydraulic_conductivity, lambda_1, lambda_2 = generating_random_eigenvalues(mu_1, mu_2, std, num_samples)

filename_1 = "output/Distribution_xx.pdf"
filename_2 = "output/Distribution_yy.pdf"

plot_result(lambda_1, lambda_2, filename_1, filename_2)

# Save the random values to a CSV file
file_name = 'output/ahct_random_scaling.npy'

write_result(hydraulic_conductivity, file_name)

# Calculate mean and standard deviation of the resulting values
mean_hc_1 = np.mean(hydraulic_conductivity[:, 0])
std_hc_1 = np.std(hydraulic_conductivity[:, 0])
mean_hc_2 = np.mean(hydraulic_conductivity[:, 3])
std_hc_2 = np.std(hydraulic_conductivity[:, 3])

# Print the results
print(f"Target Mean: {mu_1}, {mu_2}")
print(f"Actual Mean Hydraulic Conductivity: {mean_hc_1}, {mean_hc_2}")
print(f"Standard Deviation: {std_hc_1}, {std_hc_2}")