import matplotlib.pyplot as plt
import numpy as np
from rose_diagram import plot_rose_diagram, extract_bar_parameters
import scienceplots

plt.style.use(['science'])

# Disable LaTeX rendering to avoid missing font issues
plt.rcParams['text.usetex'] = False

# Original histogram data (bin edges and frequencies)
bin_edges_original = np.array([90, 100, 110, 120, 130, 140])
frequencies_original = np.array([9, 14, 4, 22, 4])


# Generate random data based on the original histogram
def generate_data_from_histogram(bin_edges, frequencies, num_samples):
    samples = np.zeros(num_samples)
    for i in range(num_samples):
        # Choose a random bin based on the histogram frequencies
        bin_index = np.random.choice(len(bin_edges) - 1, p=frequencies / np.sum(frequencies))
        # Generate a random value within the chosen bin
        lower_bound, upper_bound = bin_edges[bin_index], bin_edges[bin_index + 1]
        samples[i] = np.random.uniform(lower_bound, upper_bound)
    return samples


# Generate data
num_samples_generated = 1000
generated_data = generate_data_from_histogram(
    bin_edges_original, frequencies_original, num_samples_generated
)

# Save generated data
np.save("output/rose_diagram.npy", generated_data)

# Validate generated data
# Compare summary statistics (mean, standard deviation)
mean_generated = np.mean(generated_data)
std_generated = np.std(generated_data)

print(
    f"Summary statistics for generated data: Mean={mean_generated:.2f}, Standard Deviation={std_generated:.2f}"
)

# Plot original rose diagram
theta = np.radians(bin_edges_original[0:-1])
plot_rose_diagram(theta, frequencies_original, gradation=10.)
plt.savefig('output/rose_diagram_Heilweil.pdf')

# Plot rose diagram of generated data
generated_data = np.load('output/rose_diagram.npy')
random_angles = np.radians(generated_data)
theta, count = extract_bar_parameters(random_angles=random_angles)
plot_rose_diagram(theta, count)
plt.savefig('output/rose_diagram_generated_data_from_rose_diagram.pdf')
