import matplotlib.pyplot as plt
import numpy as np

# Original histogram data (bin edges and frequencies)
bin_edges_original = np.array([90, 100, 110, 120, 130, 140], dtype=np.float64)
frequencies_original = np.array([9, 14, 4, 22, 4], dtype=np.int64)


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
num_samples_generated = 100
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

# Plot original histogram
plt.figure(figsize=(10, 5))
lt.subplot(1, 2, 1)
plt.bar(
    bin_edges_original[:-1],
    frequencies_original,
    width=np.diff(bin_edges_original),
    align="edge",
    color="blue",
    alpha=0.7,
    label="Original histogram",
)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Fractures")
plt.legend()

# Plot histogram of generated data
plt.subplot(1, 2, 2)
plt.hist(generated_data, bins=30, color="orange", alpha=0.7, label="Generated Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Generated Data")
plt.legend()

plt.tight_layout()
plt.savefig("../output/histogram_regenerated_data.pdf")
