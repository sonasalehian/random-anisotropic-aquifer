import matplotlib.pyplot as plt
import numpy as np

# Function to plot an ellipse
def plot_ellipse(ax, center, radius_x, radius_y, angle, label):
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius_x * np.cos(theta) * np.cos(angle) - radius_y * np.sin(theta) * np.sin(angle)
    y = center[1] + radius_x * np.cos(theta) * np.sin(angle) + radius_y * np.sin(theta) * np.cos(angle)
    ax.plot(x, y, label=label)

# Load hydraulic conductivity data from CSV
elliptic_radius = np.loadtxt('hpc_output/random_hc_fixed_orientation.csv', delimiter=',')


# Randomly select 10 radius values
selected_radius_indices = np.random.choice(len(elliptic_radius), size=10, replace=False)
selected_radius_values = elliptic_radius[selected_radius_indices, :]

# Set up the subplots
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 7))

# Plot 10 ellipses with randomly selected radii
for i, (radius_minor, radius_major) in enumerate(selected_radius_values):
    center = (0, 0)
    angle = np.radians(45)  # Convert angle to radians

    # Calculate the subplot indices
    row = i // 5
    col = i % 5

    # Plot in the specified subplot
    plot_ellipse(axes[row, col], center, radius_major, radius_minor, angle, label=f'Ellipse {i+1}')

    # Set axis limits for better visualization
    axes[row, col].set_xlim(-radius_major, radius_major)
    axes[row, col].set_ylim(-radius_major, radius_major)

    # Add legend
    axes[row, col].legend()

    # Set aspect ratio to 'equal' for a true circle
    axes[row, col].set_aspect('equal', adjustable='box')
    
    # Set title with rounded radius ratio
    ratio = radius_major / radius_minor
    axes[row, col].set_title(f'Radius Ratio: {ratio:.2f}')

# Adjust layout for better spacing
plt.tight_layout()

# Adjust layout for better spacing
plt.subplots_adjust(top=0.9)  # Increase the top parameter to adjust spacing

# Display the plot
plt.suptitle('Ellipses with 45-degree Tilt and Randomly Selected Radius', y=1.0)

plt.savefig("hpc_output/elliptics.png")  # save as png