# # Plot ellipses of random values with fixed orientation
# import matplotlib.pyplot as plt
# import numpy as np

# # Function to plot an ellipse
# def plot_ellipse(ax, center, radius_x, radius_y, angle, label):
#     theta = np.linspace(0, 2*np.pi, 100)
#     x = center[0] + radius_x * np.cos(theta) * np.cos(angle) - radius_y * np.sin(theta) * np.sin(angle)
#     y = center[1] + radius_x * np.cos(theta) * np.sin(angle) + radius_y * np.sin(theta) * np.cos(angle)
#     ax.plot(x, y, label=label)

# # Load hydraulic conductivity data from CSV
# elliptic_radius = np.loadtxt('hpc_output/random_hc_20filtered_fixed_orientation.csv', delimiter=',')


# # Randomly select 10 radius values
# selected_radius_indices = np.random.choice(len(elliptic_radius), size=10, replace=False)
# selected_radius_values = elliptic_radius[selected_radius_indices, :]

# # Set up the subplots
# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 7))

# # Plot 10 ellipses with randomly selected radii
# for i, (radius_minor, radius_major) in enumerate(selected_radius_values):
#     center = (0, 0)
#     angle = np.radians(54)  # Convert angle to radians

#     # Calculate the subplot indices
#     row = i // 5
#     col = i % 5

#     # Plot in the specified subplot
#     plot_ellipse(axes[row, col], center, radius_major, radius_minor, angle, label=f'Ellipse {i+1}')

#     # Set axis limits for better visualization
#     axes[row, col].set_xlim(-radius_major, radius_major)
#     axes[row, col].set_ylim(-radius_major, radius_major)

#     # Add legend
#     axes[row, col].legend()

#     # Set aspect ratio to 'equal' for a true circle
#     axes[row, col].set_aspect('equal', adjustable='box')
    
#     # Set title with rounded radius ratio
#     ratio = radius_major / radius_minor
#     axes[row, col].set_title(f'Radius Ratio: {ratio:.2f}')

# # Adjust layout for better spacing
# plt.tight_layout()

# # Adjust layout for better spacing
# plt.subplots_adjust(top=0.9)  # Increase the top parameter to adjust spacing

# # Display the plot
# plt.suptitle('Ellipses with 36-degree and Randomly Selected Radius', y=1.0)

# plt.savefig("hpc_output/elliptics.png")  # save as png


# Plot ellipses of random values with fixed scaling
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# Function to plot an ellipse
def plot_ellipse(ax, center, radius_x, radius_y, angle, label):
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius_x * np.cos(theta) * np.cos(angle) - radius_y * np.sin(theta) * np.sin(angle)
    y = center[1] + radius_x * np.cos(theta) * np.sin(angle) + radius_y * np.sin(theta) * np.cos(angle)
    ax.plot(x, y, label=label)


# Function to plot a line passing through (0, 0) with a specified angle
def plot_line(ax, angle, label):
    x = np.linspace(-2, 2, 100)
    y = np.tan(angle) * x
    ax.plot(x, y, label=label, linestyle='--', color='tab:gray')


# Load hydraulic conductivity data from CSV
elliptic_radius = np.loadtxt('output/data/random_hc_fixed_scaling.csv', delimiter=',')


# Randomly select 10 radius values
selected_radius_indices = np.random.choice(len(elliptic_radius), size=10, replace=False)
selected_tensors = elliptic_radius[selected_radius_indices, :]

# Set up the subplots
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 7))
k_x_origin = 1.1E-11
k_y_origin = 4.7E-13

# Plot 10 ellipses with randomly selected radii
for i, (k_x, k_xy, k_yx, k_y) in enumerate(selected_tensors):
    center = (0, 0)
    
    k = np.array([[k_x, k_xy],[k_yx, k_y]])
    u, R = la.eig(k)
    S = 0.5*(R-np.transpose(R))
    alpha = (0.5*np.trace(np.dot(S, np.transpose(S))))**(0.5)
    W = (np.arcsin(alpha)/alpha)*S
    phi = W[1, 0]
    # print(W)
    # print(phi)
    angle = np.radians(54) + phi # Convert angle to radians

    # Calculate the subplot indices
    row = i // 5
    col = i % 5

    # Plot in the specified subplot
    plot_ellipse(axes[row, col], center, k_x_origin, k_y_origin, angle, label=f'Ellipse {i+1}')

    # Plot the line passing through (0, 0) at the 54 deg angle
    plot_line(axes[row, col], np.radians(54), label=f'Line 36 deg')

    # Set axis limits for better visualization
    axes[row, col].set_xlim(-2*k_x_origin, 2*k_x_origin)
    axes[row, col].set_ylim(-2*k_x_origin, 2*k_x_origin)

    # Add legend
    axes[row, col].legend()

    # Set aspect ratio to 'equal' for a true circle
    axes[row, col].set_aspect('equal', adjustable='box')
    
    # Set title with rounded radius ratio
    phi_deg = np.rad2deg(phi)
    axes[row, col].set_title(f'Angle variation in degree: {phi_deg:.2f}')

# Adjust layout for better spacing
plt.tight_layout()

# Adjust layout for better spacing
plt.subplots_adjust(top=0.9)  # Increase the top parameter to adjust spacing

# Display the plot
plt.suptitle('Ellipses with fixed scaling and random orientation', y=1.0)

plt.savefig("output/plots/elliptics_fixed_scaling.png")  # save as png