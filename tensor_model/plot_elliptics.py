import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scienceplots

plt.style.use(['science'])

#  Function to plot an ellipse
def plot_ellipse(center, radius_x, radius_y, angle, label):
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius_x * np.cos(theta) * np.cos(angle) - radius_y * np.sin(theta) * np.sin(angle)
    y = center[1] + radius_x * np.cos(theta) * np.sin(angle) + radius_y * np.sin(theta) * np.cos(angle)
    # ax.plot(x, y, label=label, linewidth=0.5)
    plt.plot(x, y, color='tab:purple', linewidth=0.5, alpha=0.5)

# Function to plot a line passing through (0, 0) with a specified angle
def plot_line(angle, label):
    x = np.linspace(-2, 2, 100)
    y = np.tan(angle) * x
    plt.plot(x, y, label=label, linestyle='--', color='tab:gray')

# Load hydraulic conductivity data
elliptic_radius = np.load('output/data/ahct_random_scaling.npy')

# Load hydraulic conductivity data 
elliptic_angle = np.load('output/data/ahct_random_rotation.npy')

# Randomly select 10 radius values
selected_indices = np.random.choice(len(elliptic_radius), size=10, replace=False)
print(selected_indices)
# selected_indices = [5501, 6012, 1547, 7138, 3152, 5157, 7596, 1490, 4820, 2888]

k_x_origin = 1.1E-11
k_y_origin = 4.7E-13


# --------------------------------------------------random_scaling-----------------------------------------------------------

selected_radius_values = elliptic_radius[selected_indices[::2], :]

# Set up the subplots
plt.figure(figsize=(4, 3))

# Plot 10 ellipses with randomly selected radii
for i, (radius_major, _, _, radius_minor) in enumerate(selected_radius_values):
    center = (0, 0)
    angle = np.radians(340)  # Convert angle to radians
    ratio = radius_major / radius_minor
    plot_ellipse(center, radius_major, radius_minor, angle, label=f'Radius Ratio: {ratio:.2f}')

# Set axis limits for better visualization
plt.xlim(-1.5*k_x_origin, 1.5*k_x_origin)
plt.ylim(-1.5*k_x_origin, 1.5*k_x_origin)

# Adjust layout for better spacing
plt.tight_layout()

# Adjust layout for better spacing
plt.subplots_adjust(top=0.9)  # Increase the top parameter to adjust spacing

plt.xticks([])
plt.yticks([])

plt.xlabel('East')
plt.ylabel('North')

plt.savefig("./output/plots/ellipses_random_scaling.pdf")  # save as png


#-----------------------------------------------------random_rotation-----------------------------------------------------

selected_tensors = elliptic_angle[selected_indices, :]

# Set up the subplots
plt.figure(figsize=(4, 3))

# Plot 10 ellipses with randomly selected radii
for i, (k_x, k_xy, k_yx, k_y) in enumerate(selected_tensors):
    center = (0, 0)
    
    k = np.array([[k_x, k_xy],[k_yx, k_y]])
    u, R = la.eig(k)
    S = 0.5*(R-np.transpose(R))
    alpha = (0.5*np.trace(np.dot(S, np.transpose(S))))**(0.5)
    W = (np.arcsin(alpha)/alpha)*S
    phi = W[1, 0]
    angle = np.radians(340) - phi # Convert angle to radians

    plot_ellipse(center, k_x_origin, k_y_origin, angle, label=f'Ellipse {i+1}')

plot_line(np.radians(340), label='Line 110 degrees')

# Set axis limits for better visualization
plt.xlim(-1.5*k_x_origin, 1.5*k_x_origin)
plt.ylim(-1.5*k_x_origin, 1.5*k_x_origin)

# Add legend
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()

# Adjust layout for better spacing
plt.subplots_adjust(top=0.9)  # Increase the top parameter to adjust spacing

plt.xticks([])
plt.yticks([])

plt.xlabel('East')
plt.ylabel('North')

plt.savefig('./output/plots/ellipses_random_rotation.pdf')


#------------------------------------------------random_scaling_and_rotation--------------------------------------------------

selected_tensors = elliptic_angle[selected_indices, :]
selected_radius_values = elliptic_radius[selected_indices, :]

# Set up the subplots
plt.figure(figsize=(4, 3))

# Plot 10 ellipses with randomly selected indices
for i, (k_x, k_xy, k_yx, k_y) in enumerate(selected_tensors):
    center = (0, 0)
    
    k = np.array([[k_x, k_xy],[k_yx, k_y]])
    u, R = la.eig(k)
    S = 0.5*(R-np.transpose(R))
    alpha = (0.5*np.trace(np.dot(S, np.transpose(S))))**(0.5)
    W = (np.arcsin(alpha)/alpha)*S
    phi = W[1, 0]
    angle = np.radians(340) - phi # Convert angle to radians

    (radius_major, _, _, radius_minor) = selected_radius_values[i,:]

    plot_ellipse(center, radius_major, radius_minor, angle, label=f'Ellipse {i+1}')

plot_line(np.radians(340), label='Line 110 degrees')

# Set axis limits for better visualization
plt.xlim(-1.5*k_x_origin, 1.5*k_x_origin)
plt.ylim(-1.5*k_x_origin, 1.5*k_x_origin)

# Add legend
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()

# Adjust layout for better spacing
plt.subplots_adjust(top=0.9)  # Increase the top parameter to adjust spacing

plt.xticks([])
plt.yticks([])

plt.xlabel('East')
plt.ylabel('North')

plt.savefig('./output/plots/ellipses_random_scaling_and_rotation.pdf')