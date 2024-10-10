import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scienceplots

plt.style.use(['science'])

# Disable LaTeX rendering to avoid missing font issues
plt.rcParams['text.usetex'] = False

k_x_origin = 1.1E-11
k_y_origin = 4.7E-13

#  Function to plot an ellipse
def plot_ellipse(center, radius_x, radius_y, angle):
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius_x * np.cos(theta) * np.cos(angle) - radius_y * np.sin(theta) * np.sin(angle)
    y = center[1] + radius_x * np.cos(theta) * np.sin(angle) + radius_y * np.sin(theta) * np.cos(angle)
    plt.plot(x, y, color='tab:purple', linewidth=0.5, alpha=0.5)
    # Set plt 
    plt.xlim(-1.5*k_x_origin, 1.5*k_x_origin)
    plt.ylim(-1.5*k_x_origin, 1.5*k_x_origin)
    # plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Increase the top parameter to adjust spacing
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('East')
    plt.ylabel('North')
    
# Function to plot a line passing through (0, 0) with a specified angle
def plot_line(angle, label, shift=0):
    x = np.linspace(-2, 2, 100)
    y = np.tan(angle) * x
    plt.plot(x, y-shift, label=label, linestyle='--', color='tab:gray')
    plt.legend()

# Load AHC tensor with random scaling
elliptic_radius = np.load('output/ahct_random_scaling.npy')

# Load AHC tensor with random rotation
elliptic_angle = np.load('output/ahct_random_rotation.npy')

# Randomly select 10 tensor
selected_indices = np.random.choice(len(elliptic_radius), size=10, replace=False)
print(selected_indices)
# selected_indices = [2219,  848, 4251, 3281,  696, 4595, 5457, 7109, 4812,  342]

# --------------------------------------------------random_scaling-----------------------------------------------------------

selected_radius_values = elliptic_radius[selected_indices, :, :]

plt.figure(figsize=(4, 3))

# Plot 10 ellipses with randomly selected tensor with random scaling
for i, k in enumerate(selected_radius_values):
    radius_major = k[0, 0]
    radius_minor = k[1, 1]
    center = (0, 0)
    angle = np.radians(340) # fixed angle
    ratio = radius_major / radius_minor
    plot_ellipse(center, radius_major, radius_minor, angle)

plot_line(np.radians(340), label=r'$x$ direction (shifted)', shift=5*k_y_origin)

plt.savefig("output/ellipses_random_scaling.pdf")  # save as png


#-----------------------------------------------------random_rotation-----------------------------------------------------

selected_tensors = elliptic_angle[selected_indices, :, :]

plt.figure(figsize=(4, 3))

# Plot 10 ellipses with randomly selected tensor with random rotation
for i, k in enumerate(selected_tensors):
    center = (0, 0)
    u, R = la.eig(k)
    S = 0.5*(R-np.transpose(R))
    alpha = (0.5*np.trace(np.dot(S, np.transpose(S))))**(0.5)
    W = (np.arcsin(alpha)/alpha)*S
    phi = W[1, 0]
    angle = np.radians(340) - phi # Convert angle to radians

    plot_ellipse(center, k_x_origin, k_y_origin, angle)

plot_line(np.radians(340), label=r'$x$ direction')

plt.savefig('output/ellipses_random_rotation.pdf')


#------------------------------------------------random_scaling_and_rotation--------------------------------------------------

plt.figure(figsize=(4, 3))

# Plot 10 ellipses with randomly selected tensor
for i, k in enumerate(selected_tensors):
    center = (0, 0)
    u, R = la.eig(k)
    S = 0.5*(R-np.transpose(R))
    alpha = (0.5*np.trace(np.dot(S, np.transpose(S))))**(0.5)
    W = (np.arcsin(alpha)/alpha)*S
    phi = W[1, 0]
    angle = np.radians(340) - phi

    radius_major = selected_radius_values[i, 0, 0]
    radius_minor = selected_radius_values[i, 1, 1]

    plot_ellipse(center, radius_major, radius_minor, angle)

plot_line(np.radians(340), label=r'$x$ direction')

plt.savefig('output/ellipses_random_scaling_and_rotation.pdf')
