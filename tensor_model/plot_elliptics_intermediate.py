import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scienceplots

plt.style.use(['science'])

# Disable LaTeX rendering to avoid missing font issues
# plt.rcParams['text.usetex'] = False

k_y_origin = 4.7E-13
k_x_origin = 3*k_y_origin

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
def plot_line(angle, label, linestyle='--', shift=0):
    x = np.linspace(-2, 2, 100)
    y = np.tan(angle) * x
    plt.plot(x, y-shift, label=label, linestyle=linestyle, color='tab:gray')
    plt.legend()

# Load AHC tensor with random scaling
elliptic_radius = np.load('output/intermediate/ahct_random_scaling.npy')

# Load AHC tensor with random rotation
elliptic_angle = np.load('output/intermediate/ahct_random_rotation.npy')

# Randomly select 10 tensor
# selected_indices = np.random.choice(len(elliptic_radius), size=20, replace=False)
selected_indices = [3609, 1339, 3917,  258, 2353, 5509, 5948, 1214, 4343, 
5116, 4410, 1079,  256, 1144, 4225,  152, 1399,  830, 6142, 1972]
print(selected_indices)

# # --------------------------------------------------random_scaling-----------------------------------------------------------

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

plot_line(np.radians(340), label=r'$x$ direction (shifted)', shift=2*k_y_origin)

plt.savefig("output/intermediate/ellipses_random_scaling.pdf")  # save as png
plt.close()


# #-----------------------------------------------------random_rotation-----------------------------------------------------

selected_tensors = elliptic_angle[selected_indices, :, :]
random_angles = np.load('../angle_model/output/intermediate/random_rotation_angle.npy')
selected_angles = random_angles[selected_indices]

plt.figure(figsize=(4, 3))

# Plot 10 ellipses with randomly selected tensor with random rotation
for i, k in enumerate(selected_tensors):
    center = (0, 0)
    # u, R = la.eig(k)
    # print(k)
    # print(u)
    # print(R)
    # S = 0.5*(R-np.transpose(R))
    # print(S)
    # alpha = (0.5*np.trace(np.dot(S, np.transpose(S))))**(0.5)
    # print(alpha)
    # W = (np.arcsin(alpha)/alpha)*S
    # print(W)
    # phi = W[1, 0]
    # if u == [4.70e-13+0.j, 1.41e-12+0.j]:
    #     phi
    phi = selected_angles[i]
    angle = np.radians(90) - phi # Convert angle to radians

    plot_ellipse(center, k_x_origin, k_y_origin, angle)

plot_line(np.radians(340), label=r'$x$ direction')

plt.savefig('output/intermediate/ellipses_random_rotation.pdf')
plt.close()

#------------------------------------------------random_scaling_and_rotation--------------------------------------------------

plt.figure(figsize=(4, 4))

# Plot 10 ellipses with randomly selected tensor
for i, k in enumerate(selected_tensors):
    center = (0, 0)
    # u, R = la.eig(k)
    # S = 0.5*(R-np.transpose(R))
    # alpha = (0.5*np.trace(np.dot(S, np.transpose(S))))**(0.5)
    # W = (np.arcsin(alpha)/alpha)*S
    # phi = W[1, 0]
    phi = selected_angles[i]
    angle = np.radians(90) - phi

    radius_major = selected_radius_values[i, 0, 0]
    radius_minor = selected_radius_values[i, 1, 1]

    plot_ellipse(center, radius_major, radius_minor, angle)

plot_line(np.radians(340), label=r'$x$ direction')
plot_line(np.radians(70), label=r'$y$ direction', linestyle='dotted')

# Get current axis and set equal aspect
ax = plt.gca()  # Get current axis
ax.set_aspect('equal', 'box')

plt.savefig('output/intermediate/ellipses_random_scaling_and_rotation.pdf')
