import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib import colors
from rose_diagram import plot_rose_diagram
from default_parameters import create_default_parameters

parameters = create_default_parameters()

def plot_angle_distribution(random_angles):
    # Plot the histogram of the generated angles
    plt.figure(figsize=(8, 6))
    N, bins, patches = plt.hist(random_angles, bins=50)

    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    plt.hist(random_angles, bins=50, density=True, alpha=0.5)
    plt.title('Random Rotation Angles from von Mises Distribution')
    plt.xlabel('Angle (radian)')
    plt.ylabel('Frequency')
    plt.savefig("output/plots/rotation_angle_distribution.png")  # save as png

random_angles = np.load('./output/data/random_rotation_angle.npy')
num_samples = len(random_angles)
random_angles = random_angles - np.radians(110.0)
plot_angle_distribution(random_angles)

plot_rose_diagram(random_angles=random_angles)
plt.savefig("output/plots/rotation_angle_distribution_circular.png")  # save as png

k_initial = np.array([[parameters["k_x_aqfr"],0],[0,parameters["k_y_aqfr"]]])
R = [np.empty((2, 2)) for _ in range(num_samples)]
k = [np.empty((2, 2)) for _ in range(num_samples)]
for i, angle in enumerate(random_angles):
    W = np.array([[0, -angle], [angle, 0]])
    R = np.identity(2) + (np.sin(angle)/angle) * W + ((1- np.cos(angle))/angle**2) * (W @ W)
    k[i] = R @ (k_initial @ np.transpose(R))

# Save the random hydraulic conductivity tensor
file_name = 'output/data/ahct_random_rotation.npy'

if os.path.exists(file_name):
    os.remove(file_name)

# Save the list of arrays to a CSV file
np.save(file_name, np.array(k).reshape(num_samples, -1))

print(f"Filtered values have been saved to {file_name}")
