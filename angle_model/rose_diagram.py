import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import scienceplots

plt.style.use(['science'])

def extract_bar_parameters(random_angles, gradation=5.0):
    angle = np.radians(gradation)
    patches = int(np.radians(360.)/angle)
    theta = np.arange(0,np.radians(360.),angle)
    count = [0]*patches
    for item in random_angles:
        temp = int((item - item%angle)/angle)
        count[temp] += 1
    return theta, count

def plot_rose_diagram(theta, count, gradation=5.0):
    width = np.radians(gradation)  # Width of each bin in radians
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    # Normalize the count to use density instead of frequency
    total_count = sum(count)
    density = [c / total_count for c in count]  # Convert counts to densities

    rmax = max(density) + 0.05
    ax.set_rlim(0, rmax)
    ax.set_theta_offset(np.pi/2)
    ax.set_thetamin(90)
    ax.set_thetamax(180)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees([np.pi / 2, 3 * np.pi / 4, np.pi]), 
                      labels=[r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'], 
                      fontsize=14)
    
    # Set y-ticks and y-tick labels
    y_ticks = [0.1, 0.2, 0.3, 0.4]  # Specify your desired tick positions
    ax.set_yticks(y_ticks)  # Set the y-tick positions
    ax.set_yticklabels([str(tick) for tick in y_ticks])  # Set the y-tick labels

    # Set axis labels
    ax.set_xlabel(r'Angle ($\mathrm{rad}$)')
    ax.set_ylabel(r'Density ($\mathrm{rad^{-1}}$)')

    # project strike distribution as histogram bars
    theta = theta + np.radians(gradation / 2)  # Centering the bars
    patches = ax.bar(theta, density, width=width)


# How to use the functions
# random_angles = np.load('./output/data/generated_data_from_rose_diagram.npy')
# random_angles = np.radians(random_angles) # if the angles are in degree
# random_angles = np.load('./output/random_rotation_angle.npy')
# theta, count = extract_bar_parameters(random_angles=random_angles)
# plot_rose_diagram(theta, count)
# plt.savefig('./output/rose_diagram_random_rotation_angle.pdf')
