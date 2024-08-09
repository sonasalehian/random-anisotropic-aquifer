import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import scienceplots

plt.style.use(['science'])

def extract_bar_parameters(random_angles, gradation=5.):
    angle = np.radians(gradation)
    patches = int(np.radians(360.)/angle)
    theta = np.arange(0,np.radians(360.),angle)
    count = [0]*patches
    for item in random_angles:
        temp = int((item - item%angle)/angle)
        count[temp] += 1
    return theta, count

def plot_rose_diagram(theta, count, gradation=5.0):
    # angle = np.radians(gradation)
    # patches = int(np.radians(360.)/angle)
    # width = angle * np.ones(patches)
    width = np.radians(gradation)  # Width of each bin in radians

    # force square figure and square axes looks better for polar, IMO
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    rmax = max(count) + 1
    ax.set_rlim(0,rmax)
    ax.set_theta_offset(np.pi/2)
    ax.set_thetamin(90)
    ax.set_thetamax(180)
    ax.set_thetagrids(np.arange(90,181,10))
    ax.set_yticklabels([])  # Hide r-axis labels
    ax.set_theta_direction(-1)

    # project strike distribution as histogram bars
    theta = theta + np.radians(gradation / 2)  # Centering the bars
    patches = ax.bar(theta, count, width=width)


# How to use the functions
# random_angles = np.load('./output/data/generated_data_from_rose_diagram.npy')
# # random_angles = np.radians(random_angles) # if the angles are in degree
# random_angles = np.load('./output/data/random_rotation_angle.npy')
# theta, count, width = extract_bar_parameters(random_angles=random_angles)
# plot_rose_diagram(theta, count, width)
# plt.savefig('./output/plots/rose_diagram_random_rotation_angle_5deg_quarter_noclr.pdf')