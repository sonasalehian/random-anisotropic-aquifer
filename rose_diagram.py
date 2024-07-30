import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import scienceplots

plt.style.use(['science'])

def plot_rose_diagram(random_angles):
    gradation = 5.
    angle = np.radians(gradation)
    patches = int(np.radians(360.)/angle)
    theta = np.arange(0,np.radians(360.),angle)
    count = [0]*patches
    for i, item in enumerate(random_angles):
        temp = int((item - item%angle)/angle)
        count[temp] += 1
    width = angle * np.ones(patches)

    # force square figure and square axes looks better for polar, IMO
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    rmax = max(count) + 1

    ax.set_rlim(0,rmax)
    ax.set_theta_offset(np.pi/2)
    ax.set_thetamin(90)
    ax.set_thetamax(180)
    ax.set_thetagrids(np.arange(90,181,10))
    ax.set_rgrids([1000,2000,3000], labels=[])
    # ax.set_rgrids([10,20,30,40], labels=[])
    ax.set_theta_direction(-1)

    # project strike distribution as histogram bars
    theta = theta + np.radians(5)
    print(count)
    cs = [c / (max(count)/1.6) for c in count]
    # colors = plt.cm.viridis(cs)
    patches = ax.bar(theta, count, width=width)
    
    # # Add colorbar
    # cmap = plt.cm.viridis
    # norm = mpl.colors.Normalize(vmin=0, vmax=max(count))
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.1)


# random_angles = np.load('./output/data/generated_data_from_rose_diagram.npy')
# random_angles = np.radians(random_angles)
random_angles = np.load('./output/data/random_rotation_angle.npy')
plot_rose_diagram(random_angles=random_angles)
plt.savefig('./output/plots/rose_diagram_random_rotation_angle_5deg_quarter_noclr.pdf')