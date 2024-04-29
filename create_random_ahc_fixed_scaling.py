import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib import colors

def plot_angle_distribution(random_angles):
    # Convert angles to degrees for visualization (optional)
    # random_angles_deg = np.degrees(random_angles)

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
    plt.savefig("output/plots/Rotation_angle_distribution.png")  # save as png

def circular_hist(ax, x, bins=30, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    colors = plt.cm.viridis(n / 1000.)
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor=colors, fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches, colors

# # Set the parameters for the von Mises distribution
# mean_angle = 0  # Mean angle in radians
# sigma = np.radians(10.0)
# kappa = 1 / (sigma**2)  # Concentration parameter, controls the dispersion of the distribution

# # Generate random samples from the von Mises distribution
# num_samples = 80
# random_angles = np.random.vonmises(mean_angle, kappa, num_samples)

random_angles = np.load('./output/data/random_rotation_angle.npy')
num_samples = len(random_angles)
random_angles = random_angles - np.radians(110.0)
plot_angle_distribution(random_angles)

# Circular distribution plot
plt.figure(figsize=(8, 6))
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))

# Visualise by area of bins
circular_hist(ax, random_angles)
plt.savefig("output/plots/Rotation_angle_distribution_circular.png")  # save as png

k_initial = np.array([[1.1e-11,0],[0,4.7e-13]])
R = [np.empty((2, 2)) for _ in range(num_samples)]
k = [np.empty((2, 2)) for _ in range(num_samples)]
for i, angle in enumerate(random_angles):
    W = np.array([[0, -angle], [angle, 0]])
    R = np.identity(2) + (np.sin(angle)/angle) * W + ((1- np.cos(angle))/angle**2) * np.dot(W, W)
    k[i] = np.dot(R, np.dot(k_initial, np.transpose(R)))

# Save the random hydraulic conductivity tensor
file_name = 'output/data/ahct_random_rotation.npy'

if os.path.exists(file_name):
    os.remove(file_name)
else:
    print(f"The {file_name} does not exist to remove")

# Save the list of arrays to a CSV file
np.save(file_name, np.array(k).reshape(num_samples, -1))

print(f"Filtered values have been saved to {file_name}")
