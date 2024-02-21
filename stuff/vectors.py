import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Set seed for reproducibility
# np.random.seed(42)

N = 10
pd = lognorm(s=0.5, loc=0, scale=np.exp(-26))
r = pd.rvs(size=(N, 6))


# Rearrange components
r = np.column_stack((r[:, :3], r[:, 1], r[:, 3:5], r[:, [2, 4]], r[:, 5]))
print(r)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set 'AutoScale' to 'on' or a specific value to scale arrows
ax.quiver(0, 0, 0, r[:, 0], r[:, 1], r[:, 2], color='r', label='Vectors')

ax.set_xlim([0, max(r[:, 0])])
ax.set_ylim([0, max(r[:, 1])])
ax.set_zlim([0, max(r[:, 2])])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Save the plot
plt.savefig("output/vectors.png")  # save as png