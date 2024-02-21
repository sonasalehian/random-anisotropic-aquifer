import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



usage = pd.read_csv("output/memory_usage_kutta.csv")
mu = np.zeros((usage.shape))
data = pd.DataFrame(usage, columns=['Time (s)','Memory Usage (bytes)'])
mu[:, :] = data
print(max(mu[:,1]))


# Plot pressure x direction
fig = plt.figure(figsize=(8, 6))

plt.plot(mu[:, 0]/60, mu[:, 1]/(10**9), label="memory usage", color='lightseagreen', linestyle='solid')

plt.xlabel("time (min)", fontsize=10)
plt.ylabel("memory usage (gigabytes)", fontsize=10)
plt.legend(fontsize=10)

plt.savefig("output/graph_memory_usage_kutta.png")  # save as png