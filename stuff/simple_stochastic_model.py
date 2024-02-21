import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm



mu, sigma = 0.0, 0.1 # mean and standard deviation
x = np.random.lognormal(mu, sigma, 5)

# x = lognorm.rvs(sigma, size=10000)

plt.figure(1)
count_x, bins_x, ignored_x = plt.hist(x, 100, density=True, align='mid', color='pink', alpha=0.6, label="x")

# xx = np.linspace(min(bins_x), max(bins_x), 10000)
# pdf_x = (np.exp(-(np.log(xx) - mu)**2 / (2 * sigma**2))
#        / (xx * sigma * np.sqrt(2 * np.pi)))

# plt.plot(xx, pdf_x, label="pdf of x", linewidth=2, color='r')

xx = np.linspace(lognorm.ppf(0.01, sigma), lognorm.ppf(0.99, sigma), 10000)
plt.plot(xx, lognorm.pdf(xx, sigma), 'r-', lw=2, alpha=1, label='lognorm pdf')

plt.axis('tight')
plt.legend(fontsize=10)
plt.savefig("output/histogram+pdf_x.png")  # save as png

def f(x):
    return x**2

y = f(x)

count_y, bins_y, ignored_y = plt.hist(y, 1000, density=True, align='mid', color='b', label="y")
plt.legend(fontsize=10)
plt.savefig("output/histogram_x,y.png")  # save as png