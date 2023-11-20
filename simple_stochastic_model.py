import numpy as np
import matplotlib.pyplot as plt


mu, sigma = 0.0, 0.5 # mean and standard deviation
x = np.random.lognormal(mu, sigma, 10000)

plt.figure(1)
count_x, bins_x, ignored_x = plt.hist(x, 100, density=True, align='mid', color='r', label="x")

plt.savefig("output/histogram_x.png")  # save as png

# print(x)

def f(x):
    return x**2

y = f(x)


count_y, bins_y, ignored_y = plt.hist(y, 100, density=True, align='mid', color='b', label="y")
plt.legend(fontsize=10)
plt.savefig("output/histogram_x,y.png")  # save as png

plt.figure(2)
xx = np.linspace(min(bins_x), max(bins_x), 10000)
pdf_x = (np.exp(-(np.log(xx) - mu)**2 / (2 * sigma**2))
       / (xx * sigma * np.sqrt(2 * np.pi)))

plt.plot(xx, pdf_x, label="pdf of x", linewidth=2, color='r')

yy = np.linspace(min(bins_y), max(bins_y), 10000)
mu_y = mu**(1/2)
sigma_y = sigma**(1/2)
pdf_y = (np.exp(-(np.log(yy) - mu_y)**2 / (2 * sigma_y**2))
        / (yy * sigma_y * np.sqrt(2 * np.pi)))

plt.plot(yy, pdf_y, label="pdf of y", linewidth=1, color='b')
plt.axis('tight')
plt.legend(fontsize=10)
plt.savefig("output/pdf_x,y.png")  # save as png
