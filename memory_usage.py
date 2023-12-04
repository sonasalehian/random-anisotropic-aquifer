import threading
import psutil
import time
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.stats import lognorm

def monitor_memory(memory_usage_data, stop_event):
    process = psutil.Process()

    while not stop_event.is_set():
        # Get memory usage in bytes
        memory_usage = process.memory_info().rss

        # Convert to megabytes for easier reading
        memory_mb = memory_usage / (1024 ** 2)

        # Record the current time and memory usage
        current_time = datetime.now()
        memory_usage_data.append((current_time, memory_mb))

        # Print memory usage
        print(f"Memory Usage: {memory_mb:.2f} MB")

        # Optional: Sleep for a short duration to control the frequency of updates
        time.sleep(0.1)

def plot_memory_usage(memory_usage_data):
    times, usages = zip(*memory_usage_data)

    plt.figure(1)
    plt.plot(times, usages)
    plt.title('Memory Usage Over Time')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage (MB)')
    plt.savefig("output/memory_usage.png")  # save as png

if __name__ == "__main__":
    # Start memory monitoring in a separate thread
    memory_data = []
    stop_event = threading.Event()
    memory_thread = threading.Thread(target=monitor_memory, args=(memory_data, stop_event))
    memory_thread.start()

    # Begining of the code
    mu, sigma = 0.0, 0.5 # mean and standard deviation

    x = lognorm.rvs(sigma, size=10000)

    plt.figure(2)
    count_x, bins_x, ignored_x = plt.hist(x, 100, density=True, align='mid', color='pink', alpha=0.6, label="x")

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
    
    # End of the code

    # Signal the memory monitoring thread to stop
    stop_event.set()

    # Optionally, wait for the memory monitoring thread to finish
    memory_thread.join()

    # Plot memory usage after the model execution
    plot_memory_usage(memory_data)
