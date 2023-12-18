import numpy as np
from scipy import stats
import sys


def calculate_statistics(data):
    min = np.min(data)
    max = np.max(data)
    mean = np.mean(data)
    # mode = stats.mode(data, keepdims=True).mode[0]
    median = np.median(data)
    data_range = np.ptp(data)
    quartiles = np.percentile(data, [25, 50, 75])
    iqr = stats.iqr(data)
    variance = np.var(data)
    std_deviation = np.std(data)
    
    return min, max, mean, median, data_range, quartiles, iqr, variance, std_deviation


def read_numbers_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        numbers = [float(line.strip()) for line in lines]
    return numbers


def statistical_analysis(file_path):    
    results = read_numbers_from_file(file_path)
    
    min, max, mean, median, data_range, quartiles, iqr, variance, std_deviation = calculate_statistics(results)
    
    print(f"Min: {min}")
    print(f"Max: {max}")
    print(f"Mean: {mean}")
    # print(f"Mode: {mode}")
    print(f"Median: {median}")
    print(f"Range: {data_range}")
    print(f"Quartiles: Q1 = {quartiles[0]}, Q2 = {quartiles[1]}, Q3 = {quartiles[2]}")
    print(f"Interquartile Range: {iqr}")
    print(f"Variance: {variance}")
    print(f"Standard Deviation: {std_deviation}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 statistical_analysis.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    # file_path = float(sys.argv[1])  # Read parameter from command line argument

    statistical_analysis(file_path)

