#!/usr/bin/python3
import re
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

data = []
with open("throughput.log") as fh:
    for line in fh:
        # [INTERNODE] 16 -> 8 time = 0.018645 sec (858.133609 MB/sec)
        value = re.search(r".*\((?P<VALUE>.*)\sMB\/sec\).*", line)
        if value :
            data.append(float(value.group(1)))

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=30, density=True, alpha=0.6)

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.xlabel('Throughput, MB/sec')

plt.savefig('throughput.png')
