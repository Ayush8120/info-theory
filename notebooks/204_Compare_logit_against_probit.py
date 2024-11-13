"""
Visualizes probit and logit PDF and CDF with identical standard deviation 1
First version: 10/29/2024
This version: 10/29/2024
https://northeastern-datalab.github.io/cs7840/fa24/calendar.html
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, logistic


# Generate a range of x values
x = np.linspace(-4, 4, 1000)

# Standard normal distribution (mean=0, std=1)
normal_pdf = norm.pdf(x, loc=0, scale=1)
normal_cdf = norm.cdf(x, loc=0, scale=1)

# Logit distribution (mean=0, scale to match std deviation of standard normal)
# Find scale parameter s such that std (var) is 1.
# Formula: var = s^2 pi^2 / 3
logit_scale = 1 * np.sqrt(3) / np.pi  # To match standard deviation of normal (std=1)
logit_pdf = logistic.pdf(x, loc=0, scale=logit_scale)
logit_cdf = logistic.cdf(x, loc=0, scale=logit_scale)

# Set font size and line width globally
plt.rcParams.update({'font.size': 16, 'lines.linewidth': 3})



# Plot 1: Probability Density Function (PDF)
plt.figure(figsize=(8, 6))
plt.plot(x, normal_pdf, label='Probit', color='blue')
plt.plot(x, logit_pdf, label='Logit (same $\sigma$)', color='red', linestyle='--')
plt.title('Probit vs. Logit (PDF, same standard deviation)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/Fig_204_pdf_comparison_probit_logit.png')
plt.show()



# Plot 2: Cumulative Distribution Function (CDF)
plt.figure(figsize=(8, 6))
plt.plot(x, normal_cdf, label='Probit', color='blue')
plt.plot(x, logit_cdf, label='Logit (same $\sigma$)', color='red', linestyle='--')
plt.title('Probit vs. Logit (CDF, same standard deviation)')
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/Fig_204_cdf_comparison_probit_logit.png')
plt.show()
