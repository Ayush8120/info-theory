"""
Plot function for Sterling approximation for lg(n!). Recall lg(x) = log2(x) = ln(x) / ln(2)
First version: 11/9/2024
This version: 11/9/2024
https://northeastern-datalab.github.io/cs7840/fa24/calendar.html
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln


n_values = np.arange(2, 10000)

# Calculate exact binary logarithm of n! using gammaln function for high precision
# gamma(n+1) = n!, gammaln(n+1) = ln(n!)
log2_factorial_exact = gammaln(n_values + 1) / np.log(2)

# 2. Calculate Stirling's approximation for binary logarithm of n!
log2_factorial_stirling = n_values * np.log2(n_values) - n_values * np.log2(np.e)

# 3. Calculate a looser Stirling approximation
log2_factorial_stirling_loose = n_values * np.log2(n_values)


# Plot all the total values
plt.figure(figsize=(6, 6))
plt.loglog(n_values, log2_factorial_stirling_loose,
           label="Stirling's Approx. (loose)", linewidth=4, color='orange', linestyle='-')
plt.loglog(n_values, log2_factorial_exact,
           label='$\mathrm{lg}(n!)$', linewidth=4, color='blue')
plt.loglog(n_values, log2_factorial_stirling,
           label="Stirling's Approx.", linewidth=4, color='green', linestyle='--')

plt.xlabel("$n$", fontsize=24)
plt.ylabel("$\mathrm{lg}(n!)$", fontsize=24)
plt.legend(fontsize=14, facecolor='white', framealpha=1)
plt.grid(True, which="both", linestyle='-', color='lightgray', linewidth=0.5)

plt.xticks([10**i for i in range(0, 6)], [f"$10^{i}$" for i in range(0, 6)], fontsize=18)
plt.yticks([10**i for i in range(0, 8)], [f"$10^{i}$" for i in range(0, 8)], fontsize=18)
plt.xlim((1,1e4))
plt.ylim((1,2e5))

plt.tight_layout()
plt.savefig('figures/Fig_224_Sterling_approximation_1.png')
plt.show()


# Plot everything as relative values
plt.figure(figsize=(6, 6))
plt.loglog(n_values, log2_factorial_stirling_loose/log2_factorial_exact,
           label="Stirling (loose) / $\mathrm{lg}(n!)$", linewidth=4, color='orange')
plt.loglog(n_values, log2_factorial_stirling/log2_factorial_exact,
           label="Stirling / $\mathrm{lg}(n!)$", linewidth=4, color='green', linestyle='--')

plt.xlabel("$n$", fontsize=24)
plt.ylabel("Approximation ratio", fontsize=24)
plt.legend(fontsize=18, facecolor='white', framealpha=1)
plt.grid(True, which="both", linestyle='-', color='lightgray', linewidth=0.5)

plt.tick_params(axis='x', labelsize=18)
# plt.tick_params(axis='y', labelsize=18)
plt.yticks([0.6, 0.7, 0.8, 0.9, 1, 2], [f"${i}$" for i in [0.6, 0.7, 0.8, 0.9, 1, 2]], fontsize=18)
# plt.xticks([10**i for i in range(0, 6)], [f"$10^{i}$" for i in range(0, 6)], fontsize=18)
# plt.yticks([10**i for i in range(0, 8)], [f"$10^{i}$" for i in range(0, 8)], fontsize=18)
# plt.xlim((1,1e4))
plt.ylim((0.5,2))

plt.tight_layout()
plt.savefig('figures/Fig_224_Sterling_approximation_2.png')
plt.show()