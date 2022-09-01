#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import bisect

# %%
N = 100
mu = np.linspace(0.01, 1, N)
k = 1.0
sigma = k * mu
def cdf_below(x):
    return stats.norm.cdf(x, loc=mu, scale=sigma)

cdf_below(0)
# %%
dists = [np.ones(N) / N]

gamma = 0.1
T = 100
for i in range(T):
    current_dist = dists[-1]
    x0 = bisect(lambda x: current_dist.dot(cdf_below(x)) - gamma, -50, 50)
    new_dist = current_dist * cdf_below(x0)
    new_dist /= np.sum(new_dist)
    dists.append(new_dist)

# %%
for i in range(0, T, 10):
    # print(len(new_dist[i]))
    plt.plot(mu, dists[i])
# %%
