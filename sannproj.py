import numpy as np
from numpy.linalg import matrix_power
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# == Pre-values
x1range = np.linspace(-1, 1, 41) # [-1, -0.95,..., 0.95, 1]
x2range = np.linspace(-1, 1, 41)  # [-1, -0.95,..., 0.95, 1]
w = np.array([0, 2.5, -0.5])


## Get the grid (matrix of x1 & x2),  [-1, -0.95,..., 0.95, 1] x  [-1, -0.95,..., 0.95, 1]
X1, X2 = np.meshgrid(x1range, x2range)
x1_flat = X1.flatten()
x2_flat = X2.flatten()


def power(my_list, pow):
    return [ x**pow for x in my_list ]



# Get vector phi φ(x)
phi = np.vstack((np.ones_like(x1_flat), x1_flat**2, x2_flat**3)).T

# Generate epsilon ~ N(0, sigma^2)
sigma = 0.3 # Can also be changed to 0.5, 0.8 or 1.2
my = 0
epsilon = np.random.normal(my, sigma**2, size=len(x1_flat))
# 
# == Get training data ==
T_training = phi @ w + epsilon

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1_flat, x2_flat, T_training, c=T_training, cmap='plasma', s=10)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('t')
ax.set_title(f'Data genererad från modellen med σ = {sigma}')
plt.tight_layout()
plt.show()