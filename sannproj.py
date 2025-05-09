import numpy as np
from numpy.linalg import matrix_power
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



def generate_data(x1, x2, w, sigma):
    """
    Generate data based on the model t = w^T * phi + epsilon
    where phi is a vector of features and epsilon is Gaussian noise.
    """
    # Create the feature vector phi
    phi = np.vstack((np.ones_like(x1), x1**2, x2**3))
    
    # Generate Gaussian noise
    epsilon = np.random.normal(0, sigma, size=len(x1))
    
    # Calculate the target variable t
    t = w.T @ phi + epsilon
    
    return t

def plot_3d_surface(x1, x2, t, sigma, text):
    """
    Plot the 3D surface of the generated data.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, t, c=t, cmap='plasma', s=10)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('t')
    ax.set_title(f'{text} genererad från modellen med σ = {sigma}')
    plt.tight_layout()
    


# == Pre-values =======================
w = np.array([0, 2.5, -0.5])
sigma = 0.3 # eller 0.5 0.8 1.2
# ======================================


# All x1,x2
x1range = np.linspace(-1, 1, 41) # [-1, -0.95,..., 0.95, 1]
x2range = np.linspace(-1, 1, 41)  # [-1, -0.95,..., 0.95, 1]

## Get the grid (matrix of x1 & x2),  [-1, -0.95,..., 0.95, 1] x  [-1, -0.95,..., 0.95, 1]
X1, X2 = np.meshgrid(x1range, x2range)
x1_flat = X1.flatten()
x2_flat = X2.flatten()

# == Get all t data ==
Full_data_set = generate_data(x1_flat, x2_flat, w, sigma)

plot_3d_surface(x1_flat, x2_flat, Full_data_set, sigma, "Alla data från x1,x2")


# === 2 ===
# All x1,x2
x1rangeTraining = np.linspace(-0.3, 0.3, 13) # [-0.3, -0.25,..., 0.25, 0.3]
x2rangeTraining = np.linspace(-0.3, 0.3, 13) # [-0.3, -0.25,..., 0.25, 0.3]

## Get the grid (matrix of x1 & x2),  [-0.3, -0.25,..., 0.25, 0.3] x  [-0.3, -0.25,..., 0.25, 0.3]
X1, X2 = np.meshgrid(x1rangeTraining, x2rangeTraining)
x1_flatTr = X1.flatten()
x2_flatTr = X2.flatten()
traning_subset = generate_data(x1_flatTr, x2_flatTr, w, sigma)

plot_3d_surface(x1_flatTr, x2_flatTr, traning_subset, sigma, "Träningsdata")


# == Test data ==
x1rangeTest = np.concatenate((np.linspace(-1, 0.35, 14),np.linspace(0.35, 1, 14)))
x2rangeTest = np.concatenate((np.linspace(-1, 0.35, 14),np.linspace(0.35, 1, 14)))

X1, X2 = np.meshgrid(x1rangeTest, x2rangeTest)
x1_flatTest = X1.flatten()
x2_flatTest = X2.flatten()
test_subset = generate_data(x1_flatTest, x2_flatTest, w, sigma) + np.random.normal(0, sigma, size=len(x1_flatTest))

plot_3d_surface(x1_flatTest, x2_flatTest, test_subset, sigma, "Testdata")

# == 3 == Maximum Likelihood
X = np.vstack((np.ones_like(x1_flatTr), x1_flatTr**2, x2_flatTr**3)).T #  φ(x) = Xest
t = np.array(traning_subset)
w_ml = np.linalg.inv(X.T @ X) @ X.T @ t  # Ekvation nr 21

# Generera test output utifrån nya w.
phi = np.vstack((np.ones_like(x1_flatTest), x1_flatTest**2, x2_flatTest**3))
t = w_ml.T @  phi # Vår gissade w och phi(x1,x2) ger oss gissning av t
t_true = w.T @  phi # Test with no noice and correct w

plot_3d_surface(x1_flatTest, x2_flatTest, t, sigma, "uppg3. ML-prediktion")

# Mean square error
MSE = sum((t-t_true)**2)/len(t) # Ekvation nr ? , fanns i slides
print("Mean square Error (ML): " + str(MSE))


# == 4 == Bayesiansk linjär regression

alpha = 0.3  # Du kan variera mellan {0.3, 0.7, 2.0}
alphaVals = [0.3, 0.7, 2.0]
beta = 1 / sigma**2

# Träningsdesignmatris (Phi)
Xext = np.vstack((np.ones_like(x1_flatTr), x1_flatTr**2, x2_flatTr**3)).T  # N x D
t_train = np.array(traning_subset)

# Testdesignmatris (Phi_*)
Phi_test = np.vstack((np.ones_like(x1_flatTest), x1_flatTest**2, x2_flatTest**3)).T  # M x D

# Posterior: S_N och m_N
S_N_inv = alpha * np.eye(Xext.shape[1]) + beta * Xext.T @ Xext  # D x D
S_N = np.linalg.inv(S_N_inv)
m_N = beta * S_N @ Xext.T @ t_train  # D x 1

# === Prediktivt medelvärde och varians ===

# Prediktivt medelvärde för alla testpunkter
mu_N = Phi_test @ m_N  # M x 1

# Prediktiv varians för alla testpunkter (vektoriserat)
sigma2_N = 1 / beta + np.sum(Phi_test @ S_N * Phi_test, axis=1)  # M x 1

# === Plot ===
plot_3d_surface(x1_flatTest, x2_flatTest, mu_N, sigma, f"4. Bayesianskt prediktivt medelvärde med alpha = {alpha} och")

# Plotta osäkerheten som separat yta
plot_3d_surface(x1_flatTest, x2_flatTest, sigma2_N, sigma, f"4. Bayesiansk prediktiv varians alpha = {alpha} och")




# == 5. Jämför Maximum Likelihood med Bayesiansk linjär regression

# MSE för ML printas i uppgift 3
MSEBay = sum((mu_N-t_true)**2)/len(t)
print("Mean square Error (Bayesian): " + str(MSEBay))



# == Visa alla figurer tsm ==
plt.show()