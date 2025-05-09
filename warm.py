import numpy as np
import math
import random
import matplotlib
matplotlib.use('TkAgg')  # eller 'Qt5Agg' om du har Qt installerat
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
#from scipy.spatial.distance import cdist

# == 1 == Rita priorfördelningen

w0list = np.linspace(-2.0, 2.0, 200)
w1list = np . linspace (-2.0, 2.0, 200)
W0arr, W1arr = np.meshgrid(w0list, w1list )
pos = np.dstack((W0arr, W1arr))

mu = np.array([0.0, 0.0])
alpha = 2.0
Cov = (1.0 / alpha) * np.identity(2)

# set your mu vector and Cov array
rv = multivariate_normal(mu,Cov)
Wpriorpdf = rv.pdf(pos)


plt.figure(figsize=(8, 6))
plt.contour(W0arr, W1arr, Wpriorpdf)
plt.title('Prior')
plt.xlabel('w0')
plt.ylabel('w1')


## 2 ##
# Parametrar
w = [-1.2, 0.9]
sigma2 = 0.2 # variera mellan 0.1, 0.2, 0.4 and 0.8

# Skapa hela träningsdatasetet en gång
X_training = np.linspace(-1.0, 1.0, 201)
noise = np.random.normal(0, sigma2, size=X_training.shape)
T_training = w[0] + w[1] * X_training + noise

# Kombinera till (x, t) par
full_training_data = [[float(xi), float(ti)] for xi, ti in zip(X_training, T_training)]

# Exempel:
subset1 = random.sample(full_training_data, 3)
subset2 = random.sample(full_training_data, 10)
subset3 = random.sample(full_training_data, 20)
subset4 = random.sample(full_training_data, 100)

def compute_log_likelihood(w0_grid, w1_grid, training_data, sigma2):
    log_likelihood = np.zeros(w0_grid.shape)
    for xi, ti in training_data:
        pred = w0_grid + w1_grid * xi
        log_likelihood += -0.5 * np.log(2 * np.pi * sigma2) - ((ti - pred)**2) / (2 * sigma2)
    return np.exp(log_likelihood)  # för att få tillbaka sannolikhetsfördelningen

likelihood = [
compute_log_likelihood(W0arr, W1arr, subset1, sigma2),
compute_log_likelihood(W0arr, W1arr, subset2, sigma2),
compute_log_likelihood(W0arr, W1arr, subset3, sigma2),
compute_log_likelihood(W0arr, W1arr, subset4, sigma2)]


plt.figure(figsize=(8, 6))
for i in range(len(likelihood)):
    plt.subplot(2, 2, i+1)
    plt.contour(W0arr, W1arr, likelihood[i])
    plt.title('Subset ' + str(i+1))
    plt.xlabel('w0')
    plt.ylabel('w1')


# == 3 == 

# Posterior enligt Eqs. 27 & 28
def compute_posterior(Phi, t, alpha, beta):
    S_N_inv = alpha * np.eye(2) + beta * Phi.T @ Phi
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N @ Phi.T @ t
    return m_N, S_N

# Förbered träningsdata
x_train = np.array([x for x, t in full_training_data])
t_train = np.array([t for x, t in full_training_data])

Phi = np.vstack([np.ones_like(x_train), x_train]).T

alpha = 2.0
beta = 1.0 / sigma2  # 1 / 0.2

m_N, S_N = compute_posterior(Phi, t_train, alpha, beta)

posterior = multivariate_normal(mean=m_N, cov=S_N)
posterior_pdf = posterior.pdf(pos)

plt.figure(figsize=(8, 6))
plt.contour(W0arr, W1arr, posterior_pdf)
plt.title('Posterior')
plt.xlabel('w0')    
plt.ylabel('w1')

# == 4 == 

# Skapa testdata 
x_test = np.concatenate((np.linspace(-1.5, -1.1, 5), np.linspace(1.1, 1.5, 5)))
t_test = w[0] + w[1] * x_test + np.random.normal(0, sigma2, size=x_test.shape)

# Ta 5 samples från posterior
ws_samples = np.random.multivariate_normal(m_N, S_N, 5)

# Skapa x-område för linjer
x_plot = np.linspace(-2, 2, 200)

# Plotta träningsdata och testdata
x_train = np.array([x for x, t in full_training_data])
t_train = np.array([t for x, t in full_training_data])
plt.figure(figsize=(8,6))
plt.scatter(x_train, t_train, color='black', marker='x', label='Training data')
plt.scatter(x_test, t_test, color='red',marker="x",label="Test data")

# Plotta linjerna
for w in ws_samples:
    y_plot = w[0] + w[1] * x_plot
    plt.plot(x_plot, y_plot, label=f"w0={w[0]:.2f}, w1={w[1]:.2f}")
    
plt.legend()
plt.title("Samples från posteriorn")

# == 5 ==

Phi_test = np.vstack((np.ones_like(x_test), x_test)).T

mean_pred = Phi_test @ m_N
var_pred = 1 / beta + np.sum(Phi_test @ S_N * Phi_test, axis=1)
std_pred = np.sqrt(var_pred)

plt.figure(figsize=(8,6))
plt.errorbar(x_test, mean_pred, yerr=std_pred, fmt='o', label="Bayesiansk prediktion")
plt.scatter(x_train, t_train, color='black', marker="x", label="Träningsdata")

# == 6 ==
# Maximum Likelihood-estimat
X = np.vstack((np.ones_like(X_training), X_training)).T
t = np.array(T_training)
w_ml = np.linalg.inv(X.T @ X) @ X.T @ t

# Rita ML-linje (orange)
y_ml = w_ml[0] + w_ml[1] * x_plot
plt.plot(x_plot, y_ml, 'orange', linewidth=2, label="Maximum Likelihood estimat") 
plt.legend()
plt.title("Bayesiansk prediktiv fördelning")
plt.show()