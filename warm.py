import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')  # eller 'Qt5Agg' om du har Qt installerat
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
#from scipy.spatial.distance import cdist

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


plt.contour(W0arr, W1arr, Wpriorpdf)
plt.show()

## 2 ##
# Parametrar
w = [-1.2, 0.9]
sigma2 = 0.2

# Skapa hela träningsdatasetet en gång
X_training = np.linspace(-1.0, 1.0, 201)
noise = np.random.normal(0, sigma2, size=X_training.shape)
T_training = w[0] + w[1] * X_training + noise

# Kombinera till (x, t) par
full_training_data = [[float(xi), float(ti)] for xi, ti in zip(X_training, T_training)]

# Exempel:
subset1 = full_training_data[:10]
subset2 = full_training_data[:20]
subset3 = full_training_data[:100]

def compute_log_likelihood(w0_grid, w1_grid, training_data, sigma2):
    log_likelihood = np.zeros(w0_grid.shape)
    for xi, ti in training_data:
        pred = w0_grid + w1_grid * xi
        log_likelihood += -0.5 * np.log(2 * np.pi * sigma2) - ((ti - pred)**2) / (2 * sigma2)
    return np.exp(log_likelihood)  # för att få tillbaka sannolikhetsfördelningen

likelihood_pdf = compute_log_likelihood(W0arr, W1arr, full_training_data, sigma2)

plt.contour(W0arr, W1arr, likelihood_pdf)
plt.show()


# == 3 == 

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

plt.contour(W0arr, W1arr, posterior_pdf)
plt.show()

# == 4 == 
# 1. Rita 5 samples från posterior
ws_samples = np.random.multivariate_normal(m_N, S_N, 5)

# 2. Skapa x-område för linjer
x_plot = np.linspace(-2, 2, 200)





# 3. Plotta linjerna
plt.figure(figsize=(8,6))
for w in ws_samples:
    y_plot = w[0] + w[1] * x_plot
    plt.plot(x_plot, y_plot, label=f"w0={w[0]:.2f}, w1={w[1]:.2f}")

# 4. Plotta träningsdata 
x_train = np.array([x for x, t in full_training_data])
t_train = np.array([t for x, t in full_training_data])
plt.scatter(x_train, t_train, color='black', marker='x', label='Training data')
def predict_posterior(x_star_list, m_N, S_N, beta):
    x_star_list = np.array(x_star_list)
    X_star = np.vstack((np.ones_like(x_star_list), x_star_list)).T  # [1, x*]
    
    mean_preds = X_star @ m_N
    var_preds = 1 / beta + np.sum(X_star @ S_N * X_star, axis=1)
    std_preds = np.sqrt(var_preds)
    
    return mean_preds, std_preds

# === DINA EGETA TESTPUNKTER HÄR ===
x_custom = [-2,-1.2, 0.0, 1.3, 2]  # ← Ändra fritt

# === BERÄKNA PREDIKTIV FÖRDELNING ===
mu_custom, std_custom = predict_posterior(x_custom, m_N, S_N, beta)

# === PLOT ===
plt.errorbar(x_custom, mu_custom, yerr=std_custom, fmt='o', color='red', capsize=5, label='Prediktion med osäkerhet')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)

plt.show()

# == 6 ==
# Maximum Likelihood-estimat
X = np.vstack((np.ones_like(X_training), T_training)).T
t = np.array(T_training)
w_ml = np.linalg.inv(X.T @ X) @ X.T @ t

# Rita ML-linje (orange)
y_ml = w_ml[0] + w_ml[1] * x_plot
plt.plot(x_plot, y_ml, 'orange', label='ML-prediktion', linewidth=2)
plt.show()

