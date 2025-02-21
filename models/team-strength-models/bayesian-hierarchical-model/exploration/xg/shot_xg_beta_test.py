import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\Users\Owner\dev\football-analytics\models\team-strength-models\bayesian-hierarchical-model\exploration\xg\shot_data_prem_2023.csv")


import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def zero_inflated_beta_pdf(x, alpha, beta, pi):
    """
    PDF of zero-inflated beta distribution
    pi: probability of zero
    alpha, beta: beta distribution parameters
    """
    # Handle zeros separately
    if np.isscalar(x):
        if x == 0:
            return pi
        else:
            return (1 - pi) * stats.beta.pdf(x, alpha, beta)
    else:
        pdf = np.zeros_like(x)
        pdf[x == 0] = pi
        non_zero = x > 0
        pdf[non_zero] = (1 - pi) * stats.beta.pdf(x[non_zero], alpha, beta)
        return pdf

def zero_inflated_beta_nll(params, data):
    """Negative log-likelihood of zero-inflated beta"""
    alpha, beta, pi = params
    if alpha <= 0 or beta <= 0 or pi < 0 or pi > 1:
        return np.inf
    
    # Calculate log likelihood
    non_zero = data > 0
    n_zero = np.sum(~non_zero)
    
    ll_zero = n_zero * np.log(pi) if pi > 0 else -np.inf
    ll_non_zero = np.sum(np.log(1 - pi) + stats.beta.logpdf(data[non_zero], alpha, beta))
    
    return -(ll_zero + ll_non_zero)

def fit_zero_inflated_beta(data):
    """Fit zero-inflated beta distribution to data"""
    # Initial guess: use method of moments for beta parameters
    non_zero = data[data > 0]
    mean = np.mean(non_zero)
    var = np.var(non_zero)
    
    # Method of moments estimates for beta parameters
    temp = mean * (1 - mean) / var - 1
    alpha_guess = mean * temp
    beta_guess = (1 - mean) * temp
    pi_guess = np.mean(data == 0)
    
    # Optimize parameters
    result = minimize(zero_inflated_beta_nll, 
                     [alpha_guess, beta_guess, pi_guess], 
                     args=(data,),
                     bounds=[(0.001, None), (0.001, None), (0, 1)])
    
    return result.x

def plot_zero_inflated_beta_fit(data, params):
    """Plot histogram of data with fitted zero-inflated beta"""
    alpha, beta, pi = params
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of non-zero values
    non_zero = data[data > 0]
    plt.hist(data, bins=50, density=True, alpha=0.7, label='Data')
    
    # Plot fitted distribution
    x = np.linspace(0, 1, 1000)
    y = zero_inflated_beta_pdf(x, alpha, beta, pi)
    plt.plot(x, y, 'r-', label=f'Fitted ZIB(α={alpha:.2f}, β={beta:.2f}, π={pi:.2f})')
    
    # Add zero component as a spike
    if pi > 0:
        plt.plot([0], [pi], 'ro', markersize=10, label='Zero component')
    
    plt.title('Data vs Fitted Zero-Inflated Beta Distribution')
    plt.legend()
    plt.show()

params = fit_zero_inflated_beta(df["xG"])
print(f"Fitted parameters:\nAlpha: {params[0]:.3f}\nBeta: {params[1]:.3f}\nPi (zero probability): {params[2]:.3f}")

# Visualize the fit
plot_zero_inflated_beta_fit(df["xG"], params)