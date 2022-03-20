from jax import grad, jit, vmap, value_and_grad
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import jax
from jax import random
from functools import partial
from jax.example_libraries import optimizers

jax.config.update('jax_platform_name', 'cpu')

from sklearn import datasets

key = random.PRNGKey(0)

def generate_data(shape=(100,)):
    return random.normal(key, shape=shape) + 5

@jit
def log_priors(theta):
    mu_prior = jsp.stats.norm.logpdf(theta[0], loc=4, scale=10)
    sigma_prior = jsp.stats.norm.logpdf(theta[1], loc=1, scale=0.25)
    return mu_prior + sigma_prior

@jit
def log_likelihood(theta, y):
    mu = theta[0]
    sigma = theta[1]

    log_likel = jsp.stats.norm.logpdf(y, mu, sigma).sum(axis=0)

    return log_likel

@jit
def log_posterior(theta, y):
    return (log_likelihood(theta, y) + log_priors(theta))

def log_var_approx(theta, x):
    D = len(theta)//2
    mu, sigma = theta[:D,], theta[D:,]
    # eta_m = (random.normal(key=key, shape=(2, 100)) * sigma) + mu.reshape(-1, 1)
    return jsp.stats.multivariate_normal.logpdf(x.T, mu.reshape(D), jnp.diag(sigma))

def log_var_sample(theta):
    D = len(theta)//2
    mu, sigma = theta[:D,], theta[D:,]
    # print(sigma.shape, mu.shape)
    eta_m = (random.normal(key=key, shape=(D, 50)) * sigma.reshape(-1,1)) + mu.reshape(-1, 1)
    return eta_m


data = generate_data().reshape(-1,1)
log_posterior = partial(log_posterior, y=data)

def main():
    theta = np.array([4.,0.5,0.01,0.01])
    lr = 1e-4

    @jit
    def elbo(theta):
        eta = log_var_sample(theta)
        return jnp.mean(log_var_approx(theta, eta) - log_posterior(eta))

    elbo = jit(value_and_grad(elbo))

    opt_init, opt_update, get_params = optimizers.sgd(step_size=1e-3)
    opt_state = opt_init(theta)
    
    for i in range(1000):
        loss, der = elbo(theta)
        # theta = theta - lr * der

        opt_state = opt_update(i, der, opt_state)
        theta = get_params(opt_state)
        # print(opt_state)
        if i % 10 == 0:
            print(theta)
    return


def advi():
    theta = np.array([4.,0.5,0.01,0.01])
    lr = 1e-4

    @jit
    def elbo(theta):
        eta = log_var_sample(theta)
        return jnp.mean(log_var_approx(theta, eta) - log_posterior(eta))

    elbo = jit(value_and_grad(elbo))

    opt_init, opt_update, get_params = optimizers.sgd(step_size=1e-3)
    opt_state = opt_init(theta)
    
    for i in range(1000):
        loss, der = elbo(theta)
        # theta = theta - lr * der

        opt_state = opt_update(i, der, opt_state)
        theta = get_params(opt_state)
        # print(opt_state)
        if i % 10 == 0:
            print(theta)
    return



if __name__ == '__main__':
    # print(generate_data())
    log = advi()
    # print(log['ELBO'][-1], log['theta'][-1])
    # x, y, coef = datasets.make_regression(n_samples=100, n_features=10, n_informative=3, bias=5, coef=True)
    # print(x.shape, y.shape, coef)
