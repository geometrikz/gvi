from jax import grad, jit, vmap, value_and_grad
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import jax
from jax import random
from functools import partial
from jax.example_libraries import optimizers

from sklearn import datasets

key = random.PRNGKey(0)

def generate_data(shape=(100,)):
    return random.normal(key, shape=shape) + 5

def genereate_regression_data():
    return

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
    # D = len(theta)//2
    # theta = theta[:D,]
    return (log_likelihood(theta, y) + log_priors(theta))
    # return -(log_likel.mean() + priors(theta).mean())

def log_var_approx(theta, x):
    D = len(theta)//2
    mu, sigma = theta[:D,], theta[D:,]
    # eta_m = (random.normal(key=key, shape=(2, 100)) * sigma) + mu.reshape(-1, 1)
    # print(jnp.diag(sigma))
    return jsp.stats.multivariate_normal.logpdf(x.T, mu.reshape(D), jnp.diag(sigma))
    # return jsp.stats.norm.logpdf(x[0], mu[0], sigma[0]) + jsp.stats.norm.logpdf(x[1], mu[1], sigma[1])

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

def blr():
    D = 5
    x, y, coef = datasets.make_regression(n_samples=100, n_features=5, n_informative=5, bias=0, noise=5, coef=True)
    y = y.reshape(-1,1)
    theta = jnp.zeros(D*2)

    @jit
    def log_priors(theta):
        # Independent priors
        mu_prior = jsp.stats.norm.logpdf(theta, loc=0, scale=10).sum()
        
        return mu_prior

    @jit
    def log_likelihood(theta, y, x):
        mu = jnp.dot(x, theta)
        log_likel = jsp.stats.norm.logpdf(y, mu, 5).sum(axis=0)

        return log_likel

    @jit
    def log_posterior(theta, y, x):
        return (log_likelihood(theta, y, x) + log_priors(theta))

    def log_var_approx(theta, samples):
        D = len(theta)//2
        mu, mu_var = theta[:D,], jnp.exp(theta[D:,])
        # eta_m = (random.normal(key=key, shape=(2, 100)) * sigma) + mu.reshape(-1, 1)
        # print(jnp.diag(sigma))
        return jsp.stats.multivariate_normal.logpdf(samples.T, mu.reshape(D), jnp.diag(mu_var))
        # return jsp.stats.norm.logpdf(x[0], mu[0], sigma[0]) + jsp.stats.norm.logpdf(x[1], mu[1], sigma[1])

    def log_var_sample(theta):
        D = len(theta)//2
        mu, sigma = theta[:D,], jnp.exp(theta[D:,])
        # print(sigma.shape, mu.shape)
        eta_m = (random.normal(key=key, shape=(D, 50)) * sigma.reshape(-1,1)) + mu.reshape(-1, 1)
        return eta_m

    log_posterior = partial(log_posterior, y=y, x=x)

    @jit
    def elbo(theta):
        eta = log_var_sample(theta)
        return jnp.mean(log_var_approx(theta, eta) - log_posterior(eta))

    elbo = jit(value_and_grad(elbo))

    opt_init, opt_update, get_params = optimizers.sgd(step_size=1e-3)
    opt_state = opt_init(theta)
    
    epochs = 100
    logs = {'loss':jnp.zeros(epochs)}

    for i in range(epochs):
        loss, der = elbo(theta)

        logs['loss'] = logs['loss'].at[i].set(loss)

        # theta = theta - lr * der
        opt_state = opt_update(i, der, opt_state)
        theta = get_params(opt_state)
        # print(opt_state)
        if i % 10 == 0:
            print(theta[:D], loss)
    
    print(coef)

    



    

def run():
    data = generate_data()
    # posterior = random.normal(key).pdf()
    # print(priors(5))
    # print(gauss_likelihood(data, [10,1]))
    init_theta = np.array([2.6, 1, 0.01, 0.01])
    theta = init_theta
    # ELBO = posterior(theta, data)
    ELBO = -np.inf

    # print(grad(posterior, argnums=0)(theta, data))
    lr = 1e-4

    log = {'ELBO':[ELBO], 'theta':[theta]}
    # to_minimize = partial(posterior, y=data.reshape(-1,1))

    def to_minimize2(theta, y):
        mu = theta[0]
        sigma = theta[1]
        var_sigma = theta[2:].reshape(-1,1)
        
        eta_m = (random.normal(key=key, shape=(2, 100)) * var_sigma) + jnp.array([mu, sigma]).reshape(-1, 1)

        log_likel = jsp.stats.norm.logpdf(y, eta_m[0], eta_m[1]).sum(axis=0)


        mu_prior = jsp.stats.norm.logpdf(eta_m[0], loc=0, scale=10)
        sigma_prior = jsp.stats.norm.logpdf(eta_m[1], loc=1, scale=0.25)

        log_var_approx = jsp.stats.norm.logpdf(eta_m[0], mu, var_sigma[0]) + jsp.stats.norm.logpdf(eta_m[1], sigma, var_sigma[1])

        prior = mu_prior + sigma_prior

        return -jnp.mean(log_likel + prior - log_var_approx)
        # return posterior(eta_m, y=y)



    elbo_fun = jit(value_and_grad(partial(to_minimize2, y=data.reshape(-1,1))))

    # for i in range(100):
    # eta_m = np.random.normal(jnp.exp(theta[0]), jnp.exp(theta[1]), size=(2, 25))
    # print(jsp.stats.norm.logpdf(data.reshape(-1,1), eta_m[0], eta_m[1]).sum(axis=0))
    # print(priors(eta_m))
    # print(elbo_fun(init_theta))
    # ELBO, ELBO_grad = elbo_fun(eta_m)

    from scipy.optimize import minimize

    def np_elbo(theta):
        f, g = elbo_fun(theta)
        return -float(np.array(f)), np.array(g, dtype='float64')
    
    print(np_elbo(init_theta))

    # result = minimize(np_elbo, init_theta, method='L-BFGS-B', jac=True)
    # print(result)
    # print(result['x'])
    # nabla_theta = grad(posterior, argnums=0)(eta_m, data)
    print(theta)
    for i in range(1000):
        ELBO = np_elbo(theta)[0]
        theta = theta - lr * np_elbo(theta)[1]
        # theta = jnp.exp(theta)

        log['ELBO'].append(ELBO)
        log['theta'].append(theta)
        
        if i % 100 == 0:
            print(ELBO, theta)

    return log

if __name__ == '__main__':
    # print(generate_data())
    log = blr()
    # print(log['ELBO'][-1], log['theta'][-1])
    # x, y, coef = datasets.make_regression(n_samples=100, n_features=10, n_informative=3, bias=5, coef=True)
    # print(x.shape, y.shape, coef)
