from jax import grad, jit, vmap, value_and_grad
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import jax
from jax import random
from functools import partial


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

def main():
    data = generate_data().reshape(-1,1)
    theta = np.array([4.,0.5,0.01,0.01])
    lr = 1e-4
    partial_log_posterior = partial(log_posterior, y=data)
    # eta = log_var_sample(theta)
    # partial_log_likelihood = partial(log_likelihood, y=data)
    # print("LOGGG", eta[0].shape)
    # print(partial_log_posterior(eta).shape, log_priors(eta).shape)
    
    # log_var = log_var_approx(theta, eta)
    # log_post = partial_log_posterior(eta)

    # print(log_post.shape, log_var.shape)
    
    # print(elbo(theta))

    # def elbo(theta, y):
    #     # eta = log_var_sample(theta)
    #     D=2
    #     mu, sigma = theta[:D,], theta[D:,]
    #     eta = (random.normal(key=key, shape=(D, 50)) * sigma.reshape(-1,1)) + mu.reshape(-1, 1)

    #     log_likel = jsp.stats.norm.logpdf(y, eta[0], eta[1]).sum(axis=0)
    #     mu_prior = jsp.stats.norm.logpdf(eta[0], loc=0, scale=10)
    #     sigma_prior = jsp.stats.norm.logpdf(eta[1], loc=1, scale=0.25)
    #     log_post = log_likel + mu_prior + sigma_prior

    #     log_var_approx2 = jsp.stats.norm.logpdf(eta[0], mu[0], sigma[0]) + jsp.stats.norm.logpdf(eta[1], mu[1], sigma[1])

    #     # return jnp.mean(log_var_approx(theta, eta) - partial_log_posterior(eta))
    #     return jnp.mean(log_var_approx2 - log_post)

    def elbo(theta):
        eta = log_var_sample(theta)
        return jnp.mean(log_var_approx(theta, eta) - partial_log_posterior(eta))
        # return jnp.mean(log_var_approx(theta, eta) - log_post)

    # elbo = jit(value_and_grad(partial(elbo, y=data)))
    elbo = jit(value_and_grad(elbo))

    for i in range(500):

        # eta = log_var_sample(theta)
        # elbo = lambda theta, eta: jnp.mean(log_var_approx(theta, eta) - partial_log_posterior(eta))
        eval = elbo(theta)
        theta = theta - lr * eval[1]
        if i % 10 == 0:
            print(theta)
    return


    

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
    log = main()
    # print(log['ELBO'][-1], log['theta'][-1])
