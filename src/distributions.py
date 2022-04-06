# import torch
import math
import jax
import jax.numpy as jnp
from numpy.random import choice
from jax.scipy.special import logsumexp

jax.config.update("jax_platform_name", "cpu")


class Gaussian(object):
    def __init__(self, d):
        # d is dimension of the random variable space
        # we only consider diagonal covariation matrices
        self.d = d
        self.key = jax.random.PRNGKey(0)

    def _atleast_2d(self, array):
        if len(array.shape) < 2:
            return array[None, :]
        return array

    def unflatten(self, params):
        params = self._atleast_2d(params)
        N = params.shape[0]
        # first d params are mu values
        mus = params[:, : self.d]
        # following params define the covariance matrix
        log_sigmas = params[:, self.d :]
        return {"mus": mus, "sigmas": jnp.exp(log_sigmas)}

    def log_pdf(self, params, X):
        theta = self.unflatten(params)
        if len(X.shape) == 1 and self.d == 1:
            # in one-dimensional case need to add a dimension so that each row is an observation
            X = X[:, None]

        X = self._atleast_2d(X)
        mus = theta["mus"]
        sigmas = theta["sigmas"]

        log_pdf = (
            -0.5 * mus.shape[1] * jnp.log(jnp.array(2 * math.pi))
            - 0.5 * jnp.sum(jnp.log(sigmas), axis=1)
            - 0.5 * jnp.sum((X[:, None, :] - mus) ** 2 / sigmas, axis=2)
        )

        if log_pdf.shape[1] == 1:
            log_pdf = log_pdf[:, 0]
        return log_pdf

    def generate_samples_for_one_component(self, param, num_samples):
        # TODO Change this to not mean-field
        epsilons = jax.random.normal(self.key, shape=(num_samples, self.d))

        # epsilons = torch.randn(num_samples, self.d)
        mu = param[:, : self.d]

        log_sigma = param[:, self.d :]
        # print(log_sigma.shape) # TODO: Remove
        std = jnp.exp(log_sigma / 2)
        # print(epsilons.shape, mu.shape, std.shape) #TODO: Remove
        return mu + epsilons * std

    def _get_paired_param(self, param_a, param_b, flatten=False):
        # TODO Investigate
        theta = self.unflatten(jnp.stack((param_a, param_b)))
        mus = theta["mus"]
        sigmas = theta["sigmas"]

        paired_sigma = 2.0 / (1 / sigmas[0, :] + 1 / sigmas[1, :])
        paired_mu = (
            0.5 * paired_sigma * (mus[0, :] / sigmas[0, :] + mus[1, :] / sigmas[1, :])
        )

        if not flatten:
            return paired_mu, paired_sigma
        else:
            return jnp.cat((paired_mu, jnp.log(paired_sigma)))

    def generate_samples_for_paired_distribution(self, param_a, param_b, num_samples):
        paired_mu, paired_sigma = self._get_paired_param(param_a, param_b)

        return paired_mu + jnp.sqrt(paired_sigma) * jax.random.normal(
            key=self.key, shape=(num_samples, self.d)
        )

    def log_sqrt_pair_integral(self, new_param, old_params):
        old_params = self._atleast_2d(old_params)
        mu_new = new_param[: self.d]
        mus_old = old_params[:, : self.d]

        log_sigma_new = new_param[self.d :]
        log_sigmas_old = old_params[:, self.d :]
        print(log_sigma_new.shape, log_sigmas_old.shape)
        log_sigmas_all = jnp.log(jnp.array(0.5)) + logsumexp(
            jnp.stack([log_sigma_new.reshape(log_sigmas_old.shape), log_sigmas_old]),
            axis=0,  # TODO log_sigma_new.reshape might break
        )

        return (
            -0.125 * jnp.sum(jnp.exp(-log_sigmas_all) * (mu_new - mus_old) ** 2, axis=1)
            - 0.5 * jnp.sum(log_sigmas_all, axis=1)
            + 0.25 * jnp.sum(log_sigma_new)
            + 0.25 * jnp.sum(log_sigmas_old, axis=1)
        )

    def params_init(self, params, weights, inflation):
        # initialization with heuristics

        params = self._atleast_2d(params)

        i = params.shape[0]
        if i == 0:
            mu = jax.random.normal(key=self.key, shape=(self.d,)) * inflation

            log_sigma = jnp.zeros(self.d)
            new_param = jnp.concatenate((mu, log_sigma), axis=0)
        else:
            mus = params[:, : self.d]
            probs = (weights**2) / (weights**2).sum()
            print(probs.shape)
            print(jnp.arange(i).shape)
            k = jax.random.choice(key=self.key, a=jnp.arange(i), p=probs)

            log_sigmas = params[:, self.d :]
            mu = mus[k, :] + jax.random.normal(self.key, (self.d,)) * jnp.sqrt(
                jnp.array(inflation)
            ) * jnp.exp(log_sigmas[k, :])
            log_sigma = jax.random.normal(self.key, (self.d,)) + log_sigmas[k, :]
            new_param = jnp.concatenate((mu, log_sigma), axis=0)
        return new_param

    def print_perf(self, itr, x, obj):
        if itr == 0:
            print(
                "{:^30}|{:^30}|{:^30}|{:^30}".format(
                    "Iteration", "mu", "log_sigma", "Boosting Obj"
                )
            )
        if self.diag:
            print(
                "{:^30}|{:^30}|{:^30}|{:^30.3f}".format(
                    itr,
                    str(x[: min(self.d, 4)]),
                    str(x[self.d : self.d + min(self.d, 4)]),
                    obj,
                )
            )


if __name__ == "__main__":
    # g = Gaussian(1)
    X = jnp.array([[0, 0]])
    # params = jnp.array([[0, 0]])
    # print(g._atleast_2d(jnp.array([0])))
    # print(g.unflatten(jnp.array([0, 0])))
    # print(g.log_pdf(params, X))
    # print(g.generate_samples_for_one_component(params, 100))
    # print(jnp.stack((1, 0)))
    # print(g._get_paired_param(1, 0))
    # print(g.generate_samples_for_paired_distribution(0, 1, 100))
    # print(g.log_sqrt_pair_integral(jnp.array([0.5, 0.5]), params))
    # print(g.params_init(params, jnp.array([1]), jnp.array([1, 1])))
