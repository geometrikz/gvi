from jax import grad, jit, vmap, value_and_grad
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import jax
from jax import random
from functools import partial
from jax.example_libraries import optimizers
from scipy.optimize import nnls

jax.config.update("jax_platform_name", "cpu")


class UBVI:
    def __init__(
        self,
        target_log_pdf,
        component_dist,
        n_samples=100,
        n_logfg_samples=100,
        **kwargs
    ):

        self.target_log_pdf = target_log_pdf
        self.n_samples = n_samples
        self.n_logfg_samples = n_logfg_samples
        self.Z = jnp.empty((0, 0))
        self._logfg = jnp.empty(0)
        self._logfgsum = -jnp.inf

    def _compute_weights(self):
        Znew = jnp.exp(
            self.component_dist.log_sqrt_pair_integral(self.params[-1, :], self.params)
        )

        Zold = self.Z
        self.Z = jnp.zeros((self.params.shape[0], self.params.shape[0]))
        self.Z[:-1, :-1] = Zold
        self.Z[-1, :] = Znew
        self.Z[:, -1] = Znew

        logfg_old = self._logfg
        self._logfg = jnp.zeros(self.params.shape[0])

        self._logfg[:-1] = logfg_old
        self._logfg[-1] = self._logfg_est(self.params[-1, :])

        if self.params.shape[0] == 1:
            w = jnp.array([1.0])
        else:
            print(self.Z)

            Linv = jnp.invert(jnp.linalg.cholesky(self.Z))

            d = jnp.exp(self._logfg - self._logfg.max())
            b = nnls(np.array(Linv), -np.einsum("ij,j->i", Linc, d))[0]

            lbd = np.einsum("ij,j->i", Linv, b + d)
            w = np.max(
                np.zeros(1),
                np.einsum("ij,j->i", Linv.T, lbd / np.sqrt(((lbd**2).sum()))),
            )

        self._logfgsum = np.logsumexp(
            np.concatenate(
                (-np.array(np.inf)[None], self._logfg[w > 0] + torch.log(w[w > 0])),
                0,
            ),
            0,
        )
        return w

    def _hellsq_estimate(self):
        samples = self._sample_g(self.n_samples)
        lf = 0.5 * self.target_log_pdf(samples)
        lg = self._logg(samples)
        ln = torch.log(torch.tensor(self.n_samples, dtype=torch.float32))
        return 1.0 - torch.exp(
            torch.logsumexp(lf - lg - ln, 0)
            - 0.5 * torch.logsumexp(2 * lf - 2 * lg - ln, 0)
        )


if __name__ == "__main__":
    from distributions import Gaussian

    cauchy = lambda x: -jnp.log(1 + (x**2).sum(axis=-1))
    test = UBVI(
        cauchy,
        Gaussian(1),
        num_opt_steps=1000,
        n_samples=100,
        n_init=100,
        init_inflation=100,
        n_logfg_samples=100,
    )
