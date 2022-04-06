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