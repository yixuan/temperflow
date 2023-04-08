import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import temperflow.option as opts
from temperflow.flow import Energy

# Whether to JIT compile functions
JIT = opts.opts["jit"]

# A mixture of (standard) normal distributions that forms a circle
class GaussianCircle(Energy):
    def __init__(self, modes=8, radius=10.0, scale=1.0):
        locs = [[
            math.cos(2.0 * math.pi * i / modes) * radius,
            math.sin(2.0 * math.pi * i / modes) * radius
        ] for i in range(modes)]
        self.distr = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=tf.ones(modes) / modes),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=locs, scale_diag=tf.constant(scale, shape=(modes, 2))
            )
        )

    # Implement the energy function
    @tf.function(jit_compile=JIT)
    def energy(self, x):
        return -self.distr.log_prob(x)

    # Explicitly turn off JIT compilation, since the random seed
    # will be ignored by the XLA compiler
    @tf.function(jit_compile=False)
    def sample(self, n):
        return self.distr.sample(n)

# Unit test
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from density import GaussianCircle

    np.random.seed(123)
    tf.random.set_seed(123)
    fig = plt.figure(figsize=(10, 10))
    model = GaussianCircle(radius=10.0, scale=1.0)
    ngrid = 100
    z1 = np.linspace(-15.0, 15.0, num=ngrid)
    z2 = np.linspace(-15.0, 15.0, num=ngrid)
    zv1, zv2 = np.meshgrid(z1, z2)
    grids = np.hstack((zv1.reshape(-1, 1), zv2.reshape(-1, 1))).astype(np.float32)
    grids = tf.constant(grids)
    energy = model.energy(grids)
    plt.contourf(zv1, zv2, energy.numpy().reshape(ngrid, ngrid), levels=15)
    plt.colorbar()
    plt.show()

# A mixture of (standard) normal distributions that forms grids
class GaussianGrid(Energy):
    def __init__(self, num_on_edge=5, width=10.0, scale=1.0):
        modes = num_on_edge ** 2
        x = np.linspace(-width / 2.0, width / 2.0, num=num_on_edge)
        X, Y = np.meshgrid(x, x)
        locs = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1))).astype(np.float32)
        self.distr = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=tf.ones(modes) / modes),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=locs, scale_diag=tf.constant(scale, shape=(modes, 2))
            )
        )

    # Implement the energy function
    @tf.function(jit_compile=JIT)
    def energy(self, x):
        return -self.distr.log_prob(x)

    # Explicitly turn off JIT compilation, since the random seed
    # will be ignored by the XLA compiler
    @tf.function(jit_compile=False)
    def sample(self, n):
        return self.distr.sample(n)

# Unit test
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from density import GaussianGrid

    np.random.seed(123)
    tf.random.set_seed(123)
    fig = plt.figure(figsize=(10, 10))
    model = GaussianGrid(width=10.0, scale=0.3)
    ngrid = 100
    z1 = np.linspace(-8.0, 8.0, num=ngrid)
    z2 = np.linspace(-8.0, 8.0, num=ngrid)
    zv1, zv2 = np.meshgrid(z1, z2)
    grids = np.hstack((zv1.reshape(-1, 1), zv2.reshape(-1, 1))).astype(np.float32)
    grids = tf.constant(grids)
    energy = model.energy(grids)
    plt.contourf(zv1, zv2, energy.numpy().reshape(ngrid, ngrid), levels=15)
    plt.colorbar()
    plt.show()

# A mixture of two bivariate normals that form a cross
class GaussianCross(Energy):
    def __init__(self, scale=1.0, rho=0.9):
        modes = 8
        loc = [[3.0, 3.0], [-3.0, 3.0], [3.0, -3.0], [-3.0, -3.0]]
        L1 = scale * np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
        L2 = scale * np.linalg.cholesky([[1.0, -rho], [-rho, 1.0]])
        L = [L1, L2]
        locs = np.array([x for x in loc for y in L]).astype(np.float32)
        Ls = np.array([y for x in loc for y in L]).astype(np.float32)
        self.distr = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=tf.ones(modes) / modes),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=locs, scale_tril=Ls
            )
        )

    # Implement the energy function
    @tf.function(jit_compile=JIT)
    def energy(self, x):
        return -self.distr.log_prob(x)

    # Explicitly turn off JIT compilation, since the random seed
    # will be ignored by the XLA compiler
    @tf.function(jit_compile=False)
    def sample(self, n):
        return self.distr.sample(n)

# Unit test
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from density import GaussianCross

    np.random.seed(123)
    tf.random.set_seed(123)
    fig = plt.figure(figsize=(10, 10))
    model = GaussianCross(scale=1.0, rho=0.9)
    ngrid = 100
    z1 = np.linspace(-8.0, 8.0, num=ngrid)
    z2 = np.linspace(-8.0, 8.0, num=ngrid)
    zv1, zv2 = np.meshgrid(z1, z2)
    grids = np.hstack((zv1.reshape(-1, 1), zv2.reshape(-1, 1))).astype(np.float32)
    grids = tf.constant(grids)
    energy = model.energy(grids)
    plt.contourf(zv1, zv2, energy.numpy().reshape(ngrid, ngrid), levels=15)
    plt.colorbar()
    plt.show()

# Mixture normal marginals with Clayton copula
class ClaytonNormalMix(Energy):
    # The generator, "phi" function
    # phi(u) = u^(-theta) - 1 = exp(-theta * logu) - 1
    #
    # def phi(self, u):
    #     return tf.math.pow(u, -self.theta) - 1.0
    #
    def log_phi(self, logu):
        # Since 0 < u < 1, we have logu < 0, and hence phi(u) > 0
        # If u is close to 0, z = -theta * log(u) >> 0
        #     log[phi(u)] = log(exp(z) - 1) = z + log(1 - exp(-z))  #1
        # If u is close to 1, v = 1 - u, w = log(v)
        #     u^(-theta) = (1 - v)^(-t) = 1 + t * v + r(v)
        #     phi(u) = t * exp(w) + r(v)
        #     log[phi(u)] = log(t) + w + log(1 + r(v) / (t * v))    #2
        #     r(v) / (t * v) = c1 * v + c2 * v^2 + O(v^3)
        #     c1 = (t + 1) / 2, c2 = (t + 1)(t + 2) / 6
        #
        # Algorithm #1 fails when u = 1
        # Algorithm #2 is accurate when 1 - u -> 0, i.e., w << 0
        # Set cutoff w0 = -15, v0 = 1 - u0 ~ 3e-7
        # When logu > log(u0), use Algorithm #2
        logu0 = math.log1p(-math.exp(-15.0))
        safe_logu = tf.math.minimum(logu, 0.5 * logu0)
        z = -self.theta * safe_logu
        alg1 = z + tfp.math.log1mexp(z)
        return alg1
        # v = tf.math.exp(log1u)
        # c1 = 0.5 * (self.theta + 1.0)
        # c2 = (self.theta + 1.0) * (self.theta + 2.0) / 6.0
        # r = c1 * v + c2 * tf.math.square(v)
        # alg2 = math.log(self.theta) + log1u + tf.math.log1p(r)
        # return tf.where(logu > logu0, alg2, alg1)

    # Inverse generator, "psi" function
    def psi(self, x):
        return tf.pow(x + 1.0, -1.0 / self.theta)

    # log|psi^(d) (x)| + C, psi^(d) - d-th derivative of psi
    #
    # def log_psi_d(self, x):
    #     return -(self.dim + 1.0 / self.theta) * tf.math.log1p(x)
    #
    def log_psi_d(self, logx):
        # z = log(x), x > 0
        # log(1 + x) = log(1 + exp(z)) = softplus(z)
        log1px = tf.math.softplus(logx)
        return -(self.dim + 1.0 / self.theta) * log1px

    # See Proposition 4.2 of https://arxiv.org/pdf/0908.3750.pdf
    #
    # def log_dcopula(self, u):
    #     u = 1e-8 + (1.0 - 2e-8) * u
    #     numer = self.log_psi_d(tf.math.reduce_sum(self.phi(u), axis=-1))
    #     denom = (self.theta + 1.0) * tf.math.reduce_sum(tf.math.log(u), axis=-1)
    #     return numer - denom
    #
    def log_dcopula(self, logu):
        log_phi = self.log_phi(logu)
        log_phi_sum = tf.math.reduce_logsumexp(log_phi, axis=-1)
        numer = self.log_psi_d(log_phi_sum)
        denom = (self.theta + 1.0) * tf.math.reduce_sum(logu, axis=-1)
        return numer - denom

    # Marginal
    # First mix_dim dimensions are mixture distributions,
    # and the rest are unimodal distributions
    #
    # def mcdf(self, x):
    #     if self.mix_dim == self.dim:
    #         return self.marginal_mix.cdf(x)
    #     cdf_mix = self.marginal_mix.cdf(x[..., :self.mix_dim])
    #     cdf_uni = self.marginal_uni.cdf(x[..., self.mix_dim:])
    #     return tf.concat((cdf_mix, cdf_uni), axis=-1)

    def mlog_cdf(self, x):
        if self.mix_dim == self.dim:
            return self.marginal_mix.log_cdf(x)
        logcdf_mix = self.marginal_mix.log_cdf(x[..., :self.mix_dim])
        logcdf_uni = self.marginal_uni.log_cdf(x[..., self.mix_dim:])
        return tf.concat((logcdf_mix, logcdf_uni), axis=-1)

    def mlog_pdf(self, x):
        if self.mix_dim == self.dim:
            return self.marginal_mix.log_prob(x)
        logpdf_mix = self.marginal_mix.log_prob(x[..., :self.mix_dim])
        logpdf_uni = self.marginal_uni.log_prob(x[..., self.mix_dim:])
        return tf.concat((logpdf_mix, logpdf_uni), axis=-1)

    # Not yet implemented!
    def micdf(self, u):
        if self.mix_dim == self.dim:
            return self.marginal_mix.quantile(u)
        icdf_mix = self.marginal_mix.quantile(u[..., :self.mix_dim])
        icdf_uni = self.marginal_uni.quantile(u[..., self.mix_dim:])
        return tf.concat((icdf_mix, icdf_uni), axis=-1)

    def __init__(self, dim=2, mix_dim=None, theta=2.0):
        super().__init__()

        self.dim = dim
        self.mix_dim = dim if mix_dim is None else mix_dim
        self.theta = theta
        self.gamma = tfd.Gamma(
            concentration=1.0 / theta, rate=1.0)
        self.marginal_mix = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.7, 0.3]),
            components_distribution=tfd.Normal(loc=[-1.0, 1.0], scale=[0.2, 0.2])
        )
        self.marginal_uni = tfd.Normal(loc=[0.0], scale=[0.5])

    # Implement the energy function
    def log_prob(self, x):
        # return self.log_dcopula(self.mcdf(x)) + tf.math.reduce_sum(self.mlog_pdf(x), axis=-1)
        logu = self.mlog_cdf(x)
        # We have taken care of u that is close to 0,
        # but we also need to deal with u that is close to 1
        logu = tf.math.minimum(logu, -1e-6)
        return self.log_dcopula(logu) + tf.math.reduce_sum(self.mlog_pdf(x), axis=-1)

    # Implement the energy function
    @tf.function(jit_compile=JIT)
    def energy(self, x):
        return -self.log_prob(x)

    # See Algorithm 6.3 of
    # https://www.uio.no/studier/emner/matnat/math/STK4520/h10/undervisningsmateriale/copula.pdf
    def sample_clayton(self, n):
        z = self.gamma.sample(n)
        z = tf.tile(z[..., None], (1, self.dim))
        v = tf.random.uniform(shape=(n, self.dim))
        return self.psi(-tf.math.log(v) / z)

    @tf.function(jit_compile=False)
    def sample(self, n):
        u = self.sample_clayton(n)
        return self.micdf(u)

# Unit test
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from density import ClaytonNormalMix

    model = ClaytonNormalMix(dim=2)
    ngrid = 100
    z1 = np.linspace(-3.0, 3.0, num=ngrid)
    z2 = np.linspace(-3.0, 3.0, num=ngrid)
    zv1, zv2 = np.meshgrid(z1, z2)
    grids = np.hstack((zv1.reshape(-1, 1), zv2.reshape(-1, 1))).astype(np.float32)
    grids = tf.constant(grids)
    energy = model.energy(grids)
    plt.contourf(zv1, zv2, energy.numpy().reshape(ngrid, ngrid), levels=15)
    plt.colorbar()
    plt.show()

    # Test numerical stability
    p = 8
    model = ClaytonNormalMix(dim=p)
    z = np.linspace(-6.0, 6.0, num=300, dtype=np.float32)
    zv = z.reshape(-1, 1).repeat(p, axis=1)
    e = model.energy(tf.constant(zv))
    plt.plot(z, e.numpy())
    plt.show()
