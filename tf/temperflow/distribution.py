import os
import tempfile
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import temperflow.option as opts

# Whether to JIT compile functions
JIT = opts.opts["jit"]
JIT_DEBUG = opts.opts["debug"]

class TransformedDistribution(tf.Module):
    def __init__(self, dim, bijector, base_distr=None):
        super().__init__()
        self.dim = dim
        self.base = base_distr
        if base_distr is None:
            self.base = tfd.Independent(
                distribution=tfd.Normal(loc=tf.zeros(dim), scale=tf.ones(dim)),
                reinterpreted_batch_ndims=1)
        self.bijector = bijector

    # When JIT compilation is enabled, seed needs to be a [2] Tensor
    # to make the result reproducible
    # Therefore, when this function is used in a higher-level
    # function that is JIT compiled (e.g. TemperFlow), be sure to
    # pass in a Tensor for seed. A typical choice is
    #     seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
    # Here we turn off JIT compilation, so that when this function
    # is directly used (such as generating samples), we can simply
    # let seed=None
    @tf.function(jit_compile=False)
    def sample(self, n, log_pdf=False, seed=None):
        print(f"***** tracing TransformedDistribution.sample(log_pdf={log_pdf}) *****") if JIT_DEBUG else None
        z = self.base.sample(n, seed=seed)
        x, logdet = self.bijector(z)
        if not log_pdf:
            return x
        logp = self.base.log_prob(z) - logdet
        return x, logp

    @tf.function(jit_compile=JIT)
    def log_prob(self, x):
        print("***** tracing TransformedDistribution.log_prob() *****") if JIT_DEBUG else None
        z, logdet = self.bijector.inverse(x)
        logp = self.base.log_prob(z) + logdet
        return logp

    def copy(self):
        return TransformedDistribution(
            self.dim, self.bijector.copy(), self.base.copy())

    def serialized_copy(self):
        # Save current model to a temporary folder
        temp_dir = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        tf.saved_model.save(self, temp_dir)
        # Load the serialized model
        new_model = tf.saved_model.load(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir)
        return new_model

    def copy_params_from(self, other):
        # This function does not copy the base distribution
        if self.dim != other.dim:
            raise ValueError("dimensions do not match")
        if type(self.bijector) != type(other.bijector):
            raise ValueError("types of bijectors do not match")
        if len(self.bijector.variables) != len(other.bijector.variables):
            raise ValueError("parameter lengths do not match")
        for self_var, other_var in zip(self.bijector.variables, other.bijector.variables):
            self_var.assign(other_var.value())

    def save_params(self, path):
        # https://stackoverflow.com/a/62806106
        params = [var.numpy() for var in self.bijector.variables]
        np.savez_compressed(path, *params)
        print(f"parameters saved to {path}")

    def load_params(self, path):
        # Load parameters
        data = np.load(path)
        params = [data[k] for k in data]
        if len(self.bijector.variables) != len(params):
            raise ValueError("parameter lengths do not match")
        for var, param in zip(self.bijector.variables, params):
            var.assign(tf.constant(param))
        print(f"loaded parameters from {path}")

# class Test:
#     def __init__(self):
#         self.var = tf.Variable(tf.constant(1.0))
#
#     @tf.function(jit_compile=True)
#     def retvar(self):
#         return self.var
#
#     def reassign(self):
#         self.var.assign_add(tf.constant(1.0))
#
#     def modify(self):
#         self.var = tf.Variable(tf.constant(-1.0))
#
# test = Test()
# print(test.retvar())
# test.reassign()
# print(test.retvar())
# test.modify()
# print(test.retvar())

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfb = tfp.bijectors
    tfd = tfp.distributions
    import temperflow.option as opts

    opts.set_opts(jit=True)
    from temperflow.bijector import TFBWrapper
    from temperflow.distribution import TransformedDistribution

    # tfp.distributions
    tf.random.set_seed(123)
    base = tfd.Independent(
      distribution=tfd.Normal(loc=[-1.0, 1.0], scale=[0.1, 0.5]),
      reinterpreted_batch_ndims=1)
    print("z =\n", base.sample(3))
    bij = tfb.Exp()
    trans = tfd.TransformedDistribution(base, bij)
    x = trans.sample(3)
    logp = trans.log_prob(x)
    print("x =\n", x)
    print("logp =\n", logp)
    x2 = tf.random.uniform((3, 2), minval=0.0, maxval=5.0)
    logp2 = trans.log_prob(x2)
    print("logp2 =\n", logp2)

    # Ours
    tf.random.set_seed(123)
    print("z =\n", base.sample(3))
    bij = TFBWrapper(tfb.Exp())
    trans = TransformedDistribution(dim=2, bijector=bij, base_distr=base)
    x, logp = trans.sample(3, log_pdf=True)
    print("x =\n", x)
    print("logp =\n", logp)
    logp2 = trans.log_prob(x2)
    print("logp2 =\n", logp2)

    # Serialized copy
    tf.random.set_seed(123)
    trans_s = trans.serialized_copy()
    x, logp = trans_s.sample(3, log_pdf=True)
    print("x =\n", x)
    print("logp =\n", logp)
    logp2 = trans_s.log_prob(x2)
    print("logp2 =\n", logp2)

    # Copy
    tf.random.set_seed(123)
    trans_c = trans.copy()
    x, logp = trans_c.sample(3, log_pdf=True)
    print("x =\n", x)
    print("logp =\n", logp)
    logp2 = trans_c.log_prob(x2)
    print("logp2 =\n", logp2)

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.bijector import AutoRegSpline
    from temperflow.distribution import TransformedDistribution

    tf.random.set_seed(123)
    dim = 5
    bij = AutoRegSpline(dim=dim, perm_dim=2)
    trans = TransformedDistribution(dim=dim, bijector=bij)
    seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
    x, logp = trans.sample(3, log_pdf=True, seed=seed)
    print("x =\n", x)
    print()
    print("logp =\n", logp)
    print()
    logp2 = trans.log_prob(tf.identity(x))
    print("logp2 =\n", logp2)

    # Copy
    trans_c = trans.copy()
    x, logp = trans_c.sample(3, log_pdf=True, seed=seed)
    print("x =\n", x)
    print()
    print("logp =\n", logp)
    print()
    logp2 = trans_c.log_prob(tf.identity(x))
    print("logp2 =\n", logp2)

    # Copy params
    bij2 = AutoRegSpline(dim=dim, perm_dim=2)
    trans_c2 = TransformedDistribution(dim=dim, bijector=bij2)
    trans_c2.copy_params_from(trans)
    # Test results
    x, logp = trans_c2.sample(3, log_pdf=True, seed=seed)
    print("x =\n", x)
    print()
    print("logp =\n", logp)
    print()
    logp2 = trans_c2.log_prob(tf.identity(x))
    print("logp2 =\n", logp2)

# Demo
if __name__ == "__main__":
    import math
    import time
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    from temperflow.bijector import AutoRegSpline, Inverse
    from temperflow.distribution import TransformedDistribution
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Data generator
    def distr_circle(modes=8, radius=10.0, scale=1.0):
        locs = [[
            math.cos(2.0 * math.pi * i / modes) * radius,
            math.sin(2.0 * math.pi * i / modes) * radius
        ] for i in range(modes)]
        distr = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=tf.ones(modes) / modes),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=locs, scale_diag=tf.constant(scale, shape=(modes, 2))
            )
        )
        return distr

    # Observed data
    tf.random.set_seed(123)
    circle = distr_circle(modes=8, radius=5.0, scale=0.5)
    dat = circle.sample(1000)
    dat = pd.DataFrame(dat.numpy(), columns=["x1", "x2"])
    fig = plt.figure()
    sns.scatterplot(data=dat, x="x1", y="x2")
    plt.show()

    # Distribution
    tf.random.set_seed(123)
    bij = AutoRegSpline(dim=2, count_bins=32, bound=8.0)
    # Inverse should be faster in training
    # bij = Inverse(bij)
    distr = TransformedDistribution(dim=2, bijector=bij)
    opt = tf.keras.optimizers.Adam()

    # Training functions
    @tf.function(jit_compile=True)
    def lossfn(x):
        return -tf.reduce_mean(distr.log_prob(x))

    # Problematic with jit_compile=True, not sure why
    @tf.function
    def train_model_step(bs=1000):
        x = circle.sample(bs)
        with tf.GradientTape() as tape:
            loss = lossfn(x)
        params = distr.trainable_variables
        grad = tape.gradient(loss, params)
        opt.apply_gradients(zip(grad, params))
        return loss

    def train_model(nepoch=1000, bs=1000):
        for i in range(nepoch):
            loss = train_model_step(bs=bs)
            if i % 100 == 0:
                print(f"epoch {i}, loss = {loss.numpy()}")

    # Training process
    tf.random.set_seed(123)
    # Warmup stage for JIT compiling
    t1 = time.time()
    train_model(nepoch=2, bs=1000)
    t2 = time.time()
    print(f"warmup finished in {t2 - t1} seconds")
    t2 = time.time()
    # Formal training
    train_model(nepoch=2000, bs=1000)
    t3 = time.time()
    print(f"training finished in {t3 - t2} seconds")

    # Visualize fitted result
    tf.random.set_seed(123)
    dat = distr.sample(1000)
    dat = pd.DataFrame(dat.numpy(), columns=["x1", "x2"])
    fig = plt.figure()
    sns.scatterplot(data=dat, x="x1", y="x2")
    plt.show()

# Benchmarking
if __name__ == "__main__":
    import time
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    from temperflow.bijector import AutoRegSpline, Inverse
    from temperflow.distribution import TransformedDistribution

    # Distribution
    tf.random.set_seed(123)
    dim = 32
    bs = 2000
    nrep = 1000
    base = tfd.Independent(
        distribution=tfd.Normal(loc=tf.zeros(dim), scale=tf.ones(dim)),
        reinterpreted_batch_ndims=1)
    bij = AutoRegSpline(dim=dim, perm_dim=2, count_bins=64,
                        bound=8.0, hlayers=[64, 64])
    # bij = Inverse(bij)
    distr = TransformedDistribution(dim, bij, base)

    @tf.function(jit_compile=True)
    def test_fun(distr, bs, seed=None):
        x, logp = distr.sample(bs, log_pdf=True, seed=seed)
        x2 = tf.identity(x)
        logp2 = distr.log_prob(x2)
        return x, logp, logp2

    tf.random.set_seed(123)
    seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
    t1 = time.time()
    _ = test_fun(distr, bs, seed=seed)
    t2 = time.time()
    print(f"warmup finished in {t2 - t1} seconds")
    t1 = time.time()
    for i in range(nrep):
        _ = test_fun(distr, bs, seed=seed)
    t2 = time.time()
    print(f"finished in {t2 - t1} seconds")
    print(_[0])
