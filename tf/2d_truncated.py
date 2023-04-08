import time
import os
import json
import math
import numpy as np

import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
import tensorflow_probability as tfp
tfd = tfp.distributions

import temperflow.option as opts
opts.set_opts(jit=False, debug=False)
from temperflow.bijector import Spline2D
from temperflow.distribution import TransformedDistribution
from temperflow.flow import TemperFlow

from utils import get_distr, vis_energy, vis_trace, rej_sampling, vis_sampling_res

# Paths
cache_path = "cache"
model_path = "model"
cache_prefix = f"{cache_path}/2d-trunc"
model_prefix = f"{model_path}/2d-trunc"
os.makedirs(cache_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# The distribution to be sampled from
np.random.seed(123)
tf.random.set_seed(123)
dim = 2
energy = get_distr("circle")

# Visualize the energy function
bound = 5.0
vis_energy(energy, xlim=(-bound, bound), ylim=(-bound, bound))

# Base distribution
class Uniform2D(tfd.Distribution):
    def __init__(self, bound=1.0):
        super().__init__(
            dtype=None, reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=False, allow_nan_stats=True)
        self.dim = 2
        self.logden = -self.dim * math.log(2.0 * bound)
        self.impl = tfd.Independent(
            distribution=tfd.Uniform(low=tf.fill([self.dim], -bound),
                                     high=tf.fill([self.dim], bound)),
            reinterpreted_batch_ndims=1)

    def sample(self, *args, **kwargs):
        return self.impl.sample(*args, **kwargs)

    def log_prob(self, value, name="log_prob", **kwargs):
        return tf.fill([value.shape[0]], self.logden)



# Bijector
np.random.seed(123)
tf.random.set_seed(123)
bijector = Spline2D(count_bins=32, bound=bound, hlayers=[64, 64])
basedistr = Uniform2D(bound=bound)
bijdistr = TransformedDistribution(dim=dim, bijector=bijector, base_distr=basedistr)
temper = TemperFlow(energy, bijdistr)

t1 = time.time()
trace = temper.train_flow(beta0=0.1, kl_decay=0.5, nepoch=(1001, 1001, 0),
                          bs=2000, lr=0.001, max_betas=100, verbose=2,
                          checkpoint=cache_prefix, recover=False)
t2 = time.time()
print(f"{t2 - t1} seconds")

# Save model and traces
bijdistr.save_params(f"{model_prefix}-bijdistr.npz")
with open(f"{model_prefix}-trace.json", mode="w") as f:
    json.dump(trace, f, indent=2)

# Restore model and traces
bijdistr.load_params(f"{model_prefix}-bijdistr.npz")
with open(f"{model_prefix}-trace.json", mode="r") as f:
    trace = json.load(f)

# Visualize traces
vis_trace(trace)

# Rejection sampling
np.random.seed(123)
tf.random.set_seed(123)
xacc = rej_sampling(energy, bijdistr, nsamp=100000)

# Visualize sampling results
vis_sampling_res(bijdistr, trace, xacc, nsamp=5000,
                 xlim=(-bound, bound), ylim=(-bound, bound))

# Get sampled points at each stage
def get_sample(bijdistr, n, seed=None):
    return bijdistr.sample(n, seed=seed)

betas = [f"β={math.exp(lbeta):.3f}" for lbeta in trace["logbeta"]]
states = ["Init"] + betas + ["Final"]

np.random.seed(123)
tf.random.set_seed(123)
nsamp = 5000
seed = 123
bijector = Spline2D(count_bins=32, bound=bound, hlayers=[64, 64])
basedistr = Uniform2D(bound=bound)
bijdistr = TransformedDistribution(dim=dim, bijector=bijector, base_distr=basedistr)

# Init
samples = [get_sample(bijdistr, nsamp, seed=seed)]
# Betas
for i, _ in enumerate(betas):
    print(i)
    bijdistr.load_params(f"{cache_prefix}_{i}.npz")
    samples.append(get_sample(bijdistr, nsamp, seed=seed))
# Final
bijdistr.load_params(f"{model_prefix}-bijdistr.npz")
samples.append(get_sample(bijdistr, nsamp, seed=seed))

# Save data
samples = np.array(samples)
np.savez_compressed(f"{model_prefix}-sample.npz",
                    states=states, samples=samples)
