################################################################
# This section is intended to be run in interactive mode
data = "circle"
bijname = "affine"
exp_sampling = True
################################################################

################################################################
# This section is intended to be run in script mode
# In script mode, save plot as PDF
import matplotlib
matplotlib.use("PDF")
import argparse

parser = argparse.ArgumentParser(description="2D experiments")
parser.add_argument(
    "--data", default="circle", type=str,
    choices=["circle", "grid", "cross"], dest="data")
parser.add_argument(
    "--bijector", default="spline", type=str,
    choices=["affine", "lrspline", "qrspline"], dest="bijname")
parser.add_argument(
    "--exp_sampling", default=False, dest="exp_sampling",
    action=argparse.BooleanOptionalAction)

args = parser.parse_args()
data = args.data
bijname = args.bijname
exp_sampling = args.exp_sampling
################################################################

import time
import os
import json
import numpy as np
import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
import tensorflow_probability as tfp
tfd = tfp.distributions

import temperflow.option as opts
opts.set_opts(jit=True, debug=False)
from temperflow.bijector import Spline2D, AutoRegAffine
from temperflow.distribution import TransformedDistribution
from temperflow.flow import TemperFlow

from utils import get_distr, cond_run_experiment, vis_energy, vis_trace, \
    rej_sampling, vis_sampling_res

# Paths
cache_path = "cache"
model_path = "model"
cache_prefix = f"{cache_path}/{data}-{bijname}"
model_prefix = f"{model_path}/{data}-{bijname}"
os.makedirs(cache_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# The distribution to be sampled from
np.random.seed(123)
tf.random.set_seed(123)
energy = get_distr(data)

# Visualize the energy function
vis_energy(energy)

# Bijector
np.random.seed(123)
tf.random.set_seed(123)
if bijname == "affine":
    bijector = AutoRegAffine(dim=2, depth=16, hlayers=[64, 64])
    lr = 0.0001  # Large learning rates generate NaN losses
elif bijname == "lrspline":
    bijector = Spline2D(count_bins=32, bound=None, hlayers=[64, 64])
    lr = 0.001
else:
    bijector = Spline2D(count_bins=32, bound=None, hlayers=[64, 64], spline="quadratic")
    lr = 0.001
bijdistr = TransformedDistribution(dim=2, bijector=bijector)
temper = TemperFlow(energy, bijdistr)

# Run the main sampling experiment
def run_exp_sampling():
    t1 = time.time()
    trace = temper.train_flow(beta0=0.1, kl_decay=0.6, nepoch=(1001, 1001, 0),
                              bs=2000, lr=lr, max_betas=100, verbose=2,
                              checkpoint=cache_prefix, recover=False)
    t2 = time.time()
    print(f"{t2 - t1} seconds")

    # Save model and traces
    bijdistr.save_params(f"{model_prefix}-bijdistr.npz")
    with open(f"{model_prefix}-trace.json", mode="w") as f:
        json.dump(trace, f, indent=2)

cond_run_experiment(
    "Main sampling experiment", exp_sampling, run_exp_sampling)

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
vis_sampling_res(bijdistr, trace, xacc, nsamp=5000)

# Save data
np.random.seed(123)
tf.random.set_seed(123)
xsamp = bijdistr.sample(n=5000, seed=123)
np.savez_compressed(f"{model_prefix}-sample.npz",
                    sample=xsamp.numpy())
