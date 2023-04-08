################################################################
# This section is intended to be run in interactive mode
dim = 8
alpha = 0.7
spline = "linear"
exp_sampling = True
exp_benchmark = True
################################################################

################################################################
# This section is intended to be run in script mode
import argparse

parser = argparse.ArgumentParser(description="Clayton copula experiments")
parser.add_argument(
    "--dim", default=8, type=int, dest="dim")
parser.add_argument(
    "--alpha", default=0.7, type=float, dest="alpha")
parser.add_argument(
    "--spline", default="linear", type=str,
    choices=["linear", "quadratic"], dest="spline")
parser.add_argument(
    "--exp_sampling", default=False, dest="exp_sampling",
    action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--exp_benchmark", default=False, dest="exp_benchmark",
    action=argparse.BooleanOptionalAction)

args = parser.parse_args()
dim = args.dim
alpha = args.alpha
spline = args.spline
exp_sampling = args.exp_sampling
exp_benchmark = args.exp_benchmark
################################################################

import time
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
import tensorflow_probability as tfp
tfd = tfp.distributions

import temperflow.option as opts
opts.set_opts(jit=True, debug=False)
from temperflow.bijector import AutoRegSpline, BlockAutoRegSpline
from temperflow.distribution import TransformedDistribution
from temperflow.flow import TemperFlow

from utils import get_distr, cond_run_experiment, vis_trace, rej_sampling

# Paths
cache_path = "cache"
model_path = "model"
cache_prefix = f"{cache_path}/clayton"
model_prefix = f"{model_path}/clayton-dim{dim}-{spline}-alpha{alpha:.1f}"
os.makedirs(cache_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# The distribution to be sampled from
np.random.seed(123)
tf.random.set_seed(123)
energy = get_distr("clayton", dim=dim)

# Visualization of marginal distributions
def vis_marginal(samp):
    gdat = pd.DataFrame(samp, columns=[f"x{i + 1}" for i in range(dim)])
    fig = plt.figure(figsize=(12, 6))
    # True density curve
    x0 = np.linspace(-3.0, 3.0, num=300, dtype=np.float32)
    y0 = np.exp(energy.marginal_mix.log_prob(tf.constant(x0)).numpy())
    mix = pd.DataFrame(np.stack((x0, y0), axis=1), columns=["x", "density"])
    sns.lineplot(data=mix, x="x", y="density", label="True density", color="lightgreen", linewidth=3)
    for i in range(dim):
        if i < 10:
            sns.kdeplot(data=gdat, x=f"x{i + 1}", label=f"x{i + 1}")
        else:
            sns.kdeplot(data=gdat, x=f"x{i + 1}")
    plt.legend()
    plt.show()

# Pairwise scatterplots of latent variables
def vis_pair(samp, nvars):
    gdat = pd.DataFrame(samp, columns=[f"x{i + 1}" for i in range(dim)])
    if nvars > 0:
        g = sns.pairplot(gdat.iloc[:, :nvars], plot_kws={"alpha": 0.2, "s": 10})
    else:
        g = sns.pairplot(gdat.iloc[:, nvars:], plot_kws={"alpha": 0.2, "s": 10})
    g.set(xlim=(-3, 3), ylim=(-3, 3))
    plt.show()



# Bijector
np.random.seed(123)
tf.random.set_seed(123)
if dim < 16:
    bijector = AutoRegSpline(
        dim=dim, perm_dim=2, count_bins=32, bound=6.0, hlayers=[64, 64], spline=spline)
else:
    bijector = BlockAutoRegSpline(
        dim=dim, block_size=8, count_bins=32, bound=6.0, hlayers=[128, 128], spline=spline)
bijdistr = TransformedDistribution(dim=dim, bijector=bijector)
temper = TemperFlow(energy, bijdistr)

# Run the main sampling experiment
def run_exp_sampling():
    # Compiling model and warmup
    t1 = time.time()
    temper.train_one_beta(beta=0.1, fn="kl", bs=2000, nepoch=11, lr=0.0001, verbose=True)
    temper.update_ref()
    temper.train_one_beta(beta=0.1, fn="l2", ref=True, bs=2000, nepoch=11, lr=0.0001, verbose=True)
    t2 = time.time()
    print(f"{t2 - t1} seconds")

    t1 = time.time()
    trace = temper.train_flow(beta0=0.1, kl_decay=alpha, nepoch=(1001, 1001, 101),
                              bs=2000, lr=0.001, max_betas=100, verbose=2,
                              checkpoint=cache_prefix, recover=False)
    t2 = time.time()
    print(f"{t2 - t1} seconds")

    # Save model and traces
    bijdistr.save_params(f"{model_prefix}-bijdistr.npz")
    with open(f"{model_prefix}-losses.json", mode="w") as f:
        json.dump(trace, f, indent=2)

cond_run_experiment(
    "Main sampling experiment", exp_sampling, run_exp_sampling)

# Restore model and traces
bijdistr.load_params(f"{model_prefix}-bijdistr.npz")
with open(f"{model_prefix}-losses.json", mode="r") as f:
    trace = json.load(f)

# Visualize traces
vis_trace(trace)

# Marginal densities
np.random.seed(123)
tf.random.set_seed(123)
nsamp = 5000
seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
xsamp = bijdistr.sample(nsamp, seed=seed).numpy()
vis_marginal(xsamp)

# Pairwise scatterplots
vis_pair(xsamp, nvars=8)
vis_pair(xsamp, nvars=-8)

# Rejection sampling
np.random.seed(123)
tf.random.set_seed(123)
xacc = rej_sampling(energy, bijdistr, nsamp=100000, maxit=200)

# Visualization of marginal distributions
vis_marginal(xacc[:nsamp, :])

# Pairwise scatterplots
vis_pair(xacc[:nsamp, :], nvars=8)
vis_pair(xacc[:nsamp, :], nvars=-8)

# Save data
np.random.seed(123)
tf.random.set_seed(123)
nrep = 100
nsamp = 1000
xest = bijdistr.sample(nrep * nsamp).numpy().reshape(nrep, nsamp, dim)
xacc = xacc[:(nrep * nsamp)].reshape(nrep, nsamp, dim)
np.savez_compressed(f"{model_prefix}.npz", xest=xest, xacc=xacc)



# Timing benchmark
np.random.seed(123)
tf.random.set_seed(123)
nsamp = 10000
nrep = 30
nwarmup = 10
benchmark = dict()

# Run benchmarking
def run_exp_benchmark():
    time_tf = []
    for i in range(nrep + nwarmup):
        t1 = time.time()
        _ = bijdistr.sample(nsamp)
        # Force synchronization
        sync = _[0].numpy()
        t2 = time.time()
        # The first few iterations are warm-ups
        if i >= nwarmup:
            time_tf.append(t2 - t1)
    print(np.median(time_tf))
    benchmark["temperflow"] = time_tf

    time_rej = []
    for i in range(nrep + nwarmup):
        t1 = time.time()
        _ = rej_sampling(energy, bijdistr, nsamp=nsamp, verbose=False)
        t2 = time.time()
        # The first few iterations are warm-ups
        if i >= nwarmup:
            time_rej.append(t2 - t1)
    print(np.median(time_rej))
    benchmark["temperflow_rej"] = time_rej

    # Save benchmarking results
    with open(f"{model_prefix}-benchmark.json", mode="w") as f:
        json.dump(benchmark, f, indent=2)
    print("finished")

cond_run_experiment(
    "Benchmarks", exp_benchmark, run_exp_benchmark)
