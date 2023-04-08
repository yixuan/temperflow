################################################################
# This section is intended to be run in interactive mode
dim = 8
mcmc_thinning = 1
exp_sampling = True
exp_mcmc = True
exp_benchmark = True
################################################################

################################################################
# This section is intended to be run in script mode
import argparse

parser = argparse.ArgumentParser(description="Clayton copula experiments")
parser.add_argument(
    "--dim", default=8, type=int, dest="dim")
parser.add_argument(
    "--mcmc_thinning", default=1, type=int, dest="mcmc_thinning")
parser.add_argument(
    "--exp_sampling", default=False, dest="exp_sampling",
    action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--exp_mcmc", default=False, dest="exp_mcmc",
    action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--exp_benchmark", default=False, dest="exp_benchmark",
    action=argparse.BooleanOptionalAction)

args = parser.parse_args()
dim = args.dim
mcmc_thinning = args.mcmc_thinning
exp_sampling = args.exp_sampling
exp_mcmc = args.exp_mcmc
exp_benchmark = args.exp_benchmark
################################################################

import time
import os
import json
import math
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
model_prefix = f"{model_path}/clayton-dim{dim}"
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
        dim=dim, perm_dim=2, count_bins=32, bound=6.0, hlayers=[64, 64])
else:
    bijector = BlockAutoRegSpline(
        dim=dim, block_size=8, count_bins=32, bound=6.0, hlayers=[128, 128])
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

    # t1 = time.time()
    # temper.train_one_beta(beta=0.01, fn="kl", bs=2000, nepoch=2001, lr=0.001, verbose=True)
    # t2 = time.time()
    # print(f"{t2 - t1} seconds")

    t1 = time.time()
    trace = temper.train_flow(beta0=0.1, kl_decay=0.7, nepoch=(1001, 1001, 101),
                              bs=2000, lr=0.001, max_betas=100, verbose=2,
                              checkpoint=f"{cache_path}/clayton", recover=False)
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
xacc = rej_sampling(energy, bijdistr, nsamp=100000)

# Visualization of marginal distributions
vis_marginal(xacc[:nsamp, :])

# Pairwise scatterplots
vis_pair(xacc[:nsamp, :], nvars=8)
vis_pair(xacc[:nsamp, :], nvars=-8)



# MCMC
np.random.seed(123)
tf.random.set_seed(123)
num_results = 1000
num_chains = 100

# Kernel for M-H
kernel_mh = tfp.mcmc.RandomWalkMetropolis(
    target_log_prob_fn=energy.log_prob,
    new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=0.1)
)
# Kernel for HMC
kernel_hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=energy.log_prob,
    step_size=0.1,
    num_leapfrog_steps=5
)
# Kernel for parallel tempering
inv_temp = tf.exp(tf.linspace(0.0, math.log(0.1), num=5))
step_size = 0.1 / tf.sqrt(inv_temp)
step_size = tf.tile(tf.reshape(step_size, shape=(5, 1)), multiples=(1, num_chains))
step_size = tf.reshape(step_size, shape=(5, num_chains, 1))
kernel_pt = tfp.mcmc.ReplicaExchangeMC(
    target_log_prob_fn=energy.log_prob,
    inverse_temperatures=inv_temp,
    make_kernel_fn=lambda target_log_prob_fn: tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size))
)
# Initial state of the chain
init_state = tf.random.normal(shape=(num_chains, dim))

@tf.function
def run_mcmc(kernel, num_results, init_state, trace=False):
    trace_fn = (lambda current_state, kernel_results: kernel_results) if trace else None
    res = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=kernel,
        num_burnin_steps=200,
        num_steps_between_results=mcmc_thinning,  # Thinning
        trace_fn=trace_fn,
        seed=123)
    sample, trace_res = res if trace else (res, None)
    return sample, trace_res

# Run the MCMC experiment
def run_exp_mcmc():
    t1 = time.time()
    mh, trace_mh = run_mcmc(kernel_mh, num_results, init_state, trace=True)
    t2 = time.time()
    mh = tf.transpose(mh, perm=[1, 0, 2])
    print(f"{t2 - t1} seconds")
    print(mh.shape)
    print(tf.reduce_mean(tf.cast(trace_mh.is_accepted, dtype=tf.float32)))

    # vis_marginal(mh[1])
    # vis_marginal(mh[:, 1, :])

    t1 = time.time()
    hmc, trace_hmc = run_mcmc(kernel_hmc, num_results, init_state, trace=True)
    t2 = time.time()
    hmc = tf.transpose(hmc, perm=[1, 0, 2])
    print(f"{t2 - t1} seconds")
    print(hmc.shape)
    print(tf.reduce_mean(tf.cast(trace_hmc.is_accepted, dtype=tf.float32)))

    # vis_marginal(hmc[1])
    # vis_marginal(hmc[:, 1, :])

    t1 = time.time()
    pt, trace_pt = run_mcmc(kernel_pt, num_results, init_state, trace=True)
    t2 = time.time()
    pt = tf.transpose(pt, perm=[1, 0, 2])
    print(f"{t2 - t1} seconds")
    print(pt.shape)

    # vis_marginal(pt[1])
    # vis_pair(pt[1], nvars=8)
    # vis_pair(pt[1], nvars=-8)

    # vis_marginal(pt[:, 1, :])
    # vis_pair(pt[:, 1, :], nvars=8)
    # vis_pair(pt[:, 1, :], nvars=-8)

    # Save data
    np.random.seed(123)
    tf.random.set_seed(123)
    nrep = num_chains
    nsamp = num_results
    bijdistr.load_params(f"{model_prefix}-bijdistr.npz")
    xest = bijdistr.sample(nrep * nsamp).numpy().reshape(nrep, nsamp, dim)
    xacc2 = xacc[:(nrep * nsamp)].reshape(nrep, nsamp, dim)
    filename = f"{model_prefix}.npz" if mcmc_thinning == 1 else f"{model_prefix}-mcmc{mcmc_thinning}.npz"
    np.savez_compressed(filename,
        xest=xest, xacc=xacc2, mh=mh.numpy(), hmc=hmc.numpy(), pt=pt.numpy())

cond_run_experiment(
    "MCMC experiment", exp_mcmc, run_exp_mcmc)



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

    time_mh = []
    for i in range(nrep + nwarmup):
        init_state = tf.random.normal(shape=(1, dim))
        t1 = time.time()
        _ = run_mcmc(kernel_mh, nsamp, init_state, trace=False)
        t2 = time.time()
        # The first few iterations are warm-ups
        if i >= nwarmup:
            time_mh.append(t2 - t1)
    print(np.median(time_mh))
    benchmark["mh"] = time_mh

    time_hmc = []
    for i in range(nrep + nwarmup):
        init_state = tf.random.normal(shape=(1, dim))
        t1 = time.time()
        _ = run_mcmc(kernel_hmc, nsamp, init_state, trace=False)
        t2 = time.time()
        # The first few iterations are warm-ups
        if i >= nwarmup:
            time_hmc.append(t2 - t1)
    print(np.median(time_hmc))
    benchmark["hmc"] = time_hmc

    time_pt = []
    # Kernel for parallel tempering
    num_chains = 1
    inv_temp = tf.exp(tf.linspace(0.0, math.log(0.1), num=5))
    step_size = 0.1 / tf.sqrt(inv_temp)
    step_size = tf.tile(tf.reshape(step_size, shape=(5, 1)), multiples=(1, num_chains))
    step_size = tf.reshape(step_size, shape=(5, num_chains, 1))
    kernel_pt = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=energy.log_prob,
        inverse_temperatures=inv_temp,
        make_kernel_fn=lambda target_log_prob_fn: tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob_fn,
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size))
    )
    for i in range(nrep + nwarmup):
        init_state = tf.random.normal(shape=(1, dim))
        t1 = time.time()
        _ = run_mcmc(kernel_pt, nsamp, init_state, trace=False)
        t2 = time.time()
        # The first few iterations are warm-ups
        if i >= nwarmup:
            time_pt.append(t2 - t1)
    print(np.median(time_pt))
    benchmark["pt"] = time_pt

    # Save benchmarking results
    with open(f"{model_prefix}-benchmark.json", mode="w") as f:
        json.dump(benchmark, f, indent=2)
    print("finished")

cond_run_experiment(
    "Benchmarks", exp_benchmark, run_exp_benchmark)
