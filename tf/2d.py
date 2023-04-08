################################################################
# This section is intended to be run in interactive mode
data = "circle"
exp_sampling = True
exp_ablation = True
exp_mcmc = True
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
    "--exp_sampling", default=False, dest="exp_sampling",
    action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--exp_ablation", default=False, dest="exp_ablation",
    action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--exp_mcmc", default=False, dest="exp_mcmc",
    action=argparse.BooleanOptionalAction)

args = parser.parse_args()
data = args.data
exp_sampling = args.exp_sampling
exp_ablation = args.exp_ablation
exp_mcmc = args.exp_mcmc
################################################################

import time
import os
import json
import math
import numpy as np
import ot
import matplotlib.pyplot as plt

import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
import tensorflow_probability as tfp
tfd = tfp.distributions

import temperflow.option as opts
opts.set_opts(jit=True, debug=False)
from temperflow.bijector import Spline2D
from temperflow.distribution import TransformedDistribution
from temperflow.flow import TemperFlow

from utils import get_distr, cond_run_experiment, vis_energy, vis_trace, \
    rej_sampling, vis_sampling_res, vis_sampling_res_simple

# Paths
cache_path = "cache"
model_path = "model"
cache_prefix = f"{cache_path}/{data}"
model_prefix = f"{model_path}/{data}"
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
bijector = Spline2D(count_bins=32, bound=None, hlayers=[64, 64])
bijdistr = TransformedDistribution(dim=2, bijector=bijector)
temper = TemperFlow(energy, bijdistr)

# t1 = time.time()
# temper.train_one_beta(beta=0.1, fn="kl", bs=2000, nepoch=1001, lr=0.001, verbose=2)
# t2 = time.time()
# print(f"{t2 - t1} seconds")

# Run the main sampling experiment
def run_exp_sampling():
    t1 = time.time()
    trace = temper.train_flow(beta0=0.1, kl_decay=0.5, nepoch=(1001, 1001, 0),
                              bs=2000, lr=0.001, max_betas=100, verbose=2,
                              checkpoint=f"{cache_path}/{data}", recover=False)
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



# Ablation experiment
np.random.seed(123)
tf.random.set_seed(123)

# Run the ablation experiment
def run_exp_ablation():
    def compute_w(xsamp):
        # Compute Wasserstein distance with the true sample
        w = ot.emd2(a=np.ones(nsamp) / nsamp, b=np.ones(nsamp) / nsamp,
                    M=ot.dist(xtrue, xsamp, metric="minkowski", p=1))
        print(f"W = {w}")
        return w

    betas = np.exp(np.linspace(math.log(0.1), 0.0, 10, dtype=np.float32))
    nsamp = 2000
    xtrue = energy.sample(nsamp).numpy()
    xtrue2 = energy.sample(nsamp).numpy()
    w0 = compute_w(xtrue2)
    ############ Use KL only ############
    np.random.seed(123)
    tf.random.set_seed(123)
    bijector = Spline2D(count_bins=32, bound=None, hlayers=[64, 64])
    bijdistr = TransformedDistribution(dim=2, bijector=bijector)
    temper = TemperFlow(energy, bijdistr)
    temper.train_one_beta(beta=1.0, fn="kl", bs=2000, nepoch=2001, lr=0.001, verbose=2)
    # Visualize the result
    xsamp = bijdistr.sample(nsamp).numpy()
    label = f"Adj. W = {compute_w(xsamp) - w0:.3f}"
    vis_sampling_res_simple(bijdistr, nsamp=nsamp, xlab=label)
    plt.savefig(f"{model_prefix}-ablation-kl-only.pdf", bbox_inches="tight")
    ############ Use KL+tempering ############
    np.random.seed(123)
    tf.random.set_seed(123)
    bijector = Spline2D(count_bins=32, bound=None, hlayers=[64, 64])
    bijdistr = TransformedDistribution(dim=2, bijector=bijector)
    temper = TemperFlow(energy, bijdistr)
    for beta in betas:
        print(f"beta = {beta}")
        temper.train_one_beta(beta=beta, fn="kl", bs=2000, nepoch=2001, lr=0.001, verbose=2)
    # Visualize the result
    xsamp = bijdistr.sample(nsamp).numpy()
    label = f"Adj. W = {compute_w(xsamp) - w0:.3f}"
    vis_sampling_res_simple(bijdistr, nsamp=nsamp, xlab=label)
    plt.savefig(f"{model_prefix}-ablation-kl.pdf", bbox_inches="tight")
    ############ Use TemperFlow ############
    np.random.seed(123)
    tf.random.set_seed(123)
    bijector = Spline2D(count_bins=32, bound=None, hlayers=[64, 64])
    bijdistr = TransformedDistribution(dim=2, bijector=bijector)
    temper = TemperFlow(energy, bijdistr)
    for i, beta in enumerate(betas):
        print(f"beta = {beta}")
        if i == 0:
            temper.train_one_beta(beta=beta, fn="kl", bs=2000, nepoch=2001, lr=0.001, verbose=2)
        else:
            temper.train_one_beta(beta=beta, fn="l2", ref=True, bs=2000, nepoch=2001, lr=0.001, verbose=2)
        temper.update_ref()
    # Visualize the result
    xsamp = bijdistr.sample(nsamp).numpy()
    label = f"Adj. W = {compute_w(xsamp) - w0:.3f}"
    vis_sampling_res_simple(bijdistr, nsamp=nsamp, xlab=label)
    plt.savefig(f"{model_prefix}-ablation-l2.pdf", bbox_inches="tight")

cond_run_experiment(
    "Ablation experiment", exp_ablation, run_exp_ablation)



# MCMC
np.random.seed(123)
tf.random.set_seed(123)
num_results = 1000
num_chains = 100

# Kernel for M-H
kernel_mh = tfp.mcmc.RandomWalkMetropolis(
    target_log_prob_fn=energy.distr.log_prob,
    new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=0.2)
)
# Kernel for HMC
kernel_hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=energy.distr.log_prob,
    step_size=0.2,
    num_leapfrog_steps=5
)
# Kernel for parallel tempering
inv_temp = tf.exp(tf.linspace(0.0, math.log(0.1), num=5))
step_size = 0.2 / tf.sqrt(inv_temp)
step_size = tf.tile(tf.reshape(step_size, shape=(5, 1)), multiples=(1, num_chains))
step_size = tf.reshape(step_size, shape=(5, num_chains, 1))
kernel_pt = tfp.mcmc.ReplicaExchangeMC(
    target_log_prob_fn=energy.distr.log_prob,
    inverse_temperatures=inv_temp,
    make_kernel_fn=lambda target_log_prob_fn: tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size))
)

# Initial state of the chain
init_state = tf.random.normal(shape=(num_chains, 2))

@tf.function
def run_mcmc(kernel, num_results, init_state, trace=False):
    trace_fn = (lambda current_state, kernel_results: kernel_results) if trace else None
    res = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=kernel,
        num_burnin_steps=200,
        num_steps_between_results=1,  # Thinning
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

    t1 = time.time()
    hmc, trace_hmc = run_mcmc(kernel_hmc, num_results, init_state, trace=True)
    t2 = time.time()
    hmc = tf.transpose(hmc, perm=[1, 0, 2])
    print(f"{t2 - t1} seconds")
    print(hmc.shape)
    print(tf.reduce_mean(tf.cast(trace_hmc.is_accepted, dtype=tf.float32)))

    t1 = time.time()
    pt, trace_pt = run_mcmc(kernel_pt, num_results, init_state, trace=True)
    t2 = time.time()
    pt = tf.transpose(pt, perm=[1, 0, 2])
    print(f"{t2 - t1} seconds")
    print(pt.shape)

    # Save data
    np.random.seed(123)
    tf.random.set_seed(123)
    dim = 2
    nrep = num_chains
    nsamp = num_results
    bijdistr.load_params(f"{model_prefix}-bijdistr.npz")
    # True sample
    xtrue = energy.sample(nrep * nsamp).numpy().reshape(nrep, nsamp, dim)
    xtrue2 = energy.sample(nrep * nsamp).numpy().reshape(nrep, nsamp, dim)
    # Sample from the model
    xest = bijdistr.sample(nrep * nsamp).numpy().reshape(nrep, nsamp, dim)
    # Rejection sampling
    xacc2 = xacc[:(nrep * nsamp)].reshape(nrep, nsamp, dim)
    np.savez_compressed(f"{model_prefix}.npz",
        xtrue=xtrue, xtrue2=xtrue2, xest=xest, xacc=xacc2,
        mh=mh.numpy(), hmc=hmc.numpy(), pt=pt.numpy())
    print("finished")

cond_run_experiment(
    "MCMC experiment", exp_mcmc, run_exp_mcmc)
